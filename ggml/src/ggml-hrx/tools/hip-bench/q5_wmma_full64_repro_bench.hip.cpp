#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../../kernels/mul_mat_vec_q5_k_wmma16_vk64_padded44_w64_full64_wg256.hip.cpp"

#define HIP_CHECK(expr) do { \
    hipError_t _err = (expr); \
    if (_err != hipSuccess) { \
        std::fprintf(stderr, "%s:%d: HIP error: %s\n", __FILE__, __LINE__, hipGetErrorString(_err)); \
        std::exit(2); \
    } \
} while (0)

template <typename T>
struct device_buffer {
    T * ptr = nullptr;
    size_t count = 0;

    explicit device_buffer(size_t count) : count(count) {
        HIP_CHECK(hipMalloc(&ptr, count * sizeof(T)));
    }

    ~device_buffer() {
        if (ptr) {
            (void) hipFree(ptr);
        }
    }

    device_buffer(const device_buffer &) = delete;
    device_buffer & operator=(const device_buffer &) = delete;
};

static float half_bits_to_float(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h >> 15) & 1u;
    const uint32_t exp = static_cast<uint32_t>(h >> 10) & 31u;
    const uint32_t mant = static_cast<uint32_t>(h) & 1023u;
    if (exp == 0) {
        const float value = std::ldexp(static_cast<float>(mant), -24);
        return sign ? -value : value;
    }
    if (exp == 31) {
        return mant ? NAN : (sign ? -INFINITY : INFINITY);
    }
    const float value = std::ldexp(1.0f + static_cast<float>(mant) / 1024.0f, static_cast<int>(exp) - 15);
    return sign ? -value : value;
}

static void get_scale_min(int group, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (group < 4) {
        *d = q[group] & 63u;
        *m = q[group + 4] & 63u;
    } else {
        *d = (q[group + 4] & 0x0fu) | ((q[group - 4] >> 6) << 4);
        *m = (q[group + 4] >> 4) | ((q[group] >> 6) << 4);
    }
}

static int q5_value(const hrx_block_q5_K_wmma_vk128_lhs & block, int in_block) {
    const int group = in_block >> 5;
    const int q_index = (group >> 1) * 32 + (in_block & 31);
    const uint8_t packed = block.qs[q_index];
    const uint8_t lo = (packed >> ((group & 1) * 4)) & 0x0f;
    const uint8_t hi = ((block.qh[in_block & 31] >> group) & 1) << 4;
    return lo | hi;
}

static float q5_dequant(
        const std::vector<hrx_block_q5_K_wmma_vk128_lhs> & blocks,
        int row,
        int k_index,
        int blocks_per_row) {
    const hrx_block_q5_K_wmma_vk128_lhs & block =
        blocks[static_cast<size_t>(row) * blocks_per_row + (k_index >> 8)];
    const int in_block = k_index & 255;
    const int group = in_block >> 5;
    uint8_t sc = 0;
    uint8_t m = 0;
    get_scale_min(group, block.scales, &sc, &m);
    const float d = half_bits_to_float(block.d) * static_cast<float>(sc);
    const float dmin = half_bits_to_float(block.dmin) * static_cast<float>(m);
    return d * static_cast<float>(q5_value(block, in_block)) - dmin;
}

struct case_config {
    const char * name;
    uint16_t d_bits;
    uint16_t dmin_bits;
    int scale_mask;
    float rhs_scale;
};

static float rhs_value(int col, int k_index, const case_config & config) {
    const int raw = (col * 19 + k_index * 7 + 23) & 63;
    return (static_cast<float>(raw) - 31.0f) * config.rhs_scale;
}

static void fill_q5(
        std::vector<hrx_block_q5_K_wmma_vk128_lhs> & blocks,
        int rows,
        int blocks_per_row,
        const case_config & config) {
    for (int row = 0; row < rows; ++row) {
        for (int block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
            hrx_block_q5_K_wmma_vk128_lhs & block =
                blocks[static_cast<size_t>(row) * blocks_per_row + block_idx];
            block.d = config.d_bits;
            block.dmin = config.dmin_bits;
            for (int i = 0; i < 12; ++i) {
                block.scales[i] = static_cast<uint8_t>(1 + ((row * 3 + block_idx * 5 + i * 11) & config.scale_mask));
            }
            for (int i = 0; i < 32; ++i) {
                block.qh[i] = static_cast<uint8_t>((row * 17 + block_idx * 13 + i * 5) & 0xff);
            }
            for (int i = 0; i < 128; ++i) {
                const uint8_t lo = static_cast<uint8_t>((row * 5 + block_idx * 7 + i) & 15);
                const uint8_t hi = static_cast<uint8_t>((row * 11 + block_idx * 3 + i * 9) & 15);
                block.qs[i] = static_cast<uint8_t>(lo | (hi << 4));
            }
        }
    }
}

static void fill_rhs(std::vector<float> & rhs, int k, int cols, const case_config & config) {
    for (int col = 0; col < cols; ++col) {
        for (int kk = 0; kk < k; ++kk) {
            rhs[static_cast<size_t>(col) * k + kk] = rhs_value(col, kk, config);
        }
    }
}

static std::vector<float> cpu_reference(
        const std::vector<hrx_block_q5_K_wmma_vk128_lhs> & blocks,
        const std::vector<float> & rhs,
        int k,
        int rows,
        int cols) {
    const int blocks_per_row = k / 256;
    std::vector<float> ref(static_cast<size_t>(rows) * cols, 0.0f);
    for (int col = 0; col < cols; ++col) {
        for (int row = 0; row < rows; ++row) {
            float sum = 0.0f;
            for (int kk = 0; kk < k; ++kk) {
                sum += q5_dequant(blocks, row, kk, blocks_per_row) * rhs[static_cast<size_t>(col) * k + kk];
            }
            ref[static_cast<size_t>(col) * rows + row] = sum;
        }
    }
    return ref;
}

static int run_case(int rows, int cols, int k, const case_config & config) {
    const int blocks_per_row = k / 256;
    std::vector<hrx_block_q5_K_wmma_vk128_lhs> h_src0(static_cast<size_t>(rows) * blocks_per_row);
    std::vector<float> h_src1(static_cast<size_t>(cols) * k);
    std::vector<float> h_dst(static_cast<size_t>(rows) * cols, -777.0f);
    fill_q5(h_src0, rows, blocks_per_row, config);
    fill_rhs(h_src1, k, cols, config);

    device_buffer<hrx_block_q5_K_wmma_vk128_lhs> d_src0(h_src0.size());
    device_buffer<float> d_src1(h_src1.size());
    device_buffer<float> d_dst(h_dst.size());
    HIP_CHECK(hipMemcpy(d_src0.ptr, h_src0.data(), h_src0.size() * sizeof(h_src0[0]), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_src1.ptr, h_src1.data(), h_src1.size() * sizeof(h_src1[0]), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_dst.ptr, h_dst.data(), h_dst.size() * sizeof(h_dst[0]), hipMemcpyHostToDevice));

    dim3 grid((rows + 63) / 64, (cols + 63) / 64, 1);
    hipLaunchKernelGGL(
        hrx_mul_mat_vec_q5_k_wmma16x16_vk64_padded44_w64_full64_f16acc_wg256_f32,
        grid,
        dim3(256, 1, 1),
        0,
        0,
        d_src0.ptr,
        d_src1.ptr,
        d_dst.ptr,
        static_cast<long long>(k),
        static_cast<long long>(rows),
        static_cast<long long>(cols));
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_dst.data(), d_dst.ptr, h_dst.size() * sizeof(h_dst[0]), hipMemcpyDeviceToHost));

    const std::vector<float> ref = cpu_reference(h_src0, h_src1, k, rows, cols);
    size_t nan_count = 0;
    size_t inf_count = 0;
    size_t sentinel_count = 0;
    double max_abs = 0.0;
    double max_rel = 0.0;
    size_t max_idx = 0;
    for (size_t i = 0; i < h_dst.size(); ++i) {
        const float actual = h_dst[i];
        if (std::isnan(actual)) {
            ++nan_count;
            continue;
        }
        if (std::isinf(actual)) {
            ++inf_count;
            continue;
        }
        if (actual == -777.0f) {
            ++sentinel_count;
        }
        const double diff = std::abs(static_cast<double>(actual) - static_cast<double>(ref[i]));
        const double denom = std::max(1.0, std::abs(static_cast<double>(ref[i])));
        const double rel = diff / denom;
        if (diff > max_abs) {
            max_abs = diff;
            max_rel = rel;
            max_idx = i;
        }
    }

    std::printf(
        "q5-full64-repro profile=%s rows=%d cols=%d k=%d nan=%zu inf=%zu sentinel=%zu max_abs=%g max_rel=%g idx=%zu actual=%g ref=%g\n",
        config.name,
        rows,
        cols,
        k,
        nan_count,
        inf_count,
        sentinel_count,
        max_abs,
        max_rel,
        max_idx,
        h_dst[max_idx],
        ref[max_idx]);
    return (nan_count == 0 && inf_count == 0 && sentinel_count == 0) ? 0 : 1;
}

int main() {
    const case_config small = {"small", 0x2000u, 0x0000u, 3, 0.00390625f};
    const case_config stress = {"stress", 0x3400u, 0x2c00u, 31, 0.015625f};
    int status = 0;
    status |= run_case(64, 33, 256, small);
    status |= run_case(64, 33, 512, small);
    status |= run_case(64, 33, 3584, small);
    status |= run_case(64, 64, 3584, small);
    status |= run_case(128, 33, 3584, small);
    status |= run_case(64, 33, 256, stress);
    status |= run_case(64, 33, 512, stress);
    status |= run_case(64, 33, 3584, stress);
    status |= run_case(64, 64, 3584, stress);
    status |= run_case(128, 33, 3584, stress);
    return status;
}
