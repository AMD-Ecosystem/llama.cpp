#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#if HRX_Q6_REPRO_ACCEPTED_VK64
#include "../../kernels/mul_mat_vec_q6_k_wmma16_vk64_padded44_w64_wg256.hip.cpp"
#define HRX_Q6_REPRO_KERNEL hrx_mul_mat_vec_q6_k_wmma16x16_vk64_padded44_w64_f16acc_wg256_f32
#define HRX_Q6_REPRO_LABEL "q6-vk64-accepted-repro"
#else
#include "../../kernels/mul_mat_vec_q6_k_wmma16_vk64_padded44_w64_ring96_copyab_wg256.hip.cpp"
#define HRX_Q6_REPRO_KERNEL hrx_mul_mat_vec_q6_k_wmma16x16_vk64_padded44_w64_ring96_copyab_f16acc_wg256_f32
#define HRX_Q6_REPRO_LABEL "q6-ring96-copyab-repro"
#endif

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

static int q6_value(const hrx_block_q6_K_wmma_vk128_lhs & block, int in_block) {
    const int half = in_block / 128;
    const int idx = in_block - half * 128;
    const int lane = idx & 31;
    const int ql_base = half * 64;
    const int qh_base = half * 32;

    int q = 0;
    if (idx < 32) {
        q = (block.ql[ql_base + lane] & 0x0f) | (((block.qh[qh_base + lane] >> 0) & 3) << 4);
    } else if (idx < 64) {
        q = (block.ql[ql_base + lane + 32] & 0x0f) | (((block.qh[qh_base + lane] >> 2) & 3) << 4);
    } else if (idx < 96) {
        q = (block.ql[ql_base + lane] >> 4) | (((block.qh[qh_base + lane] >> 4) & 3) << 4);
    } else {
        q = (block.ql[ql_base + lane + 32] >> 4) | (((block.qh[qh_base + lane] >> 6) & 3) << 4);
    }
    return q - 32;
}

static int q6_scale(const hrx_block_q6_K_wmma_vk128_lhs & block, int group, int lane) {
    const int half = group >> 2;
    const int group_in_half = group & 3;
    return static_cast<int>(block.scales[half * 8 + group_in_half * 2 + lane / 16]);
}

static float q6_dequant(
        const std::vector<hrx_block_q6_K_wmma_vk128_lhs> & blocks,
        int row,
        int k_index,
        int blocks_per_row) {
    const hrx_block_q6_K_wmma_vk128_lhs & block =
        blocks[static_cast<size_t>(row) * blocks_per_row + (k_index >> 8)];
    const int in_block = k_index & 255;
    const int group = in_block >> 5;
    const int lane = in_block & 31;
    const float d = half_bits_to_float(block.d) * static_cast<float>(q6_scale(block, group, lane));
    return d * static_cast<float>(q6_value(block, in_block));
}

struct case_config {
    const char * name;
    uint16_t d_bits;
    int scale_mask;
    float rhs_scale;
};

static float rhs_value(int col, int k_index, const case_config & config) {
    const int raw = (col * 19 + k_index * 7 + 23) & 63;
    return (static_cast<float>(raw) - 31.0f) * config.rhs_scale;
}

static void fill_q6(
        std::vector<hrx_block_q6_K_wmma_vk128_lhs> & blocks,
        int rows,
        int blocks_per_row,
        const case_config & config) {
    for (int row = 0; row < rows; ++row) {
        for (int block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
            hrx_block_q6_K_wmma_vk128_lhs & block =
                blocks[static_cast<size_t>(row) * blocks_per_row + block_idx];
            block.d = config.d_bits;
            for (int i = 0; i < 16; ++i) {
                block.scales[i] = static_cast<int8_t>(1 + ((row * 3 + block_idx * 5 + i * 11) & config.scale_mask));
            }
            for (int i = 0; i < 64; ++i) {
                block.qh[i] = static_cast<uint8_t>((row * 17 + block_idx * 13 + i * 5) & 0xff);
            }
            for (int i = 0; i < 128; ++i) {
                const uint8_t lo = static_cast<uint8_t>((row * 5 + block_idx * 7 + i) & 15);
                const uint8_t hi = static_cast<uint8_t>((row * 11 + block_idx * 3 + i * 9) & 15);
                block.ql[i] = static_cast<uint8_t>(lo | (hi << 4));
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
        const std::vector<hrx_block_q6_K_wmma_vk128_lhs> & blocks,
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
                sum += q6_dequant(blocks, row, kk, blocks_per_row) * rhs[static_cast<size_t>(col) * k + kk];
            }
            ref[static_cast<size_t>(col) * rows + row] = sum;
        }
    }
    return ref;
}

static std::vector<float> ring96_cross_reference(
        const std::vector<hrx_block_q6_K_wmma_vk128_lhs> & blocks,
        const std::vector<float> & rhs,
        int k,
        int rows,
        int cols) {
    const int blocks_per_row = k / 256;
    std::vector<float> ref(static_cast<size_t>(rows) * cols, 0.0f);
    for (int col = 0; col < cols; ++col) {
        const int col_tile_base = (col / 64) * 64;
        const int col_local = col - col_tile_base;
        const int col_lane = col_local & 15;
        const int stored_col_tile = (col_local >> 4) & 3;
        const int source_col_tile = (stored_col_tile + 2) & 3;
        const int source_col = col_tile_base + source_col_tile * 16 + col_lane;
        for (int row = 0; row < rows; ++row) {
            float sum = 0.0f;
            if (source_col < cols) {
                for (int k0 = 0; k0 < k; k0 += 32) {
                    for (int kk = 0; kk < 16; ++kk) {
                        sum += q6_dequant(blocks, row, k0 + kk, blocks_per_row) *
                            rhs[static_cast<size_t>(source_col) * k + k0 + 16 + kk];
                    }
                }
            }
            ref[static_cast<size_t>(col) * rows + row] = sum;
        }
    }
    return ref;
}

static int run_case(int rows, int cols, int k, const case_config & config) {
    if ((k % 256) != 0) {
        std::fprintf(stderr, "k must be a multiple of 256\n");
        return 2;
    }

    const int blocks_per_row = k / 256;
    std::vector<hrx_block_q6_K_wmma_vk128_lhs> h_src0(static_cast<size_t>(rows) * blocks_per_row);
    std::vector<float> h_src1(static_cast<size_t>(cols) * k);
    std::vector<float> h_dst(static_cast<size_t>(rows) * cols, -777.0f);

    fill_q6(h_src0, rows, blocks_per_row, config);
    fill_rhs(h_src1, k, cols, config);

    device_buffer<hrx_block_q6_K_wmma_vk128_lhs> d_src0(h_src0.size());
    device_buffer<float> d_src1(h_src1.size());
    device_buffer<float> d_dst(h_dst.size());
    HIP_CHECK(hipMemcpy(d_src0.ptr, h_src0.data(), h_src0.size() * sizeof(h_src0[0]), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_src1.ptr, h_src1.data(), h_src1.size() * sizeof(h_src1[0]), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_dst.ptr, h_dst.data(), h_dst.size() * sizeof(h_dst[0]), hipMemcpyHostToDevice));

    const dim3 grid((rows + 63) / 64, (cols + 63) / 64, 1);
    hipLaunchKernelGGL(
        HRX_Q6_REPRO_KERNEL,
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
    const std::vector<float> cross_ref = ring96_cross_reference(h_src0, h_src1, k, rows, cols);
    size_t nan_count = 0;
    size_t inf_count = 0;
    size_t sentinel_count = 0;
    size_t bad_count = 0;
    size_t cross_bad_count = 0;
    size_t bad_groups[16] = {};
    double max_group_abs[16] = {};
    double max_abs = 0.0;
    double max_rel = 0.0;
    double cross_max_abs = 0.0;
    double cross_max_rel = 0.0;
    size_t cross_max_idx = 0;
    size_t max_idx = 0;
    for (size_t i = 0; i < h_dst.size(); ++i) {
        const float actual = h_dst[i];
        const int row = static_cast<int>(i % static_cast<size_t>(rows));
        const int col = static_cast<int>(i / static_cast<size_t>(rows));
        const int row_local = row & 63;
        const int col_local = col & 63;
        const int group = (col_local >> 4) * 4 + (row_local >> 4);
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
        const double cross_diff = std::abs(static_cast<double>(actual) - static_cast<double>(cross_ref[i]));
        const double cross_denom = std::max(1.0, std::abs(static_cast<double>(cross_ref[i])));
        const double cross_rel = cross_diff / cross_denom;
        if (diff > 0.25) {
            ++bad_count;
            if (group >= 0 && group < 16) {
                ++bad_groups[group];
            }
        }
        if (cross_diff > 0.25) {
            ++cross_bad_count;
        }
        if (group >= 0 && group < 16 && diff > max_group_abs[group]) {
            max_group_abs[group] = diff;
        }
        if (diff > max_abs) {
            max_abs = diff;
            max_rel = rel;
            max_idx = i;
        }
        if (cross_diff > cross_max_abs) {
            cross_max_abs = cross_diff;
            cross_max_rel = cross_rel;
            cross_max_idx = i;
        }
    }

    const int max_row = static_cast<int>(max_idx % static_cast<size_t>(rows));
    const int max_col = static_cast<int>(max_idx / static_cast<size_t>(rows));
    const int cross_max_row = static_cast<int>(cross_max_idx % static_cast<size_t>(rows));
    const int cross_max_col = static_cast<int>(cross_max_idx / static_cast<size_t>(rows));
    std::printf(
        "%s profile=%s rows=%d cols=%d k=%d elements=%zu nan=%zu inf=%zu sentinel=%zu bad_gt_0p25=%zu max_abs=%g max_rel=%g idx=%zu row=%d col=%d actual=%g ref=%g cross_bad_gt_0p25=%zu cross_max_abs=%g cross_max_rel=%g cross_idx=%zu cross_row=%d cross_col=%d cross_ref=%g bad_groups=",
        HRX_Q6_REPRO_LABEL,
        config.name,
        rows,
        cols,
        k,
        h_dst.size(),
        nan_count,
        inf_count,
        sentinel_count,
        bad_count,
        max_abs,
        max_rel,
        max_idx,
        max_row,
        max_col,
        h_dst[max_idx],
        ref[max_idx],
        cross_bad_count,
        cross_max_abs,
        cross_max_rel,
        cross_max_idx,
        cross_max_row,
        cross_max_col,
        cross_ref[cross_max_idx]);
    bool first = true;
    for (int group = 0; group < 16; ++group) {
        if (bad_groups[group] == 0 && max_group_abs[group] == 0.0) {
            continue;
        }
        if (!first) {
            std::printf(",");
        }
        std::printf("%d:%zu/%g", group, bad_groups[group], max_group_abs[group]);
        first = false;
    }
    std::printf("\n");
    return (nan_count == 0 && inf_count == 0 && sentinel_count == 0) ? 0 : 1;
}

int main() {
    const case_config small = {"small", 0x2000u, 3, 0.00390625f};
    const case_config stress = {"stress", 0x3400u, 31, 0.015625f};
    int status = 0;
    status |= run_case(64, 33, 256, small);
    status |= run_case(64, 33, 512, small);
    status |= run_case(64, 33, 3584, small);
    status |= run_case(64, 64, 3584, small);
    status |= run_case(128, 33, 3584, small);
    status |= run_case(64, 33, 3584, stress);
    return status;
}
