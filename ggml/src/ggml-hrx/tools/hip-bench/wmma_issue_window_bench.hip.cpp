#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define HIP_CHECK(expr) do { \
    hipError_t _err = (expr); \
    if (_err != hipSuccess) { \
        std::fprintf(stderr, "%s:%d: HIP error: %s\n", __FILE__, __LINE__, hipGetErrorString(_err)); \
        std::exit(2); \
    } \
} while (0)

typedef _Float16 half16_vec __attribute__((ext_vector_type(16)));
typedef _Float16 half8_vec __attribute__((ext_vector_type(8)));
typedef uint64_t u64x4_vec __attribute__((ext_vector_type(4)));

static __device__ __forceinline__ uint64_t wmma_issue_window_ds_read_b64(
        const __attribute__((address_space(3))) uint64_t * ptr) {
    uint64_t value = 0;
    asm volatile("ds_read_b64 %0, %1 offset:0\n"
                 : "=v"(value)
                 : "v"(ptr)
                 : "memory");
    return value;
}

static __device__ __forceinline__ half16_vec wmma_issue_window_load_fragment(
        const __attribute__((address_space(3))) uint64_t * base,
        unsigned int lane,
        unsigned int frag) {
    const unsigned int index = frag * 256u + lane * 4u;
    u64x4_vec raw;
    raw[0] = wmma_issue_window_ds_read_b64(base + index + 0u);
    raw[1] = wmma_issue_window_ds_read_b64(base + index + 1u);
    raw[2] = wmma_issue_window_ds_read_b64(base + index + 2u);
    raw[3] = wmma_issue_window_ds_read_b64(base + index + 3u);
    return __builtin_bit_cast(half16_vec, raw);
}

static __device__ __forceinline__ void wmma_issue_window_sink_fragments(
        const half16_vec & a0, const half16_vec & a1, const half16_vec & a2, const half16_vec & a3,
        const half16_vec & a4, const half16_vec & a5, const half16_vec & a6, const half16_vec & a7,
        const half16_vec & b0, const half16_vec & b1, const half16_vec & b2, const half16_vec & b3,
        const half16_vec & b4, const half16_vec & b5, const half16_vec & b6, const half16_vec & b7) {
    asm volatile(""
                 :
                 : "v"(a0), "v"(a1), "v"(a2), "v"(a3), "v"(a4), "v"(a5), "v"(a6), "v"(a7),
                   "v"(b0), "v"(b1), "v"(b2), "v"(b3), "v"(b4), "v"(b5), "v"(b6), "v"(b7)
                 : "memory");
}

static __device__ __forceinline__ half16_vec wmma_issue_window_constant_fragment(float value) {
    half16_vec result;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        result[i] = static_cast<_Float16>(value);
    }
    return result;
}

template <int wait_lgkmcnt>
__global__ __launch_bounds__(64, 1)
void wmma_issue_window_probe(float * dst) {
    __shared__ uint64_t sh[16 * 64 * 4];
    const unsigned int lane = __builtin_amdgcn_workitem_id_x() & 63u;

#pragma unroll
    for (unsigned int frag = 0; frag < 16u; ++frag) {
#pragma unroll
        for (unsigned int item = 0; item < 4u; ++item) {
            const uint64_t lo = static_cast<uint64_t>(0x3c00u + ((frag + item + lane) & 7u));
            const uint64_t packed = lo | (lo << 16) | (lo << 32) | (lo << 48);
            sh[frag * 256u + lane * 4u + item] = packed;
        }
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    const __attribute__((address_space(3))) uint64_t * lds =
        (const __attribute__((address_space(3))) uint64_t *) sh;

    const half16_vec a0 = wmma_issue_window_load_fragment(lds, lane, 0u);
    const half16_vec b0 = wmma_issue_window_load_fragment(lds, lane, 8u);
    const half16_vec a1 = wmma_issue_window_load_fragment(lds, lane, 1u);
    const half16_vec b1 = wmma_issue_window_load_fragment(lds, lane, 9u);
    const half16_vec a2 = wmma_issue_window_load_fragment(lds, lane, 2u);
    const half16_vec b2 = wmma_issue_window_load_fragment(lds, lane, 10u);
    const half16_vec a3 = wmma_issue_window_load_fragment(lds, lane, 3u);
    const half16_vec b3 = wmma_issue_window_load_fragment(lds, lane, 11u);
    const half16_vec a4 = wmma_issue_window_load_fragment(lds, lane, 4u);
    const half16_vec b4 = wmma_issue_window_load_fragment(lds, lane, 12u);
    const half16_vec a5 = wmma_issue_window_load_fragment(lds, lane, 5u);
    const half16_vec b5 = wmma_issue_window_load_fragment(lds, lane, 13u);
    const half16_vec a6 = wmma_issue_window_load_fragment(lds, lane, 6u);
    const half16_vec b6 = wmma_issue_window_load_fragment(lds, lane, 14u);
    const half16_vec a7 = wmma_issue_window_load_fragment(lds, lane, 7u);
    const half16_vec b7 = wmma_issue_window_load_fragment(lds, lane, 15u);

    wmma_issue_window_sink_fragments(a0, a1, a2, a3, a4, a5, a6, a7, b0, b1, b2, b3, b4, b5, b6, b7);

    if constexpr (wait_lgkmcnt == 51) {
        asm volatile("s_waitcnt lgkmcnt(51)\n" ::: "memory");
    } else {
        asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    }

    const half16_vec ones = wmma_issue_window_constant_fragment(1.0f);
    const half16_vec twos = wmma_issue_window_constant_fragment(2.0f);
    half8_vec acc;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        acc[i] = static_cast<_Float16>(0.0f);
    }

    acc = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(ones, twos, acc, false);
    acc = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(ones, twos, acc, true);
    acc = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(ones, twos, acc, false);
    acc = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(ones, twos, acc, true);
    acc = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(ones, twos, acc, false);
    acc = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(ones, twos, acc, true);
    acc = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(ones, twos, acc, false);
    acc = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(ones, twos, acc, true);

#pragma unroll
    for (int i = 0; i < 8; ++i) {
        dst[lane * 8u + static_cast<unsigned int>(i)] = static_cast<float>(acc[i]);
    }
}

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

static void check_finite(const std::vector<float> & values) {
    size_t bad = 0;
    for (float value : values) {
        if (!(value == value)) {
            ++bad;
        }
    }
    std::printf("check: elements=%zu nan=%zu\n", values.size(), bad);
    if (bad != 0) {
        std::exit(1);
    }
}

int main(int argc, char ** argv) {
    std::string mode = "lgkm51";
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], "--mode=", 7) == 0) {
            mode = argv[i] + 7;
        } else {
            std::fprintf(stderr, "usage: %s [--mode=lgkm51|wait0]\n", argv[0]);
            return 2;
        }
    }

    const size_t count = 64u * 8u;
    device_buffer<float> d_out(count);
    std::vector<float> h_out(count);
    HIP_CHECK(hipMemset(d_out.ptr, 0, count * sizeof(float)));

    if (mode == "lgkm51") {
        hipLaunchKernelGGL((wmma_issue_window_probe<51>), dim3(1), dim3(64), 0, 0, d_out.ptr);
    } else if (mode == "wait0") {
        hipLaunchKernelGGL((wmma_issue_window_probe<0>), dim3(1), dim3(64), 0, 0, d_out.ptr);
    } else {
        std::fprintf(stderr, "unknown mode: %s\n", mode.c_str());
        return 2;
    }

    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_out.data(), d_out.ptr, count * sizeof(float), hipMemcpyDeviceToHost));

    double checksum = 0.0;
    for (float value : h_out) {
        checksum += static_cast<double>(value);
    }

    std::printf("wmma-issue-window mode=%s elements=%zu checksum=%.6f\n",
        mode.c_str(),
        h_out.size(),
        checksum);
    check_finite(h_out);
    return 0;
}
