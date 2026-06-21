#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define HIP_CHECK(expr) do { \
    hipError_t _err = (expr); \
    if (_err != hipSuccess) { \
        std::fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(_err)); \
        std::exit(1); \
    } \
} while (0)

using half16_vec = _Float16 __attribute__((ext_vector_type(16)));
using u32x4_vec = uint32_t __attribute__((ext_vector_type(4)));

static __device__ __forceinline__ uint32_t pack_f16_pair(float lo, float hi) {
    const uint32_t lo_bits = static_cast<uint32_t>(
        __builtin_bit_cast(uint16_t, static_cast<_Float16>(lo)));
    const uint32_t hi_bits = static_cast<uint32_t>(
        __builtin_bit_cast(uint16_t, static_cast<_Float16>(hi)));
    return lo_bits | (hi_bits << 16);
}

__global__ __launch_bounds__(64, 1)
void wmma_f16_mixed_compact_probe(uint32_t * dst) {
    const uint32_t lane = __builtin_amdgcn_workitem_id_x() & 63u;

    half16_vec a;
    half16_vec b;
    u32x4_vec acc;

#pragma unroll
    for (int i = 0; i < 16; ++i) {
        a[i] = static_cast<_Float16>(1.0f);
        b[i] = static_cast<_Float16>(1.0f);
    }

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        acc[i] = pack_f16_pair(
            static_cast<float>((lane & 15u) + static_cast<uint32_t>(i * 2)),
            static_cast<float>((lane & 15u) + static_cast<uint32_t>(i * 2 + 1)));
    }

    u32x4_vec out = acc;
    asm volatile("v_wmma_f16_16x16x16_f16 %0, %1, %2, %0\n"
                 : "+v"(out)
                 : "v"(a), "v"(b)
                 : "memory");

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        dst[lane * 4u + static_cast<uint32_t>(i)] = out[i];
    }
}

int main() {
    constexpr size_t lanes = 64;
    constexpr size_t words_per_lane = 4;
    constexpr size_t count = lanes * words_per_lane;

    uint32_t * d_out = nullptr;
    HIP_CHECK(hipMalloc(&d_out, count * sizeof(uint32_t)));
    HIP_CHECK(hipMemset(d_out, 0, count * sizeof(uint32_t)));

    hipLaunchKernelGGL(wmma_f16_mixed_compact_probe, dim3(1), dim3(64), 0, 0, d_out);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<uint32_t> h_out(count, 0u);
    HIP_CHECK(hipMemcpy(h_out.data(), d_out, count * sizeof(uint32_t), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_out));

    size_t zero_words = 0;
    for (uint32_t value : h_out) {
        if (value == 0u) {
            ++zero_words;
        }
    }
    std::printf("wmma-f16-mixed-compact lanes=%zu words=%zu zero_words=%zu first=0x%08x\n",
        lanes, count, zero_words, h_out.empty() ? 0u : h_out[0]);
    return zero_words == count ? 1 : 0;
}
