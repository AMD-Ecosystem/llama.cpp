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

__global__ __launch_bounds__(64, 1)
void wmma_f16_raw_radv_compact_encoding_probe(uint32_t * dst) {
    uint32_t o0 = 0;
    uint32_t o1 = 0;
    uint32_t o2 = 0;
    uint32_t o3 = 0;

    // RADV Q6_K medium emits compact f16 accumulator operands such as:
    // v_wmma_f16_16x16x16_f16 v[40:43], v[56:63], v[64:71], v[40:43]
    // LLVM MC rejects that symbolic operand form for gfx1151, so this probe
    // injects the RADV instruction encoding directly as a diagnostic.
    asm volatile(
        "v_mov_b32 v40, 0\n\t"
        "v_mov_b32 v41, 0\n\t"
        "v_mov_b32 v42, 0\n\t"
        "v_mov_b32 v43, 0\n\t"
        "v_mov_b32 v56, 0x3c003c00\n\t"
        "v_mov_b32 v57, 0x3c003c00\n\t"
        "v_mov_b32 v58, 0x3c003c00\n\t"
        "v_mov_b32 v59, 0x3c003c00\n\t"
        "v_mov_b32 v60, 0x3c003c00\n\t"
        "v_mov_b32 v61, 0x3c003c00\n\t"
        "v_mov_b32 v62, 0x3c003c00\n\t"
        "v_mov_b32 v63, 0x3c003c00\n\t"
        "v_mov_b32 v64, 0x3c003c00\n\t"
        "v_mov_b32 v65, 0x3c003c00\n\t"
        "v_mov_b32 v66, 0x3c003c00\n\t"
        "v_mov_b32 v67, 0x3c003c00\n\t"
        "v_mov_b32 v68, 0x3c003c00\n\t"
        "v_mov_b32 v69, 0x3c003c00\n\t"
        "v_mov_b32 v70, 0x3c003c00\n\t"
        "v_mov_b32 v71, 0x3c003c00\n\t"
        ".long 0xcc424028\n\t"
        ".long 0x1ca28138\n\t"
        "v_mov_b32 %0, v40\n\t"
        "v_mov_b32 %1, v41\n\t"
        "v_mov_b32 %2, v42\n\t"
        "v_mov_b32 %3, v43\n\t"
        : "=v"(o0), "=v"(o1), "=v"(o2), "=v"(o3)
        :
        : "memory");

    const uint32_t lane = __builtin_amdgcn_workitem_id_x() & 63u;
    dst[lane * 4u + 0u] = o0;
    dst[lane * 4u + 1u] = o1;
    dst[lane * 4u + 2u] = o2;
    dst[lane * 4u + 3u] = o3;
}

int main() {
    constexpr size_t lanes = 64;
    constexpr size_t words_per_lane = 4;
    constexpr size_t count = lanes * words_per_lane;

    uint32_t * d_symbolic = nullptr;
    uint32_t * d_raw = nullptr;
    HIP_CHECK(hipMalloc(&d_symbolic, count * sizeof(uint32_t)));
    HIP_CHECK(hipMalloc(&d_raw, count * sizeof(uint32_t)));
    HIP_CHECK(hipMemset(d_symbolic, 0, count * sizeof(uint32_t)));
    HIP_CHECK(hipMemset(d_raw, 0, count * sizeof(uint32_t)));

    hipLaunchKernelGGL(wmma_f16_mixed_compact_probe, dim3(1), dim3(64), 0, 0, d_symbolic);
    HIP_CHECK(hipGetLastError());
    hipLaunchKernelGGL(wmma_f16_raw_radv_compact_encoding_probe, dim3(1), dim3(64), 0, 0, d_raw);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<uint32_t> h_symbolic(count, 0u);
    std::vector<uint32_t> h_raw(count, 0u);
    HIP_CHECK(hipMemcpy(h_symbolic.data(), d_symbolic, count * sizeof(uint32_t), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_raw.data(), d_raw, count * sizeof(uint32_t), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_symbolic));
    HIP_CHECK(hipFree(d_raw));

    size_t symbolic_zero_words = 0;
    for (uint32_t value : h_symbolic) {
        if (value == 0u) {
            ++symbolic_zero_words;
        }
    }

    size_t raw_zero_words = 0;
    size_t raw_expected_words = 0;
    for (uint32_t value : h_raw) {
        if (value == 0u) {
            ++raw_zero_words;
        }
        if (value == 0x4c004c00u) {
            ++raw_expected_words;
        }
    }

    std::printf("wmma-f16-mixed-compact-symbolic lanes=%zu words=%zu zero_words=%zu first=0x%08x\n",
        lanes, count, symbolic_zero_words, h_symbolic.empty() ? 0u : h_symbolic[0]);
    std::printf("wmma-f16-raw-radv-compact-encoding lanes=%zu words=%zu zero_words=%zu expected_16x2_words=%zu first=0x%08x\n",
        lanes, count, raw_zero_words, raw_expected_words, h_raw.empty() ? 0u : h_raw[0]);
    return symbolic_zero_words == count || raw_zero_words == count ? 1 : 0;
}
