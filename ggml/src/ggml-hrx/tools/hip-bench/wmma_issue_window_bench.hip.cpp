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
typedef uint32_t u32x8_vec __attribute__((ext_vector_type(8)));
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

template <int wait_lgkmcnt>
static __device__ __forceinline__ half16_vec wmma_issue_window_dependent_constant(
        const half16_vec & a0, const half16_vec & a1, const half16_vec & a2, const half16_vec & a3,
        const half16_vec & a4, const half16_vec & a5, const half16_vec & a6, const half16_vec & a7,
        const half16_vec & b0, const half16_vec & b1, const half16_vec & b2, const half16_vec & b3,
        const half16_vec & b4, const half16_vec & b5, const half16_vec & b6, const half16_vec & b7) {
    u32x8_vec bits;
    if constexpr (wait_lgkmcnt == 51) {
        asm volatile("s_waitcnt lgkmcnt(51)\n\t"
                     "v_mov_b32 %0, 0x3c003c00\n\t"
                     "v_mov_b32 %1, 0x3c003c00\n\t"
                     "v_mov_b32 %2, 0x3c003c00\n\t"
                     "v_mov_b32 %3, 0x3c003c00\n\t"
                     "v_mov_b32 %4, 0x3c003c00\n\t"
                     "v_mov_b32 %5, 0x3c003c00\n\t"
                     "v_mov_b32 %6, 0x3c003c00\n\t"
                     "v_mov_b32 %7, 0x3c003c00\n\t"
                     : "=v"(bits[0]), "=v"(bits[1]), "=v"(bits[2]), "=v"(bits[3]),
                       "=v"(bits[4]), "=v"(bits[5]), "=v"(bits[6]), "=v"(bits[7])
                     : "v"(a0), "v"(a1), "v"(a2), "v"(a3), "v"(a4), "v"(a5), "v"(a6), "v"(a7),
                       "v"(b0), "v"(b1), "v"(b2), "v"(b3), "v"(b4), "v"(b5), "v"(b6), "v"(b7)
                     : "memory");
    } else {
        asm volatile("s_waitcnt lgkmcnt(0)\n\t"
                     "v_mov_b32 %0, 0x3c003c00\n\t"
                     "v_mov_b32 %1, 0x3c003c00\n\t"
                     "v_mov_b32 %2, 0x3c003c00\n\t"
                     "v_mov_b32 %3, 0x3c003c00\n\t"
                     "v_mov_b32 %4, 0x3c003c00\n\t"
                     "v_mov_b32 %5, 0x3c003c00\n\t"
                     "v_mov_b32 %6, 0x3c003c00\n\t"
                     "v_mov_b32 %7, 0x3c003c00\n\t"
                     : "=v"(bits[0]), "=v"(bits[1]), "=v"(bits[2]), "=v"(bits[3]),
                       "=v"(bits[4]), "=v"(bits[5]), "=v"(bits[6]), "=v"(bits[7])
                 : "v"(a0), "v"(a1), "v"(a2), "v"(a3), "v"(a4), "v"(a5), "v"(a6), "v"(a7),
                   "v"(b0), "v"(b1), "v"(b2), "v"(b3), "v"(b4), "v"(b5), "v"(b6), "v"(b7)
                 : "memory");
    }
    return __builtin_bit_cast(half16_vec, bits);
}

template <int wait_lgkmcnt>
static __device__ __forceinline__ half16_vec wmma_issue_window_dep_copy_initial(
        const half16_vec & src,
        const half16_vec & dep0, const half16_vec & dep1, const half16_vec & dep2,
        const half16_vec & dep3, const half16_vec & dep4, const half16_vec & dep5,
        const half16_vec & dep6) {
    const u32x8_vec in = __builtin_bit_cast(u32x8_vec, src);
    u32x8_vec out;
    asm volatile("s_waitcnt lgkmcnt(%16)\n\t"
                 "v_mov_b32 %0, %8\n\t"
                 "v_mov_b32 %1, %9\n\t"
                 "v_mov_b32 %2, %10\n\t"
                 "v_mov_b32 %3, %11\n\t"
                 "v_mov_b32 %4, %12\n\t"
                 "v_mov_b32 %5, %13\n\t"
                 "v_mov_b32 %6, %14\n\t"
                 "v_mov_b32 %7, %15\n\t"
                 : "=v"(out[0]), "=v"(out[1]), "=v"(out[2]), "=v"(out[3]),
                   "=v"(out[4]), "=v"(out[5]), "=v"(out[6]), "=v"(out[7])
                 : "v"(in[0]), "v"(in[1]), "v"(in[2]), "v"(in[3]),
                   "v"(in[4]), "v"(in[5]), "v"(in[6]), "v"(in[7]),
                   "n"(wait_lgkmcnt),
                   "v"(dep0), "v"(dep1), "v"(dep2), "v"(dep3),
                   "v"(dep4), "v"(dep5), "v"(dep6)
                 : "memory");
    return __builtin_bit_cast(half16_vec, out);
}

template <int wait_lgkmcnt>
static __device__ __forceinline__ half16_vec wmma_issue_window_dep_copy_after_acc(
        const half16_vec & src,
        const half16_vec & token,
        const half8_vec & prev_acc) {
    const u32x8_vec in = __builtin_bit_cast(u32x8_vec, src);
    u32x8_vec out;
    asm volatile("s_waitcnt lgkmcnt(%16)\n\t"
                 "v_mov_b32 %0, %8\n\t"
                 "v_mov_b32 %1, %9\n\t"
                 "v_mov_b32 %2, %10\n\t"
                 "v_mov_b32 %3, %11\n\t"
                 "v_mov_b32 %4, %12\n\t"
                 "v_mov_b32 %5, %13\n\t"
                 "v_mov_b32 %6, %14\n\t"
                 "v_mov_b32 %7, %15\n\t"
                 : "=v"(out[0]), "=v"(out[1]), "=v"(out[2]), "=v"(out[3]),
                   "=v"(out[4]), "=v"(out[5]), "=v"(out[6]), "=v"(out[7])
                 : "v"(in[0]), "v"(in[1]), "v"(in[2]), "v"(in[3]),
                   "v"(in[4]), "v"(in[5]), "v"(in[6]), "v"(in[7]),
                   "n"(wait_lgkmcnt), "v"(token), "v"(prev_acc)
                 : "memory");
    return __builtin_bit_cast(half16_vec, out);
}

static __device__ __forceinline__ half16_vec wmma_issue_window_dep_copy_after_token(
        const half16_vec & src,
        const half16_vec & token) {
    const u32x8_vec in = __builtin_bit_cast(u32x8_vec, src);
    u32x8_vec out;
    asm volatile("v_mov_b32 %0, %8\n\t"
                 "v_mov_b32 %1, %9\n\t"
                 "v_mov_b32 %2, %10\n\t"
                 "v_mov_b32 %3, %11\n\t"
                 "v_mov_b32 %4, %12\n\t"
                 "v_mov_b32 %5, %13\n\t"
                 "v_mov_b32 %6, %14\n\t"
                 "v_mov_b32 %7, %15\n\t"
                 : "=v"(out[0]), "=v"(out[1]), "=v"(out[2]), "=v"(out[3]),
                   "=v"(out[4]), "=v"(out[5]), "=v"(out[6]), "=v"(out[7])
                 : "v"(in[0]), "v"(in[1]), "v"(in[2]), "v"(in[3]),
                   "v"(in[4]), "v"(in[5]), "v"(in[6]), "v"(in[7]),
                   "v"(token)
                 : "memory");
    return __builtin_bit_cast(half16_vec, out);
}

#define WMMA_ISSUE_WINDOW_DEP_WMMA_INITIAL(ACC, TILE, A, B, W, D0, D1, D2, D3, D4, D5, D6) \
    do { \
        const half16_vec a_dep = wmma_issue_window_dep_copy_initial<W>((A), (D0), (D1), (D2), (D3), (D4), (D5), (D6)); \
        const half16_vec b_dep = wmma_issue_window_dep_copy_after_token((B), a_dep); \
        (ACC)[TILE] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(a_dep, b_dep, (ACC)[TILE], false); \
    } while (0)

#define WMMA_ISSUE_WINDOW_DEP_WMMA_AFTER(ACC, TILE, A, B, W, TOKEN, PREV) \
    do { \
        const half16_vec a_dep = wmma_issue_window_dep_copy_after_acc<W>((A), (TOKEN), (ACC)[PREV]); \
        const half16_vec b_dep = wmma_issue_window_dep_copy_after_token((B), a_dep); \
        (ACC)[TILE] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(a_dep, b_dep, (ACC)[TILE], false); \
    } while (0)

#define WMMA_ISSUE_WINDOW_DIRECT_WMMA(ACC, TILE, A, B) \
    do { \
        (ACC)[TILE] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64((A), (B), (ACC)[TILE], false); \
    } while (0)

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

    const half16_vec ones =
        wmma_issue_window_dependent_constant<wait_lgkmcnt>(a0, a1, a2, a3, a4, a5, a6, a7,
                                                           b0, b1, b2, b3, b4, b5, b6, b7);
    half8_vec acc;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        acc[i] = static_cast<_Float16>(0.0f);
    }

    acc = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(ones, ones, acc, false);
    acc = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(ones, ones, acc, true);
    acc = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(ones, ones, acc, false);
    acc = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(ones, ones, acc, true);
    acc = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(ones, ones, acc, false);
    acc = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(ones, ones, acc, true);
    acc = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(ones, ones, acc, false);
    acc = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(ones, ones, acc, true);

#pragma unroll
    for (int i = 0; i < 8; ++i) {
        dst[lane * 8u + static_cast<unsigned int>(i)] = static_cast<float>(acc[i]);
    }
}

template <int wait_lgkmcnt>
__global__ __launch_bounds__(64, 1)
void wmma_issue_window_realfrag16_probe(float * dst) {
    __shared__ uint64_t sh[16 * 64 * 4];
    const unsigned int lane = __builtin_amdgcn_workitem_id_x() & 63u;

#pragma unroll
    for (unsigned int frag = 0; frag < 8u; ++frag) {
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
    const half16_vec a1 = wmma_issue_window_load_fragment(lds, lane, 1u);
    const half16_vec a2 = wmma_issue_window_load_fragment(lds, lane, 2u);
    const half16_vec a3 = wmma_issue_window_load_fragment(lds, lane, 3u);
    const half16_vec b0 = wmma_issue_window_load_fragment(lds, lane, 4u);
    const half16_vec b1 = wmma_issue_window_load_fragment(lds, lane, 5u);
    const half16_vec b2 = wmma_issue_window_load_fragment(lds, lane, 6u);
    const half16_vec b3 = wmma_issue_window_load_fragment(lds, lane, 7u);

    half8_vec acc[16];
#pragma unroll
    for (int t = 0; t < 16; ++t) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            acc[t][i] = static_cast<_Float16>(0.0f);
        }
    }

    WMMA_ISSUE_WINDOW_DEP_WMMA_INITIAL(acc, 0, a0, b0, wait_lgkmcnt, a1, a2, a3, b0, b1, b2, b3);
    WMMA_ISSUE_WINDOW_DEP_WMMA_AFTER(acc, 1, a1, b0, 47, a0, 0);
    WMMA_ISSUE_WINDOW_DEP_WMMA_AFTER(acc, 2, a2, b0, 43, a1, 1);
    WMMA_ISSUE_WINDOW_DEP_WMMA_AFTER(acc, 3, a3, b0, 39, a2, 2);
    WMMA_ISSUE_WINDOW_DEP_WMMA_AFTER(acc, 4, a0, b1, 40, a3, 3);
    WMMA_ISSUE_WINDOW_DEP_WMMA_AFTER(acc, 5, a1, b1, 36, b1, 4);
    WMMA_ISSUE_WINDOW_DEP_WMMA_AFTER(acc, 6, a2, b1, 32, a1, 5);
    WMMA_ISSUE_WINDOW_DEP_WMMA_AFTER(acc, 7, a3, b1, 24, a2, 6);
    WMMA_ISSUE_WINDOW_DEP_WMMA_AFTER(acc, 8, a0, b2, 20, a3, 7);
    WMMA_ISSUE_WINDOW_DEP_WMMA_AFTER(acc, 9, a1, b2, 16, b2, 8);
    WMMA_ISSUE_WINDOW_DEP_WMMA_AFTER(acc, 10, a2, b2, 12, a1, 9);
    WMMA_ISSUE_WINDOW_DEP_WMMA_AFTER(acc, 11, a3, b2, 8, a2, 10);
    WMMA_ISSUE_WINDOW_DEP_WMMA_AFTER(acc, 12, a0, b3, 4, a3, 11);
    WMMA_ISSUE_WINDOW_DEP_WMMA_AFTER(acc, 13, a1, b3, 0, b3, 12);
    WMMA_ISSUE_WINDOW_DEP_WMMA_AFTER(acc, 14, a2, b3, 0, a1, 13);
    WMMA_ISSUE_WINDOW_DEP_WMMA_AFTER(acc, 15, a3, b3, 0, a2, 14);

#pragma unroll
    for (int t = 0; t < 16; ++t) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            dst[lane * 128u + static_cast<unsigned int>(t * 8 + i)] = static_cast<float>(acc[t][i]);
        }
    }
}

__global__ __launch_bounds__(64, 1)
void wmma_issue_window_realfrag16_direct_probe(float * dst) {
    __shared__ uint64_t sh[16 * 64 * 4];
    const unsigned int lane = __builtin_amdgcn_workitem_id_x() & 63u;

#pragma unroll
    for (unsigned int frag = 0; frag < 8u; ++frag) {
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
    const half16_vec a1 = wmma_issue_window_load_fragment(lds, lane, 1u);
    const half16_vec a2 = wmma_issue_window_load_fragment(lds, lane, 2u);
    const half16_vec a3 = wmma_issue_window_load_fragment(lds, lane, 3u);
    const half16_vec b0 = wmma_issue_window_load_fragment(lds, lane, 4u);
    const half16_vec b1 = wmma_issue_window_load_fragment(lds, lane, 5u);
    const half16_vec b2 = wmma_issue_window_load_fragment(lds, lane, 6u);
    const half16_vec b3 = wmma_issue_window_load_fragment(lds, lane, 7u);
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");

    half8_vec acc[16];
#pragma unroll
    for (int t = 0; t < 16; ++t) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            acc[t][i] = static_cast<_Float16>(0.0f);
        }
    }

    WMMA_ISSUE_WINDOW_DIRECT_WMMA(acc, 0, a0, b0);
    WMMA_ISSUE_WINDOW_DIRECT_WMMA(acc, 1, a1, b0);
    WMMA_ISSUE_WINDOW_DIRECT_WMMA(acc, 2, a2, b0);
    WMMA_ISSUE_WINDOW_DIRECT_WMMA(acc, 3, a3, b0);
    WMMA_ISSUE_WINDOW_DIRECT_WMMA(acc, 4, a0, b1);
    WMMA_ISSUE_WINDOW_DIRECT_WMMA(acc, 5, a1, b1);
    WMMA_ISSUE_WINDOW_DIRECT_WMMA(acc, 6, a2, b1);
    WMMA_ISSUE_WINDOW_DIRECT_WMMA(acc, 7, a3, b1);
    WMMA_ISSUE_WINDOW_DIRECT_WMMA(acc, 8, a0, b2);
    WMMA_ISSUE_WINDOW_DIRECT_WMMA(acc, 9, a1, b2);
    WMMA_ISSUE_WINDOW_DIRECT_WMMA(acc, 10, a2, b2);
    WMMA_ISSUE_WINDOW_DIRECT_WMMA(acc, 11, a3, b2);
    WMMA_ISSUE_WINDOW_DIRECT_WMMA(acc, 12, a0, b3);
    WMMA_ISSUE_WINDOW_DIRECT_WMMA(acc, 13, a1, b3);
    WMMA_ISSUE_WINDOW_DIRECT_WMMA(acc, 14, a2, b3);
    WMMA_ISSUE_WINDOW_DIRECT_WMMA(acc, 15, a3, b3);

#pragma unroll
    for (int t = 0; t < 16; ++t) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            dst[lane * 128u + static_cast<unsigned int>(t * 8 + i)] = static_cast<float>(acc[t][i]);
        }
    }
}

template <int wait_lgkmcnt>
__global__ __launch_bounds__(64, 1)
void wmma_issue_window_realfrag8_probe(float * dst) {
    __shared__ uint64_t sh[12 * 64 * 4];
    const unsigned int lane = __builtin_amdgcn_workitem_id_x() & 63u;

#pragma unroll
    for (unsigned int frag = 0; frag < 6u; ++frag) {
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
    const half16_vec a1 = wmma_issue_window_load_fragment(lds, lane, 1u);
    const half16_vec a2 = wmma_issue_window_load_fragment(lds, lane, 2u);
    const half16_vec a3 = wmma_issue_window_load_fragment(lds, lane, 3u);
    const half16_vec b0 = wmma_issue_window_load_fragment(lds, lane, 4u);
    const half16_vec b1 = wmma_issue_window_load_fragment(lds, lane, 5u);

    half8_vec acc[8];
#pragma unroll
    for (int t = 0; t < 8; ++t) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            acc[t][i] = static_cast<_Float16>(0.0f);
        }
    }

    WMMA_ISSUE_WINDOW_DEP_WMMA_INITIAL(acc, 0, a0, b0, wait_lgkmcnt, a1, a2, a3, b0, b1, a0, b0);
    WMMA_ISSUE_WINDOW_DEP_WMMA_AFTER(acc, 1, a1, b0, 47, a0, 0);
    WMMA_ISSUE_WINDOW_DEP_WMMA_AFTER(acc, 2, a2, b0, 43, a1, 1);
    WMMA_ISSUE_WINDOW_DEP_WMMA_AFTER(acc, 3, a3, b0, 39, a2, 2);
    WMMA_ISSUE_WINDOW_DEP_WMMA_AFTER(acc, 4, a0, b1, 40, a3, 3);
    WMMA_ISSUE_WINDOW_DEP_WMMA_AFTER(acc, 5, a1, b1, 36, b1, 4);
    WMMA_ISSUE_WINDOW_DEP_WMMA_AFTER(acc, 6, a2, b1, 32, a1, 5);
    WMMA_ISSUE_WINDOW_DEP_WMMA_AFTER(acc, 7, a3, b1, 24, a2, 6);

#pragma unroll
    for (int t = 0; t < 8; ++t) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            dst[lane * 128u + static_cast<unsigned int>(t * 8 + i)] = static_cast<float>(acc[t][i]);
        }
    }
}

__global__ __launch_bounds__(64, 1)
void wmma_issue_window_realfrag8_direct_probe(float * dst) {
    __shared__ uint64_t sh[12 * 64 * 4];
    const unsigned int lane = __builtin_amdgcn_workitem_id_x() & 63u;

#pragma unroll
    for (unsigned int frag = 0; frag < 6u; ++frag) {
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
    const half16_vec a1 = wmma_issue_window_load_fragment(lds, lane, 1u);
    const half16_vec a2 = wmma_issue_window_load_fragment(lds, lane, 2u);
    const half16_vec a3 = wmma_issue_window_load_fragment(lds, lane, 3u);
    const half16_vec b0 = wmma_issue_window_load_fragment(lds, lane, 4u);
    const half16_vec b1 = wmma_issue_window_load_fragment(lds, lane, 5u);
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");

    half8_vec acc[8];
#pragma unroll
    for (int t = 0; t < 8; ++t) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            acc[t][i] = static_cast<_Float16>(0.0f);
        }
    }

    WMMA_ISSUE_WINDOW_DIRECT_WMMA(acc, 0, a0, b0);
    WMMA_ISSUE_WINDOW_DIRECT_WMMA(acc, 1, a1, b0);
    WMMA_ISSUE_WINDOW_DIRECT_WMMA(acc, 2, a2, b0);
    WMMA_ISSUE_WINDOW_DIRECT_WMMA(acc, 3, a3, b0);
    WMMA_ISSUE_WINDOW_DIRECT_WMMA(acc, 4, a0, b1);
    WMMA_ISSUE_WINDOW_DIRECT_WMMA(acc, 5, a1, b1);
    WMMA_ISSUE_WINDOW_DIRECT_WMMA(acc, 6, a2, b1);
    WMMA_ISSUE_WINDOW_DIRECT_WMMA(acc, 7, a3, b1);

#pragma unroll
    for (int t = 0; t < 8; ++t) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            dst[lane * 128u + static_cast<unsigned int>(t * 8 + i)] = static_cast<float>(acc[t][i]);
        }
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
    size_t bad_even_slots = 0;
    size_t bad_odd_slots = 0;
    size_t first_bad = values.size();
    for (size_t idx = 0; idx < values.size(); ++idx) {
        if (values[idx] == values[idx]) {
            continue;
        }
        if (first_bad == values.size()) {
            first_bad = idx;
        }
        ++bad;
        const size_t slot = idx & 7u;
        if ((slot & 1u) == 0u) {
            ++bad_even_slots;
        } else {
            ++bad_odd_slots;
        }
    }
    std::printf("check: elements=%zu nan=%zu nan_even=%zu nan_odd=%zu",
        values.size(), bad, bad_even_slots, bad_odd_slots);
    if (first_bad != values.size()) {
        const size_t slot = first_bad & 7u;
        const size_t lane = (first_bad / 128u) & 63u;
        const size_t tile = (first_bad / 8u) & 15u;
        std::printf(" first_nan_tile=%zu first_nan_lane=%zu first_nan_slot=%zu", tile, lane, slot);
    }
    std::printf("\n");
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
            std::fprintf(stderr, "usage: %s [--mode=lgkm51|wait0|realfrag16|realfrag8|realfrag16-direct|realfrag8-direct]\n", argv[0]);
            return 2;
        }
    }

    const size_t count = 64u * 16u * 8u;
    device_buffer<float> d_out(count);
    std::vector<float> h_out(count);
    HIP_CHECK(hipMemset(d_out.ptr, 0, count * sizeof(float)));

    if (mode == "lgkm51") {
        hipLaunchKernelGGL((wmma_issue_window_probe<51>), dim3(1), dim3(64), 0, 0, d_out.ptr);
    } else if (mode == "wait0") {
        hipLaunchKernelGGL((wmma_issue_window_probe<0>), dim3(1), dim3(64), 0, 0, d_out.ptr);
    } else if (mode == "realfrag16") {
        hipLaunchKernelGGL((wmma_issue_window_realfrag16_probe<51>), dim3(1), dim3(64), 0, 0, d_out.ptr);
    } else if (mode == "realfrag8") {
        hipLaunchKernelGGL((wmma_issue_window_realfrag8_probe<51>), dim3(1), dim3(64), 0, 0, d_out.ptr);
    } else if (mode == "realfrag16-direct") {
        hipLaunchKernelGGL(wmma_issue_window_realfrag16_direct_probe, dim3(1), dim3(64), 0, 0, d_out.ptr);
    } else if (mode == "realfrag8-direct") {
        hipLaunchKernelGGL(wmma_issue_window_realfrag8_direct_probe, dim3(1), dim3(64), 0, 0, d_out.ptr);
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
