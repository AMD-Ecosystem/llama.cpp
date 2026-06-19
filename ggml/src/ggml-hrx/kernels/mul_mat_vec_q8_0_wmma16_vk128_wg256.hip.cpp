#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <stdint.h>

#ifndef HRX_Q8_0_WMMA_VK128_EXPORT
#define HRX_Q8_0_WMMA_VK128_EXPORT hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_f16acc_wg256_f32
#endif

#ifndef HRX_Q8_0_WMMA_VK128_SHARED_STRIDE
#define HRX_Q8_0_WMMA_VK128_SHARED_STRIDE 40
#endif

#ifndef HRX_Q8_0_WMMA_VK128_PREFETCH_FRAGS
#define HRX_Q8_0_WMMA_VK128_PREFETCH_FRAGS 0
#endif

#ifndef HRX_Q8_0_WMMA_VK128_PAIR64_TILE_MAP
#define HRX_Q8_0_WMMA_VK128_PAIR64_TILE_MAP 0
#endif

#ifndef HRX_Q8_0_WMMA_VK128_W64
#define HRX_Q8_0_WMMA_VK128_W64 1
#endif

#ifndef HRX_Q8_0_WMMA_VK128_W64_OPSEL
#define HRX_Q8_0_WMMA_VK128_W64_OPSEL 0
#endif

#ifndef HRX_Q8_0_WMMA_VK128_W64_H4LOAD
#define HRX_Q8_0_WMMA_VK128_W64_H4LOAD 0
#endif

#ifndef HRX_Q8_0_WMMA_VK128_W64_B64GROUP
#define HRX_Q8_0_WMMA_VK128_W64_B64GROUP 0
#endif

#ifndef HRX_Q8_0_WMMA_VK128_W64_B64GROUP_NOWAIT
#define HRX_Q8_0_WMMA_VK128_W64_B64GROUP_NOWAIT 0
#endif

#ifndef HRX_Q8_0_WMMA_VK128_W64_B64GROUP_STREAM_COL
#define HRX_Q8_0_WMMA_VK128_W64_B64GROUP_STREAM_COL 0
#endif

#ifndef HRX_Q8_0_WMMA_VK128_W64_B64GROUP_STREAM_ROW
#define HRX_Q8_0_WMMA_VK128_W64_B64GROUP_STREAM_ROW 0
#endif

#ifndef HRX_Q8_0_WMMA_VK128_W64_B64GROUP_NAMED_FRAGS
#define HRX_Q8_0_WMMA_VK128_W64_B64GROUP_NAMED_FRAGS 0
#endif

#ifndef HRX_Q8_0_WMMA_VK128_W64_B64GROUP_STAGED_WAIT
#define HRX_Q8_0_WMMA_VK128_W64_B64GROUP_STAGED_WAIT 0
#endif

#ifndef HRX_Q8_0_WMMA_VK128_W64_B64GROUP_STAGED_WAIT0
#define HRX_Q8_0_WMMA_VK128_W64_B64GROUP_STAGED_WAIT0 12
#endif

#ifndef HRX_Q8_0_WMMA_VK128_W64_B64GROUP_STAGED_WAIT1
#define HRX_Q8_0_WMMA_VK128_W64_B64GROUP_STAGED_WAIT1 8
#endif

#ifndef HRX_Q8_0_WMMA_VK128_W64_B64GROUP_STAGED_WAIT2
#define HRX_Q8_0_WMMA_VK128_W64_B64GROUP_STAGED_WAIT2 4
#endif

#ifndef HRX_Q8_0_WMMA_VK128_W64_B64GROUP_STAGED_WAIT3
#define HRX_Q8_0_WMMA_VK128_W64_B64GROUP_STAGED_WAIT3 0
#endif

#ifndef HRX_Q8_0_WMMA_VK128_W64_B64GROUP_PREUSE_FRAGS
#define HRX_Q8_0_WMMA_VK128_W64_B64GROUP_PREUSE_FRAGS 0
#endif

#ifndef HRX_Q8_0_WMMA_VK128_W64_B64GROUP_DEPENDENT_WAIT
#define HRX_Q8_0_WMMA_VK128_W64_B64GROUP_DEPENDENT_WAIT 0
#endif

#ifndef HRX_Q8_0_WMMA_VK128_W64_B64GROUP_DEPENDENT_WAIT_K2
#define HRX_Q8_0_WMMA_VK128_W64_B64GROUP_DEPENDENT_WAIT_K2 0
#endif

#ifndef HRX_Q8_0_WMMA_VK128_W64_B64GROUP_DEPENDENT_WAIT_K2_PREUSE
#define HRX_Q8_0_WMMA_VK128_W64_B64GROUP_DEPENDENT_WAIT_K2_PREUSE 1
#endif

#ifndef HRX_Q8_0_WMMA_VK128_W64_B64GROUP_DEPENDENT_WAIT_K2_DIRECT
#define HRX_Q8_0_WMMA_VK128_W64_B64GROUP_DEPENDENT_WAIT_K2_DIRECT 0
#endif

#ifndef HRX_Q8_0_WMMA_VK128_W64_B64GROUP_DEPENDENT_WAIT_K2_RAW
#define HRX_Q8_0_WMMA_VK128_W64_B64GROUP_DEPENDENT_WAIT_K2_RAW 0
#endif

#ifndef HRX_Q8_0_WMMA_VK128_W64_MEDIUMFRAG12_COMBINED96
#define HRX_Q8_0_WMMA_VK128_W64_MEDIUMFRAG12_COMBINED96 0
#endif

#ifndef HRX_Q8_0_WMMA_VK128_W64_MOTIF192
#define HRX_Q8_0_WMMA_VK128_W64_MOTIF192 0
#endif

#ifndef HRX_Q8_0_WMMA_VK128_W64_PHASE96
#define HRX_Q8_0_WMMA_VK128_W64_PHASE96 0
#endif

#ifndef HRX_Q8_0_WMMA_VK128_W64_PHASE96_GROUP_BASE
#define HRX_Q8_0_WMMA_VK128_W64_PHASE96_GROUP_BASE 0
#endif

#ifndef HRX_Q8_0_WMMA_VK128_W64_PHASE96_COL_START
#define HRX_Q8_0_WMMA_VK128_W64_PHASE96_COL_START 0
#endif

#ifndef HRX_Q8_0_WMMA_VK128_W64_PHASE96_COL_COUNT
#define HRX_Q8_0_WMMA_VK128_W64_PHASE96_COL_COUNT 2
#endif

#ifndef HRX_Q8_0_WMMA_VK128_STORE_STAGE
#define HRX_Q8_0_WMMA_VK128_STORE_STAGE 0
#endif

#ifndef HRX_Q8_0_WMMA_VK128_STORE_STAGE_DUAL_HALF
#define HRX_Q8_0_WMMA_VK128_STORE_STAGE_DUAL_HALF 0
#endif

#ifndef HRX_Q8_0_WMMA_VK128_STORE_STAGE_FAST_HALF
#define HRX_Q8_0_WMMA_VK128_STORE_STAGE_FAST_HALF 0
#endif

#ifndef HRX_Q8_0_WMMA_VK128_STORE_STAGE_FAST_HALF_SELECTED
#define HRX_Q8_0_WMMA_VK128_STORE_STAGE_FAST_HALF_SELECTED 0
#endif

#ifndef HRX_Q8_0_WMMA_VK128_STORE_STAGE_FAST_HALF_SPLIT_SELECTED
#define HRX_Q8_0_WMMA_VK128_STORE_STAGE_FAST_HALF_SPLIT_SELECTED 0
#endif

#ifndef HRX_Q8_0_WMMA_VK128_STAGE_ALLOC
#define HRX_Q8_0_WMMA_VK128_STAGE_ALLOC 0
#endif

#ifndef HRX_Q8_0_WMMA_VK128_FULL_TILE_STORE
#define HRX_Q8_0_WMMA_VK128_FULL_TILE_STORE 0
#endif

#ifndef HRX_Q8_0_WMMA_VK128_FULL_TILE_STORE_PAIR
#define HRX_Q8_0_WMMA_VK128_FULL_TILE_STORE_PAIR 0
#endif

#ifndef HRX_Q8_0_WMMA_VK128_BUFFER_STORE
#define HRX_Q8_0_WMMA_VK128_BUFFER_STORE 0
#endif

#ifndef HRX_Q8_0_WMMA_VK128_PACK_STAGE_B32
#define HRX_Q8_0_WMMA_VK128_PACK_STAGE_B32 0
#endif

#ifndef HRX_Q8_0_WMMA_VK128_FAST_HALF_DUMMY_LOAD
#define HRX_Q8_0_WMMA_VK128_FAST_HALF_DUMMY_LOAD 0
#endif

#ifndef HRX_Q8_0_WMMA_VK128_COPY_B_FRAG
#define HRX_Q8_0_WMMA_VK128_COPY_B_FRAG 0
#endif

#ifndef HRX_Q8_0_WMMA_VK128_COPY_A_FRAG
#define HRX_Q8_0_WMMA_VK128_COPY_A_FRAG 0
#endif

#ifndef HRX_Q8_0_WMMA_VK128_COPY_B_FRAG_MIN_COL_SUB
#define HRX_Q8_0_WMMA_VK128_COPY_B_FRAG_MIN_COL_SUB 0
#endif

#ifndef HRX_Q8_0_WMMA_VK128_USE_ASM_WMMA
#define HRX_Q8_0_WMMA_VK128_USE_ASM_WMMA 0
#endif

#ifndef HRX_Q8_0_WMMA_VK128_ASSUME_FULL_TILES
#define HRX_Q8_0_WMMA_VK128_ASSUME_FULL_TILES 0
#endif

#ifndef HRX_Q8_0_WMMA_VK128_LAUNCH_MIN_BLOCKS
#define HRX_Q8_0_WMMA_VK128_LAUNCH_MIN_BLOCKS 1
#endif

struct hrx_block_q8_0_wmma_vk128_lhs {
    unsigned short d;
    int8_t qs[32];
};

typedef _Float16 hrx_q8_0_wmma_vk128_half16_vec __attribute__((ext_vector_type(16)));
typedef _Float16 hrx_q8_0_wmma_vk128_half8_vec __attribute__((ext_vector_type(8)));
typedef _Float16 hrx_q8_0_wmma_vk128_half4_vec __attribute__((ext_vector_type(4)));
typedef uint32_t hrx_q8_0_wmma_vk128_u32x8_vec __attribute__((ext_vector_type(8)));
typedef uint64_t hrx_q8_0_wmma_vk128_u64x4_vec __attribute__((ext_vector_type(4)));
typedef const __attribute__((address_space(3))) _Float16 * hrx_q8_0_wmma_vk128_lds_half_ptr;
typedef volatile __attribute__((address_space(3))) _Float16 * hrx_q8_0_wmma_vk128_lds_volatile_half_ptr;
typedef __attribute__((address_space(3))) uint16_t * hrx_q8_0_wmma_vk128_lds_u16_ptr;
typedef const __attribute__((address_space(3))) uint16_t * hrx_q8_0_wmma_vk128_lds_const_u16_ptr;
typedef __attribute__((address_space(3))) uint32_t * hrx_q8_0_wmma_vk128_lds_u32_ptr;

#if HRX_Q8_0_WMMA_VK128_COPY_A_FRAG || HRX_Q8_0_WMMA_VK128_COPY_B_FRAG
static __device__ __forceinline__ hrx_q8_0_wmma_vk128_half16_vec hrx_q8_0_wmma_vk128_copy_frag(
        hrx_q8_0_wmma_vk128_half16_vec frag) {
    const hrx_q8_0_wmma_vk128_u32x8_vec in =
        __builtin_bit_cast(hrx_q8_0_wmma_vk128_u32x8_vec, frag);
    hrx_q8_0_wmma_vk128_u32x8_vec out;
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
                   "v"(in[4]), "v"(in[5]), "v"(in[6]), "v"(in[7])
                 : "memory");
    return __builtin_bit_cast(hrx_q8_0_wmma_vk128_half16_vec, out);
}
#endif

#if HRX_Q8_0_WMMA_VK128_W64_B64GROUP_PREUSE_FRAGS
static __device__ __forceinline__ void hrx_q8_0_wmma_vk128_preuse_fragments(
        hrx_q8_0_wmma_vk128_half16_vec a0,
        hrx_q8_0_wmma_vk128_half16_vec a1,
        hrx_q8_0_wmma_vk128_half16_vec a2,
        hrx_q8_0_wmma_vk128_half16_vec a3,
        hrx_q8_0_wmma_vk128_half16_vec b0,
        hrx_q8_0_wmma_vk128_half16_vec b1,
        hrx_q8_0_wmma_vk128_half16_vec b2,
        hrx_q8_0_wmma_vk128_half16_vec b3) {
    asm volatile("" ::
        "v"(a0), "v"(a1), "v"(a2), "v"(a3),
        "v"(b0), "v"(b1), "v"(b2), "v"(b3) : "memory");
}
#endif

#if HRX_Q8_0_WMMA_VK128_W64_B64GROUP_DEPENDENT_WAIT_K2
static __device__ __forceinline__ void hrx_q8_0_wmma_vk128_preuse_fragments_k2(
        hrx_q8_0_wmma_vk128_half16_vec a00,
        hrx_q8_0_wmma_vk128_half16_vec a01,
        hrx_q8_0_wmma_vk128_half16_vec a02,
        hrx_q8_0_wmma_vk128_half16_vec a03,
        hrx_q8_0_wmma_vk128_half16_vec b00,
        hrx_q8_0_wmma_vk128_half16_vec b01,
        hrx_q8_0_wmma_vk128_half16_vec b02,
        hrx_q8_0_wmma_vk128_half16_vec b03,
        hrx_q8_0_wmma_vk128_half16_vec a10,
        hrx_q8_0_wmma_vk128_half16_vec a11,
        hrx_q8_0_wmma_vk128_half16_vec a12,
        hrx_q8_0_wmma_vk128_half16_vec a13,
        hrx_q8_0_wmma_vk128_half16_vec b10,
        hrx_q8_0_wmma_vk128_half16_vec b11,
        hrx_q8_0_wmma_vk128_half16_vec b12,
        hrx_q8_0_wmma_vk128_half16_vec b13) {
    asm volatile("" ::
        "v"(a00), "v"(a01), "v"(a02), "v"(a03),
        "v"(b00), "v"(b01), "v"(b02), "v"(b03),
        "v"(a10), "v"(a11), "v"(a12), "v"(a13),
        "v"(b10), "v"(b11), "v"(b12), "v"(b13) : "memory");
}
#endif

#if HRX_Q8_0_WMMA_VK128_W64_B64GROUP_DEPENDENT_WAIT_K2_RAW
static __device__ __forceinline__ void hrx_q8_0_wmma_vk128_preuse_raw_fragments_k2(
        hrx_q8_0_wmma_vk128_u64x4_vec a00,
        hrx_q8_0_wmma_vk128_u64x4_vec a01,
        hrx_q8_0_wmma_vk128_u64x4_vec a02,
        hrx_q8_0_wmma_vk128_u64x4_vec a03,
        hrx_q8_0_wmma_vk128_u64x4_vec b00,
        hrx_q8_0_wmma_vk128_u64x4_vec b01,
        hrx_q8_0_wmma_vk128_u64x4_vec b02,
        hrx_q8_0_wmma_vk128_u64x4_vec b03,
        hrx_q8_0_wmma_vk128_u64x4_vec a10,
        hrx_q8_0_wmma_vk128_u64x4_vec a11,
        hrx_q8_0_wmma_vk128_u64x4_vec a12,
        hrx_q8_0_wmma_vk128_u64x4_vec a13,
        hrx_q8_0_wmma_vk128_u64x4_vec b10,
        hrx_q8_0_wmma_vk128_u64x4_vec b11,
        hrx_q8_0_wmma_vk128_u64x4_vec b12,
        hrx_q8_0_wmma_vk128_u64x4_vec b13) {
    asm volatile("" ::
        "v"(a00), "v"(a01), "v"(a02), "v"(a03),
        "v"(b00), "v"(b01), "v"(b02), "v"(b03),
        "v"(a10), "v"(a11), "v"(a12), "v"(a13),
        "v"(b10), "v"(b11), "v"(b12), "v"(b13) : "memory");
}

static __device__ __forceinline__ hrx_q8_0_wmma_vk128_half16_vec
hrx_q8_0_wmma_vk128_raw_frag_to_half16(hrx_q8_0_wmma_vk128_u64x4_vec raw) {
    return __builtin_bit_cast(hrx_q8_0_wmma_vk128_half16_vec, raw);
}
#endif

#if HRX_Q8_0_WMMA_VK128_W64_B64GROUP_DEPENDENT_WAIT
template <int wait_lgkmcnt>
static __device__ __forceinline__ hrx_q8_0_wmma_vk128_half16_vec hrx_q8_0_wmma_vk128_dep_copy_initial(
        hrx_q8_0_wmma_vk128_half16_vec src,
        hrx_q8_0_wmma_vk128_half16_vec dep0,
        hrx_q8_0_wmma_vk128_half16_vec dep1,
        hrx_q8_0_wmma_vk128_half16_vec dep2,
        hrx_q8_0_wmma_vk128_half16_vec dep3,
        hrx_q8_0_wmma_vk128_half16_vec dep4,
        hrx_q8_0_wmma_vk128_half16_vec dep5,
        hrx_q8_0_wmma_vk128_half16_vec dep6) {
    const hrx_q8_0_wmma_vk128_u32x8_vec in =
        __builtin_bit_cast(hrx_q8_0_wmma_vk128_u32x8_vec, src);
    hrx_q8_0_wmma_vk128_u32x8_vec out;
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
    return __builtin_bit_cast(hrx_q8_0_wmma_vk128_half16_vec, out);
}

template <int wait_lgkmcnt>
static __device__ __forceinline__ hrx_q8_0_wmma_vk128_half16_vec hrx_q8_0_wmma_vk128_dep_copy_after_acc(
        hrx_q8_0_wmma_vk128_half16_vec src,
        hrx_q8_0_wmma_vk128_half16_vec token,
        hrx_q8_0_wmma_vk128_half8_vec prev_acc) {
    const hrx_q8_0_wmma_vk128_u32x8_vec in =
        __builtin_bit_cast(hrx_q8_0_wmma_vk128_u32x8_vec, src);
    hrx_q8_0_wmma_vk128_u32x8_vec out;
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
    return __builtin_bit_cast(hrx_q8_0_wmma_vk128_half16_vec, out);
}

static __device__ __forceinline__ hrx_q8_0_wmma_vk128_half16_vec hrx_q8_0_wmma_vk128_dep_copy_after_token(
        hrx_q8_0_wmma_vk128_half16_vec src,
        hrx_q8_0_wmma_vk128_half16_vec token) {
    const hrx_q8_0_wmma_vk128_u32x8_vec in =
        __builtin_bit_cast(hrx_q8_0_wmma_vk128_u32x8_vec, src);
    hrx_q8_0_wmma_vk128_u32x8_vec out;
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
    return __builtin_bit_cast(hrx_q8_0_wmma_vk128_half16_vec, out);
}

#define HRX_Q8_0_WMMA_VK128_DEP_WMMA_INITIAL(TILE, A, B, W, D0, D1, D2, D3, D4, D5, D6) \
    do { \
        const hrx_q8_0_wmma_vk128_half16_vec a_dep = \
            hrx_q8_0_wmma_vk128_dep_copy_initial<W>((A), (D0), (D1), (D2), (D3), (D4), (D5), (D6)); \
        const hrx_q8_0_wmma_vk128_half16_vec b_dep = \
            hrx_q8_0_wmma_vk128_dep_copy_after_token((B), a_dep); \
        acc[TILE] = hrx_q8_0_wmma_vk128_wmma_f16_w64(a_dep, b_dep, acc[TILE]); \
    } while (0)

#define HRX_Q8_0_WMMA_VK128_DEP_WMMA_AFTER(TILE, A, B, W, TOKEN, PREV) \
    do { \
        const hrx_q8_0_wmma_vk128_half16_vec a_dep = \
            hrx_q8_0_wmma_vk128_dep_copy_after_acc<W>((A), (TOKEN), acc[PREV]); \
        const hrx_q8_0_wmma_vk128_half16_vec b_dep = \
            hrx_q8_0_wmma_vk128_dep_copy_after_token((B), a_dep); \
        acc[TILE] = hrx_q8_0_wmma_vk128_wmma_f16_w64(a_dep, b_dep, acc[TILE]); \
    } while (0)
#endif

static __device__ __forceinline__ hrx_q8_0_wmma_vk128_half8_vec hrx_q8_0_wmma_vk128_wmma_f16_w64(
        hrx_q8_0_wmma_vk128_half16_vec a_frag,
        hrx_q8_0_wmma_vk128_half16_vec b_frag,
        hrx_q8_0_wmma_vk128_half8_vec acc) {
#if HRX_Q8_0_WMMA_VK128_USE_ASM_WMMA
    hrx_q8_0_wmma_vk128_half8_vec out;
#if HRX_Q8_0_WMMA_VK128_W64_OPSEL
    asm volatile("v_wmma_f16_16x16x16_f16 %0, %1, %2, %3 op_sel:[0,0,1]\n"
                 : "=v"(out)
                 : "v"(a_frag), "v"(b_frag), "v"(acc)
                 : "memory");
#else
    asm volatile("v_wmma_f16_16x16x16_f16 %0, %1, %2, %3\n"
                 : "=v"(out)
                 : "v"(a_frag), "v"(b_frag), "v"(acc)
                 : "memory");
#endif
    return out;
#else
    return __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(
        a_frag,
        b_frag,
        acc,
        HRX_Q8_0_WMMA_VK128_W64_OPSEL != 0);
#endif
}

#define HRX_Q8_0_WMMA_VK128_WAIT_WMMA(TILE, A, B, W) \
    do { \
        asm volatile("s_waitcnt lgkmcnt(%0)\n" :: "n"(W) : "memory"); \
        acc[TILE] = hrx_q8_0_wmma_vk128_wmma_f16_w64((A), (B), acc[TILE]); \
    } while (0)

#if HRX_Q8_0_WMMA_VK128_BUFFER_STORE
static constexpr int HRX_Q8_0_WMMA_VK128_RAW_BUFFER_FLAGS_GFX11 = 0x31004000;

static __device__ __forceinline__ __amdgpu_buffer_rsrc_t hrx_q8_0_wmma_vk128_make_dst_rsrc(float * dst) {
    return __builtin_amdgcn_make_buffer_rsrc(
        dst,
        static_cast<unsigned short>(0),
        0xffffffffull,
        HRX_Q8_0_WMMA_VK128_RAW_BUFFER_FLAGS_GFX11);
}

static __device__ __forceinline__ void hrx_q8_0_wmma_vk128_buffer_store_f32(
        __amdgpu_buffer_rsrc_t dst_rsrc,
        long long elem_offset,
        float value) {
    __builtin_amdgcn_raw_buffer_store_b32(
        __builtin_bit_cast(int, value),
        dst_rsrc,
        static_cast<int>(elem_offset * static_cast<long long>(sizeof(float))),
        0,
        0);
}
#endif

static __device__ __forceinline__ uint32_t hrx_q8_0_wmma_vk128_pack_f16x2(_Float16 lo, _Float16 hi) {
    union {
        _Float16 h[2];
        uint32_t u;
    } pack;
    pack.h[0] = lo;
    pack.h[1] = hi;
    return pack.u;
}

static __device__ __forceinline__ _Float16 hrx_q8_0_wmma_vk128_unpack_f16x2(uint32_t bits, int idx) {
    union {
        uint32_t u;
        _Float16 h[2];
    } pack;
    pack.u = bits;
    return pack.h[idx];
}

static __device__ __forceinline__ uint16_t hrx_q8_0_wmma_vk128_f16_to_u16(_Float16 value) {
    union {
        _Float16 h;
        uint16_t u;
    } pack;
    pack.h = value;
    return pack.u;
}

static __device__ __forceinline__ _Float16 hrx_q8_0_wmma_vk128_u16_to_f16(uint32_t value) {
    union {
        uint16_t u;
        _Float16 h;
    } pack;
    pack.u = static_cast<uint16_t>(value);
    return pack.h;
}

static __device__ __forceinline__ uint32_t hrx_q8_0_wmma_vk128_f16_pair_to_u32(_Float16 lo, _Float16 hi) {
    union {
        _Float16 h[2];
        uint32_t u;
    } pack;
    pack.h[0] = lo;
    pack.h[1] = hi;
    return pack.u;
}

#if HRX_Q8_0_WMMA_VK128_PACK_STAGE_B32
static __device__ __forceinline__ void hrx_q8_0_wmma_vk128_ds_store_u32(
        hrx_q8_0_wmma_vk128_lds_u32_ptr ptr,
        uint32_t value) {
    asm volatile("ds_write_b32 %0, %1 offset:0\n"
                 :
                 : "v"(ptr), "v"(value)
                 : "memory");
}
#endif

static __device__ __forceinline__ hrx_q8_0_wmma_vk128_half16_vec hrx_q8_0_wmma_vk128_duplicate_input(
        _Float16 x0, _Float16 x1, _Float16 x2, _Float16 x3,
        _Float16 x4, _Float16 x5, _Float16 x6, _Float16 x7) {
    constexpr int SWAP16_CTRL = (16 << 10) | 0x1f;
    const uint32_t p0 = hrx_q8_0_wmma_vk128_pack_f16x2(x0, x1);
    const uint32_t p1 = hrx_q8_0_wmma_vk128_pack_f16x2(x2, x3);
    const uint32_t p2 = hrx_q8_0_wmma_vk128_pack_f16x2(x4, x5);
    const uint32_t p3 = hrx_q8_0_wmma_vk128_pack_f16x2(x6, x7);
    const uint32_t s0 = static_cast<uint32_t>(__builtin_amdgcn_ds_swizzle(static_cast<int32_t>(p0), SWAP16_CTRL));
    const uint32_t s1 = static_cast<uint32_t>(__builtin_amdgcn_ds_swizzle(static_cast<int32_t>(p1), SWAP16_CTRL));
    const uint32_t s2 = static_cast<uint32_t>(__builtin_amdgcn_ds_swizzle(static_cast<int32_t>(p2), SWAP16_CTRL));
    const uint32_t s3 = static_cast<uint32_t>(__builtin_amdgcn_ds_swizzle(static_cast<int32_t>(p3), SWAP16_CTRL));

    hrx_q8_0_wmma_vk128_half16_vec result;
    result[0] = x0;
    result[1] = x1;
    result[2] = x2;
    result[3] = x3;
    result[4] = x4;
    result[5] = x5;
    result[6] = x6;
    result[7] = x7;
    result[8] = hrx_q8_0_wmma_vk128_unpack_f16x2(s0, 0);
    result[9] = hrx_q8_0_wmma_vk128_unpack_f16x2(s0, 1);
    result[10] = hrx_q8_0_wmma_vk128_unpack_f16x2(s1, 0);
    result[11] = hrx_q8_0_wmma_vk128_unpack_f16x2(s1, 1);
    result[12] = hrx_q8_0_wmma_vk128_unpack_f16x2(s2, 0);
    result[13] = hrx_q8_0_wmma_vk128_unpack_f16x2(s2, 1);
    result[14] = hrx_q8_0_wmma_vk128_unpack_f16x2(s3, 0);
    result[15] = hrx_q8_0_wmma_vk128_unpack_f16x2(s3, 1);
    return result;
}

static __device__ __forceinline__ _Float16 hrx_q8_0_wmma_vk128_load_a_value(
        const hrx_block_q8_0_wmma_vk128_lhs * src0,
        long long row,
        long long k_index,
        long long blocks_per_row) {
    const hrx_block_q8_0_wmma_vk128_lhs * block = src0 + row * blocks_per_row + (k_index >> 5);
    const float d = __half2float(__ushort_as_half(block->d));
    return static_cast<_Float16>(d * static_cast<float>(block->qs[k_index & 31]));
}

static __device__ __forceinline__ hrx_q8_0_wmma_vk128_half16_vec hrx_q8_0_wmma_vk128_load_a_frag(
        const _Float16 * sh_a,
        int row_tile,
        int k_tile,
        unsigned int lane) {
    constexpr int SHARED_STRIDE = HRX_Q8_0_WMMA_VK128_SHARED_STRIDE;
    const int row = row_tile * 16 + static_cast<int>(lane & 15u);
    const int k_base = k_tile * 16 + static_cast<int>(lane >> 4) * 8;
    return hrx_q8_0_wmma_vk128_duplicate_input(
        sh_a[row * SHARED_STRIDE + k_base + 0],
        sh_a[row * SHARED_STRIDE + k_base + 1],
        sh_a[row * SHARED_STRIDE + k_base + 2],
        sh_a[row * SHARED_STRIDE + k_base + 3],
        sh_a[row * SHARED_STRIDE + k_base + 4],
        sh_a[row * SHARED_STRIDE + k_base + 5],
        sh_a[row * SHARED_STRIDE + k_base + 6],
        sh_a[row * SHARED_STRIDE + k_base + 7]);
}

static __device__ __forceinline__ hrx_q8_0_wmma_vk128_half16_vec hrx_q8_0_wmma_vk128_load_b_frag(
        const _Float16 * sh_b,
        int col_tile,
        int k_tile,
        unsigned int lane) {
    constexpr int SHARED_STRIDE = HRX_Q8_0_WMMA_VK128_SHARED_STRIDE;
    const int col = col_tile * 16 + static_cast<int>(lane & 15u);
    const int k_base = k_tile * 16 + static_cast<int>(lane >> 4) * 8;
    return hrx_q8_0_wmma_vk128_duplicate_input(
        sh_b[col * SHARED_STRIDE + k_base + 0],
        sh_b[col * SHARED_STRIDE + k_base + 1],
        sh_b[col * SHARED_STRIDE + k_base + 2],
        sh_b[col * SHARED_STRIDE + k_base + 3],
        sh_b[col * SHARED_STRIDE + k_base + 4],
        sh_b[col * SHARED_STRIDE + k_base + 5],
        sh_b[col * SHARED_STRIDE + k_base + 6],
        sh_b[col * SHARED_STRIDE + k_base + 7]);
}

static __device__ __forceinline__ hrx_q8_0_wmma_vk128_half16_vec hrx_q8_0_wmma_vk128_load_a_frag_w64(
        const _Float16 * sh_a,
        int row_tile,
        int k_tile,
        unsigned int lane) {
    constexpr int SHARED_STRIDE = HRX_Q8_0_WMMA_VK128_SHARED_STRIDE;
    const int row = row_tile * 16 + static_cast<int>(lane & 15u);
    const int k_base = k_tile * 16;
    hrx_q8_0_wmma_vk128_half16_vec result;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        result[i] = sh_a[row * SHARED_STRIDE + k_base + i];
    }
    return result;
}

static __device__ __forceinline__ hrx_q8_0_wmma_vk128_half16_vec hrx_q8_0_wmma_vk128_load_b_frag_w64(
        const _Float16 * sh_b,
        int col_tile,
        int k_tile,
        unsigned int lane) {
    constexpr int SHARED_STRIDE = HRX_Q8_0_WMMA_VK128_SHARED_STRIDE;
    const int col = col_tile * 16 + static_cast<int>(lane & 15u);
    const int k_base = k_tile * 16;
    hrx_q8_0_wmma_vk128_half16_vec result;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        result[i] = sh_b[col * SHARED_STRIDE + k_base + i];
    }
    return result;
}

static __device__ __forceinline__ void hrx_q8_0_wmma_vk128_append_half4(
        hrx_q8_0_wmma_vk128_half16_vec * result,
        int base,
        hrx_q8_0_wmma_vk128_half4_vec values) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        (*result)[base + i] = values[i];
    }
}

static __device__ __forceinline__ hrx_q8_0_wmma_vk128_half4_vec hrx_q8_0_wmma_vk128_ds_read_b64_h4(
        hrx_q8_0_wmma_vk128_lds_half_ptr ptr) {
    const __attribute__((address_space(3))) uint64_t * lds_ptr =
        (const __attribute__((address_space(3))) uint64_t *) ptr;
    hrx_q8_0_wmma_vk128_half4_vec value;
    asm volatile("ds_read_b64 %0, %1 offset:0\n"
                 : "=v"(value)
                 : "v"(lds_ptr)
                 : "memory");
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    return value;
}

static __device__ __forceinline__ hrx_q8_0_wmma_vk128_half4_vec hrx_q8_0_wmma_vk128_ds_read_b64_h4_nowait(
        hrx_q8_0_wmma_vk128_lds_half_ptr ptr) {
    const __attribute__((address_space(3))) uint64_t * lds_ptr =
        (const __attribute__((address_space(3))) uint64_t *) ptr;
    hrx_q8_0_wmma_vk128_half4_vec value;
    asm volatile("ds_read_b64 %0, %1 offset:0\n"
                 : "=v"(value)
                 : "v"(lds_ptr)
                 : "memory");
    return value;
}

static __device__ __forceinline__ uint64_t hrx_q8_0_wmma_vk128_ds_read_b64_u64_nowait(
        hrx_q8_0_wmma_vk128_lds_half_ptr ptr) {
    const __attribute__((address_space(3))) uint64_t * lds_ptr =
        (const __attribute__((address_space(3))) uint64_t *) ptr;
    uint64_t value;
    asm volatile("ds_read_b64 %0, %1 offset:0\n"
                 : "=v"(value)
                 : "v"(lds_ptr)
                 : "memory");
    return value;
}

static __device__ __forceinline__ hrx_q8_0_wmma_vk128_half16_vec hrx_q8_0_wmma_vk128_load_a_frag_w64_b64asm(
        hrx_q8_0_wmma_vk128_lds_half_ptr sh_a,
        int row_tile,
        int k_tile,
        unsigned int lane) {
    constexpr int SHARED_STRIDE = HRX_Q8_0_WMMA_VK128_SHARED_STRIDE;
    const int row = row_tile * 16 + static_cast<int>(lane & 15u);
    const int k_base = k_tile * 16;
    hrx_q8_0_wmma_vk128_lds_half_ptr row_ptr = sh_a + row * SHARED_STRIDE + k_base;
    hrx_q8_0_wmma_vk128_half16_vec result;
    hrx_q8_0_wmma_vk128_append_half4(&result, 0, hrx_q8_0_wmma_vk128_ds_read_b64_h4(row_ptr + 0));
    hrx_q8_0_wmma_vk128_append_half4(&result, 4, hrx_q8_0_wmma_vk128_ds_read_b64_h4(row_ptr + 4));
    hrx_q8_0_wmma_vk128_append_half4(&result, 8, hrx_q8_0_wmma_vk128_ds_read_b64_h4(row_ptr + 8));
    hrx_q8_0_wmma_vk128_append_half4(&result, 12, hrx_q8_0_wmma_vk128_ds_read_b64_h4(row_ptr + 12));
    return result;
}

static __device__ __forceinline__ hrx_q8_0_wmma_vk128_half16_vec hrx_q8_0_wmma_vk128_load_a_frag_w64_b64asm_nowait(
        hrx_q8_0_wmma_vk128_lds_half_ptr sh_a,
        int row_tile,
        int k_tile,
        unsigned int lane) {
    constexpr int SHARED_STRIDE = HRX_Q8_0_WMMA_VK128_SHARED_STRIDE;
    const int row = row_tile * 16 + static_cast<int>(lane & 15u);
    const int k_base = k_tile * 16;
    hrx_q8_0_wmma_vk128_lds_half_ptr row_ptr = sh_a + row * SHARED_STRIDE + k_base;
    hrx_q8_0_wmma_vk128_half16_vec result;
    hrx_q8_0_wmma_vk128_append_half4(&result, 0, hrx_q8_0_wmma_vk128_ds_read_b64_h4_nowait(row_ptr + 0));
    hrx_q8_0_wmma_vk128_append_half4(&result, 4, hrx_q8_0_wmma_vk128_ds_read_b64_h4_nowait(row_ptr + 4));
    hrx_q8_0_wmma_vk128_append_half4(&result, 8, hrx_q8_0_wmma_vk128_ds_read_b64_h4_nowait(row_ptr + 8));
    hrx_q8_0_wmma_vk128_append_half4(&result, 12, hrx_q8_0_wmma_vk128_ds_read_b64_h4_nowait(row_ptr + 12));
    return result;
}

static __device__ __forceinline__ hrx_q8_0_wmma_vk128_u64x4_vec hrx_q8_0_wmma_vk128_load_a_raw_frag_w64_b64asm_nowait(
        hrx_q8_0_wmma_vk128_lds_half_ptr sh_a,
        int row_tile,
        int k_tile,
        unsigned int lane) {
    constexpr int SHARED_STRIDE = HRX_Q8_0_WMMA_VK128_SHARED_STRIDE;
    const int row = row_tile * 16 + static_cast<int>(lane & 15u);
    const int k_base = k_tile * 16;
    hrx_q8_0_wmma_vk128_lds_half_ptr row_ptr = sh_a + row * SHARED_STRIDE + k_base;
    hrx_q8_0_wmma_vk128_u64x4_vec result;
    result[0] = hrx_q8_0_wmma_vk128_ds_read_b64_u64_nowait(row_ptr + 0);
    result[1] = hrx_q8_0_wmma_vk128_ds_read_b64_u64_nowait(row_ptr + 4);
    result[2] = hrx_q8_0_wmma_vk128_ds_read_b64_u64_nowait(row_ptr + 8);
    result[3] = hrx_q8_0_wmma_vk128_ds_read_b64_u64_nowait(row_ptr + 12);
    return result;
}

static __device__ __forceinline__ hrx_q8_0_wmma_vk128_half16_vec hrx_q8_0_wmma_vk128_load_b_frag_w64_b64asm(
        hrx_q8_0_wmma_vk128_lds_half_ptr sh_b,
        int col_tile,
        int k_tile,
        unsigned int lane) {
    constexpr int SHARED_STRIDE = HRX_Q8_0_WMMA_VK128_SHARED_STRIDE;
    const int col = col_tile * 16 + static_cast<int>(lane & 15u);
    const int k_base = k_tile * 16;
    hrx_q8_0_wmma_vk128_lds_half_ptr col_ptr = sh_b + col * SHARED_STRIDE + k_base;
    hrx_q8_0_wmma_vk128_half16_vec result;
    hrx_q8_0_wmma_vk128_append_half4(&result, 0, hrx_q8_0_wmma_vk128_ds_read_b64_h4(col_ptr + 0));
    hrx_q8_0_wmma_vk128_append_half4(&result, 4, hrx_q8_0_wmma_vk128_ds_read_b64_h4(col_ptr + 4));
    hrx_q8_0_wmma_vk128_append_half4(&result, 8, hrx_q8_0_wmma_vk128_ds_read_b64_h4(col_ptr + 8));
    hrx_q8_0_wmma_vk128_append_half4(&result, 12, hrx_q8_0_wmma_vk128_ds_read_b64_h4(col_ptr + 12));
    return result;
}

static __device__ __forceinline__ hrx_q8_0_wmma_vk128_half16_vec hrx_q8_0_wmma_vk128_load_b_frag_w64_b64asm_nowait(
        hrx_q8_0_wmma_vk128_lds_half_ptr sh_b,
        int col_tile,
        int k_tile,
        unsigned int lane) {
    constexpr int SHARED_STRIDE = HRX_Q8_0_WMMA_VK128_SHARED_STRIDE;
    const int col = col_tile * 16 + static_cast<int>(lane & 15u);
    const int k_base = k_tile * 16;
    hrx_q8_0_wmma_vk128_lds_half_ptr col_ptr = sh_b + col * SHARED_STRIDE + k_base;
    hrx_q8_0_wmma_vk128_half16_vec result;
    hrx_q8_0_wmma_vk128_append_half4(&result, 0, hrx_q8_0_wmma_vk128_ds_read_b64_h4_nowait(col_ptr + 0));
    hrx_q8_0_wmma_vk128_append_half4(&result, 4, hrx_q8_0_wmma_vk128_ds_read_b64_h4_nowait(col_ptr + 4));
    hrx_q8_0_wmma_vk128_append_half4(&result, 8, hrx_q8_0_wmma_vk128_ds_read_b64_h4_nowait(col_ptr + 8));
    hrx_q8_0_wmma_vk128_append_half4(&result, 12, hrx_q8_0_wmma_vk128_ds_read_b64_h4_nowait(col_ptr + 12));
    return result;
}

static __device__ __forceinline__ hrx_q8_0_wmma_vk128_u64x4_vec hrx_q8_0_wmma_vk128_load_b_raw_frag_w64_b64asm_nowait(
        hrx_q8_0_wmma_vk128_lds_half_ptr sh_b,
        int col_tile,
        int k_tile,
        unsigned int lane) {
    constexpr int SHARED_STRIDE = HRX_Q8_0_WMMA_VK128_SHARED_STRIDE;
    const int col = col_tile * 16 + static_cast<int>(lane & 15u);
    const int k_base = k_tile * 16;
    hrx_q8_0_wmma_vk128_lds_half_ptr col_ptr = sh_b + col * SHARED_STRIDE + k_base;
    hrx_q8_0_wmma_vk128_u64x4_vec result;
    result[0] = hrx_q8_0_wmma_vk128_ds_read_b64_u64_nowait(col_ptr + 0);
    result[1] = hrx_q8_0_wmma_vk128_ds_read_b64_u64_nowait(col_ptr + 4);
    result[2] = hrx_q8_0_wmma_vk128_ds_read_b64_u64_nowait(col_ptr + 8);
    result[3] = hrx_q8_0_wmma_vk128_ds_read_b64_u64_nowait(col_ptr + 12);
    return result;
}

static __device__ __forceinline__ hrx_q8_0_wmma_vk128_half16_vec hrx_q8_0_wmma_vk128_load_a_frag_w64_h4(
        const _Float16 * sh_a,
        int row_tile,
        int k_tile,
        unsigned int lane) {
    constexpr int SHARED_STRIDE = HRX_Q8_0_WMMA_VK128_SHARED_STRIDE;
    const int row = row_tile * 16 + static_cast<int>(lane & 15u);
    const int k_base = k_tile * 16;
    const hrx_q8_0_wmma_vk128_half4_vec * row_ptr =
        reinterpret_cast<const hrx_q8_0_wmma_vk128_half4_vec *>(sh_a + row * SHARED_STRIDE + k_base);
    hrx_q8_0_wmma_vk128_half16_vec result;
    hrx_q8_0_wmma_vk128_append_half4(&result, 0, row_ptr[0]);
    hrx_q8_0_wmma_vk128_append_half4(&result, 4, row_ptr[1]);
    hrx_q8_0_wmma_vk128_append_half4(&result, 8, row_ptr[2]);
    hrx_q8_0_wmma_vk128_append_half4(&result, 12, row_ptr[3]);
    return result;
}

static __device__ __forceinline__ hrx_q8_0_wmma_vk128_half16_vec hrx_q8_0_wmma_vk128_load_b_frag_w64_h4(
        const _Float16 * sh_b,
        int col_tile,
        int k_tile,
        unsigned int lane) {
    constexpr int SHARED_STRIDE = HRX_Q8_0_WMMA_VK128_SHARED_STRIDE;
    const int col = col_tile * 16 + static_cast<int>(lane & 15u);
    const int k_base = k_tile * 16;
    const hrx_q8_0_wmma_vk128_half4_vec * col_ptr =
        reinterpret_cast<const hrx_q8_0_wmma_vk128_half4_vec *>(sh_b + col * SHARED_STRIDE + k_base);
    hrx_q8_0_wmma_vk128_half16_vec result;
    hrx_q8_0_wmma_vk128_append_half4(&result, 0, col_ptr[0]);
    hrx_q8_0_wmma_vk128_append_half4(&result, 4, col_ptr[1]);
    hrx_q8_0_wmma_vk128_append_half4(&result, 8, col_ptr[2]);
    hrx_q8_0_wmma_vk128_append_half4(&result, 12, col_ptr[3]);
    return result;
}

static __device__ __forceinline__ void hrx_q8_0_wmma_vk128_store_acc_f16_row_major(
        float * dst,
        long long rows_stride,
        long long row0,
        long long col0,
        long long rows,
        long long cols,
        hrx_q8_0_wmma_vk128_half16_vec acc,
        unsigned int lane) {
    const long long row_base = row0 + static_cast<long long>(lane >> 4);
    const long long col = col0 + static_cast<long long>(lane & 15u);
    if (col >= cols) {
        return;
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const long long row = row_base + static_cast<long long>(i * 2);
        if (row < rows) {
            dst[col * rows_stride + row] = static_cast<float>(acc[i * 2]);
        }
    }
}

static __device__ __forceinline__ void hrx_q8_0_wmma_vk128_store_acc_f16_row_major_w64(
        float * dst,
        long long rows_stride,
        long long row0,
        long long col0,
        long long rows,
        long long cols,
        hrx_q8_0_wmma_vk128_half8_vec acc,
        unsigned int lane) {
    const long long row_lane = static_cast<long long>(lane >> 4);
    const long long col = col0 + static_cast<long long>(lane & 15u);
    if (col >= cols) {
        return;
    }
#pragma unroll
    for (int reg = 0; reg < 4; ++reg) {
        const long long row = row0 + row_lane + static_cast<long long>(reg * 4);
        if (row < rows) {
            dst[col * rows_stride + row] =
                static_cast<float>(acc[reg * 2 + HRX_Q8_0_WMMA_VK128_W64_OPSEL]);
        }
    }
}

static __device__ __forceinline__ void hrx_q8_0_wmma_vk128_store_acc_f16_row_major_w64_full(
        float * dst,
        long long rows_stride,
        long long row0,
        long long col0,
        hrx_q8_0_wmma_vk128_half8_vec acc,
        unsigned int lane) {
    const long long row_lane = static_cast<long long>(lane >> 4);
    const long long col = col0 + static_cast<long long>(lane & 15u);
#pragma unroll
    for (int reg = 0; reg < 4; ++reg) {
        const long long row = row0 + row_lane + static_cast<long long>(reg * 4);
        dst[col * rows_stride + row] =
            static_cast<float>(acc[reg * 2 + HRX_Q8_0_WMMA_VK128_W64_OPSEL]);
    }
}

#if HRX_Q8_0_WMMA_VK128_BUFFER_STORE
static __device__ __forceinline__ void hrx_q8_0_wmma_vk128_store_acc_f16_row_major_w64_buffer_full(
        __amdgpu_buffer_rsrc_t dst_rsrc,
        long long rows_stride,
        long long row0,
        long long col0,
        hrx_q8_0_wmma_vk128_half8_vec acc,
        unsigned int lane) {
    const long long row_lane = static_cast<long long>(lane >> 4);
    const long long col = col0 + static_cast<long long>(lane & 15u);
#pragma unroll
    for (int reg = 0; reg < 4; ++reg) {
        const long long row = row0 + row_lane + static_cast<long long>(reg * 4);
        hrx_q8_0_wmma_vk128_buffer_store_f32(
            dst_rsrc,
            col * rows_stride + row,
            static_cast<float>(acc[reg * 2 + HRX_Q8_0_WMMA_VK128_W64_OPSEL]));
    }
}

#if HRX_Q8_0_WMMA_VK128_FULL_TILE_STORE_PAIR
static __device__ __forceinline__ void hrx_q8_0_wmma_vk128_store_acc_f16_row_major_w64_buffer_full_pair(
        __amdgpu_buffer_rsrc_t dst_rsrc,
        long long rows_stride,
        long long row0,
        long long col0,
        hrx_q8_0_wmma_vk128_half8_vec acc,
        unsigned int lane) {
    // Diagnostic only. gfx11 wave64 OPSEL high/low halves map to the same D
    // coordinates; row_hi is not a valid second row band for correctness.
    const long long row_lane = static_cast<long long>(lane >> 4);
    const long long col = col0 + static_cast<long long>(lane & 15u);
#pragma unroll
    for (int reg = 0; reg < 4; ++reg) {
        const long long row_lo = row0 + row_lane + static_cast<long long>(reg * 4);
        const long long row_hi = row_lo + 16;
        hrx_q8_0_wmma_vk128_buffer_store_f32(
            dst_rsrc,
            col * rows_stride + row_lo,
            static_cast<float>(acc[reg * 2 + 0]));
        hrx_q8_0_wmma_vk128_buffer_store_f32(
            dst_rsrc,
            col * rows_stride + row_hi,
            static_cast<float>(acc[reg * 2 + 1]));
    }
}
#endif

static __device__ __forceinline__ void hrx_q8_0_wmma_vk128_store_acc_f16_row_major_w64_buffer(
        __amdgpu_buffer_rsrc_t dst_rsrc,
        long long rows_stride,
        long long row0,
        long long col0,
        long long rows,
        long long cols,
        hrx_q8_0_wmma_vk128_half8_vec acc,
        unsigned int lane) {
    const long long row_lane = static_cast<long long>(lane >> 4);
    const long long col = col0 + static_cast<long long>(lane & 15u);
    if (col >= cols) {
        return;
    }
#pragma unroll
    for (int reg = 0; reg < 4; ++reg) {
        const long long row = row0 + row_lane + static_cast<long long>(reg * 4);
        if (row < rows) {
            hrx_q8_0_wmma_vk128_buffer_store_f32(
                dst_rsrc,
                col * rows_stride + row,
                static_cast<float>(acc[reg * 2 + HRX_Q8_0_WMMA_VK128_W64_OPSEL]));
        }
    }
}
#endif

static __device__ __forceinline__ void hrx_q8_0_wmma_vk128_store_acc_f16_row_major_w64_stage(
        float * dst,
        long long rows_stride,
        long long row0,
        long long col0,
        long long rows,
        long long cols,
        hrx_q8_0_wmma_vk128_half8_vec acc,
        unsigned int lane,
        unsigned int wave,
        _Float16 * sh_store) {
    constexpr int TILE_STRIDE = 16 * 16;
    const int row_lane = static_cast<int>(lane >> 4);
    const int col_lane = static_cast<int>(lane & 15u);
    const int tile_base = static_cast<int>(wave) * TILE_STRIDE;
    const int col_major_base = tile_base + col_lane * 16 + row_lane;
#pragma unroll
    for (int reg = 0; reg < 4; ++reg) {
        sh_store[col_major_base + reg * 4] = acc[reg * 2 + HRX_Q8_0_WMMA_VK128_W64_OPSEL];
    }
    __syncthreads();

    const long long col = col0 + static_cast<long long>(col_lane);
    if (col < cols) {
#pragma unroll
        for (int reg = 0; reg < 4; ++reg) {
            const long long row = row0 + static_cast<long long>(row_lane + reg * 4);
            if (row < rows) {
                dst[col * rows_stride + row] = static_cast<float>(sh_store[col_major_base + reg * 4]);
            }
        }
    }
    __syncthreads();
}

#if HRX_Q8_0_WMMA_VK128_STORE_STAGE_DUAL_HALF || HRX_Q8_0_WMMA_VK128_STORE_STAGE_FAST_HALF || HRX_Q8_0_WMMA_VK128_W64_PHASE96 || HRX_Q8_0_WMMA_VK128_W64_MOTIF192
static __device__ __forceinline__ void hrx_q8_0_wmma_vk128_ds_store_u16(
        hrx_q8_0_wmma_vk128_lds_u16_ptr ptr,
        uint16_t value) {
    asm volatile("ds_write_b16 %0, %1 offset:0\n"
                 :
                 : "v"(ptr), "v"(static_cast<uint32_t>(value))
                 : "memory");
}

static __device__ __forceinline__ uint32_t hrx_q8_0_wmma_vk128_ds_load_u16_d16(
        hrx_q8_0_wmma_vk128_lds_const_u16_ptr ptr) {
    uint32_t value = 0;
    asm volatile("ds_read_u16_d16 %0, %1 offset:0\n"
                 : "=v"(value)
                 : "v"(ptr)
                 : "memory");
    return value;
}
#endif

#if HRX_Q8_0_WMMA_VK128_STORE_STAGE_DUAL_HALF
static __device__ __forceinline__ void hrx_q8_0_wmma_vk128_store_acc_f16_row_major_w64_dual_half_stage(
#if HRX_Q8_0_WMMA_VK128_BUFFER_STORE
        __amdgpu_buffer_rsrc_t dst_rsrc,
#else
        float * dst,
#endif
        long long rows_stride,
        long long row0,
        long long col0,
        long long rows,
        long long cols,
        hrx_q8_0_wmma_vk128_half8_vec acc,
        unsigned int lane,
        unsigned int wave,
        hrx_q8_0_wmma_vk128_lds_volatile_half_ptr sh_store) {
    constexpr int TILE_STRIDE = 16 * 16;
    const int row_lane = static_cast<int>(lane >> 4);
    const int col_lane = static_cast<int>(lane & 15u);
    const int tile_base = static_cast<int>(wave) * TILE_STRIDE;
    const int col_major_base = tile_base + col_lane * 16 + row_lane;
    hrx_q8_0_wmma_vk128_lds_u16_ptr sh_u16 =
        (hrx_q8_0_wmma_vk128_lds_u16_ptr) sh_store;
    constexpr int SELECTED_OPSEL = HRX_Q8_0_WMMA_VK128_W64_OPSEL;
    constexpr int OTHER_OPSEL = SELECTED_OPSEL == 0 ? 1 : 0;
#pragma unroll
    for (int reg = 0; reg < 4; ++reg) {
        const int offset = col_major_base + reg * 4;
        hrx_q8_0_wmma_vk128_ds_store_u16(
            sh_u16 + offset,
            hrx_q8_0_wmma_vk128_f16_to_u16(acc[reg * 2 + OTHER_OPSEL]));
        hrx_q8_0_wmma_vk128_ds_store_u16(
            sh_u16 + offset,
            hrx_q8_0_wmma_vk128_f16_to_u16(acc[reg * 2 + SELECTED_OPSEL]));
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    const long long col = col0 + static_cast<long long>(col_lane);
    if (col < cols) {
#pragma unroll
        for (int reg = 0; reg < 4; ++reg) {
            const int offset = col_major_base + reg * 4;
            const _Float16 probe = hrx_q8_0_wmma_vk128_u16_to_f16(
                hrx_q8_0_wmma_vk128_ds_load_u16_d16(sh_u16 + offset));
            const _Float16 selected = hrx_q8_0_wmma_vk128_u16_to_f16(
                hrx_q8_0_wmma_vk128_ds_load_u16_d16(sh_u16 + offset));
            asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
            (void) probe;
            const long long row = row0 + static_cast<long long>(row_lane + reg * 4);
            if (row < rows) {
#if HRX_Q8_0_WMMA_VK128_BUFFER_STORE
                hrx_q8_0_wmma_vk128_buffer_store_f32(
                    dst_rsrc,
                    col * rows_stride + row,
                    static_cast<float>(selected));
#else
                dst[col * rows_stride + row] = static_cast<float>(selected);
#endif
            }
        }
    }
    __syncthreads();
}
#endif

#if HRX_Q8_0_WMMA_VK128_STORE_STAGE_FAST_HALF && HRX_Q8_0_WMMA_VK128_BUFFER_STORE
static __device__ __forceinline__ void hrx_q8_0_wmma_vk128_store_acc_f16_row_major_w64_fast_half_buffer_selected(
        __amdgpu_buffer_rsrc_t dst_rsrc,
        long long rows_stride,
        long long row0,
        long long col0,
        hrx_q8_0_wmma_vk128_half8_vec acc,
        unsigned int lane,
        unsigned int wave,
        hrx_q8_0_wmma_vk128_lds_volatile_half_ptr sh_store) {
    constexpr int TILE_STRIDE = 16 * 16;
    constexpr int SELECTED_OPSEL = HRX_Q8_0_WMMA_VK128_W64_OPSEL;
    constexpr int OTHER_OPSEL = SELECTED_OPSEL == 0 ? 1 : 0;
    const int row_lane = static_cast<int>(lane >> 4);
    const int col_lane = static_cast<int>(lane & 15u);
    const int tile_base = static_cast<int>(wave) * TILE_STRIDE;
    const int col_major_base = tile_base + col_lane * 16 + row_lane;
    hrx_q8_0_wmma_vk128_lds_u16_ptr sh_u16 =
        (hrx_q8_0_wmma_vk128_lds_u16_ptr) sh_store;
    const long long col = col0 + static_cast<long long>(col_lane);
#pragma unroll
    for (int reg = 0; reg < 4; ++reg) {
        const int selected_offset =
            col_major_base + ((reg * 4 + SELECTED_OPSEL * 2) & 15);
        const int other_offset =
            col_major_base + ((reg * 4 + OTHER_OPSEL * 2) & 15);
        hrx_q8_0_wmma_vk128_ds_store_u16(
            sh_u16 + selected_offset,
            hrx_q8_0_wmma_vk128_f16_to_u16(acc[reg * 2 + SELECTED_OPSEL]));
        hrx_q8_0_wmma_vk128_ds_store_u16(
            sh_u16 + other_offset,
            hrx_q8_0_wmma_vk128_f16_to_u16(acc[reg * 2 + OTHER_OPSEL]));
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");

#pragma unroll
    for (int reg = 0; reg < 4; ++reg) {
        const int selected_offset =
            col_major_base + ((reg * 4 + SELECTED_OPSEL * 2) & 15);
        const int other_offset =
            col_major_base + ((reg * 4 + OTHER_OPSEL * 2) & 15);
        const _Float16 selected = hrx_q8_0_wmma_vk128_u16_to_f16(
            hrx_q8_0_wmma_vk128_ds_load_u16_d16(sh_u16 + selected_offset));
        const uint32_t other_bits = hrx_q8_0_wmma_vk128_ds_load_u16_d16(sh_u16 + other_offset);
        (void) other_bits;
        asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
        const long long row = row0 + static_cast<long long>(row_lane + reg * 4);
        hrx_q8_0_wmma_vk128_buffer_store_f32(
            dst_rsrc,
            col * rows_stride + row,
            static_cast<float>(selected));
    }
}

static __device__ __forceinline__ void hrx_q8_0_wmma_vk128_store_acc_f16_row_major_w64_fast_half_buffer_split_selected(
        __amdgpu_buffer_rsrc_t dst_rsrc,
        long long rows_stride,
        long long row0,
        long long col0,
        hrx_q8_0_wmma_vk128_half8_vec acc,
        unsigned int lane,
        unsigned int wave,
        hrx_q8_0_wmma_vk128_lds_volatile_half_ptr sh_store) {
    constexpr int TILE_STRIDE = 16 * 16;
    constexpr int SELECTED_OPSEL = HRX_Q8_0_WMMA_VK128_W64_OPSEL;
    constexpr int OTHER_OPSEL = SELECTED_OPSEL == 0 ? 1 : 0;
    const int row_lane = static_cast<int>(lane >> 4);
    const int col_lane = static_cast<int>(lane & 15u);
    const int tile_base = static_cast<int>(wave) * TILE_STRIDE;
    const int col_major_base = tile_base + col_lane * 16 + row_lane;
    const long long col = col0 + static_cast<long long>(col_lane);
    hrx_q8_0_wmma_vk128_lds_u16_ptr sh_u16 =
        (hrx_q8_0_wmma_vk128_lds_u16_ptr) sh_store;

#pragma unroll
    for (int reg_base = 0; reg_base < 4; reg_base += 2) {
#pragma unroll
        for (int reg_delta = 0; reg_delta < 2; ++reg_delta) {
            const int reg = reg_base + reg_delta;
            const int selected_offset =
                col_major_base + ((reg * 4 + SELECTED_OPSEL * 2) & 15);
            const int other_offset =
                col_major_base + ((reg * 4 + OTHER_OPSEL * 2) & 15);
            hrx_q8_0_wmma_vk128_ds_store_u16(
                sh_u16 + selected_offset,
                hrx_q8_0_wmma_vk128_f16_to_u16(acc[reg * 2 + SELECTED_OPSEL]));
            hrx_q8_0_wmma_vk128_ds_store_u16(
                sh_u16 + other_offset,
                hrx_q8_0_wmma_vk128_f16_to_u16(acc[reg * 2 + OTHER_OPSEL]));
        }
        asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");

#pragma unroll
        for (int reg_delta = 0; reg_delta < 2; ++reg_delta) {
            const int reg = reg_base + reg_delta;
            const int selected_offset =
                col_major_base + ((reg * 4 + SELECTED_OPSEL * 2) & 15);
            const int other_offset =
                col_major_base + ((reg * 4 + OTHER_OPSEL * 2) & 15);
            const _Float16 selected = hrx_q8_0_wmma_vk128_u16_to_f16(
                hrx_q8_0_wmma_vk128_ds_load_u16_d16(sh_u16 + selected_offset));
            const uint32_t other_bits = hrx_q8_0_wmma_vk128_ds_load_u16_d16(sh_u16 + other_offset);
            (void) other_bits;
            asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
            const long long row = row0 + static_cast<long long>(row_lane + reg * 4);
            hrx_q8_0_wmma_vk128_buffer_store_f32(
                dst_rsrc,
                col * rows_stride + row,
                static_cast<float>(selected));
        }
    }
}

static __device__ __forceinline__ void hrx_q8_0_wmma_vk128_store_acc_f16_row_major_w64_fast_half_buffer_full_pair(
        __amdgpu_buffer_rsrc_t dst_rsrc,
        long long rows_stride,
        long long row0,
        long long col0,
        hrx_q8_0_wmma_vk128_half8_vec acc,
        unsigned int lane,
        unsigned int wave,
        hrx_q8_0_wmma_vk128_lds_volatile_half_ptr sh_store) {
    // Diagnostic only. The high-half write is useful for static RADV store
    // surface experiments, but does not represent another 16x16 output band.
    constexpr int TILE_STRIDE = 16 * 16;
    const int row_lane = static_cast<int>(lane >> 4);
    const int col_lane = static_cast<int>(lane & 15u);
    const int tile_base = static_cast<int>(wave) * TILE_STRIDE;
    const int col_major_base = tile_base + col_lane * 16 + row_lane;
    hrx_q8_0_wmma_vk128_lds_u16_ptr sh_u16 =
        (hrx_q8_0_wmma_vk128_lds_u16_ptr) sh_store;
    const long long col = col0 + static_cast<long long>(col_lane);
#pragma unroll
    for (int reg = 0; reg < 4; ++reg) {
        const int lo_offset = col_major_base + reg * 4;
        const int hi_offset = col_major_base + ((reg * 4 + 2) & 15);
        hrx_q8_0_wmma_vk128_ds_store_u16(
            sh_u16 + lo_offset,
            hrx_q8_0_wmma_vk128_f16_to_u16(acc[reg * 2 + 0]));
        hrx_q8_0_wmma_vk128_ds_store_u16(
            sh_u16 + hi_offset,
            hrx_q8_0_wmma_vk128_f16_to_u16(acc[reg * 2 + 1]));
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");

#pragma unroll
    for (int reg = 0; reg < 4; ++reg) {
        const int lo_offset = col_major_base + reg * 4;
        const int hi_offset = col_major_base + ((reg * 4 + 2) & 15);
#if HRX_Q8_0_WMMA_VK128_FAST_HALF_DUMMY_LOAD
        const uint32_t lo_bits = hrx_q8_0_wmma_vk128_ds_load_u16_d16(sh_u16 + lo_offset);
        const uint32_t hi_bits = hrx_q8_0_wmma_vk128_ds_load_u16_d16(sh_u16 + hi_offset);
        (void) lo_bits;
        (void) hi_bits;
        asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
        const float lo = static_cast<float>(acc[reg * 2 + 0]);
        const float hi = static_cast<float>(acc[reg * 2 + 1]);
#else
        const _Float16 lo = hrx_q8_0_wmma_vk128_u16_to_f16(
            hrx_q8_0_wmma_vk128_ds_load_u16_d16(sh_u16 + lo_offset));
        const _Float16 hi = hrx_q8_0_wmma_vk128_u16_to_f16(
            hrx_q8_0_wmma_vk128_ds_load_u16_d16(sh_u16 + hi_offset));
        asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
#endif
        const long long row_lo = row0 + static_cast<long long>(row_lane + reg * 4);
        const long long row_hi = row_lo + 16;
        hrx_q8_0_wmma_vk128_buffer_store_f32(
            dst_rsrc,
            col * rows_stride + row_lo,
            lo);
        hrx_q8_0_wmma_vk128_buffer_store_f32(
            dst_rsrc,
            col * rows_stride + row_hi,
            hi);
    }
}
#endif

#if HRX_Q8_0_WMMA_VK128_W64_MEDIUMFRAG12_COMBINED96 || HRX_Q8_0_WMMA_VK128_W64_PHASE96 || HRX_Q8_0_WMMA_VK128_W64_MOTIF192
static __device__ __forceinline__ hrx_q8_0_wmma_vk128_lds_u16_ptr hrx_q8_0_wmma_vk128_combined96_stage_ptr(
        _Float16 * sh_store,
        int index) {
    return ((hrx_q8_0_wmma_vk128_lds_u16_ptr) sh_store) + index;
}

static __device__ __forceinline__ hrx_q8_0_wmma_vk128_lds_const_u16_ptr hrx_q8_0_wmma_vk128_combined96_stage_const_ptr(
        const _Float16 * sh_store,
        int index) {
    return ((hrx_q8_0_wmma_vk128_lds_const_u16_ptr) sh_store) + index;
}

static __device__ __forceinline__ int hrx_q8_0_wmma_vk128_combined96_stage_index(
        unsigned int wave,
        int group,
        int slot,
        unsigned int lane) {
    const int stage_group = group - 8;
    const int row_lane = static_cast<int>(lane >> 4);
    const int col_lane = static_cast<int>(lane & 15u);
    return static_cast<int>(wave) * 16 * 16 * 16 +
        stage_group * 16 * 16 + col_lane * 16 + row_lane + slot * 4;
}

static __device__ __forceinline__ void hrx_q8_0_wmma_vk128_combined96_raw_store_slot(
        __amdgpu_buffer_rsrc_t dst_rsrc,
        long long rows_stride,
        long long row_base,
        long long col_base,
        long long rows,
        long long cols,
        const hrx_q8_0_wmma_vk128_half8_vec * acc,
        unsigned int wave,
        int group,
        int slot,
        unsigned int lane) {
    const int wave_row = static_cast<int>(wave & 1u);
    const int wave_col = static_cast<int>(wave >> 1);
    const int row_lane = static_cast<int>(lane >> 4);
    const int col_lane = static_cast<int>(lane & 15u);
    const int acc_index = group & 7;
    const long long row = row_base + static_cast<long long>(wave_row * 64 + (group & 3) * 16 + row_lane + slot * 4);
    const long long col = col_base + static_cast<long long>(wave_col * 64 + ((group >> 2) & 3) * 16 + col_lane);
    if (row < rows && col < cols) {
        hrx_q8_0_wmma_vk128_buffer_store_f32(
            dst_rsrc,
            col * rows_stride + row,
            static_cast<float>(acc[acc_index][slot * 2 + HRX_Q8_0_WMMA_VK128_W64_OPSEL]));
    }
}

static __device__ __forceinline__ void hrx_q8_0_wmma_vk128_combined96_stage_store_slot(
        _Float16 * sh_store,
        const hrx_q8_0_wmma_vk128_half8_vec * acc,
        unsigned int wave,
        int group,
        int slot,
        unsigned int lane) {
    const int acc_index = group & 7;
    hrx_q8_0_wmma_vk128_ds_store_u16(
        hrx_q8_0_wmma_vk128_combined96_stage_ptr(
            sh_store, hrx_q8_0_wmma_vk128_combined96_stage_index(wave, group, slot, lane)),
        hrx_q8_0_wmma_vk128_f16_to_u16(acc[acc_index][slot * 2 + HRX_Q8_0_WMMA_VK128_W64_OPSEL]));
}

static __device__ __forceinline__ void hrx_q8_0_wmma_vk128_combined96_stage_load_store_slot(
        __amdgpu_buffer_rsrc_t dst_rsrc,
        long long rows_stride,
        long long row_base,
        long long col_base,
        long long rows,
        long long cols,
        const _Float16 * sh_store,
        unsigned int wave,
        int group,
        int slot,
        unsigned int lane) {
    const int wave_row = static_cast<int>(wave & 1u);
    const int wave_col = static_cast<int>(wave >> 1);
    const int row_lane = static_cast<int>(lane >> 4);
    const int col_lane = static_cast<int>(lane & 15u);
    const long long row = row_base + static_cast<long long>(wave_row * 64 + (group & 3) * 16 + row_lane + slot * 4);
    const long long col = col_base + static_cast<long long>(wave_col * 64 + ((group >> 2) & 3) * 16 + col_lane);
    const _Float16 value = hrx_q8_0_wmma_vk128_u16_to_f16(
        hrx_q8_0_wmma_vk128_ds_load_u16_d16(
            hrx_q8_0_wmma_vk128_combined96_stage_const_ptr(
                sh_store, hrx_q8_0_wmma_vk128_combined96_stage_index(wave, group, slot, lane))));
    if (row < rows && col < cols) {
        hrx_q8_0_wmma_vk128_buffer_store_f32(dst_rsrc, col * rows_stride + row, static_cast<float>(value));
    }
}

#define HRX_Q8_0_WMMA_VK128_COMBINED96_RAW_STORE_GROUP(GROUP_ID) do { \
    hrx_q8_0_wmma_vk128_combined96_raw_store_slot(dst_rsrc, rows, row_base, col_base, rows, cols, acc, wave, (GROUP_ID), 0, lane); \
    hrx_q8_0_wmma_vk128_combined96_raw_store_slot(dst_rsrc, rows, row_base, col_base, rows, cols, acc, wave, (GROUP_ID), 1, lane); \
    hrx_q8_0_wmma_vk128_combined96_raw_store_slot(dst_rsrc, rows, row_base, col_base, rows, cols, acc, wave, (GROUP_ID), 2, lane); \
    hrx_q8_0_wmma_vk128_combined96_raw_store_slot(dst_rsrc, rows, row_base, col_base, rows, cols, acc, wave, (GROUP_ID), 3, lane); \
} while (0)

#define HRX_Q8_0_WMMA_VK128_COMBINED96_STAGE_STORE_GROUP(GROUP_ID) do { \
    hrx_q8_0_wmma_vk128_combined96_stage_store_slot(sh_store, acc, wave, (GROUP_ID), 0, lane); \
    hrx_q8_0_wmma_vk128_combined96_stage_store_slot(sh_store, acc, wave, (GROUP_ID), 1, lane); \
    hrx_q8_0_wmma_vk128_combined96_stage_store_slot(sh_store, acc, wave, (GROUP_ID), 2, lane); \
    hrx_q8_0_wmma_vk128_combined96_stage_store_slot(sh_store, acc, wave, (GROUP_ID), 3, lane); \
} while (0)

#define HRX_Q8_0_WMMA_VK128_COMBINED96_STAGE_LOAD_STORE_GROUP(GROUP_ID) do { \
    hrx_q8_0_wmma_vk128_combined96_stage_load_store_slot(dst_rsrc, rows, row_base, col_base, rows, cols, sh_store, wave, (GROUP_ID), 0, lane); \
    hrx_q8_0_wmma_vk128_combined96_stage_load_store_slot(dst_rsrc, rows, row_base, col_base, rows, cols, sh_store, wave, (GROUP_ID), 1, lane); \
    hrx_q8_0_wmma_vk128_combined96_stage_load_store_slot(dst_rsrc, rows, row_base, col_base, rows, cols, sh_store, wave, (GROUP_ID), 2, lane); \
    hrx_q8_0_wmma_vk128_combined96_stage_load_store_slot(dst_rsrc, rows, row_base, col_base, rows, cols, sh_store, wave, (GROUP_ID), 3, lane); \
} while (0)

#define HRX_Q8_0_WMMA_VK128_COMBINED96_GROUPS_0_7(MACRO) do { \
    MACRO(0); MACRO(1); MACRO(2); MACRO(3); \
    MACRO(4); MACRO(5); MACRO(6); MACRO(7); \
} while (0)

#define HRX_Q8_0_WMMA_VK128_COMBINED96_GROUPS_8_23(MACRO) do { \
    MACRO(8);  MACRO(9);  MACRO(10); MACRO(11); \
    MACRO(12); MACRO(13); MACRO(14); MACRO(15); \
    MACRO(16); MACRO(17); MACRO(18); MACRO(19); \
    MACRO(20); MACRO(21); MACRO(22); MACRO(23); \
} while (0)

#define HRX_Q8_0_WMMA_VK128_PHASE96_RAW_STORE_SLOT(GROUP_ID, SLOT_ID) do { \
    hrx_q8_0_wmma_vk128_combined96_raw_store_slot(dst_rsrc, rows, row_base, col_base, rows, cols, acc, wave, (GROUP_ID), (SLOT_ID), lane); \
} while (0)

#define HRX_Q8_0_WMMA_VK128_PHASE96_RAW_STORE_GROUP(GROUP_ID) do { \
    HRX_Q8_0_WMMA_VK128_PHASE96_RAW_STORE_SLOT((GROUP_ID), 0); \
    HRX_Q8_0_WMMA_VK128_PHASE96_RAW_STORE_SLOT((GROUP_ID), 1); \
    HRX_Q8_0_WMMA_VK128_PHASE96_RAW_STORE_SLOT((GROUP_ID), 2); \
    HRX_Q8_0_WMMA_VK128_PHASE96_RAW_STORE_SLOT((GROUP_ID), 3); \
} while (0)

#define HRX_Q8_0_WMMA_VK128_PHASE96_GROUPS_0_7(MACRO) do { \
    MACRO(0); MACRO(1); MACRO(2); MACRO(3); \
    MACRO(4); MACRO(5); MACRO(6); MACRO(7); \
} while (0)

#define HRX_Q8_0_WMMA_VK128_PHASE96_GROUPS_8_15(MACRO) do { \
    MACRO(8);  MACRO(9);  MACRO(10); MACRO(11); \
    MACRO(12); MACRO(13); MACRO(14); MACRO(15); \
} while (0)

#if HRX_Q8_0_WMMA_VK128_W64_MOTIF192
static __device__ __forceinline__ int hrx_q8_0_wmma_vk128_motif192_stage_index(
        unsigned int wave,
        int group,
        int slot,
        unsigned int lane) {
    const int group16 = group & 15;
    const int row_lane = static_cast<int>(lane >> 4);
    const int col_lane = static_cast<int>(lane & 15u);
    return static_cast<int>(wave) * 16 * 16 + col_lane * 16 + row_lane + slot * 4;
}

static __device__ __forceinline__ void hrx_q8_0_wmma_vk128_motif192_raw_store_slot(
        __amdgpu_buffer_rsrc_t dst_rsrc,
        long long rows_stride,
        long long row_base,
        long long col_base,
        long long rows,
        long long cols,
        const hrx_q8_0_wmma_vk128_half8_vec * acc,
        unsigned int wave,
        int group,
        int slot,
        unsigned int lane) {
    const int group16 = group & 15;
    hrx_q8_0_wmma_vk128_combined96_raw_store_slot(
        dst_rsrc, rows_stride, row_base, col_base, rows, cols, acc, wave, group16, slot, lane);
}

static __device__ __forceinline__ void hrx_q8_0_wmma_vk128_motif192_stage_store_slot(
        _Float16 * sh_store,
        const hrx_q8_0_wmma_vk128_half8_vec * acc,
        unsigned int wave,
        int group,
        int slot,
        unsigned int lane) {
    const int acc_index = group & 15;
    hrx_q8_0_wmma_vk128_ds_store_u16(
        hrx_q8_0_wmma_vk128_combined96_stage_ptr(
            sh_store, hrx_q8_0_wmma_vk128_motif192_stage_index(wave, group, slot, lane)),
        hrx_q8_0_wmma_vk128_f16_to_u16(acc[acc_index][slot * 2 + HRX_Q8_0_WMMA_VK128_W64_OPSEL]));
}

static __device__ __forceinline__ void hrx_q8_0_wmma_vk128_motif192_stage_load_store_slot(
        __amdgpu_buffer_rsrc_t dst_rsrc,
        long long rows_stride,
        long long row_base,
        long long col_base,
        long long rows,
        long long cols,
        const _Float16 * sh_store,
        unsigned int wave,
        int group,
        int slot,
        unsigned int lane) {
    const int group16 = group & 15;
    const int wave_row = static_cast<int>(wave & 1u);
    const int wave_col = static_cast<int>(wave >> 1);
    const int row_lane = static_cast<int>(lane >> 4);
    const int col_lane = static_cast<int>(lane & 15u);
    const long long row = row_base + static_cast<long long>(wave_row * 64 + (group16 & 3) * 16 + row_lane + slot * 4);
    const long long col = col_base + static_cast<long long>(wave_col * 64 + ((group16 >> 2) & 3) * 16 + col_lane);
    const _Float16 value = hrx_q8_0_wmma_vk128_u16_to_f16(
        hrx_q8_0_wmma_vk128_ds_load_u16_d16(
            hrx_q8_0_wmma_vk128_combined96_stage_const_ptr(
                sh_store, hrx_q8_0_wmma_vk128_motif192_stage_index(wave, group, slot, lane))));
    if (row < rows && col < cols) {
        hrx_q8_0_wmma_vk128_buffer_store_f32(dst_rsrc, col * rows_stride + row, static_cast<float>(value));
    }
}

#define HRX_Q8_0_WMMA_VK128_MOTIF192_RAW_STORE_GROUP(GROUP_ID) do { \
    hrx_q8_0_wmma_vk128_motif192_raw_store_slot(dst_rsrc, rows, row_base, col_base, rows, cols, acc, wave, (GROUP_ID), 0, lane); \
    hrx_q8_0_wmma_vk128_motif192_raw_store_slot(dst_rsrc, rows, row_base, col_base, rows, cols, acc, wave, (GROUP_ID), 1, lane); \
    hrx_q8_0_wmma_vk128_motif192_raw_store_slot(dst_rsrc, rows, row_base, col_base, rows, cols, acc, wave, (GROUP_ID), 2, lane); \
    hrx_q8_0_wmma_vk128_motif192_raw_store_slot(dst_rsrc, rows, row_base, col_base, rows, cols, acc, wave, (GROUP_ID), 3, lane); \
} while (0)

#define HRX_Q8_0_WMMA_VK128_MOTIF192_STAGE_STORE_GROUP(GROUP_ID) do { \
    hrx_q8_0_wmma_vk128_motif192_stage_store_slot(sh_store, acc, wave, (GROUP_ID), 0, lane); \
    hrx_q8_0_wmma_vk128_motif192_stage_store_slot(sh_store, acc, wave, (GROUP_ID), 1, lane); \
    hrx_q8_0_wmma_vk128_motif192_stage_store_slot(sh_store, acc, wave, (GROUP_ID), 2, lane); \
    hrx_q8_0_wmma_vk128_motif192_stage_store_slot(sh_store, acc, wave, (GROUP_ID), 3, lane); \
} while (0)

#define HRX_Q8_0_WMMA_VK128_MOTIF192_STAGE_LOAD_STORE_GROUP(GROUP_ID) do { \
    hrx_q8_0_wmma_vk128_motif192_stage_load_store_slot(dst_rsrc, rows, row_base, col_base, rows, cols, sh_store, wave, (GROUP_ID), 0, lane); \
    hrx_q8_0_wmma_vk128_motif192_stage_load_store_slot(dst_rsrc, rows, row_base, col_base, rows, cols, sh_store, wave, (GROUP_ID), 1, lane); \
    hrx_q8_0_wmma_vk128_motif192_stage_load_store_slot(dst_rsrc, rows, row_base, col_base, rows, cols, sh_store, wave, (GROUP_ID), 2, lane); \
    hrx_q8_0_wmma_vk128_motif192_stage_load_store_slot(dst_rsrc, rows, row_base, col_base, rows, cols, sh_store, wave, (GROUP_ID), 3, lane); \
} while (0)

#define HRX_Q8_0_WMMA_VK128_MOTIF192_GROUP(GROUP_ID) do { \
    HRX_Q8_0_WMMA_VK128_MOTIF192_RAW_STORE_GROUP((GROUP_ID)); \
    HRX_Q8_0_WMMA_VK128_MOTIF192_STAGE_STORE_GROUP((GROUP_ID) + 16); \
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory"); \
    HRX_Q8_0_WMMA_VK128_MOTIF192_STAGE_LOAD_STORE_GROUP((GROUP_ID) + 16); \
    HRX_Q8_0_WMMA_VK128_MOTIF192_STAGE_STORE_GROUP((GROUP_ID) + 32); \
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory"); \
    HRX_Q8_0_WMMA_VK128_MOTIF192_STAGE_LOAD_STORE_GROUP((GROUP_ID) + 32); \
} while (0)

#define HRX_Q8_0_WMMA_VK128_MOTIF192_GROUPS_0_15() do { \
    HRX_Q8_0_WMMA_VK128_MOTIF192_GROUP(0);  HRX_Q8_0_WMMA_VK128_MOTIF192_GROUP(1); \
    HRX_Q8_0_WMMA_VK128_MOTIF192_GROUP(2);  HRX_Q8_0_WMMA_VK128_MOTIF192_GROUP(3); \
    HRX_Q8_0_WMMA_VK128_MOTIF192_GROUP(4);  HRX_Q8_0_WMMA_VK128_MOTIF192_GROUP(5); \
    HRX_Q8_0_WMMA_VK128_MOTIF192_GROUP(6);  HRX_Q8_0_WMMA_VK128_MOTIF192_GROUP(7); \
    HRX_Q8_0_WMMA_VK128_MOTIF192_GROUP(8);  HRX_Q8_0_WMMA_VK128_MOTIF192_GROUP(9); \
    HRX_Q8_0_WMMA_VK128_MOTIF192_GROUP(10); HRX_Q8_0_WMMA_VK128_MOTIF192_GROUP(11); \
    HRX_Q8_0_WMMA_VK128_MOTIF192_GROUP(12); HRX_Q8_0_WMMA_VK128_MOTIF192_GROUP(13); \
    HRX_Q8_0_WMMA_VK128_MOTIF192_GROUP(14); HRX_Q8_0_WMMA_VK128_MOTIF192_GROUP(15); \
} while (0)
#endif
#endif

static __device__ __forceinline__ void hrx_q8_0_wmma_vk128_tile_map(
        int wave,
        int tile_iter,
        int * row_tile,
        int * col_tile) {
    constexpr int ROW_TILES = 8;
    constexpr int WAVE_COUNT = 8;
#if HRX_Q8_0_WMMA_VK128_PAIR64_TILE_MAP
    const int pair = wave >> 1;
    const int lane_wave = wave & 1;
    const int tile = lane_wave + tile_iter * 2;
    const int pair_row = pair & 1;
    const int pair_col = pair >> 1;
    *row_tile = pair_row * 4 + (tile & 3);
    *col_tile = pair_col * 4 + (tile >> 2);
#elif HRX_Q8_0_WMMA_VK128_W64
    const int wave_row = wave & 1;
    const int wave_col = wave >> 1;
    *row_tile = wave_row * 4 + (tile_iter & 3);
    *col_tile = wave_col * 4 + (tile_iter >> 2);
#else
    const int tile = wave + tile_iter * WAVE_COUNT;
    *row_tile = tile & (ROW_TILES - 1);
    *col_tile = tile >> 3;
#endif
}

extern "C" __global__ __launch_bounds__(256, HRX_Q8_0_WMMA_VK128_LAUNCH_MIN_BLOCKS)
void HRX_Q8_0_WMMA_VK128_EXPORT(
        const hrx_block_q8_0_wmma_vk128_lhs * src0,
        const float * src1,
        float * dst,
        long long k,
        long long rows,
        long long cols) {
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 32;
    constexpr int SHARED_STRIDE = HRX_Q8_0_WMMA_VK128_SHARED_STRIDE;
#if HRX_Q8_0_WMMA_VK128_W64
    constexpr int WAVE = 64;
#else
    constexpr int WAVE = 32;
#endif
    constexpr int ROW_TILES = BM / 16;
    constexpr int COL_TILES = BN / 16;
    constexpr int TILE_COUNT = ROW_TILES * COL_TILES;
    constexpr int WAVE_COUNT = 256 / WAVE;
    constexpr int TILES_PER_WAVE = TILE_COUNT / WAVE_COUNT;

    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int wave = tid / WAVE;
    const unsigned int lane = tid & static_cast<unsigned int>(WAVE - 1);
    const long long row_base = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * BM;
    const long long col_base = static_cast<long long>(__builtin_amdgcn_workgroup_id_y()) * BN;
    if (row_base >= rows || col_base >= cols) {
        return;
    }

#if HRX_Q8_0_WMMA_VK128_BUFFER_STORE
    const __amdgpu_buffer_rsrc_t dst_rsrc = hrx_q8_0_wmma_vk128_make_dst_rsrc(dst);
#endif

    __shared__ _Float16 sh_a[BM * SHARED_STRIDE];
    __shared__ _Float16 sh_b[BN * SHARED_STRIDE];
#if HRX_Q8_0_WMMA_VK128_W64_MEDIUMFRAG12_COMBINED96
    __shared__ _Float16 sh_store[WAVE_COUNT * 16 * 16 * 16];
#elif HRX_Q8_0_WMMA_VK128_W64_MOTIF192
    __shared__ _Float16 sh_store[WAVE_COUNT * 16 * 16];
#elif HRX_Q8_0_WMMA_VK128_STORE_STAGE
    __shared__ _Float16 sh_store[WAVE_COUNT * 16 * 16];
#elif HRX_Q8_0_WMMA_VK128_STORE_STAGE_DUAL_HALF
    __shared__ _Float16 sh_store[WAVE_COUNT * 16 * 16];
#elif HRX_Q8_0_WMMA_VK128_STORE_STAGE_FAST_HALF
    __shared__ _Float16 sh_store[WAVE_COUNT * 16 * 16];
#elif HRX_Q8_0_WMMA_VK128_STAGE_ALLOC
    __shared__ volatile _Float16 sh_stage_alloc[WAVE_COUNT * 16 * 16];
#endif

    const long long blocks_per_row = k / 32;
    const _Float16 zero = static_cast<_Float16>(0.0f);
#if HRX_Q8_0_WMMA_VK128_STAGE_ALLOC
    if (tid < WAVE_COUNT * 16 * 16) {
        sh_stage_alloc[tid] = zero;
    }
#endif
#if HRX_Q8_0_WMMA_VK128_W64
    constexpr int ACC_TILES =
        HRX_Q8_0_WMMA_VK128_W64_MOTIF192 ? TILES_PER_WAVE :
        (HRX_Q8_0_WMMA_VK128_W64_MEDIUMFRAG12_COMBINED96 || HRX_Q8_0_WMMA_VK128_W64_PHASE96) ?
        8 : TILES_PER_WAVE;
    hrx_q8_0_wmma_vk128_half8_vec acc[ACC_TILES] = {};
#else
    hrx_q8_0_wmma_vk128_half16_vec acc[TILES_PER_WAVE] = {};
#endif

    for (long long k0 = 0; k0 < k; k0 += BK) {
#if HRX_Q8_0_WMMA_VK128_PACK_STAGE_B32
        hrx_q8_0_wmma_vk128_lds_u32_ptr sh_a_u32 = (hrx_q8_0_wmma_vk128_lds_u32_ptr) sh_a;
        hrx_q8_0_wmma_vk128_lds_u32_ptr sh_b_u32 = (hrx_q8_0_wmma_vk128_lds_u32_ptr) sh_b;
        constexpr int PACKS_PER_K = BK / 2;
        constexpr int SHARED_STRIDE_PACKS = SHARED_STRIDE / 2;
        for (int idx = static_cast<int>(tid); idx < BM * PACKS_PER_K; idx += 256) {
            const int r = idx / PACKS_PER_K;
            const int kk_pair = idx - r * PACKS_PER_K;
            const int kk = kk_pair * 2;
            const long long row = row_base + static_cast<long long>(r);
#if HRX_Q8_0_WMMA_VK128_ASSUME_FULL_TILES
            const _Float16 a0 = hrx_q8_0_wmma_vk128_load_a_value(src0, row, k0 + kk + 0, blocks_per_row);
            const _Float16 a1 = hrx_q8_0_wmma_vk128_load_a_value(src0, row, k0 + kk + 1, blocks_per_row);
#else
            const _Float16 a0 = row < rows ?
                hrx_q8_0_wmma_vk128_load_a_value(src0, row, k0 + kk + 0, blocks_per_row) : zero;
            const _Float16 a1 = row < rows ?
                hrx_q8_0_wmma_vk128_load_a_value(src0, row, k0 + kk + 1, blocks_per_row) : zero;
#endif
            hrx_q8_0_wmma_vk128_ds_store_u32(
                sh_a_u32 + r * SHARED_STRIDE_PACKS + kk_pair,
                hrx_q8_0_wmma_vk128_f16_pair_to_u32(a0, a1));
        }
        for (int idx = static_cast<int>(tid); idx < BN * PACKS_PER_K; idx += 256) {
            const int c = idx / PACKS_PER_K;
            const int kk_pair = idx - c * PACKS_PER_K;
            const int kk = kk_pair * 2;
            const long long col = col_base + static_cast<long long>(c);
#if HRX_Q8_0_WMMA_VK128_ASSUME_FULL_TILES
            const _Float16 b0 = static_cast<_Float16>(src1[col * k + k0 + kk + 0]);
            const _Float16 b1 = static_cast<_Float16>(src1[col * k + k0 + kk + 1]);
#else
            const _Float16 b0 = col < cols ? static_cast<_Float16>(src1[col * k + k0 + kk + 0]) : zero;
            const _Float16 b1 = col < cols ? static_cast<_Float16>(src1[col * k + k0 + kk + 1]) : zero;
#endif
            hrx_q8_0_wmma_vk128_ds_store_u32(
                sh_b_u32 + c * SHARED_STRIDE_PACKS + kk_pair,
                hrx_q8_0_wmma_vk128_f16_pair_to_u32(b0, b1));
        }
#else
        for (int idx = static_cast<int>(tid); idx < BM * BK; idx += 256) {
            const int r = idx / BK;
            const int kk = idx - r * BK;
            const long long row = row_base + static_cast<long long>(r);
            sh_a[r * SHARED_STRIDE + kk] = row < rows ?
                hrx_q8_0_wmma_vk128_load_a_value(src0, row, k0 + kk, blocks_per_row) : zero;
        }
        for (int idx = static_cast<int>(tid); idx < BN * BK; idx += 256) {
            const int c = idx / BK;
            const int kk = idx - c * BK;
            const long long col = col_base + static_cast<long long>(c);
            sh_b[c * SHARED_STRIDE + kk] = col < cols ? static_cast<_Float16>(src1[col * k + k0 + kk]) : zero;
        }
#endif
        __syncthreads();

#if HRX_Q8_0_WMMA_VK128_W64_B64GROUP
        {
            hrx_q8_0_wmma_vk128_lds_half_ptr sh_a_lds =
                (hrx_q8_0_wmma_vk128_lds_half_ptr) sh_a;
            hrx_q8_0_wmma_vk128_lds_half_ptr sh_b_lds =
                (hrx_q8_0_wmma_vk128_lds_half_ptr) sh_b;
            const int wave_row = static_cast<int>(wave & 1u);
            const int wave_col = static_cast<int>(wave >> 1);
#if HRX_Q8_0_WMMA_VK128_W64_MEDIUMFRAG12_COMBINED96
            const hrx_q8_0_wmma_vk128_half16_vec a0 =
                hrx_q8_0_wmma_vk128_load_a_frag_w64_b64asm_nowait(sh_a_lds, wave_row * 4 + 0, 0, lane);
            const hrx_q8_0_wmma_vk128_half16_vec a1 =
                hrx_q8_0_wmma_vk128_load_a_frag_w64_b64asm_nowait(sh_a_lds, wave_row * 4 + 1, 0, lane);
            const hrx_q8_0_wmma_vk128_half16_vec a2 =
                hrx_q8_0_wmma_vk128_load_a_frag_w64_b64asm_nowait(sh_a_lds, wave_row * 4 + 2, 0, lane);
            const hrx_q8_0_wmma_vk128_half16_vec a3 =
                hrx_q8_0_wmma_vk128_load_a_frag_w64_b64asm_nowait(sh_a_lds, wave_row * 4 + 3, 0, lane);
            const hrx_q8_0_wmma_vk128_half16_vec b0 =
                hrx_q8_0_wmma_vk128_load_b_frag_w64_b64asm_nowait(sh_b_lds, wave_col * 4 + 0, 0, lane);
            const hrx_q8_0_wmma_vk128_half16_vec b1 =
                hrx_q8_0_wmma_vk128_load_b_frag_w64_b64asm_nowait(sh_b_lds, wave_col * 4 + 1, 0, lane);
            const hrx_q8_0_wmma_vk128_half16_vec a4 =
                hrx_q8_0_wmma_vk128_load_a_frag_w64_b64asm_nowait(sh_a_lds, wave_row * 4 + 0, 1, lane);
            const hrx_q8_0_wmma_vk128_half16_vec a5 =
                hrx_q8_0_wmma_vk128_load_a_frag_w64_b64asm_nowait(sh_a_lds, wave_row * 4 + 1, 1, lane);
            const hrx_q8_0_wmma_vk128_half16_vec a6 =
                hrx_q8_0_wmma_vk128_load_a_frag_w64_b64asm_nowait(sh_a_lds, wave_row * 4 + 2, 1, lane);
            const hrx_q8_0_wmma_vk128_half16_vec a7 =
                hrx_q8_0_wmma_vk128_load_a_frag_w64_b64asm_nowait(sh_a_lds, wave_row * 4 + 3, 1, lane);
            const hrx_q8_0_wmma_vk128_half16_vec b2 =
                hrx_q8_0_wmma_vk128_load_b_frag_w64_b64asm_nowait(sh_b_lds, wave_col * 4 + 0, 1, lane);
            const hrx_q8_0_wmma_vk128_half16_vec b3 =
                hrx_q8_0_wmma_vk128_load_b_frag_w64_b64asm_nowait(sh_b_lds, wave_col * 4 + 1, 1, lane);

            HRX_Q8_0_WMMA_VK128_DEP_WMMA_INITIAL(0, a0, b0, 40, a1, a2, a3, b0, b1, a4, b2);
            HRX_Q8_0_WMMA_VK128_DEP_WMMA_AFTER(1, a1, b0, 39, a0, 0);
            HRX_Q8_0_WMMA_VK128_DEP_WMMA_AFTER(2, a2, b0, 35, a1, 1);
            HRX_Q8_0_WMMA_VK128_DEP_WMMA_AFTER(3, a3, b0, 31, a2, 2);
            HRX_Q8_0_WMMA_VK128_DEP_WMMA_AFTER(4, a0, b1, 27, a3, 3);
            HRX_Q8_0_WMMA_VK128_DEP_WMMA_AFTER(5, a1, b1, 23, b1, 4);
            HRX_Q8_0_WMMA_VK128_DEP_WMMA_AFTER(6, a2, b1, 19, a1, 5);
            HRX_Q8_0_WMMA_VK128_DEP_WMMA_AFTER(7, a3, b1, 15, a2, 6);
            HRX_Q8_0_WMMA_VK128_DEP_WMMA_AFTER(0, a4, b2, 11, a3, 7);
            HRX_Q8_0_WMMA_VK128_DEP_WMMA_AFTER(1, a5, b2, 7, b2, 0);
            HRX_Q8_0_WMMA_VK128_DEP_WMMA_AFTER(2, a6, b2, 3, a5, 1);
            HRX_Q8_0_WMMA_VK128_DEP_WMMA_AFTER(3, a7, b2, 0, a6, 2);
            HRX_Q8_0_WMMA_VK128_DEP_WMMA_AFTER(4, a4, b3, 0, a7, 3);
            HRX_Q8_0_WMMA_VK128_DEP_WMMA_AFTER(5, a5, b3, 0, b3, 4);
            HRX_Q8_0_WMMA_VK128_DEP_WMMA_AFTER(6, a6, b3, 0, a5, 5);
            HRX_Q8_0_WMMA_VK128_DEP_WMMA_AFTER(7, a7, b3, 0, a6, 6);
#elif HRX_Q8_0_WMMA_VK128_W64_MOTIF192
#pragma unroll
            for (int k_tile = 0; k_tile < 2; ++k_tile) {
                hrx_q8_0_wmma_vk128_half16_vec a_frag[4];
                hrx_q8_0_wmma_vk128_half16_vec b_frag[4];
#pragma unroll
                for (int row_sub = 0; row_sub < 4; ++row_sub) {
                    a_frag[row_sub] =
                        hrx_q8_0_wmma_vk128_load_a_frag_w64_b64asm_nowait(
                            sh_a_lds, wave_row * 4 + row_sub, k_tile, lane);
                }
#pragma unroll
                for (int col_sub = 0; col_sub < 4; ++col_sub) {
                    b_frag[col_sub] =
                        hrx_q8_0_wmma_vk128_load_b_frag_w64_b64asm_nowait(
                            sh_b_lds, wave_col * 4 + col_sub, k_tile, lane);
                }
                asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
#if HRX_Q8_0_WMMA_VK128_COPY_A_FRAG
#pragma unroll
                for (int row_sub = 0; row_sub < 4; ++row_sub) {
                    a_frag[row_sub] = hrx_q8_0_wmma_vk128_copy_frag(a_frag[row_sub]);
                }
#endif
#if HRX_Q8_0_WMMA_VK128_COPY_B_FRAG
#pragma unroll
                for (int col_sub = 0; col_sub < 4; ++col_sub) {
                    if (col_sub >= HRX_Q8_0_WMMA_VK128_COPY_B_FRAG_MIN_COL_SUB) {
                        b_frag[col_sub] = hrx_q8_0_wmma_vk128_copy_frag(b_frag[col_sub]);
                    }
                }
#endif
#pragma unroll
                for (int col_sub = 0; col_sub < 4; ++col_sub) {
#pragma unroll
                    for (int row_sub = 0; row_sub < 4; ++row_sub) {
                        const int tile_iter = col_sub * 4 + row_sub;
                        acc[tile_iter] = hrx_q8_0_wmma_vk128_wmma_f16_w64(a_frag[row_sub], b_frag[col_sub], acc[tile_iter]);
                    }
                }
            }
#elif HRX_Q8_0_WMMA_VK128_W64_PHASE96
#pragma unroll
            for (int k_tile = 0; k_tile < 2; ++k_tile) {
                hrx_q8_0_wmma_vk128_half16_vec a_frag[4];
                hrx_q8_0_wmma_vk128_half16_vec b_frag[4];
#pragma unroll
                for (int row_sub = 0; row_sub < 4; ++row_sub) {
                    a_frag[row_sub] =
                        hrx_q8_0_wmma_vk128_load_a_frag_w64_b64asm_nowait(
                            sh_a_lds, wave_row * 4 + row_sub, k_tile, lane);
                }
#pragma unroll
                for (int col_sub = 0; col_sub < 4; ++col_sub) {
                    b_frag[col_sub] =
                        hrx_q8_0_wmma_vk128_load_b_frag_w64_b64asm_nowait(
                            sh_b_lds, wave_col * 4 + col_sub, k_tile, lane);
                }
                asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
#if HRX_Q8_0_WMMA_VK128_COPY_A_FRAG
#pragma unroll
                for (int row_sub = 0; row_sub < 4; ++row_sub) {
                    a_frag[row_sub] = hrx_q8_0_wmma_vk128_copy_frag(a_frag[row_sub]);
                }
#endif
#if HRX_Q8_0_WMMA_VK128_COPY_B_FRAG
#pragma unroll
                for (int col_sub = 0; col_sub < 4; ++col_sub) {
                    if (col_sub >= HRX_Q8_0_WMMA_VK128_COPY_B_FRAG_MIN_COL_SUB) {
                        b_frag[col_sub] = hrx_q8_0_wmma_vk128_copy_frag(b_frag[col_sub]);
                    }
                }
#endif
#pragma unroll
                for (int col_delta = 0; col_delta < HRX_Q8_0_WMMA_VK128_W64_PHASE96_COL_COUNT; ++col_delta) {
                    const int col_sub = HRX_Q8_0_WMMA_VK128_W64_PHASE96_COL_START + col_delta;
#pragma unroll
                    for (int row_sub = 0; row_sub < 4; ++row_sub) {
                        const int local = col_delta * 4 + row_sub;
                        acc[local] = hrx_q8_0_wmma_vk128_wmma_f16_w64(a_frag[row_sub], b_frag[col_sub], acc[local]);
                    }
                }
            }
#elif HRX_Q8_0_WMMA_VK128_W64_B64GROUP_DEPENDENT_WAIT_K2
#if HRX_Q8_0_WMMA_VK128_W64_B64GROUP_DEPENDENT_WAIT_K2_RAW
            const hrx_q8_0_wmma_vk128_u64x4_vec a00_raw =
                hrx_q8_0_wmma_vk128_load_a_raw_frag_w64_b64asm_nowait(sh_a_lds, wave_row * 4 + 0, 0, lane);
            const hrx_q8_0_wmma_vk128_u64x4_vec a01_raw =
                hrx_q8_0_wmma_vk128_load_a_raw_frag_w64_b64asm_nowait(sh_a_lds, wave_row * 4 + 1, 0, lane);
            const hrx_q8_0_wmma_vk128_u64x4_vec a02_raw =
                hrx_q8_0_wmma_vk128_load_a_raw_frag_w64_b64asm_nowait(sh_a_lds, wave_row * 4 + 2, 0, lane);
            const hrx_q8_0_wmma_vk128_u64x4_vec a03_raw =
                hrx_q8_0_wmma_vk128_load_a_raw_frag_w64_b64asm_nowait(sh_a_lds, wave_row * 4 + 3, 0, lane);
            const hrx_q8_0_wmma_vk128_u64x4_vec b00_raw =
                hrx_q8_0_wmma_vk128_load_b_raw_frag_w64_b64asm_nowait(sh_b_lds, wave_col * 4 + 0, 0, lane);
            const hrx_q8_0_wmma_vk128_u64x4_vec b01_raw =
                hrx_q8_0_wmma_vk128_load_b_raw_frag_w64_b64asm_nowait(sh_b_lds, wave_col * 4 + 1, 0, lane);
            const hrx_q8_0_wmma_vk128_u64x4_vec b02_raw =
                hrx_q8_0_wmma_vk128_load_b_raw_frag_w64_b64asm_nowait(sh_b_lds, wave_col * 4 + 2, 0, lane);
            const hrx_q8_0_wmma_vk128_u64x4_vec b03_raw =
                hrx_q8_0_wmma_vk128_load_b_raw_frag_w64_b64asm_nowait(sh_b_lds, wave_col * 4 + 3, 0, lane);
            const hrx_q8_0_wmma_vk128_u64x4_vec a10_raw =
                hrx_q8_0_wmma_vk128_load_a_raw_frag_w64_b64asm_nowait(sh_a_lds, wave_row * 4 + 0, 1, lane);
            const hrx_q8_0_wmma_vk128_u64x4_vec a11_raw =
                hrx_q8_0_wmma_vk128_load_a_raw_frag_w64_b64asm_nowait(sh_a_lds, wave_row * 4 + 1, 1, lane);
            const hrx_q8_0_wmma_vk128_u64x4_vec a12_raw =
                hrx_q8_0_wmma_vk128_load_a_raw_frag_w64_b64asm_nowait(sh_a_lds, wave_row * 4 + 2, 1, lane);
            const hrx_q8_0_wmma_vk128_u64x4_vec a13_raw =
                hrx_q8_0_wmma_vk128_load_a_raw_frag_w64_b64asm_nowait(sh_a_lds, wave_row * 4 + 3, 1, lane);
            const hrx_q8_0_wmma_vk128_u64x4_vec b10_raw =
                hrx_q8_0_wmma_vk128_load_b_raw_frag_w64_b64asm_nowait(sh_b_lds, wave_col * 4 + 0, 1, lane);
            const hrx_q8_0_wmma_vk128_u64x4_vec b11_raw =
                hrx_q8_0_wmma_vk128_load_b_raw_frag_w64_b64asm_nowait(sh_b_lds, wave_col * 4 + 1, 1, lane);
            const hrx_q8_0_wmma_vk128_u64x4_vec b12_raw =
                hrx_q8_0_wmma_vk128_load_b_raw_frag_w64_b64asm_nowait(sh_b_lds, wave_col * 4 + 2, 1, lane);
            const hrx_q8_0_wmma_vk128_u64x4_vec b13_raw =
                hrx_q8_0_wmma_vk128_load_b_raw_frag_w64_b64asm_nowait(sh_b_lds, wave_col * 4 + 3, 1, lane);

#if HRX_Q8_0_WMMA_VK128_W64_B64GROUP_DEPENDENT_WAIT_K2_PREUSE
            hrx_q8_0_wmma_vk128_preuse_raw_fragments_k2(
                a00_raw, a01_raw, a02_raw, a03_raw, b00_raw, b01_raw, b02_raw, b03_raw,
                a10_raw, a11_raw, a12_raw, a13_raw, b10_raw, b11_raw, b12_raw, b13_raw);
#endif

            const hrx_q8_0_wmma_vk128_half16_vec a00 =
                hrx_q8_0_wmma_vk128_raw_frag_to_half16(a00_raw);
            const hrx_q8_0_wmma_vk128_half16_vec a01 =
                hrx_q8_0_wmma_vk128_raw_frag_to_half16(a01_raw);
            const hrx_q8_0_wmma_vk128_half16_vec a02 =
                hrx_q8_0_wmma_vk128_raw_frag_to_half16(a02_raw);
            const hrx_q8_0_wmma_vk128_half16_vec a03 =
                hrx_q8_0_wmma_vk128_raw_frag_to_half16(a03_raw);
            const hrx_q8_0_wmma_vk128_half16_vec b00 =
                hrx_q8_0_wmma_vk128_raw_frag_to_half16(b00_raw);
            const hrx_q8_0_wmma_vk128_half16_vec b01 =
                hrx_q8_0_wmma_vk128_raw_frag_to_half16(b01_raw);
            const hrx_q8_0_wmma_vk128_half16_vec b02 =
                hrx_q8_0_wmma_vk128_raw_frag_to_half16(b02_raw);
            const hrx_q8_0_wmma_vk128_half16_vec b03 =
                hrx_q8_0_wmma_vk128_raw_frag_to_half16(b03_raw);
            const hrx_q8_0_wmma_vk128_half16_vec a10 =
                hrx_q8_0_wmma_vk128_raw_frag_to_half16(a10_raw);
            const hrx_q8_0_wmma_vk128_half16_vec a11 =
                hrx_q8_0_wmma_vk128_raw_frag_to_half16(a11_raw);
            const hrx_q8_0_wmma_vk128_half16_vec a12 =
                hrx_q8_0_wmma_vk128_raw_frag_to_half16(a12_raw);
            const hrx_q8_0_wmma_vk128_half16_vec a13 =
                hrx_q8_0_wmma_vk128_raw_frag_to_half16(a13_raw);
            const hrx_q8_0_wmma_vk128_half16_vec b10 =
                hrx_q8_0_wmma_vk128_raw_frag_to_half16(b10_raw);
            const hrx_q8_0_wmma_vk128_half16_vec b11 =
                hrx_q8_0_wmma_vk128_raw_frag_to_half16(b11_raw);
            const hrx_q8_0_wmma_vk128_half16_vec b12 =
                hrx_q8_0_wmma_vk128_raw_frag_to_half16(b12_raw);
            const hrx_q8_0_wmma_vk128_half16_vec b13 =
                hrx_q8_0_wmma_vk128_raw_frag_to_half16(b13_raw);
#else
            const hrx_q8_0_wmma_vk128_half16_vec a00 =
                hrx_q8_0_wmma_vk128_load_a_frag_w64_b64asm_nowait(sh_a_lds, wave_row * 4 + 0, 0, lane);
            const hrx_q8_0_wmma_vk128_half16_vec a01 =
                hrx_q8_0_wmma_vk128_load_a_frag_w64_b64asm_nowait(sh_a_lds, wave_row * 4 + 1, 0, lane);
            const hrx_q8_0_wmma_vk128_half16_vec a02 =
                hrx_q8_0_wmma_vk128_load_a_frag_w64_b64asm_nowait(sh_a_lds, wave_row * 4 + 2, 0, lane);
            const hrx_q8_0_wmma_vk128_half16_vec a03 =
                hrx_q8_0_wmma_vk128_load_a_frag_w64_b64asm_nowait(sh_a_lds, wave_row * 4 + 3, 0, lane);
            const hrx_q8_0_wmma_vk128_half16_vec b00 =
                hrx_q8_0_wmma_vk128_load_b_frag_w64_b64asm_nowait(sh_b_lds, wave_col * 4 + 0, 0, lane);
            const hrx_q8_0_wmma_vk128_half16_vec b01 =
                hrx_q8_0_wmma_vk128_load_b_frag_w64_b64asm_nowait(sh_b_lds, wave_col * 4 + 1, 0, lane);
            const hrx_q8_0_wmma_vk128_half16_vec b02 =
                hrx_q8_0_wmma_vk128_load_b_frag_w64_b64asm_nowait(sh_b_lds, wave_col * 4 + 2, 0, lane);
            const hrx_q8_0_wmma_vk128_half16_vec b03 =
                hrx_q8_0_wmma_vk128_load_b_frag_w64_b64asm_nowait(sh_b_lds, wave_col * 4 + 3, 0, lane);
            const hrx_q8_0_wmma_vk128_half16_vec a10 =
                hrx_q8_0_wmma_vk128_load_a_frag_w64_b64asm_nowait(sh_a_lds, wave_row * 4 + 0, 1, lane);
            const hrx_q8_0_wmma_vk128_half16_vec a11 =
                hrx_q8_0_wmma_vk128_load_a_frag_w64_b64asm_nowait(sh_a_lds, wave_row * 4 + 1, 1, lane);
            const hrx_q8_0_wmma_vk128_half16_vec a12 =
                hrx_q8_0_wmma_vk128_load_a_frag_w64_b64asm_nowait(sh_a_lds, wave_row * 4 + 2, 1, lane);
            const hrx_q8_0_wmma_vk128_half16_vec a13 =
                hrx_q8_0_wmma_vk128_load_a_frag_w64_b64asm_nowait(sh_a_lds, wave_row * 4 + 3, 1, lane);
            const hrx_q8_0_wmma_vk128_half16_vec b10 =
                hrx_q8_0_wmma_vk128_load_b_frag_w64_b64asm_nowait(sh_b_lds, wave_col * 4 + 0, 1, lane);
            const hrx_q8_0_wmma_vk128_half16_vec b11 =
                hrx_q8_0_wmma_vk128_load_b_frag_w64_b64asm_nowait(sh_b_lds, wave_col * 4 + 1, 1, lane);
            const hrx_q8_0_wmma_vk128_half16_vec b12 =
                hrx_q8_0_wmma_vk128_load_b_frag_w64_b64asm_nowait(sh_b_lds, wave_col * 4 + 2, 1, lane);
            const hrx_q8_0_wmma_vk128_half16_vec b13 =
                hrx_q8_0_wmma_vk128_load_b_frag_w64_b64asm_nowait(sh_b_lds, wave_col * 4 + 3, 1, lane);

#if HRX_Q8_0_WMMA_VK128_W64_B64GROUP_DEPENDENT_WAIT_K2_PREUSE
            hrx_q8_0_wmma_vk128_preuse_fragments_k2(
                a00, a01, a02, a03, b00, b01, b02, b03,
                a10, a11, a12, a13, b10, b11, b12, b13);
#endif
#endif

#if HRX_Q8_0_WMMA_VK128_W64_B64GROUP_DEPENDENT_WAIT_K2_DIRECT
            HRX_Q8_0_WMMA_VK128_WAIT_WMMA(0, a00, b00, 51);
            HRX_Q8_0_WMMA_VK128_WAIT_WMMA(1, a01, b00, 47);
            HRX_Q8_0_WMMA_VK128_WAIT_WMMA(2, a02, b00, 43);
            HRX_Q8_0_WMMA_VK128_WAIT_WMMA(3, a03, b00, 39);
            HRX_Q8_0_WMMA_VK128_WAIT_WMMA(4, a00, b01, 40);
            HRX_Q8_0_WMMA_VK128_WAIT_WMMA(5, a01, b01, 36);
            HRX_Q8_0_WMMA_VK128_WAIT_WMMA(6, a02, b01, 32);
            HRX_Q8_0_WMMA_VK128_WAIT_WMMA(7, a03, b01, 24);
            HRX_Q8_0_WMMA_VK128_WAIT_WMMA(8, a00, b02, 20);
            HRX_Q8_0_WMMA_VK128_WAIT_WMMA(9, a01, b02, 16);
            HRX_Q8_0_WMMA_VK128_WAIT_WMMA(10, a02, b02, 12);
            HRX_Q8_0_WMMA_VK128_WAIT_WMMA(11, a03, b02, 8);
            HRX_Q8_0_WMMA_VK128_WAIT_WMMA(12, a00, b03, 4);
            HRX_Q8_0_WMMA_VK128_WAIT_WMMA(13, a01, b03, 0);
            HRX_Q8_0_WMMA_VK128_WAIT_WMMA(14, a02, b03, 0);
            HRX_Q8_0_WMMA_VK128_WAIT_WMMA(15, a03, b03, 0);

            HRX_Q8_0_WMMA_VK128_WAIT_WMMA(0, a10, b10, 51);
            HRX_Q8_0_WMMA_VK128_WAIT_WMMA(1, a11, b10, 47);
            HRX_Q8_0_WMMA_VK128_WAIT_WMMA(2, a12, b10, 43);
            HRX_Q8_0_WMMA_VK128_WAIT_WMMA(3, a13, b10, 39);
            HRX_Q8_0_WMMA_VK128_WAIT_WMMA(4, a10, b11, 40);
            HRX_Q8_0_WMMA_VK128_WAIT_WMMA(5, a11, b11, 36);
            HRX_Q8_0_WMMA_VK128_WAIT_WMMA(6, a12, b11, 32);
            HRX_Q8_0_WMMA_VK128_WAIT_WMMA(7, a13, b11, 24);
            HRX_Q8_0_WMMA_VK128_WAIT_WMMA(8, a10, b12, 20);
            HRX_Q8_0_WMMA_VK128_WAIT_WMMA(9, a11, b12, 16);
            HRX_Q8_0_WMMA_VK128_WAIT_WMMA(10, a12, b12, 12);
            HRX_Q8_0_WMMA_VK128_WAIT_WMMA(11, a13, b12, 8);
            HRX_Q8_0_WMMA_VK128_WAIT_WMMA(12, a10, b13, 4);
            HRX_Q8_0_WMMA_VK128_WAIT_WMMA(13, a11, b13, 0);
            HRX_Q8_0_WMMA_VK128_WAIT_WMMA(14, a12, b13, 0);
            HRX_Q8_0_WMMA_VK128_WAIT_WMMA(15, a13, b13, 0);
#else
            HRX_Q8_0_WMMA_VK128_DEP_WMMA_INITIAL(0, a00, b00, 51, a01, a02, a03, b00, b01, b02, b03);
            HRX_Q8_0_WMMA_VK128_DEP_WMMA_AFTER(1, a01, b00, 47, a00, 0);
            HRX_Q8_0_WMMA_VK128_DEP_WMMA_AFTER(2, a02, b00, 43, a01, 1);
            HRX_Q8_0_WMMA_VK128_DEP_WMMA_AFTER(3, a03, b00, 39, a02, 2);
            HRX_Q8_0_WMMA_VK128_DEP_WMMA_AFTER(4, a00, b01, 40, a03, 3);
            HRX_Q8_0_WMMA_VK128_DEP_WMMA_AFTER(5, a01, b01, 36, b01, 4);
            HRX_Q8_0_WMMA_VK128_DEP_WMMA_AFTER(6, a02, b01, 32, a01, 5);
            HRX_Q8_0_WMMA_VK128_DEP_WMMA_AFTER(7, a03, b01, 24, a02, 6);
            HRX_Q8_0_WMMA_VK128_DEP_WMMA_AFTER(8, a00, b02, 20, a03, 7);
            HRX_Q8_0_WMMA_VK128_DEP_WMMA_AFTER(9, a01, b02, 16, b02, 8);
            HRX_Q8_0_WMMA_VK128_DEP_WMMA_AFTER(10, a02, b02, 12, a01, 9);
            HRX_Q8_0_WMMA_VK128_DEP_WMMA_AFTER(11, a03, b02, 8, a02, 10);
            HRX_Q8_0_WMMA_VK128_DEP_WMMA_AFTER(12, a00, b03, 4, a03, 11);
            HRX_Q8_0_WMMA_VK128_DEP_WMMA_AFTER(13, a01, b03, 0, b03, 12);
            HRX_Q8_0_WMMA_VK128_DEP_WMMA_AFTER(14, a02, b03, 0, a01, 13);
            HRX_Q8_0_WMMA_VK128_DEP_WMMA_AFTER(15, a03, b03, 0, a02, 14);

            HRX_Q8_0_WMMA_VK128_DEP_WMMA_AFTER(0, a10, b10, 51, b10, 15);
            HRX_Q8_0_WMMA_VK128_DEP_WMMA_AFTER(1, a11, b10, 47, a10, 0);
            HRX_Q8_0_WMMA_VK128_DEP_WMMA_AFTER(2, a12, b10, 43, a11, 1);
            HRX_Q8_0_WMMA_VK128_DEP_WMMA_AFTER(3, a13, b10, 39, a12, 2);
            HRX_Q8_0_WMMA_VK128_DEP_WMMA_AFTER(4, a10, b11, 40, a13, 3);
            HRX_Q8_0_WMMA_VK128_DEP_WMMA_AFTER(5, a11, b11, 36, b11, 4);
            HRX_Q8_0_WMMA_VK128_DEP_WMMA_AFTER(6, a12, b11, 32, a11, 5);
            HRX_Q8_0_WMMA_VK128_DEP_WMMA_AFTER(7, a13, b11, 24, a12, 6);
            HRX_Q8_0_WMMA_VK128_DEP_WMMA_AFTER(8, a10, b12, 20, a13, 7);
            HRX_Q8_0_WMMA_VK128_DEP_WMMA_AFTER(9, a11, b12, 16, b12, 8);
            HRX_Q8_0_WMMA_VK128_DEP_WMMA_AFTER(10, a12, b12, 12, a11, 9);
            HRX_Q8_0_WMMA_VK128_DEP_WMMA_AFTER(11, a13, b12, 8, a12, 10);
            HRX_Q8_0_WMMA_VK128_DEP_WMMA_AFTER(12, a10, b13, 4, a13, 11);
            HRX_Q8_0_WMMA_VK128_DEP_WMMA_AFTER(13, a11, b13, 0, b13, 12);
            HRX_Q8_0_WMMA_VK128_DEP_WMMA_AFTER(14, a12, b13, 0, a11, 13);
            HRX_Q8_0_WMMA_VK128_DEP_WMMA_AFTER(15, a13, b13, 0, a12, 14);
#endif
#else
#pragma unroll
            for (int k_tile = 0; k_tile < 2; ++k_tile) {
                hrx_q8_0_wmma_vk128_half16_vec a_frag[4];
#if HRX_Q8_0_WMMA_VK128_W64_B64GROUP_STREAM_COL
#pragma unroll
                for (int row_sub = 0; row_sub < 4; ++row_sub) {
                    a_frag[row_sub] = hrx_q8_0_wmma_vk128_load_a_frag_w64_b64asm(
                        sh_a_lds, wave_row * 4 + row_sub, k_tile, lane);
                }
                asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
#pragma unroll
                for (int col_sub = 0; col_sub < 4; ++col_sub) {
                    const hrx_q8_0_wmma_vk128_half16_vec b_frag =
                        hrx_q8_0_wmma_vk128_load_b_frag_w64_b64asm(
                            sh_b_lds, wave_col * 4 + col_sub, k_tile, lane);
                    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
#pragma unroll
                    for (int row_sub = 0; row_sub < 4; ++row_sub) {
                        const int tile_iter = col_sub * 4 + row_sub;
                        acc[tile_iter] = hrx_q8_0_wmma_vk128_wmma_f16_w64(a_frag[row_sub], b_frag, acc[tile_iter]);
                    }
                }
#elif HRX_Q8_0_WMMA_VK128_W64_B64GROUP_STREAM_ROW
                hrx_q8_0_wmma_vk128_half16_vec b_frag[4];
#pragma unroll
                for (int col_sub = 0; col_sub < 4; ++col_sub) {
                    b_frag[col_sub] = hrx_q8_0_wmma_vk128_load_b_frag_w64_b64asm(
                        sh_b_lds, wave_col * 4 + col_sub, k_tile, lane);
                }
                asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
#if HRX_Q8_0_WMMA_VK128_COPY_B_FRAG
#pragma unroll
                for (int col_sub = 0; col_sub < 4; ++col_sub) {
                    if (col_sub >= HRX_Q8_0_WMMA_VK128_COPY_B_FRAG_MIN_COL_SUB) {
                        b_frag[col_sub] = hrx_q8_0_wmma_vk128_copy_frag(b_frag[col_sub]);
                    }
                }
#endif
#pragma unroll
                for (int row_sub = 0; row_sub < 4; ++row_sub) {
                    hrx_q8_0_wmma_vk128_half16_vec a_frag_row =
                        hrx_q8_0_wmma_vk128_load_a_frag_w64_b64asm(
                            sh_a_lds, wave_row * 4 + row_sub, k_tile, lane);
                    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
#if HRX_Q8_0_WMMA_VK128_COPY_A_FRAG
                    a_frag_row = hrx_q8_0_wmma_vk128_copy_frag(a_frag_row);
#endif
#pragma unroll
                    for (int col_sub = 0; col_sub < 4; ++col_sub) {
                        const int tile_iter = col_sub * 4 + row_sub;
                        acc[tile_iter] = hrx_q8_0_wmma_vk128_wmma_f16_w64(a_frag_row, b_frag[col_sub], acc[tile_iter]);
                    }
                }
#elif HRX_Q8_0_WMMA_VK128_W64_B64GROUP_NAMED_FRAGS
#if HRX_Q8_0_WMMA_VK128_W64_B64GROUP_NOWAIT
                const hrx_q8_0_wmma_vk128_half16_vec a0 =
                    hrx_q8_0_wmma_vk128_load_a_frag_w64_b64asm_nowait(sh_a_lds, wave_row * 4 + 0, k_tile, lane);
                const hrx_q8_0_wmma_vk128_half16_vec a1 =
                    hrx_q8_0_wmma_vk128_load_a_frag_w64_b64asm_nowait(sh_a_lds, wave_row * 4 + 1, k_tile, lane);
                const hrx_q8_0_wmma_vk128_half16_vec a2 =
                    hrx_q8_0_wmma_vk128_load_a_frag_w64_b64asm_nowait(sh_a_lds, wave_row * 4 + 2, k_tile, lane);
                const hrx_q8_0_wmma_vk128_half16_vec a3 =
                    hrx_q8_0_wmma_vk128_load_a_frag_w64_b64asm_nowait(sh_a_lds, wave_row * 4 + 3, k_tile, lane);
                const hrx_q8_0_wmma_vk128_half16_vec b0 =
                    hrx_q8_0_wmma_vk128_load_b_frag_w64_b64asm_nowait(sh_b_lds, wave_col * 4 + 0, k_tile, lane);
                const hrx_q8_0_wmma_vk128_half16_vec b1 =
                    hrx_q8_0_wmma_vk128_load_b_frag_w64_b64asm_nowait(sh_b_lds, wave_col * 4 + 1, k_tile, lane);
                const hrx_q8_0_wmma_vk128_half16_vec b2 =
                    hrx_q8_0_wmma_vk128_load_b_frag_w64_b64asm_nowait(sh_b_lds, wave_col * 4 + 2, k_tile, lane);
                const hrx_q8_0_wmma_vk128_half16_vec b3 =
                    hrx_q8_0_wmma_vk128_load_b_frag_w64_b64asm_nowait(sh_b_lds, wave_col * 4 + 3, k_tile, lane);
#else
                const hrx_q8_0_wmma_vk128_half16_vec a0 =
                    hrx_q8_0_wmma_vk128_load_a_frag_w64_b64asm(sh_a_lds, wave_row * 4 + 0, k_tile, lane);
                const hrx_q8_0_wmma_vk128_half16_vec a1 =
                    hrx_q8_0_wmma_vk128_load_a_frag_w64_b64asm(sh_a_lds, wave_row * 4 + 1, k_tile, lane);
                const hrx_q8_0_wmma_vk128_half16_vec a2 =
                    hrx_q8_0_wmma_vk128_load_a_frag_w64_b64asm(sh_a_lds, wave_row * 4 + 2, k_tile, lane);
                const hrx_q8_0_wmma_vk128_half16_vec a3 =
                    hrx_q8_0_wmma_vk128_load_a_frag_w64_b64asm(sh_a_lds, wave_row * 4 + 3, k_tile, lane);
                const hrx_q8_0_wmma_vk128_half16_vec b0 =
                    hrx_q8_0_wmma_vk128_load_b_frag_w64_b64asm(sh_b_lds, wave_col * 4 + 0, k_tile, lane);
                const hrx_q8_0_wmma_vk128_half16_vec b1 =
                    hrx_q8_0_wmma_vk128_load_b_frag_w64_b64asm(sh_b_lds, wave_col * 4 + 1, k_tile, lane);
                const hrx_q8_0_wmma_vk128_half16_vec b2 =
                    hrx_q8_0_wmma_vk128_load_b_frag_w64_b64asm(sh_b_lds, wave_col * 4 + 2, k_tile, lane);
                const hrx_q8_0_wmma_vk128_half16_vec b3 =
                    hrx_q8_0_wmma_vk128_load_b_frag_w64_b64asm(sh_b_lds, wave_col * 4 + 3, k_tile, lane);
#endif
#if HRX_Q8_0_WMMA_VK128_W64_B64GROUP_PREUSE_FRAGS
                hrx_q8_0_wmma_vk128_preuse_fragments(a0, a1, a2, a3, b0, b1, b2, b3);
#endif
#if HRX_Q8_0_WMMA_VK128_W64_B64GROUP_DEPENDENT_WAIT
                HRX_Q8_0_WMMA_VK128_DEP_WMMA_INITIAL(0, a0, b0, 51, a1, a2, a3, b0, b1, b2, b3);
                HRX_Q8_0_WMMA_VK128_DEP_WMMA_AFTER(1, a1, b0, 47, a0, 0);
                HRX_Q8_0_WMMA_VK128_DEP_WMMA_AFTER(2, a2, b0, 43, a1, 1);
                HRX_Q8_0_WMMA_VK128_DEP_WMMA_AFTER(3, a3, b0, 39, a2, 2);
                HRX_Q8_0_WMMA_VK128_DEP_WMMA_AFTER(4, a0, b1, 40, a3, 3);
                HRX_Q8_0_WMMA_VK128_DEP_WMMA_AFTER(5, a1, b1, 36, b1, 4);
                HRX_Q8_0_WMMA_VK128_DEP_WMMA_AFTER(6, a2, b1, 32, a1, 5);
                HRX_Q8_0_WMMA_VK128_DEP_WMMA_AFTER(7, a3, b1, 24, a2, 6);
                HRX_Q8_0_WMMA_VK128_DEP_WMMA_AFTER(8, a0, b2, 20, a3, 7);
                HRX_Q8_0_WMMA_VK128_DEP_WMMA_AFTER(9, a1, b2, 16, b2, 8);
                HRX_Q8_0_WMMA_VK128_DEP_WMMA_AFTER(10, a2, b2, 12, a1, 9);
                HRX_Q8_0_WMMA_VK128_DEP_WMMA_AFTER(11, a3, b2, 8, a2, 10);
                HRX_Q8_0_WMMA_VK128_DEP_WMMA_AFTER(12, a0, b3, 4, a3, 11);
                HRX_Q8_0_WMMA_VK128_DEP_WMMA_AFTER(13, a1, b3, 0, b3, 12);
                HRX_Q8_0_WMMA_VK128_DEP_WMMA_AFTER(14, a2, b3, 0, a1, 13);
                HRX_Q8_0_WMMA_VK128_DEP_WMMA_AFTER(15, a3, b3, 0, a2, 14);
#else
#if HRX_Q8_0_WMMA_VK128_W64_B64GROUP_STAGED_WAIT
                asm volatile("s_waitcnt lgkmcnt(%0)\n" :: "n"(HRX_Q8_0_WMMA_VK128_W64_B64GROUP_STAGED_WAIT0) : "memory");
#else
                asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
#endif
                acc[0] = hrx_q8_0_wmma_vk128_wmma_f16_w64(a0, b0, acc[0]);
                acc[1] = hrx_q8_0_wmma_vk128_wmma_f16_w64(a1, b0, acc[1]);
                acc[2] = hrx_q8_0_wmma_vk128_wmma_f16_w64(a2, b0, acc[2]);
                acc[3] = hrx_q8_0_wmma_vk128_wmma_f16_w64(a3, b0, acc[3]);
#if HRX_Q8_0_WMMA_VK128_W64_B64GROUP_STAGED_WAIT
                asm volatile("s_waitcnt lgkmcnt(%0)\n" :: "n"(HRX_Q8_0_WMMA_VK128_W64_B64GROUP_STAGED_WAIT1) : "memory");
#endif
                acc[4] = hrx_q8_0_wmma_vk128_wmma_f16_w64(a0, b1, acc[4]);
                acc[5] = hrx_q8_0_wmma_vk128_wmma_f16_w64(a1, b1, acc[5]);
                acc[6] = hrx_q8_0_wmma_vk128_wmma_f16_w64(a2, b1, acc[6]);
                acc[7] = hrx_q8_0_wmma_vk128_wmma_f16_w64(a3, b1, acc[7]);
#if HRX_Q8_0_WMMA_VK128_W64_B64GROUP_STAGED_WAIT
                asm volatile("s_waitcnt lgkmcnt(%0)\n" :: "n"(HRX_Q8_0_WMMA_VK128_W64_B64GROUP_STAGED_WAIT2) : "memory");
#endif
                acc[8] = hrx_q8_0_wmma_vk128_wmma_f16_w64(a0, b2, acc[8]);
                acc[9] = hrx_q8_0_wmma_vk128_wmma_f16_w64(a1, b2, acc[9]);
                acc[10] = hrx_q8_0_wmma_vk128_wmma_f16_w64(a2, b2, acc[10]);
                acc[11] = hrx_q8_0_wmma_vk128_wmma_f16_w64(a3, b2, acc[11]);
#if HRX_Q8_0_WMMA_VK128_W64_B64GROUP_STAGED_WAIT
                asm volatile("s_waitcnt lgkmcnt(%0)\n" :: "n"(HRX_Q8_0_WMMA_VK128_W64_B64GROUP_STAGED_WAIT3) : "memory");
#endif
                acc[12] = hrx_q8_0_wmma_vk128_wmma_f16_w64(a0, b3, acc[12]);
                acc[13] = hrx_q8_0_wmma_vk128_wmma_f16_w64(a1, b3, acc[13]);
                acc[14] = hrx_q8_0_wmma_vk128_wmma_f16_w64(a2, b3, acc[14]);
                acc[15] = hrx_q8_0_wmma_vk128_wmma_f16_w64(a3, b3, acc[15]);
#endif
#else
                hrx_q8_0_wmma_vk128_half16_vec b_frag[4];
#pragma unroll
                for (int row_sub = 0; row_sub < 4; ++row_sub) {
#if HRX_Q8_0_WMMA_VK128_W64_B64GROUP_NOWAIT
                    a_frag[row_sub] = hrx_q8_0_wmma_vk128_load_a_frag_w64_b64asm_nowait(
                        sh_a_lds, wave_row * 4 + row_sub, k_tile, lane);
#else
                    a_frag[row_sub] = hrx_q8_0_wmma_vk128_load_a_frag_w64_b64asm(
                        sh_a_lds, wave_row * 4 + row_sub, k_tile, lane);
#endif
                }
#pragma unroll
                for (int col_sub = 0; col_sub < 4; ++col_sub) {
#if HRX_Q8_0_WMMA_VK128_W64_B64GROUP_NOWAIT
                    b_frag[col_sub] = hrx_q8_0_wmma_vk128_load_b_frag_w64_b64asm_nowait(
                        sh_b_lds, wave_col * 4 + col_sub, k_tile, lane);
#else
                    b_frag[col_sub] = hrx_q8_0_wmma_vk128_load_b_frag_w64_b64asm(
                        sh_b_lds, wave_col * 4 + col_sub, k_tile, lane);
#endif
                }
                asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
#if HRX_Q8_0_WMMA_VK128_COPY_A_FRAG
#pragma unroll
                for (int row_sub = 0; row_sub < 4; ++row_sub) {
                    a_frag[row_sub] = hrx_q8_0_wmma_vk128_copy_frag(a_frag[row_sub]);
                }
#endif
#if HRX_Q8_0_WMMA_VK128_COPY_B_FRAG
#pragma unroll
                for (int col_sub = 0; col_sub < 4; ++col_sub) {
                    if (col_sub >= HRX_Q8_0_WMMA_VK128_COPY_B_FRAG_MIN_COL_SUB) {
                        b_frag[col_sub] = hrx_q8_0_wmma_vk128_copy_frag(b_frag[col_sub]);
                    }
                }
#endif
#pragma unroll
                for (int col_sub = 0; col_sub < 4; ++col_sub) {
#pragma unroll
                    for (int row_sub = 0; row_sub < 4; ++row_sub) {
                        const int tile_iter = col_sub * 4 + row_sub;
                        acc[tile_iter] = hrx_q8_0_wmma_vk128_wmma_f16_w64(a_frag[row_sub], b_frag[col_sub], acc[tile_iter]);
                    }
                }
#endif
            }
#endif
        }
#else
#pragma unroll
        for (int tile_iter = 0; tile_iter < TILES_PER_WAVE; ++tile_iter) {
            int row_tile = 0;
            int col_tile = 0;
            hrx_q8_0_wmma_vk128_tile_map(static_cast<int>(wave), tile_iter, &row_tile, &col_tile);
#if HRX_Q8_0_WMMA_VK128_W64
#pragma unroll
            for (int k_tile = 0; k_tile < 2; ++k_tile) {
#if HRX_Q8_0_WMMA_VK128_W64_H4LOAD
                const hrx_q8_0_wmma_vk128_half16_vec a =
                    hrx_q8_0_wmma_vk128_load_a_frag_w64_h4(sh_a, row_tile, k_tile, lane);
                const hrx_q8_0_wmma_vk128_half16_vec b =
                    hrx_q8_0_wmma_vk128_load_b_frag_w64_h4(sh_b, col_tile, k_tile, lane);
#else
                const hrx_q8_0_wmma_vk128_half16_vec a =
                    hrx_q8_0_wmma_vk128_load_a_frag_w64(sh_a, row_tile, k_tile, lane);
                const hrx_q8_0_wmma_vk128_half16_vec b =
                    hrx_q8_0_wmma_vk128_load_b_frag_w64(sh_b, col_tile, k_tile, lane);
#endif
                acc[tile_iter] = hrx_q8_0_wmma_vk128_wmma_f16_w64(a, b, acc[tile_iter]);
            }
#else
#if HRX_Q8_0_WMMA_VK128_PREFETCH_FRAGS
            hrx_q8_0_wmma_vk128_half16_vec a_frag[2];
            hrx_q8_0_wmma_vk128_half16_vec b_frag[2];
#pragma unroll
            for (int k_tile = 0; k_tile < 2; ++k_tile) {
                a_frag[k_tile] = hrx_q8_0_wmma_vk128_load_a_frag(sh_a, row_tile, k_tile, lane);
                b_frag[k_tile] = hrx_q8_0_wmma_vk128_load_b_frag(sh_b, col_tile, k_tile, lane);
            }
#pragma unroll
            for (int k_tile = 0; k_tile < 2; ++k_tile) {
                acc[tile_iter] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(
                    a_frag[k_tile], b_frag[k_tile], acc[tile_iter], false);
            }
#else
#pragma unroll
            for (int k_tile = 0; k_tile < 2; ++k_tile) {
                const hrx_q8_0_wmma_vk128_half16_vec a =
                    hrx_q8_0_wmma_vk128_load_a_frag(sh_a, row_tile, k_tile, lane);
                const hrx_q8_0_wmma_vk128_half16_vec b =
                    hrx_q8_0_wmma_vk128_load_b_frag(sh_b, col_tile, k_tile, lane);
                acc[tile_iter] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(a, b, acc[tile_iter], false);
            }
#endif
#endif
        }
#endif

        __syncthreads();
    }

#if HRX_Q8_0_WMMA_VK128_W64_MEDIUMFRAG12_COMBINED96
    HRX_Q8_0_WMMA_VK128_COMBINED96_GROUPS_0_7(HRX_Q8_0_WMMA_VK128_COMBINED96_RAW_STORE_GROUP);
    HRX_Q8_0_WMMA_VK128_COMBINED96_GROUPS_8_23(HRX_Q8_0_WMMA_VK128_COMBINED96_STAGE_STORE_GROUP);
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();
    HRX_Q8_0_WMMA_VK128_COMBINED96_GROUPS_8_23(HRX_Q8_0_WMMA_VK128_COMBINED96_STAGE_LOAD_STORE_GROUP);
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();
#elif HRX_Q8_0_WMMA_VK128_W64_MOTIF192
    HRX_Q8_0_WMMA_VK128_MOTIF192_GROUPS_0_15();
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();
#elif HRX_Q8_0_WMMA_VK128_W64_PHASE96
#if HRX_Q8_0_WMMA_VK128_W64_PHASE96_GROUP_BASE == 0
    HRX_Q8_0_WMMA_VK128_PHASE96_GROUPS_0_7(HRX_Q8_0_WMMA_VK128_PHASE96_RAW_STORE_GROUP);
#elif HRX_Q8_0_WMMA_VK128_W64_PHASE96_GROUP_BASE == 8
    HRX_Q8_0_WMMA_VK128_PHASE96_GROUPS_8_15(HRX_Q8_0_WMMA_VK128_PHASE96_RAW_STORE_GROUP);
#else
#error "unsupported HRX_Q8_0_WMMA_VK128_W64_PHASE96_GROUP_BASE"
#endif
#else
#pragma unroll
    for (int tile_iter = 0; tile_iter < TILES_PER_WAVE; ++tile_iter) {
        int row_tile = 0;
        int col_tile = 0;
        hrx_q8_0_wmma_vk128_tile_map(static_cast<int>(wave), tile_iter, &row_tile, &col_tile);
#if HRX_Q8_0_WMMA_VK128_W64
#if HRX_Q8_0_WMMA_VK128_STORE_STAGE_DUAL_HALF
        hrx_q8_0_wmma_vk128_store_acc_f16_row_major_w64_dual_half_stage(
#if HRX_Q8_0_WMMA_VK128_BUFFER_STORE
            dst_rsrc,
#else
            dst,
#endif
            rows,
            row_base + static_cast<long long>(row_tile * 16),
            col_base + static_cast<long long>(col_tile * 16),
            rows,
            cols,
            acc[tile_iter],
            lane,
            wave,
            (hrx_q8_0_wmma_vk128_lds_volatile_half_ptr) sh_store);
#elif HRX_Q8_0_WMMA_VK128_STORE_STAGE
        hrx_q8_0_wmma_vk128_store_acc_f16_row_major_w64_stage(
            dst,
            rows,
            row_base + static_cast<long long>(row_tile * 16),
            col_base + static_cast<long long>(col_tile * 16),
            rows,
            cols,
            acc[tile_iter],
            lane,
            wave,
            sh_store);
#else
#if HRX_Q8_0_WMMA_VK128_FULL_TILE_STORE
        const long long tile_row0 = row_base + static_cast<long long>(row_tile * 16);
        const long long tile_col0 = col_base + static_cast<long long>(col_tile * 16);
#if HRX_Q8_0_WMMA_VK128_ASSUME_FULL_TILES
#if HRX_Q8_0_WMMA_VK128_BUFFER_STORE
#if HRX_Q8_0_WMMA_VK128_STORE_STAGE_FAST_HALF
#if HRX_Q8_0_WMMA_VK128_STORE_STAGE_FAST_HALF_SPLIT_SELECTED
        hrx_q8_0_wmma_vk128_store_acc_f16_row_major_w64_fast_half_buffer_split_selected(
            dst_rsrc,
            rows,
            tile_row0,
            tile_col0,
            acc[tile_iter],
            lane,
            wave,
            (hrx_q8_0_wmma_vk128_lds_volatile_half_ptr) sh_store);
#elif HRX_Q8_0_WMMA_VK128_STORE_STAGE_FAST_HALF_SELECTED
        hrx_q8_0_wmma_vk128_store_acc_f16_row_major_w64_fast_half_buffer_selected(
            dst_rsrc,
            rows,
            tile_row0,
            tile_col0,
            acc[tile_iter],
            lane,
            wave,
            (hrx_q8_0_wmma_vk128_lds_volatile_half_ptr) sh_store);
#else
        hrx_q8_0_wmma_vk128_store_acc_f16_row_major_w64_fast_half_buffer_full_pair(
            dst_rsrc,
            rows,
            tile_row0,
            tile_col0,
            acc[tile_iter],
            lane,
            wave,
            (hrx_q8_0_wmma_vk128_lds_volatile_half_ptr) sh_store);
#endif
#elif HRX_Q8_0_WMMA_VK128_FULL_TILE_STORE_PAIR
        hrx_q8_0_wmma_vk128_store_acc_f16_row_major_w64_buffer_full_pair(
            dst_rsrc,
            rows,
            tile_row0,
            tile_col0,
            acc[tile_iter],
            lane);
#else
        hrx_q8_0_wmma_vk128_store_acc_f16_row_major_w64_buffer_full(
            dst_rsrc,
            rows,
            tile_row0,
            tile_col0,
            acc[tile_iter],
            lane);
#endif
#else
        hrx_q8_0_wmma_vk128_store_acc_f16_row_major_w64_full(
            dst,
            rows,
            tile_row0,
            tile_col0,
            acc[tile_iter],
            lane);
#endif
#else
        if (tile_row0 + 16 <= rows && tile_col0 + 16 <= cols) {
#if HRX_Q8_0_WMMA_VK128_BUFFER_STORE
#if HRX_Q8_0_WMMA_VK128_STORE_STAGE_FAST_HALF
#if HRX_Q8_0_WMMA_VK128_STORE_STAGE_FAST_HALF_SPLIT_SELECTED
            hrx_q8_0_wmma_vk128_store_acc_f16_row_major_w64_fast_half_buffer_split_selected(
                dst_rsrc,
                rows,
                tile_row0,
                tile_col0,
                acc[tile_iter],
                lane,
                wave,
                (hrx_q8_0_wmma_vk128_lds_volatile_half_ptr) sh_store);
#elif HRX_Q8_0_WMMA_VK128_STORE_STAGE_FAST_HALF_SELECTED
            hrx_q8_0_wmma_vk128_store_acc_f16_row_major_w64_fast_half_buffer_selected(
                dst_rsrc,
                rows,
                tile_row0,
                tile_col0,
                acc[tile_iter],
                lane,
                wave,
                (hrx_q8_0_wmma_vk128_lds_volatile_half_ptr) sh_store);
#else
            hrx_q8_0_wmma_vk128_store_acc_f16_row_major_w64_fast_half_buffer_full_pair(
                dst_rsrc,
                rows,
                tile_row0,
                tile_col0,
                acc[tile_iter],
                lane,
                wave,
                (hrx_q8_0_wmma_vk128_lds_volatile_half_ptr) sh_store);
#endif
#elif HRX_Q8_0_WMMA_VK128_FULL_TILE_STORE_PAIR
            hrx_q8_0_wmma_vk128_store_acc_f16_row_major_w64_buffer_full_pair(
                dst_rsrc,
                rows,
                tile_row0,
                tile_col0,
                acc[tile_iter],
                lane);
#else
            hrx_q8_0_wmma_vk128_store_acc_f16_row_major_w64_buffer_full(
                dst_rsrc,
                rows,
                tile_row0,
                tile_col0,
                acc[tile_iter],
                lane);
#endif
#else
            hrx_q8_0_wmma_vk128_store_acc_f16_row_major_w64_full(
                dst,
                rows,
                tile_row0,
                tile_col0,
                acc[tile_iter],
                lane);
#endif
        } else {
#if HRX_Q8_0_WMMA_VK128_BUFFER_STORE
            hrx_q8_0_wmma_vk128_store_acc_f16_row_major_w64_buffer(
                dst_rsrc,
                rows,
                tile_row0,
                tile_col0,
                rows,
                cols,
                acc[tile_iter],
                lane);
#else
            hrx_q8_0_wmma_vk128_store_acc_f16_row_major_w64(
                dst,
                rows,
                tile_row0,
                tile_col0,
                rows,
                cols,
                acc[tile_iter],
                lane);
#endif
        }
#endif
#else
        hrx_q8_0_wmma_vk128_store_acc_f16_row_major_w64(
            dst,
            rows,
            row_base + static_cast<long long>(row_tile * 16),
            col_base + static_cast<long long>(col_tile * 16),
            rows,
            cols,
            acc[tile_iter],
            lane);
#endif
#endif
#else
        hrx_q8_0_wmma_vk128_store_acc_f16_row_major(
            dst,
            rows,
            row_base + static_cast<long long>(row_tile * 16),
            col_base + static_cast<long long>(col_tile * 16),
            rows,
            cols,
            acc[tile_iter],
            lane);
#endif
    }
#endif
}
