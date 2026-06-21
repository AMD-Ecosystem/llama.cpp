#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <stdint.h>

#ifndef HRX_Q5_K_WMMA_VK128_EXPORT
#define HRX_Q5_K_WMMA_VK128_EXPORT hrx_mul_mat_vec_q5_k_wmma16x16_vk128_f16acc_wg256_f32
#endif

#ifndef HRX_Q5_K_WMMA_VK128_SHARED_STRIDE
#define HRX_Q5_K_WMMA_VK128_SHARED_STRIDE 32
#endif

#ifndef HRX_Q5_K_WMMA_VK128_BM
#define HRX_Q5_K_WMMA_VK128_BM 128
#endif

#ifndef HRX_Q5_K_WMMA_VK128_BN
#define HRX_Q5_K_WMMA_VK128_BN 128
#endif

#ifndef HRX_Q5_K_WMMA_VK128_PREFETCH_FRAGS
#define HRX_Q5_K_WMMA_VK128_PREFETCH_FRAGS 0
#endif

#ifndef HRX_Q5_K_WMMA_VK128_PAIR64_TILE_MAP
#define HRX_Q5_K_WMMA_VK128_PAIR64_TILE_MAP 0
#endif

#ifndef HRX_Q5_K_WMMA_VK128_W64
#define HRX_Q5_K_WMMA_VK128_W64 0
#endif

#ifndef HRX_Q5_K_WMMA_VK128_W64_OPSEL
#define HRX_Q5_K_WMMA_VK128_W64_OPSEL 0
#endif

#ifndef HRX_Q5_K_WMMA_VK128_W64_H4LOAD
#define HRX_Q5_K_WMMA_VK128_W64_H4LOAD 0
#endif

#ifndef HRX_Q5_K_WMMA_VK128_W64_B64ASM
#define HRX_Q5_K_WMMA_VK128_W64_B64ASM 0
#endif

#ifndef HRX_Q5_K_WMMA_VK128_W64_B64GROUP
#define HRX_Q5_K_WMMA_VK128_W64_B64GROUP 0
#endif

#ifndef HRX_Q5_K_WMMA_VK128_W64_B64GROUP_RADVLADDER
#define HRX_Q5_K_WMMA_VK128_W64_B64GROUP_RADVLADDER 0
#endif

#ifndef HRX_Q5_K_WMMA_VK128_W64_VK64_GROUPK2
#define HRX_Q5_K_WMMA_VK128_W64_VK64_GROUPK2 0
#endif

#ifndef HRX_Q5_K_WMMA_VK128_W64_VK64_GROUPK2_WAIT
#define HRX_Q5_K_WMMA_VK128_W64_VK64_GROUPK2_WAIT 0
#endif

#ifndef HRX_Q5_K_WMMA_VK128_W64_VK64_COMBINED96
#define HRX_Q5_K_WMMA_VK128_W64_VK64_COMBINED96 0
#endif

#ifndef HRX_Q5_K_WMMA_VK128_W64_VK64_FULL64
#define HRX_Q5_K_WMMA_VK128_W64_VK64_FULL64 0
#endif

#ifndef HRX_Q5_K_WMMA_VK128_FULL_TILE_STORE
#define HRX_Q5_K_WMMA_VK128_FULL_TILE_STORE 0
#endif

#ifndef HRX_Q5_K_WMMA_VK128_STORE_STAGE
#define HRX_Q5_K_WMMA_VK128_STORE_STAGE 0
#endif

#ifndef HRX_Q5_K_WMMA_VK128_STORE_STAGE_BATCH_TILES
#define HRX_Q5_K_WMMA_VK128_STORE_STAGE_BATCH_TILES 0
#endif

#ifndef HRX_Q5_K_WMMA_VK128_STORE_STAGE_FAST_HALF
#define HRX_Q5_K_WMMA_VK128_STORE_STAGE_FAST_HALF 0
#endif

#ifndef HRX_Q5_K_WMMA_VK128_STORE_STAGE_FAST_HALF_SELECTED
#define HRX_Q5_K_WMMA_VK128_STORE_STAGE_FAST_HALF_SELECTED 0
#endif

#ifndef HRX_Q5_K_WMMA_VK128_MOTIF192_FULL_WRITEBACK
#define HRX_Q5_K_WMMA_VK128_MOTIF192_FULL_WRITEBACK 0
#endif

#ifndef HRX_Q5_K_WMMA_VK128_BUFFER_STORE
#define HRX_Q5_K_WMMA_VK128_BUFFER_STORE 0
#endif

#ifndef HRX_Q5_K_WMMA_VK128_PACK_STAGE_B32
#define HRX_Q5_K_WMMA_VK128_PACK_STAGE_B32 0
#endif

struct hrx_block_q5_K_wmma_vk128_lhs {
    unsigned short d;
    unsigned short dmin;
    uint8_t scales[12];
    uint8_t qh[32];
    uint8_t qs[128];
};

typedef _Float16 hrx_q5_k_wmma_vk128_half16_vec __attribute__((ext_vector_type(16)));
typedef _Float16 hrx_q5_k_wmma_vk128_half8_vec __attribute__((ext_vector_type(8)));
typedef _Float16 hrx_q5_k_wmma_vk128_half4_vec __attribute__((ext_vector_type(4)));
typedef const __attribute__((address_space(3))) _Float16 * hrx_q5_k_wmma_vk128_lds_half_ptr;
typedef volatile __attribute__((address_space(3))) _Float16 * hrx_q5_k_wmma_vk128_lds_volatile_half_ptr;
typedef __attribute__((address_space(3))) uint16_t * hrx_q5_k_wmma_vk128_lds_u16_ptr;
typedef const __attribute__((address_space(3))) uint16_t * hrx_q5_k_wmma_vk128_lds_const_u16_ptr;
typedef __attribute__((address_space(3))) uint32_t * hrx_q5_k_wmma_vk128_lds_u32_ptr;

#if HRX_Q5_K_WMMA_VK128_BUFFER_STORE
static constexpr int HRX_Q5_K_WMMA_VK128_RAW_BUFFER_FLAGS_GFX11 = 0x31004000;

static __device__ __forceinline__ __amdgpu_buffer_rsrc_t hrx_q5_k_wmma_vk128_make_dst_rsrc(float * dst) {
    return __builtin_amdgcn_make_buffer_rsrc(
        dst,
        static_cast<unsigned short>(0),
        0xffffffffull,
        HRX_Q5_K_WMMA_VK128_RAW_BUFFER_FLAGS_GFX11);
}

static __device__ __forceinline__ void hrx_q5_k_wmma_vk128_buffer_store_f32(
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

static __device__ __forceinline__ uint32_t hrx_q5_k_wmma_vk128_pack_f16x2(_Float16 lo, _Float16 hi) {
    union {
        _Float16 h[2];
        uint32_t u;
    } pack;
    pack.h[0] = lo;
    pack.h[1] = hi;
    return pack.u;
}

static __device__ __forceinline__ _Float16 hrx_q5_k_wmma_vk128_unpack_f16x2(uint32_t bits, int idx) {
    union {
        uint32_t u;
        _Float16 h[2];
    } pack;
    pack.u = bits;
    return pack.h[idx];
}

static __device__ __forceinline__ uint16_t hrx_q5_k_wmma_vk128_f16_to_u16(_Float16 value) {
    union {
        _Float16 h;
        uint16_t u;
    } pack;
    pack.h = value;
    return pack.u;
}

static __device__ __forceinline__ _Float16 hrx_q5_k_wmma_vk128_u16_to_f16(uint32_t value) {
    union {
        uint16_t u;
        _Float16 h;
    } pack;
    pack.u = static_cast<uint16_t>(value);
    return pack.h;
}

static __device__ __forceinline__ uint32_t hrx_q5_k_wmma_vk128_f16_pair_to_u32(_Float16 lo, _Float16 hi) {
    union {
        _Float16 h[2];
        uint32_t u;
    } pack;
    pack.h[0] = lo;
    pack.h[1] = hi;
    return pack.u;
}

#if HRX_Q5_K_WMMA_VK128_PACK_STAGE_B32
static __device__ __forceinline__ void hrx_q5_k_wmma_vk128_ds_store_u32(
        hrx_q5_k_wmma_vk128_lds_u32_ptr ptr,
        uint32_t value) {
    asm volatile("ds_write_b32 %0, %1 offset:0\n"
                 :
                 : "v"(ptr), "v"(value)
                 : "memory");
}
#endif

#if HRX_Q5_K_WMMA_VK128_STORE_STAGE_FAST_HALF
static __device__ __forceinline__ void hrx_q5_k_wmma_vk128_ds_store_u16(
        hrx_q5_k_wmma_vk128_lds_u16_ptr ptr,
        uint16_t value) {
    asm volatile("ds_write_b16 %0, %1 offset:0\n"
                 :
                 : "v"(ptr), "v"(static_cast<uint32_t>(value))
                 : "memory");
}

static __device__ __forceinline__ uint32_t hrx_q5_k_wmma_vk128_ds_load_u16_d16(
        hrx_q5_k_wmma_vk128_lds_const_u16_ptr ptr) {
    uint32_t value = 0;
    asm volatile("ds_read_u16_d16 %0, %1 offset:0\n"
                 : "=v"(value)
                 : "v"(ptr)
                 : "memory");
    return value;
}
#endif

static __device__ __forceinline__ hrx_q5_k_wmma_vk128_half16_vec hrx_q5_k_wmma_vk128_duplicate_input(
        _Float16 x0, _Float16 x1, _Float16 x2, _Float16 x3,
        _Float16 x4, _Float16 x5, _Float16 x6, _Float16 x7) {
    constexpr int SWAP16_CTRL = (16 << 10) | 0x1f;
    const uint32_t p0 = hrx_q5_k_wmma_vk128_pack_f16x2(x0, x1);
    const uint32_t p1 = hrx_q5_k_wmma_vk128_pack_f16x2(x2, x3);
    const uint32_t p2 = hrx_q5_k_wmma_vk128_pack_f16x2(x4, x5);
    const uint32_t p3 = hrx_q5_k_wmma_vk128_pack_f16x2(x6, x7);
    const uint32_t s0 = static_cast<uint32_t>(__builtin_amdgcn_ds_swizzle(static_cast<int32_t>(p0), SWAP16_CTRL));
    const uint32_t s1 = static_cast<uint32_t>(__builtin_amdgcn_ds_swizzle(static_cast<int32_t>(p1), SWAP16_CTRL));
    const uint32_t s2 = static_cast<uint32_t>(__builtin_amdgcn_ds_swizzle(static_cast<int32_t>(p2), SWAP16_CTRL));
    const uint32_t s3 = static_cast<uint32_t>(__builtin_amdgcn_ds_swizzle(static_cast<int32_t>(p3), SWAP16_CTRL));

    hrx_q5_k_wmma_vk128_half16_vec result;
    result[0] = x0;
    result[1] = x1;
    result[2] = x2;
    result[3] = x3;
    result[4] = x4;
    result[5] = x5;
    result[6] = x6;
    result[7] = x7;
    result[8] = hrx_q5_k_wmma_vk128_unpack_f16x2(s0, 0);
    result[9] = hrx_q5_k_wmma_vk128_unpack_f16x2(s0, 1);
    result[10] = hrx_q5_k_wmma_vk128_unpack_f16x2(s1, 0);
    result[11] = hrx_q5_k_wmma_vk128_unpack_f16x2(s1, 1);
    result[12] = hrx_q5_k_wmma_vk128_unpack_f16x2(s2, 0);
    result[13] = hrx_q5_k_wmma_vk128_unpack_f16x2(s2, 1);
    result[14] = hrx_q5_k_wmma_vk128_unpack_f16x2(s3, 0);
    result[15] = hrx_q5_k_wmma_vk128_unpack_f16x2(s3, 1);
    return result;
}

static __device__ __forceinline__ void hrx_q5_k_wmma_vk128_get_scale_min(
        int group, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (group < 4) {
        *d = q[group] & 63;
        *m = q[group + 4] & 63;
    } else {
        *d = (q[group + 4] & 0xF) | ((q[group - 4] >> 6) << 4);
        *m = (q[group + 4] >> 4) | ((q[group] >> 6) << 4);
    }
}

static __device__ __forceinline__ int hrx_q5_k_wmma_vk128_value(
        const hrx_block_q5_K_wmma_vk128_lhs * block,
        int in_block) {
    const int group = in_block >> 5;
    const int q_index = (group >> 1) * 32 + (in_block & 31);
    const uint8_t packed = block->qs[q_index];
    const uint8_t lo = (packed >> ((group & 1) * 4)) & 0x0f;
    const uint8_t hi = ((block->qh[in_block & 31] >> group) & 1) << 4;
    return lo | hi;
}

static __device__ __forceinline__ _Float16 hrx_q5_k_wmma_vk128_load_a_value(
        const hrx_block_q5_K_wmma_vk128_lhs * src0,
        long long row,
        long long k_index,
        long long blocks_per_row) {
    const hrx_block_q5_K_wmma_vk128_lhs * block = src0 + row * blocks_per_row + (k_index >> 8);
    const int in_block = static_cast<int>(k_index & 255);
    const int group = in_block >> 5;
    uint8_t sc = 0;
    uint8_t m = 0;
    hrx_q5_k_wmma_vk128_get_scale_min(group, block->scales, &sc, &m);
    const float d = __half2float(__ushort_as_half(block->d)) * static_cast<float>(sc);
    const float dmin = __half2float(__ushort_as_half(block->dmin)) * static_cast<float>(m);
    return static_cast<_Float16>(
        d * static_cast<float>(hrx_q5_k_wmma_vk128_value(block, in_block)) - dmin);
}

static __device__ __forceinline__ hrx_q5_k_wmma_vk128_half16_vec hrx_q5_k_wmma_vk128_load_a_frag(
        const _Float16 * sh_a,
        int row_tile,
        int k_tile,
        unsigned int lane) {
    constexpr int SHARED_STRIDE = HRX_Q5_K_WMMA_VK128_SHARED_STRIDE;
    const int row = row_tile * 16 + static_cast<int>(lane & 15u);
    const int k_base = k_tile * 16 + static_cast<int>(lane >> 4) * 8;
    return hrx_q5_k_wmma_vk128_duplicate_input(
        sh_a[row * SHARED_STRIDE + k_base + 0],
        sh_a[row * SHARED_STRIDE + k_base + 1],
        sh_a[row * SHARED_STRIDE + k_base + 2],
        sh_a[row * SHARED_STRIDE + k_base + 3],
        sh_a[row * SHARED_STRIDE + k_base + 4],
        sh_a[row * SHARED_STRIDE + k_base + 5],
        sh_a[row * SHARED_STRIDE + k_base + 6],
        sh_a[row * SHARED_STRIDE + k_base + 7]);
}

static __device__ __forceinline__ hrx_q5_k_wmma_vk128_half16_vec hrx_q5_k_wmma_vk128_load_b_frag(
        const _Float16 * sh_b,
        int col_tile,
        int k_tile,
        unsigned int lane) {
    constexpr int SHARED_STRIDE = HRX_Q5_K_WMMA_VK128_SHARED_STRIDE;
    const int col = col_tile * 16 + static_cast<int>(lane & 15u);
    const int k_base = k_tile * 16 + static_cast<int>(lane >> 4) * 8;
    return hrx_q5_k_wmma_vk128_duplicate_input(
        sh_b[col * SHARED_STRIDE + k_base + 0],
        sh_b[col * SHARED_STRIDE + k_base + 1],
        sh_b[col * SHARED_STRIDE + k_base + 2],
        sh_b[col * SHARED_STRIDE + k_base + 3],
        sh_b[col * SHARED_STRIDE + k_base + 4],
        sh_b[col * SHARED_STRIDE + k_base + 5],
        sh_b[col * SHARED_STRIDE + k_base + 6],
        sh_b[col * SHARED_STRIDE + k_base + 7]);
}

static __device__ __forceinline__ hrx_q5_k_wmma_vk128_half16_vec hrx_q5_k_wmma_vk128_load_a_frag_w64(
        const _Float16 * sh_a,
        int row_tile,
        int k_tile,
        unsigned int lane) {
    constexpr int SHARED_STRIDE = HRX_Q5_K_WMMA_VK128_SHARED_STRIDE;
    const int row = row_tile * 16 + static_cast<int>(lane & 15u);
    const int k_base = k_tile * 16;
    hrx_q5_k_wmma_vk128_half16_vec result;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        result[i] = sh_a[row * SHARED_STRIDE + k_base + i];
    }
    return result;
}

static __device__ __forceinline__ hrx_q5_k_wmma_vk128_half16_vec hrx_q5_k_wmma_vk128_load_b_frag_w64(
        const _Float16 * sh_b,
        int col_tile,
        int k_tile,
        unsigned int lane) {
    constexpr int SHARED_STRIDE = HRX_Q5_K_WMMA_VK128_SHARED_STRIDE;
    const int col = col_tile * 16 + static_cast<int>(lane & 15u);
    const int k_base = k_tile * 16;
    hrx_q5_k_wmma_vk128_half16_vec result;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        result[i] = sh_b[col * SHARED_STRIDE + k_base + i];
    }
    return result;
}

static __device__ __forceinline__ void hrx_q5_k_wmma_vk128_append_half4(
        hrx_q5_k_wmma_vk128_half16_vec * result,
        int base,
        hrx_q5_k_wmma_vk128_half4_vec values) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        (*result)[base + i] = values[i];
    }
}

static __device__ __forceinline__ hrx_q5_k_wmma_vk128_half4_vec hrx_q5_k_wmma_vk128_ds_read_b64_h4(
        hrx_q5_k_wmma_vk128_lds_half_ptr ptr) {
    const __attribute__((address_space(3))) uint64_t * lds_ptr =
        (const __attribute__((address_space(3))) uint64_t *) ptr;
    hrx_q5_k_wmma_vk128_half4_vec value;
    asm volatile("ds_read_b64 %0, %1 offset:0\n"
                 : "=v"(value)
                 : "v"(lds_ptr)
                 : "memory");
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    return value;
}

static __device__ __forceinline__ hrx_q5_k_wmma_vk128_half4_vec hrx_q5_k_wmma_vk128_ds_read_b64_h4_nowait(
        hrx_q5_k_wmma_vk128_lds_half_ptr ptr) {
    const __attribute__((address_space(3))) uint64_t * lds_ptr =
        (const __attribute__((address_space(3))) uint64_t *) ptr;
    hrx_q5_k_wmma_vk128_half4_vec value;
    asm volatile("ds_read_b64 %0, %1 offset:0\n"
                 : "=v"(value)
                 : "v"(lds_ptr)
                 : "memory");
    return value;
}

static __device__ __forceinline__ hrx_q5_k_wmma_vk128_half16_vec hrx_q5_k_wmma_vk128_load_a_frag_w64_b64asm(
        hrx_q5_k_wmma_vk128_lds_half_ptr sh_a,
        int row_tile,
        int k_tile,
        unsigned int lane) {
    constexpr int SHARED_STRIDE = HRX_Q5_K_WMMA_VK128_SHARED_STRIDE;
    const int row = row_tile * 16 + static_cast<int>(lane & 15u);
    const int k_base = k_tile * 16;
    hrx_q5_k_wmma_vk128_lds_half_ptr row_ptr = sh_a + row * SHARED_STRIDE + k_base;
    hrx_q5_k_wmma_vk128_half16_vec result;
    hrx_q5_k_wmma_vk128_append_half4(&result, 0, hrx_q5_k_wmma_vk128_ds_read_b64_h4(row_ptr + 0));
    hrx_q5_k_wmma_vk128_append_half4(&result, 4, hrx_q5_k_wmma_vk128_ds_read_b64_h4(row_ptr + 4));
    hrx_q5_k_wmma_vk128_append_half4(&result, 8, hrx_q5_k_wmma_vk128_ds_read_b64_h4(row_ptr + 8));
    hrx_q5_k_wmma_vk128_append_half4(&result, 12, hrx_q5_k_wmma_vk128_ds_read_b64_h4(row_ptr + 12));
    return result;
}

static __device__ __forceinline__ hrx_q5_k_wmma_vk128_half16_vec hrx_q5_k_wmma_vk128_load_a_frag_w64_b64asm_nowait(
        hrx_q5_k_wmma_vk128_lds_half_ptr sh_a,
        int row_tile,
        int k_tile,
        unsigned int lane) {
    constexpr int SHARED_STRIDE = HRX_Q5_K_WMMA_VK128_SHARED_STRIDE;
    const int row = row_tile * 16 + static_cast<int>(lane & 15u);
    const int k_base = k_tile * 16;
    hrx_q5_k_wmma_vk128_lds_half_ptr row_ptr = sh_a + row * SHARED_STRIDE + k_base;
    hrx_q5_k_wmma_vk128_half16_vec result;
    hrx_q5_k_wmma_vk128_append_half4(&result, 0, hrx_q5_k_wmma_vk128_ds_read_b64_h4_nowait(row_ptr + 0));
    hrx_q5_k_wmma_vk128_append_half4(&result, 4, hrx_q5_k_wmma_vk128_ds_read_b64_h4_nowait(row_ptr + 4));
    hrx_q5_k_wmma_vk128_append_half4(&result, 8, hrx_q5_k_wmma_vk128_ds_read_b64_h4_nowait(row_ptr + 8));
    hrx_q5_k_wmma_vk128_append_half4(&result, 12, hrx_q5_k_wmma_vk128_ds_read_b64_h4_nowait(row_ptr + 12));
    return result;
}

static __device__ __forceinline__ hrx_q5_k_wmma_vk128_half16_vec hrx_q5_k_wmma_vk128_load_b_frag_w64_b64asm(
        hrx_q5_k_wmma_vk128_lds_half_ptr sh_b,
        int col_tile,
        int k_tile,
        unsigned int lane) {
    constexpr int SHARED_STRIDE = HRX_Q5_K_WMMA_VK128_SHARED_STRIDE;
    const int col = col_tile * 16 + static_cast<int>(lane & 15u);
    const int k_base = k_tile * 16;
    hrx_q5_k_wmma_vk128_lds_half_ptr col_ptr = sh_b + col * SHARED_STRIDE + k_base;
    hrx_q5_k_wmma_vk128_half16_vec result;
    hrx_q5_k_wmma_vk128_append_half4(&result, 0, hrx_q5_k_wmma_vk128_ds_read_b64_h4(col_ptr + 0));
    hrx_q5_k_wmma_vk128_append_half4(&result, 4, hrx_q5_k_wmma_vk128_ds_read_b64_h4(col_ptr + 4));
    hrx_q5_k_wmma_vk128_append_half4(&result, 8, hrx_q5_k_wmma_vk128_ds_read_b64_h4(col_ptr + 8));
    hrx_q5_k_wmma_vk128_append_half4(&result, 12, hrx_q5_k_wmma_vk128_ds_read_b64_h4(col_ptr + 12));
    return result;
}

static __device__ __forceinline__ hrx_q5_k_wmma_vk128_half16_vec hrx_q5_k_wmma_vk128_load_b_frag_w64_b64asm_nowait(
        hrx_q5_k_wmma_vk128_lds_half_ptr sh_b,
        int col_tile,
        int k_tile,
        unsigned int lane) {
    constexpr int SHARED_STRIDE = HRX_Q5_K_WMMA_VK128_SHARED_STRIDE;
    const int col = col_tile * 16 + static_cast<int>(lane & 15u);
    const int k_base = k_tile * 16;
    hrx_q5_k_wmma_vk128_lds_half_ptr col_ptr = sh_b + col * SHARED_STRIDE + k_base;
    hrx_q5_k_wmma_vk128_half16_vec result;
    hrx_q5_k_wmma_vk128_append_half4(&result, 0, hrx_q5_k_wmma_vk128_ds_read_b64_h4_nowait(col_ptr + 0));
    hrx_q5_k_wmma_vk128_append_half4(&result, 4, hrx_q5_k_wmma_vk128_ds_read_b64_h4_nowait(col_ptr + 4));
    hrx_q5_k_wmma_vk128_append_half4(&result, 8, hrx_q5_k_wmma_vk128_ds_read_b64_h4_nowait(col_ptr + 8));
    hrx_q5_k_wmma_vk128_append_half4(&result, 12, hrx_q5_k_wmma_vk128_ds_read_b64_h4_nowait(col_ptr + 12));
    return result;
}

#define HRX_Q5_K_WMMA_VK128_WAIT_WMMA(TILE, A, B, W) \
    do { \
        asm volatile("s_waitcnt lgkmcnt(%0)\n" :: "n"(W) : "memory"); \
        acc[TILE] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64( \
            (A), (B), acc[TILE], HRX_Q5_K_WMMA_VK128_W64_OPSEL != 0); \
    } while (0)

static __device__ __forceinline__ hrx_q5_k_wmma_vk128_half16_vec hrx_q5_k_wmma_vk128_load_a_frag_w64_h4(
        const _Float16 * sh_a,
        int row_tile,
        int k_tile,
        unsigned int lane) {
    constexpr int SHARED_STRIDE = HRX_Q5_K_WMMA_VK128_SHARED_STRIDE;
    const int row = row_tile * 16 + static_cast<int>(lane & 15u);
    const int k_base = k_tile * 16;
    const hrx_q5_k_wmma_vk128_half4_vec * row_ptr =
        reinterpret_cast<const hrx_q5_k_wmma_vk128_half4_vec *>(sh_a + row * SHARED_STRIDE + k_base);
    hrx_q5_k_wmma_vk128_half16_vec result;
    hrx_q5_k_wmma_vk128_append_half4(&result, 0, row_ptr[0]);
    hrx_q5_k_wmma_vk128_append_half4(&result, 4, row_ptr[1]);
    hrx_q5_k_wmma_vk128_append_half4(&result, 8, row_ptr[2]);
    hrx_q5_k_wmma_vk128_append_half4(&result, 12, row_ptr[3]);
    return result;
}

static __device__ __forceinline__ hrx_q5_k_wmma_vk128_half16_vec hrx_q5_k_wmma_vk128_load_b_frag_w64_h4(
        const _Float16 * sh_b,
        int col_tile,
        int k_tile,
        unsigned int lane) {
    constexpr int SHARED_STRIDE = HRX_Q5_K_WMMA_VK128_SHARED_STRIDE;
    const int col = col_tile * 16 + static_cast<int>(lane & 15u);
    const int k_base = k_tile * 16;
    const hrx_q5_k_wmma_vk128_half4_vec * col_ptr =
        reinterpret_cast<const hrx_q5_k_wmma_vk128_half4_vec *>(sh_b + col * SHARED_STRIDE + k_base);
    hrx_q5_k_wmma_vk128_half16_vec result;
    hrx_q5_k_wmma_vk128_append_half4(&result, 0, col_ptr[0]);
    hrx_q5_k_wmma_vk128_append_half4(&result, 4, col_ptr[1]);
    hrx_q5_k_wmma_vk128_append_half4(&result, 8, col_ptr[2]);
    hrx_q5_k_wmma_vk128_append_half4(&result, 12, col_ptr[3]);
    return result;
}

static __device__ __forceinline__ void hrx_q5_k_wmma_vk128_store_acc_f16_row_major(
        float * dst,
        long long rows_stride,
        long long row0,
        long long col0,
        long long rows,
        long long cols,
        hrx_q5_k_wmma_vk128_half16_vec acc,
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

static __device__ __forceinline__ void hrx_q5_k_wmma_vk128_store_acc_f16_row_major_w64(
        float * dst,
        long long rows_stride,
        long long row0,
        long long col0,
        long long rows,
        long long cols,
        hrx_q5_k_wmma_vk128_half8_vec acc,
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
                static_cast<float>(acc[reg * 2 + HRX_Q5_K_WMMA_VK128_W64_OPSEL]);
        }
    }
}

static __device__ __forceinline__ void hrx_q5_k_wmma_vk128_store_acc_f16_row_major_w64_full(
        float * dst,
        long long rows_stride,
        long long row0,
        long long col0,
        hrx_q5_k_wmma_vk128_half8_vec acc,
        unsigned int lane) {
    const long long row_lane = static_cast<long long>(lane >> 4);
    const long long col = col0 + static_cast<long long>(lane & 15u);
#pragma unroll
    for (int reg = 0; reg < 4; ++reg) {
        const long long row = row0 + row_lane + static_cast<long long>(reg * 4);
        dst[col * rows_stride + row] =
            static_cast<float>(acc[reg * 2 + HRX_Q5_K_WMMA_VK128_W64_OPSEL]);
    }
}

#if HRX_Q5_K_WMMA_VK128_BUFFER_STORE
static __device__ __forceinline__ void hrx_q5_k_wmma_vk128_store_acc_f16_row_major_w64_buffer_full(
        __amdgpu_buffer_rsrc_t dst_rsrc,
        long long rows_stride,
        long long row0,
        long long col0,
        hrx_q5_k_wmma_vk128_half8_vec acc,
        unsigned int lane) {
    const long long row_lane = static_cast<long long>(lane >> 4);
    const long long col = col0 + static_cast<long long>(lane & 15u);
#pragma unroll
    for (int reg = 0; reg < 4; ++reg) {
        const long long row = row0 + row_lane + static_cast<long long>(reg * 4);
        hrx_q5_k_wmma_vk128_buffer_store_f32(
            dst_rsrc,
            col * rows_stride + row,
            static_cast<float>(acc[reg * 2 + HRX_Q5_K_WMMA_VK128_W64_OPSEL]));
    }
}

static __device__ __forceinline__ void hrx_q5_k_wmma_vk128_store_acc_f16_row_major_w64_buffer(
        __amdgpu_buffer_rsrc_t dst_rsrc,
        long long rows_stride,
        long long row0,
        long long col0,
        long long rows,
        long long cols,
        hrx_q5_k_wmma_vk128_half8_vec acc,
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
            hrx_q5_k_wmma_vk128_buffer_store_f32(
                dst_rsrc,
                col * rows_stride + row,
                static_cast<float>(acc[reg * 2 + HRX_Q5_K_WMMA_VK128_W64_OPSEL]));
        }
    }
}
#endif

#if HRX_Q5_K_WMMA_VK128_STORE_STAGE_FAST_HALF && HRX_Q5_K_WMMA_VK128_BUFFER_STORE
static __device__ __forceinline__ int hrx_q5_k_wmma_vk128_motif192_stage_index(
        unsigned int wave,
        int group,
        int slot,
        unsigned int lane) {
    (void) group;
    const int row_lane = static_cast<int>(lane >> 4);
    const int col_lane = static_cast<int>(lane & 15u);
    return static_cast<int>(wave) * 16 * 16 + col_lane * 16 + row_lane + slot * 4;
}

static __device__ __forceinline__ void hrx_q5_k_wmma_vk128_motif192_raw_store_slot(
        __amdgpu_buffer_rsrc_t dst_rsrc,
        long long rows_stride,
        long long row_base,
        long long col_base,
        long long rows,
        long long cols,
        const hrx_q5_k_wmma_vk128_half8_vec * acc,
        unsigned int wave,
        int group,
        int slot,
        unsigned int lane) {
    const int group16 = group & 15;
    const int wave_row = static_cast<int>(wave & 1u);
    const int wave_col = static_cast<int>(wave >> 1);
    const int row_lane = static_cast<int>(lane >> 4);
    const int col_lane = static_cast<int>(lane & 15u);
    const long long row = row_base + static_cast<long long>(
        wave_row * 64 + (group16 & 3) * 16 + row_lane + slot * 4);
    const long long col = col_base + static_cast<long long>(
        wave_col * 64 + ((group16 >> 2) & 3) * 16 + col_lane);
    if (row < rows && col < cols) {
        hrx_q5_k_wmma_vk128_buffer_store_f32(
            dst_rsrc,
            col * rows_stride + row,
            static_cast<float>(acc[group16][slot * 2 + HRX_Q5_K_WMMA_VK128_W64_OPSEL]));
    }
}

static __device__ __forceinline__ void hrx_q5_k_wmma_vk128_motif192_stage_store_slot(
        hrx_q5_k_wmma_vk128_lds_volatile_half_ptr sh_store,
        const hrx_q5_k_wmma_vk128_half8_vec * acc,
        unsigned int wave,
        int group,
        int slot,
        unsigned int lane) {
    const int acc_index = group & 15;
    hrx_q5_k_wmma_vk128_lds_u16_ptr sh_u16 =
        (hrx_q5_k_wmma_vk128_lds_u16_ptr) sh_store;
    hrx_q5_k_wmma_vk128_ds_store_u16(
        sh_u16 + hrx_q5_k_wmma_vk128_motif192_stage_index(wave, group, slot, lane),
        hrx_q5_k_wmma_vk128_f16_to_u16(acc[acc_index][slot * 2 + HRX_Q5_K_WMMA_VK128_W64_OPSEL]));
}

static __device__ __forceinline__ void hrx_q5_k_wmma_vk128_motif192_stage_load_store_slot(
        __amdgpu_buffer_rsrc_t dst_rsrc,
        long long rows_stride,
        long long row_base,
        long long col_base,
        long long rows,
        long long cols,
        hrx_q5_k_wmma_vk128_lds_volatile_half_ptr sh_store,
        unsigned int wave,
        int group,
        int slot,
        unsigned int lane) {
    const int group16 = group & 15;
    const int wave_row = static_cast<int>(wave & 1u);
    const int wave_col = static_cast<int>(wave >> 1);
    const int row_lane = static_cast<int>(lane >> 4);
    const int col_lane = static_cast<int>(lane & 15u);
    const long long row = row_base + static_cast<long long>(
        wave_row * 64 + (group16 & 3) * 16 + row_lane + slot * 4);
    const long long col = col_base + static_cast<long long>(
        wave_col * 64 + ((group16 >> 2) & 3) * 16 + col_lane);
    hrx_q5_k_wmma_vk128_lds_const_u16_ptr sh_u16 =
        (hrx_q5_k_wmma_vk128_lds_const_u16_ptr) sh_store;
    const _Float16 value = hrx_q5_k_wmma_vk128_u16_to_f16(
        hrx_q5_k_wmma_vk128_ds_load_u16_d16(
            sh_u16 + hrx_q5_k_wmma_vk128_motif192_stage_index(wave, group, slot, lane)));
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    if (row < rows && col < cols) {
        hrx_q5_k_wmma_vk128_buffer_store_f32(dst_rsrc, col * rows_stride + row, static_cast<float>(value));
    }
}

#define HRX_Q5_K_WMMA_VK128_MOTIF192_RAW_STORE_GROUP(GROUP_ID) do { \
    hrx_q5_k_wmma_vk128_motif192_raw_store_slot(dst_rsrc, rows, row_base, col_base, rows, cols, acc, wave, (GROUP_ID), 0, lane); \
    hrx_q5_k_wmma_vk128_motif192_raw_store_slot(dst_rsrc, rows, row_base, col_base, rows, cols, acc, wave, (GROUP_ID), 1, lane); \
    hrx_q5_k_wmma_vk128_motif192_raw_store_slot(dst_rsrc, rows, row_base, col_base, rows, cols, acc, wave, (GROUP_ID), 2, lane); \
    hrx_q5_k_wmma_vk128_motif192_raw_store_slot(dst_rsrc, rows, row_base, col_base, rows, cols, acc, wave, (GROUP_ID), 3, lane); \
} while (0)

#define HRX_Q5_K_WMMA_VK128_MOTIF192_STAGE_STORE_GROUP(GROUP_ID) do { \
    hrx_q5_k_wmma_vk128_motif192_stage_store_slot((hrx_q5_k_wmma_vk128_lds_volatile_half_ptr) sh_store, acc, wave, (GROUP_ID), 0, lane); \
    hrx_q5_k_wmma_vk128_motif192_stage_store_slot((hrx_q5_k_wmma_vk128_lds_volatile_half_ptr) sh_store, acc, wave, (GROUP_ID), 1, lane); \
    hrx_q5_k_wmma_vk128_motif192_stage_store_slot((hrx_q5_k_wmma_vk128_lds_volatile_half_ptr) sh_store, acc, wave, (GROUP_ID), 2, lane); \
    hrx_q5_k_wmma_vk128_motif192_stage_store_slot((hrx_q5_k_wmma_vk128_lds_volatile_half_ptr) sh_store, acc, wave, (GROUP_ID), 3, lane); \
} while (0)

#define HRX_Q5_K_WMMA_VK128_MOTIF192_STAGE_LOAD_STORE_GROUP(GROUP_ID) do { \
    hrx_q5_k_wmma_vk128_motif192_stage_load_store_slot(dst_rsrc, rows, row_base, col_base, rows, cols, (hrx_q5_k_wmma_vk128_lds_volatile_half_ptr) sh_store, wave, (GROUP_ID), 0, lane); \
    hrx_q5_k_wmma_vk128_motif192_stage_load_store_slot(dst_rsrc, rows, row_base, col_base, rows, cols, (hrx_q5_k_wmma_vk128_lds_volatile_half_ptr) sh_store, wave, (GROUP_ID), 1, lane); \
    hrx_q5_k_wmma_vk128_motif192_stage_load_store_slot(dst_rsrc, rows, row_base, col_base, rows, cols, (hrx_q5_k_wmma_vk128_lds_volatile_half_ptr) sh_store, wave, (GROUP_ID), 2, lane); \
    hrx_q5_k_wmma_vk128_motif192_stage_load_store_slot(dst_rsrc, rows, row_base, col_base, rows, cols, (hrx_q5_k_wmma_vk128_lds_volatile_half_ptr) sh_store, wave, (GROUP_ID), 3, lane); \
} while (0)

#define HRX_Q5_K_WMMA_VK128_MOTIF192_GROUP(GROUP_ID) do { \
    HRX_Q5_K_WMMA_VK128_MOTIF192_RAW_STORE_GROUP((GROUP_ID)); \
    HRX_Q5_K_WMMA_VK128_MOTIF192_STAGE_STORE_GROUP((GROUP_ID) + 16); \
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory"); \
    HRX_Q5_K_WMMA_VK128_MOTIF192_STAGE_LOAD_STORE_GROUP((GROUP_ID) + 16); \
    HRX_Q5_K_WMMA_VK128_MOTIF192_STAGE_STORE_GROUP((GROUP_ID) + 32); \
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory"); \
    HRX_Q5_K_WMMA_VK128_MOTIF192_STAGE_LOAD_STORE_GROUP((GROUP_ID) + 32); \
} while (0)

#define HRX_Q5_K_WMMA_VK128_MOTIF192_GROUPS_0_15() do { \
    HRX_Q5_K_WMMA_VK128_MOTIF192_GROUP(0);  HRX_Q5_K_WMMA_VK128_MOTIF192_GROUP(1); \
    HRX_Q5_K_WMMA_VK128_MOTIF192_GROUP(2);  HRX_Q5_K_WMMA_VK128_MOTIF192_GROUP(3); \
    HRX_Q5_K_WMMA_VK128_MOTIF192_GROUP(4);  HRX_Q5_K_WMMA_VK128_MOTIF192_GROUP(5); \
    HRX_Q5_K_WMMA_VK128_MOTIF192_GROUP(6);  HRX_Q5_K_WMMA_VK128_MOTIF192_GROUP(7); \
    HRX_Q5_K_WMMA_VK128_MOTIF192_GROUP(8);  HRX_Q5_K_WMMA_VK128_MOTIF192_GROUP(9); \
    HRX_Q5_K_WMMA_VK128_MOTIF192_GROUP(10); HRX_Q5_K_WMMA_VK128_MOTIF192_GROUP(11); \
    HRX_Q5_K_WMMA_VK128_MOTIF192_GROUP(12); HRX_Q5_K_WMMA_VK128_MOTIF192_GROUP(13); \
    HRX_Q5_K_WMMA_VK128_MOTIF192_GROUP(14); HRX_Q5_K_WMMA_VK128_MOTIF192_GROUP(15); \
} while (0)

static __device__ __forceinline__ void hrx_q5_k_wmma_vk128_store_acc_f16_row_major_w64_fast_half_buffer_selected(
        __amdgpu_buffer_rsrc_t dst_rsrc,
        long long rows_stride,
        long long row0,
        long long col0,
        hrx_q5_k_wmma_vk128_half8_vec acc,
        unsigned int lane,
        unsigned int wave,
        hrx_q5_k_wmma_vk128_lds_volatile_half_ptr sh_store) {
    constexpr int TILE_STRIDE = 16 * 16;
    constexpr int SELECTED_OPSEL = HRX_Q5_K_WMMA_VK128_W64_OPSEL;
    constexpr int OTHER_OPSEL = SELECTED_OPSEL == 0 ? 1 : 0;
    const int row_lane = static_cast<int>(lane >> 4);
    const int col_lane = static_cast<int>(lane & 15u);
    const int tile_base = static_cast<int>(wave) * TILE_STRIDE;
    const int col_major_base = tile_base + col_lane * 16 + row_lane;
    hrx_q5_k_wmma_vk128_lds_u16_ptr sh_u16 =
        (hrx_q5_k_wmma_vk128_lds_u16_ptr) sh_store;
    const long long col = col0 + static_cast<long long>(col_lane);
#pragma unroll
    for (int reg = 0; reg < 4; ++reg) {
        const int selected_offset =
            col_major_base + ((reg * 4 + SELECTED_OPSEL * 2) & 15);
        const int other_offset =
            col_major_base + ((reg * 4 + OTHER_OPSEL * 2) & 15);
        hrx_q5_k_wmma_vk128_ds_store_u16(
            sh_u16 + selected_offset,
            hrx_q5_k_wmma_vk128_f16_to_u16(acc[reg * 2 + SELECTED_OPSEL]));
        hrx_q5_k_wmma_vk128_ds_store_u16(
            sh_u16 + other_offset,
            hrx_q5_k_wmma_vk128_f16_to_u16(acc[reg * 2 + OTHER_OPSEL]));
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");

#pragma unroll
    for (int reg = 0; reg < 4; ++reg) {
        const int selected_offset =
            col_major_base + ((reg * 4 + SELECTED_OPSEL * 2) & 15);
        const int other_offset =
            col_major_base + ((reg * 4 + OTHER_OPSEL * 2) & 15);
        const _Float16 selected = hrx_q5_k_wmma_vk128_u16_to_f16(
            hrx_q5_k_wmma_vk128_ds_load_u16_d16(sh_u16 + selected_offset));
        const uint32_t other_bits = hrx_q5_k_wmma_vk128_ds_load_u16_d16(sh_u16 + other_offset);
        (void) other_bits;
        asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
        const long long row = row0 + static_cast<long long>(row_lane + reg * 4);
        hrx_q5_k_wmma_vk128_buffer_store_f32(
            dst_rsrc,
            col * rows_stride + row,
            static_cast<float>(selected));
    }
}
#endif

#if (HRX_Q5_K_WMMA_VK128_W64_VK64_COMBINED96 || HRX_Q5_K_WMMA_VK128_W64_VK64_FULL64) && HRX_Q5_K_WMMA_VK128_BUFFER_STORE
#if HRX_Q5_K_WMMA_VK128_W64_VK64_COMBINED96 && HRX_Q5_K_WMMA_VK128_STORE_STAGE_FAST_HALF
static __device__ __forceinline__ hrx_q5_k_wmma_vk128_lds_u16_ptr hrx_q5_k_wmma_vk128_combined96_stage_ptr(
        _Float16 * sh_a,
        _Float16 * sh_b,
        int index) {
    constexpr int SH_A_HALF_COUNT = HRX_Q5_K_WMMA_VK128_BM * HRX_Q5_K_WMMA_VK128_SHARED_STRIDE;
    hrx_q5_k_wmma_vk128_lds_u16_ptr sh_a_u16 = (hrx_q5_k_wmma_vk128_lds_u16_ptr) sh_a;
    hrx_q5_k_wmma_vk128_lds_u16_ptr sh_b_u16 = (hrx_q5_k_wmma_vk128_lds_u16_ptr) sh_b;
    return index < SH_A_HALF_COUNT ? sh_a_u16 + index : sh_b_u16 + (index - SH_A_HALF_COUNT);
}

static __device__ __forceinline__ hrx_q5_k_wmma_vk128_lds_const_u16_ptr hrx_q5_k_wmma_vk128_combined96_stage_const_ptr(
        const _Float16 * sh_a,
        const _Float16 * sh_b,
        int index) {
    constexpr int SH_A_HALF_COUNT = HRX_Q5_K_WMMA_VK128_BM * HRX_Q5_K_WMMA_VK128_SHARED_STRIDE;
    hrx_q5_k_wmma_vk128_lds_const_u16_ptr sh_a_u16 = (hrx_q5_k_wmma_vk128_lds_const_u16_ptr) sh_a;
    hrx_q5_k_wmma_vk128_lds_const_u16_ptr sh_b_u16 = (hrx_q5_k_wmma_vk128_lds_const_u16_ptr) sh_b;
    return index < SH_A_HALF_COUNT ? sh_a_u16 + index : sh_b_u16 + (index - SH_A_HALF_COUNT);
}

static __device__ __forceinline__ int hrx_q5_k_wmma_vk128_combined96_stage_index(
        int group,
        int slot,
        unsigned int lane) {
    const int stage_group = group - 8;
    const int row_lane = static_cast<int>(lane >> 4);
    const int col_lane = static_cast<int>(lane & 15u);
    return stage_group * 16 * 16 + col_lane * 16 + row_lane + slot * 4;
}
#endif

static __device__ __forceinline__ void hrx_q5_k_wmma_vk128_combined96_raw_store_slot(
        __amdgpu_buffer_rsrc_t dst_rsrc,
        long long rows_stride,
        long long row_base,
        long long col_base,
        long long rows,
        long long cols,
        const hrx_q5_k_wmma_vk128_half8_vec * acc,
        int group,
        int slot,
        unsigned int lane) {
    const int row_lane = static_cast<int>(lane >> 4);
    const int col_lane = static_cast<int>(lane & 15u);
#if HRX_Q5_K_WMMA_VK128_W64_VK64_FULL64
    const int acc_index = group;
#else
    const int acc_index = group & 7;
#endif
    const long long row = row_base + static_cast<long long>((group & 3) * 16 + row_lane + slot * 4);
    const long long col = col_base + static_cast<long long>(((group >> 2) & 3) * 16 + col_lane);
    if (row < rows && col < cols) {
        hrx_q5_k_wmma_vk128_buffer_store_f32(
            dst_rsrc,
            col * rows_stride + row,
            static_cast<float>(acc[acc_index][slot * 2 + HRX_Q5_K_WMMA_VK128_W64_OPSEL]));
    }
}

#if HRX_Q5_K_WMMA_VK128_W64_VK64_COMBINED96 && HRX_Q5_K_WMMA_VK128_STORE_STAGE_FAST_HALF
static __device__ __forceinline__ void hrx_q5_k_wmma_vk128_combined96_stage_store_slot(
        _Float16 * sh_a,
        _Float16 * sh_b,
        const hrx_q5_k_wmma_vk128_half8_vec * acc,
        int group,
        int slot,
        unsigned int lane) {
    const int acc_index = group & 7;
    hrx_q5_k_wmma_vk128_ds_store_u16(
        hrx_q5_k_wmma_vk128_combined96_stage_ptr(
            sh_a, sh_b, hrx_q5_k_wmma_vk128_combined96_stage_index(group, slot, lane)),
        hrx_q5_k_wmma_vk128_f16_to_u16(acc[acc_index][slot * 2 + HRX_Q5_K_WMMA_VK128_W64_OPSEL]));
}

static __device__ __forceinline__ void hrx_q5_k_wmma_vk128_combined96_stage_load_store_slot(
        __amdgpu_buffer_rsrc_t dst_rsrc,
        long long rows_stride,
        long long row_base,
        long long col_base,
        long long rows,
        long long cols,
        const _Float16 * sh_a,
        const _Float16 * sh_b,
        int group,
        int slot,
        unsigned int lane) {
    const int row_lane = static_cast<int>(lane >> 4);
    const int col_lane = static_cast<int>(lane & 15u);
    const long long row = row_base + static_cast<long long>((group & 3) * 16 + row_lane + slot * 4);
    const long long col = col_base + static_cast<long long>(((group >> 2) & 3) * 16 + col_lane);
    const _Float16 value = hrx_q5_k_wmma_vk128_u16_to_f16(
        hrx_q5_k_wmma_vk128_ds_load_u16_d16(
            hrx_q5_k_wmma_vk128_combined96_stage_const_ptr(
                sh_a, sh_b, hrx_q5_k_wmma_vk128_combined96_stage_index(group, slot, lane))));
    if (row < rows && col < cols) {
        hrx_q5_k_wmma_vk128_buffer_store_f32(dst_rsrc, col * rows_stride + row, static_cast<float>(value));
    }
}
#endif

#define HRX_Q5_K_WMMA_VK128_COMBINED96_RAW_STORE_GROUP(GROUP_ID) do { \
    hrx_q5_k_wmma_vk128_combined96_raw_store_slot(dst_rsrc, rows, row_base, col_base, rows, cols, acc, (GROUP_ID), 0, lane); \
    hrx_q5_k_wmma_vk128_combined96_raw_store_slot(dst_rsrc, rows, row_base, col_base, rows, cols, acc, (GROUP_ID), 1, lane); \
    hrx_q5_k_wmma_vk128_combined96_raw_store_slot(dst_rsrc, rows, row_base, col_base, rows, cols, acc, (GROUP_ID), 2, lane); \
    hrx_q5_k_wmma_vk128_combined96_raw_store_slot(dst_rsrc, rows, row_base, col_base, rows, cols, acc, (GROUP_ID), 3, lane); \
} while (0)

#if HRX_Q5_K_WMMA_VK128_W64_VK64_COMBINED96 && HRX_Q5_K_WMMA_VK128_STORE_STAGE_FAST_HALF
#define HRX_Q5_K_WMMA_VK128_COMBINED96_STAGE_STORE_GROUP(GROUP_ID) do { \
    hrx_q5_k_wmma_vk128_combined96_stage_store_slot(sh_a, sh_b, acc, (GROUP_ID), 0, lane); \
    hrx_q5_k_wmma_vk128_combined96_stage_store_slot(sh_a, sh_b, acc, (GROUP_ID), 1, lane); \
    hrx_q5_k_wmma_vk128_combined96_stage_store_slot(sh_a, sh_b, acc, (GROUP_ID), 2, lane); \
    hrx_q5_k_wmma_vk128_combined96_stage_store_slot(sh_a, sh_b, acc, (GROUP_ID), 3, lane); \
} while (0)

#define HRX_Q5_K_WMMA_VK128_COMBINED96_STAGE_LOAD_STORE_GROUP(GROUP_ID) do { \
    hrx_q5_k_wmma_vk128_combined96_stage_load_store_slot(dst_rsrc, rows, row_base, col_base, rows, cols, sh_a, sh_b, (GROUP_ID), 0, lane); \
    hrx_q5_k_wmma_vk128_combined96_stage_load_store_slot(dst_rsrc, rows, row_base, col_base, rows, cols, sh_a, sh_b, (GROUP_ID), 1, lane); \
    hrx_q5_k_wmma_vk128_combined96_stage_load_store_slot(dst_rsrc, rows, row_base, col_base, rows, cols, sh_a, sh_b, (GROUP_ID), 2, lane); \
    hrx_q5_k_wmma_vk128_combined96_stage_load_store_slot(dst_rsrc, rows, row_base, col_base, rows, cols, sh_a, sh_b, (GROUP_ID), 3, lane); \
} while (0)
#endif

#define HRX_Q5_K_WMMA_VK128_COMBINED96_GROUPS_0_7(MACRO) do { \
    MACRO(0); MACRO(1); MACRO(2); MACRO(3); \
    MACRO(4); MACRO(5); MACRO(6); MACRO(7); \
} while (0)

#define HRX_Q5_K_WMMA_VK128_COMBINED96_GROUPS_8_15(MACRO) do { \
    MACRO(8);  MACRO(9);  MACRO(10); MACRO(11); \
    MACRO(12); MACRO(13); MACRO(14); MACRO(15); \
} while (0)

#define HRX_Q5_K_WMMA_VK128_COMBINED96_GROUPS_8_23(MACRO) do { \
    MACRO(8);  MACRO(9);  MACRO(10); MACRO(11); \
    MACRO(12); MACRO(13); MACRO(14); MACRO(15); \
    MACRO(16); MACRO(17); MACRO(18); MACRO(19); \
    MACRO(20); MACRO(21); MACRO(22); MACRO(23); \
} while (0)
#endif

static __device__ __forceinline__ void hrx_q5_k_wmma_vk128_store_acc_f16_row_major_w64_stage(
        float * dst,
        long long rows_stride,
        long long row0,
        long long col0,
        long long rows,
        long long cols,
        hrx_q5_k_wmma_vk128_half8_vec acc,
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
        sh_store[col_major_base + reg * 4] = acc[reg * 2 + HRX_Q5_K_WMMA_VK128_W64_OPSEL];
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

static __device__ __forceinline__ void hrx_q5_k_wmma_vk128_stage_acc_f16_row_major_w64(
        hrx_q5_k_wmma_vk128_half8_vec acc,
        unsigned int lane,
        unsigned int wave,
        int batch_slot,
        _Float16 * sh_store) {
    constexpr int TILE_STRIDE = 16 * 16;
    constexpr int BATCH_TILES = HRX_Q5_K_WMMA_VK128_STORE_STAGE_BATCH_TILES;
    const int row_lane = static_cast<int>(lane >> 4);
    const int col_lane = static_cast<int>(lane & 15u);
    const int tile_base = (static_cast<int>(wave) * BATCH_TILES + batch_slot) * TILE_STRIDE;
    const int col_major_base = tile_base + col_lane * 16 + row_lane;
#pragma unroll
    for (int reg = 0; reg < 4; ++reg) {
        sh_store[col_major_base + reg * 4] = acc[reg * 2 + HRX_Q5_K_WMMA_VK128_W64_OPSEL];
    }
}

static __device__ __forceinline__ void hrx_q5_k_wmma_vk128_store_staged_acc_f16_row_major_w64(
        float * dst,
        long long rows_stride,
        long long row0,
        long long col0,
        long long rows,
        long long cols,
        unsigned int lane,
        unsigned int wave,
        int batch_slot,
        _Float16 * sh_store) {
    constexpr int TILE_STRIDE = 16 * 16;
    constexpr int BATCH_TILES = HRX_Q5_K_WMMA_VK128_STORE_STAGE_BATCH_TILES;
    const int row_lane = static_cast<int>(lane >> 4);
    const int col_lane = static_cast<int>(lane & 15u);
    const int tile_base = (static_cast<int>(wave) * BATCH_TILES + batch_slot) * TILE_STRIDE;
    const int col_major_base = tile_base + col_lane * 16 + row_lane;
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
}

static __device__ __forceinline__ void hrx_q5_k_wmma_vk128_tile_map(
        int wave,
        int tile_iter,
        int * row_tile,
        int * col_tile) {
    constexpr int ROW_TILES = HRX_Q5_K_WMMA_VK128_BM / 16;
#if HRX_Q5_K_WMMA_VK128_W64
    constexpr int WAVE_COUNT = 4;
#else
    constexpr int WAVE_COUNT = 8;
#endif
    constexpr int TILE_COUNT = (HRX_Q5_K_WMMA_VK128_BM / 16) * (HRX_Q5_K_WMMA_VK128_BN / 16);
    constexpr int TILES_PER_WAVE = TILE_COUNT / WAVE_COUNT;
#if HRX_Q5_K_WMMA_VK128_PAIR64_TILE_MAP
    const int pair = wave >> 1;
    const int lane_wave = wave & 1;
    const int tile = lane_wave + tile_iter * 2;
    const int pair_row = pair & 1;
    const int pair_col = pair >> 1;
    *row_tile = pair_row * 4 + (tile & 3);
    *col_tile = pair_col * 4 + (tile >> 2);
#elif HRX_Q5_K_WMMA_VK128_W64 && HRX_Q5_K_WMMA_VK128_BM == 128 && HRX_Q5_K_WMMA_VK128_BN == 128
    const int wave_row = wave & 1;
    const int wave_col = wave >> 1;
    *row_tile = wave_row * 4 + (tile_iter & 3);
    *col_tile = wave_col * 4 + (tile_iter >> 2);
#elif HRX_Q5_K_WMMA_VK128_W64 && HRX_Q5_K_WMMA_VK128_BM == 64 && HRX_Q5_K_WMMA_VK128_BN == 64
    const int tile = wave * TILES_PER_WAVE + tile_iter;
    *row_tile = tile % ROW_TILES;
    *col_tile = tile / ROW_TILES;
#else
    const int tile = wave + tile_iter * WAVE_COUNT;
    *row_tile = tile % ROW_TILES;
    *col_tile = tile / ROW_TILES;
#endif
}

extern "C" __global__ __launch_bounds__(256, 1)
void HRX_Q5_K_WMMA_VK128_EXPORT(
        const hrx_block_q5_K_wmma_vk128_lhs * src0,
        const float * src1,
        float * dst,
        long long k,
        long long rows,
        long long cols) {
    constexpr int BM = HRX_Q5_K_WMMA_VK128_BM;
    constexpr int BN = HRX_Q5_K_WMMA_VK128_BN;
    constexpr int BK = 32;
    constexpr int SHARED_STRIDE = HRX_Q5_K_WMMA_VK128_SHARED_STRIDE;
#if HRX_Q5_K_WMMA_VK128_W64
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

#if HRX_Q5_K_WMMA_VK128_BUFFER_STORE
    const __amdgpu_buffer_rsrc_t dst_rsrc = hrx_q5_k_wmma_vk128_make_dst_rsrc(dst);
#endif

    __shared__ _Float16 sh_a[BM * SHARED_STRIDE];
    __shared__ _Float16 sh_b[BN * SHARED_STRIDE];
#if HRX_Q5_K_WMMA_VK128_STORE_STAGE
#if HRX_Q5_K_WMMA_VK128_STORE_STAGE_BATCH_TILES > 0
    __shared__ _Float16 sh_store[WAVE_COUNT * HRX_Q5_K_WMMA_VK128_STORE_STAGE_BATCH_TILES * 16 * 16];
#else
    __shared__ _Float16 sh_store[WAVE_COUNT * 16 * 16];
#endif
#elif HRX_Q5_K_WMMA_VK128_STORE_STAGE_FAST_HALF && !HRX_Q5_K_WMMA_VK128_W64_VK64_COMBINED96
    __shared__ _Float16 sh_store[WAVE_COUNT * 16 * 16];
#endif

    const long long blocks_per_row = k / 256;
    const _Float16 zero = static_cast<_Float16>(0.0f);
#if HRX_Q5_K_WMMA_VK128_W64
    constexpr int ACC_TILES = HRX_Q5_K_WMMA_VK128_W64_VK64_FULL64 ? 16 :
        (HRX_Q5_K_WMMA_VK128_W64_VK64_COMBINED96 ? 8 : TILES_PER_WAVE);
    hrx_q5_k_wmma_vk128_half8_vec acc[ACC_TILES] = {};
#else
    hrx_q5_k_wmma_vk128_half16_vec acc[TILES_PER_WAVE] = {};
#endif

    for (long long k0 = 0; k0 < k; k0 += BK) {
#if HRX_Q5_K_WMMA_VK128_PACK_STAGE_B32
        hrx_q5_k_wmma_vk128_lds_u32_ptr sh_a_u32 = (hrx_q5_k_wmma_vk128_lds_u32_ptr) sh_a;
        hrx_q5_k_wmma_vk128_lds_u32_ptr sh_b_u32 = (hrx_q5_k_wmma_vk128_lds_u32_ptr) sh_b;
        constexpr int PACKS_PER_K = BK / 2;
        constexpr int SHARED_STRIDE_PACKS = SHARED_STRIDE / 2;
        for (int idx = static_cast<int>(tid); idx < BM * PACKS_PER_K; idx += 256) {
            const int r = idx / PACKS_PER_K;
            const int kk_pair = idx - r * PACKS_PER_K;
            const int kk = kk_pair * 2;
            const long long row = row_base + static_cast<long long>(r);
            const _Float16 a0 = row < rows ?
                hrx_q5_k_wmma_vk128_load_a_value(src0, row, k0 + kk + 0, blocks_per_row) : zero;
            const _Float16 a1 = row < rows ?
                hrx_q5_k_wmma_vk128_load_a_value(src0, row, k0 + kk + 1, blocks_per_row) : zero;
            hrx_q5_k_wmma_vk128_ds_store_u32(
                sh_a_u32 + r * SHARED_STRIDE_PACKS + kk_pair,
                hrx_q5_k_wmma_vk128_f16_pair_to_u32(a0, a1));
        }
        for (int idx = static_cast<int>(tid); idx < BN * PACKS_PER_K; idx += 256) {
            const int c = idx / PACKS_PER_K;
            const int kk_pair = idx - c * PACKS_PER_K;
            const int kk = kk_pair * 2;
            const long long col = col_base + static_cast<long long>(c);
            const _Float16 b0 = col < cols ? static_cast<_Float16>(src1[col * k + k0 + kk + 0]) : zero;
            const _Float16 b1 = col < cols ? static_cast<_Float16>(src1[col * k + k0 + kk + 1]) : zero;
            hrx_q5_k_wmma_vk128_ds_store_u32(
                sh_b_u32 + c * SHARED_STRIDE_PACKS + kk_pair,
                hrx_q5_k_wmma_vk128_f16_pair_to_u32(b0, b1));
        }
#else
        for (int idx = static_cast<int>(tid); idx < BM * BK; idx += 256) {
            const int r = idx / BK;
            const int kk = idx - r * BK;
            const long long row = row_base + static_cast<long long>(r);
            sh_a[r * SHARED_STRIDE + kk] = row < rows ?
                hrx_q5_k_wmma_vk128_load_a_value(src0, row, k0 + kk, blocks_per_row) : zero;
        }
        for (int idx = static_cast<int>(tid); idx < BN * BK; idx += 256) {
            const int c = idx / BK;
            const int kk = idx - c * BK;
            const long long col = col_base + static_cast<long long>(c);
            sh_b[c * SHARED_STRIDE + kk] = col < cols ? static_cast<_Float16>(src1[col * k + k0 + kk]) : zero;
        }
#endif
        __syncthreads();

#if HRX_Q5_K_WMMA_VK128_W64_VK64_FULL64
        if (wave == 0) {
            hrx_q5_k_wmma_vk128_lds_half_ptr sh_a_lds =
                (hrx_q5_k_wmma_vk128_lds_half_ptr) sh_a;
            hrx_q5_k_wmma_vk128_lds_half_ptr sh_b_lds =
                (hrx_q5_k_wmma_vk128_lds_half_ptr) sh_b;
            hrx_q5_k_wmma_vk128_half16_vec a_frag[2][4];
            hrx_q5_k_wmma_vk128_half16_vec b_frag[2][4];
#pragma unroll
            for (int k_tile = 0; k_tile < 2; ++k_tile) {
#pragma unroll
                for (int row_sub = 0; row_sub < 4; ++row_sub) {
                    a_frag[k_tile][row_sub] =
                        hrx_q5_k_wmma_vk128_load_a_frag_w64_b64asm_nowait(
                            sh_a_lds, row_sub, k_tile, lane);
                }
#pragma unroll
                for (int col_sub = 0; col_sub < 4; ++col_sub) {
                    b_frag[k_tile][col_sub] =
                        hrx_q5_k_wmma_vk128_load_b_frag_w64_b64asm_nowait(
                            sh_b_lds, col_sub, k_tile, lane);
                }
            }
            asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
#pragma unroll
            for (int k_tile = 0; k_tile < 2; ++k_tile) {
#pragma unroll
                for (int col_sub = 0; col_sub < 4; ++col_sub) {
#pragma unroll
                    for (int row_sub = 0; row_sub < 4; ++row_sub) {
                        const int tile = col_sub * 4 + row_sub;
                        acc[tile] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(
                            a_frag[k_tile][row_sub],
                            b_frag[k_tile][col_sub],
                            acc[tile],
                            HRX_Q5_K_WMMA_VK128_W64_OPSEL != 0);
                    }
                }
            }
        }
#elif HRX_Q5_K_WMMA_VK128_W64_VK64_COMBINED96
        if (wave == 0) {
            hrx_q5_k_wmma_vk128_lds_half_ptr sh_a_lds =
                (hrx_q5_k_wmma_vk128_lds_half_ptr) sh_a;
            hrx_q5_k_wmma_vk128_lds_half_ptr sh_b_lds =
                (hrx_q5_k_wmma_vk128_lds_half_ptr) sh_b;
            const hrx_q5_k_wmma_vk128_half16_vec a0 =
                hrx_q5_k_wmma_vk128_load_a_frag_w64_b64asm_nowait(sh_a_lds, 0, 0, lane);
            const hrx_q5_k_wmma_vk128_half16_vec a1 =
                hrx_q5_k_wmma_vk128_load_a_frag_w64_b64asm_nowait(sh_a_lds, 1, 0, lane);
            const hrx_q5_k_wmma_vk128_half16_vec a2 =
                hrx_q5_k_wmma_vk128_load_a_frag_w64_b64asm_nowait(sh_a_lds, 2, 0, lane);
            const hrx_q5_k_wmma_vk128_half16_vec a3 =
                hrx_q5_k_wmma_vk128_load_a_frag_w64_b64asm_nowait(sh_a_lds, 3, 0, lane);
            const hrx_q5_k_wmma_vk128_half16_vec b0 =
                hrx_q5_k_wmma_vk128_load_b_frag_w64_b64asm_nowait(sh_b_lds, 0, 0, lane);
            const hrx_q5_k_wmma_vk128_half16_vec b1 =
                hrx_q5_k_wmma_vk128_load_b_frag_w64_b64asm_nowait(sh_b_lds, 1, 0, lane);
            const hrx_q5_k_wmma_vk128_half16_vec a4 =
                hrx_q5_k_wmma_vk128_load_a_frag_w64_b64asm_nowait(sh_a_lds, 0, 1, lane);
            const hrx_q5_k_wmma_vk128_half16_vec a5 =
                hrx_q5_k_wmma_vk128_load_a_frag_w64_b64asm_nowait(sh_a_lds, 1, 1, lane);
            const hrx_q5_k_wmma_vk128_half16_vec a6 =
                hrx_q5_k_wmma_vk128_load_a_frag_w64_b64asm_nowait(sh_a_lds, 2, 1, lane);
            const hrx_q5_k_wmma_vk128_half16_vec a7 =
                hrx_q5_k_wmma_vk128_load_a_frag_w64_b64asm_nowait(sh_a_lds, 3, 1, lane);
            const hrx_q5_k_wmma_vk128_half16_vec b2 =
                hrx_q5_k_wmma_vk128_load_b_frag_w64_b64asm_nowait(sh_b_lds, 0, 1, lane);
            const hrx_q5_k_wmma_vk128_half16_vec b3 =
                hrx_q5_k_wmma_vk128_load_b_frag_w64_b64asm_nowait(sh_b_lds, 1, 1, lane);

            asm volatile("s_waitcnt lgkmcnt(40)\n" ::: "memory");
            acc[0] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(a0, b0, acc[0], false);
            acc[1] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(a1, b0, acc[1], false);
            acc[2] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(a2, b0, acc[2], false);
            acc[3] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(a3, b0, acc[3], false);
            acc[4] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(a0, b1, acc[4], false);
            acc[5] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(a1, b1, acc[5], false);
            acc[6] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(a2, b1, acc[6], false);
            acc[7] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(a3, b1, acc[7], false);
            acc[0] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(a4, b2, acc[0], false);
            acc[1] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(a5, b2, acc[1], false);
            acc[2] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(a6, b2, acc[2], false);
            acc[3] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(a7, b2, acc[3], false);
            acc[4] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(a4, b3, acc[4], false);
            acc[5] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(a5, b3, acc[5], false);
            acc[6] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(a6, b3, acc[6], false);
            acc[7] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(a7, b3, acc[7], false);
        }
#elif HRX_Q5_K_WMMA_VK128_W64_VK64_GROUPK2 || HRX_Q5_K_WMMA_VK128_W64_VK64_GROUPK2_WAIT
        {
            hrx_q5_k_wmma_vk128_lds_half_ptr sh_a_lds =
                (hrx_q5_k_wmma_vk128_lds_half_ptr) sh_a;
            hrx_q5_k_wmma_vk128_lds_half_ptr sh_b_lds =
                (hrx_q5_k_wmma_vk128_lds_half_ptr) sh_b;
            const int col_tile = static_cast<int>(wave);
            hrx_q5_k_wmma_vk128_half16_vec a_frag[2][4];
            hrx_q5_k_wmma_vk128_half16_vec b_frag[2];
#pragma unroll
            for (int k_tile = 0; k_tile < 2; ++k_tile) {
#pragma unroll
                for (int row_sub = 0; row_sub < 4; ++row_sub) {
#if HRX_Q5_K_WMMA_VK128_W64_VK64_GROUPK2_WAIT
                    a_frag[k_tile][row_sub] = hrx_q5_k_wmma_vk128_load_a_frag_w64_b64asm(
                        sh_a_lds, row_sub, k_tile, lane);
#else
                    a_frag[k_tile][row_sub] = hrx_q5_k_wmma_vk128_load_a_frag_w64_b64asm_nowait(
                        sh_a_lds, row_sub, k_tile, lane);
#endif
                }
#if HRX_Q5_K_WMMA_VK128_W64_VK64_GROUPK2_WAIT
                b_frag[k_tile] = hrx_q5_k_wmma_vk128_load_b_frag_w64_b64asm(
                    sh_b_lds, col_tile, k_tile, lane);
#else
                b_frag[k_tile] = hrx_q5_k_wmma_vk128_load_b_frag_w64_b64asm_nowait(
                    sh_b_lds, col_tile, k_tile, lane);
#endif
            }
            asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
#pragma unroll
            for (int k_tile = 0; k_tile < 2; ++k_tile) {
#pragma unroll
                for (int row_sub = 0; row_sub < 4; ++row_sub) {
                    acc[row_sub] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(
                        a_frag[k_tile][row_sub],
                        b_frag[k_tile],
                        acc[row_sub],
                        HRX_Q5_K_WMMA_VK128_W64_OPSEL != 0);
                }
            }
        }
#elif HRX_Q5_K_WMMA_VK128_W64_B64GROUP
        {
            hrx_q5_k_wmma_vk128_lds_half_ptr sh_a_lds =
                (hrx_q5_k_wmma_vk128_lds_half_ptr) sh_a;
            hrx_q5_k_wmma_vk128_lds_half_ptr sh_b_lds =
                (hrx_q5_k_wmma_vk128_lds_half_ptr) sh_b;
            const int wave_row = static_cast<int>(wave & 1u);
            const int wave_col = static_cast<int>(wave >> 1);
#pragma unroll
            for (int k_tile = 0; k_tile < 2; ++k_tile) {
                hrx_q5_k_wmma_vk128_half16_vec a_frag[4];
                hrx_q5_k_wmma_vk128_half16_vec b_frag[4];
#if HRX_Q5_K_WMMA_VK128_W64_B64GROUP_RADVLADDER
                a_frag[0] = hrx_q5_k_wmma_vk128_load_a_frag_w64_b64asm_nowait(
                    sh_a_lds, wave_row * 4 + 0, k_tile, lane);
#pragma unroll
                for (int col_sub = 0; col_sub < 4; ++col_sub) {
                    b_frag[col_sub] = hrx_q5_k_wmma_vk128_load_b_frag_w64_b64asm_nowait(
                        sh_b_lds, wave_col * 4 + col_sub, k_tile, lane);
                }
#pragma unroll
                for (int row_sub = 1; row_sub < 4; ++row_sub) {
                    a_frag[row_sub] = hrx_q5_k_wmma_vk128_load_a_frag_w64_b64asm_nowait(
                        sh_a_lds, wave_row * 4 + row_sub, k_tile, lane);
                }
                HRX_Q5_K_WMMA_VK128_WAIT_WMMA(0, a_frag[0], b_frag[0], 12);
                HRX_Q5_K_WMMA_VK128_WAIT_WMMA(4, a_frag[0], b_frag[1], 12);
                HRX_Q5_K_WMMA_VK128_WAIT_WMMA(8, a_frag[0], b_frag[2], 12);
                HRX_Q5_K_WMMA_VK128_WAIT_WMMA(12, a_frag[0], b_frag[3], 12);
                HRX_Q5_K_WMMA_VK128_WAIT_WMMA(1, a_frag[1], b_frag[0], 8);
                HRX_Q5_K_WMMA_VK128_WAIT_WMMA(5, a_frag[1], b_frag[1], 8);
                HRX_Q5_K_WMMA_VK128_WAIT_WMMA(9, a_frag[1], b_frag[2], 8);
                HRX_Q5_K_WMMA_VK128_WAIT_WMMA(13, a_frag[1], b_frag[3], 8);
                HRX_Q5_K_WMMA_VK128_WAIT_WMMA(2, a_frag[2], b_frag[0], 4);
                HRX_Q5_K_WMMA_VK128_WAIT_WMMA(6, a_frag[2], b_frag[1], 4);
                HRX_Q5_K_WMMA_VK128_WAIT_WMMA(10, a_frag[2], b_frag[2], 4);
                HRX_Q5_K_WMMA_VK128_WAIT_WMMA(14, a_frag[2], b_frag[3], 4);
                HRX_Q5_K_WMMA_VK128_WAIT_WMMA(3, a_frag[3], b_frag[0], 0);
                HRX_Q5_K_WMMA_VK128_WAIT_WMMA(7, a_frag[3], b_frag[1], 0);
                HRX_Q5_K_WMMA_VK128_WAIT_WMMA(11, a_frag[3], b_frag[2], 0);
                HRX_Q5_K_WMMA_VK128_WAIT_WMMA(15, a_frag[3], b_frag[3], 0);
#else
#pragma unroll
                for (int row_sub = 0; row_sub < 4; ++row_sub) {
                    a_frag[row_sub] = hrx_q5_k_wmma_vk128_load_a_frag_w64_b64asm(
                        sh_a_lds, wave_row * 4 + row_sub, k_tile, lane);
                }
#pragma unroll
                for (int col_sub = 0; col_sub < 4; ++col_sub) {
                    b_frag[col_sub] = hrx_q5_k_wmma_vk128_load_b_frag_w64_b64asm(
                        sh_b_lds, wave_col * 4 + col_sub, k_tile, lane);
                }
                asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
#pragma unroll
                for (int col_sub = 0; col_sub < 4; ++col_sub) {
#pragma unroll
                    for (int row_sub = 0; row_sub < 4; ++row_sub) {
                        const int tile_iter = col_sub * 4 + row_sub;
                        acc[tile_iter] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(
                            a_frag[row_sub],
                            b_frag[col_sub],
                            acc[tile_iter],
                            HRX_Q5_K_WMMA_VK128_W64_OPSEL != 0);
                    }
                }
#endif
            }
        }
#else
#pragma unroll
        for (int tile_iter = 0; tile_iter < TILES_PER_WAVE; ++tile_iter) {
            int row_tile = 0;
            int col_tile = 0;
            hrx_q5_k_wmma_vk128_tile_map(static_cast<int>(wave), tile_iter, &row_tile, &col_tile);
#if HRX_Q5_K_WMMA_VK128_W64
#pragma unroll
            for (int k_tile = 0; k_tile < 2; ++k_tile) {
#if HRX_Q5_K_WMMA_VK128_W64_B64ASM
                hrx_q5_k_wmma_vk128_lds_half_ptr sh_a_lds =
                    (hrx_q5_k_wmma_vk128_lds_half_ptr) sh_a;
                hrx_q5_k_wmma_vk128_lds_half_ptr sh_b_lds =
                    (hrx_q5_k_wmma_vk128_lds_half_ptr) sh_b;
                const hrx_q5_k_wmma_vk128_half16_vec a =
                    hrx_q5_k_wmma_vk128_load_a_frag_w64_b64asm(sh_a_lds, row_tile, k_tile, lane);
                const hrx_q5_k_wmma_vk128_half16_vec b =
                    hrx_q5_k_wmma_vk128_load_b_frag_w64_b64asm(sh_b_lds, col_tile, k_tile, lane);
#elif HRX_Q5_K_WMMA_VK128_W64_H4LOAD
                const hrx_q5_k_wmma_vk128_half16_vec a =
                    hrx_q5_k_wmma_vk128_load_a_frag_w64_h4(sh_a, row_tile, k_tile, lane);
                const hrx_q5_k_wmma_vk128_half16_vec b =
                    hrx_q5_k_wmma_vk128_load_b_frag_w64_h4(sh_b, col_tile, k_tile, lane);
#else
                const hrx_q5_k_wmma_vk128_half16_vec a =
                    hrx_q5_k_wmma_vk128_load_a_frag_w64(sh_a, row_tile, k_tile, lane);
                const hrx_q5_k_wmma_vk128_half16_vec b =
                    hrx_q5_k_wmma_vk128_load_b_frag_w64(sh_b, col_tile, k_tile, lane);
#endif
                acc[tile_iter] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(
                    a, b, acc[tile_iter], HRX_Q5_K_WMMA_VK128_W64_OPSEL != 0);
            }
#else
#if HRX_Q5_K_WMMA_VK128_PREFETCH_FRAGS
            hrx_q5_k_wmma_vk128_half16_vec a_frag[2];
            hrx_q5_k_wmma_vk128_half16_vec b_frag[2];
#pragma unroll
            for (int k_tile = 0; k_tile < 2; ++k_tile) {
                a_frag[k_tile] = hrx_q5_k_wmma_vk128_load_a_frag(sh_a, row_tile, k_tile, lane);
                b_frag[k_tile] = hrx_q5_k_wmma_vk128_load_b_frag(sh_b, col_tile, k_tile, lane);
            }
#pragma unroll
            for (int k_tile = 0; k_tile < 2; ++k_tile) {
                acc[tile_iter] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(
                    a_frag[k_tile], b_frag[k_tile], acc[tile_iter], false);
            }
#else
#pragma unroll
            for (int k_tile = 0; k_tile < 2; ++k_tile) {
                const hrx_q5_k_wmma_vk128_half16_vec a =
                    hrx_q5_k_wmma_vk128_load_a_frag(sh_a, row_tile, k_tile, lane);
                const hrx_q5_k_wmma_vk128_half16_vec b =
                    hrx_q5_k_wmma_vk128_load_b_frag(sh_b, col_tile, k_tile, lane);
                acc[tile_iter] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(a, b, acc[tile_iter], false);
            }
#endif
#endif
        }
#endif

        __syncthreads();
    }

#if HRX_Q5_K_WMMA_VK128_W64_VK64_FULL64
    if (wave == 0) {
        HRX_Q5_K_WMMA_VK128_COMBINED96_GROUPS_0_7(HRX_Q5_K_WMMA_VK128_COMBINED96_RAW_STORE_GROUP);
        HRX_Q5_K_WMMA_VK128_COMBINED96_GROUPS_8_15(HRX_Q5_K_WMMA_VK128_COMBINED96_RAW_STORE_GROUP);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();
#elif HRX_Q5_K_WMMA_VK128_W64_VK64_COMBINED96
    if (wave == 0) {
        HRX_Q5_K_WMMA_VK128_COMBINED96_GROUPS_0_7(HRX_Q5_K_WMMA_VK128_COMBINED96_RAW_STORE_GROUP);
        HRX_Q5_K_WMMA_VK128_COMBINED96_GROUPS_8_23(HRX_Q5_K_WMMA_VK128_COMBINED96_STAGE_STORE_GROUP);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();
    if (wave == 0) {
        HRX_Q5_K_WMMA_VK128_COMBINED96_GROUPS_8_23(HRX_Q5_K_WMMA_VK128_COMBINED96_STAGE_LOAD_STORE_GROUP);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();
#elif HRX_Q5_K_WMMA_VK128_W64 && HRX_Q5_K_WMMA_VK128_STORE_STAGE && HRX_Q5_K_WMMA_VK128_STORE_STAGE_BATCH_TILES > 0
    constexpr int STORE_BATCH_TILES = HRX_Q5_K_WMMA_VK128_STORE_STAGE_BATCH_TILES;
#pragma unroll
    for (int batch_start = 0; batch_start < TILES_PER_WAVE; batch_start += STORE_BATCH_TILES) {
#pragma unroll
        for (int batch_slot = 0; batch_slot < STORE_BATCH_TILES; ++batch_slot) {
            const int tile_iter = batch_start + batch_slot;
            if (tile_iter < TILES_PER_WAVE) {
                hrx_q5_k_wmma_vk128_stage_acc_f16_row_major_w64(
                    acc[tile_iter],
                    lane,
                    wave,
                    batch_slot,
                    sh_store);
            }
        }
        __syncthreads();
#pragma unroll
        for (int batch_slot = 0; batch_slot < STORE_BATCH_TILES; ++batch_slot) {
            const int tile_iter = batch_start + batch_slot;
            if (tile_iter < TILES_PER_WAVE) {
                int row_tile = 0;
                int col_tile = 0;
                hrx_q5_k_wmma_vk128_tile_map(static_cast<int>(wave), tile_iter, &row_tile, &col_tile);
                hrx_q5_k_wmma_vk128_store_staged_acc_f16_row_major_w64(
                    dst,
                    rows,
                    row_base + static_cast<long long>(row_tile * 16),
                    col_base + static_cast<long long>(col_tile * 16),
                    rows,
                    cols,
                    lane,
                    wave,
                    batch_slot,
                    sh_store);
            }
        }
        __syncthreads();
    }
#elif HRX_Q5_K_WMMA_VK128_W64 && HRX_Q5_K_WMMA_VK128_MOTIF192_FULL_WRITEBACK
    HRX_Q5_K_WMMA_VK128_MOTIF192_GROUPS_0_15();
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();
#else
#pragma unroll
    for (int tile_iter = 0; tile_iter < TILES_PER_WAVE; ++tile_iter) {
        int row_tile = 0;
        int col_tile = 0;
        hrx_q5_k_wmma_vk128_tile_map(static_cast<int>(wave), tile_iter, &row_tile, &col_tile);
#if HRX_Q5_K_WMMA_VK128_W64
#if HRX_Q5_K_WMMA_VK128_STORE_STAGE
        hrx_q5_k_wmma_vk128_store_acc_f16_row_major_w64_stage(
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
#if HRX_Q5_K_WMMA_VK128_FULL_TILE_STORE
        const long long tile_row0 = row_base + static_cast<long long>(row_tile * 16);
        const long long tile_col0 = col_base + static_cast<long long>(col_tile * 16);
        if (tile_row0 + 16 <= rows && tile_col0 + 16 <= cols) {
#if HRX_Q5_K_WMMA_VK128_BUFFER_STORE
#if HRX_Q5_K_WMMA_VK128_STORE_STAGE_FAST_HALF && HRX_Q5_K_WMMA_VK128_STORE_STAGE_FAST_HALF_SELECTED
            hrx_q5_k_wmma_vk128_store_acc_f16_row_major_w64_fast_half_buffer_selected(
                dst_rsrc,
                rows,
                tile_row0,
                tile_col0,
                acc[tile_iter],
                lane,
                wave,
                (hrx_q5_k_wmma_vk128_lds_volatile_half_ptr) sh_store);
#else
            hrx_q5_k_wmma_vk128_store_acc_f16_row_major_w64_buffer_full(
                dst_rsrc,
                rows,
                tile_row0,
                tile_col0,
                acc[tile_iter],
                lane);
#endif
#else
            hrx_q5_k_wmma_vk128_store_acc_f16_row_major_w64_full(
                dst,
                rows,
                tile_row0,
                tile_col0,
                acc[tile_iter],
                lane);
#endif
        } else {
#if HRX_Q5_K_WMMA_VK128_BUFFER_STORE
            hrx_q5_k_wmma_vk128_store_acc_f16_row_major_w64_buffer(
                dst_rsrc,
                rows,
                tile_row0,
                tile_col0,
                rows,
                cols,
                acc[tile_iter],
                lane);
#else
            hrx_q5_k_wmma_vk128_store_acc_f16_row_major_w64(
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
#else
        hrx_q5_k_wmma_vk128_store_acc_f16_row_major_w64(
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
        hrx_q5_k_wmma_vk128_store_acc_f16_row_major(
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
