#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <stdint.h>

#ifndef HRX_Q6_K_WMMA_VK128_EXPORT
#define HRX_Q6_K_WMMA_VK128_EXPORT hrx_mul_mat_vec_q6_k_wmma16x16_vk128_f16acc_wg256_f32
#endif

#ifndef HRX_Q6_K_WMMA_VK128_SHARED_STRIDE
#define HRX_Q6_K_WMMA_VK128_SHARED_STRIDE 32
#endif

#ifndef HRX_Q6_K_WMMA_VK128_BM
#define HRX_Q6_K_WMMA_VK128_BM 128
#endif

#ifndef HRX_Q6_K_WMMA_VK128_BN
#define HRX_Q6_K_WMMA_VK128_BN 128
#endif

#ifndef HRX_Q6_K_WMMA_VK128_WG_SIZE
#define HRX_Q6_K_WMMA_VK128_WG_SIZE 256
#endif

#ifndef HRX_Q6_K_WMMA_VK128_PREFETCH_FRAGS
#define HRX_Q6_K_WMMA_VK128_PREFETCH_FRAGS 0
#endif

#ifndef HRX_Q6_K_WMMA_VK128_PAIR64_TILE_MAP
#define HRX_Q6_K_WMMA_VK128_PAIR64_TILE_MAP 0
#endif

#ifndef HRX_Q6_K_WMMA_VK128_W64
#define HRX_Q6_K_WMMA_VK128_W64 0
#endif

#ifndef HRX_Q6_K_WMMA_VK128_W64_OPSEL
#define HRX_Q6_K_WMMA_VK128_W64_OPSEL 0
#endif

#ifndef HRX_Q6_K_WMMA_VK128_W64_H4LOAD
#define HRX_Q6_K_WMMA_VK128_W64_H4LOAD 0
#endif

#ifndef HRX_Q6_K_WMMA_VK128_W64_B64ASM
#define HRX_Q6_K_WMMA_VK128_W64_B64ASM 0
#endif

#ifndef HRX_Q6_K_WMMA_VK128_W64_B64GROUP
#define HRX_Q6_K_WMMA_VK128_W64_B64GROUP 0
#endif

#ifndef HRX_Q6_K_WMMA_VK128_W64_VK64_GROUPK2
#define HRX_Q6_K_WMMA_VK128_W64_VK64_GROUPK2 0
#endif

#ifndef HRX_Q6_K_WMMA_VK128_W64_VK64_GROUPK2_WAIT
#define HRX_Q6_K_WMMA_VK128_W64_VK64_GROUPK2_WAIT 0
#endif

#ifndef HRX_Q6_K_WMMA_VK128_W64_VK64_RING96
#define HRX_Q6_K_WMMA_VK128_W64_VK64_RING96 0
#endif

#ifndef HRX_Q6_K_WMMA_VK128_FULL_TILE_STORE
#define HRX_Q6_K_WMMA_VK128_FULL_TILE_STORE 0
#endif

#ifndef HRX_Q6_K_WMMA_VK128_STORE_STAGE
#define HRX_Q6_K_WMMA_VK128_STORE_STAGE 0
#endif

#ifndef HRX_Q6_K_WMMA_VK128_STORE_STAGE_FAST_HALF
#define HRX_Q6_K_WMMA_VK128_STORE_STAGE_FAST_HALF 0
#endif

#ifndef HRX_Q6_K_WMMA_VK128_BUFFER_STORE
#define HRX_Q6_K_WMMA_VK128_BUFFER_STORE 0
#endif

struct hrx_block_q6_K_wmma_vk128_lhs {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    unsigned short d;
};

typedef _Float16 hrx_q6_k_wmma_vk128_half16_vec __attribute__((ext_vector_type(16)));
typedef _Float16 hrx_q6_k_wmma_vk128_half8_vec __attribute__((ext_vector_type(8)));
typedef _Float16 hrx_q6_k_wmma_vk128_half4_vec __attribute__((ext_vector_type(4)));
typedef const __attribute__((address_space(3))) _Float16 * hrx_q6_k_wmma_vk128_lds_half_ptr;
typedef volatile __attribute__((address_space(3))) _Float16 * hrx_q6_k_wmma_vk128_lds_volatile_half_ptr;
typedef __attribute__((address_space(3))) uint16_t * hrx_q6_k_wmma_vk128_lds_u16_ptr;
typedef const __attribute__((address_space(3))) uint16_t * hrx_q6_k_wmma_vk128_lds_const_u16_ptr;

#if HRX_Q6_K_WMMA_VK128_BUFFER_STORE
static constexpr int HRX_Q6_K_WMMA_VK128_RAW_BUFFER_FLAGS_GFX11 = 0x31004000;

static __device__ __forceinline__ __amdgpu_buffer_rsrc_t hrx_q6_k_wmma_vk128_make_dst_rsrc(float * dst) {
    return __builtin_amdgcn_make_buffer_rsrc(
        dst,
        static_cast<unsigned short>(0),
        0xffffffffull,
        HRX_Q6_K_WMMA_VK128_RAW_BUFFER_FLAGS_GFX11);
}

static __device__ __forceinline__ void hrx_q6_k_wmma_vk128_buffer_store_f32(
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

static __device__ __forceinline__ uint32_t hrx_q6_k_wmma_vk128_pack_f16x2(_Float16 lo, _Float16 hi) {
    union {
        _Float16 h[2];
        uint32_t u;
    } pack;
    pack.h[0] = lo;
    pack.h[1] = hi;
    return pack.u;
}

static __device__ __forceinline__ _Float16 hrx_q6_k_wmma_vk128_unpack_f16x2(uint32_t bits, int idx) {
    union {
        uint32_t u;
        _Float16 h[2];
    } pack;
    pack.u = bits;
    return pack.h[idx];
}

static __device__ __forceinline__ uint16_t hrx_q6_k_wmma_vk128_f16_to_u16(_Float16 value) {
    union {
        _Float16 h;
        uint16_t u;
    } pack;
    pack.h = value;
    return pack.u;
}

static __device__ __forceinline__ _Float16 hrx_q6_k_wmma_vk128_u16_to_f16(uint32_t value) {
    union {
        uint16_t u;
        _Float16 h;
    } pack;
    pack.u = static_cast<uint16_t>(value);
    return pack.h;
}

static __device__ __forceinline__ hrx_q6_k_wmma_vk128_half16_vec hrx_q6_k_wmma_vk128_duplicate_input(
        _Float16 x0, _Float16 x1, _Float16 x2, _Float16 x3,
        _Float16 x4, _Float16 x5, _Float16 x6, _Float16 x7) {
    constexpr int SWAP16_CTRL = (16 << 10) | 0x1f;
    const uint32_t p0 = hrx_q6_k_wmma_vk128_pack_f16x2(x0, x1);
    const uint32_t p1 = hrx_q6_k_wmma_vk128_pack_f16x2(x2, x3);
    const uint32_t p2 = hrx_q6_k_wmma_vk128_pack_f16x2(x4, x5);
    const uint32_t p3 = hrx_q6_k_wmma_vk128_pack_f16x2(x6, x7);
    const uint32_t s0 = static_cast<uint32_t>(__builtin_amdgcn_ds_swizzle(static_cast<int32_t>(p0), SWAP16_CTRL));
    const uint32_t s1 = static_cast<uint32_t>(__builtin_amdgcn_ds_swizzle(static_cast<int32_t>(p1), SWAP16_CTRL));
    const uint32_t s2 = static_cast<uint32_t>(__builtin_amdgcn_ds_swizzle(static_cast<int32_t>(p2), SWAP16_CTRL));
    const uint32_t s3 = static_cast<uint32_t>(__builtin_amdgcn_ds_swizzle(static_cast<int32_t>(p3), SWAP16_CTRL));

    hrx_q6_k_wmma_vk128_half16_vec result;
    result[0] = x0;
    result[1] = x1;
    result[2] = x2;
    result[3] = x3;
    result[4] = x4;
    result[5] = x5;
    result[6] = x6;
    result[7] = x7;
    result[8] = hrx_q6_k_wmma_vk128_unpack_f16x2(s0, 0);
    result[9] = hrx_q6_k_wmma_vk128_unpack_f16x2(s0, 1);
    result[10] = hrx_q6_k_wmma_vk128_unpack_f16x2(s1, 0);
    result[11] = hrx_q6_k_wmma_vk128_unpack_f16x2(s1, 1);
    result[12] = hrx_q6_k_wmma_vk128_unpack_f16x2(s2, 0);
    result[13] = hrx_q6_k_wmma_vk128_unpack_f16x2(s2, 1);
    result[14] = hrx_q6_k_wmma_vk128_unpack_f16x2(s3, 0);
    result[15] = hrx_q6_k_wmma_vk128_unpack_f16x2(s3, 1);
    return result;
}

static __device__ __forceinline__ int hrx_q6_k_wmma_vk128_value(
        const hrx_block_q6_K_wmma_vk128_lhs * block,
        int in_block) {
    const int half = in_block / 128;
    const int idx = in_block - half * 128;
    const int lane = idx & 31;
    const int ql_base = half * 64;
    const int qh_base = half * 32;

    int q = 0;
    if (idx < 32) {
        q = (block->ql[ql_base + lane] & 0x0F) | (((block->qh[qh_base + lane] >> 0) & 3) << 4);
    } else if (idx < 64) {
        q = (block->ql[ql_base + lane + 32] & 0x0F) | (((block->qh[qh_base + lane] >> 2) & 3) << 4);
    } else if (idx < 96) {
        q = (block->ql[ql_base + lane] >> 4) | (((block->qh[qh_base + lane] >> 4) & 3) << 4);
    } else {
        q = (block->ql[ql_base + lane + 32] >> 4) | (((block->qh[qh_base + lane] >> 6) & 3) << 4);
    }
    return q - 32;
}

static __device__ __forceinline__ int hrx_q6_k_wmma_vk128_scale(
        const hrx_block_q6_K_wmma_vk128_lhs * block,
        int group,
        int lane) {
    const int half = group >> 2;
    const int group_in_half = group & 3;
    return static_cast<int>(block->scales[half * 8 + group_in_half * 2 + lane / 16]);
}

static __device__ __forceinline__ _Float16 hrx_q6_k_wmma_vk128_load_a_value(
        const hrx_block_q6_K_wmma_vk128_lhs * src0,
        long long row,
        long long k_index,
        long long blocks_per_row) {
    const hrx_block_q6_K_wmma_vk128_lhs * block = src0 + row * blocks_per_row + (k_index >> 8);
    const int in_block = static_cast<int>(k_index & 255);
    const int group = in_block >> 5;
    const int lane = in_block & 31;
    const float d =
        __half2float(__ushort_as_half(block->d)) *
        static_cast<float>(hrx_q6_k_wmma_vk128_scale(block, group, lane));
    return static_cast<_Float16>(d * static_cast<float>(hrx_q6_k_wmma_vk128_value(block, in_block)));
}

static __device__ __forceinline__ hrx_q6_k_wmma_vk128_half16_vec hrx_q6_k_wmma_vk128_load_a_frag(
        const _Float16 * sh_a,
        int row_tile,
        int k_tile,
        unsigned int lane) {
    constexpr int SHARED_STRIDE = HRX_Q6_K_WMMA_VK128_SHARED_STRIDE;
    const int row = row_tile * 16 + static_cast<int>(lane & 15u);
    const int k_base = k_tile * 16 + static_cast<int>(lane >> 4) * 8;
    return hrx_q6_k_wmma_vk128_duplicate_input(
        sh_a[row * SHARED_STRIDE + k_base + 0],
        sh_a[row * SHARED_STRIDE + k_base + 1],
        sh_a[row * SHARED_STRIDE + k_base + 2],
        sh_a[row * SHARED_STRIDE + k_base + 3],
        sh_a[row * SHARED_STRIDE + k_base + 4],
        sh_a[row * SHARED_STRIDE + k_base + 5],
        sh_a[row * SHARED_STRIDE + k_base + 6],
        sh_a[row * SHARED_STRIDE + k_base + 7]);
}

static __device__ __forceinline__ hrx_q6_k_wmma_vk128_half16_vec hrx_q6_k_wmma_vk128_load_b_frag(
        const _Float16 * sh_b,
        int col_tile,
        int k_tile,
        unsigned int lane) {
    constexpr int SHARED_STRIDE = HRX_Q6_K_WMMA_VK128_SHARED_STRIDE;
    const int col = col_tile * 16 + static_cast<int>(lane & 15u);
    const int k_base = k_tile * 16 + static_cast<int>(lane >> 4) * 8;
    return hrx_q6_k_wmma_vk128_duplicate_input(
        sh_b[col * SHARED_STRIDE + k_base + 0],
        sh_b[col * SHARED_STRIDE + k_base + 1],
        sh_b[col * SHARED_STRIDE + k_base + 2],
        sh_b[col * SHARED_STRIDE + k_base + 3],
        sh_b[col * SHARED_STRIDE + k_base + 4],
        sh_b[col * SHARED_STRIDE + k_base + 5],
        sh_b[col * SHARED_STRIDE + k_base + 6],
        sh_b[col * SHARED_STRIDE + k_base + 7]);
}

static __device__ __forceinline__ hrx_q6_k_wmma_vk128_half16_vec hrx_q6_k_wmma_vk128_load_a_frag_w64(
        const _Float16 * sh_a,
        int row_tile,
        int k_tile,
        unsigned int lane) {
    constexpr int SHARED_STRIDE = HRX_Q6_K_WMMA_VK128_SHARED_STRIDE;
    const int row = row_tile * 16 + static_cast<int>(lane & 15u);
    const int k_base = k_tile * 16;
    hrx_q6_k_wmma_vk128_half16_vec result;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        result[i] = sh_a[row * SHARED_STRIDE + k_base + i];
    }
    return result;
}

static __device__ __forceinline__ hrx_q6_k_wmma_vk128_half16_vec hrx_q6_k_wmma_vk128_load_b_frag_w64(
        const _Float16 * sh_b,
        int col_tile,
        int k_tile,
        unsigned int lane) {
    constexpr int SHARED_STRIDE = HRX_Q6_K_WMMA_VK128_SHARED_STRIDE;
    const int col = col_tile * 16 + static_cast<int>(lane & 15u);
    const int k_base = k_tile * 16;
    hrx_q6_k_wmma_vk128_half16_vec result;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        result[i] = sh_b[col * SHARED_STRIDE + k_base + i];
    }
    return result;
}

static __device__ __forceinline__ void hrx_q6_k_wmma_vk128_append_half4(
        hrx_q6_k_wmma_vk128_half16_vec * result,
        int base,
        hrx_q6_k_wmma_vk128_half4_vec values) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        (*result)[base + i] = values[i];
    }
}

static __device__ __forceinline__ hrx_q6_k_wmma_vk128_half4_vec hrx_q6_k_wmma_vk128_ds_read_b64_h4(
        hrx_q6_k_wmma_vk128_lds_half_ptr ptr) {
    const __attribute__((address_space(3))) uint64_t * lds_ptr =
        (const __attribute__((address_space(3))) uint64_t *) ptr;
    hrx_q6_k_wmma_vk128_half4_vec value;
    asm volatile("ds_read_b64 %0, %1 offset:0\n"
                 : "=v"(value)
                 : "v"(lds_ptr)
                 : "memory");
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    return value;
}

static __device__ __forceinline__ hrx_q6_k_wmma_vk128_half4_vec hrx_q6_k_wmma_vk128_ds_read_b64_h4_nowait(
        hrx_q6_k_wmma_vk128_lds_half_ptr ptr) {
    const __attribute__((address_space(3))) uint64_t * lds_ptr =
        (const __attribute__((address_space(3))) uint64_t *) ptr;
    hrx_q6_k_wmma_vk128_half4_vec value;
    asm volatile("ds_read_b64 %0, %1 offset:0\n"
                 : "=v"(value)
                 : "v"(lds_ptr)
                 : "memory");
    return value;
}

static __device__ __forceinline__ hrx_q6_k_wmma_vk128_half16_vec hrx_q6_k_wmma_vk128_load_a_frag_w64_b64asm(
        hrx_q6_k_wmma_vk128_lds_half_ptr sh_a,
        int row_tile,
        int k_tile,
        unsigned int lane) {
    constexpr int SHARED_STRIDE = HRX_Q6_K_WMMA_VK128_SHARED_STRIDE;
    const int row = row_tile * 16 + static_cast<int>(lane & 15u);
    const int k_base = k_tile * 16;
    hrx_q6_k_wmma_vk128_lds_half_ptr row_ptr = sh_a + row * SHARED_STRIDE + k_base;
    hrx_q6_k_wmma_vk128_half16_vec result;
    hrx_q6_k_wmma_vk128_append_half4(&result, 0, hrx_q6_k_wmma_vk128_ds_read_b64_h4(row_ptr + 0));
    hrx_q6_k_wmma_vk128_append_half4(&result, 4, hrx_q6_k_wmma_vk128_ds_read_b64_h4(row_ptr + 4));
    hrx_q6_k_wmma_vk128_append_half4(&result, 8, hrx_q6_k_wmma_vk128_ds_read_b64_h4(row_ptr + 8));
    hrx_q6_k_wmma_vk128_append_half4(&result, 12, hrx_q6_k_wmma_vk128_ds_read_b64_h4(row_ptr + 12));
    return result;
}

static __device__ __forceinline__ hrx_q6_k_wmma_vk128_half16_vec hrx_q6_k_wmma_vk128_load_a_frag_w64_b64asm_nowait(
        hrx_q6_k_wmma_vk128_lds_half_ptr sh_a,
        int row_tile,
        int k_tile,
        unsigned int lane) {
    constexpr int SHARED_STRIDE = HRX_Q6_K_WMMA_VK128_SHARED_STRIDE;
    const int row = row_tile * 16 + static_cast<int>(lane & 15u);
    const int k_base = k_tile * 16;
    hrx_q6_k_wmma_vk128_lds_half_ptr row_ptr = sh_a + row * SHARED_STRIDE + k_base;
    hrx_q6_k_wmma_vk128_half16_vec result;
    hrx_q6_k_wmma_vk128_append_half4(&result, 0, hrx_q6_k_wmma_vk128_ds_read_b64_h4_nowait(row_ptr + 0));
    hrx_q6_k_wmma_vk128_append_half4(&result, 4, hrx_q6_k_wmma_vk128_ds_read_b64_h4_nowait(row_ptr + 4));
    hrx_q6_k_wmma_vk128_append_half4(&result, 8, hrx_q6_k_wmma_vk128_ds_read_b64_h4_nowait(row_ptr + 8));
    hrx_q6_k_wmma_vk128_append_half4(&result, 12, hrx_q6_k_wmma_vk128_ds_read_b64_h4_nowait(row_ptr + 12));
    return result;
}

static __device__ __forceinline__ hrx_q6_k_wmma_vk128_half16_vec hrx_q6_k_wmma_vk128_load_b_frag_w64_b64asm(
        hrx_q6_k_wmma_vk128_lds_half_ptr sh_b,
        int col_tile,
        int k_tile,
        unsigned int lane) {
    constexpr int SHARED_STRIDE = HRX_Q6_K_WMMA_VK128_SHARED_STRIDE;
    const int col = col_tile * 16 + static_cast<int>(lane & 15u);
    const int k_base = k_tile * 16;
    hrx_q6_k_wmma_vk128_lds_half_ptr col_ptr = sh_b + col * SHARED_STRIDE + k_base;
    hrx_q6_k_wmma_vk128_half16_vec result;
    hrx_q6_k_wmma_vk128_append_half4(&result, 0, hrx_q6_k_wmma_vk128_ds_read_b64_h4(col_ptr + 0));
    hrx_q6_k_wmma_vk128_append_half4(&result, 4, hrx_q6_k_wmma_vk128_ds_read_b64_h4(col_ptr + 4));
    hrx_q6_k_wmma_vk128_append_half4(&result, 8, hrx_q6_k_wmma_vk128_ds_read_b64_h4(col_ptr + 8));
    hrx_q6_k_wmma_vk128_append_half4(&result, 12, hrx_q6_k_wmma_vk128_ds_read_b64_h4(col_ptr + 12));
    return result;
}

static __device__ __forceinline__ hrx_q6_k_wmma_vk128_half16_vec hrx_q6_k_wmma_vk128_load_b_frag_w64_b64asm_nowait(
        hrx_q6_k_wmma_vk128_lds_half_ptr sh_b,
        int col_tile,
        int k_tile,
        unsigned int lane) {
    constexpr int SHARED_STRIDE = HRX_Q6_K_WMMA_VK128_SHARED_STRIDE;
    const int col = col_tile * 16 + static_cast<int>(lane & 15u);
    const int k_base = k_tile * 16;
    hrx_q6_k_wmma_vk128_lds_half_ptr col_ptr = sh_b + col * SHARED_STRIDE + k_base;
    hrx_q6_k_wmma_vk128_half16_vec result;
    hrx_q6_k_wmma_vk128_append_half4(&result, 0, hrx_q6_k_wmma_vk128_ds_read_b64_h4_nowait(col_ptr + 0));
    hrx_q6_k_wmma_vk128_append_half4(&result, 4, hrx_q6_k_wmma_vk128_ds_read_b64_h4_nowait(col_ptr + 4));
    hrx_q6_k_wmma_vk128_append_half4(&result, 8, hrx_q6_k_wmma_vk128_ds_read_b64_h4_nowait(col_ptr + 8));
    hrx_q6_k_wmma_vk128_append_half4(&result, 12, hrx_q6_k_wmma_vk128_ds_read_b64_h4_nowait(col_ptr + 12));
    return result;
}

static __device__ __forceinline__ hrx_q6_k_wmma_vk128_half16_vec hrx_q6_k_wmma_vk128_load_a_frag_w64_h4(
        const _Float16 * sh_a,
        int row_tile,
        int k_tile,
        unsigned int lane) {
    constexpr int SHARED_STRIDE = HRX_Q6_K_WMMA_VK128_SHARED_STRIDE;
    const int row = row_tile * 16 + static_cast<int>(lane & 15u);
    const int k_base = k_tile * 16;
    const hrx_q6_k_wmma_vk128_half4_vec * row_ptr =
        reinterpret_cast<const hrx_q6_k_wmma_vk128_half4_vec *>(sh_a + row * SHARED_STRIDE + k_base);
    hrx_q6_k_wmma_vk128_half16_vec result;
    hrx_q6_k_wmma_vk128_append_half4(&result, 0, row_ptr[0]);
    hrx_q6_k_wmma_vk128_append_half4(&result, 4, row_ptr[1]);
    hrx_q6_k_wmma_vk128_append_half4(&result, 8, row_ptr[2]);
    hrx_q6_k_wmma_vk128_append_half4(&result, 12, row_ptr[3]);
    return result;
}

static __device__ __forceinline__ hrx_q6_k_wmma_vk128_half16_vec hrx_q6_k_wmma_vk128_load_b_frag_w64_h4(
        const _Float16 * sh_b,
        int col_tile,
        int k_tile,
        unsigned int lane) {
    constexpr int SHARED_STRIDE = HRX_Q6_K_WMMA_VK128_SHARED_STRIDE;
    const int col = col_tile * 16 + static_cast<int>(lane & 15u);
    const int k_base = k_tile * 16;
    const hrx_q6_k_wmma_vk128_half4_vec * col_ptr =
        reinterpret_cast<const hrx_q6_k_wmma_vk128_half4_vec *>(sh_b + col * SHARED_STRIDE + k_base);
    hrx_q6_k_wmma_vk128_half16_vec result;
    hrx_q6_k_wmma_vk128_append_half4(&result, 0, col_ptr[0]);
    hrx_q6_k_wmma_vk128_append_half4(&result, 4, col_ptr[1]);
    hrx_q6_k_wmma_vk128_append_half4(&result, 8, col_ptr[2]);
    hrx_q6_k_wmma_vk128_append_half4(&result, 12, col_ptr[3]);
    return result;
}

static __device__ __forceinline__ void hrx_q6_k_wmma_vk128_store_acc_f16_row_major(
        float * dst,
        long long rows_stride,
        long long row0,
        long long col0,
        long long rows,
        long long cols,
        hrx_q6_k_wmma_vk128_half16_vec acc,
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

static __device__ __forceinline__ void hrx_q6_k_wmma_vk128_store_acc_f16_row_major_w64(
        float * dst,
        long long rows_stride,
        long long row0,
        long long col0,
        long long rows,
        long long cols,
        hrx_q6_k_wmma_vk128_half8_vec acc,
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
                static_cast<float>(acc[reg * 2 + HRX_Q6_K_WMMA_VK128_W64_OPSEL]);
        }
    }
}

static __device__ __forceinline__ void hrx_q6_k_wmma_vk128_store_acc_f16_row_major_w64_full(
        float * dst,
        long long rows_stride,
        long long row0,
        long long col0,
        hrx_q6_k_wmma_vk128_half8_vec acc,
        unsigned int lane) {
    const long long row_lane = static_cast<long long>(lane >> 4);
    const long long col = col0 + static_cast<long long>(lane & 15u);
#pragma unroll
    for (int reg = 0; reg < 4; ++reg) {
        const long long row = row0 + row_lane + static_cast<long long>(reg * 4);
        dst[col * rows_stride + row] =
            static_cast<float>(acc[reg * 2 + HRX_Q6_K_WMMA_VK128_W64_OPSEL]);
    }
}

#if HRX_Q6_K_WMMA_VK128_BUFFER_STORE
static __device__ __forceinline__ void hrx_q6_k_wmma_vk128_store_acc_f16_row_major_w64_buffer_full(
        __amdgpu_buffer_rsrc_t dst_rsrc,
        long long rows_stride,
        long long row0,
        long long col0,
        hrx_q6_k_wmma_vk128_half8_vec acc,
        unsigned int lane) {
    const long long row_lane = static_cast<long long>(lane >> 4);
    const long long col = col0 + static_cast<long long>(lane & 15u);
#pragma unroll
    for (int reg = 0; reg < 4; ++reg) {
        const long long row = row0 + row_lane + static_cast<long long>(reg * 4);
        hrx_q6_k_wmma_vk128_buffer_store_f32(
            dst_rsrc,
            col * rows_stride + row,
            static_cast<float>(acc[reg * 2 + HRX_Q6_K_WMMA_VK128_W64_OPSEL]));
    }
}

static __device__ __forceinline__ void hrx_q6_k_wmma_vk128_store_acc_f16_row_major_w64_buffer(
        __amdgpu_buffer_rsrc_t dst_rsrc,
        long long rows_stride,
        long long row0,
        long long col0,
        long long rows,
        long long cols,
        hrx_q6_k_wmma_vk128_half8_vec acc,
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
            hrx_q6_k_wmma_vk128_buffer_store_f32(
                dst_rsrc,
                col * rows_stride + row,
                static_cast<float>(acc[reg * 2 + HRX_Q6_K_WMMA_VK128_W64_OPSEL]));
        }
    }
}
#endif

static __device__ __forceinline__ void hrx_q6_k_wmma_vk128_store_acc_f16_row_major_w64_stage(
        float * dst,
        long long rows_stride,
        long long row0,
        long long col0,
        long long rows,
        long long cols,
        hrx_q6_k_wmma_vk128_half8_vec acc,
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
        sh_store[col_major_base + reg * 4] = acc[reg * 2 + HRX_Q6_K_WMMA_VK128_W64_OPSEL];
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

#if HRX_Q6_K_WMMA_VK128_STORE_STAGE_FAST_HALF && HRX_Q6_K_WMMA_VK128_BUFFER_STORE
static __device__ __forceinline__ void hrx_q6_k_wmma_vk128_ds_store_u16(
        hrx_q6_k_wmma_vk128_lds_u16_ptr ptr,
        uint16_t value) {
    asm volatile("ds_write_b16 %0, %1 offset:0\n"
                 :
                 : "v"(ptr), "v"(static_cast<uint32_t>(value))
                 : "memory");
}

static __device__ __forceinline__ uint32_t hrx_q6_k_wmma_vk128_ds_load_u16_d16(
        hrx_q6_k_wmma_vk128_lds_const_u16_ptr ptr) {
    uint32_t value = 0;
    asm volatile("ds_read_u16_d16 %0, %1 offset:0\n"
                 : "=v"(value)
                 : "v"(ptr)
                 : "memory");
    return value;
}

static __device__ __forceinline__ void hrx_q6_k_wmma_vk128_store_acc_f16_row_major_w64_fast_half_buffer_selected(
        __amdgpu_buffer_rsrc_t dst_rsrc,
        long long rows_stride,
        long long row0,
        long long col0,
        long long rows,
        long long cols,
        hrx_q6_k_wmma_vk128_half8_vec acc,
        unsigned int lane,
        unsigned int wave,
        hrx_q6_k_wmma_vk128_lds_volatile_half_ptr sh_store) {
    constexpr int TILE_STRIDE = 16 * 16;
    constexpr int SELECTED_OPSEL = HRX_Q6_K_WMMA_VK128_W64_OPSEL;
    constexpr int OTHER_OPSEL = SELECTED_OPSEL == 0 ? 1 : 0;
    const int row_lane = static_cast<int>(lane >> 4);
    const int col_lane = static_cast<int>(lane & 15u);
    const int tile_base = static_cast<int>(wave) * TILE_STRIDE * 2;
    const int dummy_tile_base = tile_base + TILE_STRIDE;
    const int col_major_base = tile_base + col_lane * 16 + row_lane;
    const int dummy_col_major_base = dummy_tile_base + col_lane * 16 + row_lane;
    hrx_q6_k_wmma_vk128_lds_u16_ptr sh_u16 =
        (hrx_q6_k_wmma_vk128_lds_u16_ptr) sh_store;
    const long long col = col0 + static_cast<long long>(col_lane);

#pragma unroll
    for (int reg = 0; reg < 4; ++reg) {
        const int selected_offset = col_major_base + reg * 4;
        const int other_offset = dummy_col_major_base + reg * 4;
        hrx_q6_k_wmma_vk128_ds_store_u16(
            sh_u16 + selected_offset,
            hrx_q6_k_wmma_vk128_f16_to_u16(acc[reg * 2 + SELECTED_OPSEL]));
        hrx_q6_k_wmma_vk128_ds_store_u16(
            sh_u16 + other_offset,
            hrx_q6_k_wmma_vk128_f16_to_u16(acc[reg * 2 + OTHER_OPSEL]));
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");

#pragma unroll
    for (int reg = 0; reg < 4; ++reg) {
        const int selected_offset = col_major_base + reg * 4;
        const int other_offset = dummy_col_major_base + reg * 4;
        const _Float16 selected = hrx_q6_k_wmma_vk128_u16_to_f16(
            hrx_q6_k_wmma_vk128_ds_load_u16_d16(sh_u16 + selected_offset));
        const uint32_t other_bits = hrx_q6_k_wmma_vk128_ds_load_u16_d16(sh_u16 + other_offset);
        (void) other_bits;
        asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
        const long long row = row0 + static_cast<long long>(row_lane + reg * 4);
        if (row < rows && col < cols) {
            hrx_q6_k_wmma_vk128_buffer_store_f32(
                dst_rsrc,
                col * rows_stride + row,
                static_cast<float>(selected));
        }
    }
}
#endif

static __device__ __forceinline__ void hrx_q6_k_wmma_vk128_tile_map(
        int wave,
        int tile_iter,
        int * row_tile,
        int * col_tile) {
    constexpr int ROW_TILES = HRX_Q6_K_WMMA_VK128_BM / 16;
#if HRX_Q6_K_WMMA_VK128_W64
    constexpr int WAVE = 64;
#else
    constexpr int WAVE = 32;
#endif
    constexpr int WAVE_COUNT = HRX_Q6_K_WMMA_VK128_WG_SIZE / WAVE;
    constexpr int TILE_COUNT = (HRX_Q6_K_WMMA_VK128_BM / 16) * (HRX_Q6_K_WMMA_VK128_BN / 16);
    constexpr int TILES_PER_WAVE = TILE_COUNT / WAVE_COUNT;
#if HRX_Q6_K_WMMA_VK128_PAIR64_TILE_MAP
    const int pair = wave >> 1;
    const int lane_wave = wave & 1;
    const int tile = lane_wave + tile_iter * 2;
    const int pair_row = pair & 1;
    const int pair_col = pair >> 1;
    *row_tile = pair_row * 4 + (tile & 3);
    *col_tile = pair_col * 4 + (tile >> 2);
#elif HRX_Q6_K_WMMA_VK128_W64 && HRX_Q6_K_WMMA_VK128_BM == 128 && HRX_Q6_K_WMMA_VK128_BN == 128
    const int wave_row = wave & 1;
    const int wave_col = wave >> 1;
    *row_tile = wave_row * 4 + (tile_iter & 3);
    *col_tile = wave_col * 4 + (tile_iter >> 2);
#elif HRX_Q6_K_WMMA_VK128_W64 && HRX_Q6_K_WMMA_VK128_BM == 64 && HRX_Q6_K_WMMA_VK128_BN == 64
    const int tile = wave * TILES_PER_WAVE + tile_iter;
    *row_tile = tile % ROW_TILES;
    *col_tile = tile / ROW_TILES;
#else
    const int tile = wave + tile_iter * WAVE_COUNT;
    *row_tile = tile & (ROW_TILES - 1);
    *col_tile = tile >> 3;
#endif
}

extern "C" __global__ __launch_bounds__(HRX_Q6_K_WMMA_VK128_WG_SIZE, 1)
void HRX_Q6_K_WMMA_VK128_EXPORT(
        const hrx_block_q6_K_wmma_vk128_lhs * src0,
        const float * src1,
        float * dst,
        long long k,
        long long rows,
        long long cols) {
    constexpr int BM = HRX_Q6_K_WMMA_VK128_BM;
    constexpr int BN = HRX_Q6_K_WMMA_VK128_BN;
    constexpr int BK = 32;
    constexpr int SHARED_STRIDE = HRX_Q6_K_WMMA_VK128_SHARED_STRIDE;
#if HRX_Q6_K_WMMA_VK128_W64
    constexpr int WAVE = 64;
#else
    constexpr int WAVE = 32;
#endif
    constexpr int WG_SIZE = HRX_Q6_K_WMMA_VK128_WG_SIZE;
    constexpr int ROW_TILES = BM / 16;
    constexpr int COL_TILES = BN / 16;
    constexpr int TILE_COUNT = ROW_TILES * COL_TILES;
    constexpr int WAVE_COUNT = WG_SIZE / WAVE;
    constexpr int TILES_PER_WAVE = TILE_COUNT / WAVE_COUNT;

    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int wave = tid / WAVE;
    const unsigned int lane = tid & static_cast<unsigned int>(WAVE - 1);
    const long long row_base = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * BM;
    const long long col_base = static_cast<long long>(__builtin_amdgcn_workgroup_id_y()) * BN;
    if (row_base >= rows || col_base >= cols) {
        return;
    }

    __shared__ _Float16 sh_a[BM * SHARED_STRIDE];
    __shared__ _Float16 sh_b[BN * SHARED_STRIDE];
#if HRX_Q6_K_WMMA_VK128_STORE_STAGE_FAST_HALF
    __shared__ _Float16 sh_store[WAVE_COUNT * 2 * 16 * 16];
#elif HRX_Q6_K_WMMA_VK128_STORE_STAGE
    __shared__ _Float16 sh_store[WAVE_COUNT * 16 * 16];
#endif

    const long long blocks_per_row = k / 256;
    const _Float16 zero = static_cast<_Float16>(0.0f);
#if HRX_Q6_K_WMMA_VK128_BUFFER_STORE
    const __amdgpu_buffer_rsrc_t dst_rsrc = hrx_q6_k_wmma_vk128_make_dst_rsrc(dst);
#endif
#if HRX_Q6_K_WMMA_VK128_W64_VK64_RING96
    hrx_q6_k_wmma_vk128_half8_vec acc[16] = {};
#elif HRX_Q6_K_WMMA_VK128_W64
    hrx_q6_k_wmma_vk128_half8_vec acc[TILES_PER_WAVE] = {};
#else
    hrx_q6_k_wmma_vk128_half16_vec acc[TILES_PER_WAVE] = {};
#endif

    for (long long k0 = 0; k0 < k; k0 += BK) {
        for (int idx = static_cast<int>(tid); idx < BM * BK; idx += WG_SIZE) {
            const int r = idx / BK;
            const int kk = idx - r * BK;
            const long long row = row_base + static_cast<long long>(r);
            sh_a[r * SHARED_STRIDE + kk] = row < rows ?
                hrx_q6_k_wmma_vk128_load_a_value(src0, row, k0 + kk, blocks_per_row) : zero;
        }
        for (int idx = static_cast<int>(tid); idx < BN * BK; idx += WG_SIZE) {
            const int c = idx / BK;
            const int kk = idx - c * BK;
            const long long col = col_base + static_cast<long long>(c);
            sh_b[c * SHARED_STRIDE + kk] = col < cols ? static_cast<_Float16>(src1[col * k + k0 + kk]) : zero;
        }
        __syncthreads();

#if HRX_Q6_K_WMMA_VK128_W64_VK64_RING96
        if (tid < 64u) {
            hrx_q6_k_wmma_vk128_lds_half_ptr sh_a_lds =
                (hrx_q6_k_wmma_vk128_lds_half_ptr) sh_a;
            hrx_q6_k_wmma_vk128_lds_half_ptr sh_b_lds =
                (hrx_q6_k_wmma_vk128_lds_half_ptr) sh_b;
            hrx_q6_k_wmma_vk128_half16_vec a_frag[6];
            hrx_q6_k_wmma_vk128_half16_vec b_frag[6];
#pragma unroll
            for (int frag = 0; frag < 6; ++frag) {
                a_frag[frag] = hrx_q6_k_wmma_vk128_load_a_frag_w64_b64asm_nowait(
                    sh_a_lds, frag & 3, 0, lane);
                b_frag[frag] = hrx_q6_k_wmma_vk128_load_b_frag_w64_b64asm_nowait(
                    sh_b_lds, (frag + 2) & 3, 1, lane);
            }
            asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
#pragma unroll
            for (int group = 0; group < 16; ++group) {
                const int row_frag = group & 3;
                const int col_frag = (group >> 2) & 3;
                acc[group] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(
                    a_frag[row_frag],
                    b_frag[col_frag],
                    acc[group],
                    HRX_Q6_K_WMMA_VK128_W64_OPSEL != 0);
            }
        }
#elif HRX_Q6_K_WMMA_VK128_W64_VK64_GROUPK2 || HRX_Q6_K_WMMA_VK128_W64_VK64_GROUPK2_WAIT
        {
            hrx_q6_k_wmma_vk128_lds_half_ptr sh_a_lds =
                (hrx_q6_k_wmma_vk128_lds_half_ptr) sh_a;
            hrx_q6_k_wmma_vk128_lds_half_ptr sh_b_lds =
                (hrx_q6_k_wmma_vk128_lds_half_ptr) sh_b;
            const int col_tile = static_cast<int>(wave);
            hrx_q6_k_wmma_vk128_half16_vec a_frag[2][4];
            hrx_q6_k_wmma_vk128_half16_vec b_frag[2];
#pragma unroll
            for (int k_tile = 0; k_tile < 2; ++k_tile) {
#pragma unroll
                for (int row_sub = 0; row_sub < 4; ++row_sub) {
#if HRX_Q6_K_WMMA_VK128_W64_VK64_GROUPK2_WAIT
                    a_frag[k_tile][row_sub] = hrx_q6_k_wmma_vk128_load_a_frag_w64_b64asm(
                        sh_a_lds, row_sub, k_tile, lane);
#else
                    a_frag[k_tile][row_sub] = hrx_q6_k_wmma_vk128_load_a_frag_w64_b64asm_nowait(
                        sh_a_lds, row_sub, k_tile, lane);
#endif
                }
#if HRX_Q6_K_WMMA_VK128_W64_VK64_GROUPK2_WAIT
                b_frag[k_tile] = hrx_q6_k_wmma_vk128_load_b_frag_w64_b64asm(
                    sh_b_lds, col_tile, k_tile, lane);
#else
                b_frag[k_tile] = hrx_q6_k_wmma_vk128_load_b_frag_w64_b64asm_nowait(
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
                        HRX_Q6_K_WMMA_VK128_W64_OPSEL != 0);
                }
            }
        }
#elif HRX_Q6_K_WMMA_VK128_W64_B64GROUP
        {
            hrx_q6_k_wmma_vk128_lds_half_ptr sh_a_lds =
                (hrx_q6_k_wmma_vk128_lds_half_ptr) sh_a;
            hrx_q6_k_wmma_vk128_lds_half_ptr sh_b_lds =
                (hrx_q6_k_wmma_vk128_lds_half_ptr) sh_b;
            const int wave_row = static_cast<int>(wave & 1u);
            const int wave_col = static_cast<int>(wave >> 1);
#pragma unroll
            for (int k_tile = 0; k_tile < 2; ++k_tile) {
                hrx_q6_k_wmma_vk128_half16_vec a_frag[4];
                hrx_q6_k_wmma_vk128_half16_vec b_frag[4];
#pragma unroll
                for (int row_sub = 0; row_sub < 4; ++row_sub) {
                    a_frag[row_sub] = hrx_q6_k_wmma_vk128_load_a_frag_w64_b64asm(
                        sh_a_lds, wave_row * 4 + row_sub, k_tile, lane);
                }
#pragma unroll
                for (int col_sub = 0; col_sub < 4; ++col_sub) {
                    b_frag[col_sub] = hrx_q6_k_wmma_vk128_load_b_frag_w64_b64asm(
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
                            HRX_Q6_K_WMMA_VK128_W64_OPSEL != 0);
                    }
                }
            }
        }
#else
#pragma unroll
        for (int tile_iter = 0; tile_iter < TILES_PER_WAVE; ++tile_iter) {
            int row_tile = 0;
            int col_tile = 0;
            hrx_q6_k_wmma_vk128_tile_map(static_cast<int>(wave), tile_iter, &row_tile, &col_tile);
#if HRX_Q6_K_WMMA_VK128_W64
#pragma unroll
            for (int k_tile = 0; k_tile < 2; ++k_tile) {
#if HRX_Q6_K_WMMA_VK128_W64_B64ASM
                hrx_q6_k_wmma_vk128_lds_half_ptr sh_a_lds =
                    (hrx_q6_k_wmma_vk128_lds_half_ptr) sh_a;
                hrx_q6_k_wmma_vk128_lds_half_ptr sh_b_lds =
                    (hrx_q6_k_wmma_vk128_lds_half_ptr) sh_b;
                const hrx_q6_k_wmma_vk128_half16_vec a =
                    hrx_q6_k_wmma_vk128_load_a_frag_w64_b64asm(sh_a_lds, row_tile, k_tile, lane);
                const hrx_q6_k_wmma_vk128_half16_vec b =
                    hrx_q6_k_wmma_vk128_load_b_frag_w64_b64asm(sh_b_lds, col_tile, k_tile, lane);
#elif HRX_Q6_K_WMMA_VK128_W64_H4LOAD
                const hrx_q6_k_wmma_vk128_half16_vec a =
                    hrx_q6_k_wmma_vk128_load_a_frag_w64_h4(sh_a, row_tile, k_tile, lane);
                const hrx_q6_k_wmma_vk128_half16_vec b =
                    hrx_q6_k_wmma_vk128_load_b_frag_w64_h4(sh_b, col_tile, k_tile, lane);
#else
                const hrx_q6_k_wmma_vk128_half16_vec a =
                    hrx_q6_k_wmma_vk128_load_a_frag_w64(sh_a, row_tile, k_tile, lane);
                const hrx_q6_k_wmma_vk128_half16_vec b =
                    hrx_q6_k_wmma_vk128_load_b_frag_w64(sh_b, col_tile, k_tile, lane);
#endif
                acc[tile_iter] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(
                    a, b, acc[tile_iter], HRX_Q6_K_WMMA_VK128_W64_OPSEL != 0);
            }
#else
#if HRX_Q6_K_WMMA_VK128_PREFETCH_FRAGS
            hrx_q6_k_wmma_vk128_half16_vec a_frag[2];
            hrx_q6_k_wmma_vk128_half16_vec b_frag[2];
#pragma unroll
            for (int k_tile = 0; k_tile < 2; ++k_tile) {
                a_frag[k_tile] = hrx_q6_k_wmma_vk128_load_a_frag(sh_a, row_tile, k_tile, lane);
                b_frag[k_tile] = hrx_q6_k_wmma_vk128_load_b_frag(sh_b, col_tile, k_tile, lane);
            }
#pragma unroll
            for (int k_tile = 0; k_tile < 2; ++k_tile) {
                acc[tile_iter] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(
                    a_frag[k_tile], b_frag[k_tile], acc[tile_iter], false);
            }
#else
#pragma unroll
            for (int k_tile = 0; k_tile < 2; ++k_tile) {
                const hrx_q6_k_wmma_vk128_half16_vec a =
                    hrx_q6_k_wmma_vk128_load_a_frag(sh_a, row_tile, k_tile, lane);
                const hrx_q6_k_wmma_vk128_half16_vec b =
                    hrx_q6_k_wmma_vk128_load_b_frag(sh_b, col_tile, k_tile, lane);
                acc[tile_iter] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(a, b, acc[tile_iter], false);
            }
#endif
#endif
        }
#endif

        __syncthreads();
    }

#if HRX_Q6_K_WMMA_VK128_W64_VK64_RING96
#pragma unroll
    for (int group = 0; group < 16; ++group) {
        if (tid < 64u) {
            const int row_tile = group & 3;
            const int col_tile = (group >> 2) & 3;
#if HRX_Q6_K_WMMA_VK128_BUFFER_STORE
            hrx_q6_k_wmma_vk128_store_acc_f16_row_major_w64_buffer(
                dst_rsrc,
                rows,
                row_base + static_cast<long long>(row_tile * 16),
                col_base + static_cast<long long>(col_tile * 16),
                rows,
                cols,
                acc[group],
                lane);
#else
            hrx_q6_k_wmma_vk128_store_acc_f16_row_major_w64(
                dst,
                rows,
                row_base + static_cast<long long>(row_tile * 16),
                col_base + static_cast<long long>(col_tile * 16),
                rows,
                cols,
                acc[group],
                lane);
#endif
        }
    }
#else
#pragma unroll
    for (int tile_iter = 0; tile_iter < TILES_PER_WAVE; ++tile_iter) {
        int row_tile = 0;
        int col_tile = 0;
        hrx_q6_k_wmma_vk128_tile_map(static_cast<int>(wave), tile_iter, &row_tile, &col_tile);
#if HRX_Q6_K_WMMA_VK128_W64
#if HRX_Q6_K_WMMA_VK128_STORE_STAGE_FAST_HALF
        hrx_q6_k_wmma_vk128_store_acc_f16_row_major_w64_fast_half_buffer_selected(
            dst_rsrc,
            rows,
            row_base + static_cast<long long>(row_tile * 16),
            col_base + static_cast<long long>(col_tile * 16),
            rows,
            cols,
            acc[tile_iter],
            lane,
            wave,
            (hrx_q6_k_wmma_vk128_lds_volatile_half_ptr) sh_store);
#elif HRX_Q6_K_WMMA_VK128_STORE_STAGE
        hrx_q6_k_wmma_vk128_store_acc_f16_row_major_w64_stage(
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
#if HRX_Q6_K_WMMA_VK128_FULL_TILE_STORE
        const long long tile_row0 = row_base + static_cast<long long>(row_tile * 16);
        const long long tile_col0 = col_base + static_cast<long long>(col_tile * 16);
        if (tile_row0 + 16 <= rows && tile_col0 + 16 <= cols) {
#if HRX_Q6_K_WMMA_VK128_BUFFER_STORE
            hrx_q6_k_wmma_vk128_store_acc_f16_row_major_w64_buffer_full(
                dst_rsrc,
                rows,
                tile_row0,
                tile_col0,
                acc[tile_iter],
                lane);
#else
            hrx_q6_k_wmma_vk128_store_acc_f16_row_major_w64_full(
                dst,
                rows,
                tile_row0,
                tile_col0,
                acc[tile_iter],
                lane);
#endif
        } else {
#if HRX_Q6_K_WMMA_VK128_BUFFER_STORE
            hrx_q6_k_wmma_vk128_store_acc_f16_row_major_w64_buffer(
                dst_rsrc,
                rows,
                tile_row0,
                tile_col0,
                rows,
                cols,
                acc[tile_iter],
                lane);
#else
            hrx_q6_k_wmma_vk128_store_acc_f16_row_major_w64(
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
#if HRX_Q6_K_WMMA_VK128_BUFFER_STORE
        hrx_q6_k_wmma_vk128_store_acc_f16_row_major_w64_buffer(
            dst_rsrc,
            rows,
            row_base + static_cast<long long>(row_tile * 16),
            col_base + static_cast<long long>(col_tile * 16),
            rows,
            cols,
            acc[tile_iter],
            lane);
#else
        hrx_q6_k_wmma_vk128_store_acc_f16_row_major_w64(
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
#endif
#else
        hrx_q6_k_wmma_vk128_store_acc_f16_row_major(
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
