#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <stdint.h>

#ifndef HRX_Q4_K_WMMA_VK128_EXPORT
#define HRX_Q4_K_WMMA_VK128_EXPORT hrx_mul_mat_vec_q4_k_wmma16x16_vk128_f16acc_wg256_f32
#endif

#ifndef HRX_Q4_K_WMMA_VK128_SHARED_STRIDE
#define HRX_Q4_K_WMMA_VK128_SHARED_STRIDE 32
#endif

#ifndef HRX_Q4_K_WMMA_VK128_PREFETCH_FRAGS
#define HRX_Q4_K_WMMA_VK128_PREFETCH_FRAGS 0
#endif

#ifndef HRX_Q4_K_WMMA_VK128_PAIR64_TILE_MAP
#define HRX_Q4_K_WMMA_VK128_PAIR64_TILE_MAP 0
#endif

#ifndef HRX_Q4_K_WMMA_VK128_W64
#define HRX_Q4_K_WMMA_VK128_W64 0
#endif

#ifndef HRX_Q4_K_WMMA_VK128_W64_OPSEL
#define HRX_Q4_K_WMMA_VK128_W64_OPSEL 0
#endif

#ifndef HRX_Q4_K_WMMA_VK128_W64_H4LOAD
#define HRX_Q4_K_WMMA_VK128_W64_H4LOAD 0
#endif

#ifndef HRX_Q4_K_WMMA_VK128_W64_B64ASM
#define HRX_Q4_K_WMMA_VK128_W64_B64ASM 0
#endif

#ifndef HRX_Q4_K_WMMA_VK128_W64_B64GROUP
#define HRX_Q4_K_WMMA_VK128_W64_B64GROUP 0
#endif

struct hrx_block_q4_K_wmma_vk128_lhs {
    unsigned short d;
    unsigned short dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};

typedef _Float16 hrx_q4_k_wmma_vk128_half16_vec __attribute__((ext_vector_type(16)));
typedef _Float16 hrx_q4_k_wmma_vk128_half8_vec __attribute__((ext_vector_type(8)));
typedef _Float16 hrx_q4_k_wmma_vk128_half4_vec __attribute__((ext_vector_type(4)));
typedef const __attribute__((address_space(3))) _Float16 * hrx_q4_k_wmma_vk128_lds_half_ptr;

static __device__ __forceinline__ uint32_t hrx_q4_k_wmma_vk128_pack_f16x2(_Float16 lo, _Float16 hi) {
    union {
        _Float16 h[2];
        uint32_t u;
    } pack;
    pack.h[0] = lo;
    pack.h[1] = hi;
    return pack.u;
}

static __device__ __forceinline__ _Float16 hrx_q4_k_wmma_vk128_unpack_f16x2(uint32_t bits, int idx) {
    union {
        uint32_t u;
        _Float16 h[2];
    } pack;
    pack.u = bits;
    return pack.h[idx];
}

static __device__ __forceinline__ hrx_q4_k_wmma_vk128_half16_vec hrx_q4_k_wmma_vk128_duplicate_input(
        _Float16 x0, _Float16 x1, _Float16 x2, _Float16 x3,
        _Float16 x4, _Float16 x5, _Float16 x6, _Float16 x7) {
    constexpr int SWAP16_CTRL = (16 << 10) | 0x1f;
    const uint32_t p0 = hrx_q4_k_wmma_vk128_pack_f16x2(x0, x1);
    const uint32_t p1 = hrx_q4_k_wmma_vk128_pack_f16x2(x2, x3);
    const uint32_t p2 = hrx_q4_k_wmma_vk128_pack_f16x2(x4, x5);
    const uint32_t p3 = hrx_q4_k_wmma_vk128_pack_f16x2(x6, x7);
    const uint32_t s0 = static_cast<uint32_t>(__builtin_amdgcn_ds_swizzle(static_cast<int32_t>(p0), SWAP16_CTRL));
    const uint32_t s1 = static_cast<uint32_t>(__builtin_amdgcn_ds_swizzle(static_cast<int32_t>(p1), SWAP16_CTRL));
    const uint32_t s2 = static_cast<uint32_t>(__builtin_amdgcn_ds_swizzle(static_cast<int32_t>(p2), SWAP16_CTRL));
    const uint32_t s3 = static_cast<uint32_t>(__builtin_amdgcn_ds_swizzle(static_cast<int32_t>(p3), SWAP16_CTRL));

    hrx_q4_k_wmma_vk128_half16_vec result;
    result[0] = x0;
    result[1] = x1;
    result[2] = x2;
    result[3] = x3;
    result[4] = x4;
    result[5] = x5;
    result[6] = x6;
    result[7] = x7;
    result[8] = hrx_q4_k_wmma_vk128_unpack_f16x2(s0, 0);
    result[9] = hrx_q4_k_wmma_vk128_unpack_f16x2(s0, 1);
    result[10] = hrx_q4_k_wmma_vk128_unpack_f16x2(s1, 0);
    result[11] = hrx_q4_k_wmma_vk128_unpack_f16x2(s1, 1);
    result[12] = hrx_q4_k_wmma_vk128_unpack_f16x2(s2, 0);
    result[13] = hrx_q4_k_wmma_vk128_unpack_f16x2(s2, 1);
    result[14] = hrx_q4_k_wmma_vk128_unpack_f16x2(s3, 0);
    result[15] = hrx_q4_k_wmma_vk128_unpack_f16x2(s3, 1);
    return result;
}

static __device__ __forceinline__ void hrx_q4_k_wmma_vk128_get_scale_min(
        int group, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (group < 4) {
        *d = q[group] & 63;
        *m = q[group + 4] & 63;
    } else {
        *d = (q[group + 4] & 0xF) | ((q[group - 4] >> 6) << 4);
        *m = (q[group + 4] >> 4) | ((q[group] >> 6) << 4);
    }
}

static __device__ __forceinline__ int hrx_q4_k_wmma_vk128_value(
        const hrx_block_q4_K_wmma_vk128_lhs * block,
        int in_block) {
    const int group = in_block >> 5;
    const int q_index = (group >> 1) * 32 + (in_block & 31);
    const uint8_t packed = block->qs[q_index];
    return (packed >> ((group & 1) * 4)) & 0x0f;
}

static __device__ __forceinline__ _Float16 hrx_q4_k_wmma_vk128_load_a_value(
        const hrx_block_q4_K_wmma_vk128_lhs * src0,
        long long row,
        long long k_index,
        long long blocks_per_row) {
    const hrx_block_q4_K_wmma_vk128_lhs * block = src0 + row * blocks_per_row + (k_index >> 8);
    const int in_block = static_cast<int>(k_index & 255);
    const int group = in_block >> 5;
    uint8_t sc = 0;
    uint8_t m = 0;
    hrx_q4_k_wmma_vk128_get_scale_min(group, block->scales, &sc, &m);
    const float d = __half2float(__ushort_as_half(block->d)) * static_cast<float>(sc);
    const float dmin = __half2float(__ushort_as_half(block->dmin)) * static_cast<float>(m);
    return static_cast<_Float16>(
        d * static_cast<float>(hrx_q4_k_wmma_vk128_value(block, in_block)) - dmin);
}

static __device__ __forceinline__ hrx_q4_k_wmma_vk128_half16_vec hrx_q4_k_wmma_vk128_load_a_frag(
        const _Float16 * sh_a,
        int row_tile,
        int k_tile,
        unsigned int lane) {
    constexpr int SHARED_STRIDE = HRX_Q4_K_WMMA_VK128_SHARED_STRIDE;
    const int row = row_tile * 16 + static_cast<int>(lane & 15u);
    const int k_base = k_tile * 16 + static_cast<int>(lane >> 4) * 8;
    return hrx_q4_k_wmma_vk128_duplicate_input(
        sh_a[row * SHARED_STRIDE + k_base + 0],
        sh_a[row * SHARED_STRIDE + k_base + 1],
        sh_a[row * SHARED_STRIDE + k_base + 2],
        sh_a[row * SHARED_STRIDE + k_base + 3],
        sh_a[row * SHARED_STRIDE + k_base + 4],
        sh_a[row * SHARED_STRIDE + k_base + 5],
        sh_a[row * SHARED_STRIDE + k_base + 6],
        sh_a[row * SHARED_STRIDE + k_base + 7]);
}

static __device__ __forceinline__ hrx_q4_k_wmma_vk128_half16_vec hrx_q4_k_wmma_vk128_load_b_frag(
        const _Float16 * sh_b,
        int col_tile,
        int k_tile,
        unsigned int lane) {
    constexpr int SHARED_STRIDE = HRX_Q4_K_WMMA_VK128_SHARED_STRIDE;
    const int col = col_tile * 16 + static_cast<int>(lane & 15u);
    const int k_base = k_tile * 16 + static_cast<int>(lane >> 4) * 8;
    return hrx_q4_k_wmma_vk128_duplicate_input(
        sh_b[col * SHARED_STRIDE + k_base + 0],
        sh_b[col * SHARED_STRIDE + k_base + 1],
        sh_b[col * SHARED_STRIDE + k_base + 2],
        sh_b[col * SHARED_STRIDE + k_base + 3],
        sh_b[col * SHARED_STRIDE + k_base + 4],
        sh_b[col * SHARED_STRIDE + k_base + 5],
        sh_b[col * SHARED_STRIDE + k_base + 6],
        sh_b[col * SHARED_STRIDE + k_base + 7]);
}

static __device__ __forceinline__ hrx_q4_k_wmma_vk128_half16_vec hrx_q4_k_wmma_vk128_load_a_frag_w64(
        const _Float16 * sh_a,
        int row_tile,
        int k_tile,
        unsigned int lane) {
    constexpr int SHARED_STRIDE = HRX_Q4_K_WMMA_VK128_SHARED_STRIDE;
    const int row = row_tile * 16 + static_cast<int>(lane & 15u);
    const int k_base = k_tile * 16;
    hrx_q4_k_wmma_vk128_half16_vec result;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        result[i] = sh_a[row * SHARED_STRIDE + k_base + i];
    }
    return result;
}

static __device__ __forceinline__ hrx_q4_k_wmma_vk128_half16_vec hrx_q4_k_wmma_vk128_load_b_frag_w64(
        const _Float16 * sh_b,
        int col_tile,
        int k_tile,
        unsigned int lane) {
    constexpr int SHARED_STRIDE = HRX_Q4_K_WMMA_VK128_SHARED_STRIDE;
    const int col = col_tile * 16 + static_cast<int>(lane & 15u);
    const int k_base = k_tile * 16;
    hrx_q4_k_wmma_vk128_half16_vec result;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        result[i] = sh_b[col * SHARED_STRIDE + k_base + i];
    }
    return result;
}

static __device__ __forceinline__ void hrx_q4_k_wmma_vk128_append_half4(
        hrx_q4_k_wmma_vk128_half16_vec * result,
        int base,
        hrx_q4_k_wmma_vk128_half4_vec values) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        (*result)[base + i] = values[i];
    }
}

static __device__ __forceinline__ hrx_q4_k_wmma_vk128_half4_vec hrx_q4_k_wmma_vk128_ds_read_b64_h4(
        hrx_q4_k_wmma_vk128_lds_half_ptr ptr) {
    const __attribute__((address_space(3))) uint64_t * lds_ptr =
        (const __attribute__((address_space(3))) uint64_t *) ptr;
    hrx_q4_k_wmma_vk128_half4_vec value;
    asm volatile("ds_read_b64 %0, %1 offset:0\n"
                 : "=v"(value)
                 : "v"(lds_ptr)
                 : "memory");
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    return value;
}

static __device__ __forceinline__ hrx_q4_k_wmma_vk128_half4_vec hrx_q4_k_wmma_vk128_ds_read_b64_h4_nowait(
        hrx_q4_k_wmma_vk128_lds_half_ptr ptr) {
    const __attribute__((address_space(3))) uint64_t * lds_ptr =
        (const __attribute__((address_space(3))) uint64_t *) ptr;
    hrx_q4_k_wmma_vk128_half4_vec value;
    asm volatile("ds_read_b64 %0, %1 offset:0\n"
                 : "=v"(value)
                 : "v"(lds_ptr)
                 : "memory");
    return value;
}

static __device__ __forceinline__ hrx_q4_k_wmma_vk128_half16_vec hrx_q4_k_wmma_vk128_load_a_frag_w64_b64asm(
        hrx_q4_k_wmma_vk128_lds_half_ptr sh_a,
        int row_tile,
        int k_tile,
        unsigned int lane) {
    constexpr int SHARED_STRIDE = HRX_Q4_K_WMMA_VK128_SHARED_STRIDE;
    const int row = row_tile * 16 + static_cast<int>(lane & 15u);
    const int k_base = k_tile * 16;
    hrx_q4_k_wmma_vk128_lds_half_ptr row_ptr = sh_a + row * SHARED_STRIDE + k_base;
    hrx_q4_k_wmma_vk128_half16_vec result;
    hrx_q4_k_wmma_vk128_append_half4(&result, 0, hrx_q4_k_wmma_vk128_ds_read_b64_h4(row_ptr + 0));
    hrx_q4_k_wmma_vk128_append_half4(&result, 4, hrx_q4_k_wmma_vk128_ds_read_b64_h4(row_ptr + 4));
    hrx_q4_k_wmma_vk128_append_half4(&result, 8, hrx_q4_k_wmma_vk128_ds_read_b64_h4(row_ptr + 8));
    hrx_q4_k_wmma_vk128_append_half4(&result, 12, hrx_q4_k_wmma_vk128_ds_read_b64_h4(row_ptr + 12));
    return result;
}

static __device__ __forceinline__ hrx_q4_k_wmma_vk128_half16_vec hrx_q4_k_wmma_vk128_load_a_frag_w64_b64asm_nowait(
        hrx_q4_k_wmma_vk128_lds_half_ptr sh_a,
        int row_tile,
        int k_tile,
        unsigned int lane) {
    constexpr int SHARED_STRIDE = HRX_Q4_K_WMMA_VK128_SHARED_STRIDE;
    const int row = row_tile * 16 + static_cast<int>(lane & 15u);
    const int k_base = k_tile * 16;
    hrx_q4_k_wmma_vk128_lds_half_ptr row_ptr = sh_a + row * SHARED_STRIDE + k_base;
    hrx_q4_k_wmma_vk128_half16_vec result;
    hrx_q4_k_wmma_vk128_append_half4(&result, 0, hrx_q4_k_wmma_vk128_ds_read_b64_h4_nowait(row_ptr + 0));
    hrx_q4_k_wmma_vk128_append_half4(&result, 4, hrx_q4_k_wmma_vk128_ds_read_b64_h4_nowait(row_ptr + 4));
    hrx_q4_k_wmma_vk128_append_half4(&result, 8, hrx_q4_k_wmma_vk128_ds_read_b64_h4_nowait(row_ptr + 8));
    hrx_q4_k_wmma_vk128_append_half4(&result, 12, hrx_q4_k_wmma_vk128_ds_read_b64_h4_nowait(row_ptr + 12));
    return result;
}

static __device__ __forceinline__ hrx_q4_k_wmma_vk128_half16_vec hrx_q4_k_wmma_vk128_load_b_frag_w64_b64asm(
        hrx_q4_k_wmma_vk128_lds_half_ptr sh_b,
        int col_tile,
        int k_tile,
        unsigned int lane) {
    constexpr int SHARED_STRIDE = HRX_Q4_K_WMMA_VK128_SHARED_STRIDE;
    const int col = col_tile * 16 + static_cast<int>(lane & 15u);
    const int k_base = k_tile * 16;
    hrx_q4_k_wmma_vk128_lds_half_ptr col_ptr = sh_b + col * SHARED_STRIDE + k_base;
    hrx_q4_k_wmma_vk128_half16_vec result;
    hrx_q4_k_wmma_vk128_append_half4(&result, 0, hrx_q4_k_wmma_vk128_ds_read_b64_h4(col_ptr + 0));
    hrx_q4_k_wmma_vk128_append_half4(&result, 4, hrx_q4_k_wmma_vk128_ds_read_b64_h4(col_ptr + 4));
    hrx_q4_k_wmma_vk128_append_half4(&result, 8, hrx_q4_k_wmma_vk128_ds_read_b64_h4(col_ptr + 8));
    hrx_q4_k_wmma_vk128_append_half4(&result, 12, hrx_q4_k_wmma_vk128_ds_read_b64_h4(col_ptr + 12));
    return result;
}

static __device__ __forceinline__ hrx_q4_k_wmma_vk128_half16_vec hrx_q4_k_wmma_vk128_load_b_frag_w64_b64asm_nowait(
        hrx_q4_k_wmma_vk128_lds_half_ptr sh_b,
        int col_tile,
        int k_tile,
        unsigned int lane) {
    constexpr int SHARED_STRIDE = HRX_Q4_K_WMMA_VK128_SHARED_STRIDE;
    const int col = col_tile * 16 + static_cast<int>(lane & 15u);
    const int k_base = k_tile * 16;
    hrx_q4_k_wmma_vk128_lds_half_ptr col_ptr = sh_b + col * SHARED_STRIDE + k_base;
    hrx_q4_k_wmma_vk128_half16_vec result;
    hrx_q4_k_wmma_vk128_append_half4(&result, 0, hrx_q4_k_wmma_vk128_ds_read_b64_h4_nowait(col_ptr + 0));
    hrx_q4_k_wmma_vk128_append_half4(&result, 4, hrx_q4_k_wmma_vk128_ds_read_b64_h4_nowait(col_ptr + 4));
    hrx_q4_k_wmma_vk128_append_half4(&result, 8, hrx_q4_k_wmma_vk128_ds_read_b64_h4_nowait(col_ptr + 8));
    hrx_q4_k_wmma_vk128_append_half4(&result, 12, hrx_q4_k_wmma_vk128_ds_read_b64_h4_nowait(col_ptr + 12));
    return result;
}

static __device__ __forceinline__ hrx_q4_k_wmma_vk128_half16_vec hrx_q4_k_wmma_vk128_load_a_frag_w64_h4(
        const _Float16 * sh_a,
        int row_tile,
        int k_tile,
        unsigned int lane) {
    constexpr int SHARED_STRIDE = HRX_Q4_K_WMMA_VK128_SHARED_STRIDE;
    const int row = row_tile * 16 + static_cast<int>(lane & 15u);
    const int k_base = k_tile * 16;
    const hrx_q4_k_wmma_vk128_half4_vec * row_ptr =
        reinterpret_cast<const hrx_q4_k_wmma_vk128_half4_vec *>(sh_a + row * SHARED_STRIDE + k_base);
    hrx_q4_k_wmma_vk128_half16_vec result;
    hrx_q4_k_wmma_vk128_append_half4(&result, 0, row_ptr[0]);
    hrx_q4_k_wmma_vk128_append_half4(&result, 4, row_ptr[1]);
    hrx_q4_k_wmma_vk128_append_half4(&result, 8, row_ptr[2]);
    hrx_q4_k_wmma_vk128_append_half4(&result, 12, row_ptr[3]);
    return result;
}

static __device__ __forceinline__ hrx_q4_k_wmma_vk128_half16_vec hrx_q4_k_wmma_vk128_load_b_frag_w64_h4(
        const _Float16 * sh_b,
        int col_tile,
        int k_tile,
        unsigned int lane) {
    constexpr int SHARED_STRIDE = HRX_Q4_K_WMMA_VK128_SHARED_STRIDE;
    const int col = col_tile * 16 + static_cast<int>(lane & 15u);
    const int k_base = k_tile * 16;
    const hrx_q4_k_wmma_vk128_half4_vec * col_ptr =
        reinterpret_cast<const hrx_q4_k_wmma_vk128_half4_vec *>(sh_b + col * SHARED_STRIDE + k_base);
    hrx_q4_k_wmma_vk128_half16_vec result;
    hrx_q4_k_wmma_vk128_append_half4(&result, 0, col_ptr[0]);
    hrx_q4_k_wmma_vk128_append_half4(&result, 4, col_ptr[1]);
    hrx_q4_k_wmma_vk128_append_half4(&result, 8, col_ptr[2]);
    hrx_q4_k_wmma_vk128_append_half4(&result, 12, col_ptr[3]);
    return result;
}

static __device__ __forceinline__ void hrx_q4_k_wmma_vk128_store_acc_f16_row_major(
        float * dst,
        long long rows_stride,
        long long row0,
        long long col0,
        long long rows,
        long long cols,
        hrx_q4_k_wmma_vk128_half16_vec acc,
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

static __device__ __forceinline__ void hrx_q4_k_wmma_vk128_store_acc_f16_row_major_w64(
        float * dst,
        long long rows_stride,
        long long row0,
        long long col0,
        long long rows,
        long long cols,
        hrx_q4_k_wmma_vk128_half8_vec acc,
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
                static_cast<float>(acc[reg * 2 + HRX_Q4_K_WMMA_VK128_W64_OPSEL]);
        }
    }
}

static __device__ __forceinline__ void hrx_q4_k_wmma_vk128_tile_map(
        int wave,
        int tile_iter,
        int * row_tile,
        int * col_tile) {
    constexpr int ROW_TILES = 8;
    constexpr int WAVE_COUNT = 8;
#if HRX_Q4_K_WMMA_VK128_PAIR64_TILE_MAP
    const int pair = wave >> 1;
    const int lane_wave = wave & 1;
    const int tile = lane_wave + tile_iter * 2;
    const int pair_row = pair & 1;
    const int pair_col = pair >> 1;
    *row_tile = pair_row * 4 + (tile & 3);
    *col_tile = pair_col * 4 + (tile >> 2);
#elif HRX_Q4_K_WMMA_VK128_W64
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

extern "C" __global__ __launch_bounds__(256, 1)
void HRX_Q4_K_WMMA_VK128_EXPORT(
        const hrx_block_q4_K_wmma_vk128_lhs * src0,
        const float * src1,
        float * dst,
        long long k,
        long long rows,
        long long cols) {
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 32;
    constexpr int SHARED_STRIDE = HRX_Q4_K_WMMA_VK128_SHARED_STRIDE;
#if HRX_Q4_K_WMMA_VK128_W64
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

    __shared__ _Float16 sh_a[BM * SHARED_STRIDE];
    __shared__ _Float16 sh_b[BN * SHARED_STRIDE];

    const long long blocks_per_row = k / 256;
    const _Float16 zero = static_cast<_Float16>(0.0f);
#if HRX_Q4_K_WMMA_VK128_W64
    hrx_q4_k_wmma_vk128_half8_vec acc[TILES_PER_WAVE] = {};
#else
    hrx_q4_k_wmma_vk128_half16_vec acc[TILES_PER_WAVE] = {};
#endif

    for (long long k0 = 0; k0 < k; k0 += BK) {
        for (int idx = static_cast<int>(tid); idx < BM * BK; idx += 256) {
            const int r = idx / BK;
            const int kk = idx - r * BK;
            const long long row = row_base + static_cast<long long>(r);
            sh_a[r * SHARED_STRIDE + kk] = row < rows ?
                hrx_q4_k_wmma_vk128_load_a_value(src0, row, k0 + kk, blocks_per_row) : zero;
        }
        for (int idx = static_cast<int>(tid); idx < BN * BK; idx += 256) {
            const int c = idx / BK;
            const int kk = idx - c * BK;
            const long long col = col_base + static_cast<long long>(c);
            sh_b[c * SHARED_STRIDE + kk] = col < cols ? static_cast<_Float16>(src1[col * k + k0 + kk]) : zero;
        }
        __syncthreads();

#if HRX_Q4_K_WMMA_VK128_W64_B64GROUP
        {
            hrx_q4_k_wmma_vk128_lds_half_ptr sh_a_lds =
                (hrx_q4_k_wmma_vk128_lds_half_ptr) sh_a;
            hrx_q4_k_wmma_vk128_lds_half_ptr sh_b_lds =
                (hrx_q4_k_wmma_vk128_lds_half_ptr) sh_b;
            const int wave_row = static_cast<int>(wave & 1u);
            const int wave_col = static_cast<int>(wave >> 1);
#pragma unroll
            for (int k_tile = 0; k_tile < 2; ++k_tile) {
                hrx_q4_k_wmma_vk128_half16_vec a_frag[4];
                hrx_q4_k_wmma_vk128_half16_vec b_frag[4];
#pragma unroll
                for (int row_sub = 0; row_sub < 4; ++row_sub) {
                    a_frag[row_sub] = hrx_q4_k_wmma_vk128_load_a_frag_w64_b64asm(
                        sh_a_lds, wave_row * 4 + row_sub, k_tile, lane);
                }
#pragma unroll
                for (int col_sub = 0; col_sub < 4; ++col_sub) {
                    b_frag[col_sub] = hrx_q4_k_wmma_vk128_load_b_frag_w64_b64asm(
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
                            HRX_Q4_K_WMMA_VK128_W64_OPSEL != 0);
                    }
                }
            }
        }
#else
#pragma unroll
        for (int tile_iter = 0; tile_iter < TILES_PER_WAVE; ++tile_iter) {
            int row_tile = 0;
            int col_tile = 0;
            hrx_q4_k_wmma_vk128_tile_map(static_cast<int>(wave), tile_iter, &row_tile, &col_tile);
#if HRX_Q4_K_WMMA_VK128_W64
#pragma unroll
            for (int k_tile = 0; k_tile < 2; ++k_tile) {
#if HRX_Q4_K_WMMA_VK128_W64_B64ASM
                hrx_q4_k_wmma_vk128_lds_half_ptr sh_a_lds =
                    (hrx_q4_k_wmma_vk128_lds_half_ptr) sh_a;
                hrx_q4_k_wmma_vk128_lds_half_ptr sh_b_lds =
                    (hrx_q4_k_wmma_vk128_lds_half_ptr) sh_b;
                const hrx_q4_k_wmma_vk128_half16_vec a =
                    hrx_q4_k_wmma_vk128_load_a_frag_w64_b64asm(sh_a_lds, row_tile, k_tile, lane);
                const hrx_q4_k_wmma_vk128_half16_vec b =
                    hrx_q4_k_wmma_vk128_load_b_frag_w64_b64asm(sh_b_lds, col_tile, k_tile, lane);
#elif HRX_Q4_K_WMMA_VK128_W64_H4LOAD
                const hrx_q4_k_wmma_vk128_half16_vec a =
                    hrx_q4_k_wmma_vk128_load_a_frag_w64_h4(sh_a, row_tile, k_tile, lane);
                const hrx_q4_k_wmma_vk128_half16_vec b =
                    hrx_q4_k_wmma_vk128_load_b_frag_w64_h4(sh_b, col_tile, k_tile, lane);
#else
                const hrx_q4_k_wmma_vk128_half16_vec a =
                    hrx_q4_k_wmma_vk128_load_a_frag_w64(sh_a, row_tile, k_tile, lane);
                const hrx_q4_k_wmma_vk128_half16_vec b =
                    hrx_q4_k_wmma_vk128_load_b_frag_w64(sh_b, col_tile, k_tile, lane);
#endif
                acc[tile_iter] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(
                    a, b, acc[tile_iter], HRX_Q4_K_WMMA_VK128_W64_OPSEL != 0);
            }
#else
#if HRX_Q4_K_WMMA_VK128_PREFETCH_FRAGS
            hrx_q4_k_wmma_vk128_half16_vec a_frag[2];
            hrx_q4_k_wmma_vk128_half16_vec b_frag[2];
#pragma unroll
            for (int k_tile = 0; k_tile < 2; ++k_tile) {
                a_frag[k_tile] = hrx_q4_k_wmma_vk128_load_a_frag(sh_a, row_tile, k_tile, lane);
                b_frag[k_tile] = hrx_q4_k_wmma_vk128_load_b_frag(sh_b, col_tile, k_tile, lane);
            }
#pragma unroll
            for (int k_tile = 0; k_tile < 2; ++k_tile) {
                acc[tile_iter] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(
                    a_frag[k_tile], b_frag[k_tile], acc[tile_iter], false);
            }
#else
#pragma unroll
            for (int k_tile = 0; k_tile < 2; ++k_tile) {
                const hrx_q4_k_wmma_vk128_half16_vec a =
                    hrx_q4_k_wmma_vk128_load_a_frag(sh_a, row_tile, k_tile, lane);
                const hrx_q4_k_wmma_vk128_half16_vec b =
                    hrx_q4_k_wmma_vk128_load_b_frag(sh_b, col_tile, k_tile, lane);
                acc[tile_iter] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(a, b, acc[tile_iter], false);
            }
#endif
#endif
        }
#endif

        __syncthreads();
    }

#pragma unroll
    for (int tile_iter = 0; tile_iter < TILES_PER_WAVE; ++tile_iter) {
        int row_tile = 0;
        int col_tile = 0;
        hrx_q4_k_wmma_vk128_tile_map(static_cast<int>(wave), tile_iter, &row_tile, &col_tile);
#if HRX_Q4_K_WMMA_VK128_W64
        hrx_q4_k_wmma_vk128_store_acc_f16_row_major_w64(
            dst,
            rows,
            row_base + static_cast<long long>(row_tile * 16),
            col_base + static_cast<long long>(col_tile * 16),
            rows,
            cols,
            acc[tile_iter],
            lane);
#else
        hrx_q4_k_wmma_vk128_store_acc_f16_row_major(
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
}
