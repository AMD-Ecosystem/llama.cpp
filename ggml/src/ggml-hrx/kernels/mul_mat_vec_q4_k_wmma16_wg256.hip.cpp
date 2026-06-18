#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <stdint.h>

struct hrx_block_q4_K_wmma_lhs {
    unsigned short d;
    unsigned short dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};

typedef _Float16 hrx_q4_k_wmma16_wg256_half16_vec __attribute__((ext_vector_type(16)));

static __device__ __forceinline__ uint32_t hrx_q4_k_wmma16_wg256_pack_f16x2(_Float16 lo, _Float16 hi) {
    union {
        _Float16 h[2];
        uint32_t u;
    } pack;
    pack.h[0] = lo;
    pack.h[1] = hi;
    return pack.u;
}

static __device__ __forceinline__ _Float16 hrx_q4_k_wmma16_wg256_unpack_f16x2(uint32_t bits, int idx) {
    union {
        uint32_t u;
        _Float16 h[2];
    } pack;
    pack.u = bits;
    return pack.h[idx];
}

static __device__ __forceinline__ hrx_q4_k_wmma16_wg256_half16_vec hrx_q4_k_wmma16_wg256_duplicate_input(
        _Float16 x0, _Float16 x1, _Float16 x2, _Float16 x3,
        _Float16 x4, _Float16 x5, _Float16 x6, _Float16 x7) {
    constexpr int SWAP16_CTRL = (16 << 10) | 0x1f;
    const uint32_t p0 = hrx_q4_k_wmma16_wg256_pack_f16x2(x0, x1);
    const uint32_t p1 = hrx_q4_k_wmma16_wg256_pack_f16x2(x2, x3);
    const uint32_t p2 = hrx_q4_k_wmma16_wg256_pack_f16x2(x4, x5);
    const uint32_t p3 = hrx_q4_k_wmma16_wg256_pack_f16x2(x6, x7);
    const uint32_t s0 = static_cast<uint32_t>(__builtin_amdgcn_ds_swizzle(static_cast<int32_t>(p0), SWAP16_CTRL));
    const uint32_t s1 = static_cast<uint32_t>(__builtin_amdgcn_ds_swizzle(static_cast<int32_t>(p1), SWAP16_CTRL));
    const uint32_t s2 = static_cast<uint32_t>(__builtin_amdgcn_ds_swizzle(static_cast<int32_t>(p2), SWAP16_CTRL));
    const uint32_t s3 = static_cast<uint32_t>(__builtin_amdgcn_ds_swizzle(static_cast<int32_t>(p3), SWAP16_CTRL));

    hrx_q4_k_wmma16_wg256_half16_vec result;
    result[0] = x0;
    result[1] = x1;
    result[2] = x2;
    result[3] = x3;
    result[4] = x4;
    result[5] = x5;
    result[6] = x6;
    result[7] = x7;
    result[8] = hrx_q4_k_wmma16_wg256_unpack_f16x2(s0, 0);
    result[9] = hrx_q4_k_wmma16_wg256_unpack_f16x2(s0, 1);
    result[10] = hrx_q4_k_wmma16_wg256_unpack_f16x2(s1, 0);
    result[11] = hrx_q4_k_wmma16_wg256_unpack_f16x2(s1, 1);
    result[12] = hrx_q4_k_wmma16_wg256_unpack_f16x2(s2, 0);
    result[13] = hrx_q4_k_wmma16_wg256_unpack_f16x2(s2, 1);
    result[14] = hrx_q4_k_wmma16_wg256_unpack_f16x2(s3, 0);
    result[15] = hrx_q4_k_wmma16_wg256_unpack_f16x2(s3, 1);
    return result;
}

static __device__ __forceinline__ void hrx_q4_k_wmma16_wg256_get_scale_min(
        int group, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (group < 4) {
        *d = q[group] & 63;
        *m = q[group + 4] & 63;
    } else {
        *d = (q[group + 4] & 0xF) | ((q[group - 4] >> 6) << 4);
        *m = (q[group + 4] >> 4) | ((q[group] >> 6) << 4);
    }
}

static __device__ __forceinline__ int hrx_q4_k_wmma16_wg256_value(
        const hrx_block_q4_K_wmma_lhs * block,
        int in_block) {
    const int group = in_block >> 5;
    const int q_index = (group >> 1) * 32 + (in_block & 31);
    const uint8_t packed = block->qs[q_index];
    return (packed >> ((group & 1) * 4)) & 0x0f;
}

static __device__ __forceinline__ _Float16 hrx_q4_k_wmma16_wg256_load_a_value(
        const hrx_block_q4_K_wmma_lhs * src0,
        long long row,
        long long k_index,
        long long blocks_per_row) {
    const hrx_block_q4_K_wmma_lhs * block = src0 + row * blocks_per_row + (k_index >> 8);
    const int in_block = static_cast<int>(k_index & 255);
    const int group = in_block >> 5;
    uint8_t sc = 0;
    uint8_t m = 0;
    hrx_q4_k_wmma16_wg256_get_scale_min(group, block->scales, &sc, &m);
    const float d = __half2float(__ushort_as_half(block->d)) * static_cast<float>(sc);
    const float dmin = __half2float(__ushort_as_half(block->dmin)) * static_cast<float>(m);
    return static_cast<_Float16>(
        d * static_cast<float>(hrx_q4_k_wmma16_wg256_value(block, in_block)) - dmin);
}

static __device__ __forceinline__ hrx_q4_k_wmma16_wg256_half16_vec hrx_q4_k_wmma16_wg256_load_a_frag(
        const _Float16 * sh_a,
        int row_tile,
        unsigned int lane) {
    constexpr int BK = 16;
    const int row = row_tile * 16 + static_cast<int>(lane & 15u);
    const int k_base = static_cast<int>(lane >> 4) * 8;
    return hrx_q4_k_wmma16_wg256_duplicate_input(
        sh_a[row * BK + k_base + 0],
        sh_a[row * BK + k_base + 1],
        sh_a[row * BK + k_base + 2],
        sh_a[row * BK + k_base + 3],
        sh_a[row * BK + k_base + 4],
        sh_a[row * BK + k_base + 5],
        sh_a[row * BK + k_base + 6],
        sh_a[row * BK + k_base + 7]);
}

static __device__ __forceinline__ hrx_q4_k_wmma16_wg256_half16_vec hrx_q4_k_wmma16_wg256_load_b_frag(
        const _Float16 * sh_b,
        int col_tile,
        unsigned int lane) {
    constexpr int BK = 16;
    const int col = col_tile * 16 + static_cast<int>(lane & 15u);
    const int k_base = static_cast<int>(lane >> 4) * 8;
    return hrx_q4_k_wmma16_wg256_duplicate_input(
        sh_b[col * BK + k_base + 0],
        sh_b[col * BK + k_base + 1],
        sh_b[col * BK + k_base + 2],
        sh_b[col * BK + k_base + 3],
        sh_b[col * BK + k_base + 4],
        sh_b[col * BK + k_base + 5],
        sh_b[col * BK + k_base + 6],
        sh_b[col * BK + k_base + 7]);
}

static __device__ __forceinline__ void hrx_q4_k_wmma16_wg256_store_acc_f16_row_major(
        float * dst,
        long long rows_stride,
        long long row0,
        long long col0,
        long long rows,
        long long cols,
        hrx_q4_k_wmma16_wg256_half16_vec acc,
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

extern "C" __global__ __launch_bounds__(256, 1)
void hrx_mul_mat_vec_q4_k_wmma16x16_f16acc_wg256_f32(
        const hrx_block_q4_K_wmma_lhs * src0,
        const float * src1,
        float * dst,
        long long k,
        long long rows,
        long long cols) {
    constexpr int BM = 64;
    constexpr int BN = 32;
    constexpr int BK = 16;
    constexpr int WAVE = 32;

    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int wave = tid / WAVE;
    const unsigned int lane = tid & 31u;
    const int wave_row = static_cast<int>(wave & 3u);
    const int wave_col = static_cast<int>(wave >> 2);
    const long long row_base = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * BM;
    const long long col_base = static_cast<long long>(__builtin_amdgcn_workgroup_id_y()) * BN;
    const long long row0 = row_base + static_cast<long long>(wave_row * 16);
    const long long col0 = col_base + static_cast<long long>(wave_col * 16);
    if (row_base >= rows || col_base >= cols) {
        return;
    }

    __shared__ _Float16 sh_a[BM * BK];
    __shared__ _Float16 sh_b[BN * BK];

    const long long blocks_per_row = k / 256;
    hrx_q4_k_wmma16_wg256_half16_vec acc = {};
    const _Float16 zero = static_cast<_Float16>(0.0f);

    for (long long k0 = 0; k0 < k; k0 += BK) {
        for (int idx = static_cast<int>(tid); idx < BM * BK; idx += 256) {
            const int r = idx / BK;
            const int kk = idx - r * BK;
            const long long row = row_base + static_cast<long long>(r);
            sh_a[idx] = row < rows ?
                hrx_q4_k_wmma16_wg256_load_a_value(src0, row, k0 + kk, blocks_per_row) : zero;
        }
        for (int idx = static_cast<int>(tid); idx < BN * BK; idx += 256) {
            const int c = idx / BK;
            const int kk = idx - c * BK;
            const long long col = col_base + static_cast<long long>(c);
            sh_b[idx] = col < cols ? static_cast<_Float16>(src1[col * k + k0 + kk]) : zero;
        }
        __syncthreads();

        const hrx_q4_k_wmma16_wg256_half16_vec a =
            hrx_q4_k_wmma16_wg256_load_a_frag(sh_a, wave_row, lane);
        const hrx_q4_k_wmma16_wg256_half16_vec b =
            hrx_q4_k_wmma16_wg256_load_b_frag(sh_b, wave_col, lane);
        acc = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(a, b, acc, false);

        __syncthreads();
    }

    hrx_q4_k_wmma16_wg256_store_acc_f16_row_major(dst, rows, row0, col0, rows, cols, acc, lane);
}
