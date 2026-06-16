#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <stdint.h>

struct hrx_block_q4_K_q8_1_lhs {
    unsigned short d;
    unsigned short dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};

struct hrx_block_q8_1_x4_rhs_q4 {
    unsigned short ds[8];
    int qs[32];
};

struct hrx_q4_k_mmqv_a_cache {
    int qs[8];
    float d;
    float min;
};

struct hrx_q8_1_mmqv_b_cache_q4 {
    int qs[8];
    float d;
    float s;
};

static __device__ __forceinline__ void hrx_get_scale_min_k4_q4_q8_1(
        int group, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (group < 4) {
        *d = q[group] & 63;
        *m = q[group + 4] & 63;
    } else {
        *d = (q[group + 4] & 0x0F) | ((q[group - 4] >> 6) << 4);
        *m = (q[group + 4] >> 4) | ((q[group] >> 6) << 4);
    }
}

static __device__ __forceinline__ int hrx_sudot4_q4_q8_1(uint32_t qpack, int rpack) {
    return __builtin_amdgcn_sudot4(false, static_cast<int>(qpack), true, rpack, 0, false);
}

static __device__ __forceinline__ uint32_t hrx_q4_k_pack4(
        const hrx_block_q4_K_q8_1_lhs * block,
        int group,
        int iqs) {
    const int qs_base = (group >> 1) * 32 + iqs * 4;
    const uint32_t qs = *reinterpret_cast<const uint32_t *>(block->qs + qs_base);
    return (qs >> ((group & 1) * 4)) & 0x0F0F0F0Fu;
}

static __device__ __forceinline__ void hrx_q4_k_mmqv_load_a(
        hrx_q4_k_mmqv_a_cache * buf_a,
        int buf_idx,
        const hrx_block_q4_K_q8_1_lhs * src0,
        long long row,
        long long kb,
        int iqs,
        long long blocks_per_row) {
    const hrx_block_q4_K_q8_1_lhs * block = src0 + row * blocks_per_row + (kb >> 3);
    const int group = static_cast<int>(kb & 7);
    buf_a[buf_idx].qs[iqs] = static_cast<int>(hrx_q4_k_pack4(block, group, iqs));
    if (iqs == 0) {
        uint8_t sc = 0;
        uint8_t m = 0;
        hrx_get_scale_min_k4_q4_q8_1(group, block->scales, &sc, &m);
        buf_a[buf_idx].d = __half2float(__ushort_as_half(block->d)) * static_cast<float>(sc);
        buf_a[buf_idx].min = __half2float(__ushort_as_half(block->dmin)) * static_cast<float>(m);
    }
}

static __device__ __forceinline__ void hrx_q4_k_mmqv_load_b(
        hrx_q8_1_mmqv_b_cache_q4 * buf_b,
        int buf_idx,
        const hrx_block_q8_1_x4_rhs_q4 * src1,
        long long col,
        long long kb,
        int iqs_vec4,
        long long q8_blocks_per_col) {
    const long long linear_block = col * q8_blocks_per_col + kb;
    const hrx_block_q8_1_x4_rhs_q4 * rhs = src1 + (linear_block >> 2);
    const int inner = static_cast<int>(linear_block & 3);
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        buf_b[buf_idx].qs[iqs_vec4 * 4 + j] = rhs->qs[inner * 8 + iqs_vec4 * 4 + j];
    }
    if (iqs_vec4 == 0) {
        buf_b[buf_idx].d = __half2float(__ushort_as_half(rhs->ds[inner * 2 + 0]));
        buf_b[buf_idx].s = __half2float(__ushort_as_half(rhs->ds[inner * 2 + 1]));
    }
}

extern "C" __global__ void hrx2_mul_mat_vec_q4_k_q8_1_x4_mmql64x32_wg64_u32(
        const hrx_block_q4_K_q8_1_lhs * src0,
        const hrx_block_q8_1_x4_rhs_q4 * src1,
        float * dst,
        uint32_t k,
        uint32_t rows,
        uint32_t cols) {
    constexpr int BM = 64;
    constexpr int BN = 32;
    constexpr int BK_STEP = 1;
    constexpr int BLOCK_SIZE = 64;
    constexpr int WARP = 64;
    constexpr int WM = 64;
    constexpr int WN = 32;
    constexpr int WMITER = 1;
    constexpr int TM = 4;
    constexpr int TN = 2;
    constexpr int WNITER = (WM * WN) / (WARP * TM * TN * WMITER);
    constexpr int WSUBM = WM / WMITER;
    constexpr int WSUBN = WN / WNITER;
    constexpr int LOAD_VEC_A = 4;
    constexpr int LOAD_VEC_B = 16;
    constexpr int LOAD_STRIDE_A = BLOCK_SIZE * LOAD_VEC_A / 32;
    constexpr int LOAD_STRIDE_B = BLOCK_SIZE * LOAD_VEC_B / 32;

    static_assert(WNITER == 4, "unexpected Q4 MMQL tile shape");
    static_assert(WSUBM == 64 && WSUBN == 8, "unexpected Q4 MMQL subtile shape");

    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const int tiw = static_cast<int>(tid % WARP);
    const int tiwr = tiw % (WSUBM / TM);
    const int tiwc = tiw / (WSUBM / TM);

    __shared__ hrx_q4_k_mmqv_a_cache buf_a[BM * BK_STEP];
    __shared__ hrx_q8_1_mmqv_b_cache_q4 buf_b[BN * BK_STEP];

    const long long blocks_per_row = static_cast<long long>(k) / 256;
    const long long q8_blocks_per_col = static_cast<long long>(k) / 32;
    const long long row_base = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * BM;
    const long long col_base = static_cast<long long>(__builtin_amdgcn_workgroup_id_y()) * BN;
    const bool full_tile = row_base + BM <= static_cast<long long>(rows) &&
        col_base + BN <= static_cast<long long>(cols);

    float sum[WNITER * TM * TN] = {};

    for (long long kb_base = 0; kb_base < q8_blocks_per_col; kb_base += BK_STEP) {
        const int loadr_a = static_cast<int>(tid % (32 / LOAD_VEC_A));
        const int loadc_a = static_cast<int>(tid / (32 / LOAD_VEC_A));
        const int loadr_b = static_cast<int>(tid % (32 / LOAD_VEC_B));
        const int loadc_b = static_cast<int>(tid / (32 / LOAD_VEC_B));

        if (full_tile) {
            for (int r = loadc_a; r < BM; r += LOAD_STRIDE_A) {
                #pragma unroll
                for (int k_step = 0; k_step < BK_STEP; ++k_step) {
                    hrx_q4_k_mmqv_load_a(buf_a, k_step * BM + r, src0, row_base + r, kb_base + k_step, loadr_a, blocks_per_row);
                }
            }
            for (int c = loadc_b; c < BN; c += LOAD_STRIDE_B) {
                #pragma unroll
                for (int k_step = 0; k_step < BK_STEP; ++k_step) {
                    hrx_q4_k_mmqv_load_b(buf_b, k_step * BN + c, src1, col_base + c, kb_base + k_step, loadr_b, q8_blocks_per_col);
                }
            }
        } else {
            #pragma unroll
            for (int k_step = 0; k_step < BK_STEP; ++k_step) {
                for (int r = loadc_a; r < BM; r += LOAD_STRIDE_A) {
                    const int buf_idx = k_step * BM + r;
                    if (row_base + r < static_cast<long long>(rows)) {
                        hrx_q4_k_mmqv_load_a(buf_a, buf_idx, src0, row_base + r, kb_base + k_step, loadr_a, blocks_per_row);
                    } else {
                        buf_a[buf_idx].qs[loadr_a] = 0;
                        if (loadr_a == 0) {
                            buf_a[buf_idx].d = 0.0f;
                            buf_a[buf_idx].min = 0.0f;
                        }
                    }
                }
                for (int c = loadc_b; c < BN; c += LOAD_STRIDE_B) {
                    const int buf_idx = k_step * BN + c;
                    if (col_base + c < static_cast<long long>(cols)) {
                        hrx_q4_k_mmqv_load_b(buf_b, buf_idx, src1, col_base + c, kb_base + k_step, loadr_b, q8_blocks_per_col);
                    } else {
                        #pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            buf_b[buf_idx].qs[loadr_b * 4 + j] = 0;
                        }
                        if (loadr_b == 0) {
                            buf_b[buf_idx].d = 0.0f;
                            buf_b[buf_idx].s = 0.0f;
                        }
                    }
                }
            }
        }
        __syncthreads();

        #pragma unroll
        for (int k_step = 0; k_step < BK_STEP; ++k_step) {
            hrx_q4_k_mmqv_a_cache cache_a[TM];
            #pragma unroll
            for (int cr = 0; cr < TM; ++cr) {
                cache_a[cr] = buf_a[k_step * BM + tiwr * TM + cr];
            }

            #pragma unroll
            for (int wsic = 0; wsic < WNITER; ++wsic) {
                hrx_q8_1_mmqv_b_cache_q4 cache_b[TN];
                #pragma unroll
                for (int cc = 0; cc < TN; ++cc) {
                    cache_b[cc] = buf_b[k_step * BN + wsic * WSUBN + tiwc * TN + cc];
                }
                #pragma unroll
                for (int cr = 0; cr < TM; ++cr) {
                    #pragma unroll
                    for (int cc = 0; cc < TN; ++cc) {
                        int qsum = 0;
                        #pragma unroll
                        for (int iqs = 0; iqs < 8; ++iqs) {
                            qsum += hrx_sudot4_q4_q8_1(static_cast<uint32_t>(cache_a[cr].qs[iqs]), cache_b[cc].qs[iqs]);
                        }
                        sum[(wsic * TM + cr) * TN + cc] +=
                            cache_a[cr].d * cache_b[cc].d * static_cast<float>(qsum) -
                            cache_a[cr].min * cache_b[cc].s;
                    }
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int wsic = 0; wsic < WNITER; ++wsic) {
        #pragma unroll
        for (int cr = 0; cr < TM; ++cr) {
            const long long row = row_base + tiwr * TM + cr;
            #pragma unroll
            for (int cc = 0; cc < TN; ++cc) {
                const long long col = col_base + wsic * WSUBN + tiwc * TN + cc;
                if (full_tile || (row < static_cast<long long>(rows) && col < static_cast<long long>(cols))) {
                    dst[col * static_cast<long long>(rows) + row] = sum[(wsic * TM + cr) * TN + cc];
                }
            }
        }
    }
}
