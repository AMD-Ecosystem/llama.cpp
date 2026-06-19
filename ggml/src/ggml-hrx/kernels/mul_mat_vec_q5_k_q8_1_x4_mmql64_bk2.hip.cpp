#include "mul_mat_vec_q5_k_q8_1_common.hip.inc"

#ifndef HRX_Q5_K_Q8_1_X4_MMQL64_EXPORT
#define HRX_Q5_K_Q8_1_X4_MMQL64_EXPORT hrx_mul_mat_vec_q5_k_q8_1_x4_mmql64x64_bk2_wg256_f32
#endif

#ifndef HRX_Q5_K_Q8_1_X4_MMQL64_BK_STEP
#define HRX_Q5_K_Q8_1_X4_MMQL64_BK_STEP 2
#endif

#ifndef HRX_Q5_K_Q8_1_X4_MMQL64_PREFETCH_B_QUAD
#define HRX_Q5_K_Q8_1_X4_MMQL64_PREFETCH_B_QUAD 0
#endif

#ifndef HRX_Q5_K_Q8_1_X4_MMQL64_PREFETCH_B_PAIR
#define HRX_Q5_K_Q8_1_X4_MMQL64_PREFETCH_B_PAIR 0
#endif

#ifndef HRX_Q5_K_Q8_1_X4_MMQL64_ISSUE_CR_MAJOR
#define HRX_Q5_K_Q8_1_X4_MMQL64_ISSUE_CR_MAJOR 0
#endif

extern "C" __global__ void HRX_Q5_K_Q8_1_X4_MMQL64_EXPORT(
        const hrx_block_q5_K_q8_1_lhs * src0,
        const hrx_block_q8_1_x4_rhs_q5 * src1,
        float * dst,
        long long k, long long rows, long long cols) {
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK_STEP = HRX_Q5_K_Q8_1_X4_MMQL64_BK_STEP;
    constexpr int BLOCK_SIZE = 256;
    constexpr int WARP = 64;
    constexpr int WM = 64;
    constexpr int WN = 16;
    constexpr int WMITER = 1;
    constexpr int TM = 4;
    constexpr int TN = 2;
    constexpr int WNITER = (WM * WN) / (WARP * TM * TN * WMITER);
    constexpr int WSUBM = WM / WMITER;
    constexpr int WSUBN = WN / WNITER;
    constexpr int LOAD_VEC_A = 4;
    constexpr int LOAD_VEC_B = 16;

    static_assert(BK_STEP >= 1, "unexpected Q5 MMQL64 BK step");
    static_assert(WNITER == 2, "unexpected Q5 MMQL64 narrow tile shape");
    static_assert(WSUBM == 64 && WSUBN == 8, "unexpected Q5 MMQL64 BK2 subtile shape");

    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const int warp_i = static_cast<int>(tid / WARP);
    const int tiw = static_cast<int>(tid % WARP);
    const int tiwr = tiw % (WSUBM / TM);
    const int tiwc = tiw / (WSUBM / TM);
    const int warp_r = 0;
    const int warp_c = warp_i;

    __shared__ hrx_q5_k_mmqv_a_cache buf_a[BM * BK_STEP];
    __shared__ hrx_q8_1_mmqv_b_cache buf_b[BN * BK_STEP];

    const long long blocks_per_row = k / 256;
    const long long q8_blocks_per_col = k / 32;
    const long long row_base = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * BM;
    const long long col_base = static_cast<long long>(__builtin_amdgcn_workgroup_id_y()) * BN;
    if (row_base >= rows || col_base >= cols) {
        return;
    }
    const bool full_tile = row_base + BM <= rows && col_base + BN <= cols;

    float sum[WNITER * TM * TN] = {};

    for (long long kb_base = 0; kb_base < q8_blocks_per_col; kb_base += BK_STEP) {
        const int loadr_a = static_cast<int>(tid % (32 / LOAD_VEC_A));
        const int loadc_a = static_cast<int>(tid / (32 / LOAD_VEC_A));
        const int loadstride_a = BLOCK_SIZE * LOAD_VEC_A / 32;
        const int loadr_b = static_cast<int>(tid % (32 / LOAD_VEC_B));
        const int loadc_b = static_cast<int>(tid / (32 / LOAD_VEC_B));
        const int loadstride_b = BLOCK_SIZE * LOAD_VEC_B / 32;

        if (full_tile) {
            for (int r = loadc_a; r < BM; r += loadstride_a) {
                #pragma unroll
                for (int k_step = 0; k_step < BK_STEP; ++k_step) {
                    hrx_q5_k_mmqv_load_a(
                        buf_a,
                        k_step * BM + r,
                        src0,
                        row_base + r,
                        kb_base + k_step,
                        loadr_a,
                        blocks_per_row);
                }
            }
            for (int c = loadc_b; c < BN; c += loadstride_b) {
                #pragma unroll
                for (int k_step = 0; k_step < BK_STEP; ++k_step) {
                    hrx_q5_k_mmqv_load_b(
                        buf_b,
                        k_step * BN + c,
                        src1,
                        col_base + c,
                        kb_base + k_step,
                        loadr_b,
                        q8_blocks_per_col);
                }
            }
        } else {
            for (int r = loadc_a; r < BM; r += loadstride_a) {
                #pragma unroll
                for (int k_step = 0; k_step < BK_STEP; ++k_step) {
                    const int buf_idx = k_step * BM + r;
                    if (row_base + r < rows && kb_base + k_step < q8_blocks_per_col) {
                        hrx_q5_k_mmqv_load_a(
                            buf_a,
                            buf_idx,
                            src0,
                            row_base + r,
                            kb_base + k_step,
                            loadr_a,
                            blocks_per_row);
                    } else {
                        buf_a[buf_idx].qs[loadr_a] = 0;
                        if (loadr_a == 0) {
                            buf_a[buf_idx].d = 0.0f;
                            buf_a[buf_idx].min = 0.0f;
                        }
                    }
                }
            }
            for (int c = loadc_b; c < BN; c += loadstride_b) {
                #pragma unroll
                for (int k_step = 0; k_step < BK_STEP; ++k_step) {
                    const int buf_idx = k_step * BN + c;
                    if (col_base + c < cols && kb_base + k_step < q8_blocks_per_col) {
                        hrx_q5_k_mmqv_load_b(
                            buf_b,
                            buf_idx,
                            src1,
                            col_base + c,
                            kb_base + k_step,
                            loadr_b,
                            q8_blocks_per_col);
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
            hrx_q5_k_mmqv_a_cache cache_a[TM];
            #pragma unroll
            for (int cr = 0; cr < TM; ++cr) {
                cache_a[cr] = buf_a[k_step * BM + warp_r * WM + tiwr * TM + cr];
            }

#if HRX_Q5_K_Q8_1_X4_MMQL64_ISSUE_CR_MAJOR
            #pragma unroll
            for (int cr = 0; cr < TM; ++cr) {
                const hrx_q5_k_mmqv_a_cache cache_a_row = cache_a[cr];
                #pragma unroll
                for (int wsic = 0; wsic < WNITER; ++wsic) {
                    #pragma unroll
                    for (int cc = 0; cc < TN; ++cc) {
                        const hrx_q8_1_mmqv_b_cache cache_b =
                            buf_b[k_step * BN + warp_c * WN + wsic * WSUBN + tiwc * TN + cc];
                        int qsum = 0;
                        #pragma unroll
                        for (int iqs = 0; iqs < 8; ++iqs) {
                            qsum += hrx_sudot4_q5_q8_1(
                                static_cast<uint32_t>(cache_a_row.qs[iqs]), cache_b.qs[iqs]);
                        }
                        sum[(wsic * TN + cc) * TM + cr] +=
                            cache_a_row.d * hrx_q5_k_mmqv_b_cache_d(cache_b) * static_cast<float>(qsum) -
                            cache_a_row.min * hrx_q5_k_mmqv_b_cache_s(cache_b);
                    }
                }
            }
#elif HRX_Q5_K_Q8_1_X4_MMQL64_PREFETCH_B_PAIR
            #pragma unroll
            for (int wsic = 0; wsic < WNITER; ++wsic) {
                hrx_q8_1_mmqv_b_cache cache_b_pair[TN];
                #pragma unroll
                for (int cc = 0; cc < TN; ++cc) {
                    cache_b_pair[cc] =
                        buf_b[k_step * BN + warp_c * WN + wsic * WSUBN + tiwc * TN + cc];
                }
                #pragma unroll
                for (int cc = 0; cc < TN; ++cc) {
                    const hrx_q8_1_mmqv_b_cache cache_b = cache_b_pair[cc];
                    #pragma unroll
                    for (int cr = 0; cr < TM; ++cr) {
                        int qsum = 0;
                        #pragma unroll
                        for (int iqs = 0; iqs < 8; ++iqs) {
                            qsum += hrx_sudot4_q5_q8_1(
                                static_cast<uint32_t>(cache_a[cr].qs[iqs]), cache_b.qs[iqs]);
                        }
                        sum[(wsic * TN + cc) * TM + cr] +=
                            cache_a[cr].d * hrx_q5_k_mmqv_b_cache_d(cache_b) * static_cast<float>(qsum) -
                            cache_a[cr].min * hrx_q5_k_mmqv_b_cache_s(cache_b);
                    }
                }
            }
#elif HRX_Q5_K_Q8_1_X4_MMQL64_PREFETCH_B_QUAD
            hrx_q8_1_mmqv_b_cache cache_b_quad[WNITER][TN];
            #pragma unroll
            for (int wsic = 0; wsic < WNITER; ++wsic) {
                #pragma unroll
                for (int cc = 0; cc < TN; ++cc) {
                    cache_b_quad[wsic][cc] =
                        buf_b[k_step * BN + warp_c * WN + wsic * WSUBN + tiwc * TN + cc];
                }
            }
            #pragma unroll
            for (int wsic = 0; wsic < WNITER; ++wsic) {
                #pragma unroll
                for (int cc = 0; cc < TN; ++cc) {
                    const hrx_q8_1_mmqv_b_cache cache_b = cache_b_quad[wsic][cc];
                    #pragma unroll
                    for (int cr = 0; cr < TM; ++cr) {
                        int qsum = 0;
                        #pragma unroll
                        for (int iqs = 0; iqs < 8; ++iqs) {
                            qsum += hrx_sudot4_q5_q8_1(
                                static_cast<uint32_t>(cache_a[cr].qs[iqs]), cache_b.qs[iqs]);
                        }
                        sum[(wsic * TN + cc) * TM + cr] +=
                            cache_a[cr].d * hrx_q5_k_mmqv_b_cache_d(cache_b) * static_cast<float>(qsum) -
                            cache_a[cr].min * hrx_q5_k_mmqv_b_cache_s(cache_b);
                    }
                }
            }
#else
            #pragma unroll
            for (int wsic = 0; wsic < WNITER; ++wsic) {
                #pragma unroll
                for (int cc = 0; cc < TN; ++cc) {
                    const hrx_q8_1_mmqv_b_cache cache_b =
                        buf_b[k_step * BN + warp_c * WN + wsic * WSUBN + tiwc * TN + cc];
                    #pragma unroll
                    for (int cr = 0; cr < TM; ++cr) {
                        int qsum = 0;
                        #pragma unroll
                        for (int iqs = 0; iqs < 8; ++iqs) {
                            qsum += hrx_sudot4_q5_q8_1(
                                static_cast<uint32_t>(cache_a[cr].qs[iqs]), cache_b.qs[iqs]);
                        }
                        sum[(wsic * TN + cc) * TM + cr] +=
                            cache_a[cr].d * hrx_q5_k_mmqv_b_cache_d(cache_b) * static_cast<float>(qsum) -
                            cache_a[cr].min * hrx_q5_k_mmqv_b_cache_s(cache_b);
                    }
                }
            }
#endif
        }
        __syncthreads();
    }

    #pragma unroll
    for (int wsic = 0; wsic < WNITER; ++wsic) {
        #pragma unroll
        for (int cr = 0; cr < TM; ++cr) {
            const long long row = row_base + warp_r * WM + tiwr * TM + cr;
            #pragma unroll
            for (int cc = 0; cc < TN; ++cc) {
                const long long col = col_base + warp_c * WN + wsic * WSUBN + tiwc * TN + cc;
                if (full_tile || (row < rows && col < cols)) {
                    dst[col * rows + row] = sum[(wsic * TN + cc) * TM + cr];
                }
            }
        }
    }
}

#undef HRX_Q5_K_Q8_1_X4_MMQL64_EXPORT
#undef HRX_Q5_K_Q8_1_X4_MMQL64_BK_STEP
#undef HRX_Q5_K_Q8_1_X4_MMQL64_PREFETCH_B_QUAD
#undef HRX_Q5_K_Q8_1_X4_MMQL64_PREFETCH_B_PAIR
#undef HRX_Q5_K_Q8_1_X4_MMQL64_ISSUE_CR_MAJOR
