#include "../../../ggml-hrx/kernels/mul_mat_vec_q6_k_q8_1_wave64.hip.cpp"

extern "C" __global__ void hrx2_mul_mat_vec_q6_k_q8_1_x4_mmql64x128_wg256_u32(
        const hrx_block_q6_K_q8_1_lhs * src0,
        const hrx_block_q8_1_x4_rhs_q6 * src1,
        float * dst,
        uint32_t k,
        uint32_t rows,
        uint32_t cols) {
    hrx_mul_mat_vec_q6_k_q8_1_x4_mmql_wg256_impl<64, 128>(
        src0,
        src1,
        dst,
        static_cast<long long>(k),
        static_cast<long long>(rows),
        static_cast<long long>(cols));
}

extern "C" __global__ void hrx2_mul_mat_vec_q6_k_q8_1_x4_vkm64x64_wg128_w32_u32(
        const hrx_block_q6_K_q8_1_lhs * src0,
        const hrx_block_q8_1_x4_rhs_q6 * src1,
        float * dst,
        uint32_t k,
        uint32_t rows,
        uint32_t cols) {
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK_STEP = 4;
    constexpr int BLOCK_SIZE = 128;
    constexpr int WARP = 32;
    constexpr int WM = 32;
    constexpr int WN = 32;
    constexpr int WMITER = 1;
    constexpr int TM = 2;
    constexpr int TN = 2;
    constexpr int WNITER = (WM * WN) / (WARP * TM * TN * WMITER);
    constexpr int WSUBM = WM / WMITER;
    constexpr int WSUBN = WN / WNITER;
    constexpr int LOAD_VEC_A = 4;
    constexpr int LOAD_VEC_B = 16;
    constexpr int LOAD_STRIDE_A = BLOCK_SIZE * LOAD_VEC_A / 32;
    constexpr int LOAD_STRIDE_B = BLOCK_SIZE * LOAD_VEC_B / 32;

    static_assert(WNITER == 8, "unexpected Q6 VKM64x64 W32 WNITER");
    static_assert(WSUBM == 32 && WSUBN == 4, "unexpected Q6 VKM64x64 W32 subtile shape");

    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const int warp_i = static_cast<int>(tid / WARP);
    const int tiw = static_cast<int>(tid % WARP);
    const int tiwr = tiw % (WSUBM / TM);
    const int tiwc = tiw / (WSUBM / TM);
    const int warp_r = warp_i % (BM / WM);
    const int warp_c = warp_i / (BM / WM);

    __shared__ hrx_q6_k_mmqv_a_cache buf_a[BM * BK_STEP];
    __shared__ hrx_q8_1_mmqv_b_cache_q6 buf_b[BN * BK_STEP];

    const long long kk = static_cast<long long>(k);
    const long long nrows = static_cast<long long>(rows);
    const long long ncols = static_cast<long long>(cols);
    const long long blocks_per_row = kk / 256;
    const long long q8_blocks_per_col = kk / 32;
    const long long row_base = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * BM;
    const long long col_base = static_cast<long long>(__builtin_amdgcn_workgroup_id_y()) * BN;
    const bool full_tile = row_base + BM <= nrows && col_base + BN <= ncols;

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
                    hrx_q6_k_mmqv_load_a(
                        buf_a,
                        k_step * BM + r,
                        src0,
                        row_base + r,
                        kb_base + k_step,
                        loadr_a,
                        blocks_per_row);
                }
            }
            for (int c = loadc_b; c < BN; c += LOAD_STRIDE_B) {
                #pragma unroll
                for (int k_step = 0; k_step < BK_STEP; ++k_step) {
                    hrx_q6_k_mmqv_load_b(
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
            #pragma unroll
            for (int k_step = 0; k_step < BK_STEP; ++k_step) {
                for (int r = loadc_a; r < BM; r += LOAD_STRIDE_A) {
                    const int buf_idx = k_step * BM + r;
                    if (row_base + r < nrows) {
                        hrx_q6_k_mmqv_load_a(
                            buf_a,
                            buf_idx,
                            src0,
                            row_base + r,
                            kb_base + k_step,
                            loadr_a,
                            blocks_per_row);
                    } else {
                        buf_a[buf_idx].qs[loadr_a] = 0;
                        if (loadr_a == 0 || loadr_a == 4) {
                            buf_a[buf_idx].d[loadr_a >> 2] = 0.0f;
                        }
                    }
                }
                for (int c = loadc_b; c < BN; c += LOAD_STRIDE_B) {
                    const int buf_idx = k_step * BN + c;
                    if (col_base + c < ncols) {
                        hrx_q6_k_mmqv_load_b(
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
                        }
                    }
                }
            }
        }
        __syncthreads();

        #pragma unroll
        for (int k_step = 0; k_step < BK_STEP; ++k_step) {
            hrx_q6_k_mmqv_a_cache cache_a[TM];
            #pragma unroll
            for (int cr = 0; cr < TM; ++cr) {
                cache_a[cr] = buf_a[k_step * BM + warp_r * WM + tiwr * TM + cr];
            }

            #pragma unroll
            for (int wsic = 0; wsic < WNITER; ++wsic) {
                hrx_q8_1_mmqv_b_cache_q6 cache_b[TN];
                #pragma unroll
                for (int cc = 0; cc < TN; ++cc) {
                    cache_b[cc] = buf_b[k_step * BN + warp_c * WN + wsic * WSUBN + tiwc * TN + cc];
                }
                #pragma unroll
                for (int cr = 0; cr < TM; ++cr) {
                    #pragma unroll
                    for (int cc = 0; cc < TN; ++cc) {
                        int qsum0 = 0;
                        int qsum1 = 0;
                        #pragma unroll
                        for (int iqs = 0; iqs < 4; ++iqs) {
                            qsum0 += hrx_sdot4_q6_q8_1_qpack(cache_a[cr].qs[iqs], cache_b[cc].qs[iqs]);
                        }
                        #pragma unroll
                        for (int iqs = 4; iqs < 8; ++iqs) {
                            qsum1 += hrx_sdot4_q6_q8_1_qpack(cache_a[cr].qs[iqs], cache_b[cc].qs[iqs]);
                        }
                        sum[(wsic * TM + cr) * TN + cc] += cache_b[cc].d *
                            (cache_a[cr].d[0] * static_cast<float>(qsum0) +
                             cache_a[cr].d[1] * static_cast<float>(qsum1));
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
            const long long row = row_base + warp_r * WM + tiwr * TM + cr;
            #pragma unroll
            for (int cc = 0; cc < TN; ++cc) {
                const long long col = col_base + warp_c * WN + wsic * WSUBN + tiwc * TN + cc;
                if (full_tile || (row < nrows && col < ncols)) {
                    dst[col * nrows + row] = sum[(wsic * TM + cr) * TN + cc];
                }
            }
        }
    }
}
