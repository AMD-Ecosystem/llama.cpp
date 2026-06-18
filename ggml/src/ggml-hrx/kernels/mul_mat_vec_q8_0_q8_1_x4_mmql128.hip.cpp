// Q8_0 x Q8_1 x4 prompt probe: Vulkan-large-style 128x128 output
// ownership with cooperative A+B staging.
#define hrx_mul_mat_vec_q8_0_f32 hrx_mul_mat_vec_q8_0_mmql128_unused_f32
#define hrx_mul_mat_vec_q8_0_cols8_f32 hrx_mul_mat_vec_q8_0_cols8_mmql128_unused_f32
#define hrx_mul_mat_vec_q8_0_q8_1_x4_mmq128x32_wg256_f32 hrx_mul_mat_vec_q8_0_q8_1_x4_mmq128x32_mmql128_unused_f32
#define hrx_mul_mat_vec_q8_0_add_f32 hrx_mul_mat_vec_q8_0_add_mmql128_unused_f32
#define hrx_mul_mat_vec_q8_0_add_cols8_f32 hrx_mul_mat_vec_q8_0_add_cols8_mmql128_unused_f32
#define hrx_mul_mat_vec_q8_0_add_q8_1_x4_mmq128x32_wg256_f32 hrx_mul_mat_vec_q8_0_add_q8_1_x4_mmq128x32_mmql128_unused_f32
#define hrx_mul_mat_vec_q8_0_add_rows4_cols4_f32 hrx_mul_mat_vec_q8_0_add_rows4_cols4_mmql128_unused_f32

#include "mul_mat_vec_q8_0.hip.cpp"

struct hrx_q8_0_mmql_a_cache {
    int qs[8];
    float d;
};

struct hrx_q8_1_mmql_b_cache_q8 {
    int qs[8];
    float d;
};

static __device__ __forceinline__ void hrx_q8_0_mmql_load_a(
        hrx_q8_0_mmql_a_cache * buf_a,
        int buf_idx,
        const hrx_block_q8_0 * src0,
        long long row,
        long long kb,
        int iqs,
        long long blocks_per_row) {
    const hrx_block_q8_0 * block = src0 + row * blocks_per_row + kb;
    buf_a[buf_idx].qs[iqs] = hrx_q8_0_pack4(block, iqs);
    if (iqs == 0) {
        buf_a[buf_idx].d = __half2float(__ushort_as_half(block->d));
    }
}

static __device__ __forceinline__ void hrx_q8_0_mmql_load_b(
        hrx_q8_1_mmql_b_cache_q8 * buf_b,
        int buf_idx,
        const hrx_block_q8_1_x4_rhs_q8 * src1,
        long long col,
        long long kb,
        int iqs,
        long long q8_blocks_per_col) {
    const long long linear_block = col * q8_blocks_per_col + kb;
    const hrx_block_q8_1_x4_rhs_q8 * rhs = src1 + (linear_block >> 2);
    const int inner = static_cast<int>(linear_block & 3);
    buf_b[buf_idx].qs[iqs] = rhs->qs[inner * 8 + iqs];
    if (iqs == 0) {
        buf_b[buf_idx].d = __half2float(__ushort_as_half(rhs->ds[inner * 2 + 0]));
    }
}

extern "C" __global__ void hrx_mul_mat_vec_q8_0_q8_1_x4_mmql128x128_wg256_f32(
        const hrx_block_q8_0 * src0,
        const hrx_block_q8_1_x4_rhs_q8 * src1,
        float * dst,
        long long k,
        long long rows,
        long long cols) {
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK_STEP = 1;
    constexpr int BLOCK_SIZE = 256;
    constexpr int WARP = 64;
    constexpr int WM = 64;
    constexpr int WN = 64;
    constexpr int WMITER = 1;
    constexpr int TM = 4;
    constexpr int TN = 2;
    constexpr int WNITER = (WM * WN) / (WARP * TM * TN * WMITER);
    constexpr int WSUBM = WM / WMITER;
    constexpr int WSUBN = WN / WNITER;
    constexpr int LOAD_VEC_A = 8;
    constexpr int LOAD_VEC_B = 8;

    static_assert(WNITER == 8, "unexpected Vulkan large Q8_0 MMQ tile shape");
    static_assert(WSUBM == 64 && WSUBN == 8, "unexpected Vulkan large Q8_0 MMQ subtile shape");

    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const int warp_i = static_cast<int>(tid / WARP);
    const int tiw = static_cast<int>(tid % WARP);
    const int tiwr = tiw % (WSUBM / TM);
    const int tiwc = tiw / (WSUBM / TM);
    const int warp_r = warp_i % (BM / WM);
    const int warp_c = warp_i / (BM / WM);

    __shared__ hrx_q8_0_mmql_a_cache buf_a[BM * BK_STEP];
    __shared__ hrx_q8_1_mmql_b_cache_q8 buf_b[BN * BK_STEP];

    const long long blocks_per_row = k / 32;
    const long long q8_blocks_per_col = k / 32;
    const long long row_base = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * BM;
    const long long col_base = static_cast<long long>(__builtin_amdgcn_workgroup_id_y()) * BN;
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
                    hrx_q8_0_mmql_load_a(
                        buf_a, k_step * BM + r, src0, row_base + r,
                        kb_base + k_step, loadr_a, blocks_per_row);
                }
            }
            for (int c = loadc_b; c < BN; c += loadstride_b) {
                #pragma unroll
                for (int k_step = 0; k_step < BK_STEP; ++k_step) {
                    hrx_q8_0_mmql_load_b(
                        buf_b, k_step * BN + c, src1, col_base + c,
                        kb_base + k_step, loadr_b, q8_blocks_per_col);
                }
            }
        } else {
            #pragma unroll
            for (int k_step = 0; k_step < BK_STEP; ++k_step) {
                for (int r = loadc_a; r < BM; r += loadstride_a) {
                    const int buf_idx = k_step * BM + r;
                    if (row_base + r < rows) {
                        hrx_q8_0_mmql_load_a(
                            buf_a, buf_idx, src0, row_base + r,
                            kb_base + k_step, loadr_a, blocks_per_row);
                    } else {
                        buf_a[buf_idx].qs[loadr_a] = 0;
                        if (loadr_a == 0) {
                            buf_a[buf_idx].d = 0.0f;
                        }
                    }
                }
            }

            #pragma unroll
            for (int k_step = 0; k_step < BK_STEP; ++k_step) {
                for (int c = loadc_b; c < BN; c += loadstride_b) {
                    const int buf_idx = k_step * BN + c;
                    if (col_base + c < cols) {
                        hrx_q8_0_mmql_load_b(
                            buf_b, buf_idx, src1, col_base + c,
                            kb_base + k_step, loadr_b, q8_blocks_per_col);
                    } else {
                        buf_b[buf_idx].qs[loadr_b] = 0;
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
            hrx_q8_0_mmql_a_cache cache_a[TM];
            #pragma unroll
            for (int cr = 0; cr < TM; ++cr) {
                cache_a[cr] = buf_a[k_step * BM + warp_r * WM + tiwr * TM + cr];
            }

            #pragma unroll
            for (int wsic = 0; wsic < WNITER; ++wsic) {
                #pragma unroll
                for (int cc = 0; cc < TN; ++cc) {
                    hrx_q8_1_mmql_b_cache_q8 cache_b =
                        buf_b[k_step * BN + warp_c * WN + wsic * WSUBN + tiwc * TN + cc];
                    #pragma unroll
                    for (int cr = 0; cr < TM; ++cr) {
                        int qsum = 0;
                        #pragma unroll
                        for (int iqs = 0; iqs < 8; ++iqs) {
                            qsum += hrx_sdot4_q8_q8_1(cache_a[cr].qs[iqs], cache_b.qs[iqs]);
                        }
                        sum[(wsic * TN + cc) * TM + cr] +=
                            cache_a[cr].d * cache_b.d * static_cast<float>(qsum);
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
                if (full_tile || (row < rows && col < cols)) {
                    dst[col * rows + row] = sum[(wsic * TN + cc) * TM + cr];
                }
            }
        }
    }
}
