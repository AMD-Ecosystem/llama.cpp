// Q8_0 x Q8_1 x4 prompt probe derived from the Vulkan AMD medium integer-MMQ
// schedule: BM64/BN64/BK32, BLOCK_SIZE256, WM16/WN16, WMITER2, TM2/TN2,
// logical WARP16, cooperative A+B staging, and 16 accumulators per lane.
// The exact Vulkan BK_STEP=4 spelling spills on gfx1151 HIP, so this pressure
// pivot keeps the ownership map but stages one BK32 block at a time.
#define hrx_mul_mat_vec_q8_0_f32 hrx_mul_mat_vec_q8_0_mmq64x64_medium_unused_f32
#define hrx_mul_mat_vec_q8_0_cols8_f32 hrx_mul_mat_vec_q8_0_cols8_mmq64x64_medium_unused_f32
#define hrx_mul_mat_vec_q8_0_q8_1_x4_mmq128x32_wg256_f32 hrx_mul_mat_vec_q8_0_q8_1_x4_mmq128x32_mmq64x64_medium_unused_f32
#define hrx_mul_mat_vec_q8_0_add_f32 hrx_mul_mat_vec_q8_0_add_mmq64x64_medium_unused_f32
#define hrx_mul_mat_vec_q8_0_add_cols8_f32 hrx_mul_mat_vec_q8_0_add_cols8_mmq64x64_medium_unused_f32
#define hrx_mul_mat_vec_q8_0_add_q8_1_x4_mmq128x32_wg256_f32 hrx_mul_mat_vec_q8_0_add_q8_1_x4_mmq128x32_mmq64x64_medium_unused_f32
#define hrx_mul_mat_vec_q8_0_add_rows4_cols4_f32 hrx_mul_mat_vec_q8_0_add_rows4_cols4_mmq64x64_medium_unused_f32

#include "mul_mat_vec_q8_0.hip.cpp"

struct hrx_q8_0_medium_a_cache {
    int qs[8];
    float d;
};

struct hrx_q8_1_medium_b_cache_q8 {
    int qs[8];
    float d;
};

static __device__ __forceinline__ void hrx_q8_0_medium_load_a(
        hrx_q8_0_medium_a_cache * buf_a,
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

static __device__ __forceinline__ void hrx_q8_0_medium_load_b(
        hrx_q8_1_medium_b_cache_q8 * buf_b,
        int buf_idx,
        const hrx_block_q8_1_x4_rhs_q8 * src1,
        long long col,
        long long kb,
        int iqs_vec4,
        long long q8_blocks_per_col) {
    const long long linear_block = col * q8_blocks_per_col + kb;
    const hrx_block_q8_1_x4_rhs_q8 * rhs = src1 + (linear_block >> 2);
    const int inner = static_cast<int>(linear_block & 3);
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        buf_b[buf_idx].qs[iqs_vec4 * 4 + j] = rhs->qs[inner * 8 + iqs_vec4 * 4 + j];
    }
    if (iqs_vec4 == 0) {
        buf_b[buf_idx].d = __half2float(__ushort_as_half(rhs->ds[inner * 2 + 0]));
    }
}

static __device__ __forceinline__ float hrx_q8_0_medium_dot(
        const hrx_q8_0_medium_a_cache & a,
        const hrx_q8_1_medium_b_cache_q8 & b) {
    int qsum = 0;
    #pragma unroll
    for (int iqs = 0; iqs < 8; ++iqs) {
        qsum += hrx_sdot4_q8_q8_1(a.qs[iqs], b.qs[iqs]);
    }
    return a.d * b.d * static_cast<float>(qsum);
}

extern "C" __global__ void hrx_mul_mat_vec_q8_0_q8_1_x4_mmq64x64_medium_wg256_f32(
        const hrx_block_q8_0 * src0,
        const hrx_block_q8_1_x4_rhs_q8 * src1,
        float * dst,
        long long k,
        long long rows,
        long long cols) {
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK_STEP = 1;
    constexpr int BLOCK_SIZE = 256;
    constexpr int WARP = 16;
    constexpr int WM = 16;
    constexpr int WN = 16;
    constexpr int WMITER = 2;
    constexpr int TM = 2;
    constexpr int TN = 2;
    constexpr int WNITER = (WM * WN) / (WARP * TM * TN * WMITER);
    constexpr int WSUBM = WM / WMITER;
    constexpr int WSUBN = WN / WNITER;
    constexpr int LOAD_VEC_A = 4;
    constexpr int LOAD_VEC_B = 16;

    static_assert(WNITER == 2, "unexpected Vulkan medium Q8_0 MMQ WNITER");
    static_assert(WSUBM == 8 && WSUBN == 8, "unexpected Vulkan medium Q8_0 MMQ subtile");

    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const int warp_i = static_cast<int>(tid / WARP);
    const int tiw = static_cast<int>(tid % WARP);
    const int tiwr = tiw % (WSUBM / TM);
    const int tiwc = tiw / (WSUBM / TM);
    const int warp_r = warp_i % (BM / WM);
    const int warp_c = warp_i / (BM / WM);

    __shared__ hrx_q8_0_medium_a_cache buf_a[BM * BK_STEP];
    __shared__ hrx_q8_1_medium_b_cache_q8 buf_b[BN * BK_STEP];

    const long long blocks_per_row = k / 32;
    const long long q8_blocks_per_col = k / 32;
    const long long row_base = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * BM;
    const long long col_base = static_cast<long long>(__builtin_amdgcn_workgroup_id_y()) * BN;
    const bool full_tile = row_base + BM <= rows && col_base + BN <= cols;

    float sum[WMITER * TM * WNITER * TN] = {};

    for (long long kb_base = 0; kb_base < q8_blocks_per_col; kb_base += BK_STEP) {
        const int loadr_a = static_cast<int>(tid % (32 / LOAD_VEC_A));
        const int loadc_a = static_cast<int>(tid / (32 / LOAD_VEC_A));
        const int loadstride_a = BLOCK_SIZE * LOAD_VEC_A / 32;
        const int loadr_b = static_cast<int>(tid % (32 / LOAD_VEC_B));
        const int loadc_b = static_cast<int>(tid / (32 / LOAD_VEC_B));
        const int loadstride_b = BLOCK_SIZE * LOAD_VEC_B / 32;

        #pragma unroll
        for (int k_step = 0; k_step < BK_STEP; ++k_step) {
            for (int r = loadc_a; r < BM; r += loadstride_a) {
                const int buf_idx = k_step * BM + r;
                if ((full_tile || row_base + r < rows) && kb_base + k_step < q8_blocks_per_col) {
                    hrx_q8_0_medium_load_a(
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
                if ((full_tile || col_base + c < cols) && kb_base + k_step < q8_blocks_per_col) {
                    hrx_q8_0_medium_load_b(
                        buf_b, buf_idx, src1, col_base + c,
                        kb_base + k_step, loadr_b, q8_blocks_per_col);
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
        __syncthreads();

        #pragma unroll
        for (int k_step = 0; k_step < BK_STEP; ++k_step) {
            hrx_q8_0_medium_a_cache cache_a[WMITER * TM];
            #pragma unroll
            for (int wsir = 0; wsir < WMITER; ++wsir) {
                #pragma unroll
                for (int cr = 0; cr < TM; ++cr) {
                    const int reg_ib = wsir * TM + cr;
                    const int buf_ib = warp_r * WM + wsir * WSUBM + tiwr * TM + cr;
                    cache_a[reg_ib] = buf_a[k_step * BM + buf_ib];
                }
            }

            #pragma unroll
            for (int wsic = 0; wsic < WNITER; ++wsic) {
                #pragma unroll
                for (int cc = 0; cc < TN; ++cc) {
                    const int ib = k_step * BN + warp_c * WN + wsic * WSUBN + tiwc * TN + cc;
                    const hrx_q8_1_medium_b_cache_q8 cache_b = buf_b[ib];
                    #pragma unroll
                    for (int wsir = 0; wsir < WMITER; ++wsir) {
                        #pragma unroll
                        for (int cr = 0; cr < TM; ++cr) {
                            const int cache_a_idx = wsir * TM + cr;
                            const int sum_idx = (wsic * TN + cc) * (WMITER * TM) + wsir * TM + cr;
                            sum[sum_idx] += hrx_q8_0_medium_dot(cache_a[cache_a_idx], cache_b);
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int wsic = 0; wsic < WNITER; ++wsic) {
        #pragma unroll
        for (int wsir = 0; wsir < WMITER; ++wsir) {
            const long long row_warp = row_base + warp_r * WM + wsir * WSUBM + tiwr * TM;
            const long long col_warp = col_base + warp_c * WN + wsic * WSUBN + tiwc * TN;
            #pragma unroll
            for (int cc = 0; cc < TN; ++cc) {
                #pragma unroll
                for (int cr = 0; cr < TM; ++cr) {
                    const long long row = row_warp + cr;
                    const long long col = col_warp + cc;
                    const int sum_idx = (wsic * TN + cc) * WMITER * TM + wsir * TM + cr;
                    if (full_tile || (row < rows && col < cols)) {
                        dst[col * rows + row] = sum[sum_idx];
                    }
                }
            }
        }
    }
}
