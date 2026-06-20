// Q8_0 x Q8_1 x4 prompt full-column probe: push the accepted BN128
// split-qsum packed route from BK_STEP=2 to BK_STEP=4 while keeping wave64.
// This brackets the Vulkan oracle's low-barrier large route without changing
// tile ownership or the packed-Q8_1 scalar-dot dataflow.
#define hrx_mul_mat_vec_q8_0_f32 hrx_mul_mat_vec_q8_0_mmq64x128_splitqsum_bk4_wave64_unused_f32
#define hrx_mul_mat_vec_q8_0_cols8_f32 hrx_mul_mat_vec_q8_0_cols8_mmq64x128_splitqsum_bk4_wave64_unused_f32
#define hrx_mul_mat_vec_q8_0_q8_1_x4_mmq128x32_wg256_f32 hrx_mul_mat_vec_q8_0_q8_1_x4_mmq128x32_mmq64x128_splitqsum_bk4_wave64_unused_f32
#define hrx_mul_mat_vec_q8_0_add_f32 hrx_mul_mat_vec_q8_0_add_mmq64x128_splitqsum_bk4_wave64_unused_f32
#define hrx_mul_mat_vec_q8_0_add_cols8_f32 hrx_mul_mat_vec_q8_0_add_cols8_mmq64x128_splitqsum_bk4_wave64_unused_f32
#define hrx_mul_mat_vec_q8_0_add_q8_1_x4_mmq128x32_wg256_f32 hrx_mul_mat_vec_q8_0_add_q8_1_x4_mmq128x32_mmq64x128_splitqsum_bk4_wave64_unused_f32
#define hrx_mul_mat_vec_q8_0_add_rows4_cols4_f32 hrx_mul_mat_vec_q8_0_add_rows4_cols4_mmq64x128_splitqsum_bk4_wave64_unused_f32

#include "mul_mat_vec_q8_0.hip.cpp"

extern "C" __global__ void hrx_mul_mat_vec_q8_0_q8_1_x4_mmq64x128_splitqsum_bk4_wave64_wg256_f32(
        const hrx_block_q8_0 * src0,
        const hrx_block_q8_1_x4_rhs_q8 * src1,
        float * dst,
        long long k,
        long long rows,
        long long cols) {
    constexpr int BM = 64;
    constexpr int BN = 128;
    constexpr int BK_STEP = 4;
    constexpr int COLS_PER_THREAD = 32;
    constexpr int COL_CHUNK = 16;

    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const int row_lane = static_cast<int>(tid & 63u);
    const int col_lane = static_cast<int>(tid >> 6);
    const long long row = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * BM + row_lane;
    const long long col_base = static_cast<long long>(__builtin_amdgcn_workgroup_id_y()) * BN +
        static_cast<long long>(col_lane * COLS_PER_THREAD);
    const long long col_block_base = static_cast<long long>(__builtin_amdgcn_workgroup_id_y()) * BN;
    if (row >= rows || col_block_base >= cols) {
        return;
    }

    __shared__ int b_qs[BK_STEP][BN][8];
    __shared__ unsigned short b_d[BK_STEP][BN];

    const long long blocks_per_row = k / 32;
    const long long q8_blocks_per_col = k / 32;
    const hrx_block_q8_0 * row_blocks = src0 + row * blocks_per_row;

    float sum[COLS_PER_THREAD] = {};

    for (long long kb_base = 0; kb_base < q8_blocks_per_col; kb_base += BK_STEP) {
        #pragma unroll
        for (int load_idx = static_cast<int>(tid); load_idx < BK_STEP * BN * 8; load_idx += 256) {
            const int k_step = load_idx / (BN * 8);
            const int rem = load_idx - k_step * BN * 8;
            const int c = rem >> 3;
            const int iqs = rem & 7;
            const long long kb = kb_base + k_step;
            if (kb < q8_blocks_per_col && col_block_base + c < cols) {
                const long long linear_block = (col_block_base + c) * q8_blocks_per_col + kb;
                const hrx_block_q8_1_x4_rhs_q8 * rhs = src1 + (linear_block >> 2);
                const int inner = static_cast<int>(linear_block & 3);
                b_qs[k_step][c][iqs] = rhs->qs[inner * 8 + iqs];
                if (iqs == 0) {
                    b_d[k_step][c] = rhs->ds[inner * 2 + 0];
                }
            } else {
                b_qs[k_step][c][iqs] = 0;
                if (iqs == 0) {
                    b_d[k_step][c] = 0;
                }
            }
        }
        __syncthreads();

        #pragma unroll
        for (int k_step = 0; k_step < BK_STEP; ++k_step) {
            const long long kb = kb_base + k_step;
            if (kb >= q8_blocks_per_col) {
                continue;
            }

            const hrx_block_q8_0 * block = row_blocks + kb;
            const float d = __half2float(__ushort_as_half(block->d));

            #pragma unroll
            for (int chunk = 0; chunk < COLS_PER_THREAD; chunk += COL_CHUNK) {
                int qsum[COL_CHUNK] = {};
                #pragma unroll
                for (int iqs = 0; iqs < 8; ++iqs) {
                    const int qpack = hrx_q8_0_pack4(block, iqs);
                    #pragma unroll
                    for (int j = 0; j < COL_CHUNK; ++j) {
                        const int col = chunk + j;
                        qsum[j] += hrx_sdot4_q8_q8_1(qpack, b_qs[k_step][col_lane * COLS_PER_THREAD + col][iqs]);
                    }
                }

                #pragma unroll
                for (int j = 0; j < COL_CHUNK; ++j) {
                    const int col = chunk + j;
                    const int c = col_lane * COLS_PER_THREAD + col;
                    sum[col] += d * __half2float(__ushort_as_half(b_d[k_step][c])) * static_cast<float>(qsum[j]);
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int col = 0; col < COLS_PER_THREAD; ++col) {
        if (col_base + col < cols) {
            const long long out_idx = (col_base + col) * rows + row;
            dst[out_idx] = sum[col];
        }
    }
}
