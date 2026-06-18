#include "mul_mat_vec_q6_k_q8_1_common.hip.inc"

template <int BN>
static __device__ __forceinline__ void hrx_mul_mat_vec_q6_k_q8_1_x4_mmq64xN_wg256_impl(
        const hrx_block_q6_K_q8_1_lhs * src0,
        const hrx_block_q8_1_x4_rhs_q6 * src1,
        float * dst,
        long long k, long long rows, long long cols) {
    constexpr int BM = 64;
    constexpr int COLS_PER_THREAD = BN / 4;
    static_assert(BN == 64 || BN == 128, "unexpected Q6 direct column tile");

    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const int row_lane = static_cast<int>(tid & 63u);
    const int col_lane = static_cast<int>(tid >> 6);
    const long long row_base = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * BM;
    const long long col_block_base = static_cast<long long>(__builtin_amdgcn_workgroup_id_y()) * BN;
    const long long row = row_base + row_lane;
    const long long col_base = col_block_base + static_cast<long long>(col_lane * COLS_PER_THREAD);
    if (row_base >= rows || col_block_base >= cols) {
        return;
    }

    const bool full_tile = row_base + BM <= rows && col_block_base + BN <= cols;
    const bool row_valid = row < rows;

    __shared__ int b_qs[BN][8];
    __shared__ unsigned short b_d[BN];

    const long long blocks_per_row = k / 256;
    const long long q8_blocks_per_col = k / 32;
    const hrx_block_q6_K_q8_1_lhs * row_blocks =
        src0 + (row_valid ? row : row_base) * blocks_per_row;
    float sum[COLS_PER_THREAD] = {};

    for (long long kb = 0; kb < q8_blocks_per_col; ++kb) {
        #pragma unroll
        for (int load_idx = static_cast<int>(tid); load_idx < BN * 8; load_idx += 256) {
            const int c = load_idx >> 3;
            const int iqs = load_idx & 7;
            if (full_tile || col_block_base + c < cols) {
                const long long linear_block = (col_block_base + c) * q8_blocks_per_col + kb;
                const hrx_block_q8_1_x4_rhs_q6 * rhs = src1 + (linear_block >> 2);
                const int inner = static_cast<int>(linear_block & 3);
                b_qs[c][iqs] = rhs->qs[inner * 8 + iqs];
                if (iqs == 0) {
                    b_d[c] = rhs->ds[inner * 2 + 0];
                }
            } else {
                b_qs[c][iqs] = 0;
                if (iqs == 0) {
                    b_d[c] = 0;
                }
            }
        }
        __syncthreads();

        const hrx_block_q6_K_q8_1_lhs * block = row_blocks + (kb >> 3);
        const int group = static_cast<int>(kb & 7);
        const float d_base = row_valid ? __half2float(__ushort_as_half(block->d)) : 0.0f;
        const float d0 = row_valid ? d_base * static_cast<float>(hrx_q6_k_scale(block, group, 0)) : 0.0f;
        const float d1 = row_valid ? d_base * static_cast<float>(hrx_q6_k_scale(block, group, 16)) : 0.0f;

        int qsum0[COLS_PER_THREAD] = {};
        int qsum1[COLS_PER_THREAD] = {};
        #pragma unroll
        for (int iqs = 0; iqs < 4; ++iqs) {
            const int qpack = row_valid ? hrx_q6_k_pack4(block, group, iqs) : 0;
            #pragma unroll
            for (int col = 0; col < COLS_PER_THREAD; ++col) {
                qsum0[col] += hrx_sdot4_q6_q8_1_qpack(qpack, b_qs[col_lane * COLS_PER_THREAD + col][iqs]);
            }
        }
        #pragma unroll
        for (int iqs = 4; iqs < 8; ++iqs) {
            const int qpack = row_valid ? hrx_q6_k_pack4(block, group, iqs) : 0;
            #pragma unroll
            for (int col = 0; col < COLS_PER_THREAD; ++col) {
                qsum1[col] += hrx_sdot4_q6_q8_1_qpack(qpack, b_qs[col_lane * COLS_PER_THREAD + col][iqs]);
            }
        }

        #pragma unroll
        for (int col = 0; col < COLS_PER_THREAD; ++col) {
            const int c = col_lane * COLS_PER_THREAD + col;
            sum[col] += __half2float(__ushort_as_half(b_d[c])) *
                (d0 * static_cast<float>(qsum0[col]) + d1 * static_cast<float>(qsum1[col]));
        }

        __syncthreads();
    }

    #pragma unroll
    for (int col = 0; col < COLS_PER_THREAD; ++col) {
        if (full_tile || (row_valid && col_base + col < cols)) {
            dst[(col_base + col) * rows + row] = sum[col];
        }
    }
}

extern "C" __global__ void hrx_mul_mat_vec_q6_k_q8_1_x4_mmq64x64_wg256_f32(
        const hrx_block_q6_K_q8_1_lhs * src0,
        const hrx_block_q8_1_x4_rhs_q6 * src1,
        float * dst,
        long long k, long long rows, long long cols) {
    hrx_mul_mat_vec_q6_k_q8_1_x4_mmq64xN_wg256_impl<64>(src0, src1, dst, k, rows, cols);
}

extern "C" __global__ void hrx_mul_mat_vec_q6_k_q8_1_x4_mmq64x128_wg256_f32(
        const hrx_block_q6_K_q8_1_lhs * src0,
        const hrx_block_q8_1_x4_rhs_q6 * src1,
        float * dst,
        long long k, long long rows, long long cols) {
    hrx_mul_mat_vec_q6_k_q8_1_x4_mmq64xN_wg256_impl<128>(src0, src1, dst, k, rows, cols);
}
