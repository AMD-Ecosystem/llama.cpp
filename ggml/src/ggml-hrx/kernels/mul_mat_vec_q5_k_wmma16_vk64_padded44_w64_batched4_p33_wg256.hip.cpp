#include "mul_mat_vec_q5_k_wmma16_vk64_padded44_w64_full64_wg256.hip.cpp"

static __device__ __forceinline__ void hrx_q5_k_wmma_vk64_batched4_p33_store_slot(
        __amdgpu_buffer_rsrc_t dst_rsrc,
        long long rows_stride,
        long long row_base,
        long long col_base,
        long long rows,
        long long cols,
        const hrx_q5_k_wmma_vk128_half8_vec * acc,
        int group_base,
        int group,
        int slot,
        unsigned int lane) {
    const int row_lane = static_cast<int>(lane >> 4);
    const int col_lane = static_cast<int>(lane & 15u);
    const int local_group = group - group_base;
    const long long row = row_base + static_cast<long long>((group & 3) * 16 + row_lane + slot * 4);
    const long long col = col_base + static_cast<long long>(((group >> 2) & 3) * 16 + col_lane);
    if (row < rows && col < cols) {
        hrx_q5_k_wmma_vk128_buffer_store_f32(
            dst_rsrc,
            col * rows_stride + row,
            static_cast<float>(acc[local_group][slot * 2 + HRX_Q5_K_WMMA_VK128_W64_OPSEL]));
    }
}

extern "C" __global__ __launch_bounds__(256, 1)
void hrx_mul_mat_vec_q5_k_wmma16x16_vk64_padded44_w64_batched4_p33_f16acc_wg256_f32(
        const hrx_block_q5_K_wmma_vk128_lhs * src0,
        const float * src1,
        float * dst,
        long long k,
        long long rows,
        long long cols) {
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 32;
    constexpr int SHARED_STRIDE = 44;

    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int wave = tid >> 6u;
    const unsigned int lane = tid & 63u;
    const long long row_base = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * BM;
    const long long col_base = static_cast<long long>(__builtin_amdgcn_workgroup_id_y()) * BN;
    if (row_base >= rows || col_base >= cols) {
        return;
    }

    const __amdgpu_buffer_rsrc_t dst_rsrc = hrx_q5_k_wmma_vk128_make_dst_rsrc(dst);
    __shared__ _Float16 sh_a[BM * SHARED_STRIDE];
    __shared__ _Float16 sh_b[BN * SHARED_STRIDE];

    const long long blocks_per_row = k / 256;
    const _Float16 zero = static_cast<_Float16>(0.0f);

#pragma unroll
    for (int group_base = 0; group_base < 12; group_base += 4) {
        hrx_q5_k_wmma_vk128_half8_vec acc[4] = {};

        for (long long k0 = 0; k0 < k; k0 += BK) {
            for (int idx = static_cast<int>(tid); idx < BM * BK; idx += 256) {
                const int r = idx / BK;
                const int kk = idx - r * BK;
                const long long row = row_base + static_cast<long long>(r);
                sh_a[r * SHARED_STRIDE + kk] = row < rows ?
                    hrx_q5_k_wmma_vk128_load_a_value(src0, row, k0 + kk, blocks_per_row) : zero;
            }
            for (int idx = static_cast<int>(tid); idx < BN * BK; idx += 256) {
                const int c = idx / BK;
                const int kk = idx - c * BK;
                const long long col = col_base + static_cast<long long>(c);
                sh_b[c * SHARED_STRIDE + kk] = col < cols ? static_cast<_Float16>(src1[col * k + k0 + kk]) : zero;
            }
            __syncthreads();

            if (wave == 0) {
                hrx_q5_k_wmma_vk128_lds_half_ptr sh_a_lds =
                    (hrx_q5_k_wmma_vk128_lds_half_ptr) sh_a;
                hrx_q5_k_wmma_vk128_lds_half_ptr sh_b_lds =
                    (hrx_q5_k_wmma_vk128_lds_half_ptr) sh_b;
                hrx_q5_k_wmma_vk128_half16_vec a_frag[2][4];
                hrx_q5_k_wmma_vk128_half16_vec b_frag[2][4];
#pragma unroll
                for (int k_tile = 0; k_tile < 2; ++k_tile) {
#pragma unroll
                    for (int row_sub = 0; row_sub < 4; ++row_sub) {
                        a_frag[k_tile][row_sub] =
                            hrx_q5_k_wmma_vk128_load_a_frag_w64_b64asm_nowait(
                                sh_a_lds, row_sub, k_tile, lane);
                    }
#pragma unroll
                    for (int col_sub = 0; col_sub < 4; ++col_sub) {
                        b_frag[k_tile][col_sub] =
                            hrx_q5_k_wmma_vk128_load_b_frag_w64_b64asm_nowait(
                                sh_b_lds, col_sub, k_tile, lane);
                    }
                }
                asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
#pragma unroll
                for (int k_tile = 0; k_tile < 2; ++k_tile) {
#pragma unroll
                    for (int local = 0; local < 4; ++local) {
                        const int group = group_base + local;
                        const int row_sub = group & 3;
                        const int col_sub = group >> 2;
                        acc[local] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(
                            a_frag[k_tile][row_sub],
                            b_frag[k_tile][col_sub],
                            acc[local],
                            HRX_Q5_K_WMMA_VK128_W64_OPSEL != 0);
                    }
                }
            }
            __syncthreads();
        }

        if (wave == 0) {
#pragma unroll
            for (int local = 0; local < 4; ++local) {
                const int group = group_base + local;
#pragma unroll
                for (int slot = 0; slot < 4; ++slot) {
                    hrx_q5_k_wmma_vk64_batched4_p33_store_slot(
                        dst_rsrc, rows, row_base, col_base, rows, cols, acc, group_base, group, slot, lane);
                }
            }
        }
        asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
        __syncthreads();
    }
}
