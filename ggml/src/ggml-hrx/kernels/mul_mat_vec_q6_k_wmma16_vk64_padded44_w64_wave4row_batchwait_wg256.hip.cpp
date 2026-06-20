#include "mul_mat_vec_q6_k_wmma16_vk64_padded44_w64_wn32_bufferstore_wg128.hip.cpp"

static __device__ __forceinline__ hrx_q6_k_wmma_vk128_half16_vec
hrx_q6_k_wmma_vk64_load_a_frag_w64_b64asm_batchwait(
        hrx_q6_k_wmma_vk128_lds_half_ptr sh_a,
        int row_tile,
        int k_tile,
        unsigned int lane) {
    constexpr int SHARED_STRIDE = HRX_Q6_K_WMMA_VK128_SHARED_STRIDE;
    const int row = row_tile * 16 + static_cast<int>(lane & 15u);
    const int k_base = k_tile * 16;
    hrx_q6_k_wmma_vk128_lds_half_ptr row_ptr = sh_a + row * SHARED_STRIDE + k_base;
    const hrx_q6_k_wmma_vk128_half4_vec v0 = hrx_q6_k_wmma_vk128_ds_read_b64_h4_nowait(row_ptr + 0);
    const hrx_q6_k_wmma_vk128_half4_vec v1 = hrx_q6_k_wmma_vk128_ds_read_b64_h4_nowait(row_ptr + 4);
    const hrx_q6_k_wmma_vk128_half4_vec v2 = hrx_q6_k_wmma_vk128_ds_read_b64_h4_nowait(row_ptr + 8);
    const hrx_q6_k_wmma_vk128_half4_vec v3 = hrx_q6_k_wmma_vk128_ds_read_b64_h4_nowait(row_ptr + 12);
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    hrx_q6_k_wmma_vk128_half16_vec result;
    hrx_q6_k_wmma_vk128_append_half4(&result, 0, v0);
    hrx_q6_k_wmma_vk128_append_half4(&result, 4, v1);
    hrx_q6_k_wmma_vk128_append_half4(&result, 8, v2);
    hrx_q6_k_wmma_vk128_append_half4(&result, 12, v3);
    return result;
}

static __device__ __forceinline__ hrx_q6_k_wmma_vk128_half16_vec
hrx_q6_k_wmma_vk64_load_b_frag_w64_b64asm_batchwait(
        hrx_q6_k_wmma_vk128_lds_half_ptr sh_b,
        int col_tile,
        int k_tile,
        unsigned int lane) {
    constexpr int SHARED_STRIDE = HRX_Q6_K_WMMA_VK128_SHARED_STRIDE;
    const int col = col_tile * 16 + static_cast<int>(lane & 15u);
    const int k_base = k_tile * 16;
    hrx_q6_k_wmma_vk128_lds_half_ptr col_ptr = sh_b + col * SHARED_STRIDE + k_base;
    const hrx_q6_k_wmma_vk128_half4_vec v0 = hrx_q6_k_wmma_vk128_ds_read_b64_h4_nowait(col_ptr + 0);
    const hrx_q6_k_wmma_vk128_half4_vec v1 = hrx_q6_k_wmma_vk128_ds_read_b64_h4_nowait(col_ptr + 4);
    const hrx_q6_k_wmma_vk128_half4_vec v2 = hrx_q6_k_wmma_vk128_ds_read_b64_h4_nowait(col_ptr + 8);
    const hrx_q6_k_wmma_vk128_half4_vec v3 = hrx_q6_k_wmma_vk128_ds_read_b64_h4_nowait(col_ptr + 12);
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    hrx_q6_k_wmma_vk128_half16_vec result;
    hrx_q6_k_wmma_vk128_append_half4(&result, 0, v0);
    hrx_q6_k_wmma_vk128_append_half4(&result, 4, v1);
    hrx_q6_k_wmma_vk128_append_half4(&result, 8, v2);
    hrx_q6_k_wmma_vk128_append_half4(&result, 12, v3);
    return result;
}

static __device__ __forceinline__ void hrx_q6_k_wmma_vk64_wave4row_batchwait_store_slot(
        __amdgpu_buffer_rsrc_t dst_rsrc,
        long long rows_stride,
        long long row_base,
        long long col_base,
        long long rows,
        long long cols,
        const hrx_q6_k_wmma_vk128_half8_vec * acc,
        int row_sub,
        int col_sub,
        int slot,
        unsigned int lane) {
    const int row_lane = static_cast<int>(lane >> 4);
    const int col_lane = static_cast<int>(lane & 15u);
    const long long row = row_base + static_cast<long long>(row_sub * 16 + row_lane + slot * 4);
    const long long col = col_base + static_cast<long long>(col_sub * 16 + col_lane);
    if (row < rows && col < cols) {
        hrx_q6_k_wmma_vk128_buffer_store_f32(
            dst_rsrc,
            col * rows_stride + row,
            static_cast<float>(acc[col_sub][slot * 2 + HRX_Q6_K_WMMA_VK128_W64_OPSEL]));
    }
}

extern "C" __global__ __launch_bounds__(256, 1)
void hrx_mul_mat_vec_q6_k_wmma16x16_vk64_padded44_w64_wave4row_batchwait_f16acc_wg256_f32(
        const hrx_block_q6_K_wmma_vk128_lhs * src0,
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
    const int row_sub = static_cast<int>(wave);
    const long long row_base = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * BM;
    const long long col_base = static_cast<long long>(__builtin_amdgcn_workgroup_id_y()) * BN;
    if (row_base >= rows || col_base >= cols) {
        return;
    }

    const __amdgpu_buffer_rsrc_t dst_rsrc = hrx_q6_k_wmma_vk128_make_dst_rsrc(dst);
    __shared__ _Float16 sh_a[BM * SHARED_STRIDE];
    __shared__ _Float16 sh_b[BN * SHARED_STRIDE];

    const long long blocks_per_row = k / 256;
    const _Float16 zero = static_cast<_Float16>(0.0f);
    hrx_q6_k_wmma_vk128_half8_vec acc[4] = {};

    for (long long k0 = 0; k0 < k; k0 += BK) {
        for (int idx = static_cast<int>(tid); idx < BM * BK; idx += 256) {
            const int r = idx / BK;
            const int kk = idx - r * BK;
            const long long row = row_base + static_cast<long long>(r);
            sh_a[r * SHARED_STRIDE + kk] = row < rows ?
                hrx_q6_k_wmma_vk128_load_a_value(src0, row, k0 + kk, blocks_per_row) : zero;
        }
        for (int idx = static_cast<int>(tid); idx < BN * BK; idx += 256) {
            const int c = idx / BK;
            const int kk = idx - c * BK;
            const long long col = col_base + static_cast<long long>(c);
            sh_b[c * SHARED_STRIDE + kk] = col < cols ? static_cast<_Float16>(src1[col * k + k0 + kk]) : zero;
        }
        __syncthreads();

        hrx_q6_k_wmma_vk128_lds_half_ptr sh_a_lds =
            (hrx_q6_k_wmma_vk128_lds_half_ptr) sh_a;
        hrx_q6_k_wmma_vk128_lds_half_ptr sh_b_lds =
            (hrx_q6_k_wmma_vk128_lds_half_ptr) sh_b;
        hrx_q6_k_wmma_vk128_half16_vec a_frag[2];
        hrx_q6_k_wmma_vk128_half16_vec b_frag[2][4];
#pragma unroll
        for (int k_tile = 0; k_tile < 2; ++k_tile) {
            a_frag[k_tile] =
                hrx_q6_k_wmma_vk64_load_a_frag_w64_b64asm_batchwait(
                    sh_a_lds, row_sub, k_tile, lane);
#pragma unroll
            for (int col_sub = 0; col_sub < 4; ++col_sub) {
                b_frag[k_tile][col_sub] =
                    hrx_q6_k_wmma_vk64_load_b_frag_w64_b64asm_batchwait(
                        sh_b_lds, col_sub, k_tile, lane);
            }
        }

#pragma unroll
        for (int k_tile = 0; k_tile < 2; ++k_tile) {
#pragma unroll
            for (int col_sub = 0; col_sub < 4; ++col_sub) {
                acc[col_sub] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(
                    a_frag[k_tile],
                    b_frag[k_tile][col_sub],
                    acc[col_sub],
                    HRX_Q6_K_WMMA_VK128_W64_OPSEL != 0);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int col_sub = 0; col_sub < 4; ++col_sub) {
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            hrx_q6_k_wmma_vk64_wave4row_batchwait_store_slot(
                dst_rsrc, rows, row_base, col_base, rows, cols, acc, row_sub, col_sub, slot, lane);
        }
    }
}
