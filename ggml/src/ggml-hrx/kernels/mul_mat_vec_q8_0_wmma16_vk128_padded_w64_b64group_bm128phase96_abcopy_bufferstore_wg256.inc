#ifndef HRX_Q8_0_WMMA_VK128_BM128_PHASE96_EXPORT
#error "HRX_Q8_0_WMMA_VK128_BM128_PHASE96_EXPORT must be defined"
#endif

#ifndef HRX_Q8_0_WMMA_VK128_BM128_PHASE96_GROUP_BASE
#error "HRX_Q8_0_WMMA_VK128_BM128_PHASE96_GROUP_BASE must be defined"
#endif

#ifndef HRX_Q8_0_WMMA_VK128_BM128_PHASE96_COL_START
#error "HRX_Q8_0_WMMA_VK128_BM128_PHASE96_COL_START must be defined"
#endif

#ifndef HRX_Q8_0_WMMA_VK128_BM128_PHASE96_COL_COUNT
#error "HRX_Q8_0_WMMA_VK128_BM128_PHASE96_COL_COUNT must be defined"
#endif

#ifndef HRX_Q8_0_WMMA_VK128_HELPER_EXPORT
#define HRX_Q8_0_WMMA_VK128_HELPER_EXPORT hrx_q8_0_wmma16x16_vk128_bm128phase96_helper_unused
#endif

#define HRX_Q8_0_WMMA_VK128_EXPORT HRX_Q8_0_WMMA_VK128_HELPER_EXPORT
#define HRX_Q8_0_WMMA_VK128_SHARED_STRIDE 40
#define HRX_Q8_0_WMMA_VK128_W64 1
#define HRX_Q8_0_WMMA_VK128_W64_B64GROUP 1
#define HRX_Q8_0_WMMA_VK128_BUFFER_STORE 1
#define HRX_Q8_0_WMMA_VK128_COPY_A_FRAG 1
#define HRX_Q8_0_WMMA_VK128_COPY_B_FRAG 1
#define HRX_Q8_0_WMMA_VK128_COPY_B_FRAG_MIN_COL_SUB 0
#include "mul_mat_vec_q8_0_wmma16_vk128_wg256.hip.cpp"

static __device__ __forceinline__ void hrx_q8_0_wmma_vk128_bm128phase96_store_slot(
        __amdgpu_buffer_rsrc_t dst_rsrc,
        long long rows_stride,
        long long row_base,
        long long col_base,
        long long rows,
        long long cols,
        const hrx_q8_0_wmma_vk128_half8_vec * acc,
        unsigned int wave,
        int group,
        int slot,
        unsigned int lane) {
    const int wave_row = static_cast<int>(wave & 1u);
    const int wave_col = static_cast<int>(wave >> 1);
    const int row_lane = static_cast<int>(lane >> 4);
    const int col_lane = static_cast<int>(lane & 15u);
    const int acc_index = group & 7;
    const long long row = row_base + static_cast<long long>(
        wave_row * 64 + (group & 3) * 16 + row_lane + slot * 4);
    const long long col = col_base + static_cast<long long>(
        wave_col * 64 + ((group >> 2) & 3) * 16 + col_lane);
    if (row < rows && col < cols) {
        hrx_q8_0_wmma_vk128_buffer_store_f32(
            dst_rsrc,
            col * rows_stride + row,
            static_cast<float>(acc[acc_index][slot * 2 + HRX_Q8_0_WMMA_VK128_W64_OPSEL]));
    }
}

extern "C" __global__ __launch_bounds__(256, 1)
void HRX_Q8_0_WMMA_VK128_BM128_PHASE96_EXPORT(
        const hrx_block_q8_0_wmma_vk128_lhs * src0,
        const float * src1,
        float * dst,
        long long k,
        long long rows,
        long long cols) {
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 32;
    constexpr int SHARED_STRIDE = HRX_Q8_0_WMMA_VK128_SHARED_STRIDE;
    constexpr int GROUP_BASE = HRX_Q8_0_WMMA_VK128_BM128_PHASE96_GROUP_BASE;
    constexpr int COL_START = HRX_Q8_0_WMMA_VK128_BM128_PHASE96_COL_START;
    constexpr int COL_COUNT = HRX_Q8_0_WMMA_VK128_BM128_PHASE96_COL_COUNT;
    static_assert(COL_COUNT >= 1 && COL_COUNT <= 2, "unexpected column group count");

    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int wave = tid >> 6u;
    const unsigned int lane = tid & 63u;
    const int wave_row = static_cast<int>(wave & 1u);
    const int wave_col = static_cast<int>(wave >> 1);
    const long long row_base = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * BM;
    const long long col_base = static_cast<long long>(__builtin_amdgcn_workgroup_id_y()) * BN;
    if (row_base >= rows || col_base >= cols) {
        return;
    }

    const __amdgpu_buffer_rsrc_t dst_rsrc = hrx_q8_0_wmma_vk128_make_dst_rsrc(dst);
    __shared__ _Float16 sh_a[BM * SHARED_STRIDE];
    __shared__ _Float16 sh_b[BN * SHARED_STRIDE];

    const long long blocks_per_row = k / 32;
    const _Float16 zero = static_cast<_Float16>(0.0f);
    hrx_q8_0_wmma_vk128_half8_vec acc[8] = {};

    for (long long k0 = 0; k0 < k; k0 += BK) {
        for (int idx = static_cast<int>(tid); idx < BM * BK; idx += 256) {
            const int r = idx / BK;
            const int kk = idx - r * BK;
            const long long row = row_base + static_cast<long long>(r);
            sh_a[r * SHARED_STRIDE + kk] = row < rows ?
                hrx_q8_0_wmma_vk128_load_a_value(src0, row, k0 + kk, blocks_per_row) : zero;
        }
        for (int idx = static_cast<int>(tid); idx < BN * BK; idx += 256) {
            const int c = idx / BK;
            const int kk = idx - c * BK;
            const long long col = col_base + static_cast<long long>(c);
            sh_b[c * SHARED_STRIDE + kk] = col < cols ?
                static_cast<_Float16>(src1[col * k + k0 + kk]) : zero;
        }
        __syncthreads();

        hrx_q8_0_wmma_vk128_lds_half_ptr sh_a_lds =
            (hrx_q8_0_wmma_vk128_lds_half_ptr) sh_a;
        hrx_q8_0_wmma_vk128_lds_half_ptr sh_b_lds =
            (hrx_q8_0_wmma_vk128_lds_half_ptr) sh_b;

#pragma unroll
        for (int k_tile = 0; k_tile < 2; ++k_tile) {
            hrx_q8_0_wmma_vk128_half16_vec a_frag[4];
            hrx_q8_0_wmma_vk128_half16_vec b_frag[4];
#pragma unroll
            for (int row_sub = 0; row_sub < 4; ++row_sub) {
                a_frag[row_sub] =
                    hrx_q8_0_wmma_vk128_load_a_frag_w64_b64asm_nowait(
                        sh_a_lds, wave_row * 4 + row_sub, k_tile, lane);
            }
#pragma unroll
            for (int col_sub = 0; col_sub < 4; ++col_sub) {
                b_frag[col_sub] =
                    hrx_q8_0_wmma_vk128_load_b_frag_w64_b64asm_nowait(
                        sh_b_lds, wave_col * 4 + col_sub, k_tile, lane);
            }
            asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
#pragma unroll
            for (int col_delta = 0; col_delta < COL_COUNT; ++col_delta) {
                const int col_sub = COL_START + col_delta;
#pragma unroll
                for (int row_sub = 0; row_sub < 4; ++row_sub) {
                    const int local = col_delta * 4 + row_sub;
                    const hrx_q8_0_wmma_vk128_half16_vec a_use =
                        hrx_q8_0_wmma_vk128_copy_frag(a_frag[row_sub]);
                    const hrx_q8_0_wmma_vk128_half16_vec b_use =
                        hrx_q8_0_wmma_vk128_copy_frag(b_frag[col_sub]);
                    acc[local] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(
                        a_use,
                        b_use,
                        acc[local],
                        HRX_Q8_0_WMMA_VK128_W64_OPSEL != 0);
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int local = 0; local < 8; ++local) {
        const int group = GROUP_BASE + local;
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            hrx_q8_0_wmma_vk128_bm128phase96_store_slot(
                dst_rsrc, rows, row_base, col_base, rows, cols, acc, wave, group, slot, lane);
        }
    }
}
