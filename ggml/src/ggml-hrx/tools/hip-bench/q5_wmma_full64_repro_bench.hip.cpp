#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../../kernels/mul_mat_vec_q5_k_wmma16_vk64_padded44_w64_full64_wg256.hip.cpp"

#define HIP_CHECK(expr) do { \
    hipError_t _err = (expr); \
    if (_err != hipSuccess) { \
        std::fprintf(stderr, "%s:%d: HIP error: %s\n", __FILE__, __LINE__, hipGetErrorString(_err)); \
        std::exit(2); \
    } \
} while (0)

static __device__ __forceinline__ void q5_repro_consume_frag(
        hrx_q5_k_wmma_vk128_half16_vec frag) {
    asm volatile("" :: "v"(frag[0]), "v"(frag[4]), "v"(frag[8]), "v"(frag[12]) : "memory");
}

template <int active_groups>
__global__ __launch_bounds__(256, 1)
void q5_full64_active_groups_repro_kernel(
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
    static_assert(active_groups >= 1 && active_groups <= 16, "unexpected active group count");

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
    hrx_q5_k_wmma_vk128_half8_vec acc[active_groups] = {};

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
                for (int col_sub = 0; col_sub < 4; ++col_sub) {
#pragma unroll
                    for (int row_sub = 0; row_sub < 4; ++row_sub) {
                        const int tile = col_sub * 4 + row_sub;
                        if (tile < active_groups) {
                            acc[tile] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(
                                a_frag[k_tile][row_sub],
                                b_frag[k_tile][col_sub],
                                acc[tile],
                                HRX_Q5_K_WMMA_VK128_W64_OPSEL != 0);
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    if (wave == 0) {
#pragma unroll
        for (int group = 0; group < active_groups; ++group) {
#pragma unroll
            for (int slot = 0; slot < 4; ++slot) {
                hrx_q5_k_wmma_vk128_combined96_raw_store_slot(
                    dst_rsrc, rows, row_base, col_base, rows, cols, acc, group, slot, lane);
            }
        }
    }
}

__global__ __launch_bounds__(256, 1)
void q5_array8_b2_repro_kernel(
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
    constexpr int ACTIVE_GROUPS = 8;

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
    hrx_q5_k_wmma_vk128_half8_vec acc[ACTIVE_GROUPS] = {};

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
            hrx_q5_k_wmma_vk128_half16_vec b_frag[2][2];
#pragma unroll
            for (int k_tile = 0; k_tile < 2; ++k_tile) {
#pragma unroll
                for (int row_sub = 0; row_sub < 4; ++row_sub) {
                    a_frag[k_tile][row_sub] =
                        hrx_q5_k_wmma_vk128_load_a_frag_w64_b64asm_nowait(
                            sh_a_lds, row_sub, k_tile, lane);
                }
#pragma unroll
                for (int col_sub = 0; col_sub < 2; ++col_sub) {
                    b_frag[k_tile][col_sub] =
                        hrx_q5_k_wmma_vk128_load_b_frag_w64_b64asm_nowait(
                            sh_b_lds, col_sub, k_tile, lane);
                }
            }
            asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
#pragma unroll
            for (int k_tile = 0; k_tile < 2; ++k_tile) {
#pragma unroll
                for (int col_sub = 0; col_sub < 2; ++col_sub) {
#pragma unroll
                    for (int row_sub = 0; row_sub < 4; ++row_sub) {
                        const int tile = col_sub * 4 + row_sub;
                        acc[tile] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(
                            a_frag[k_tile][row_sub],
                            b_frag[k_tile][col_sub],
                            acc[tile],
                            HRX_Q5_K_WMMA_VK128_W64_OPSEL != 0);
                    }
                }
            }
        }
        __syncthreads();
    }

    if (wave == 0) {
#pragma unroll
        for (int group = 0; group < ACTIVE_GROUPS; ++group) {
#pragma unroll
            for (int slot = 0; slot < 4; ++slot) {
                hrx_q5_k_wmma_vk128_combined96_raw_store_slot(
                    dst_rsrc, rows, row_base, col_base, rows, cols, acc, group, slot, lane);
            }
        }
    }
}

__global__ __launch_bounds__(256, 1)
void q5_array8_fullb_noif_repro_kernel(
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
    constexpr int ACTIVE_GROUPS = 8;

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
    hrx_q5_k_wmma_vk128_half8_vec acc[ACTIVE_GROUPS] = {};

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
                q5_repro_consume_frag(b_frag[k_tile][2]);
                q5_repro_consume_frag(b_frag[k_tile][3]);
            }
            asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
#pragma unroll
            for (int k_tile = 0; k_tile < 2; ++k_tile) {
#pragma unroll
                for (int col_sub = 0; col_sub < 2; ++col_sub) {
#pragma unroll
                    for (int row_sub = 0; row_sub < 4; ++row_sub) {
                        const int tile = col_sub * 4 + row_sub;
                        acc[tile] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(
                            a_frag[k_tile][row_sub],
                            b_frag[k_tile][col_sub],
                            acc[tile],
                            HRX_Q5_K_WMMA_VK128_W64_OPSEL != 0);
                    }
                }
            }
        }
        __syncthreads();
    }

    if (wave == 0) {
#pragma unroll
        for (int group = 0; group < ACTIVE_GROUPS; ++group) {
#pragma unroll
            for (int slot = 0; slot < 4; ++slot) {
                hrx_q5_k_wmma_vk128_combined96_raw_store_slot(
                    dst_rsrc, rows, row_base, col_base, rows, cols, acc, group, slot, lane);
            }
        }
    }
}

static __device__ __forceinline__ void q5_full64_batched4_store_slot(
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

__global__ __launch_bounds__(256, 1)
void q5_full64_batched4_repro_kernel(
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
    for (int group_base = 0; group_base < 16; group_base += 4) {
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
                    q5_full64_batched4_store_slot(
                        dst_rsrc, rows, row_base, col_base, rows, cols, acc, group_base, group, slot, lane);
                }
            }
        }
        asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
        __syncthreads();
    }
}

static __device__ __forceinline__ hrx_q5_k_wmma_vk128_lds_u16_ptr q5_combined96_stage_ptr(
        _Float16 * sh_a,
        _Float16 * sh_b,
        int index) {
    constexpr int SH_A_HALF_COUNT = 64 * 44;
    hrx_q5_k_wmma_vk128_lds_u16_ptr sh_a_u16 = (hrx_q5_k_wmma_vk128_lds_u16_ptr) sh_a;
    hrx_q5_k_wmma_vk128_lds_u16_ptr sh_b_u16 = (hrx_q5_k_wmma_vk128_lds_u16_ptr) sh_b;
    return index < SH_A_HALF_COUNT ? sh_a_u16 + index : sh_b_u16 + (index - SH_A_HALF_COUNT);
}

static __device__ __forceinline__ hrx_q5_k_wmma_vk128_lds_const_u16_ptr q5_combined96_stage_const_ptr(
        const _Float16 * sh_a,
        const _Float16 * sh_b,
        int index) {
    constexpr int SH_A_HALF_COUNT = 64 * 44;
    hrx_q5_k_wmma_vk128_lds_const_u16_ptr sh_a_u16 = (hrx_q5_k_wmma_vk128_lds_const_u16_ptr) sh_a;
    hrx_q5_k_wmma_vk128_lds_const_u16_ptr sh_b_u16 = (hrx_q5_k_wmma_vk128_lds_const_u16_ptr) sh_b;
    return index < SH_A_HALF_COUNT ? sh_a_u16 + index : sh_b_u16 + (index - SH_A_HALF_COUNT);
}

static __device__ __forceinline__ int q5_combined96_stage_index(
        int group,
        int slot,
        unsigned int lane) {
    const int stage_group = group - 8;
    const int row_lane = static_cast<int>(lane >> 4);
    const int col_lane = static_cast<int>(lane & 15u);
    return stage_group * 16 * 16 + col_lane * 16 + row_lane + slot * 4;
}

static __device__ __forceinline__ void q5_combined96_ds_store_u16(
        hrx_q5_k_wmma_vk128_lds_u16_ptr ptr,
        uint16_t value) {
    asm volatile("ds_write_b16 %0, %1 offset:0\n"
                 :
                 : "v"(ptr), "v"(static_cast<uint32_t>(value))
                 : "memory");
}

static __device__ __forceinline__ uint32_t q5_combined96_ds_load_u16_d16(
        hrx_q5_k_wmma_vk128_lds_const_u16_ptr ptr) {
    uint32_t value = 0;
    asm volatile("ds_read_u16_d16 %0, %1 offset:0\n"
                 : "=v"(value)
                 : "v"(ptr)
                 : "memory");
    return value;
}

static __device__ __forceinline__ void q5_combined96_stage_store_slot(
        _Float16 * sh_a,
        _Float16 * sh_b,
        const hrx_q5_k_wmma_vk128_half8_vec * acc,
        int group,
        int slot,
        unsigned int lane) {
    const int acc_index = group & 7;
    q5_combined96_ds_store_u16(
        q5_combined96_stage_ptr(sh_a, sh_b, q5_combined96_stage_index(group, slot, lane)),
        hrx_q5_k_wmma_vk128_f16_to_u16(acc[acc_index][slot * 2 + HRX_Q5_K_WMMA_VK128_W64_OPSEL]));
}

static __device__ __forceinline__ void q5_combined96_stage_load_store_slot(
        __amdgpu_buffer_rsrc_t dst_rsrc,
        long long rows_stride,
        long long row_base,
        long long col_base,
        long long rows,
        long long cols,
        const _Float16 * sh_a,
        const _Float16 * sh_b,
        int group,
        int slot,
        unsigned int lane) {
    const int row_lane = static_cast<int>(lane >> 4);
    const int col_lane = static_cast<int>(lane & 15u);
    const long long row = row_base + static_cast<long long>((group & 3) * 16 + row_lane + slot * 4);
    const long long col = col_base + static_cast<long long>(((group >> 2) & 3) * 16 + col_lane);
    const _Float16 value = hrx_q5_k_wmma_vk128_u16_to_f16(
        q5_combined96_ds_load_u16_d16(
            q5_combined96_stage_const_ptr(sh_a, sh_b, q5_combined96_stage_index(group, slot, lane))));
    if (row < rows && col < cols) {
        hrx_q5_k_wmma_vk128_buffer_store_f32(dst_rsrc, col * rows_stride + row, static_cast<float>(value));
    }
}

static __device__ __forceinline__ void q5_combined96_consume_frag(
        hrx_q5_k_wmma_vk128_half16_vec frag) {
    asm volatile("" :: "v"(frag[0]), "v"(frag[4]), "v"(frag[8]), "v"(frag[12]) : "memory");
}

#define Q5_COMBINED96_RAW_STORE_GROUP(GROUP_ID) do { \
    hrx_q5_k_wmma_vk128_combined96_raw_store_slot(dst_rsrc, rows, row_base, col_base, rows, cols, acc, (GROUP_ID), 0, lane); \
    hrx_q5_k_wmma_vk128_combined96_raw_store_slot(dst_rsrc, rows, row_base, col_base, rows, cols, acc, (GROUP_ID), 1, lane); \
    hrx_q5_k_wmma_vk128_combined96_raw_store_slot(dst_rsrc, rows, row_base, col_base, rows, cols, acc, (GROUP_ID), 2, lane); \
    hrx_q5_k_wmma_vk128_combined96_raw_store_slot(dst_rsrc, rows, row_base, col_base, rows, cols, acc, (GROUP_ID), 3, lane); \
} while (0)

#define Q5_COMBINED96_STAGE_STORE_GROUP(GROUP_ID) do { \
    q5_combined96_stage_store_slot(sh_a, sh_b, acc, (GROUP_ID), 0, lane); \
    q5_combined96_stage_store_slot(sh_a, sh_b, acc, (GROUP_ID), 1, lane); \
    q5_combined96_stage_store_slot(sh_a, sh_b, acc, (GROUP_ID), 2, lane); \
    q5_combined96_stage_store_slot(sh_a, sh_b, acc, (GROUP_ID), 3, lane); \
} while (0)

#define Q5_COMBINED96_STAGE_LOAD_STORE_GROUP(GROUP_ID) do { \
    q5_combined96_stage_load_store_slot(dst_rsrc, rows, row_base, col_base, rows, cols, sh_a, sh_b, (GROUP_ID), 0, lane); \
    q5_combined96_stage_load_store_slot(dst_rsrc, rows, row_base, col_base, rows, cols, sh_a, sh_b, (GROUP_ID), 1, lane); \
    q5_combined96_stage_load_store_slot(dst_rsrc, rows, row_base, col_base, rows, cols, sh_a, sh_b, (GROUP_ID), 2, lane); \
    q5_combined96_stage_load_store_slot(dst_rsrc, rows, row_base, col_base, rows, cols, sh_a, sh_b, (GROUP_ID), 3, lane); \
} while (0)

#define Q5_COMBINED96_GROUPS_0_7(MACRO) do { \
    MACRO(0); MACRO(1); MACRO(2); MACRO(3); \
    MACRO(4); MACRO(5); MACRO(6); MACRO(7); \
} while (0)

#define Q5_COMBINED96_GROUPS_8_23(MACRO) do { \
    MACRO(8);  MACRO(9);  MACRO(10); MACRO(11); \
    MACRO(12); MACRO(13); MACRO(14); MACRO(15); \
    MACRO(16); MACRO(17); MACRO(18); MACRO(19); \
    MACRO(20); MACRO(21); MACRO(22); MACRO(23); \
} while (0)

template <bool store_stage, bool wait_all_frag_loads, bool pad_b_loads>
__global__ __launch_bounds__(256, 1)
void q5_combined96_repro_kernel(
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
    hrx_q5_k_wmma_vk128_half8_vec acc[8] = {};

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
            const hrx_q5_k_wmma_vk128_half16_vec a0 =
                hrx_q5_k_wmma_vk128_load_a_frag_w64_b64asm_nowait(sh_a_lds, 0, 0, lane);
            const hrx_q5_k_wmma_vk128_half16_vec a1 =
                hrx_q5_k_wmma_vk128_load_a_frag_w64_b64asm_nowait(sh_a_lds, 1, 0, lane);
            const hrx_q5_k_wmma_vk128_half16_vec a2 =
                hrx_q5_k_wmma_vk128_load_a_frag_w64_b64asm_nowait(sh_a_lds, 2, 0, lane);
            const hrx_q5_k_wmma_vk128_half16_vec a3 =
                hrx_q5_k_wmma_vk128_load_a_frag_w64_b64asm_nowait(sh_a_lds, 3, 0, lane);
            const hrx_q5_k_wmma_vk128_half16_vec b0 =
                hrx_q5_k_wmma_vk128_load_b_frag_w64_b64asm_nowait(sh_b_lds, 0, 0, lane);
            const hrx_q5_k_wmma_vk128_half16_vec b1 =
                hrx_q5_k_wmma_vk128_load_b_frag_w64_b64asm_nowait(sh_b_lds, 1, 0, lane);
            const hrx_q5_k_wmma_vk128_half16_vec a4 =
                hrx_q5_k_wmma_vk128_load_a_frag_w64_b64asm_nowait(sh_a_lds, 0, 1, lane);
            const hrx_q5_k_wmma_vk128_half16_vec a5 =
                hrx_q5_k_wmma_vk128_load_a_frag_w64_b64asm_nowait(sh_a_lds, 1, 1, lane);
            const hrx_q5_k_wmma_vk128_half16_vec a6 =
                hrx_q5_k_wmma_vk128_load_a_frag_w64_b64asm_nowait(sh_a_lds, 2, 1, lane);
            const hrx_q5_k_wmma_vk128_half16_vec a7 =
                hrx_q5_k_wmma_vk128_load_a_frag_w64_b64asm_nowait(sh_a_lds, 3, 1, lane);
            const hrx_q5_k_wmma_vk128_half16_vec b2 =
                hrx_q5_k_wmma_vk128_load_b_frag_w64_b64asm_nowait(sh_b_lds, 0, 1, lane);
            const hrx_q5_k_wmma_vk128_half16_vec b3 =
                hrx_q5_k_wmma_vk128_load_b_frag_w64_b64asm_nowait(sh_b_lds, 1, 1, lane);
            if constexpr (pad_b_loads) {
                const hrx_q5_k_wmma_vk128_half16_vec b_pad0 =
                    hrx_q5_k_wmma_vk128_load_b_frag_w64_b64asm_nowait(sh_b_lds, 2, 0, lane);
                const hrx_q5_k_wmma_vk128_half16_vec b_pad1 =
                    hrx_q5_k_wmma_vk128_load_b_frag_w64_b64asm_nowait(sh_b_lds, 3, 0, lane);
                const hrx_q5_k_wmma_vk128_half16_vec b_pad2 =
                    hrx_q5_k_wmma_vk128_load_b_frag_w64_b64asm_nowait(sh_b_lds, 2, 1, lane);
                const hrx_q5_k_wmma_vk128_half16_vec b_pad3 =
                    hrx_q5_k_wmma_vk128_load_b_frag_w64_b64asm_nowait(sh_b_lds, 3, 1, lane);
                q5_combined96_consume_frag(b_pad0);
                q5_combined96_consume_frag(b_pad1);
                q5_combined96_consume_frag(b_pad2);
                q5_combined96_consume_frag(b_pad3);
            }

            if constexpr (wait_all_frag_loads) {
                asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
            } else {
                asm volatile("s_waitcnt lgkmcnt(40)\n" ::: "memory");
            }
            acc[0] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(a0, b0, acc[0], false);
            acc[1] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(a1, b0, acc[1], false);
            acc[2] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(a2, b0, acc[2], false);
            acc[3] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(a3, b0, acc[3], false);
            acc[4] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(a0, b1, acc[4], false);
            acc[5] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(a1, b1, acc[5], false);
            acc[6] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(a2, b1, acc[6], false);
            acc[7] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(a3, b1, acc[7], false);
            acc[0] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(a4, b2, acc[0], false);
            acc[1] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(a5, b2, acc[1], false);
            acc[2] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(a6, b2, acc[2], false);
            acc[3] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(a7, b2, acc[3], false);
            acc[4] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(a4, b3, acc[4], false);
            acc[5] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(a5, b3, acc[5], false);
            acc[6] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(a6, b3, acc[6], false);
            acc[7] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(a7, b3, acc[7], false);
        }
        __syncthreads();
    }

    if (wave == 0) {
        Q5_COMBINED96_GROUPS_0_7(Q5_COMBINED96_RAW_STORE_GROUP);
        if constexpr (store_stage) {
            Q5_COMBINED96_GROUPS_8_23(Q5_COMBINED96_STAGE_STORE_GROUP);
        }
    }
    if constexpr (store_stage) {
        asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
        __syncthreads();
        if (wave == 0) {
            Q5_COMBINED96_GROUPS_8_23(Q5_COMBINED96_STAGE_LOAD_STORE_GROUP);
        }
    }
}

template <typename T>
struct device_buffer {
    T * ptr = nullptr;
    size_t count = 0;

    explicit device_buffer(size_t count) : count(count) {
        HIP_CHECK(hipMalloc(&ptr, count * sizeof(T)));
    }

    ~device_buffer() {
        if (ptr) {
            (void) hipFree(ptr);
        }
    }

    device_buffer(const device_buffer &) = delete;
    device_buffer & operator=(const device_buffer &) = delete;
};

static float half_bits_to_float(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h >> 15) & 1u;
    const uint32_t exp = static_cast<uint32_t>(h >> 10) & 31u;
    const uint32_t mant = static_cast<uint32_t>(h) & 1023u;
    if (exp == 0) {
        const float value = std::ldexp(static_cast<float>(mant), -24);
        return sign ? -value : value;
    }
    if (exp == 31) {
        return mant ? NAN : (sign ? -INFINITY : INFINITY);
    }
    const float value = std::ldexp(1.0f + static_cast<float>(mant) / 1024.0f, static_cast<int>(exp) - 15);
    return sign ? -value : value;
}

static void get_scale_min(int group, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (group < 4) {
        *d = q[group] & 63u;
        *m = q[group + 4] & 63u;
    } else {
        *d = (q[group + 4] & 0x0fu) | ((q[group - 4] >> 6) << 4);
        *m = (q[group + 4] >> 4) | ((q[group] >> 6) << 4);
    }
}

static int q5_value(const hrx_block_q5_K_wmma_vk128_lhs & block, int in_block) {
    const int group = in_block >> 5;
    const int q_index = (group >> 1) * 32 + (in_block & 31);
    const uint8_t packed = block.qs[q_index];
    const uint8_t lo = (packed >> ((group & 1) * 4)) & 0x0f;
    const uint8_t hi = ((block.qh[in_block & 31] >> group) & 1) << 4;
    return lo | hi;
}

static float q5_dequant(
        const std::vector<hrx_block_q5_K_wmma_vk128_lhs> & blocks,
        int row,
        int k_index,
        int blocks_per_row) {
    const hrx_block_q5_K_wmma_vk128_lhs & block =
        blocks[static_cast<size_t>(row) * blocks_per_row + (k_index >> 8)];
    const int in_block = k_index & 255;
    const int group = in_block >> 5;
    uint8_t sc = 0;
    uint8_t m = 0;
    get_scale_min(group, block.scales, &sc, &m);
    const float d = half_bits_to_float(block.d) * static_cast<float>(sc);
    const float dmin = half_bits_to_float(block.dmin) * static_cast<float>(m);
    return d * static_cast<float>(q5_value(block, in_block)) - dmin;
}

struct case_config {
    const char * name;
    uint16_t d_bits;
    uint16_t dmin_bits;
    int scale_mask;
    float rhs_scale;
};

static float rhs_value(int col, int k_index, const case_config & config) {
    const int raw = (col * 19 + k_index * 7 + 23) & 63;
    return (static_cast<float>(raw) - 31.0f) * config.rhs_scale;
}

static void fill_q5(
        std::vector<hrx_block_q5_K_wmma_vk128_lhs> & blocks,
        int rows,
        int blocks_per_row,
        const case_config & config) {
    for (int row = 0; row < rows; ++row) {
        for (int block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
            hrx_block_q5_K_wmma_vk128_lhs & block =
                blocks[static_cast<size_t>(row) * blocks_per_row + block_idx];
            block.d = config.d_bits;
            block.dmin = config.dmin_bits;
            for (int i = 0; i < 12; ++i) {
                block.scales[i] = static_cast<uint8_t>(1 + ((row * 3 + block_idx * 5 + i * 11) & config.scale_mask));
            }
            for (int i = 0; i < 32; ++i) {
                block.qh[i] = static_cast<uint8_t>((row * 17 + block_idx * 13 + i * 5) & 0xff);
            }
            for (int i = 0; i < 128; ++i) {
                const uint8_t lo = static_cast<uint8_t>((row * 5 + block_idx * 7 + i) & 15);
                const uint8_t hi = static_cast<uint8_t>((row * 11 + block_idx * 3 + i * 9) & 15);
                block.qs[i] = static_cast<uint8_t>(lo | (hi << 4));
            }
        }
    }
}

static void fill_rhs(std::vector<float> & rhs, int k, int cols, const case_config & config) {
    for (int col = 0; col < cols; ++col) {
        for (int kk = 0; kk < k; ++kk) {
            rhs[static_cast<size_t>(col) * k + kk] = rhs_value(col, kk, config);
        }
    }
}

static std::vector<float> cpu_reference(
        const std::vector<hrx_block_q5_K_wmma_vk128_lhs> & blocks,
        const std::vector<float> & rhs,
        int k,
        int rows,
        int cols) {
    const int blocks_per_row = k / 256;
    std::vector<float> ref(static_cast<size_t>(rows) * cols, 0.0f);
    for (int col = 0; col < cols; ++col) {
        for (int row = 0; row < rows; ++row) {
            float sum = 0.0f;
            for (int kk = 0; kk < k; ++kk) {
                sum += q5_dequant(blocks, row, kk, blocks_per_row) * rhs[static_cast<size_t>(col) * k + kk];
            }
            ref[static_cast<size_t>(col) * rows + row] = sum;
        }
    }
    return ref;
}

static bool output_is_active(size_t index, int rows, int active_groups) {
    const int row = static_cast<int>(index % static_cast<size_t>(rows));
    const int col = static_cast<int>(index / static_cast<size_t>(rows));
    const int row_local = row & 63;
    const int col_local = col & 63;
    const int group = (col_local >> 4) * 4 + (row_local >> 4);
    if (active_groups == -103 || active_groups == -102 || active_groups == -97 ||
            active_groups == -98 || active_groups == -100) {
        return group < 8;
    }
    if (active_groups <= 0) {
        return true;
    }
    return group < active_groups;
}

static const char * variant_name(int active_groups) {
    switch (active_groups) {
        case -103: return "array8-fullb-noif";
        case -102: return "array8-b2";
        case -101: return "combined96-bpad";
        case -100: return "combined96-raw8-bpad";
        case -99: return "combined96-wait0";
        case -98: return "combined96-raw8-wait0";
        case -97: return "combined96-raw8";
        case -96: return "combined96";
        case -4: return "batched4";
        case 0: return "catalog-full64";
        case 1: return "active1";
        case 4: return "active4";
        case 8: return "active8";
        case 12: return "active12";
        case 16: return "active16";
        default: return "unknown";
    }
}

static void launch_variant(
        int active_groups,
        dim3 grid,
        const hrx_block_q5_K_wmma_vk128_lhs * src0,
        const float * src1,
        float * dst,
        long long k,
        long long rows,
        long long cols) {
    switch (active_groups) {
        case -103:
            hipLaunchKernelGGL(
                q5_array8_fullb_noif_repro_kernel,
                grid,
                dim3(256, 1, 1),
                0,
                0,
                src0,
                src1,
                dst,
                k,
                rows,
                cols);
            break;
        case -102:
            hipLaunchKernelGGL(
                q5_array8_b2_repro_kernel,
                grid,
                dim3(256, 1, 1),
                0,
                0,
                src0,
                src1,
                dst,
                k,
                rows,
                cols);
            break;
        case -96:
            hipLaunchKernelGGL(
                (q5_combined96_repro_kernel<true, false, false>),
                grid,
                dim3(256, 1, 1),
                0,
                0,
                src0,
                src1,
                dst,
                k,
                rows,
                cols);
            break;
        case -97:
            hipLaunchKernelGGL(
                (q5_combined96_repro_kernel<false, false, false>),
                grid,
                dim3(256, 1, 1),
                0,
                0,
                src0,
                src1,
                dst,
                k,
                rows,
                cols);
            break;
        case -98:
            hipLaunchKernelGGL(
                (q5_combined96_repro_kernel<false, true, false>),
                grid,
                dim3(256, 1, 1),
                0,
                0,
                src0,
                src1,
                dst,
                k,
                rows,
                cols);
            break;
        case -99:
            hipLaunchKernelGGL(
                (q5_combined96_repro_kernel<true, true, false>),
                grid,
                dim3(256, 1, 1),
                0,
                0,
                src0,
                src1,
                dst,
                k,
                rows,
                cols);
            break;
        case -100:
            hipLaunchKernelGGL(
                (q5_combined96_repro_kernel<false, false, true>),
                grid,
                dim3(256, 1, 1),
                0,
                0,
                src0,
                src1,
                dst,
                k,
                rows,
                cols);
            break;
        case -101:
            hipLaunchKernelGGL(
                (q5_combined96_repro_kernel<true, false, true>),
                grid,
                dim3(256, 1, 1),
                0,
                0,
                src0,
                src1,
                dst,
                k,
                rows,
                cols);
            break;
        case -4:
            hipLaunchKernelGGL(
                q5_full64_batched4_repro_kernel,
                grid,
                dim3(256, 1, 1),
                0,
                0,
                src0,
                src1,
                dst,
                k,
                rows,
                cols);
            break;
        case 0:
            hipLaunchKernelGGL(
                hrx_mul_mat_vec_q5_k_wmma16x16_vk64_padded44_w64_full64_f16acc_wg256_f32,
                grid,
                dim3(256, 1, 1),
                0,
                0,
                src0,
                src1,
                dst,
                k,
                rows,
                cols);
            break;
        case 1:
            hipLaunchKernelGGL(
                q5_full64_active_groups_repro_kernel<1>,
                grid,
                dim3(256, 1, 1),
                0,
                0,
                src0,
                src1,
                dst,
                k,
                rows,
                cols);
            break;
        case 4:
            hipLaunchKernelGGL(
                q5_full64_active_groups_repro_kernel<4>,
                grid,
                dim3(256, 1, 1),
                0,
                0,
                src0,
                src1,
                dst,
                k,
                rows,
                cols);
            break;
        case 8:
            hipLaunchKernelGGL(
                q5_full64_active_groups_repro_kernel<8>,
                grid,
                dim3(256, 1, 1),
                0,
                0,
                src0,
                src1,
                dst,
                k,
                rows,
                cols);
            break;
        case 16:
            hipLaunchKernelGGL(
                q5_full64_active_groups_repro_kernel<16>,
                grid,
                dim3(256, 1, 1),
                0,
                0,
                src0,
                src1,
                dst,
                k,
                rows,
                cols);
            break;
        case 12:
            hipLaunchKernelGGL(
                q5_full64_active_groups_repro_kernel<12>,
                grid,
                dim3(256, 1, 1),
                0,
                0,
                src0,
                src1,
                dst,
                k,
                rows,
                cols);
            break;
        default:
            std::fprintf(stderr, "unexpected active group variant: %d\n", active_groups);
            std::exit(2);
    }
}

static int run_case(int rows, int cols, int k, const case_config & config, int active_groups) {
    const int blocks_per_row = k / 256;
    std::vector<hrx_block_q5_K_wmma_vk128_lhs> h_src0(static_cast<size_t>(rows) * blocks_per_row);
    std::vector<float> h_src1(static_cast<size_t>(cols) * k);
    std::vector<float> h_dst(static_cast<size_t>(rows) * cols, -777.0f);
    fill_q5(h_src0, rows, blocks_per_row, config);
    fill_rhs(h_src1, k, cols, config);

    device_buffer<hrx_block_q5_K_wmma_vk128_lhs> d_src0(h_src0.size());
    device_buffer<float> d_src1(h_src1.size());
    device_buffer<float> d_dst(h_dst.size());
    HIP_CHECK(hipMemcpy(d_src0.ptr, h_src0.data(), h_src0.size() * sizeof(h_src0[0]), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_src1.ptr, h_src1.data(), h_src1.size() * sizeof(h_src1[0]), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_dst.ptr, h_dst.data(), h_dst.size() * sizeof(h_dst[0]), hipMemcpyHostToDevice));

    dim3 grid((rows + 63) / 64, (cols + 63) / 64, 1);
    launch_variant(
        active_groups,
        grid,
        d_src0.ptr,
        d_src1.ptr,
        d_dst.ptr,
        static_cast<long long>(k),
        static_cast<long long>(rows),
        static_cast<long long>(cols));
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_dst.data(), d_dst.ptr, h_dst.size() * sizeof(h_dst[0]), hipMemcpyDeviceToHost));

    const std::vector<float> ref = cpu_reference(h_src0, h_src1, k, rows, cols);
    size_t active_count = 0;
    size_t nan_count = 0;
    size_t inf_count = 0;
    size_t nan_first8 = 0;
    size_t nan_rest = 0;
    size_t inf_first8 = 0;
    size_t inf_rest = 0;
    size_t sentinel_count = 0;
    double max_abs = 0.0;
    double max_rel = 0.0;
    size_t max_idx = 0;
    for (size_t i = 0; i < h_dst.size(); ++i) {
        if (!output_is_active(i, rows, active_groups)) {
            continue;
        }
        ++active_count;
        const float actual = h_dst[i];
        const int row = static_cast<int>(i % static_cast<size_t>(rows));
        const int col = static_cast<int>(i / static_cast<size_t>(rows));
        const int row_local = row & 63;
        const int col_local = col & 63;
        const int group = (col_local >> 4) * 4 + (row_local >> 4);
        if (std::isnan(actual)) {
            ++nan_count;
            if (group < 8) {
                ++nan_first8;
            } else {
                ++nan_rest;
            }
            continue;
        }
        if (std::isinf(actual)) {
            ++inf_count;
            if (group < 8) {
                ++inf_first8;
            } else {
                ++inf_rest;
            }
            continue;
        }
        if (actual == -777.0f) {
            ++sentinel_count;
        }
        const double diff = std::abs(static_cast<double>(actual) - static_cast<double>(ref[i]));
        const double denom = std::max(1.0, std::abs(static_cast<double>(ref[i])));
        const double rel = diff / denom;
        if (diff > max_abs) {
            max_abs = diff;
            max_rel = rel;
            max_idx = i;
        }
    }

    std::printf(
        "q5-full64-repro variant=%s profile=%s rows=%d cols=%d k=%d active=%zu nan=%zu nan_first8=%zu nan_rest=%zu inf=%zu inf_first8=%zu inf_rest=%zu sentinel=%zu max_abs=%g max_rel=%g idx=%zu actual=%g ref=%g\n",
        variant_name(active_groups),
        config.name,
        rows,
        cols,
        k,
        active_count,
        nan_count,
        nan_first8,
        nan_rest,
        inf_count,
        inf_first8,
        inf_rest,
        sentinel_count,
        max_abs,
        max_rel,
        max_idx,
        h_dst[max_idx],
        ref[max_idx]);
    return (nan_count == 0 && inf_count == 0 && sentinel_count == 0) ? 0 : 1;
}

int main() {
    const case_config small = {"small", 0x2000u, 0x0000u, 3, 0.00390625f};
    const case_config stress = {"stress", 0x3400u, 0x2c00u, 31, 0.015625f};
    int status = 0;
    status |= run_case(64, 33, 256, small, 0);
    status |= run_case(64, 33, 512, small, 0);
    status |= run_case(64, 33, 3584, small, 0);
    status |= run_case(64, 64, 3584, small, 0);
    status |= run_case(128, 33, 3584, small, 0);
    status |= run_case(64, 33, 256, small, -96);
    status |= run_case(64, 33, 512, small, -96);
    status |= run_case(64, 33, 3584, small, -96);
    status |= run_case(64, 64, 3584, small, -96);
    status |= run_case(64, 33, 256, small, -97);
    status |= run_case(64, 33, 3584, small, -97);
    status |= run_case(64, 64, 3584, small, -97);
    status |= run_case(64, 33, 256, small, -98);
    status |= run_case(64, 33, 3584, small, -98);
    status |= run_case(64, 64, 3584, small, -98);
    status |= run_case(64, 33, 256, small, -99);
    status |= run_case(64, 33, 3584, small, -99);
    status |= run_case(64, 33, 256, small, -100);
    status |= run_case(64, 33, 3584, small, -100);
    status |= run_case(64, 64, 3584, small, -100);
    status |= run_case(64, 33, 256, small, -101);
    status |= run_case(64, 33, 3584, small, -101);
    status |= run_case(64, 33, 256, small, -102);
    status |= run_case(64, 33, 3584, small, -102);
    status |= run_case(64, 64, 3584, small, -102);
    status |= run_case(64, 33, 256, small, -103);
    status |= run_case(64, 33, 3584, small, -103);
    status |= run_case(64, 64, 3584, small, -103);
    status |= run_case(64, 33, 3584, small, 1);
    status |= run_case(64, 33, 3584, small, 4);
    status |= run_case(64, 33, 3584, small, 8);
    status |= run_case(64, 33, 3584, small, 12);
    status |= run_case(64, 33, 3584, small, 16);
    status |= run_case(64, 33, 3584, small, -4);
    status |= run_case(64, 64, 3584, small, -4);
    status |= run_case(64, 33, 256, stress, 0);
    status |= run_case(64, 33, 512, stress, 0);
    status |= run_case(64, 33, 3584, stress, 0);
    status |= run_case(64, 64, 3584, stress, 0);
    status |= run_case(128, 33, 3584, stress, 0);
    status |= run_case(64, 33, 256, stress, -96);
    status |= run_case(64, 33, 3584, stress, -96);
    return status;
}
