#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define HRX_Q8_0_WMMA_VK128_EXPORT hrx_q8_0_wmma_repro_unused_route
#define HRX_Q8_0_WMMA_VK128_BUFFER_STORE 1
#define HRX_Q8_0_WMMA_VK128_STORE_STAGE_FAST_HALF 1
#define HRX_Q8_0_WMMA_VK128_STORE_STAGE_FAST_HALF_SPLIT_SELECTED 1
#include "../../kernels/mul_mat_vec_q8_0_wmma16_vk128_wg256.hip.cpp"

#define HIP_CHECK(expr) do { \
    hipError_t _err = (expr); \
    if (_err != hipSuccess) { \
        std::fprintf(stderr, "%s:%d: HIP error: %s\n", __FILE__, __LINE__, hipGetErrorString(_err)); \
        std::exit(2); \
    } \
} while (0)

static __device__ __forceinline__ void q8_repro_raw_store_slot(
        __amdgpu_buffer_rsrc_t dst_rsrc,
        long long rows_stride,
        long long row_base,
        long long col_base,
        long long rows,
        long long cols,
        const hrx_q8_0_wmma_vk128_half8_vec * acc,
        int acc_index,
        int group,
        int slot,
        unsigned int lane) {
    const int row_lane = static_cast<int>(lane >> 4);
    const int col_lane = static_cast<int>(lane & 15u);
    const long long row = row_base + static_cast<long long>((group & 3) * 16 + row_lane + slot * 4);
    const long long col = col_base + static_cast<long long>(((group >> 2) & 3) * 16 + col_lane);
    if (row < rows && col < cols) {
        hrx_q8_0_wmma_vk128_buffer_store_f32(
            dst_rsrc,
            col * rows_stride + row,
            static_cast<float>(acc[acc_index][slot * 2 + HRX_Q8_0_WMMA_VK128_W64_OPSEL]));
    }
}

static __device__ __forceinline__ void q8_repro_raw_store_value(
        __amdgpu_buffer_rsrc_t dst_rsrc,
        long long rows_stride,
        long long row_base,
        long long col_base,
        long long rows,
        long long cols,
        int group,
        int slot,
        unsigned int lane,
        float value) {
    const int row_lane = static_cast<int>(lane >> 4);
    const int col_lane = static_cast<int>(lane & 15u);
    const long long row = row_base + static_cast<long long>((group & 3) * 16 + row_lane + slot * 4);
    const long long col = col_base + static_cast<long long>(((group >> 2) & 3) * 16 + col_lane);
    if (row < rows && col < cols) {
        hrx_q8_0_wmma_vk128_buffer_store_f32(
            dst_rsrc,
            col * rows_stride + row,
            value);
    }
}

static constexpr int Q8_REPRO_CONTRACT_LANES = 64;
static constexpr int Q8_REPRO_CONTRACT_SLOTS = 4;
static constexpr int Q8_REPRO_CONTRACT_GROUPS = 48;
static constexpr int Q8_REPRO_CONTRACT_VALUES =
    Q8_REPRO_CONTRACT_GROUPS * Q8_REPRO_CONTRACT_SLOTS * Q8_REPRO_CONTRACT_LANES;

static __host__ __device__ __forceinline__ int q8_repro_contract_index(
        int group,
        int slot,
        unsigned int lane) {
    return (group * Q8_REPRO_CONTRACT_SLOTS + slot) * Q8_REPRO_CONTRACT_LANES + static_cast<int>(lane);
}

static __host__ __device__ __forceinline__ float q8_repro_contract_synthetic_value(
        int group,
        int slot,
        unsigned int lane) {
    const int bits = static_cast<int>(
        (static_cast<unsigned int>(group) * 1009u + static_cast<unsigned int>(slot) * 131u + lane * 17u) & 0x7fffu);
    return static_cast<float>(bits - 16384) * 0.001953125f;
}

static __device__ __forceinline__ void q8_repro_contract_store_value(
        __amdgpu_buffer_rsrc_t contract_rsrc,
        int group,
        int slot,
        unsigned int lane,
        float value) {
    const int index = q8_repro_contract_index(group, slot, lane);
    __builtin_amdgcn_raw_buffer_store_b32(
        __builtin_bit_cast(int, value),
        contract_rsrc,
        index * static_cast<int>(sizeof(float)),
        0,
        0);
}

static __device__ __forceinline__ void q8_repro_contract_store_acc(
        __amdgpu_buffer_rsrc_t contract_rsrc,
        long long rows,
        long long cols,
        const hrx_q8_0_wmma_vk128_half8_vec * acc,
        int acc_index,
        int group,
        int slot,
        unsigned int lane) {
    const int row_lane = static_cast<int>(lane >> 4);
    const int col_lane = static_cast<int>(lane & 15u);
    const long long row = static_cast<long long>((group & 3) * 16 + row_lane + slot * 4);
    const long long col = static_cast<long long>(((group >> 2) & 3) * 16 + col_lane);
    if (row < rows && col < cols) {
        q8_repro_contract_store_value(
            contract_rsrc,
            group,
            slot,
            lane,
            static_cast<float>(acc[acc_index][slot * 2 + HRX_Q8_0_WMMA_VK128_W64_OPSEL]));
    }
}

static __device__ __forceinline__ void q8_repro_contract_store_synthetic(
        __amdgpu_buffer_rsrc_t contract_rsrc,
        int group,
        int slot,
        unsigned int lane) {
    q8_repro_contract_store_value(
        contract_rsrc,
        group,
        slot,
        lane,
        q8_repro_contract_synthetic_value(group, slot, lane));
}

static __device__ __forceinline__ hrx_q8_0_wmma_vk128_half8_vec q8_repro_copy_acc(
        hrx_q8_0_wmma_vk128_half8_vec acc);

static __device__ __forceinline__ void q8_repro_selected_only_stage_store_slot(
        __amdgpu_buffer_rsrc_t dst_rsrc,
        long long rows_stride,
        long long row0,
        long long col0,
        hrx_q8_0_wmma_vk128_half8_vec acc,
        unsigned int lane,
        hrx_q8_0_wmma_vk128_lds_volatile_half_ptr sh_store) {
    constexpr int SELECTED_OPSEL = HRX_Q8_0_WMMA_VK128_W64_OPSEL;
    const int row_lane = static_cast<int>(lane >> 4);
    const int col_lane = static_cast<int>(lane & 15u);
    const int col_major_base = col_lane * 16 + row_lane;
    const long long col = col0 + static_cast<long long>(col_lane);
    hrx_q8_0_wmma_vk128_lds_u16_ptr sh_u16 =
        (hrx_q8_0_wmma_vk128_lds_u16_ptr) sh_store;

#pragma unroll
    for (int reg = 0; reg < 4; ++reg) {
        const int offset = col_major_base + reg * 4;
        hrx_q8_0_wmma_vk128_ds_store_u16(
            sh_u16 + offset,
            hrx_q8_0_wmma_vk128_f16_to_u16(acc[reg * 2 + SELECTED_OPSEL]));
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");

#pragma unroll
    for (int reg = 0; reg < 4; ++reg) {
        const int offset = col_major_base + reg * 4;
        const _Float16 selected = hrx_q8_0_wmma_vk128_u16_to_f16(
            hrx_q8_0_wmma_vk128_ds_load_u16_d16(sh_u16 + offset));
        asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
        const long long row = row0 + static_cast<long long>(row_lane + reg * 4);
        hrx_q8_0_wmma_vk128_buffer_store_f32(
            dst_rsrc,
            col * rows_stride + row,
            static_cast<float>(selected));
    }
}

static __device__ __forceinline__ void q8_repro_selected_only_stage_store_slot_regcopy(
        __amdgpu_buffer_rsrc_t dst_rsrc,
        long long rows_stride,
        long long row0,
        long long col0,
        hrx_q8_0_wmma_vk128_half8_vec acc,
        unsigned int lane,
        hrx_q8_0_wmma_vk128_lds_volatile_half_ptr sh_store) {
    constexpr int SELECTED_OPSEL = HRX_Q8_0_WMMA_VK128_W64_OPSEL;
    const int row_lane = static_cast<int>(lane >> 4);
    const int col_lane = static_cast<int>(lane & 15u);
    const int col_major_base = col_lane * 16 + row_lane;
    const long long col = col0 + static_cast<long long>(col_lane);
    hrx_q8_0_wmma_vk128_lds_u16_ptr sh_u16 =
        (hrx_q8_0_wmma_vk128_lds_u16_ptr) sh_store;

#pragma unroll
    for (int reg = 0; reg < 4; ++reg) {
        const hrx_q8_0_wmma_vk128_half8_vec acc_copy = q8_repro_copy_acc(acc);
        const int offset = col_major_base + reg * 4;
        hrx_q8_0_wmma_vk128_ds_store_u16(
            sh_u16 + offset,
            hrx_q8_0_wmma_vk128_f16_to_u16(acc_copy[reg * 2 + SELECTED_OPSEL]));
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");

#pragma unroll
    for (int reg = 0; reg < 4; ++reg) {
        const int offset = col_major_base + reg * 4;
        const _Float16 selected = hrx_q8_0_wmma_vk128_u16_to_f16(
            hrx_q8_0_wmma_vk128_ds_load_u16_d16(sh_u16 + offset));
        asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
        const long long row = row0 + static_cast<long long>(row_lane + reg * 4);
        hrx_q8_0_wmma_vk128_buffer_store_f32(
            dst_rsrc,
            col * rows_stride + row,
            static_cast<float>(selected));
    }
}

static __device__ __forceinline__ void q8_repro_consume_frag(
        hrx_q8_0_wmma_vk128_half16_vec frag) {
    asm volatile("" :: "v"(frag[0]), "v"(frag[4]), "v"(frag[8]), "v"(frag[12]) : "memory");
}

static __device__ __forceinline__ hrx_q8_0_wmma_vk128_half16_vec q8_repro_copy_frag(
        hrx_q8_0_wmma_vk128_half16_vec frag) {
    const hrx_q8_0_wmma_vk128_u32x8_vec in =
        __builtin_bit_cast(hrx_q8_0_wmma_vk128_u32x8_vec, frag);
    hrx_q8_0_wmma_vk128_u32x8_vec out;
    asm volatile("v_mov_b32 %0, %8\n\t"
                 "v_mov_b32 %1, %9\n\t"
                 "v_mov_b32 %2, %10\n\t"
                 "v_mov_b32 %3, %11\n\t"
                 "v_mov_b32 %4, %12\n\t"
                 "v_mov_b32 %5, %13\n\t"
                 "v_mov_b32 %6, %14\n\t"
                 "v_mov_b32 %7, %15\n\t"
                 : "=v"(out[0]), "=v"(out[1]), "=v"(out[2]), "=v"(out[3]),
                   "=v"(out[4]), "=v"(out[5]), "=v"(out[6]), "=v"(out[7])
                 : "v"(in[0]), "v"(in[1]), "v"(in[2]), "v"(in[3]),
                   "v"(in[4]), "v"(in[5]), "v"(in[6]), "v"(in[7])
                 : "memory");
    return __builtin_bit_cast(hrx_q8_0_wmma_vk128_half16_vec, out);
}

static __device__ __forceinline__ hrx_q8_0_wmma_vk128_half8_vec q8_repro_copy_acc(
        hrx_q8_0_wmma_vk128_half8_vec acc) {
    typedef uint32_t u32x4_vec __attribute__((ext_vector_type(4)));
    const u32x4_vec in = __builtin_bit_cast(u32x4_vec, acc);
    u32x4_vec out;
    asm volatile("v_mov_b32 %0, %4\n\t"
                 "v_mov_b32 %1, %5\n\t"
                 "v_mov_b32 %2, %6\n\t"
                 "v_mov_b32 %3, %7\n\t"
                 : "=v"(out[0]), "=v"(out[1]), "=v"(out[2]), "=v"(out[3])
                 : "v"(in[0]), "v"(in[1]), "v"(in[2]), "v"(in[3])
                 : "memory");
    return __builtin_bit_cast(hrx_q8_0_wmma_vk128_half8_vec, out);
}

__global__ __launch_bounds__(256, 1)
void q8_bfrag_dump_kernel(
        const float * src1,
        float * dump,
        long long k,
        long long cols) {
    constexpr int BN = 64;
    constexpr int BK = 32;
    constexpr int SHARED_STRIDE = HRX_Q8_0_WMMA_VK128_SHARED_STRIDE;

    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int wave = tid >> 6u;
    const unsigned int lane = tid & 63u;
    __shared__ _Float16 sh_b[BN * SHARED_STRIDE];

    const _Float16 zero = static_cast<_Float16>(0.0f);
    for (int idx = static_cast<int>(tid); idx < BN * BK; idx += 256) {
        const int c = idx / BK;
        const int kk = idx - c * BK;
        sh_b[c * SHARED_STRIDE + kk] = c < cols ? static_cast<_Float16>(src1[c * k + kk]) : zero;
    }
    __syncthreads();

    if (wave == 0) {
        hrx_q8_0_wmma_vk128_lds_half_ptr sh_b_lds =
            (hrx_q8_0_wmma_vk128_lds_half_ptr) sh_b;
        hrx_q8_0_wmma_vk128_half16_vec b_frag[2][4];
#pragma unroll
        for (int k_tile = 0; k_tile < 2; ++k_tile) {
#pragma unroll
            for (int col_sub = 0; col_sub < 4; ++col_sub) {
                b_frag[k_tile][col_sub] =
                    hrx_q8_0_wmma_vk128_load_b_frag_w64_b64asm_nowait(
                        sh_b_lds, col_sub, k_tile, lane);
            }
        }
        asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
#pragma unroll
        for (int k_tile = 0; k_tile < 2; ++k_tile) {
#pragma unroll
            for (int col_sub = 0; col_sub < 4; ++col_sub) {
#pragma unroll
                for (int elem = 0; elem < 16; ++elem) {
                    const int index = (((k_tile * 4 + col_sub) * 64 + static_cast<int>(lane)) * 16) + elem;
                    dump[index] = static_cast<float>(b_frag[k_tile][col_sub][elem]);
                }
            }
        }
    }
}

template <bool full_b>
__global__ __launch_bounds__(256, 1)
void q8_array8_repro_kernel(
        const hrx_block_q8_0_wmma_vk128_lhs * src0,
        const float * src1,
        float * dst,
        long long k,
        long long rows,
        long long cols) {
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 32;
    constexpr int SHARED_STRIDE = HRX_Q8_0_WMMA_VK128_SHARED_STRIDE;
    constexpr int ACTIVE_GROUPS = 8;

    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int wave = tid >> 6u;
    const unsigned int lane = tid & 63u;
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
    hrx_q8_0_wmma_vk128_half8_vec acc[ACTIVE_GROUPS] = {};

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
            sh_b[c * SHARED_STRIDE + kk] = col < cols ? static_cast<_Float16>(src1[col * k + k0 + kk]) : zero;
        }
        __syncthreads();

        if (wave == 0) {
            hrx_q8_0_wmma_vk128_lds_half_ptr sh_a_lds =
                (hrx_q8_0_wmma_vk128_lds_half_ptr) sh_a;
            hrx_q8_0_wmma_vk128_lds_half_ptr sh_b_lds =
                (hrx_q8_0_wmma_vk128_lds_half_ptr) sh_b;
            hrx_q8_0_wmma_vk128_half16_vec a_frag[2][4];
            hrx_q8_0_wmma_vk128_half16_vec b_frag[2][4];
#pragma unroll
            for (int k_tile = 0; k_tile < 2; ++k_tile) {
#pragma unroll
                for (int row_sub = 0; row_sub < 4; ++row_sub) {
                    a_frag[k_tile][row_sub] =
                        hrx_q8_0_wmma_vk128_load_a_frag_w64_b64asm_nowait(
                            sh_a_lds, row_sub, k_tile, lane);
                }
#pragma unroll
                for (int col_sub = 0; col_sub < 2; ++col_sub) {
                    b_frag[k_tile][col_sub] =
                        hrx_q8_0_wmma_vk128_load_b_frag_w64_b64asm_nowait(
                            sh_b_lds, col_sub, k_tile, lane);
                }
                if constexpr (full_b) {
#pragma unroll
                    for (int col_sub = 2; col_sub < 4; ++col_sub) {
                        b_frag[k_tile][col_sub] =
                            hrx_q8_0_wmma_vk128_load_b_frag_w64_b64asm_nowait(
                                sh_b_lds, col_sub, k_tile, lane);
                        q8_repro_consume_frag(b_frag[k_tile][col_sub]);
                    }
                }
            }
            asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
#pragma unroll
            for (int k_tile = 0; k_tile < 2; ++k_tile) {
#pragma unroll
                for (int col_sub = 0; col_sub < 2; ++col_sub) {
#pragma unroll
                    for (int row_sub = 0; row_sub < 4; ++row_sub) {
                        const int group = col_sub * 4 + row_sub;
                        acc[group] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(
                            a_frag[k_tile][row_sub],
                            b_frag[k_tile][col_sub],
                            acc[group],
                            HRX_Q8_0_WMMA_VK128_W64_OPSEL != 0);
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
                q8_repro_raw_store_slot(dst_rsrc, rows, row_base, col_base, rows, cols, acc, group, group, slot, lane);
            }
        }
    }
}

template <bool copy_a, bool copy_b>
__global__ __launch_bounds__(256, 1)
void q8_array16_direct_raw_repro_kernel(
        const hrx_block_q8_0_wmma_vk128_lhs * src0,
        const float * src1,
        float * dst,
        long long k,
        long long rows,
        long long cols) {
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 32;
    constexpr int SHARED_STRIDE = HRX_Q8_0_WMMA_VK128_SHARED_STRIDE;
    constexpr int ACTIVE_GROUPS = 16;

    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int wave = tid >> 6u;
    const unsigned int lane = tid & 63u;
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
    hrx_q8_0_wmma_vk128_half8_vec acc[ACTIVE_GROUPS] = {};

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
            sh_b[c * SHARED_STRIDE + kk] = col < cols ? static_cast<_Float16>(src1[col * k + k0 + kk]) : zero;
        }
        __syncthreads();

        if (wave == 0) {
            hrx_q8_0_wmma_vk128_lds_half_ptr sh_a_lds =
                (hrx_q8_0_wmma_vk128_lds_half_ptr) sh_a;
            hrx_q8_0_wmma_vk128_lds_half_ptr sh_b_lds =
                (hrx_q8_0_wmma_vk128_lds_half_ptr) sh_b;
            hrx_q8_0_wmma_vk128_half16_vec a_frag[2][4];
            hrx_q8_0_wmma_vk128_half16_vec b_frag[2][4];
#pragma unroll
            for (int k_tile = 0; k_tile < 2; ++k_tile) {
#pragma unroll
                for (int row_sub = 0; row_sub < 4; ++row_sub) {
                    a_frag[k_tile][row_sub] =
                        hrx_q8_0_wmma_vk128_load_a_frag_w64_b64asm_nowait(
                            sh_a_lds, row_sub, k_tile, lane);
                }
#pragma unroll
                for (int col_sub = 0; col_sub < 4; ++col_sub) {
                    b_frag[k_tile][col_sub] =
                        hrx_q8_0_wmma_vk128_load_b_frag_w64_b64asm_nowait(
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
                        const int group = col_sub * 4 + row_sub;
                        const hrx_q8_0_wmma_vk128_half16_vec a_use = copy_a ?
                            q8_repro_copy_frag(a_frag[k_tile][row_sub]) :
                            a_frag[k_tile][row_sub];
                        const hrx_q8_0_wmma_vk128_half16_vec b_use = copy_b ?
                            q8_repro_copy_frag(b_frag[k_tile][col_sub]) :
                            b_frag[k_tile][col_sub];
                        acc[group] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(
                            a_use,
                            b_use,
                            acc[group],
                            HRX_Q8_0_WMMA_VK128_W64_OPSEL != 0);
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
                q8_repro_raw_store_slot(dst_rsrc, rows, row_base, col_base, rows, cols, acc, group, group, slot, lane);
            }
        }
    }
}

template <bool copy_a, bool copy_b>
__global__ __launch_bounds__(256, 1)
void q8_contract_direct192_repro_kernel(
        const hrx_block_q8_0_wmma_vk128_lhs * src0,
        const float * src1,
        float * contract,
        long long k,
        long long rows,
        long long cols) {
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 32;
    constexpr int SHARED_STRIDE = HRX_Q8_0_WMMA_VK128_SHARED_STRIDE;
    constexpr int ACTIVE_GROUPS = 16;

    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int wave = tid >> 6u;
    const unsigned int lane = tid & 63u;
    const long long row_base = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * BM;
    const long long col_base = static_cast<long long>(__builtin_amdgcn_workgroup_id_y()) * BN;
    if (row_base >= rows || col_base >= cols) {
        return;
    }

    const __amdgpu_buffer_rsrc_t contract_rsrc = hrx_q8_0_wmma_vk128_make_dst_rsrc(contract);
    __shared__ _Float16 sh_a[BM * SHARED_STRIDE];
    __shared__ _Float16 sh_b[BN * SHARED_STRIDE];

    const long long blocks_per_row = k / 32;
    const _Float16 zero = static_cast<_Float16>(0.0f);
    hrx_q8_0_wmma_vk128_half8_vec acc[ACTIVE_GROUPS] = {};

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
            sh_b[c * SHARED_STRIDE + kk] = col < cols ? static_cast<_Float16>(src1[col * k + k0 + kk]) : zero;
        }
        __syncthreads();

        if (wave == 0) {
            hrx_q8_0_wmma_vk128_lds_half_ptr sh_a_lds =
                (hrx_q8_0_wmma_vk128_lds_half_ptr) sh_a;
            hrx_q8_0_wmma_vk128_lds_half_ptr sh_b_lds =
                (hrx_q8_0_wmma_vk128_lds_half_ptr) sh_b;
            hrx_q8_0_wmma_vk128_half16_vec a_frag[2][4];
            hrx_q8_0_wmma_vk128_half16_vec b_frag[2][4];
#pragma unroll
            for (int k_tile = 0; k_tile < 2; ++k_tile) {
#pragma unroll
                for (int row_sub = 0; row_sub < 4; ++row_sub) {
                    a_frag[k_tile][row_sub] =
                        hrx_q8_0_wmma_vk128_load_a_frag_w64_b64asm_nowait(
                            sh_a_lds, row_sub, k_tile, lane);
                }
#pragma unroll
                for (int col_sub = 0; col_sub < 4; ++col_sub) {
                    b_frag[k_tile][col_sub] =
                        hrx_q8_0_wmma_vk128_load_b_frag_w64_b64asm_nowait(
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
                        const int group = col_sub * 4 + row_sub;
                        const hrx_q8_0_wmma_vk128_half16_vec a_use = copy_a ?
                            q8_repro_copy_frag(a_frag[k_tile][row_sub]) :
                            a_frag[k_tile][row_sub];
                        const hrx_q8_0_wmma_vk128_half16_vec b_use = copy_b ?
                            q8_repro_copy_frag(b_frag[k_tile][col_sub]) :
                            b_frag[k_tile][col_sub];
                        acc[group] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(
                            a_use,
                            b_use,
                            acc[group],
                            HRX_Q8_0_WMMA_VK128_W64_OPSEL != 0);
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
                q8_repro_contract_store_acc(contract_rsrc, rows, cols, acc, group, group, slot, lane);
            }
        }
#pragma unroll
        for (int group = 16; group < Q8_REPRO_CONTRACT_GROUPS; ++group) {
#pragma unroll
            for (int slot = 0; slot < Q8_REPRO_CONTRACT_SLOTS; ++slot) {
                q8_repro_contract_store_synthetic(contract_rsrc, group, slot, lane);
            }
        }
    }
}

template <
    int group_base,
    int col_start,
    int col_count,
    int active_groups,
    bool consume_unused_b,
    bool copy_a = false,
    bool copy_b = false,
    bool stage_store = false,
    bool selected_only_stage = false,
    bool copy_acc_before_stage = false,
    bool copy_acc_per_stage_reg = false>
__global__ __launch_bounds__(256, 1)
void q8_array_fullb_phase_repro_kernel(
        const hrx_block_q8_0_wmma_vk128_lhs * src0,
        const float * src1,
        float * dst,
        long long k,
        long long rows,
        long long cols) {
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 32;
    constexpr int SHARED_STRIDE = HRX_Q8_0_WMMA_VK128_SHARED_STRIDE;
    static_assert(active_groups >= 1 && active_groups <= 8, "unexpected phase size");
    static_assert(col_count >= 1 && col_count <= 2, "unexpected column group count");

    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int wave = tid >> 6u;
    const unsigned int lane = tid & 63u;
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
    hrx_q8_0_wmma_vk128_half8_vec acc[active_groups] = {};

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
            sh_b[c * SHARED_STRIDE + kk] = col < cols ? static_cast<_Float16>(src1[col * k + k0 + kk]) : zero;
        }
        __syncthreads();

        if (wave == 0) {
            hrx_q8_0_wmma_vk128_lds_half_ptr sh_a_lds =
                (hrx_q8_0_wmma_vk128_lds_half_ptr) sh_a;
            hrx_q8_0_wmma_vk128_lds_half_ptr sh_b_lds =
                (hrx_q8_0_wmma_vk128_lds_half_ptr) sh_b;
            hrx_q8_0_wmma_vk128_half16_vec a_frag[2][4];
            hrx_q8_0_wmma_vk128_half16_vec b_frag[2][4];
#pragma unroll
            for (int k_tile = 0; k_tile < 2; ++k_tile) {
#pragma unroll
                for (int row_sub = 0; row_sub < 4; ++row_sub) {
                    a_frag[k_tile][row_sub] =
                        hrx_q8_0_wmma_vk128_load_a_frag_w64_b64asm_nowait(
                            sh_a_lds, row_sub, k_tile, lane);
                }
#pragma unroll
                for (int col_sub = 0; col_sub < 4; ++col_sub) {
                    b_frag[k_tile][col_sub] =
                        hrx_q8_0_wmma_vk128_load_b_frag_w64_b64asm_nowait(
                            sh_b_lds, col_sub, k_tile, lane);
                    if constexpr (consume_unused_b) {
                        if (col_sub < col_start || col_sub >= col_start + col_count) {
                            q8_repro_consume_frag(b_frag[k_tile][col_sub]);
                        }
                    }
                }
            }
            asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
#pragma unroll
            for (int k_tile = 0; k_tile < 2; ++k_tile) {
#pragma unroll
                for (int col_delta = 0; col_delta < col_count; ++col_delta) {
                    const int col_sub = col_start + col_delta;
#pragma unroll
                    for (int row_sub = 0; row_sub < 4; ++row_sub) {
                        const int local = col_delta * 4 + row_sub;
                        const int group = group_base + local;
                        const hrx_q8_0_wmma_vk128_half16_vec a_use = copy_a ?
                            q8_repro_copy_frag(a_frag[k_tile][row_sub]) :
                            a_frag[k_tile][row_sub];
                        const hrx_q8_0_wmma_vk128_half16_vec b_use = copy_b ?
                            q8_repro_copy_frag(b_frag[k_tile][col_sub]) :
                            b_frag[k_tile][col_sub];
                        acc[local] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(
                            a_use,
                            b_use,
                            acc[local],
                            HRX_Q8_0_WMMA_VK128_W64_OPSEL != 0);
                    }
                }
            }
        }
        __syncthreads();
    }

    if (wave == 0) {
        if constexpr (stage_store) {
            __shared__ _Float16 sh_store[16 * 16];
#pragma unroll
            for (int local = 0; local < active_groups; ++local) {
                const int group = group_base + local;
                const long long row0 = row_base + static_cast<long long>((group & 3) * 16);
                const long long col0 = col_base + static_cast<long long>(((group >> 2) & 3) * 16);
                const hrx_q8_0_wmma_vk128_half8_vec acc_use = copy_acc_before_stage ?
                    q8_repro_copy_acc(acc[local]) :
                    acc[local];
                if constexpr (selected_only_stage) {
                    if constexpr (copy_acc_per_stage_reg) {
                        q8_repro_selected_only_stage_store_slot_regcopy(
                            dst_rsrc,
                            rows,
                            row0,
                            col0,
                            acc_use,
                            lane,
                            (hrx_q8_0_wmma_vk128_lds_volatile_half_ptr) sh_store);
                    } else {
                        q8_repro_selected_only_stage_store_slot(
                            dst_rsrc,
                            rows,
                            row0,
                            col0,
                            acc_use,
                            lane,
                            (hrx_q8_0_wmma_vk128_lds_volatile_half_ptr) sh_store);
                    }
                } else {
                    hrx_q8_0_wmma_vk128_store_acc_f16_row_major_w64_fast_half_buffer_split_selected(
                        dst_rsrc,
                        rows,
                        row0,
                        col0,
                        acc_use,
                        lane,
                        0,
                        (hrx_q8_0_wmma_vk128_lds_volatile_half_ptr) sh_store);
                }
            }
        } else {
#pragma unroll
            for (int local = 0; local < active_groups; ++local) {
                const int group = group_base + local;
#pragma unroll
                for (int slot = 0; slot < 4; ++slot) {
                    q8_repro_raw_store_slot(dst_rsrc, rows, row_base, col_base, rows, cols, acc, local, group, slot, lane);
                }
            }
        }
    }
}

template <
    int group_base,
    int col_start,
    int col_count,
    int active_groups,
    int synthetic_group_base,
    int synthetic_groups,
    bool copy_a,
    bool copy_b>
__global__ __launch_bounds__(256, 1)
void q8_contract_phase96_repro_kernel(
        const hrx_block_q8_0_wmma_vk128_lhs * src0,
        const float * src1,
        float * contract,
        long long k,
        long long rows,
        long long cols) {
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 32;
    constexpr int SHARED_STRIDE = HRX_Q8_0_WMMA_VK128_SHARED_STRIDE;
    static_assert(active_groups >= 1 && active_groups <= 8, "unexpected phase size");
    static_assert(col_count >= 1 && col_count <= 2, "unexpected column group count");
    static_assert(synthetic_groups >= 0 && synthetic_groups <= 16, "unexpected synthetic group count");

    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int wave = tid >> 6u;
    const unsigned int lane = tid & 63u;
    const long long row_base = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * BM;
    const long long col_base = static_cast<long long>(__builtin_amdgcn_workgroup_id_y()) * BN;
    if (row_base >= rows || col_base >= cols) {
        return;
    }

    const __amdgpu_buffer_rsrc_t contract_rsrc = hrx_q8_0_wmma_vk128_make_dst_rsrc(contract);
    __shared__ _Float16 sh_a[BM * SHARED_STRIDE];
    __shared__ _Float16 sh_b[BN * SHARED_STRIDE];

    const long long blocks_per_row = k / 32;
    const _Float16 zero = static_cast<_Float16>(0.0f);
    hrx_q8_0_wmma_vk128_half8_vec acc[active_groups] = {};

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
            sh_b[c * SHARED_STRIDE + kk] = col < cols ? static_cast<_Float16>(src1[col * k + k0 + kk]) : zero;
        }
        __syncthreads();

        if (wave == 0) {
            hrx_q8_0_wmma_vk128_lds_half_ptr sh_a_lds =
                (hrx_q8_0_wmma_vk128_lds_half_ptr) sh_a;
            hrx_q8_0_wmma_vk128_lds_half_ptr sh_b_lds =
                (hrx_q8_0_wmma_vk128_lds_half_ptr) sh_b;
            hrx_q8_0_wmma_vk128_half16_vec a_frag[2][4];
            hrx_q8_0_wmma_vk128_half16_vec b_frag[2][4];
#pragma unroll
            for (int k_tile = 0; k_tile < 2; ++k_tile) {
#pragma unroll
                for (int row_sub = 0; row_sub < 4; ++row_sub) {
                    a_frag[k_tile][row_sub] =
                        hrx_q8_0_wmma_vk128_load_a_frag_w64_b64asm_nowait(
                            sh_a_lds, row_sub, k_tile, lane);
                }
#pragma unroll
                for (int col_sub = 0; col_sub < 4; ++col_sub) {
                    b_frag[k_tile][col_sub] =
                        hrx_q8_0_wmma_vk128_load_b_frag_w64_b64asm_nowait(
                            sh_b_lds, col_sub, k_tile, lane);
                }
            }
            asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
#pragma unroll
            for (int k_tile = 0; k_tile < 2; ++k_tile) {
#pragma unroll
                for (int col_delta = 0; col_delta < col_count; ++col_delta) {
                    const int col_sub = col_start + col_delta;
#pragma unroll
                    for (int row_sub = 0; row_sub < 4; ++row_sub) {
                        const int local = col_delta * 4 + row_sub;
                        const hrx_q8_0_wmma_vk128_half16_vec a_use = copy_a ?
                            q8_repro_copy_frag(a_frag[k_tile][row_sub]) :
                            a_frag[k_tile][row_sub];
                        const hrx_q8_0_wmma_vk128_half16_vec b_use = copy_b ?
                            q8_repro_copy_frag(b_frag[k_tile][col_sub]) :
                            b_frag[k_tile][col_sub];
                        acc[local] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(
                            a_use,
                            b_use,
                            acc[local],
                            HRX_Q8_0_WMMA_VK128_W64_OPSEL != 0);
                    }
                }
            }
        }
        __syncthreads();
    }

    if (wave == 0) {
#pragma unroll
        for (int local = 0; local < active_groups; ++local) {
            const int group = group_base + local;
#pragma unroll
            for (int slot = 0; slot < Q8_REPRO_CONTRACT_SLOTS; ++slot) {
                q8_repro_contract_store_acc(contract_rsrc, rows, cols, acc, local, group, slot, lane);
            }
        }
#pragma unroll
        for (int group_delta = 0; group_delta < synthetic_groups; ++group_delta) {
            const int group = synthetic_group_base + group_delta;
#pragma unroll
            for (int slot = 0; slot < Q8_REPRO_CONTRACT_SLOTS; ++slot) {
                q8_repro_contract_store_synthetic(contract_rsrc, group, slot, lane);
            }
        }
    }
}

template <int group_id, bool consume_unused_b>
__global__ __launch_bounds__(256, 1)
void q8_single_group_repro_kernel(
        const hrx_block_q8_0_wmma_vk128_lhs * src0,
        const float * src1,
        float * dst,
        long long k,
        long long rows,
        long long cols) {
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 32;
    constexpr int SHARED_STRIDE = HRX_Q8_0_WMMA_VK128_SHARED_STRIDE;
    constexpr int row_sub = group_id & 3;
    constexpr int col_sub = (group_id >> 2) & 3;

    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int wave = tid >> 6u;
    const unsigned int lane = tid & 63u;
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
    hrx_q8_0_wmma_vk128_half8_vec acc[1] = {};

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
            sh_b[c * SHARED_STRIDE + kk] = col < cols ? static_cast<_Float16>(src1[col * k + k0 + kk]) : zero;
        }
        __syncthreads();

        if (wave == 0) {
            hrx_q8_0_wmma_vk128_lds_half_ptr sh_a_lds =
                (hrx_q8_0_wmma_vk128_lds_half_ptr) sh_a;
            hrx_q8_0_wmma_vk128_lds_half_ptr sh_b_lds =
                (hrx_q8_0_wmma_vk128_lds_half_ptr) sh_b;
            hrx_q8_0_wmma_vk128_half16_vec a_frag[2][4];
            hrx_q8_0_wmma_vk128_half16_vec b_frag[2][4];
#pragma unroll
            for (int k_tile = 0; k_tile < 2; ++k_tile) {
#pragma unroll
                for (int rs = 0; rs < 4; ++rs) {
                    a_frag[k_tile][rs] =
                        hrx_q8_0_wmma_vk128_load_a_frag_w64_b64asm_nowait(
                            sh_a_lds, rs, k_tile, lane);
                    if constexpr (consume_unused_b) {
                        if (rs != row_sub) {
                            q8_repro_consume_frag(a_frag[k_tile][rs]);
                        }
                    }
                }
#pragma unroll
                for (int cs = 0; cs < 4; ++cs) {
                    b_frag[k_tile][cs] =
                        hrx_q8_0_wmma_vk128_load_b_frag_w64_b64asm_nowait(
                            sh_b_lds, cs, k_tile, lane);
                    if constexpr (consume_unused_b) {
                        if (cs != col_sub) {
                            q8_repro_consume_frag(b_frag[k_tile][cs]);
                        }
                    }
                }
            }
            asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
#pragma unroll
            for (int k_tile = 0; k_tile < 2; ++k_tile) {
                acc[0] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(
                    a_frag[k_tile][row_sub],
                    b_frag[k_tile][col_sub],
                    acc[0],
                    HRX_Q8_0_WMMA_VK128_W64_OPSEL != 0);
            }
        }
        __syncthreads();
    }

    if (wave == 0) {
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            q8_repro_raw_store_slot(dst_rsrc, rows, row_base, col_base, rows, cols, acc, 0, group_id, slot, lane);
        }
    }
}

template <
    int compute_group,
    int store_group,
    bool copy_a = false,
    bool copy_b = false,
    bool stage_store = false,
    bool selected_only_stage = false>
__global__ __launch_bounds__(256, 1)
void q8_single_group_remap_repro_kernel(
        const hrx_block_q8_0_wmma_vk128_lhs * src0,
        const float * src1,
        float * dst,
        long long k,
        long long rows,
        long long cols) {
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 32;
    constexpr int SHARED_STRIDE = HRX_Q8_0_WMMA_VK128_SHARED_STRIDE;
    constexpr int row_sub = compute_group & 3;
    constexpr int col_sub = (compute_group >> 2) & 3;

    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int wave = tid >> 6u;
    const unsigned int lane = tid & 63u;
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
    hrx_q8_0_wmma_vk128_half8_vec acc[1] = {};

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
            sh_b[c * SHARED_STRIDE + kk] = col < cols ? static_cast<_Float16>(src1[col * k + k0 + kk]) : zero;
        }
        __syncthreads();

        if (wave == 0) {
            hrx_q8_0_wmma_vk128_lds_half_ptr sh_a_lds =
                (hrx_q8_0_wmma_vk128_lds_half_ptr) sh_a;
            hrx_q8_0_wmma_vk128_lds_half_ptr sh_b_lds =
                (hrx_q8_0_wmma_vk128_lds_half_ptr) sh_b;
            hrx_q8_0_wmma_vk128_half16_vec a_frag[2][4];
            hrx_q8_0_wmma_vk128_half16_vec b_frag[2][4];
#pragma unroll
            for (int k_tile = 0; k_tile < 2; ++k_tile) {
#pragma unroll
                for (int rs = 0; rs < 4; ++rs) {
                    a_frag[k_tile][rs] =
                        hrx_q8_0_wmma_vk128_load_a_frag_w64_b64asm_nowait(
                            sh_a_lds, rs, k_tile, lane);
                }
#pragma unroll
                for (int cs = 0; cs < 4; ++cs) {
                    b_frag[k_tile][cs] =
                        hrx_q8_0_wmma_vk128_load_b_frag_w64_b64asm_nowait(
                            sh_b_lds, cs, k_tile, lane);
                }
            }
            asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
#pragma unroll
            for (int k_tile = 0; k_tile < 2; ++k_tile) {
                const hrx_q8_0_wmma_vk128_half16_vec a_use = copy_a ?
                    q8_repro_copy_frag(a_frag[k_tile][row_sub]) :
                    a_frag[k_tile][row_sub];
                const hrx_q8_0_wmma_vk128_half16_vec b_use = copy_b ?
                    q8_repro_copy_frag(b_frag[k_tile][col_sub]) :
                    b_frag[k_tile][col_sub];
                acc[0] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(
                    a_use,
                    b_use,
                    acc[0],
                    HRX_Q8_0_WMMA_VK128_W64_OPSEL != 0);
            }
        }
        __syncthreads();
    }

    if (wave == 0) {
        if constexpr (stage_store) {
            __shared__ _Float16 sh_store[16 * 16];
            const long long row0 = row_base + static_cast<long long>((store_group & 3) * 16);
            const long long col0 = col_base + static_cast<long long>(((store_group >> 2) & 3) * 16);
            if constexpr (selected_only_stage) {
                q8_repro_selected_only_stage_store_slot(
                    dst_rsrc,
                    rows,
                    row0,
                    col0,
                    acc[0],
                    lane,
                    (hrx_q8_0_wmma_vk128_lds_volatile_half_ptr) sh_store);
            } else {
                hrx_q8_0_wmma_vk128_store_acc_f16_row_major_w64_fast_half_buffer_split_selected(
                    dst_rsrc,
                    rows,
                    row0,
                    col0,
                    acc[0],
                    lane,
                    0,
                    (hrx_q8_0_wmma_vk128_lds_volatile_half_ptr) sh_store);
            }
        } else {
#pragma unroll
            for (int slot = 0; slot < 4; ++slot) {
                q8_repro_raw_store_slot(dst_rsrc, rows, row_base, col_base, rows, cols, acc, 0, store_group, slot, lane);
            }
        }
    }
}

template <int group_id, bool copy_a, bool copy_b, bool raw_first>
__global__ __launch_bounds__(256, 1)
void q8_single_group_dual_stage_repro_kernel(
        const hrx_block_q8_0_wmma_vk128_lhs * src0,
        const float * src1,
        float * raw_dst,
        float * staged_dst,
        long long k,
        long long rows,
        long long cols) {
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 32;
    constexpr int SHARED_STRIDE = HRX_Q8_0_WMMA_VK128_SHARED_STRIDE;
    constexpr int row_sub = group_id & 3;
    constexpr int col_sub = (group_id >> 2) & 3;

    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int wave = tid >> 6u;
    const unsigned int lane = tid & 63u;
    const long long row_base = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * BM;
    const long long col_base = static_cast<long long>(__builtin_amdgcn_workgroup_id_y()) * BN;
    if (row_base >= rows || col_base >= cols) {
        return;
    }

    const __amdgpu_buffer_rsrc_t raw_rsrc = hrx_q8_0_wmma_vk128_make_dst_rsrc(raw_dst);
    const __amdgpu_buffer_rsrc_t staged_rsrc = hrx_q8_0_wmma_vk128_make_dst_rsrc(staged_dst);
    __shared__ _Float16 sh_a[BM * SHARED_STRIDE];
    __shared__ _Float16 sh_b[BN * SHARED_STRIDE];

    const long long blocks_per_row = k / 32;
    const _Float16 zero = static_cast<_Float16>(0.0f);
    hrx_q8_0_wmma_vk128_half8_vec acc[1] = {};

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
            sh_b[c * SHARED_STRIDE + kk] = col < cols ? static_cast<_Float16>(src1[col * k + k0 + kk]) : zero;
        }
        __syncthreads();

        if (wave == 0) {
            hrx_q8_0_wmma_vk128_lds_half_ptr sh_a_lds =
                (hrx_q8_0_wmma_vk128_lds_half_ptr) sh_a;
            hrx_q8_0_wmma_vk128_lds_half_ptr sh_b_lds =
                (hrx_q8_0_wmma_vk128_lds_half_ptr) sh_b;
            hrx_q8_0_wmma_vk128_half16_vec a_frag[2][4];
            hrx_q8_0_wmma_vk128_half16_vec b_frag[2][4];
#pragma unroll
            for (int k_tile = 0; k_tile < 2; ++k_tile) {
#pragma unroll
                for (int rs = 0; rs < 4; ++rs) {
                    a_frag[k_tile][rs] =
                        hrx_q8_0_wmma_vk128_load_a_frag_w64_b64asm_nowait(
                            sh_a_lds, rs, k_tile, lane);
                }
#pragma unroll
                for (int cs = 0; cs < 4; ++cs) {
                    b_frag[k_tile][cs] =
                        hrx_q8_0_wmma_vk128_load_b_frag_w64_b64asm_nowait(
                            sh_b_lds, cs, k_tile, lane);
                }
            }
            asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
#pragma unroll
            for (int k_tile = 0; k_tile < 2; ++k_tile) {
                const hrx_q8_0_wmma_vk128_half16_vec a_use = copy_a ?
                    q8_repro_copy_frag(a_frag[k_tile][row_sub]) :
                    a_frag[k_tile][row_sub];
                const hrx_q8_0_wmma_vk128_half16_vec b_use = copy_b ?
                    q8_repro_copy_frag(b_frag[k_tile][col_sub]) :
                    b_frag[k_tile][col_sub];
                acc[0] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(
                    a_use,
                    b_use,
                    acc[0],
                    HRX_Q8_0_WMMA_VK128_W64_OPSEL != 0);
            }
        }
        __syncthreads();
    }

    if (wave == 0) {
        __shared__ _Float16 sh_store[16 * 16];
        const long long row0 = row_base + static_cast<long long>((group_id & 3) * 16);
        const long long col0 = col_base + static_cast<long long>(((group_id >> 2) & 3) * 16);
        if constexpr (raw_first) {
#pragma unroll
            for (int slot = 0; slot < 4; ++slot) {
                q8_repro_raw_store_slot(raw_rsrc, rows, row_base, col_base, rows, cols, acc, 0, group_id, slot, lane);
            }
            q8_repro_selected_only_stage_store_slot(
                staged_rsrc,
                rows,
                row0,
                col0,
                acc[0],
                lane,
                (hrx_q8_0_wmma_vk128_lds_volatile_half_ptr) sh_store);
        } else {
            q8_repro_selected_only_stage_store_slot(
                staged_rsrc,
                rows,
                row0,
                col0,
                acc[0],
                lane,
                (hrx_q8_0_wmma_vk128_lds_volatile_half_ptr) sh_store);
#pragma unroll
            for (int slot = 0; slot < 4; ++slot) {
                q8_repro_raw_store_slot(raw_rsrc, rows, row_base, col_base, rows, cols, acc, 0, group_id, slot, lane);
            }
        }
    }
}

template <int group_id, int mirror_col_sub>
__global__ __launch_bounds__(256, 1)
void q8_single_group_bmirror_repro_kernel(
        const hrx_block_q8_0_wmma_vk128_lhs * src0,
        const float * src1,
        float * dst,
        long long k,
        long long rows,
        long long cols) {
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 32;
    constexpr int SHARED_STRIDE = HRX_Q8_0_WMMA_VK128_SHARED_STRIDE;
    constexpr int row_sub = group_id & 3;
    constexpr int col_sub = (group_id >> 2) & 3;

    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int wave = tid >> 6u;
    const unsigned int lane = tid & 63u;
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
    hrx_q8_0_wmma_vk128_half8_vec acc[1] = {};

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
            const long long source_col = col_base + static_cast<long long>(mirror_col_sub * 16 + (c & 15));
            sh_b[c * SHARED_STRIDE + kk] = source_col < cols ?
                static_cast<_Float16>(src1[source_col * k + k0 + kk]) : zero;
        }
        __syncthreads();

        if (wave == 0) {
            hrx_q8_0_wmma_vk128_lds_half_ptr sh_a_lds =
                (hrx_q8_0_wmma_vk128_lds_half_ptr) sh_a;
            hrx_q8_0_wmma_vk128_lds_half_ptr sh_b_lds =
                (hrx_q8_0_wmma_vk128_lds_half_ptr) sh_b;
            hrx_q8_0_wmma_vk128_half16_vec a_frag[2][4];
            hrx_q8_0_wmma_vk128_half16_vec b_frag[2][4];
#pragma unroll
            for (int k_tile = 0; k_tile < 2; ++k_tile) {
#pragma unroll
                for (int rs = 0; rs < 4; ++rs) {
                    a_frag[k_tile][rs] =
                        hrx_q8_0_wmma_vk128_load_a_frag_w64_b64asm_nowait(
                            sh_a_lds, rs, k_tile, lane);
                }
#pragma unroll
                for (int cs = 0; cs < 4; ++cs) {
                    b_frag[k_tile][cs] =
                        hrx_q8_0_wmma_vk128_load_b_frag_w64_b64asm_nowait(
                            sh_b_lds, cs, k_tile, lane);
                }
            }
            asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
#pragma unroll
            for (int k_tile = 0; k_tile < 2; ++k_tile) {
                acc[0] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(
                    a_frag[k_tile][row_sub],
                    b_frag[k_tile][col_sub],
                    acc[0],
                    HRX_Q8_0_WMMA_VK128_W64_OPSEL != 0);
            }
        }
        __syncthreads();
    }

    if (wave == 0) {
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            q8_repro_raw_store_slot(dst_rsrc, rows, row_base, col_base, rows, cols, acc, 0, group_id, slot, lane);
        }
    }
}

template <int group_id, bool copy_a, bool copy_b>
__global__ __launch_bounds__(256, 1)
void q8_single_group_copy_repro_kernel(
        const hrx_block_q8_0_wmma_vk128_lhs * src0,
        const float * src1,
        float * dst,
        long long k,
        long long rows,
        long long cols) {
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 32;
    constexpr int SHARED_STRIDE = HRX_Q8_0_WMMA_VK128_SHARED_STRIDE;
    constexpr int row_sub = group_id & 3;
    constexpr int col_sub = (group_id >> 2) & 3;

    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int wave = tid >> 6u;
    const unsigned int lane = tid & 63u;
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
    hrx_q8_0_wmma_vk128_half8_vec acc[1] = {};

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
            sh_b[c * SHARED_STRIDE + kk] = col < cols ? static_cast<_Float16>(src1[col * k + k0 + kk]) : zero;
        }
        __syncthreads();

        if (wave == 0) {
            hrx_q8_0_wmma_vk128_lds_half_ptr sh_a_lds =
                (hrx_q8_0_wmma_vk128_lds_half_ptr) sh_a;
            hrx_q8_0_wmma_vk128_lds_half_ptr sh_b_lds =
                (hrx_q8_0_wmma_vk128_lds_half_ptr) sh_b;
            hrx_q8_0_wmma_vk128_half16_vec a_frag[2][4];
            hrx_q8_0_wmma_vk128_half16_vec b_frag[2][4];
#pragma unroll
            for (int k_tile = 0; k_tile < 2; ++k_tile) {
#pragma unroll
                for (int rs = 0; rs < 4; ++rs) {
                    a_frag[k_tile][rs] =
                        hrx_q8_0_wmma_vk128_load_a_frag_w64_b64asm_nowait(
                            sh_a_lds, rs, k_tile, lane);
                }
#pragma unroll
                for (int cs = 0; cs < 4; ++cs) {
                    b_frag[k_tile][cs] =
                        hrx_q8_0_wmma_vk128_load_b_frag_w64_b64asm_nowait(
                            sh_b_lds, cs, k_tile, lane);
                }
            }
            asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
#pragma unroll
            for (int k_tile = 0; k_tile < 2; ++k_tile) {
                const hrx_q8_0_wmma_vk128_half16_vec a_use = copy_a ?
                    q8_repro_copy_frag(a_frag[k_tile][row_sub]) :
                    a_frag[k_tile][row_sub];
                const hrx_q8_0_wmma_vk128_half16_vec b_use = copy_b ?
                    q8_repro_copy_frag(b_frag[k_tile][col_sub]) :
                    b_frag[k_tile][col_sub];
                acc[0] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(
                    a_use,
                    b_use,
                    acc[0],
                    HRX_Q8_0_WMMA_VK128_W64_OPSEL != 0);
            }
        }
        __syncthreads();
    }

    if (wave == 0) {
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            q8_repro_raw_store_slot(dst_rsrc, rows, row_base, col_base, rows, cols, acc, 0, group_id, slot, lane);
        }
    }
}

template <int group_id>
__global__ __launch_bounds__(256, 1)
void q8_single_group_opsel1_repro_kernel(
        const hrx_block_q8_0_wmma_vk128_lhs * src0,
        const float * src1,
        float * dst,
        long long k,
        long long rows,
        long long cols) {
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 32;
    constexpr int SHARED_STRIDE = HRX_Q8_0_WMMA_VK128_SHARED_STRIDE;
    constexpr int row_sub = group_id & 3;
    constexpr int col_sub = (group_id >> 2) & 3;

    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int wave = tid >> 6u;
    const unsigned int lane = tid & 63u;
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
    hrx_q8_0_wmma_vk128_half8_vec acc = {};

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
            sh_b[c * SHARED_STRIDE + kk] = col < cols ? static_cast<_Float16>(src1[col * k + k0 + kk]) : zero;
        }
        __syncthreads();

        if (wave == 0) {
            hrx_q8_0_wmma_vk128_lds_half_ptr sh_a_lds =
                (hrx_q8_0_wmma_vk128_lds_half_ptr) sh_a;
            hrx_q8_0_wmma_vk128_lds_half_ptr sh_b_lds =
                (hrx_q8_0_wmma_vk128_lds_half_ptr) sh_b;
            hrx_q8_0_wmma_vk128_half16_vec a_frag[2][4];
            hrx_q8_0_wmma_vk128_half16_vec b_frag[2][4];
#pragma unroll
            for (int k_tile = 0; k_tile < 2; ++k_tile) {
#pragma unroll
                for (int rs = 0; rs < 4; ++rs) {
                    a_frag[k_tile][rs] =
                        hrx_q8_0_wmma_vk128_load_a_frag_w64_b64asm_nowait(
                            sh_a_lds, rs, k_tile, lane);
                }
#pragma unroll
                for (int cs = 0; cs < 4; ++cs) {
                    b_frag[k_tile][cs] =
                        hrx_q8_0_wmma_vk128_load_b_frag_w64_b64asm_nowait(
                            sh_b_lds, cs, k_tile, lane);
                }
            }
            asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
#pragma unroll
            for (int k_tile = 0; k_tile < 2; ++k_tile) {
                acc = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(
                    a_frag[k_tile][row_sub],
                    b_frag[k_tile][col_sub],
                    acc,
                    true);
            }
        }
        __syncthreads();
    }

    if (wave == 0) {
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            q8_repro_raw_store_value(
                dst_rsrc,
                rows,
                row_base,
                col_base,
                rows,
                cols,
                group_id,
                slot,
                lane,
                static_cast<float>(acc[slot * 2 + 1]));
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

static uint16_t float_to_half_bits(float value) {
    _Float16 h = static_cast<_Float16>(value);
    uint16_t bits = 0;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
}

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

static float rhs_value(int col, int k_index) {
    const int raw = (col * 19 + k_index * 7 + 23) & 63;
    return (static_cast<float>(raw) - 31.0f) * 0.0015f;
}

static void fill_q8(std::vector<hrx_block_q8_0_wmma_vk128_lhs> & blocks, int rows, int blocks_per_row) {
    for (int row = 0; row < rows; ++row) {
        for (int block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
            hrx_block_q8_0_wmma_vk128_lhs & block =
                blocks[static_cast<size_t>(row) * blocks_per_row + block_idx];
            block.d = float_to_half_bits(0.00390625f * static_cast<float>(1 + ((row + block_idx) & 3)));
            for (int i = 0; i < 32; ++i) {
                block.qs[i] = static_cast<int8_t>(((row * 11 + block_idx * 7 + i * 5) & 31) - 16);
            }
        }
    }
}

static void fill_rhs(std::vector<float> & rhs, int k, int cols) {
    for (int col = 0; col < cols; ++col) {
        for (int kk = 0; kk < k; ++kk) {
            rhs[static_cast<size_t>(col) * k + kk] = rhs_value(col, kk);
        }
    }
}

static float q8_dequant(
        const std::vector<hrx_block_q8_0_wmma_vk128_lhs> & blocks,
        int row,
        int k_index,
        int blocks_per_row) {
    const hrx_block_q8_0_wmma_vk128_lhs & block =
        blocks[static_cast<size_t>(row) * blocks_per_row + (k_index >> 5)];
    return half_bits_to_float(block.d) * static_cast<float>(block.qs[k_index & 31]);
}

static std::vector<float> cpu_reference(
        const std::vector<hrx_block_q8_0_wmma_vk128_lhs> & blocks,
        const std::vector<float> & rhs,
        int k,
        int rows,
        int cols) {
    const int blocks_per_row = k / 32;
    std::vector<float> ref(static_cast<size_t>(rows) * cols, 0.0f);
    for (int col = 0; col < cols; ++col) {
        for (int row = 0; row < rows; ++row) {
            float sum = 0.0f;
            for (int kk = 0; kk < k; ++kk) {
                sum += q8_dequant(blocks, row, kk, blocks_per_row) * rhs[static_cast<size_t>(col) * k + kk];
            }
            ref[static_cast<size_t>(col) * rows + row] = sum;
        }
    }
    return ref;
}

static int output_group(size_t index, int rows) {
    const int row = static_cast<int>(index % static_cast<size_t>(rows));
    const int col = static_cast<int>(index / static_cast<size_t>(rows));
    const int row_local = row & 63;
    const int col_local = col & 63;
    return (col_local >> 4) * 4 + (row_local >> 4);
}

static bool remap_mode_groups(const std::string & mode, int * compute_group, int * store_group) {
    if (mode == "remap-c8-s0") {
        *compute_group = 8;
        *store_group = 0;
        return true;
    }
    if (mode == "remap-c0-s8") {
        *compute_group = 0;
        *store_group = 8;
        return true;
    }
    if (mode == "remap-c12-s0") {
        *compute_group = 12;
        *store_group = 0;
        return true;
    }
    if (mode == "remap-c12-s0-bcopy-stage-selected" ||
            mode == "remap-c12-s0-abcopy-stage-selected") {
        *compute_group = 12;
        *store_group = 0;
        return true;
    }
    if (mode == "remap-c0-s12") {
        *compute_group = 0;
        *store_group = 12;
        return true;
    }
    if (mode == "remap-c0-s12-stage-selected") {
        *compute_group = 0;
        *store_group = 12;
        return true;
    }
    return false;
}

static bool bmirror_mode_groups(const std::string & mode, int * compute_group, int * store_group) {
    if (mode == "single-group8-bmirror0") {
        *compute_group = 0;
        *store_group = 8;
        return true;
    }
    if (mode == "single-group12-bmirror0") {
        *compute_group = 0;
        *store_group = 12;
        return true;
    }
    return false;
}

static bool two_phase_copy_mode(const std::string & mode) {
    return mode == "array8-fullb-2phase-bcopy" ||
        mode == "array8-fullb-2phase-bcopy-stage" ||
        mode == "array8-fullb-2phase-abcopy";
}

static bool copy_mode_group(const std::string & mode, int * group_id) {
    if (mode == "single-group0-bcopy-stage") {
        *group_id = 0;
        return true;
    }
    if (mode == "single-group8-bcopy" || mode == "single-group8-abcopy" ||
            mode == "single-group8-bcopy-stage" || mode == "single-group8-abcopy-stage" ||
            mode == "single-group8-bcopy-stage-selected") {
        *group_id = 8;
        return true;
    }
    if (mode == "single-group12-bcopy" || mode == "single-group12-abcopy" ||
            mode == "single-group12-bcopy-stage" || mode == "single-group12-abcopy-stage" ||
            mode == "single-group12-bcopy-stage-selected" ||
            mode == "single-group12-abcopy-stage-selected" ||
            mode == "single-group12-bcopy-stage-selected-acccopy" ||
            mode == "single-group12-abcopy-stage-selected-acccopy" ||
            mode == "single-group12-bcopy-stage-selected-regcopy" ||
            mode == "single-group12-abcopy-stage-selected-regcopy") {
        *group_id = 12;
        return true;
    }
    return false;
}

static bool output_is_active(size_t index, int rows, const std::string & mode) {
    const int group = output_group(index, rows);
    int compute_group = 0;
    int store_group = 0;
    if (remap_mode_groups(mode, &compute_group, &store_group)) {
        (void) compute_group;
        return group == store_group;
    }
    if (bmirror_mode_groups(mode, &compute_group, &store_group)) {
        (void) compute_group;
        return group == store_group;
    }
    int copy_group = 0;
    if (copy_mode_group(mode, &copy_group)) {
        return group == copy_group;
    }
    if (mode == "single-group0" || mode == "single-group0-consume") {
        return group == 0;
    }
    if (mode == "single-group0-opsel1") {
        return group == 0;
    }
    if (mode == "single-group8" || mode == "single-group8-consume") {
        return group == 8;
    }
    if (mode == "single-group8-opsel1") {
        return group == 8;
    }
    if (mode == "single-group12" || mode == "single-group12-consume") {
        return group == 12;
    }
    if (mode == "single-group12-opsel1") {
        return group == 12;
    }
    if (mode == "single-group13" || mode == "single-group13-consume") {
        return group == 13;
    }
    if (mode == "array16-direct-raw" ||
            mode == "array16-direct-raw-bcopy" ||
            mode == "array16-direct-raw-abcopy" ||
            mode == "array8-fullb-2phase" || mode == "batched4" ||
            mode == "array8-fullb-2phase-consume" || mode == "batched4-consume") {
        return true;
    }
    if (two_phase_copy_mode(mode)) {
        return true;
    }
    return group < 8;
}

static float expected_value_for_output(
        size_t index,
        int rows,
        int cols,
        const std::string & mode,
        const std::vector<float> & ref) {
    int compute_group = 0;
    int store_group = 0;
    if (!remap_mode_groups(mode, &compute_group, &store_group) &&
            !bmirror_mode_groups(mode, &compute_group, &store_group)) {
        return ref[index];
    }

    const int row = static_cast<int>(index % static_cast<size_t>(rows));
    const int col = static_cast<int>(index / static_cast<size_t>(rows));
    const int row_tile_base = row & ~63;
    const int col_tile_base = col & ~63;
    const int store_row_sub = store_group & 3;
    const int store_col_sub = (store_group >> 2) & 3;
    const int compute_row_sub = compute_group & 3;
    const int compute_col_sub = (compute_group >> 2) & 3;
    const int row_inner = (row & 63) - store_row_sub * 16;
    const int col_inner = (col & 63) - store_col_sub * 16;
    const int source_row = row_tile_base + compute_row_sub * 16 + row_inner;
    const int source_col = col_tile_base + compute_col_sub * 16 + col_inner;
    if (source_row < 0 || source_row >= rows || source_col < 0 || source_col >= cols) {
        return 0.0f;
    }
    return ref[static_cast<size_t>(source_col) * rows + static_cast<size_t>(source_row)];
}

struct group_stats {
    size_t active = 0;
    size_t bad = 0;
    size_t nan = 0;
    size_t inf = 0;
    size_t sentinel = 0;
    float max_abs = 0.0f;
    bool have_first_bad = false;
    int first_row = -1;
    int first_col = -1;
    int first_lane = -1;
    int first_slot = -1;
    float first_actual = 0.0f;
    float first_expected = 0.0f;
    float first_err = 0.0f;
    int sample_count = 0;
    int sample_row[16] = {};
    int sample_col[16] = {};
    int sample_lane[16] = {};
    int sample_slot[16] = {};
    float sample_actual[16] = {};
    float sample_expected[16] = {};
    float sample_err[16] = {};
};

static void note_first_bad(
        group_stats & gs,
        size_t index,
        int rows,
        float actual,
        float expected,
        float err) {
    if (gs.have_first_bad) {
        return;
    }
    const int row = static_cast<int>(index % static_cast<size_t>(rows));
    const int col = static_cast<int>(index / static_cast<size_t>(rows));
    const int row_lane = row & 3;
    const int slot = (row >> 2) & 3;
    const int col_lane = col & 15;
    gs.have_first_bad = true;
    gs.first_row = row;
    gs.first_col = col;
    gs.first_lane = row_lane * 16 + col_lane;
    gs.first_slot = slot;
    gs.first_actual = actual;
    gs.first_expected = expected;
    gs.first_err = err;
}

static void note_bad_sample(
        group_stats & gs,
        size_t index,
        int rows,
        float actual,
        float expected,
        float err) {
    note_first_bad(gs, index, rows, actual, expected, err);
    if (gs.sample_count >= 16) {
        return;
    }
    const int row = static_cast<int>(index % static_cast<size_t>(rows));
    const int col = static_cast<int>(index / static_cast<size_t>(rows));
    const int row_lane = row & 3;
    const int slot = (row >> 2) & 3;
    const int col_lane = col & 15;
    const int sample = gs.sample_count++;
    gs.sample_row[sample] = row;
    gs.sample_col[sample] = col;
    gs.sample_lane[sample] = row_lane * 16 + col_lane;
    gs.sample_slot[sample] = slot;
    gs.sample_actual[sample] = actual;
    gs.sample_expected[sample] = expected;
    gs.sample_err[sample] = err;
}

static int run_bfrag_dump_case(int cols, int k) {
    constexpr int dump_count = 2 * 4 * 64 * 16;
    std::vector<float> h_rhs(static_cast<size_t>(cols) * k);
    std::vector<float> h_dump(dump_count, -7777.0f);
    fill_rhs(h_rhs, k, cols);

    device_buffer<float> d_rhs(h_rhs.size());
    device_buffer<float> d_dump(h_dump.size());
    HIP_CHECK(hipMemcpy(d_rhs.ptr, h_rhs.data(), h_rhs.size() * sizeof(h_rhs[0]), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_dump.ptr, h_dump.data(), h_dump.size() * sizeof(h_dump[0]), hipMemcpyHostToDevice));

    hipLaunchKernelGGL(q8_bfrag_dump_kernel, dim3(1, 1, 1), dim3(256, 1, 1), 0, 0,
        d_rhs.ptr, d_dump.ptr, k, cols);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_dump.data(), d_dump.ptr, h_dump.size() * sizeof(h_dump[0]), hipMemcpyDeviceToHost));

    size_t active = 0;
    size_t bad = 0;
    size_t sentinel = 0;
    size_t nan = 0;
    float max_abs = 0.0f;
    int first_bad_k_tile = -1;
    int first_bad_col_sub = -1;
    int first_bad_lane = -1;
    int first_bad_elem = -1;
    float first_bad_actual = 0.0f;
    float first_bad_expected = 0.0f;

    for (int k_tile = 0; k_tile < 2; ++k_tile) {
        for (int col_sub = 0; col_sub < 4; ++col_sub) {
            for (int lane = 0; lane < 64; ++lane) {
                const int col = col_sub * 16 + (lane & 15);
                for (int elem = 0; elem < 16; ++elem) {
                    const int index = (((k_tile * 4 + col_sub) * 64 + lane) * 16) + elem;
                    const float actual = h_dump[index];
                    const int k_index = k_tile * 16 + elem;
                    const float expected = col < cols ?
                        half_bits_to_float(float_to_half_bits(rhs_value(col, k_index))) : 0.0f;
                    ++active;
                    if (actual == -7777.0f) {
                        ++sentinel;
                    }
                    if (std::isnan(actual)) {
                        ++nan;
                        ++bad;
                        if (first_bad_k_tile < 0) {
                            first_bad_k_tile = k_tile;
                            first_bad_col_sub = col_sub;
                            first_bad_lane = lane;
                            first_bad_elem = elem;
                            first_bad_actual = actual;
                            first_bad_expected = expected;
                        }
                        continue;
                    }
                    const float err = std::fabs(actual - expected);
                    max_abs = std::max(max_abs, err);
                    if (err > 0.0f || actual == -7777.0f) {
                        ++bad;
                        if (first_bad_k_tile < 0) {
                            first_bad_k_tile = k_tile;
                            first_bad_col_sub = col_sub;
                            first_bad_lane = lane;
                            first_bad_elem = elem;
                            first_bad_actual = actual;
                            first_bad_expected = expected;
                        }
                    }
                }
            }
        }
    }

    std::printf(
        "bfrag-dump cols=%d k=%d active=%zu bad=%zu nan=%zu sentinel=%zu max_abs=%g\n",
        cols, k, active, bad, nan, sentinel, max_abs);
    if (first_bad_k_tile >= 0) {
        std::printf(
            "  first_bad k_tile=%d col_sub=%d lane=%d elem=%d actual=%g expected=%g\n",
            first_bad_k_tile, first_bad_col_sub, first_bad_lane, first_bad_elem,
            first_bad_actual, first_bad_expected);
    }
    return (bad == 0 && nan == 0 && sentinel == 0) ? 0 : 1;
}

static int run_dual_stage_compare_case(const std::string & mode, int rows, int cols, int k) {
    const int blocks_per_row = k / 32;
    std::vector<hrx_block_q8_0_wmma_vk128_lhs> h_q8(static_cast<size_t>(rows) * blocks_per_row);
    std::vector<float> h_rhs(static_cast<size_t>(cols) * k);
    std::vector<float> h_raw(static_cast<size_t>(rows) * cols, -7777.0f);
    std::vector<float> h_staged(static_cast<size_t>(rows) * cols, -7777.0f);
    fill_q8(h_q8, rows, blocks_per_row);
    fill_rhs(h_rhs, k, cols);
    const std::vector<float> ref = cpu_reference(h_q8, h_rhs, k, rows, cols);

    device_buffer<hrx_block_q8_0_wmma_vk128_lhs> d_q8(h_q8.size());
    device_buffer<float> d_rhs(h_rhs.size());
    device_buffer<float> d_raw(h_raw.size());
    device_buffer<float> d_staged(h_staged.size());
    HIP_CHECK(hipMemcpy(d_q8.ptr, h_q8.data(), h_q8.size() * sizeof(h_q8[0]), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_rhs.ptr, h_rhs.data(), h_rhs.size() * sizeof(h_rhs[0]), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_raw.ptr, h_raw.data(), h_raw.size() * sizeof(h_raw[0]), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_staged.ptr, h_staged.data(), h_staged.size() * sizeof(h_staged[0]), hipMemcpyHostToDevice));

    dim3 grid((rows + 63) / 64, (cols + 63) / 64, 1);
    if (mode == "single-group12-abcopy-dual-stage-raw-first") {
        hipLaunchKernelGGL((q8_single_group_dual_stage_repro_kernel<12, true, true, true>), grid, dim3(256, 1, 1), 0, 0,
            d_q8.ptr, d_rhs.ptr, d_raw.ptr, d_staged.ptr, k, rows, cols);
    } else if (mode == "single-group12-abcopy-dual-stage-stage-first") {
        hipLaunchKernelGGL((q8_single_group_dual_stage_repro_kernel<12, true, true, false>), grid, dim3(256, 1, 1), 0, 0,
            d_q8.ptr, d_rhs.ptr, d_raw.ptr, d_staged.ptr, k, rows, cols);
    } else {
        std::fprintf(stderr, "unknown dual-stage mode: %s\n", mode.c_str());
        return 2;
    }
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_raw.data(), d_raw.ptr, h_raw.size() * sizeof(h_raw[0]), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_staged.data(), d_staged.ptr, h_staged.size() * sizeof(h_staged[0]), hipMemcpyDeviceToHost));

    size_t active = 0;
    size_t raw_bad = 0;
    size_t staged_bad = 0;
    size_t mismatch = 0;
    size_t raw_nan = 0;
    size_t staged_nan = 0;
    size_t raw_inf = 0;
    size_t staged_inf = 0;
    size_t raw_sentinel = 0;
    size_t staged_sentinel = 0;
    float raw_max_abs = 0.0f;
    float staged_max_abs = 0.0f;
    float mismatch_max_abs = 0.0f;
    bool have_first_mismatch = false;
    int first_row = -1;
    int first_col = -1;
    int first_lane = -1;
    int first_slot = -1;
    float first_raw = 0.0f;
    float first_staged = 0.0f;
    float first_expected = 0.0f;
    float first_delta = 0.0f;

    for (size_t i = 0; i < h_raw.size(); ++i) {
        if (output_group(i, rows) != 12) {
            continue;
        }
        ++active;
        const float raw = h_raw[i];
        const float staged = h_staged[i];
        const float expected = ref[i];

        if (raw == -7777.0f) {
            ++raw_sentinel;
            ++raw_bad;
        } else if (std::isnan(raw)) {
            ++raw_nan;
            ++raw_bad;
        } else if (std::isinf(raw)) {
            ++raw_inf;
            ++raw_bad;
        } else {
            const float raw_err = std::fabs(raw - expected);
            raw_max_abs = std::max(raw_max_abs, raw_err);
            if (raw_err > 0.25f) {
                ++raw_bad;
            }
        }

        if (staged == -7777.0f) {
            ++staged_sentinel;
            ++staged_bad;
        } else if (std::isnan(staged)) {
            ++staged_nan;
            ++staged_bad;
        } else if (std::isinf(staged)) {
            ++staged_inf;
            ++staged_bad;
        } else {
            const float staged_err = std::fabs(staged - expected);
            staged_max_abs = std::max(staged_max_abs, staged_err);
            if (staged_err > 0.25f) {
                ++staged_bad;
            }
        }

        if (raw != -7777.0f && staged != -7777.0f &&
                !std::isnan(raw) && !std::isnan(staged) &&
                !std::isinf(raw) && !std::isinf(staged)) {
            const float delta = std::fabs(raw - staged);
            mismatch_max_abs = std::max(mismatch_max_abs, delta);
            if (delta > 0.01f) {
                ++mismatch;
                if (!have_first_mismatch) {
                    const int row = static_cast<int>(i % static_cast<size_t>(rows));
                    const int col = static_cast<int>(i / static_cast<size_t>(rows));
                    const int row_lane = row & 3;
                    const int col_lane = col & 15;
                    have_first_mismatch = true;
                    first_row = row;
                    first_col = col;
                    first_lane = row_lane * 16 + col_lane;
                    first_slot = (row >> 2) & 3;
                    first_raw = raw;
                    first_staged = staged;
                    first_expected = expected;
                    first_delta = raw - staged;
                }
            }
        }
    }

    std::printf(
        "%s rows=%d cols=%d k=%d active=%zu raw_bad=%zu staged_bad=%zu mismatch=%zu raw_nan=%zu staged_nan=%zu raw_inf=%zu staged_inf=%zu raw_sentinel=%zu staged_sentinel=%zu raw_max_abs=%g staged_max_abs=%g mismatch_max_abs=%g\n",
        mode.c_str(),
        rows,
        cols,
        k,
        active,
        raw_bad,
        staged_bad,
        mismatch,
        raw_nan,
        staged_nan,
        raw_inf,
        staged_inf,
        raw_sentinel,
        staged_sentinel,
        raw_max_abs,
        staged_max_abs,
        mismatch_max_abs);
    if (have_first_mismatch) {
        std::printf(
            "  first_mismatch row=%d col=%d lane=%d slot=%d raw=%g staged=%g expected=%g raw_minus_staged=%g\n",
            first_row,
            first_col,
            first_lane,
            first_slot,
            first_raw,
            first_staged,
            first_expected,
            first_delta);
    }
    return (raw_nan == 0 && staged_nan == 0 && raw_inf == 0 && staged_inf == 0 &&
            raw_sentinel == 0 && staged_sentinel == 0) ? 0 : 1;
}

static int run_contract_case(const std::string & mode, int rows, int cols, int k) {
    const int blocks_per_row = k / 32;
    std::vector<hrx_block_q8_0_wmma_vk128_lhs> h_q8(static_cast<size_t>(rows) * blocks_per_row);
    std::vector<float> h_rhs(static_cast<size_t>(cols) * k);
    std::vector<float> h_contract(Q8_REPRO_CONTRACT_VALUES, -7777.0f);
    fill_q8(h_q8, rows, blocks_per_row);
    fill_rhs(h_rhs, k, cols);
    const std::vector<float> ref = cpu_reference(h_q8, h_rhs, k, rows, cols);

    device_buffer<hrx_block_q8_0_wmma_vk128_lhs> d_q8(h_q8.size());
    device_buffer<float> d_rhs(h_rhs.size());
    device_buffer<float> d_contract(h_contract.size());
    HIP_CHECK(hipMemcpy(d_q8.ptr, h_q8.data(), h_q8.size() * sizeof(h_q8[0]), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_rhs.ptr, h_rhs.data(), h_rhs.size() * sizeof(h_rhs[0]), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_contract.ptr, h_contract.data(), h_contract.size() * sizeof(h_contract[0]), hipMemcpyHostToDevice));

    dim3 grid((rows + 63) / 64, (cols + 63) / 64, 1);
    if (mode == "contract-direct192-abcopy") {
        hipLaunchKernelGGL((q8_contract_direct192_repro_kernel<true, true>), grid, dim3(256, 1, 1), 0, 0,
            d_q8.ptr, d_rhs.ptr, d_contract.ptr, k, rows, cols);
    } else if (mode == "contract-phase96-abcopy") {
        hipLaunchKernelGGL((q8_contract_phase96_repro_kernel<0, 0, 2, 8, 16, 16, true, true>),
            grid, dim3(256, 1, 1), 0, 0, d_q8.ptr, d_rhs.ptr, d_contract.ptr, k, rows, cols);
        hipLaunchKernelGGL((q8_contract_phase96_repro_kernel<8, 2, 2, 8, 32, 16, true, true>),
            grid, dim3(256, 1, 1), 0, 0, d_q8.ptr, d_rhs.ptr, d_contract.ptr, k, rows, cols);
    } else {
        std::fprintf(stderr, "unknown contract mode: %s\n", mode.c_str());
        return 2;
    }
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_contract.data(), d_contract.ptr, h_contract.size() * sizeof(h_contract[0]), hipMemcpyDeviceToHost));

    size_t active = 0;
    size_t inactive = 0;
    size_t bad = 0;
    size_t nan = 0;
    size_t inf = 0;
    size_t sentinel = 0;
    size_t unexpected = 0;
    float max_abs = 0.0f;
    bool have_first_bad = false;
    int first_group = -1;
    int first_slot = -1;
    int first_lane = -1;
    float first_actual = 0.0f;
    float first_expected = 0.0f;
    float first_err = 0.0f;

    for (int group = 0; group < Q8_REPRO_CONTRACT_GROUPS; ++group) {
        for (int slot = 0; slot < Q8_REPRO_CONTRACT_SLOTS; ++slot) {
            for (int lane = 0; lane < Q8_REPRO_CONTRACT_LANES; ++lane) {
                const int index = q8_repro_contract_index(group, slot, static_cast<unsigned int>(lane));
                const float actual = h_contract[static_cast<size_t>(index)];
                bool should_be_active = true;
                float expected = 0.0f;
                if (group < 16) {
                    const int row = (group & 3) * 16 + (lane >> 4) + slot * 4;
                    const int col = ((group >> 2) & 3) * 16 + (lane & 15);
                    should_be_active = row < rows && col < cols;
                    if (should_be_active) {
                        expected = ref[static_cast<size_t>(col) * rows + static_cast<size_t>(row)];
                    }
                } else {
                    expected = q8_repro_contract_synthetic_value(group, slot, static_cast<unsigned int>(lane));
                }

                if (!should_be_active) {
                    ++inactive;
                    if (actual != -7777.0f) {
                        ++unexpected;
                        ++bad;
                        if (!have_first_bad) {
                            have_first_bad = true;
                            first_group = group;
                            first_slot = slot;
                            first_lane = lane;
                            first_actual = actual;
                            first_expected = -7777.0f;
                            first_err = INFINITY;
                        }
                    }
                    continue;
                }

                ++active;
                if (actual == -7777.0f) {
                    ++sentinel;
                    ++bad;
                    if (!have_first_bad) {
                        have_first_bad = true;
                        first_group = group;
                        first_slot = slot;
                        first_lane = lane;
                        first_actual = actual;
                        first_expected = expected;
                        first_err = INFINITY;
                    }
                    continue;
                }
                if (std::isnan(actual)) {
                    ++nan;
                    ++bad;
                    if (!have_first_bad) {
                        have_first_bad = true;
                        first_group = group;
                        first_slot = slot;
                        first_lane = lane;
                        first_actual = actual;
                        first_expected = expected;
                        first_err = NAN;
                    }
                    continue;
                }
                if (std::isinf(actual)) {
                    ++inf;
                    ++bad;
                    if (!have_first_bad) {
                        have_first_bad = true;
                        first_group = group;
                        first_slot = slot;
                        first_lane = lane;
                        first_actual = actual;
                        first_expected = expected;
                        first_err = INFINITY;
                    }
                    continue;
                }
                const float err = std::fabs(actual - expected);
                max_abs = std::max(max_abs, err);
                const float threshold = group < 16 ? 0.25f : 0.0f;
                if (err > threshold) {
                    ++bad;
                    if (!have_first_bad) {
                        have_first_bad = true;
                        first_group = group;
                        first_slot = slot;
                        first_lane = lane;
                        first_actual = actual;
                        first_expected = expected;
                        first_err = err;
                    }
                }
            }
        }
    }

    std::printf(
        "%s rows=%d cols=%d k=%d active=%zu inactive=%zu bad=%zu nan=%zu inf=%zu sentinel=%zu unexpected=%zu max_abs=%g\n",
        mode.c_str(),
        rows,
        cols,
        k,
        active,
        inactive,
        bad,
        nan,
        inf,
        sentinel,
        unexpected,
        max_abs);
    if (have_first_bad) {
        std::printf(
            "  first_bad group=%d slot=%d lane=%d actual=%g expected=%g err=%g\n",
            first_group,
            first_slot,
            first_lane,
            first_actual,
            first_expected,
            first_err);
    }
    return bad == 0 ? 0 : 1;
}

static int run_case(const std::string & mode, int rows, int cols, int k) {
    const int blocks_per_row = k / 32;
    std::vector<hrx_block_q8_0_wmma_vk128_lhs> h_q8(static_cast<size_t>(rows) * blocks_per_row);
    std::vector<float> h_rhs(static_cast<size_t>(cols) * k);
    std::vector<float> h_out(static_cast<size_t>(rows) * cols, -7777.0f);
    fill_q8(h_q8, rows, blocks_per_row);
    fill_rhs(h_rhs, k, cols);
    const std::vector<float> ref = cpu_reference(h_q8, h_rhs, k, rows, cols);

    device_buffer<hrx_block_q8_0_wmma_vk128_lhs> d_q8(h_q8.size());
    device_buffer<float> d_rhs(h_rhs.size());
    device_buffer<float> d_out(h_out.size());
    HIP_CHECK(hipMemcpy(d_q8.ptr, h_q8.data(), h_q8.size() * sizeof(h_q8[0]), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_rhs.ptr, h_rhs.data(), h_rhs.size() * sizeof(h_rhs[0]), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_out.ptr, h_out.data(), h_out.size() * sizeof(h_out[0]), hipMemcpyHostToDevice));

    dim3 grid((rows + 63) / 64, (cols + 63) / 64, 1);
    if (mode == "array8-fullb") {
        hipLaunchKernelGGL((q8_array8_repro_kernel<true>), grid, dim3(256, 1, 1), 0, 0,
            d_q8.ptr, d_rhs.ptr, d_out.ptr, k, rows, cols);
    } else if (mode == "array16-direct-raw") {
        hipLaunchKernelGGL((q8_array16_direct_raw_repro_kernel<false, false>), grid, dim3(256, 1, 1), 0, 0,
            d_q8.ptr, d_rhs.ptr, d_out.ptr, k, rows, cols);
    } else if (mode == "array16-direct-raw-bcopy") {
        hipLaunchKernelGGL((q8_array16_direct_raw_repro_kernel<false, true>), grid, dim3(256, 1, 1), 0, 0,
            d_q8.ptr, d_rhs.ptr, d_out.ptr, k, rows, cols);
    } else if (mode == "array16-direct-raw-abcopy") {
        hipLaunchKernelGGL((q8_array16_direct_raw_repro_kernel<true, true>), grid, dim3(256, 1, 1), 0, 0,
            d_q8.ptr, d_rhs.ptr, d_out.ptr, k, rows, cols);
    } else if (mode == "array8-b2") {
        hipLaunchKernelGGL((q8_array8_repro_kernel<false>), grid, dim3(256, 1, 1), 0, 0,
            d_q8.ptr, d_rhs.ptr, d_out.ptr, k, rows, cols);
    } else if (mode == "array8-fullb-2phase") {
        hipLaunchKernelGGL((q8_array_fullb_phase_repro_kernel<0, 0, 2, 8, false>), grid, dim3(256, 1, 1), 0, 0,
            d_q8.ptr, d_rhs.ptr, d_out.ptr, k, rows, cols);
        hipLaunchKernelGGL((q8_array_fullb_phase_repro_kernel<8, 2, 2, 8, false>), grid, dim3(256, 1, 1), 0, 0,
            d_q8.ptr, d_rhs.ptr, d_out.ptr, k, rows, cols);
    } else if (mode == "array8-fullb-2phase-consume") {
        hipLaunchKernelGGL((q8_array_fullb_phase_repro_kernel<0, 0, 2, 8, true>), grid, dim3(256, 1, 1), 0, 0,
            d_q8.ptr, d_rhs.ptr, d_out.ptr, k, rows, cols);
        hipLaunchKernelGGL((q8_array_fullb_phase_repro_kernel<8, 2, 2, 8, true>), grid, dim3(256, 1, 1), 0, 0,
            d_q8.ptr, d_rhs.ptr, d_out.ptr, k, rows, cols);
    } else if (mode == "array8-fullb-2phase-bcopy") {
        hipLaunchKernelGGL((q8_array_fullb_phase_repro_kernel<0, 0, 2, 8, false, false, true>), grid, dim3(256, 1, 1), 0, 0,
            d_q8.ptr, d_rhs.ptr, d_out.ptr, k, rows, cols);
        hipLaunchKernelGGL((q8_array_fullb_phase_repro_kernel<8, 2, 2, 8, false, false, true>), grid, dim3(256, 1, 1), 0, 0,
            d_q8.ptr, d_rhs.ptr, d_out.ptr, k, rows, cols);
    } else if (mode == "array8-fullb-2phase-bcopy-stage") {
        hipLaunchKernelGGL((q8_array_fullb_phase_repro_kernel<0, 0, 2, 8, false, false, true, true>), grid, dim3(256, 1, 1), 0, 0,
            d_q8.ptr, d_rhs.ptr, d_out.ptr, k, rows, cols);
        hipLaunchKernelGGL((q8_array_fullb_phase_repro_kernel<8, 2, 2, 8, false, false, true, true>), grid, dim3(256, 1, 1), 0, 0,
            d_q8.ptr, d_rhs.ptr, d_out.ptr, k, rows, cols);
    } else if (mode == "array8-fullb-2phase-abcopy") {
        hipLaunchKernelGGL((q8_array_fullb_phase_repro_kernel<0, 0, 2, 8, false, true, true>), grid, dim3(256, 1, 1), 0, 0,
            d_q8.ptr, d_rhs.ptr, d_out.ptr, k, rows, cols);
        hipLaunchKernelGGL((q8_array_fullb_phase_repro_kernel<8, 2, 2, 8, false, true, true>), grid, dim3(256, 1, 1), 0, 0,
            d_q8.ptr, d_rhs.ptr, d_out.ptr, k, rows, cols);
    } else if (mode == "batched4") {
        hipLaunchKernelGGL((q8_array_fullb_phase_repro_kernel<0, 0, 1, 4, false>), grid, dim3(256, 1, 1), 0, 0,
            d_q8.ptr, d_rhs.ptr, d_out.ptr, k, rows, cols);
        hipLaunchKernelGGL((q8_array_fullb_phase_repro_kernel<4, 1, 1, 4, false>), grid, dim3(256, 1, 1), 0, 0,
            d_q8.ptr, d_rhs.ptr, d_out.ptr, k, rows, cols);
        hipLaunchKernelGGL((q8_array_fullb_phase_repro_kernel<8, 2, 1, 4, false>), grid, dim3(256, 1, 1), 0, 0,
            d_q8.ptr, d_rhs.ptr, d_out.ptr, k, rows, cols);
        hipLaunchKernelGGL((q8_array_fullb_phase_repro_kernel<12, 3, 1, 4, false>), grid, dim3(256, 1, 1), 0, 0,
            d_q8.ptr, d_rhs.ptr, d_out.ptr, k, rows, cols);
    } else if (mode == "batched4-consume") {
        hipLaunchKernelGGL((q8_array_fullb_phase_repro_kernel<0, 0, 1, 4, true>), grid, dim3(256, 1, 1), 0, 0,
            d_q8.ptr, d_rhs.ptr, d_out.ptr, k, rows, cols);
        hipLaunchKernelGGL((q8_array_fullb_phase_repro_kernel<4, 1, 1, 4, true>), grid, dim3(256, 1, 1), 0, 0,
            d_q8.ptr, d_rhs.ptr, d_out.ptr, k, rows, cols);
        hipLaunchKernelGGL((q8_array_fullb_phase_repro_kernel<8, 2, 1, 4, true>), grid, dim3(256, 1, 1), 0, 0,
            d_q8.ptr, d_rhs.ptr, d_out.ptr, k, rows, cols);
        hipLaunchKernelGGL((q8_array_fullb_phase_repro_kernel<12, 3, 1, 4, true>), grid, dim3(256, 1, 1), 0, 0,
            d_q8.ptr, d_rhs.ptr, d_out.ptr, k, rows, cols);
    } else if (mode == "single-group0") {
        hipLaunchKernelGGL((q8_single_group_repro_kernel<0, false>), grid, dim3(256, 1, 1), 0, 0,
            d_q8.ptr, d_rhs.ptr, d_out.ptr, k, rows, cols);
    } else if (mode == "single-group0-consume") {
        hipLaunchKernelGGL((q8_single_group_repro_kernel<0, true>), grid, dim3(256, 1, 1), 0, 0,
            d_q8.ptr, d_rhs.ptr, d_out.ptr, k, rows, cols);
    } else if (mode == "single-group8") {
        hipLaunchKernelGGL((q8_single_group_repro_kernel<8, false>), grid, dim3(256, 1, 1), 0, 0,
            d_q8.ptr, d_rhs.ptr, d_out.ptr, k, rows, cols);
    } else if (mode == "single-group8-consume") {
        hipLaunchKernelGGL((q8_single_group_repro_kernel<8, true>), grid, dim3(256, 1, 1), 0, 0,
            d_q8.ptr, d_rhs.ptr, d_out.ptr, k, rows, cols);
    } else if (mode == "single-group12") {
        hipLaunchKernelGGL((q8_single_group_repro_kernel<12, false>), grid, dim3(256, 1, 1), 0, 0,
            d_q8.ptr, d_rhs.ptr, d_out.ptr, k, rows, cols);
    } else if (mode == "single-group12-consume") {
        hipLaunchKernelGGL((q8_single_group_repro_kernel<12, true>), grid, dim3(256, 1, 1), 0, 0,
            d_q8.ptr, d_rhs.ptr, d_out.ptr, k, rows, cols);
    } else if (mode == "single-group13") {
        hipLaunchKernelGGL((q8_single_group_repro_kernel<13, false>), grid, dim3(256, 1, 1), 0, 0,
            d_q8.ptr, d_rhs.ptr, d_out.ptr, k, rows, cols);
    } else if (mode == "single-group13-consume") {
        hipLaunchKernelGGL((q8_single_group_repro_kernel<13, true>), grid, dim3(256, 1, 1), 0, 0,
            d_q8.ptr, d_rhs.ptr, d_out.ptr, k, rows, cols);
    } else if (mode == "single-group0-opsel1") {
        hipLaunchKernelGGL((q8_single_group_opsel1_repro_kernel<0>), grid, dim3(256, 1, 1), 0, 0,
            d_q8.ptr, d_rhs.ptr, d_out.ptr, k, rows, cols);
    } else if (mode == "single-group8-opsel1") {
        hipLaunchKernelGGL((q8_single_group_opsel1_repro_kernel<8>), grid, dim3(256, 1, 1), 0, 0,
            d_q8.ptr, d_rhs.ptr, d_out.ptr, k, rows, cols);
    } else if (mode == "single-group12-opsel1") {
        hipLaunchKernelGGL((q8_single_group_opsel1_repro_kernel<12>), grid, dim3(256, 1, 1), 0, 0,
            d_q8.ptr, d_rhs.ptr, d_out.ptr, k, rows, cols);
    } else if (mode == "remap-c8-s0") {
        hipLaunchKernelGGL((q8_single_group_remap_repro_kernel<8, 0>), grid, dim3(256, 1, 1), 0, 0,
            d_q8.ptr, d_rhs.ptr, d_out.ptr, k, rows, cols);
    } else if (mode == "remap-c0-s8") {
        hipLaunchKernelGGL((q8_single_group_remap_repro_kernel<0, 8>), grid, dim3(256, 1, 1), 0, 0,
            d_q8.ptr, d_rhs.ptr, d_out.ptr, k, rows, cols);
    } else if (mode == "remap-c12-s0") {
        hipLaunchKernelGGL((q8_single_group_remap_repro_kernel<12, 0>), grid, dim3(256, 1, 1), 0, 0,
            d_q8.ptr, d_rhs.ptr, d_out.ptr, k, rows, cols);
    } else if (mode == "remap-c12-s0-bcopy-stage-selected") {
        hipLaunchKernelGGL((q8_single_group_remap_repro_kernel<12, 0, false, true, true, true>), grid, dim3(256, 1, 1), 0, 0,
            d_q8.ptr, d_rhs.ptr, d_out.ptr, k, rows, cols);
    } else if (mode == "remap-c12-s0-abcopy-stage-selected") {
        hipLaunchKernelGGL((q8_single_group_remap_repro_kernel<12, 0, true, true, true, true>), grid, dim3(256, 1, 1), 0, 0,
            d_q8.ptr, d_rhs.ptr, d_out.ptr, k, rows, cols);
    } else if (mode == "remap-c0-s12") {
        hipLaunchKernelGGL((q8_single_group_remap_repro_kernel<0, 12>), grid, dim3(256, 1, 1), 0, 0,
            d_q8.ptr, d_rhs.ptr, d_out.ptr, k, rows, cols);
    } else if (mode == "remap-c0-s12-stage-selected") {
        hipLaunchKernelGGL((q8_single_group_remap_repro_kernel<0, 12, false, false, true, true>), grid, dim3(256, 1, 1), 0, 0,
            d_q8.ptr, d_rhs.ptr, d_out.ptr, k, rows, cols);
    } else if (mode == "single-group8-bmirror0") {
        hipLaunchKernelGGL((q8_single_group_bmirror_repro_kernel<8, 0>), grid, dim3(256, 1, 1), 0, 0,
            d_q8.ptr, d_rhs.ptr, d_out.ptr, k, rows, cols);
    } else if (mode == "single-group12-bmirror0") {
        hipLaunchKernelGGL((q8_single_group_bmirror_repro_kernel<12, 0>), grid, dim3(256, 1, 1), 0, 0,
            d_q8.ptr, d_rhs.ptr, d_out.ptr, k, rows, cols);
    } else if (mode == "single-group8-bcopy") {
        hipLaunchKernelGGL((q8_single_group_copy_repro_kernel<8, false, true>), grid, dim3(256, 1, 1), 0, 0,
            d_q8.ptr, d_rhs.ptr, d_out.ptr, k, rows, cols);
    } else if (mode == "single-group8-abcopy") {
        hipLaunchKernelGGL((q8_single_group_copy_repro_kernel<8, true, true>), grid, dim3(256, 1, 1), 0, 0,
            d_q8.ptr, d_rhs.ptr, d_out.ptr, k, rows, cols);
    } else if (mode == "single-group12-bcopy") {
        hipLaunchKernelGGL((q8_single_group_copy_repro_kernel<12, false, true>), grid, dim3(256, 1, 1), 0, 0,
            d_q8.ptr, d_rhs.ptr, d_out.ptr, k, rows, cols);
    } else if (mode == "single-group12-abcopy") {
        hipLaunchKernelGGL((q8_single_group_copy_repro_kernel<12, true, true>), grid, dim3(256, 1, 1), 0, 0,
            d_q8.ptr, d_rhs.ptr, d_out.ptr, k, rows, cols);
    } else if (mode == "single-group0-bcopy-stage") {
        hipLaunchKernelGGL((q8_array_fullb_phase_repro_kernel<0, 0, 1, 1, false, false, true, true>), grid, dim3(256, 1, 1), 0, 0,
            d_q8.ptr, d_rhs.ptr, d_out.ptr, k, rows, cols);
    } else if (mode == "single-group8-bcopy-stage") {
        hipLaunchKernelGGL((q8_array_fullb_phase_repro_kernel<8, 2, 1, 1, false, false, true, true>), grid, dim3(256, 1, 1), 0, 0,
            d_q8.ptr, d_rhs.ptr, d_out.ptr, k, rows, cols);
    } else if (mode == "single-group8-abcopy-stage") {
        hipLaunchKernelGGL((q8_array_fullb_phase_repro_kernel<8, 2, 1, 1, false, true, true, true>), grid, dim3(256, 1, 1), 0, 0,
            d_q8.ptr, d_rhs.ptr, d_out.ptr, k, rows, cols);
    } else if (mode == "single-group8-bcopy-stage-selected") {
        hipLaunchKernelGGL((q8_array_fullb_phase_repro_kernel<8, 2, 1, 1, false, false, true, true, true>), grid, dim3(256, 1, 1), 0, 0,
            d_q8.ptr, d_rhs.ptr, d_out.ptr, k, rows, cols);
    } else if (mode == "single-group12-bcopy-stage") {
        hipLaunchKernelGGL((q8_array_fullb_phase_repro_kernel<12, 3, 1, 1, false, false, true, true>), grid, dim3(256, 1, 1), 0, 0,
            d_q8.ptr, d_rhs.ptr, d_out.ptr, k, rows, cols);
    } else if (mode == "single-group12-abcopy-stage") {
        hipLaunchKernelGGL((q8_array_fullb_phase_repro_kernel<12, 3, 1, 1, false, true, true, true>), grid, dim3(256, 1, 1), 0, 0,
            d_q8.ptr, d_rhs.ptr, d_out.ptr, k, rows, cols);
    } else if (mode == "single-group12-bcopy-stage-selected") {
        hipLaunchKernelGGL((q8_array_fullb_phase_repro_kernel<12, 3, 1, 1, false, false, true, true, true>), grid, dim3(256, 1, 1), 0, 0,
            d_q8.ptr, d_rhs.ptr, d_out.ptr, k, rows, cols);
    } else if (mode == "single-group12-abcopy-stage-selected") {
        hipLaunchKernelGGL((q8_array_fullb_phase_repro_kernel<12, 3, 1, 1, false, true, true, true, true>), grid, dim3(256, 1, 1), 0, 0,
            d_q8.ptr, d_rhs.ptr, d_out.ptr, k, rows, cols);
    } else if (mode == "single-group12-bcopy-stage-selected-acccopy") {
        hipLaunchKernelGGL((q8_array_fullb_phase_repro_kernel<12, 3, 1, 1, false, false, true, true, true, true>), grid, dim3(256, 1, 1), 0, 0,
            d_q8.ptr, d_rhs.ptr, d_out.ptr, k, rows, cols);
    } else if (mode == "single-group12-abcopy-stage-selected-acccopy") {
        hipLaunchKernelGGL((q8_array_fullb_phase_repro_kernel<12, 3, 1, 1, false, true, true, true, true, true>), grid, dim3(256, 1, 1), 0, 0,
            d_q8.ptr, d_rhs.ptr, d_out.ptr, k, rows, cols);
    } else if (mode == "single-group12-bcopy-stage-selected-regcopy") {
        hipLaunchKernelGGL((q8_array_fullb_phase_repro_kernel<12, 3, 1, 1, false, false, true, true, true, false, true>), grid, dim3(256, 1, 1), 0, 0,
            d_q8.ptr, d_rhs.ptr, d_out.ptr, k, rows, cols);
    } else if (mode == "single-group12-abcopy-stage-selected-regcopy") {
        hipLaunchKernelGGL((q8_array_fullb_phase_repro_kernel<12, 3, 1, 1, false, true, true, true, true, false, true>), grid, dim3(256, 1, 1), 0, 0,
            d_q8.ptr, d_rhs.ptr, d_out.ptr, k, rows, cols);
    } else {
        std::fprintf(stderr, "unknown mode: %s\n", mode.c_str());
        return 2;
    }
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_out.data(), d_out.ptr, h_out.size() * sizeof(h_out[0]), hipMemcpyDeviceToHost));

    size_t active = 0;
    size_t nan = 0;
    size_t inf = 0;
    size_t sentinel = 0;
    size_t bad = 0;
    float max_abs = 0.0f;
    group_stats by_group[16];
    for (size_t i = 0; i < h_out.size(); ++i) {
        if (!output_is_active(i, rows, mode)) {
            continue;
        }
        const int group = output_group(i, rows);
        group_stats & gs = by_group[group];
        ++active;
        ++gs.active;
        const float actual = h_out[i];
        if (actual == -7777.0f) {
            ++sentinel;
            ++gs.sentinel;
            note_bad_sample(gs, i, rows, actual, 0.0f, INFINITY);
        }
        if (std::isnan(actual)) {
            ++nan;
            ++bad;
            ++gs.nan;
            ++gs.bad;
            note_bad_sample(gs, i, rows, actual, 0.0f, NAN);
            continue;
        }
        if (std::isinf(actual)) {
            ++inf;
            ++bad;
            ++gs.inf;
            ++gs.bad;
            note_bad_sample(gs, i, rows, actual, 0.0f, INFINITY);
            continue;
        }
        const float expected = expected_value_for_output(i, rows, cols, mode, ref);
        const float err = std::fabs(actual - expected);
        max_abs = std::max(max_abs, err);
        gs.max_abs = std::max(gs.max_abs, err);
        if (err > 0.25f) {
            ++bad;
            ++gs.bad;
            note_bad_sample(gs, i, rows, actual, expected, err);
        }
    }

    std::printf(
        "%s rows=%d cols=%d k=%d active=%zu bad=%zu nan=%zu inf=%zu sentinel=%zu max_abs=%g\n",
        mode.c_str(), rows, cols, k, active, bad, nan, inf, sentinel, max_abs);
    for (int group = 0; group < 16; ++group) {
        const group_stats & gs = by_group[group];
        if (gs.active == 0 || (gs.bad == 0 && gs.nan == 0 && gs.inf == 0 && gs.sentinel == 0)) {
            continue;
        }
        std::printf(
            "  group=%d active=%zu bad=%zu nan=%zu inf=%zu sentinel=%zu max_abs=%g\n",
            group, gs.active, gs.bad, gs.nan, gs.inf, gs.sentinel, gs.max_abs);
        if (gs.have_first_bad) {
            std::printf(
                "    first_bad row=%d col=%d lane=%d slot=%d actual=%g expected=%g err=%g\n",
                gs.first_row, gs.first_col, gs.first_lane, gs.first_slot,
                gs.first_actual, gs.first_expected, gs.first_err);
        }
        for (int sample = 0; sample < gs.sample_count; ++sample) {
            std::printf(
                "    bad_sample[%d] row=%d col=%d lane=%d slot=%d actual=%g expected=%g err=%g\n",
                sample,
                gs.sample_row[sample],
                gs.sample_col[sample],
                gs.sample_lane[sample],
                gs.sample_slot[sample],
                gs.sample_actual[sample],
                gs.sample_expected[sample],
                gs.sample_err[sample]);
        }
    }
    return (nan == 0 && inf == 0 && sentinel == 0) ? 0 : 1;
}

int main(int argc, char ** argv) {
    std::string mode = "all";
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            mode = argv[++i];
        } else {
            std::fprintf(stderr, "usage: %s [--mode array8-fullb|array16-direct-raw|array16-direct-raw-bcopy|array16-direct-raw-abcopy|contract-direct192-abcopy|contract-phase96-abcopy|array8-b2|array8-fullb-2phase|array8-fullb-2phase-consume|array8-fullb-2phase-bcopy|array8-fullb-2phase-bcopy-stage|array8-fullb-2phase-abcopy|batched4|batched4-consume|single-group0|single-group0-consume|single-group0-opsel1|single-group0-bcopy-stage|single-group8|single-group8-consume|single-group8-opsel1|single-group8-bmirror0|single-group8-bcopy|single-group8-abcopy|single-group8-bcopy-stage|single-group8-abcopy-stage|single-group8-bcopy-stage-selected|single-group12|single-group12-consume|single-group12-opsel1|single-group12-bmirror0|single-group12-bcopy|single-group12-abcopy|single-group12-bcopy-stage|single-group12-abcopy-stage|single-group12-bcopy-stage-selected|single-group12-abcopy-stage-selected|single-group12-bcopy-stage-selected-acccopy|single-group12-abcopy-stage-selected-acccopy|single-group12-bcopy-stage-selected-regcopy|single-group12-abcopy-stage-selected-regcopy|single-group12-abcopy-dual-stage-raw-first|single-group12-abcopy-dual-stage-stage-first|single-group13|single-group13-consume|remap-c8-s0|remap-c0-s8|remap-c12-s0|remap-c12-s0-bcopy-stage-selected|remap-c12-s0-abcopy-stage-selected|remap-c0-s12|remap-c0-s12-stage-selected|bfrag-dump|all]\n", argv[0]);
            return 2;
        }
    }

    int status = 0;
    const int rows = 64;
    const int k = 4096;
    if (mode == "all" || mode == "array8-fullb") {
        status |= run_case("array8-fullb", rows, 64, k);
        status |= run_case("array8-fullb", rows, 33, k);
    }
    if (mode == "array16-direct-raw" ||
            mode == "array16-direct-raw-bcopy" ||
            mode == "array16-direct-raw-abcopy") {
        status |= run_case(mode, rows, 64, k);
        status |= run_case(mode, rows, 33, k);
    }
    if (mode == "contract-direct192-abcopy" || mode == "contract-phase96-abcopy") {
        status |= run_contract_case(mode, rows, 64, k);
        status |= run_contract_case(mode, rows, 33, k);
    }
    if (mode == "all" || mode == "array8-b2") {
        status |= run_case("array8-b2", rows, 64, k);
        status |= run_case("array8-b2", rows, 33, k);
    }
    if (mode == "all" || mode == "array8-fullb-2phase") {
        status |= run_case("array8-fullb-2phase", rows, 64, k);
        status |= run_case("array8-fullb-2phase", rows, 33, k);
    }
    if (mode == "all" || mode == "array8-fullb-2phase-consume") {
        status |= run_case("array8-fullb-2phase-consume", rows, 64, k);
        status |= run_case("array8-fullb-2phase-consume", rows, 33, k);
    }
    if (mode == "array8-fullb-2phase-bcopy") {
        status |= run_case("array8-fullb-2phase-bcopy", rows, 64, k);
        status |= run_case("array8-fullb-2phase-bcopy", rows, 33, k);
    }
    if (mode == "array8-fullb-2phase-bcopy-stage") {
        status |= run_case("array8-fullb-2phase-bcopy-stage", rows, 64, k);
    }
    if (mode == "array8-fullb-2phase-abcopy") {
        status |= run_case("array8-fullb-2phase-abcopy", rows, 64, k);
        status |= run_case("array8-fullb-2phase-abcopy", rows, 33, k);
    }
    if (mode == "all" || mode == "batched4") {
        status |= run_case("batched4", rows, 64, k);
        status |= run_case("batched4", rows, 33, k);
    }
    if (mode == "all" || mode == "batched4-consume") {
        status |= run_case("batched4-consume", rows, 64, k);
        status |= run_case("batched4-consume", rows, 33, k);
    }
    if (mode == "bfrag-dump") {
        status |= run_bfrag_dump_case(64, k);
        status |= run_bfrag_dump_case(33, k);
    }
    if (mode == "single-group12-abcopy-dual-stage-raw-first" ||
            mode == "single-group12-abcopy-dual-stage-stage-first") {
        status |= run_dual_stage_compare_case(mode, rows, 64, k);
        status |= run_dual_stage_compare_case(mode, rows, 33, k);
    }
    if (mode == "single-group0" || mode == "single-group0-consume" ||
            mode == "single-group0-opsel1" ||
            mode == "single-group0-bcopy-stage" ||
            mode == "single-group8" || mode == "single-group8-consume" ||
            mode == "single-group8-opsel1" ||
            mode == "single-group8-bmirror0" ||
            mode == "single-group8-bcopy" || mode == "single-group8-abcopy" ||
            mode == "single-group8-bcopy-stage" || mode == "single-group8-abcopy-stage" ||
            mode == "single-group8-bcopy-stage-selected" ||
            mode == "single-group12" || mode == "single-group12-consume" ||
            mode == "single-group12-opsel1" ||
            mode == "single-group12-bmirror0" ||
            mode == "single-group12-bcopy" || mode == "single-group12-abcopy" ||
            mode == "single-group12-bcopy-stage" || mode == "single-group12-abcopy-stage" ||
            mode == "single-group12-bcopy-stage-selected" ||
            mode == "single-group12-abcopy-stage-selected" ||
            mode == "single-group12-bcopy-stage-selected-acccopy" ||
            mode == "single-group12-abcopy-stage-selected-acccopy" ||
            mode == "single-group12-bcopy-stage-selected-regcopy" ||
            mode == "single-group12-abcopy-stage-selected-regcopy" ||
            mode == "single-group13" || mode == "single-group13-consume" ||
            mode == "remap-c8-s0" || mode == "remap-c0-s8" ||
            mode == "remap-c12-s0" ||
            mode == "remap-c12-s0-bcopy-stage-selected" ||
            mode == "remap-c12-s0-abcopy-stage-selected" ||
            mode == "remap-c0-s12" ||
            mode == "remap-c0-s12-stage-selected") {
        status |= run_case(mode, rows, 64, k);
        status |= run_case(mode, rows, 33, k);
    }
    return status;
}
