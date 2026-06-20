#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
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

static __device__ __forceinline__ void q8_repro_bm128_raw_store_slot(
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
    const long long row = row_base + static_cast<long long>(wave_row * 64 + (group & 3) * 16 + row_lane + slot * 4);
    const long long col = col_base + static_cast<long long>(wave_col * 64 + ((group >> 2) & 3) * 16 + col_lane);
    if (row < rows && col < cols) {
        hrx_q8_0_wmma_vk128_buffer_store_f32(
            dst_rsrc,
            col * rows_stride + row,
            static_cast<float>(acc[acc_index][slot * 2 + HRX_Q8_0_WMMA_VK128_W64_OPSEL]));
    }
}

static __device__ __forceinline__ void q8_repro_bm128_direct_raw_store_slot(
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
    const long long row = row_base + static_cast<long long>(wave_row * 64 + (group & 3) * 16 + row_lane + slot * 4);
    const long long col = col_base + static_cast<long long>(wave_col * 64 + ((group >> 2) & 3) * 16 + col_lane);
    if (row < rows && col < cols) {
        hrx_q8_0_wmma_vk128_buffer_store_f32(
            dst_rsrc,
            col * rows_stride + row,
            static_cast<float>(acc[group][slot * 2 + HRX_Q8_0_WMMA_VK128_W64_OPSEL]));
    }
}

static constexpr int Q8_REPRO_CONTRACT_LANES = 64;
static constexpr int Q8_REPRO_CONTRACT_SLOTS = 4;
static constexpr int Q8_REPRO_CONTRACT_GROUPS = 48;
static constexpr int Q8_REPRO_CONTRACT_VALUES =
    Q8_REPRO_CONTRACT_GROUPS * Q8_REPRO_CONTRACT_SLOTS * Q8_REPRO_CONTRACT_LANES;
static constexpr int Q8_REPRO_BM128_CONTRACT_ACTIVE_GROUPS = 64;
static constexpr int Q8_REPRO_BM128_CONTRACT_GROUPS = 96;
static constexpr int Q8_REPRO_BM128_CONTRACT_VALUES =
    Q8_REPRO_BM128_CONTRACT_GROUPS * Q8_REPRO_CONTRACT_SLOTS * Q8_REPRO_CONTRACT_LANES;

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

static __device__ __forceinline__ void q8_repro_bm128_contract_store_acc(
        __amdgpu_buffer_rsrc_t contract_rsrc,
        long long rows,
        long long cols,
        const hrx_q8_0_wmma_vk128_half8_vec * acc,
        unsigned int wave,
        int local_group,
        int slot,
        unsigned int lane) {
    const int wave_row = static_cast<int>(wave & 1u);
    const int wave_col = static_cast<int>(wave >> 1);
    const int row_lane = static_cast<int>(lane >> 4);
    const int col_lane = static_cast<int>(lane & 15u);
    const long long row = static_cast<long long>(
        wave_row * 64 + (local_group & 3) * 16 + row_lane + slot * 4);
    const long long col = static_cast<long long>(
        wave_col * 64 + ((local_group >> 2) & 3) * 16 + col_lane);
    if (row < rows && col < cols) {
        q8_repro_contract_store_value(
            contract_rsrc,
            static_cast<int>(wave) * 16 + local_group,
            slot,
            lane,
            static_cast<float>(acc[local_group][slot * 2 + HRX_Q8_0_WMMA_VK128_W64_OPSEL]));
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

static __device__ __forceinline__ _Float16 q8_repro_motif192_synthetic_half(
        int group,
        int slot,
        unsigned int lane) {
    return static_cast<_Float16>(q8_repro_contract_synthetic_value(group & 15, slot, lane));
}

static __device__ __forceinline__ int q8_repro_motif192_stage_index(
        unsigned int wave,
        int group,
        int slot,
        unsigned int lane) {
    const int row_lane = static_cast<int>(lane >> 4);
    const int col_lane = static_cast<int>(lane & 15u);
    return static_cast<int>(wave) * 16 * 16 + col_lane * 16 + row_lane + slot * 4;
}

static __device__ __forceinline__ int q8_repro_motif192_grouped_stage_index(
        unsigned int wave,
        int group,
        int slot,
        unsigned int lane) {
    const int group16 = group & 15;
    const int row_lane = static_cast<int>(lane >> 4);
    const int col_lane = static_cast<int>(lane & 15u);
    return ((static_cast<int>(wave) * 16 + group16) * 16 + col_lane) * 16 + row_lane + slot * 4;
}

static __device__ __forceinline__ int q8_repro_motif192_group4_stage_index(
        unsigned int wave,
        int group,
        int slot,
        unsigned int lane) {
    const int group4 = group & 3;
    const int row_lane = static_cast<int>(lane >> 4);
    const int col_lane = static_cast<int>(lane & 15u);
    return ((static_cast<int>(wave) * 4 + group4) * 16 + col_lane) * 16 + row_lane + slot * 4;
}

static __device__ __forceinline__ void q8_repro_motif192_stage_store_slot(
        _Float16 * sh_store,
        const hrx_q8_0_wmma_vk128_half8_vec * acc,
        unsigned int wave,
        int group,
        int slot,
        unsigned int lane) {
    const int acc_index = group & 15;
    hrx_q8_0_wmma_vk128_lds_u16_ptr sh_u16 =
        (hrx_q8_0_wmma_vk128_lds_u16_ptr) sh_store;
    hrx_q8_0_wmma_vk128_ds_store_u16(
        sh_u16 + q8_repro_motif192_stage_index(wave, group, slot, lane),
        hrx_q8_0_wmma_vk128_f16_to_u16(acc[acc_index][slot * 2 + HRX_Q8_0_WMMA_VK128_W64_OPSEL]));
}

static __device__ __forceinline__ void q8_repro_motif192_grouped_stage_store_slot(
        _Float16 * sh_store,
        const hrx_q8_0_wmma_vk128_half8_vec * acc,
        unsigned int wave,
        int group,
        int slot,
        unsigned int lane) {
    const int acc_index = group & 15;
    hrx_q8_0_wmma_vk128_lds_u16_ptr sh_u16 =
        (hrx_q8_0_wmma_vk128_lds_u16_ptr) sh_store;
    hrx_q8_0_wmma_vk128_ds_store_u16(
        sh_u16 + q8_repro_motif192_grouped_stage_index(wave, group, slot, lane),
        hrx_q8_0_wmma_vk128_f16_to_u16(acc[acc_index][slot * 2 + HRX_Q8_0_WMMA_VK128_W64_OPSEL]));
}

static __device__ __forceinline__ void q8_repro_motif192_group4_stage_store_slot(
        _Float16 * sh_store,
        const hrx_q8_0_wmma_vk128_half8_vec * acc,
        unsigned int wave,
        int group,
        int slot,
        unsigned int lane) {
    const int acc_index = group & 15;
    hrx_q8_0_wmma_vk128_lds_u16_ptr sh_u16 =
        (hrx_q8_0_wmma_vk128_lds_u16_ptr) sh_store;
    hrx_q8_0_wmma_vk128_ds_store_u16(
        sh_u16 + q8_repro_motif192_group4_stage_index(wave, group, slot, lane),
        hrx_q8_0_wmma_vk128_f16_to_u16(acc[acc_index][slot * 2 + HRX_Q8_0_WMMA_VK128_W64_OPSEL]));
}

static __device__ __forceinline__ void q8_repro_motif192_phase8_stage_store_slot(
        _Float16 * sh_store,
        const hrx_q8_0_wmma_vk128_half8_vec * acc,
        unsigned int wave,
        int group,
        int slot,
        unsigned int lane) {
    const int acc_index = group & 7;
    hrx_q8_0_wmma_vk128_lds_u16_ptr sh_u16 =
        (hrx_q8_0_wmma_vk128_lds_u16_ptr) sh_store;
    hrx_q8_0_wmma_vk128_ds_store_u16(
        sh_u16 + q8_repro_motif192_stage_index(wave, group, slot, lane),
        hrx_q8_0_wmma_vk128_f16_to_u16(acc[acc_index][slot * 2 + HRX_Q8_0_WMMA_VK128_W64_OPSEL]));
}

static __device__ __forceinline__ void q8_repro_motif192_stage_load_store_slot(
        __amdgpu_buffer_rsrc_t dst_rsrc,
        long long rows_stride,
        long long row_base,
        long long col_base,
        long long rows,
        long long cols,
        const _Float16 * sh_store,
        unsigned int wave,
        int group,
        int slot,
        unsigned int lane,
        bool wait_after_load) {
    const int group16 = group & 15;
    const int wave_row = static_cast<int>(wave & 1u);
    const int wave_col = static_cast<int>(wave >> 1);
    const int row_lane = static_cast<int>(lane >> 4);
    const int col_lane = static_cast<int>(lane & 15u);
    const long long row = row_base + static_cast<long long>(wave_row * 64 + (group16 & 3) * 16 + row_lane + slot * 4);
    const long long col = col_base + static_cast<long long>(wave_col * 64 + ((group16 >> 2) & 3) * 16 + col_lane);
    hrx_q8_0_wmma_vk128_lds_const_u16_ptr sh_u16 =
        (hrx_q8_0_wmma_vk128_lds_const_u16_ptr) sh_store;
    const _Float16 value = hrx_q8_0_wmma_vk128_u16_to_f16(
        hrx_q8_0_wmma_vk128_ds_load_u16_d16(
            sh_u16 + q8_repro_motif192_stage_index(wave, group, slot, lane)));
    if (wait_after_load) {
        asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    }
    if (row < rows && col < cols) {
        hrx_q8_0_wmma_vk128_buffer_store_f32(dst_rsrc, col * rows_stride + row, static_cast<float>(value));
    }
}

static __device__ __forceinline__ void q8_repro_motif192_grouped_stage_load_store_slot(
        __amdgpu_buffer_rsrc_t dst_rsrc,
        long long rows_stride,
        long long row_base,
        long long col_base,
        long long rows,
        long long cols,
        const _Float16 * sh_store,
        unsigned int wave,
        int group,
        int slot,
        unsigned int lane,
        bool wait_after_load) {
    const int group16 = group & 15;
    const int wave_row = static_cast<int>(wave & 1u);
    const int wave_col = static_cast<int>(wave >> 1);
    const int row_lane = static_cast<int>(lane >> 4);
    const int col_lane = static_cast<int>(lane & 15u);
    const long long row = row_base + static_cast<long long>(wave_row * 64 + (group16 & 3) * 16 + row_lane + slot * 4);
    const long long col = col_base + static_cast<long long>(wave_col * 64 + ((group16 >> 2) & 3) * 16 + col_lane);
    hrx_q8_0_wmma_vk128_lds_const_u16_ptr sh_u16 =
        (hrx_q8_0_wmma_vk128_lds_const_u16_ptr) sh_store;
    const _Float16 value = hrx_q8_0_wmma_vk128_u16_to_f16(
        hrx_q8_0_wmma_vk128_ds_load_u16_d16(
            sh_u16 + q8_repro_motif192_grouped_stage_index(wave, group, slot, lane)));
    if (wait_after_load) {
        asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    }
    if (row < rows && col < cols) {
        hrx_q8_0_wmma_vk128_buffer_store_f32(dst_rsrc, col * rows_stride + row, static_cast<float>(value));
    }
}

static __device__ __forceinline__ void q8_repro_motif192_group4_stage_load_store_slot(
        __amdgpu_buffer_rsrc_t dst_rsrc,
        long long rows_stride,
        long long row_base,
        long long col_base,
        long long rows,
        long long cols,
        const _Float16 * sh_store,
        unsigned int wave,
        int group,
        int slot,
        unsigned int lane,
        bool wait_after_load) {
    const int group16 = group & 15;
    const int wave_row = static_cast<int>(wave & 1u);
    const int wave_col = static_cast<int>(wave >> 1);
    const int row_lane = static_cast<int>(lane >> 4);
    const int col_lane = static_cast<int>(lane & 15u);
    const long long row = row_base + static_cast<long long>(wave_row * 64 + (group16 & 3) * 16 + row_lane + slot * 4);
    const long long col = col_base + static_cast<long long>(wave_col * 64 + ((group16 >> 2) & 3) * 16 + col_lane);
    hrx_q8_0_wmma_vk128_lds_const_u16_ptr sh_u16 =
        (hrx_q8_0_wmma_vk128_lds_const_u16_ptr) sh_store;
    const _Float16 value = hrx_q8_0_wmma_vk128_u16_to_f16(
        hrx_q8_0_wmma_vk128_ds_load_u16_d16(
            sh_u16 + q8_repro_motif192_group4_stage_index(wave, group, slot, lane)));
    if (wait_after_load) {
        asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    }
    if (row < rows && col < cols) {
        hrx_q8_0_wmma_vk128_buffer_store_f32(dst_rsrc, col * rows_stride + row, static_cast<float>(value));
    }
}

static __device__ __forceinline__ int q8_repro_accpark_index(
        unsigned int wave,
        int group,
        int slot,
        unsigned int lane) {
    return (((static_cast<int>(wave) * 16 + group) * 4 + slot) * 64) + static_cast<int>(lane);
}

static __device__ __forceinline__ void q8_repro_accpark_store_slot(
        _Float16 * sh_acc,
        const hrx_q8_0_wmma_vk128_half8_vec * acc,
        unsigned int wave,
        int group,
        int slot,
        unsigned int lane) {
    const int acc_index = group & 7;
    hrx_q8_0_wmma_vk128_lds_u16_ptr sh_u16 =
        (hrx_q8_0_wmma_vk128_lds_u16_ptr) sh_acc;
    hrx_q8_0_wmma_vk128_ds_store_u16(
        sh_u16 + q8_repro_accpark_index(wave, group, slot, lane),
        hrx_q8_0_wmma_vk128_f16_to_u16(acc[acc_index][slot * 2 + HRX_Q8_0_WMMA_VK128_W64_OPSEL]));
}

static __device__ __forceinline__ _Float16 q8_repro_accpark_load_slot(
        const _Float16 * sh_acc,
        unsigned int wave,
        int group,
        int slot,
        unsigned int lane) {
    hrx_q8_0_wmma_vk128_lds_const_u16_ptr sh_u16 =
        (hrx_q8_0_wmma_vk128_lds_const_u16_ptr) sh_acc;
    return hrx_q8_0_wmma_vk128_u16_to_f16(
        hrx_q8_0_wmma_vk128_ds_load_u16_d16(
            sh_u16 + q8_repro_accpark_index(wave, group, slot, lane)));
}

static __device__ __forceinline__ void q8_repro_accpark_load_group(
        hrx_q8_0_wmma_vk128_half8_vec * acc,
        const _Float16 * sh_acc,
        unsigned int wave,
        int group_base,
        int local,
        unsigned int lane) {
    const int group = group_base + local;
#pragma unroll
    for (int slot = 0; slot < 4; ++slot) {
        acc[local][slot * 2 + HRX_Q8_0_WMMA_VK128_W64_OPSEL] =
            q8_repro_accpark_load_slot(sh_acc, wave, group, slot, lane);
        acc[local][slot * 2 + (HRX_Q8_0_WMMA_VK128_W64_OPSEL ? 0 : 1)] = static_cast<_Float16>(0.0f);
    }
}

static __device__ __forceinline__ void q8_repro_accpark_store_group(
        _Float16 * sh_acc,
        const hrx_q8_0_wmma_vk128_half8_vec * acc,
        unsigned int wave,
        int group_base,
        int local,
        unsigned int lane) {
    const int group = group_base + local;
#pragma unroll
    for (int slot = 0; slot < 4; ++slot) {
        q8_repro_accpark_store_slot(sh_acc, acc, wave, group, slot, lane);
    }
}

static __device__ __forceinline__ int q8_repro_accpark_fullvec_index(
        unsigned int wave,
        int local,
        int elem,
        unsigned int lane) {
    return (((static_cast<int>(wave) * 8 + local) * 8 + elem) * 64) + static_cast<int>(lane);
}

static __device__ __forceinline__ void q8_repro_accpark_fullvec_store_group(
        _Float16 * sh_acc,
        const hrx_q8_0_wmma_vk128_half8_vec * acc,
        unsigned int wave,
        int local,
        unsigned int lane) {
    hrx_q8_0_wmma_vk128_lds_u16_ptr sh_u16 =
        (hrx_q8_0_wmma_vk128_lds_u16_ptr) sh_acc;
#pragma unroll
    for (int elem = 0; elem < 8; ++elem) {
        hrx_q8_0_wmma_vk128_ds_store_u16(
            sh_u16 + q8_repro_accpark_fullvec_index(wave, local, elem, lane),
            hrx_q8_0_wmma_vk128_f16_to_u16(acc[local][elem]));
    }
}

static __device__ __forceinline__ _Float16 q8_repro_accpark_fullvec_load_elem(
        const _Float16 * sh_acc,
        unsigned int wave,
        int local,
        int elem,
        unsigned int lane) {
    hrx_q8_0_wmma_vk128_lds_const_u16_ptr sh_u16 =
        (hrx_q8_0_wmma_vk128_lds_const_u16_ptr) sh_acc;
    return hrx_q8_0_wmma_vk128_u16_to_f16(
        hrx_q8_0_wmma_vk128_ds_load_u16_d16(
            sh_u16 + q8_repro_accpark_fullvec_index(wave, local, elem, lane)));
}

static __device__ __forceinline__ void q8_repro_accpark_fullvec_load_group(
        hrx_q8_0_wmma_vk128_half8_vec * acc,
        const _Float16 * sh_acc,
        unsigned int wave,
        int local,
        unsigned int lane) {
#pragma unroll
    for (int elem = 0; elem < 8; ++elem) {
        acc[local][elem] = q8_repro_accpark_fullvec_load_elem(sh_acc, wave, local, elem, lane);
    }
}

static constexpr int Q8_REPRO_MOTIF192_STORE_FULL = 0;
static constexpr int Q8_REPRO_MOTIF192_STORE_DIRECT = 1;
static constexpr int Q8_REPRO_MOTIF192_STORE_STAGE16 = 2;
static constexpr int Q8_REPRO_MOTIF192_STORE_STAGE32 = 3;

__global__ __launch_bounds__(256, 1)
void q8_motif192_synthetic_store_kernel(
        float * dst,
        long long rows,
        long long cols) {
    constexpr int BM = 128;
    constexpr int BN = 128;

    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int wave = tid >> 6u;
    const unsigned int lane = tid & 63u;
    const long long row_base = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * BM;
    const long long col_base = static_cast<long long>(__builtin_amdgcn_workgroup_id_y()) * BN;
    if (row_base >= rows || col_base >= cols) {
        return;
    }

    const __amdgpu_buffer_rsrc_t dst_rsrc = hrx_q8_0_wmma_vk128_make_dst_rsrc(dst);
    __shared__ _Float16 sh_store[4 * 16 * 16];
    hrx_q8_0_wmma_vk128_half8_vec acc[16] = {};

#pragma unroll
    for (int group = 0; group < 16; ++group) {
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            const _Float16 value = q8_repro_motif192_synthetic_half(group, slot, lane);
            acc[group][slot * 2 + 0] = value;
            acc[group][slot * 2 + 1] = value;
        }
    }

#pragma unroll
    for (int group = 0; group < 16; ++group) {
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            q8_repro_bm128_raw_store_slot(
                dst_rsrc, rows, row_base, col_base, rows, cols, acc, wave, group, slot, lane);
        }
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            q8_repro_motif192_stage_store_slot(sh_store, acc, wave, group + 16, slot, lane);
        }
        asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            q8_repro_motif192_stage_load_store_slot(
                dst_rsrc, rows, row_base, col_base, rows, cols, sh_store, wave, group + 16, slot, lane, false);
        }
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            q8_repro_motif192_stage_store_slot(sh_store, acc, wave, group + 32, slot, lane);
        }
        asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            q8_repro_motif192_stage_load_store_slot(
                dst_rsrc, rows, row_base, col_base, rows, cols, sh_store, wave, group + 32, slot, lane, false);
        }
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();
}

__global__ __launch_bounds__(256, 1)
void q8_motif192_wmma_store_kernel(
        float * dst,
        long long rows,
        long long cols,
        int store_mode,
        bool wait_after_load) {
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 32;
    constexpr int SHARED_STRIDE = HRX_Q8_0_WMMA_VK128_SHARED_STRIDE;

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
    __shared__ _Float16 sh_store[4 * 16 * 16];
    hrx_q8_0_wmma_vk128_half8_vec acc[16] = {};

    for (int idx = static_cast<int>(tid); idx < BM * BK; idx += 256) {
        const int r = idx / BK;
        const int kk = idx - r * BK;
        const int row_sub = (r & 63) >> 4;
        sh_a[r * SHARED_STRIDE + kk] = static_cast<_Float16>(1.0f + static_cast<float>(row_sub));
    }
    for (int idx = static_cast<int>(tid); idx < BN * BK; idx += 256) {
        const int c = idx / BK;
        const int kk = idx - c * BK;
        const int col_sub = (c & 63) >> 4;
        sh_b[c * SHARED_STRIDE + kk] = static_cast<_Float16>(1.0f + static_cast<float>(col_sub));
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
        for (int col_sub = 0; col_sub < 4; ++col_sub) {
#pragma unroll
            for (int row_sub = 0; row_sub < 4; ++row_sub) {
                const int local = col_sub * 4 + row_sub;
                acc[local] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(
                    a_frag[row_sub],
                    b_frag[col_sub],
                    acc[local],
                    HRX_Q8_0_WMMA_VK128_W64_OPSEL != 0);
            }
        }
    }

#pragma unroll
    for (int group = 0; group < 16; ++group) {
        if (store_mode == Q8_REPRO_MOTIF192_STORE_FULL ||
                store_mode == Q8_REPRO_MOTIF192_STORE_DIRECT) {
#pragma unroll
            for (int slot = 0; slot < 4; ++slot) {
                q8_repro_bm128_raw_store_slot(
                    dst_rsrc, rows, row_base, col_base, rows, cols, acc, wave, group, slot, lane);
            }
        }
        if (store_mode == Q8_REPRO_MOTIF192_STORE_FULL ||
                store_mode == Q8_REPRO_MOTIF192_STORE_STAGE16) {
#pragma unroll
            for (int slot = 0; slot < 4; ++slot) {
                q8_repro_motif192_stage_store_slot(sh_store, acc, wave, group + 16, slot, lane);
            }
            asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
#pragma unroll
            for (int slot = 0; slot < 4; ++slot) {
                q8_repro_motif192_stage_load_store_slot(
                    dst_rsrc, rows, row_base, col_base, rows, cols, sh_store, wave, group + 16, slot, lane, wait_after_load);
            }
        }
        if (store_mode == Q8_REPRO_MOTIF192_STORE_FULL ||
                store_mode == Q8_REPRO_MOTIF192_STORE_STAGE32) {
#pragma unroll
            for (int slot = 0; slot < 4; ++slot) {
                q8_repro_motif192_stage_store_slot(sh_store, acc, wave, group + 32, slot, lane);
            }
            asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
#pragma unroll
            for (int slot = 0; slot < 4; ++slot) {
                q8_repro_motif192_stage_load_store_slot(
                    dst_rsrc, rows, row_base, col_base, rows, cols, sh_store, wave, group + 32, slot, lane, wait_after_load);
            }
        }
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();
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

static __device__ __forceinline__ hrx_q8_0_wmma_vk128_half8_vec q8_repro_wmma_f16_w64_asm(
        hrx_q8_0_wmma_vk128_half16_vec a_frag,
        hrx_q8_0_wmma_vk128_half16_vec b_frag,
        hrx_q8_0_wmma_vk128_half8_vec acc) {
    hrx_q8_0_wmma_vk128_half8_vec out;
#if HRX_Q8_0_WMMA_VK128_W64_OPSEL
    asm volatile("v_wmma_f16_16x16x16_f16 %0, %1, %2, %3 op_sel:[0,0,1]\n"
                 : "=v"(out)
                 : "v"(a_frag), "v"(b_frag), "v"(acc)
                 : "memory");
#else
    asm volatile("v_wmma_f16_16x16x16_f16 %0, %1, %2, %3\n"
                 : "=v"(out)
                 : "v"(a_frag), "v"(b_frag), "v"(acc)
                 : "memory");
#endif
    return out;
}

static __device__ __forceinline__ hrx_q8_0_wmma_vk128_half8_vec q8_repro_wmma_f16_w64_asm_inout(
        hrx_q8_0_wmma_vk128_half16_vec a_frag,
        hrx_q8_0_wmma_vk128_half16_vec b_frag,
        hrx_q8_0_wmma_vk128_half8_vec acc) {
    hrx_q8_0_wmma_vk128_half8_vec out = acc;
#if HRX_Q8_0_WMMA_VK128_W64_OPSEL
    asm volatile("v_wmma_f16_16x16x16_f16 %0, %1, %2, %0 op_sel:[0,0,1]\n"
                 : "+v"(out)
                 : "v"(a_frag), "v"(b_frag)
                 : "memory");
#else
    asm volatile("v_wmma_f16_16x16x16_f16 %0, %1, %2, %0\n"
                 : "+v"(out)
                 : "v"(a_frag), "v"(b_frag)
                 : "memory");
#endif
    return out;
}

template <int wait_lgkmcnt>
static __device__ __forceinline__ hrx_q8_0_wmma_vk128_half16_vec q8_repro_dep_copy_initial(
        hrx_q8_0_wmma_vk128_half16_vec src,
        hrx_q8_0_wmma_vk128_half16_vec dep0,
        hrx_q8_0_wmma_vk128_half16_vec dep1,
        hrx_q8_0_wmma_vk128_half16_vec dep2,
        hrx_q8_0_wmma_vk128_half16_vec dep3,
        hrx_q8_0_wmma_vk128_half16_vec dep4,
        hrx_q8_0_wmma_vk128_half16_vec dep5,
        hrx_q8_0_wmma_vk128_half16_vec dep6) {
    const hrx_q8_0_wmma_vk128_u32x8_vec in =
        __builtin_bit_cast(hrx_q8_0_wmma_vk128_u32x8_vec, src);
    hrx_q8_0_wmma_vk128_u32x8_vec out;
    asm volatile("s_waitcnt lgkmcnt(%16)\n\t"
                 "v_mov_b32 %0, %8\n\t"
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
                   "v"(in[4]), "v"(in[5]), "v"(in[6]), "v"(in[7]),
                   "n"(wait_lgkmcnt),
                   "v"(dep0), "v"(dep1), "v"(dep2), "v"(dep3),
                   "v"(dep4), "v"(dep5), "v"(dep6)
                 : "memory");
    return __builtin_bit_cast(hrx_q8_0_wmma_vk128_half16_vec, out);
}

template <int wait_lgkmcnt>
static __device__ __forceinline__ hrx_q8_0_wmma_vk128_half16_vec q8_repro_dep_copy_after_acc(
        hrx_q8_0_wmma_vk128_half16_vec src,
        hrx_q8_0_wmma_vk128_half16_vec token,
        hrx_q8_0_wmma_vk128_half8_vec prev_acc) {
    const hrx_q8_0_wmma_vk128_u32x8_vec in =
        __builtin_bit_cast(hrx_q8_0_wmma_vk128_u32x8_vec, src);
    hrx_q8_0_wmma_vk128_u32x8_vec out;
    asm volatile("s_waitcnt lgkmcnt(%16)\n\t"
                 "v_mov_b32 %0, %8\n\t"
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
                   "v"(in[4]), "v"(in[5]), "v"(in[6]), "v"(in[7]),
                   "n"(wait_lgkmcnt), "v"(token), "v"(prev_acc)
                 : "memory");
    return __builtin_bit_cast(hrx_q8_0_wmma_vk128_half16_vec, out);
}

static __device__ __forceinline__ hrx_q8_0_wmma_vk128_half16_vec q8_repro_dep_copy_after_token(
        hrx_q8_0_wmma_vk128_half16_vec src,
        hrx_q8_0_wmma_vk128_half16_vec token) {
    const hrx_q8_0_wmma_vk128_u32x8_vec in =
        __builtin_bit_cast(hrx_q8_0_wmma_vk128_u32x8_vec, src);
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
                   "v"(in[4]), "v"(in[5]), "v"(in[6]), "v"(in[7]),
                   "v"(token)
                 : "memory");
    return __builtin_bit_cast(hrx_q8_0_wmma_vk128_half16_vec, out);
}

#define Q8_REPRO_DEP_WMMA_INITIAL(ACC, TILE, A, B, W, D0, D1, D2, D3, D4, D5, D6) \
    do { \
        const hrx_q8_0_wmma_vk128_half16_vec a_dep = \
            q8_repro_dep_copy_initial<W>((A), (D0), (D1), (D2), (D3), (D4), (D5), (D6)); \
        const hrx_q8_0_wmma_vk128_half16_vec b_dep = q8_repro_dep_copy_after_token((B), a_dep); \
        (ACC)[TILE] = q8_repro_wmma_f16_w64_asm(a_dep, b_dep, (ACC)[TILE]); \
    } while (0)

#define Q8_REPRO_DEP_WMMA_AFTER(ACC, TILE, A, B, W, TOKEN, PREV) \
    do { \
        const hrx_q8_0_wmma_vk128_half16_vec a_dep = \
            q8_repro_dep_copy_after_acc<W>((A), (TOKEN), (ACC)[PREV]); \
        const hrx_q8_0_wmma_vk128_half16_vec b_dep = q8_repro_dep_copy_after_token((B), a_dep); \
        (ACC)[TILE] = q8_repro_wmma_f16_w64_asm(a_dep, b_dep, (ACC)[TILE]); \
    } while (0)

#define Q8_REPRO_WAIT_WMMA(ACC, TILE, A, B, W) \
    do { \
        asm volatile("s_waitcnt lgkmcnt(%0)\n" :: "n"(W) : "memory"); \
        (ACC)[TILE] = q8_repro_wmma_f16_w64_asm((A), (B), (ACC)[TILE]); \
    } while (0)

template <bool dependent_ladder>
static __device__ __forceinline__ void q8_repro_issue_k2_wmma_ladder(
        hrx_q8_0_wmma_vk128_half8_vec * acc,
        const hrx_q8_0_wmma_vk128_half16_vec a_frag[2][4],
        const hrx_q8_0_wmma_vk128_half16_vec b_frag[2][4]) {
    if constexpr (dependent_ladder) {
        Q8_REPRO_DEP_WMMA_INITIAL(acc, 0, a_frag[0][0], b_frag[0][0], 51,
            a_frag[0][1], a_frag[0][2], a_frag[0][3],
            b_frag[0][0], b_frag[0][1], b_frag[0][2], b_frag[0][3]);
        Q8_REPRO_DEP_WMMA_AFTER(acc, 1, a_frag[0][1], b_frag[0][0], 47, a_frag[0][0], 0);
        Q8_REPRO_DEP_WMMA_AFTER(acc, 2, a_frag[0][2], b_frag[0][0], 43, a_frag[0][1], 1);
        Q8_REPRO_DEP_WMMA_AFTER(acc, 3, a_frag[0][3], b_frag[0][0], 39, a_frag[0][2], 2);
        Q8_REPRO_DEP_WMMA_AFTER(acc, 4, a_frag[0][0], b_frag[0][1], 40, a_frag[0][3], 3);
        Q8_REPRO_DEP_WMMA_AFTER(acc, 5, a_frag[0][1], b_frag[0][1], 36, b_frag[0][1], 4);
        Q8_REPRO_DEP_WMMA_AFTER(acc, 6, a_frag[0][2], b_frag[0][1], 32, a_frag[0][1], 5);
        Q8_REPRO_DEP_WMMA_AFTER(acc, 7, a_frag[0][3], b_frag[0][1], 24, a_frag[0][2], 6);
        Q8_REPRO_DEP_WMMA_AFTER(acc, 8, a_frag[0][0], b_frag[0][2], 20, a_frag[0][3], 7);
        Q8_REPRO_DEP_WMMA_AFTER(acc, 9, a_frag[0][1], b_frag[0][2], 16, b_frag[0][2], 8);
        Q8_REPRO_DEP_WMMA_AFTER(acc, 10, a_frag[0][2], b_frag[0][2], 12, a_frag[0][1], 9);
        Q8_REPRO_DEP_WMMA_AFTER(acc, 11, a_frag[0][3], b_frag[0][2], 8, a_frag[0][2], 10);
        Q8_REPRO_DEP_WMMA_AFTER(acc, 12, a_frag[0][0], b_frag[0][3], 4, a_frag[0][3], 11);
        Q8_REPRO_DEP_WMMA_AFTER(acc, 13, a_frag[0][1], b_frag[0][3], 0, b_frag[0][3], 12);
        Q8_REPRO_DEP_WMMA_AFTER(acc, 14, a_frag[0][2], b_frag[0][3], 0, a_frag[0][1], 13);
        Q8_REPRO_DEP_WMMA_AFTER(acc, 15, a_frag[0][3], b_frag[0][3], 0, a_frag[0][2], 14);

        Q8_REPRO_DEP_WMMA_AFTER(acc, 0, a_frag[1][0], b_frag[1][0], 51, b_frag[1][0], 15);
        Q8_REPRO_DEP_WMMA_AFTER(acc, 1, a_frag[1][1], b_frag[1][0], 47, a_frag[1][0], 0);
        Q8_REPRO_DEP_WMMA_AFTER(acc, 2, a_frag[1][2], b_frag[1][0], 43, a_frag[1][1], 1);
        Q8_REPRO_DEP_WMMA_AFTER(acc, 3, a_frag[1][3], b_frag[1][0], 39, a_frag[1][2], 2);
        Q8_REPRO_DEP_WMMA_AFTER(acc, 4, a_frag[1][0], b_frag[1][1], 40, a_frag[1][3], 3);
        Q8_REPRO_DEP_WMMA_AFTER(acc, 5, a_frag[1][1], b_frag[1][1], 36, b_frag[1][1], 4);
        Q8_REPRO_DEP_WMMA_AFTER(acc, 6, a_frag[1][2], b_frag[1][1], 32, a_frag[1][1], 5);
        Q8_REPRO_DEP_WMMA_AFTER(acc, 7, a_frag[1][3], b_frag[1][1], 24, a_frag[1][2], 6);
        Q8_REPRO_DEP_WMMA_AFTER(acc, 8, a_frag[1][0], b_frag[1][2], 20, a_frag[1][3], 7);
        Q8_REPRO_DEP_WMMA_AFTER(acc, 9, a_frag[1][1], b_frag[1][2], 16, b_frag[1][2], 8);
        Q8_REPRO_DEP_WMMA_AFTER(acc, 10, a_frag[1][2], b_frag[1][2], 12, a_frag[1][1], 9);
        Q8_REPRO_DEP_WMMA_AFTER(acc, 11, a_frag[1][3], b_frag[1][2], 8, a_frag[1][2], 10);
        Q8_REPRO_DEP_WMMA_AFTER(acc, 12, a_frag[1][0], b_frag[1][3], 4, a_frag[1][3], 11);
        Q8_REPRO_DEP_WMMA_AFTER(acc, 13, a_frag[1][1], b_frag[1][3], 0, b_frag[1][3], 12);
        Q8_REPRO_DEP_WMMA_AFTER(acc, 14, a_frag[1][2], b_frag[1][3], 0, a_frag[1][1], 13);
        Q8_REPRO_DEP_WMMA_AFTER(acc, 15, a_frag[1][3], b_frag[1][3], 0, a_frag[1][2], 14);
    } else {
        Q8_REPRO_WAIT_WMMA(acc, 0, a_frag[0][0], b_frag[0][0], 51);
        Q8_REPRO_WAIT_WMMA(acc, 1, a_frag[0][1], b_frag[0][0], 47);
        Q8_REPRO_WAIT_WMMA(acc, 2, a_frag[0][2], b_frag[0][0], 43);
        Q8_REPRO_WAIT_WMMA(acc, 3, a_frag[0][3], b_frag[0][0], 39);
        Q8_REPRO_WAIT_WMMA(acc, 4, a_frag[0][0], b_frag[0][1], 40);
        Q8_REPRO_WAIT_WMMA(acc, 5, a_frag[0][1], b_frag[0][1], 36);
        Q8_REPRO_WAIT_WMMA(acc, 6, a_frag[0][2], b_frag[0][1], 32);
        Q8_REPRO_WAIT_WMMA(acc, 7, a_frag[0][3], b_frag[0][1], 24);
        Q8_REPRO_WAIT_WMMA(acc, 8, a_frag[0][0], b_frag[0][2], 20);
        Q8_REPRO_WAIT_WMMA(acc, 9, a_frag[0][1], b_frag[0][2], 16);
        Q8_REPRO_WAIT_WMMA(acc, 10, a_frag[0][2], b_frag[0][2], 12);
        Q8_REPRO_WAIT_WMMA(acc, 11, a_frag[0][3], b_frag[0][2], 8);
        Q8_REPRO_WAIT_WMMA(acc, 12, a_frag[0][0], b_frag[0][3], 4);
        Q8_REPRO_WAIT_WMMA(acc, 13, a_frag[0][1], b_frag[0][3], 0);
        Q8_REPRO_WAIT_WMMA(acc, 14, a_frag[0][2], b_frag[0][3], 0);
        Q8_REPRO_WAIT_WMMA(acc, 15, a_frag[0][3], b_frag[0][3], 0);

        Q8_REPRO_WAIT_WMMA(acc, 0, a_frag[1][0], b_frag[1][0], 51);
        Q8_REPRO_WAIT_WMMA(acc, 1, a_frag[1][1], b_frag[1][0], 47);
        Q8_REPRO_WAIT_WMMA(acc, 2, a_frag[1][2], b_frag[1][0], 43);
        Q8_REPRO_WAIT_WMMA(acc, 3, a_frag[1][3], b_frag[1][0], 39);
        Q8_REPRO_WAIT_WMMA(acc, 4, a_frag[1][0], b_frag[1][1], 40);
        Q8_REPRO_WAIT_WMMA(acc, 5, a_frag[1][1], b_frag[1][1], 36);
        Q8_REPRO_WAIT_WMMA(acc, 6, a_frag[1][2], b_frag[1][1], 32);
        Q8_REPRO_WAIT_WMMA(acc, 7, a_frag[1][3], b_frag[1][1], 24);
        Q8_REPRO_WAIT_WMMA(acc, 8, a_frag[1][0], b_frag[1][2], 20);
        Q8_REPRO_WAIT_WMMA(acc, 9, a_frag[1][1], b_frag[1][2], 16);
        Q8_REPRO_WAIT_WMMA(acc, 10, a_frag[1][2], b_frag[1][2], 12);
        Q8_REPRO_WAIT_WMMA(acc, 11, a_frag[1][3], b_frag[1][2], 8);
        Q8_REPRO_WAIT_WMMA(acc, 12, a_frag[1][0], b_frag[1][3], 4);
        Q8_REPRO_WAIT_WMMA(acc, 13, a_frag[1][1], b_frag[1][3], 0);
        Q8_REPRO_WAIT_WMMA(acc, 14, a_frag[1][2], b_frag[1][3], 0);
        Q8_REPRO_WAIT_WMMA(acc, 15, a_frag[1][3], b_frag[1][3], 0);
    }
}

template <int col_start>
static __device__ __forceinline__ void q8_repro_issue_k2_wmma_ladder_phase8(
        hrx_q8_0_wmma_vk128_half8_vec * acc,
        const hrx_q8_0_wmma_vk128_half16_vec a_frag[2][4],
        const hrx_q8_0_wmma_vk128_half16_vec b_frag[2][4]) {
    static_assert(col_start == 0 || col_start == 2, "phase8 expects two 64-column phases");
    if constexpr (col_start == 0) {
        Q8_REPRO_WAIT_WMMA(acc, 0, a_frag[0][0], b_frag[0][0], 51);
        Q8_REPRO_WAIT_WMMA(acc, 1, a_frag[0][1], b_frag[0][0], 47);
        Q8_REPRO_WAIT_WMMA(acc, 2, a_frag[0][2], b_frag[0][0], 43);
        Q8_REPRO_WAIT_WMMA(acc, 3, a_frag[0][3], b_frag[0][0], 39);
        Q8_REPRO_WAIT_WMMA(acc, 4, a_frag[0][0], b_frag[0][1], 40);
        Q8_REPRO_WAIT_WMMA(acc, 5, a_frag[0][1], b_frag[0][1], 36);
        Q8_REPRO_WAIT_WMMA(acc, 6, a_frag[0][2], b_frag[0][1], 32);
        Q8_REPRO_WAIT_WMMA(acc, 7, a_frag[0][3], b_frag[0][1], 24);
        Q8_REPRO_WAIT_WMMA(acc, 0, a_frag[1][0], b_frag[1][0], 51);
        Q8_REPRO_WAIT_WMMA(acc, 1, a_frag[1][1], b_frag[1][0], 47);
        Q8_REPRO_WAIT_WMMA(acc, 2, a_frag[1][2], b_frag[1][0], 43);
        Q8_REPRO_WAIT_WMMA(acc, 3, a_frag[1][3], b_frag[1][0], 39);
        Q8_REPRO_WAIT_WMMA(acc, 4, a_frag[1][0], b_frag[1][1], 40);
        Q8_REPRO_WAIT_WMMA(acc, 5, a_frag[1][1], b_frag[1][1], 36);
        Q8_REPRO_WAIT_WMMA(acc, 6, a_frag[1][2], b_frag[1][1], 32);
        Q8_REPRO_WAIT_WMMA(acc, 7, a_frag[1][3], b_frag[1][1], 24);
    } else {
        Q8_REPRO_WAIT_WMMA(acc, 0, a_frag[0][0], b_frag[0][2], 20);
        Q8_REPRO_WAIT_WMMA(acc, 1, a_frag[0][1], b_frag[0][2], 16);
        Q8_REPRO_WAIT_WMMA(acc, 2, a_frag[0][2], b_frag[0][2], 12);
        Q8_REPRO_WAIT_WMMA(acc, 3, a_frag[0][3], b_frag[0][2], 8);
        Q8_REPRO_WAIT_WMMA(acc, 4, a_frag[0][0], b_frag[0][3], 4);
        Q8_REPRO_WAIT_WMMA(acc, 5, a_frag[0][1], b_frag[0][3], 0);
        Q8_REPRO_WAIT_WMMA(acc, 6, a_frag[0][2], b_frag[0][3], 0);
        Q8_REPRO_WAIT_WMMA(acc, 7, a_frag[0][3], b_frag[0][3], 0);
        Q8_REPRO_WAIT_WMMA(acc, 0, a_frag[1][0], b_frag[1][2], 20);
        Q8_REPRO_WAIT_WMMA(acc, 1, a_frag[1][1], b_frag[1][2], 16);
        Q8_REPRO_WAIT_WMMA(acc, 2, a_frag[1][2], b_frag[1][2], 12);
        Q8_REPRO_WAIT_WMMA(acc, 3, a_frag[1][3], b_frag[1][2], 8);
        Q8_REPRO_WAIT_WMMA(acc, 4, a_frag[1][0], b_frag[1][3], 4);
        Q8_REPRO_WAIT_WMMA(acc, 5, a_frag[1][1], b_frag[1][3], 0);
        Q8_REPRO_WAIT_WMMA(acc, 6, a_frag[1][2], b_frag[1][3], 0);
        Q8_REPRO_WAIT_WMMA(acc, 7, a_frag[1][3], b_frag[1][3], 0);
    }
}

#define Q8_REPRO_STREAM_WMMA(ACC, TILE, SH_A, SH_B, WAVE_ROW, WAVE_COL, LANE, K_TILE, ROW_SUB, COL_SUB) \
    do { \
        const hrx_q8_0_wmma_vk128_half16_vec a_stream = \
            hrx_q8_0_wmma_vk128_load_a_frag_w64_b64asm_nowait( \
                (SH_A), (WAVE_ROW) * 4 + (ROW_SUB), (K_TILE), (LANE)); \
        const hrx_q8_0_wmma_vk128_half16_vec b_stream = \
            hrx_q8_0_wmma_vk128_load_b_frag_w64_b64asm_nowait( \
                (SH_B), (WAVE_COL) * 4 + (COL_SUB), (K_TILE), (LANE)); \
        Q8_REPRO_WAIT_WMMA((ACC), (TILE), a_stream, b_stream, 0); \
    } while (0)

static __device__ __forceinline__ void q8_repro_issue_k2_wmma_ladder_streamfrag(
        hrx_q8_0_wmma_vk128_half8_vec * acc,
        hrx_q8_0_wmma_vk128_lds_half_ptr sh_a_lds,
        hrx_q8_0_wmma_vk128_lds_half_ptr sh_b_lds,
        int wave_row,
        int wave_col,
        unsigned int lane) {
    Q8_REPRO_STREAM_WMMA(acc, 0, sh_a_lds, sh_b_lds, wave_row, wave_col, lane, 0, 0, 0);
    Q8_REPRO_STREAM_WMMA(acc, 1, sh_a_lds, sh_b_lds, wave_row, wave_col, lane, 0, 1, 0);
    Q8_REPRO_STREAM_WMMA(acc, 2, sh_a_lds, sh_b_lds, wave_row, wave_col, lane, 0, 2, 0);
    Q8_REPRO_STREAM_WMMA(acc, 3, sh_a_lds, sh_b_lds, wave_row, wave_col, lane, 0, 3, 0);
    Q8_REPRO_STREAM_WMMA(acc, 4, sh_a_lds, sh_b_lds, wave_row, wave_col, lane, 0, 0, 1);
    Q8_REPRO_STREAM_WMMA(acc, 5, sh_a_lds, sh_b_lds, wave_row, wave_col, lane, 0, 1, 1);
    Q8_REPRO_STREAM_WMMA(acc, 6, sh_a_lds, sh_b_lds, wave_row, wave_col, lane, 0, 2, 1);
    Q8_REPRO_STREAM_WMMA(acc, 7, sh_a_lds, sh_b_lds, wave_row, wave_col, lane, 0, 3, 1);
    Q8_REPRO_STREAM_WMMA(acc, 8, sh_a_lds, sh_b_lds, wave_row, wave_col, lane, 0, 0, 2);
    Q8_REPRO_STREAM_WMMA(acc, 9, sh_a_lds, sh_b_lds, wave_row, wave_col, lane, 0, 1, 2);
    Q8_REPRO_STREAM_WMMA(acc, 10, sh_a_lds, sh_b_lds, wave_row, wave_col, lane, 0, 2, 2);
    Q8_REPRO_STREAM_WMMA(acc, 11, sh_a_lds, sh_b_lds, wave_row, wave_col, lane, 0, 3, 2);
    Q8_REPRO_STREAM_WMMA(acc, 12, sh_a_lds, sh_b_lds, wave_row, wave_col, lane, 0, 0, 3);
    Q8_REPRO_STREAM_WMMA(acc, 13, sh_a_lds, sh_b_lds, wave_row, wave_col, lane, 0, 1, 3);
    Q8_REPRO_STREAM_WMMA(acc, 14, sh_a_lds, sh_b_lds, wave_row, wave_col, lane, 0, 2, 3);
    Q8_REPRO_STREAM_WMMA(acc, 15, sh_a_lds, sh_b_lds, wave_row, wave_col, lane, 0, 3, 3);

    Q8_REPRO_STREAM_WMMA(acc, 0, sh_a_lds, sh_b_lds, wave_row, wave_col, lane, 1, 0, 0);
    Q8_REPRO_STREAM_WMMA(acc, 1, sh_a_lds, sh_b_lds, wave_row, wave_col, lane, 1, 1, 0);
    Q8_REPRO_STREAM_WMMA(acc, 2, sh_a_lds, sh_b_lds, wave_row, wave_col, lane, 1, 2, 0);
    Q8_REPRO_STREAM_WMMA(acc, 3, sh_a_lds, sh_b_lds, wave_row, wave_col, lane, 1, 3, 0);
    Q8_REPRO_STREAM_WMMA(acc, 4, sh_a_lds, sh_b_lds, wave_row, wave_col, lane, 1, 0, 1);
    Q8_REPRO_STREAM_WMMA(acc, 5, sh_a_lds, sh_b_lds, wave_row, wave_col, lane, 1, 1, 1);
    Q8_REPRO_STREAM_WMMA(acc, 6, sh_a_lds, sh_b_lds, wave_row, wave_col, lane, 1, 2, 1);
    Q8_REPRO_STREAM_WMMA(acc, 7, sh_a_lds, sh_b_lds, wave_row, wave_col, lane, 1, 3, 1);
    Q8_REPRO_STREAM_WMMA(acc, 8, sh_a_lds, sh_b_lds, wave_row, wave_col, lane, 1, 0, 2);
    Q8_REPRO_STREAM_WMMA(acc, 9, sh_a_lds, sh_b_lds, wave_row, wave_col, lane, 1, 1, 2);
    Q8_REPRO_STREAM_WMMA(acc, 10, sh_a_lds, sh_b_lds, wave_row, wave_col, lane, 1, 2, 2);
    Q8_REPRO_STREAM_WMMA(acc, 11, sh_a_lds, sh_b_lds, wave_row, wave_col, lane, 1, 3, 2);
    Q8_REPRO_STREAM_WMMA(acc, 12, sh_a_lds, sh_b_lds, wave_row, wave_col, lane, 1, 0, 3);
    Q8_REPRO_STREAM_WMMA(acc, 13, sh_a_lds, sh_b_lds, wave_row, wave_col, lane, 1, 1, 3);
    Q8_REPRO_STREAM_WMMA(acc, 14, sh_a_lds, sh_b_lds, wave_row, wave_col, lane, 1, 2, 3);
    Q8_REPRO_STREAM_WMMA(acc, 15, sh_a_lds, sh_b_lds, wave_row, wave_col, lane, 1, 3, 3);
}

static __device__ __forceinline__ void q8_repro_issue_one_ktile_wmma_ladder(
        hrx_q8_0_wmma_vk128_half8_vec * acc,
        const hrx_q8_0_wmma_vk128_half16_vec a_frag[4],
        const hrx_q8_0_wmma_vk128_half16_vec b_frag[4]) {
    Q8_REPRO_WAIT_WMMA(acc, 0, a_frag[0], b_frag[0], 12);
    Q8_REPRO_WAIT_WMMA(acc, 1, a_frag[1], b_frag[0], 12);
    Q8_REPRO_WAIT_WMMA(acc, 2, a_frag[2], b_frag[0], 12);
    Q8_REPRO_WAIT_WMMA(acc, 3, a_frag[3], b_frag[0], 12);
    Q8_REPRO_WAIT_WMMA(acc, 4, a_frag[0], b_frag[1], 8);
    Q8_REPRO_WAIT_WMMA(acc, 5, a_frag[1], b_frag[1], 8);
    Q8_REPRO_WAIT_WMMA(acc, 6, a_frag[2], b_frag[1], 8);
    Q8_REPRO_WAIT_WMMA(acc, 7, a_frag[3], b_frag[1], 8);
    Q8_REPRO_WAIT_WMMA(acc, 8, a_frag[0], b_frag[2], 4);
    Q8_REPRO_WAIT_WMMA(acc, 9, a_frag[1], b_frag[2], 4);
    Q8_REPRO_WAIT_WMMA(acc, 10, a_frag[2], b_frag[2], 4);
    Q8_REPRO_WAIT_WMMA(acc, 11, a_frag[3], b_frag[2], 4);
    Q8_REPRO_WAIT_WMMA(acc, 12, a_frag[0], b_frag[3], 0);
    Q8_REPRO_WAIT_WMMA(acc, 13, a_frag[1], b_frag[3], 0);
    Q8_REPRO_WAIT_WMMA(acc, 14, a_frag[2], b_frag[3], 0);
    Q8_REPRO_WAIT_WMMA(acc, 15, a_frag[3], b_frag[3], 0);
}

static __device__ __forceinline__ void q8_repro_issue_k2_wmma_ladder_ktilefrag(
        hrx_q8_0_wmma_vk128_half8_vec * acc,
        hrx_q8_0_wmma_vk128_lds_half_ptr sh_a_lds,
        hrx_q8_0_wmma_vk128_lds_half_ptr sh_b_lds,
        int wave_row,
        int wave_col,
        unsigned int lane) {
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
        q8_repro_issue_one_ktile_wmma_ladder(acc, a_frag, b_frag);
    }
}

template <int col_pair>
static __device__ __forceinline__ void q8_repro_issue_one_colpair_wmma_ladder(
        hrx_q8_0_wmma_vk128_half8_vec * acc,
        const hrx_q8_0_wmma_vk128_half16_vec a_frag[4],
        const hrx_q8_0_wmma_vk128_half16_vec b_frag[2]) {
    static_assert(col_pair == 0 || col_pair == 1, "colpair expects two 64-column pairs");
    constexpr int base = col_pair * 8;
    Q8_REPRO_WAIT_WMMA(acc, base + 0, a_frag[0], b_frag[0], 4);
    Q8_REPRO_WAIT_WMMA(acc, base + 1, a_frag[1], b_frag[0], 4);
    Q8_REPRO_WAIT_WMMA(acc, base + 2, a_frag[2], b_frag[0], 4);
    Q8_REPRO_WAIT_WMMA(acc, base + 3, a_frag[3], b_frag[0], 4);
    Q8_REPRO_WAIT_WMMA(acc, base + 4, a_frag[0], b_frag[1], 0);
    Q8_REPRO_WAIT_WMMA(acc, base + 5, a_frag[1], b_frag[1], 0);
    Q8_REPRO_WAIT_WMMA(acc, base + 6, a_frag[2], b_frag[1], 0);
    Q8_REPRO_WAIT_WMMA(acc, base + 7, a_frag[3], b_frag[1], 0);
}

static __device__ __forceinline__ void q8_repro_issue_k2_wmma_ladder_colpairfrag(
        hrx_q8_0_wmma_vk128_half8_vec * acc,
        hrx_q8_0_wmma_vk128_lds_half_ptr sh_a_lds,
        hrx_q8_0_wmma_vk128_lds_half_ptr sh_b_lds,
        int wave_row,
        int wave_col,
        unsigned int lane) {
#pragma unroll
    for (int k_tile = 0; k_tile < 2; ++k_tile) {
#pragma unroll
        for (int col_pair = 0; col_pair < 2; ++col_pair) {
            hrx_q8_0_wmma_vk128_half16_vec a_frag[4];
            hrx_q8_0_wmma_vk128_half16_vec b_frag[2];
#pragma unroll
            for (int row_sub = 0; row_sub < 4; ++row_sub) {
                a_frag[row_sub] =
                    hrx_q8_0_wmma_vk128_load_a_frag_w64_b64asm_nowait(
                        sh_a_lds, wave_row * 4 + row_sub, k_tile, lane);
            }
#pragma unroll
            for (int local_col = 0; local_col < 2; ++local_col) {
                b_frag[local_col] =
                    hrx_q8_0_wmma_vk128_load_b_frag_w64_b64asm_nowait(
                        sh_b_lds, wave_col * 4 + col_pair * 2 + local_col, k_tile, lane);
            }
            if (col_pair == 0) {
                q8_repro_issue_one_colpair_wmma_ladder<0>(acc, a_frag, b_frag);
            } else {
                q8_repro_issue_one_colpair_wmma_ladder<1>(acc, a_frag, b_frag);
            }
        }
    }
}

static __device__ __forceinline__ void q8_repro_prefetch_k2_fragments_for_wait51(
        hrx_q8_0_wmma_vk128_lds_half_ptr sh_a_lds,
        hrx_q8_0_wmma_vk128_lds_half_ptr sh_b_lds,
        int wave_row,
        int wave_col,
        unsigned int lane) {
#pragma unroll
    for (int k_tile = 0; k_tile < 2; ++k_tile) {
#pragma unroll
        for (int row_sub = 0; row_sub < 4; ++row_sub) {
            const hrx_q8_0_wmma_vk128_half16_vec frag =
                hrx_q8_0_wmma_vk128_load_a_frag_w64_b64asm_nowait(
                    sh_a_lds, wave_row * 4 + row_sub, k_tile, lane);
            asm volatile("" :: "v"(frag[0]), "v"(frag[4]), "v"(frag[8]), "v"(frag[12]) : "memory");
        }
#pragma unroll
        for (int col_sub = 0; col_sub < 4; ++col_sub) {
            const hrx_q8_0_wmma_vk128_half16_vec frag =
                hrx_q8_0_wmma_vk128_load_b_frag_w64_b64asm_nowait(
                    sh_b_lds, wave_col * 4 + col_sub, k_tile, lane);
            asm volatile("" :: "v"(frag[0]), "v"(frag[4]), "v"(frag[8]), "v"(frag[12]) : "memory");
        }
    }
}

template <bool dependent_ladder>
__global__ __launch_bounds__(256, 1)
void q8_motif192_wmma_k2_store_kernel(
        float * dst,
        long long rows,
        long long cols,
        bool wait_after_load) {
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 32;
    constexpr int SHARED_STRIDE = HRX_Q8_0_WMMA_VK128_SHARED_STRIDE;

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
    __shared__ _Float16 sh_store[4 * 16 * 16];
    hrx_q8_0_wmma_vk128_half8_vec acc[16] = {};

    for (int idx = static_cast<int>(tid); idx < BM * BK; idx += 256) {
        const int r = idx / BK;
        const int kk = idx - r * BK;
        const int row_sub = (r & 63) >> 4;
        sh_a[r * SHARED_STRIDE + kk] = static_cast<_Float16>(1.0f + static_cast<float>(row_sub));
    }
    for (int idx = static_cast<int>(tid); idx < BN * BK; idx += 256) {
        const int c = idx / BK;
        const int kk = idx - c * BK;
        const int col_sub = (c & 63) >> 4;
        sh_b[c * SHARED_STRIDE + kk] = static_cast<_Float16>(1.0f + static_cast<float>(col_sub));
    }
    __syncthreads();

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
                    sh_a_lds, wave_row * 4 + row_sub, k_tile, lane);
        }
#pragma unroll
        for (int col_sub = 0; col_sub < 4; ++col_sub) {
            b_frag[k_tile][col_sub] =
                hrx_q8_0_wmma_vk128_load_b_frag_w64_b64asm_nowait(
                    sh_b_lds, wave_col * 4 + col_sub, k_tile, lane);
        }
    }

    q8_repro_issue_k2_wmma_ladder<dependent_ladder>(acc, a_frag, b_frag);

#pragma unroll
    for (int group = 0; group < 16; ++group) {
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            q8_repro_bm128_raw_store_slot(
                dst_rsrc, rows, row_base, col_base, rows, cols, acc, wave, group, slot, lane);
        }
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            q8_repro_motif192_stage_store_slot(sh_store, acc, wave, group + 16, slot, lane);
        }
        asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            q8_repro_motif192_stage_load_store_slot(
                dst_rsrc, rows, row_base, col_base, rows, cols, sh_store, wave, group + 16, slot, lane, wait_after_load);
        }
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            q8_repro_motif192_stage_store_slot(sh_store, acc, wave, group + 32, slot, lane);
        }
        asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            q8_repro_motif192_stage_load_store_slot(
                dst_rsrc, rows, row_base, col_base, rows, cols, sh_store, wave, group + 32, slot, lane, wait_after_load);
        }
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();
}

template <bool dependent_ladder>
__global__ __launch_bounds__(256, 1)
void q8_motif192_wmma_k2_realdata_k32_store_kernel(
        const hrx_block_q8_0_wmma_vk128_lhs * src0,
        const float * src1,
        float * dst,
        long long k,
        long long rows,
        long long cols,
        bool wait_after_load) {
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 32;
    constexpr int SHARED_STRIDE = HRX_Q8_0_WMMA_VK128_SHARED_STRIDE;

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

    const long long blocks_per_row = k >> 5;
    const __amdgpu_buffer_rsrc_t dst_rsrc = hrx_q8_0_wmma_vk128_make_dst_rsrc(dst);
    __shared__ _Float16 sh_a[BM * SHARED_STRIDE];
    __shared__ _Float16 sh_b[BN * SHARED_STRIDE];
    __shared__ _Float16 sh_store[4 * 16 * 16];
    hrx_q8_0_wmma_vk128_half8_vec acc[16] = {};

    for (int idx = static_cast<int>(tid); idx < BM * BK; idx += 256) {
        const int r = idx / BK;
        const int kk = idx - r * BK;
        const long long row = row_base + r;
        sh_a[r * SHARED_STRIDE + kk] =
            row < rows ? hrx_q8_0_wmma_vk128_load_a_value(src0, row, kk, blocks_per_row) : static_cast<_Float16>(0.0f);
    }
    for (int idx = static_cast<int>(tid); idx < BN * BK; idx += 256) {
        const int c = idx / BK;
        const int kk = idx - c * BK;
        const long long col = col_base + c;
        sh_b[c * SHARED_STRIDE + kk] =
            col < cols ? static_cast<_Float16>(src1[col * k + kk]) : static_cast<_Float16>(0.0f);
    }
    __syncthreads();

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
                    sh_a_lds, wave_row * 4 + row_sub, k_tile, lane);
        }
#pragma unroll
        for (int col_sub = 0; col_sub < 4; ++col_sub) {
            b_frag[k_tile][col_sub] =
                hrx_q8_0_wmma_vk128_load_b_frag_w64_b64asm_nowait(
                    sh_b_lds, wave_col * 4 + col_sub, k_tile, lane);
        }
    }

    q8_repro_issue_k2_wmma_ladder<dependent_ladder>(acc, a_frag, b_frag);

#pragma unroll
    for (int group = 0; group < 16; ++group) {
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            q8_repro_bm128_raw_store_slot(
                dst_rsrc, rows, row_base, col_base, rows, cols, acc, wave, group, slot, lane);
        }
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            q8_repro_motif192_stage_store_slot(sh_store, acc, wave, group + 16, slot, lane);
        }
        asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            q8_repro_motif192_stage_load_store_slot(
                dst_rsrc, rows, row_base, col_base, rows, cols, sh_store, wave, group + 16, slot, lane, wait_after_load);
        }
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            q8_repro_motif192_stage_store_slot(sh_store, acc, wave, group + 32, slot, lane);
        }
        asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            q8_repro_motif192_stage_load_store_slot(
                dst_rsrc, rows, row_base, col_base, rows, cols, sh_store, wave, group + 32, slot, lane, wait_after_load);
        }
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();
}

template <bool dependent_ladder>
__global__ __launch_bounds__(256, 1)
void q8_motif192_wmma_k2_realdata_fullk_store_kernel(
        const hrx_block_q8_0_wmma_vk128_lhs * src0,
        const float * src1,
        float * dst,
        long long k,
        long long rows,
        long long cols,
        bool wait_after_load) {
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 32;
    constexpr int SHARED_STRIDE = HRX_Q8_0_WMMA_VK128_SHARED_STRIDE;

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

    const long long blocks_per_row = k >> 5;
    const __amdgpu_buffer_rsrc_t dst_rsrc = hrx_q8_0_wmma_vk128_make_dst_rsrc(dst);
    __shared__ _Float16 sh_a[BM * SHARED_STRIDE];
    __shared__ _Float16 sh_b[BN * SHARED_STRIDE];
    __shared__ _Float16 sh_store[4 * 16 * 16];
    hrx_q8_0_wmma_vk128_half8_vec acc[16] = {};

    for (long long k0 = 0; k0 < k; k0 += BK) {
        for (int idx = static_cast<int>(tid); idx < BM * BK; idx += 256) {
            const int r = idx / BK;
            const int kk = idx - r * BK;
            const long long row = row_base + r;
            sh_a[r * SHARED_STRIDE + kk] =
                row < rows ? hrx_q8_0_wmma_vk128_load_a_value(src0, row, k0 + kk, blocks_per_row) : static_cast<_Float16>(0.0f);
        }
        for (int idx = static_cast<int>(tid); idx < BN * BK; idx += 256) {
            const int c = idx / BK;
            const int kk = idx - c * BK;
            const long long col = col_base + c;
            sh_b[c * SHARED_STRIDE + kk] =
                col < cols ? static_cast<_Float16>(src1[col * k + k0 + kk]) : static_cast<_Float16>(0.0f);
        }
        __syncthreads();

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
                        sh_a_lds, wave_row * 4 + row_sub, k_tile, lane);
            }
#pragma unroll
            for (int col_sub = 0; col_sub < 4; ++col_sub) {
                b_frag[k_tile][col_sub] =
                    hrx_q8_0_wmma_vk128_load_b_frag_w64_b64asm_nowait(
                        sh_b_lds, wave_col * 4 + col_sub, k_tile, lane);
            }
        }

        q8_repro_issue_k2_wmma_ladder<dependent_ladder>(acc, a_frag, b_frag);
        __syncthreads();
    }

#pragma unroll
    for (int group = 0; group < 16; ++group) {
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            q8_repro_bm128_raw_store_slot(
                dst_rsrc, rows, row_base, col_base, rows, cols, acc, wave, group, slot, lane);
        }
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            q8_repro_motif192_stage_store_slot(sh_store, acc, wave, group + 16, slot, lane);
        }
        asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            q8_repro_motif192_stage_load_store_slot(
                dst_rsrc, rows, row_base, col_base, rows, cols, sh_store, wave, group + 16, slot, lane, wait_after_load);
        }
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            q8_repro_motif192_stage_store_slot(sh_store, acc, wave, group + 32, slot, lane);
        }
        asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            q8_repro_motif192_stage_load_store_slot(
                dst_rsrc, rows, row_base, col_base, rows, cols, sh_store, wave, group + 32, slot, lane, wait_after_load);
        }
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();
}

template <int group_base, int col_start>
__global__ __launch_bounds__(256, 1)
void q8_motif192_wmma_k2_realdata_fullk_phase8_store_kernel(
        const hrx_block_q8_0_wmma_vk128_lhs * src0,
        const float * src1,
        float * dst,
        long long k,
        long long rows,
        long long cols,
        bool wait_after_load) {
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 32;
    constexpr int SHARED_STRIDE = HRX_Q8_0_WMMA_VK128_SHARED_STRIDE;
    static_assert((group_base == 0 && col_start == 0) || (group_base == 8 && col_start == 2),
        "phase8 expects lower or upper half of the 128x128 tile");

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

    const long long blocks_per_row = k >> 5;
    const __amdgpu_buffer_rsrc_t dst_rsrc = hrx_q8_0_wmma_vk128_make_dst_rsrc(dst);
    __shared__ _Float16 sh_a[BM * SHARED_STRIDE];
    __shared__ _Float16 sh_b[BN * SHARED_STRIDE];
    __shared__ _Float16 sh_store[4 * 16 * 16];
    hrx_q8_0_wmma_vk128_half8_vec acc[8] = {};

    for (long long k0 = 0; k0 < k; k0 += BK) {
        for (int idx = static_cast<int>(tid); idx < BM * BK; idx += 256) {
            const int r = idx / BK;
            const int kk = idx - r * BK;
            const long long row = row_base + r;
            sh_a[r * SHARED_STRIDE + kk] =
                row < rows ? hrx_q8_0_wmma_vk128_load_a_value(src0, row, k0 + kk, blocks_per_row) : static_cast<_Float16>(0.0f);
        }
        for (int idx = static_cast<int>(tid); idx < BN * BK; idx += 256) {
            const int c = idx / BK;
            const int kk = idx - c * BK;
            const long long col = col_base + c;
            sh_b[c * SHARED_STRIDE + kk] =
                col < cols ? static_cast<_Float16>(src1[col * k + k0 + kk]) : static_cast<_Float16>(0.0f);
        }
        __syncthreads();

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
                        sh_a_lds, wave_row * 4 + row_sub, k_tile, lane);
            }
#pragma unroll
            for (int col_sub = 0; col_sub < 4; ++col_sub) {
                b_frag[k_tile][col_sub] =
                    hrx_q8_0_wmma_vk128_load_b_frag_w64_b64asm_nowait(
                        sh_b_lds, wave_col * 4 + col_sub, k_tile, lane);
            }
        }

        q8_repro_issue_k2_wmma_ladder_phase8<col_start>(acc, a_frag, b_frag);
        __syncthreads();
    }

#pragma unroll
    for (int local = 0; local < 8; ++local) {
        const int group = group_base + local;
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            q8_repro_bm128_raw_store_slot(
                dst_rsrc, rows, row_base, col_base, rows, cols, acc, wave, group, slot, lane);
        }
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            q8_repro_motif192_phase8_stage_store_slot(sh_store, acc, wave, group + 16, slot, lane);
        }
        asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            q8_repro_motif192_stage_load_store_slot(
                dst_rsrc, rows, row_base, col_base, rows, cols, sh_store, wave, group + 16, slot, lane, wait_after_load);
        }
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            q8_repro_motif192_phase8_stage_store_slot(sh_store, acc, wave, group + 32, slot, lane);
        }
        asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            q8_repro_motif192_stage_load_store_slot(
                dst_rsrc, rows, row_base, col_base, rows, cols, sh_store, wave, group + 32, slot, lane, wait_after_load);
        }
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();
}

__global__ __launch_bounds__(256, 1)
void q8_motif192_wmma_k2_realdata_fullk_streamfrag_store_kernel(
        const hrx_block_q8_0_wmma_vk128_lhs * src0,
        const float * src1,
        float * dst,
        long long k,
        long long rows,
        long long cols,
        bool wait_after_load) {
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 32;
    constexpr int SHARED_STRIDE = HRX_Q8_0_WMMA_VK128_SHARED_STRIDE;

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

    const long long blocks_per_row = k >> 5;
    const __amdgpu_buffer_rsrc_t dst_rsrc = hrx_q8_0_wmma_vk128_make_dst_rsrc(dst);
    __shared__ _Float16 sh_a[BM * SHARED_STRIDE];
    __shared__ _Float16 sh_b[BN * SHARED_STRIDE];
    __shared__ _Float16 sh_store[4 * 16 * 16];
    hrx_q8_0_wmma_vk128_half8_vec acc[16] = {};

    for (long long k0 = 0; k0 < k; k0 += BK) {
        for (int idx = static_cast<int>(tid); idx < BM * BK; idx += 256) {
            const int r = idx / BK;
            const int kk = idx - r * BK;
            const long long row = row_base + r;
            sh_a[r * SHARED_STRIDE + kk] =
                row < rows ? hrx_q8_0_wmma_vk128_load_a_value(src0, row, k0 + kk, blocks_per_row) : static_cast<_Float16>(0.0f);
        }
        for (int idx = static_cast<int>(tid); idx < BN * BK; idx += 256) {
            const int c = idx / BK;
            const int kk = idx - c * BK;
            const long long col = col_base + c;
            sh_b[c * SHARED_STRIDE + kk] =
                col < cols ? static_cast<_Float16>(src1[col * k + k0 + kk]) : static_cast<_Float16>(0.0f);
        }
        __syncthreads();

        hrx_q8_0_wmma_vk128_lds_half_ptr sh_a_lds =
            (hrx_q8_0_wmma_vk128_lds_half_ptr) sh_a;
        hrx_q8_0_wmma_vk128_lds_half_ptr sh_b_lds =
            (hrx_q8_0_wmma_vk128_lds_half_ptr) sh_b;
        q8_repro_issue_k2_wmma_ladder_streamfrag(acc, sh_a_lds, sh_b_lds, wave_row, wave_col, lane);
        __syncthreads();
    }

#pragma unroll
    for (int group = 0; group < 16; ++group) {
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            q8_repro_bm128_raw_store_slot(
                dst_rsrc, rows, row_base, col_base, rows, cols, acc, wave, group, slot, lane);
        }
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            q8_repro_motif192_stage_store_slot(sh_store, acc, wave, group + 16, slot, lane);
        }
        asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            q8_repro_motif192_stage_load_store_slot(
                dst_rsrc, rows, row_base, col_base, rows, cols, sh_store, wave, group + 16, slot, lane, wait_after_load);
        }
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            q8_repro_motif192_stage_store_slot(sh_store, acc, wave, group + 32, slot, lane);
        }
        asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            q8_repro_motif192_stage_load_store_slot(
                dst_rsrc, rows, row_base, col_base, rows, cols, sh_store, wave, group + 32, slot, lane, wait_after_load);
        }
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();
}

__global__ __launch_bounds__(256, 1)
void q8_motif192_wmma_k2_realdata_fullk_ktilefrag_store_kernel(
        const hrx_block_q8_0_wmma_vk128_lhs * src0,
        const float * src1,
        float * dst,
        long long k,
        long long rows,
        long long cols,
        bool wait_after_load) {
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 32;
    constexpr int SHARED_STRIDE = HRX_Q8_0_WMMA_VK128_SHARED_STRIDE;

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

    const long long blocks_per_row = k >> 5;
    const __amdgpu_buffer_rsrc_t dst_rsrc = hrx_q8_0_wmma_vk128_make_dst_rsrc(dst);
    __shared__ _Float16 sh_a[BM * SHARED_STRIDE];
    __shared__ _Float16 sh_b[BN * SHARED_STRIDE];
    __shared__ _Float16 sh_store[4 * 16 * 16];
    hrx_q8_0_wmma_vk128_half8_vec acc[16] = {};

    for (long long k0 = 0; k0 < k; k0 += BK) {
        for (int idx = static_cast<int>(tid); idx < BM * BK; idx += 256) {
            const int r = idx / BK;
            const int kk = idx - r * BK;
            const long long row = row_base + r;
            sh_a[r * SHARED_STRIDE + kk] =
                row < rows ? hrx_q8_0_wmma_vk128_load_a_value(src0, row, k0 + kk, blocks_per_row) : static_cast<_Float16>(0.0f);
        }
        for (int idx = static_cast<int>(tid); idx < BN * BK; idx += 256) {
            const int c = idx / BK;
            const int kk = idx - c * BK;
            const long long col = col_base + c;
            sh_b[c * SHARED_STRIDE + kk] =
                col < cols ? static_cast<_Float16>(src1[col * k + k0 + kk]) : static_cast<_Float16>(0.0f);
        }
        __syncthreads();

        hrx_q8_0_wmma_vk128_lds_half_ptr sh_a_lds =
            (hrx_q8_0_wmma_vk128_lds_half_ptr) sh_a;
        hrx_q8_0_wmma_vk128_lds_half_ptr sh_b_lds =
            (hrx_q8_0_wmma_vk128_lds_half_ptr) sh_b;
        q8_repro_issue_k2_wmma_ladder_ktilefrag(acc, sh_a_lds, sh_b_lds, wave_row, wave_col, lane);
        __syncthreads();
    }

#pragma unroll
    for (int group = 0; group < 16; ++group) {
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            q8_repro_bm128_raw_store_slot(
                dst_rsrc, rows, row_base, col_base, rows, cols, acc, wave, group, slot, lane);
        }
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            q8_repro_motif192_stage_store_slot(sh_store, acc, wave, group + 16, slot, lane);
        }
        asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            q8_repro_motif192_stage_load_store_slot(
                dst_rsrc, rows, row_base, col_base, rows, cols, sh_store, wave, group + 16, slot, lane, wait_after_load);
        }
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            q8_repro_motif192_stage_store_slot(sh_store, acc, wave, group + 32, slot, lane);
        }
        asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            q8_repro_motif192_stage_load_store_slot(
                dst_rsrc, rows, row_base, col_base, rows, cols, sh_store, wave, group + 32, slot, lane, wait_after_load);
        }
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();
}

__global__ __launch_bounds__(256, 1)
void q8_motif192_wmma_k2_realdata_fullk_ktilefrag_storebatch_kernel(
        const hrx_block_q8_0_wmma_vk128_lhs * src0,
        const float * src1,
        float * dst,
        long long k,
        long long rows,
        long long cols,
        bool wait_after_load) {
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 32;
    constexpr int SHARED_STRIDE = HRX_Q8_0_WMMA_VK128_SHARED_STRIDE;

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

    const long long blocks_per_row = k >> 5;
    const __amdgpu_buffer_rsrc_t dst_rsrc = hrx_q8_0_wmma_vk128_make_dst_rsrc(dst);
    __shared__ _Float16 sh_a[BM * SHARED_STRIDE];
    __shared__ _Float16 sh_b[BN * SHARED_STRIDE];
    __shared__ _Float16 sh_store[16 * 4 * 16 * 16];
    hrx_q8_0_wmma_vk128_half8_vec acc[16] = {};

    for (long long k0 = 0; k0 < k; k0 += BK) {
        for (int idx = static_cast<int>(tid); idx < BM * BK; idx += 256) {
            const int r = idx / BK;
            const int kk = idx - r * BK;
            const long long row = row_base + r;
            sh_a[r * SHARED_STRIDE + kk] =
                row < rows ? hrx_q8_0_wmma_vk128_load_a_value(src0, row, k0 + kk, blocks_per_row) : static_cast<_Float16>(0.0f);
        }
        for (int idx = static_cast<int>(tid); idx < BN * BK; idx += 256) {
            const int c = idx / BK;
            const int kk = idx - c * BK;
            const long long col = col_base + c;
            sh_b[c * SHARED_STRIDE + kk] =
                col < cols ? static_cast<_Float16>(src1[col * k + k0 + kk]) : static_cast<_Float16>(0.0f);
        }
        __syncthreads();

        hrx_q8_0_wmma_vk128_lds_half_ptr sh_a_lds =
            (hrx_q8_0_wmma_vk128_lds_half_ptr) sh_a;
        hrx_q8_0_wmma_vk128_lds_half_ptr sh_b_lds =
            (hrx_q8_0_wmma_vk128_lds_half_ptr) sh_b;
        q8_repro_issue_k2_wmma_ladder_ktilefrag(acc, sh_a_lds, sh_b_lds, wave_row, wave_col, lane);
        __syncthreads();
    }

#pragma unroll
    for (int group = 0; group < 16; ++group) {
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            q8_repro_bm128_raw_store_slot(
                dst_rsrc, rows, row_base, col_base, rows, cols, acc, wave, group, slot, lane);
        }
    }

#pragma unroll
    for (int group = 0; group < 16; ++group) {
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            q8_repro_motif192_grouped_stage_store_slot(sh_store, acc, wave, group + 16, slot, lane);
        }
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
#pragma unroll
    for (int group = 0; group < 16; ++group) {
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            q8_repro_motif192_grouped_stage_load_store_slot(
                dst_rsrc, rows, row_base, col_base, rows, cols, sh_store, wave, group + 16, slot, lane, wait_after_load);
        }
    }

#pragma unroll
    for (int group = 0; group < 16; ++group) {
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            q8_repro_motif192_grouped_stage_store_slot(sh_store, acc, wave, group + 32, slot, lane);
        }
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
#pragma unroll
    for (int group = 0; group < 16; ++group) {
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            q8_repro_motif192_grouped_stage_load_store_slot(
                dst_rsrc, rows, row_base, col_base, rows, cols, sh_store, wave, group + 32, slot, lane, wait_after_load);
        }
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();
}

__global__ __launch_bounds__(256, 1)
void q8_motif192_wmma_k2_realdata_fullk_ktilefrag_storebatch4_kernel(
        const hrx_block_q8_0_wmma_vk128_lhs * src0,
        const float * src1,
        float * dst,
        long long k,
        long long rows,
        long long cols,
        bool wait_after_load) {
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 32;
    constexpr int SHARED_STRIDE = HRX_Q8_0_WMMA_VK128_SHARED_STRIDE;

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

    const long long blocks_per_row = k >> 5;
    const __amdgpu_buffer_rsrc_t dst_rsrc = hrx_q8_0_wmma_vk128_make_dst_rsrc(dst);
    __shared__ _Float16 sh_a[BM * SHARED_STRIDE];
    __shared__ _Float16 sh_b[BN * SHARED_STRIDE];
    __shared__ _Float16 sh_store[4 * 4 * 16 * 16];
    hrx_q8_0_wmma_vk128_half8_vec acc[16] = {};

    for (long long k0 = 0; k0 < k; k0 += BK) {
        for (int idx = static_cast<int>(tid); idx < BM * BK; idx += 256) {
            const int r = idx / BK;
            const int kk = idx - r * BK;
            const long long row = row_base + r;
            sh_a[r * SHARED_STRIDE + kk] =
                row < rows ? hrx_q8_0_wmma_vk128_load_a_value(src0, row, k0 + kk, blocks_per_row) : static_cast<_Float16>(0.0f);
        }
        for (int idx = static_cast<int>(tid); idx < BN * BK; idx += 256) {
            const int c = idx / BK;
            const int kk = idx - c * BK;
            const long long col = col_base + c;
            sh_b[c * SHARED_STRIDE + kk] =
                col < cols ? static_cast<_Float16>(src1[col * k + k0 + kk]) : static_cast<_Float16>(0.0f);
        }
        __syncthreads();

        hrx_q8_0_wmma_vk128_lds_half_ptr sh_a_lds =
            (hrx_q8_0_wmma_vk128_lds_half_ptr) sh_a;
        hrx_q8_0_wmma_vk128_lds_half_ptr sh_b_lds =
            (hrx_q8_0_wmma_vk128_lds_half_ptr) sh_b;
        q8_repro_issue_k2_wmma_ladder_ktilefrag(acc, sh_a_lds, sh_b_lds, wave_row, wave_col, lane);
        __syncthreads();
    }

#pragma unroll
    for (int group = 0; group < 16; ++group) {
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            q8_repro_bm128_raw_store_slot(
                dst_rsrc, rows, row_base, col_base, rows, cols, acc, wave, group, slot, lane);
        }
    }

#pragma unroll
    for (int group_base = 0; group_base < 16; group_base += 4) {
#pragma unroll
        for (int local = 0; local < 4; ++local) {
            const int group = group_base + local + 16;
#pragma unroll
            for (int slot = 0; slot < 4; ++slot) {
                q8_repro_motif192_group4_stage_store_slot(sh_store, acc, wave, group, slot, lane);
            }
        }
        asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
#pragma unroll
        for (int local = 0; local < 4; ++local) {
            const int group = group_base + local + 16;
#pragma unroll
            for (int slot = 0; slot < 4; ++slot) {
                q8_repro_motif192_group4_stage_load_store_slot(
                    dst_rsrc, rows, row_base, col_base, rows, cols, sh_store, wave, group, slot, lane, wait_after_load);
            }
        }
    }

#pragma unroll
    for (int group_base = 0; group_base < 16; group_base += 4) {
#pragma unroll
        for (int local = 0; local < 4; ++local) {
            const int group = group_base + local + 32;
#pragma unroll
            for (int slot = 0; slot < 4; ++slot) {
                q8_repro_motif192_group4_stage_store_slot(sh_store, acc, wave, group, slot, lane);
            }
        }
        asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
#pragma unroll
        for (int local = 0; local < 4; ++local) {
            const int group = group_base + local + 32;
#pragma unroll
            for (int slot = 0; slot < 4; ++slot) {
                q8_repro_motif192_group4_stage_load_store_slot(
                    dst_rsrc, rows, row_base, col_base, rows, cols, sh_store, wave, group, slot, lane, wait_after_load);
            }
        }
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();
}

__global__ __launch_bounds__(256, 1)
void q8_motif192_wmma_k2_realdata_fullk_colpairfrag_store_kernel(
        const hrx_block_q8_0_wmma_vk128_lhs * src0,
        const float * src1,
        float * dst,
        long long k,
        long long rows,
        long long cols,
        bool wait_after_load) {
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 32;
    constexpr int SHARED_STRIDE = HRX_Q8_0_WMMA_VK128_SHARED_STRIDE;

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

    const long long blocks_per_row = k >> 5;
    const __amdgpu_buffer_rsrc_t dst_rsrc = hrx_q8_0_wmma_vk128_make_dst_rsrc(dst);
    __shared__ _Float16 sh_a[BM * SHARED_STRIDE];
    __shared__ _Float16 sh_b[BN * SHARED_STRIDE];
    __shared__ _Float16 sh_store[4 * 16 * 16];
    hrx_q8_0_wmma_vk128_half8_vec acc[16] = {};

    for (long long k0 = 0; k0 < k; k0 += BK) {
        for (int idx = static_cast<int>(tid); idx < BM * BK; idx += 256) {
            const int r = idx / BK;
            const int kk = idx - r * BK;
            const long long row = row_base + r;
            sh_a[r * SHARED_STRIDE + kk] =
                row < rows ? hrx_q8_0_wmma_vk128_load_a_value(src0, row, k0 + kk, blocks_per_row) : static_cast<_Float16>(0.0f);
        }
        for (int idx = static_cast<int>(tid); idx < BN * BK; idx += 256) {
            const int c = idx / BK;
            const int kk = idx - c * BK;
            const long long col = col_base + c;
            sh_b[c * SHARED_STRIDE + kk] =
                col < cols ? static_cast<_Float16>(src1[col * k + k0 + kk]) : static_cast<_Float16>(0.0f);
        }
        __syncthreads();

        hrx_q8_0_wmma_vk128_lds_half_ptr sh_a_lds =
            (hrx_q8_0_wmma_vk128_lds_half_ptr) sh_a;
        hrx_q8_0_wmma_vk128_lds_half_ptr sh_b_lds =
            (hrx_q8_0_wmma_vk128_lds_half_ptr) sh_b;
        q8_repro_issue_k2_wmma_ladder_colpairfrag(acc, sh_a_lds, sh_b_lds, wave_row, wave_col, lane);
        __syncthreads();
    }

#pragma unroll
    for (int group = 0; group < 16; ++group) {
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            q8_repro_bm128_raw_store_slot(
                dst_rsrc, rows, row_base, col_base, rows, cols, acc, wave, group, slot, lane);
        }
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            q8_repro_motif192_stage_store_slot(sh_store, acc, wave, group + 16, slot, lane);
        }
        asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            q8_repro_motif192_stage_load_store_slot(
                dst_rsrc, rows, row_base, col_base, rows, cols, sh_store, wave, group + 16, slot, lane, wait_after_load);
        }
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            q8_repro_motif192_stage_store_slot(sh_store, acc, wave, group + 32, slot, lane);
        }
        asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            q8_repro_motif192_stage_load_store_slot(
                dst_rsrc, rows, row_base, col_base, rows, cols, sh_store, wave, group + 32, slot, lane, wait_after_load);
        }
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();
}

template <int group_base, int col_start>
static __device__ __forceinline__ void q8_motif192_wmma_k2_realdata_fullk_phase8_seq_phase(
        const hrx_block_q8_0_wmma_vk128_lhs * src0,
        const float * src1,
        __amdgpu_buffer_rsrc_t dst_rsrc,
        _Float16 * sh_a,
        _Float16 * sh_b,
        _Float16 * sh_store,
        long long k,
        long long rows,
        long long cols,
        long long blocks_per_row,
        long long row_base,
        long long col_base,
        unsigned int tid,
        unsigned int wave,
        unsigned int lane,
        int wave_row,
        int wave_col,
        bool wait_after_load) {
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 32;
    constexpr int SHARED_STRIDE = HRX_Q8_0_WMMA_VK128_SHARED_STRIDE;
    static_assert((group_base == 0 && col_start == 0) || (group_base == 8 && col_start == 2),
        "phase8seq expects lower or upper half of the 128x128 tile");

    hrx_q8_0_wmma_vk128_half8_vec acc[8] = {};

    for (long long k0 = 0; k0 < k; k0 += BK) {
        for (int idx = static_cast<int>(tid); idx < BM * BK; idx += 256) {
            const int r = idx / BK;
            const int kk = idx - r * BK;
            const long long row = row_base + r;
            sh_a[r * SHARED_STRIDE + kk] =
                row < rows ? hrx_q8_0_wmma_vk128_load_a_value(src0, row, k0 + kk, blocks_per_row) : static_cast<_Float16>(0.0f);
        }
        for (int idx = static_cast<int>(tid); idx < BN * BK; idx += 256) {
            const int c = idx / BK;
            const int kk = idx - c * BK;
            const long long col = col_base + c;
            sh_b[c * SHARED_STRIDE + kk] =
                col < cols ? static_cast<_Float16>(src1[col * k + k0 + kk]) : static_cast<_Float16>(0.0f);
        }
        __syncthreads();

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
                        sh_a_lds, wave_row * 4 + row_sub, k_tile, lane);
            }
#pragma unroll
            for (int col_sub = 0; col_sub < 4; ++col_sub) {
                b_frag[k_tile][col_sub] =
                    hrx_q8_0_wmma_vk128_load_b_frag_w64_b64asm_nowait(
                        sh_b_lds, wave_col * 4 + col_sub, k_tile, lane);
            }
        }

        q8_repro_issue_k2_wmma_ladder_phase8<col_start>(acc, a_frag, b_frag);
        __syncthreads();
    }

#pragma unroll
    for (int local = 0; local < 8; ++local) {
        const int group = group_base + local;
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            q8_repro_bm128_raw_store_slot(
                dst_rsrc, rows, row_base, col_base, rows, cols, acc, wave, group, slot, lane);
        }
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            q8_repro_motif192_phase8_stage_store_slot(sh_store, acc, wave, group + 16, slot, lane);
        }
        asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            q8_repro_motif192_stage_load_store_slot(
                dst_rsrc, rows, row_base, col_base, rows, cols, sh_store, wave, group + 16, slot, lane, wait_after_load);
        }
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            q8_repro_motif192_phase8_stage_store_slot(sh_store, acc, wave, group + 32, slot, lane);
        }
        asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            q8_repro_motif192_stage_load_store_slot(
                dst_rsrc, rows, row_base, col_base, rows, cols, sh_store, wave, group + 32, slot, lane, wait_after_load);
        }
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();
}

__global__ __launch_bounds__(256, 1)
void q8_motif192_wmma_k2_realdata_fullk_phase8seq_store_kernel(
        const hrx_block_q8_0_wmma_vk128_lhs * src0,
        const float * src1,
        float * dst,
        long long k,
        long long rows,
        long long cols,
        bool wait_after_load) {
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 32;
    constexpr int SHARED_STRIDE = HRX_Q8_0_WMMA_VK128_SHARED_STRIDE;
    (void) BK;

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

    const long long blocks_per_row = k >> 5;
    const __amdgpu_buffer_rsrc_t dst_rsrc = hrx_q8_0_wmma_vk128_make_dst_rsrc(dst);
    __shared__ _Float16 sh_a[BM * SHARED_STRIDE];
    __shared__ _Float16 sh_b[BN * SHARED_STRIDE];
    __shared__ _Float16 sh_store[4 * 16 * 16];

    q8_motif192_wmma_k2_realdata_fullk_phase8_seq_phase<0, 0>(
        src0, src1, dst_rsrc, sh_a, sh_b, sh_store, k, rows, cols, blocks_per_row,
        row_base, col_base, tid, wave, lane, wave_row, wave_col, wait_after_load);
    q8_motif192_wmma_k2_realdata_fullk_phase8_seq_phase<8, 2>(
        src0, src1, dst_rsrc, sh_a, sh_b, sh_store, k, rows, cols, blocks_per_row,
        row_base, col_base, tid, wave, lane, wave_row, wave_col, wait_after_load);
}

template <int group_base, int col_start>
static __device__ __forceinline__ void q8_motif192_wmma_k2_realdata_fullk_accpark_phase(
        const hrx_q8_0_wmma_vk128_half16_vec a_frag[2][4],
        const hrx_q8_0_wmma_vk128_half16_vec b_frag[2][4],
        _Float16 * sh_acc,
        unsigned int wave,
        unsigned int lane,
        bool first_k) {
    static_assert((group_base == 0 && col_start == 0) || (group_base == 8 && col_start == 2),
        "accpark expects lower or upper half of the 128x128 tile");
    hrx_q8_0_wmma_vk128_half8_vec acc[8] = {};
    if (!first_k) {
#pragma unroll
        for (int local = 0; local < 8; ++local) {
            q8_repro_accpark_load_group(acc, sh_acc, wave, group_base, local, lane);
        }
        asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    }
    q8_repro_issue_k2_wmma_ladder_phase8<col_start>(acc, a_frag, b_frag);
#pragma unroll
    for (int local = 0; local < 8; ++local) {
        q8_repro_accpark_store_group(sh_acc, acc, wave, group_base, local, lane);
    }
}

__global__ __launch_bounds__(256, 1)
void q8_motif192_wmma_k2_realdata_fullk_accpark_store_kernel(
        const hrx_block_q8_0_wmma_vk128_lhs * src0,
        const float * src1,
        float * dst,
        long long k,
        long long rows,
        long long cols,
        bool wait_after_load) {
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 32;
    constexpr int SHARED_STRIDE = HRX_Q8_0_WMMA_VK128_SHARED_STRIDE;

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

    const long long blocks_per_row = k >> 5;
    const __amdgpu_buffer_rsrc_t dst_rsrc = hrx_q8_0_wmma_vk128_make_dst_rsrc(dst);
    __shared__ _Float16 sh_a[BM * SHARED_STRIDE];
    __shared__ _Float16 sh_b[BN * SHARED_STRIDE];
    __shared__ _Float16 sh_store[4 * 16 * 16];
    __shared__ _Float16 sh_acc[4 * 16 * 4 * 64];

    for (long long k0 = 0; k0 < k; k0 += BK) {
        for (int idx = static_cast<int>(tid); idx < BM * BK; idx += 256) {
            const int r = idx / BK;
            const int kk = idx - r * BK;
            const long long row = row_base + r;
            sh_a[r * SHARED_STRIDE + kk] =
                row < rows ? hrx_q8_0_wmma_vk128_load_a_value(src0, row, k0 + kk, blocks_per_row) : static_cast<_Float16>(0.0f);
        }
        for (int idx = static_cast<int>(tid); idx < BN * BK; idx += 256) {
            const int c = idx / BK;
            const int kk = idx - c * BK;
            const long long col = col_base + c;
            sh_b[c * SHARED_STRIDE + kk] =
                col < cols ? static_cast<_Float16>(src1[col * k + k0 + kk]) : static_cast<_Float16>(0.0f);
        }
        __syncthreads();

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
                        sh_a_lds, wave_row * 4 + row_sub, k_tile, lane);
            }
#pragma unroll
            for (int col_sub = 0; col_sub < 4; ++col_sub) {
                b_frag[k_tile][col_sub] =
                    hrx_q8_0_wmma_vk128_load_b_frag_w64_b64asm_nowait(
                        sh_b_lds, wave_col * 4 + col_sub, k_tile, lane);
            }
        }

        const bool first_k = k0 == 0;
        q8_motif192_wmma_k2_realdata_fullk_accpark_phase<0, 0>(a_frag, b_frag, sh_acc, wave, lane, first_k);
        asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
        q8_motif192_wmma_k2_realdata_fullk_accpark_phase<8, 2>(a_frag, b_frag, sh_acc, wave, lane, first_k);
        asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
        __syncthreads();
    }

#pragma unroll
    for (int group = 0; group < 16; ++group) {
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            const _Float16 value = q8_repro_accpark_load_slot(sh_acc, wave, group, slot, lane);
            asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
            const int group16 = group & 15;
            const int wave_row_store = static_cast<int>(wave & 1u);
            const int wave_col_store = static_cast<int>(wave >> 1);
            const int row_lane = static_cast<int>(lane >> 4);
            const int col_lane = static_cast<int>(lane & 15u);
            const long long row = row_base + static_cast<long long>(wave_row_store * 64 + (group16 & 3) * 16 + row_lane + slot * 4);
            const long long col = col_base + static_cast<long long>(wave_col_store * 64 + ((group16 >> 2) & 3) * 16 + col_lane);
            if (row < rows && col < cols) {
                hrx_q8_0_wmma_vk128_buffer_store_f32(dst_rsrc, col * rows + row, static_cast<float>(value));
            }
        }
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            const _Float16 value = q8_repro_accpark_load_slot(sh_acc, wave, group, slot, lane);
            hrx_q8_0_wmma_vk128_lds_u16_ptr sh_u16 =
                (hrx_q8_0_wmma_vk128_lds_u16_ptr) sh_store;
            hrx_q8_0_wmma_vk128_ds_store_u16(
                sh_u16 + q8_repro_motif192_stage_index(wave, group + 16, slot, lane),
                hrx_q8_0_wmma_vk128_f16_to_u16(value));
        }
        asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            q8_repro_motif192_stage_load_store_slot(
                dst_rsrc, rows, row_base, col_base, rows, cols, sh_store, wave, group + 16, slot, lane, wait_after_load);
        }
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            const _Float16 value = q8_repro_accpark_load_slot(sh_acc, wave, group, slot, lane);
            hrx_q8_0_wmma_vk128_lds_u16_ptr sh_u16 =
                (hrx_q8_0_wmma_vk128_lds_u16_ptr) sh_store;
            hrx_q8_0_wmma_vk128_ds_store_u16(
                sh_u16 + q8_repro_motif192_stage_index(wave, group + 32, slot, lane),
                hrx_q8_0_wmma_vk128_f16_to_u16(value));
        }
        asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            q8_repro_motif192_stage_load_store_slot(
                dst_rsrc, rows, row_base, col_base, rows, cols, sh_store, wave, group + 32, slot, lane, wait_after_load);
        }
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();
}

template <int group_base, int col_start>
static __device__ __forceinline__ void q8_motif192_wmma_k2_realdata_fullk_accparkfull8_phase(
        const hrx_q8_0_wmma_vk128_half16_vec a_frag[2][4],
        const hrx_q8_0_wmma_vk128_half16_vec b_frag[2][4],
        _Float16 * sh_acc,
        unsigned int wave,
        unsigned int lane,
        bool first_k) {
    static_assert(group_base == 0 && col_start == 0,
        "accparkfull8 parks only the lower half of the 128x128 tile");
    hrx_q8_0_wmma_vk128_half8_vec acc[8] = {};
    if (!first_k) {
#pragma unroll
        for (int local = 0; local < 8; ++local) {
            q8_repro_accpark_fullvec_load_group(acc, sh_acc, wave, local, lane);
        }
        asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    }
    q8_repro_issue_k2_wmma_ladder_phase8<col_start>(acc, a_frag, b_frag);
#pragma unroll
    for (int local = 0; local < 8; ++local) {
        q8_repro_accpark_fullvec_store_group(sh_acc, acc, wave, local, lane);
    }
}

__global__ __launch_bounds__(256, 1)
void q8_motif192_wmma_k2_realdata_fullk_accparkfull8_store_kernel(
        const hrx_block_q8_0_wmma_vk128_lhs * src0,
        const float * src1,
        float * dst,
        long long k,
        long long rows,
        long long cols,
        bool wait_after_load) {
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 32;
    constexpr int SHARED_STRIDE = HRX_Q8_0_WMMA_VK128_SHARED_STRIDE;

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

    const long long blocks_per_row = k >> 5;
    const __amdgpu_buffer_rsrc_t dst_rsrc = hrx_q8_0_wmma_vk128_make_dst_rsrc(dst);
    __shared__ _Float16 sh_a[BM * SHARED_STRIDE];
    __shared__ _Float16 sh_b[BN * SHARED_STRIDE];
    __shared__ _Float16 sh_store[4 * 16 * 16];
    __shared__ _Float16 sh_acc[4 * 8 * 8 * 64];
    hrx_q8_0_wmma_vk128_half8_vec acc_upper[8] = {};

    for (long long k0 = 0; k0 < k; k0 += BK) {
        for (int idx = static_cast<int>(tid); idx < BM * BK; idx += 256) {
            const int r = idx / BK;
            const int kk = idx - r * BK;
            const long long row = row_base + r;
            sh_a[r * SHARED_STRIDE + kk] =
                row < rows ? hrx_q8_0_wmma_vk128_load_a_value(src0, row, k0 + kk, blocks_per_row) : static_cast<_Float16>(0.0f);
        }
        for (int idx = static_cast<int>(tid); idx < BN * BK; idx += 256) {
            const int c = idx / BK;
            const int kk = idx - c * BK;
            const long long col = col_base + c;
            sh_b[c * SHARED_STRIDE + kk] =
                col < cols ? static_cast<_Float16>(src1[col * k + k0 + kk]) : static_cast<_Float16>(0.0f);
        }
        __syncthreads();

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
                        sh_a_lds, wave_row * 4 + row_sub, k_tile, lane);
            }
#pragma unroll
            for (int col_sub = 0; col_sub < 4; ++col_sub) {
                b_frag[k_tile][col_sub] =
                    hrx_q8_0_wmma_vk128_load_b_frag_w64_b64asm_nowait(
                        sh_b_lds, wave_col * 4 + col_sub, k_tile, lane);
            }
        }

        const bool first_k = k0 == 0;
        q8_motif192_wmma_k2_realdata_fullk_accparkfull8_phase<0, 0>(
            a_frag, b_frag, sh_acc, wave, lane, first_k);
        asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
        q8_repro_issue_k2_wmma_ladder_phase8<2>(acc_upper, a_frag, b_frag);
        asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
        __syncthreads();
    }

#pragma unroll
    for (int group = 0; group < 8; ++group) {
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            const _Float16 value = q8_repro_accpark_fullvec_load_elem(
                sh_acc, wave, group, slot * 2 + HRX_Q8_0_WMMA_VK128_W64_OPSEL, lane);
            asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
            q8_repro_raw_store_value(
                dst_rsrc,
                rows,
                row_base,
                col_base,
                rows,
                cols,
                group,
                slot,
                lane,
                static_cast<float>(value));
        }
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            const _Float16 value = q8_repro_accpark_fullvec_load_elem(
                sh_acc, wave, group, slot * 2 + HRX_Q8_0_WMMA_VK128_W64_OPSEL, lane);
            hrx_q8_0_wmma_vk128_lds_u16_ptr sh_u16 =
                (hrx_q8_0_wmma_vk128_lds_u16_ptr) sh_store;
            hrx_q8_0_wmma_vk128_ds_store_u16(
                sh_u16 + q8_repro_motif192_stage_index(wave, group + 16, slot, lane),
                hrx_q8_0_wmma_vk128_f16_to_u16(value));
        }
        asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            q8_repro_motif192_stage_load_store_slot(
                dst_rsrc, rows, row_base, col_base, rows, cols, sh_store, wave, group + 16, slot, lane, wait_after_load);
        }
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            const _Float16 value = q8_repro_accpark_fullvec_load_elem(
                sh_acc, wave, group, slot * 2 + HRX_Q8_0_WMMA_VK128_W64_OPSEL, lane);
            hrx_q8_0_wmma_vk128_lds_u16_ptr sh_u16 =
                (hrx_q8_0_wmma_vk128_lds_u16_ptr) sh_store;
            hrx_q8_0_wmma_vk128_ds_store_u16(
                sh_u16 + q8_repro_motif192_stage_index(wave, group + 32, slot, lane),
                hrx_q8_0_wmma_vk128_f16_to_u16(value));
        }
        asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            q8_repro_motif192_stage_load_store_slot(
                dst_rsrc, rows, row_base, col_base, rows, cols, sh_store, wave, group + 32, slot, lane, wait_after_load);
        }
    }

#pragma unroll
    for (int group = 8; group < 16; ++group) {
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            q8_repro_bm128_raw_store_slot(
                dst_rsrc, rows, row_base, col_base, rows, cols, acc_upper, wave, group, slot, lane);
        }
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            q8_repro_motif192_phase8_stage_store_slot(sh_store, acc_upper, wave, group + 16, slot, lane);
        }
        asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            q8_repro_motif192_stage_load_store_slot(
                dst_rsrc, rows, row_base, col_base, rows, cols, sh_store, wave, group + 16, slot, lane, wait_after_load);
        }
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            q8_repro_motif192_phase8_stage_store_slot(sh_store, acc_upper, wave, group + 32, slot, lane);
        }
        asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            q8_repro_motif192_stage_load_store_slot(
                dst_rsrc, rows, row_base, col_base, rows, cols, sh_store, wave, group + 32, slot, lane, wait_after_load);
        }
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();
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

template <bool copy_a, bool copy_b, bool hoist_b_copy = false>
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
                    const hrx_q8_0_wmma_vk128_half16_vec b_col_use =
                        (copy_b && hoist_b_copy) ?
                            q8_repro_copy_frag(b_frag[k_tile][col_sub]) :
                            b_frag[k_tile][col_sub];
#pragma unroll
                    for (int row_sub = 0; row_sub < 4; ++row_sub) {
                        const int group = col_sub * 4 + row_sub;
                        const hrx_q8_0_wmma_vk128_half16_vec a_use = copy_a ?
                            q8_repro_copy_frag(a_frag[k_tile][row_sub]) :
                            a_frag[k_tile][row_sub];
                        const hrx_q8_0_wmma_vk128_half16_vec b_use = copy_b ?
                            (hoist_b_copy ? b_col_use : q8_repro_copy_frag(b_frag[k_tile][col_sub])) :
                            b_col_use;
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

template <bool copy_a, bool copy_b, bool hoist_b_copy = false, int copy_b_min_col_sub = 0, bool use_asm_wmma = false>
__global__ __launch_bounds__(256, 1)
void q8_contract_bm128_direct192_repro_kernel(
        const hrx_block_q8_0_wmma_vk128_lhs * src0,
        const float * src1,
        float * contract,
        long long k,
        long long rows,
        long long cols) {
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 32;
    constexpr int SHARED_STRIDE = HRX_Q8_0_WMMA_VK128_SHARED_STRIDE;
    constexpr int ACTIVE_GROUPS = 16;

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
                        sh_a_lds, wave_row * 4 + row_sub, k_tile, lane);
            }
#pragma unroll
            for (int col_sub = 0; col_sub < 4; ++col_sub) {
                b_frag[k_tile][col_sub] =
                    hrx_q8_0_wmma_vk128_load_b_frag_w64_b64asm_nowait(
                        sh_b_lds, wave_col * 4 + col_sub, k_tile, lane);
            }
        }
        asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
#pragma unroll
        for (int k_tile = 0; k_tile < 2; ++k_tile) {
#pragma unroll
            for (int col_sub = 0; col_sub < 4; ++col_sub) {
                const bool do_b_copy = copy_b && col_sub >= copy_b_min_col_sub;
                const hrx_q8_0_wmma_vk128_half16_vec b_col_use =
                    (do_b_copy && hoist_b_copy) ?
                        q8_repro_copy_frag(b_frag[k_tile][col_sub]) :
                        b_frag[k_tile][col_sub];
#pragma unroll
                for (int row_sub = 0; row_sub < 4; ++row_sub) {
                    const int group = col_sub * 4 + row_sub;
                    const hrx_q8_0_wmma_vk128_half16_vec a_use = copy_a ?
                        q8_repro_copy_frag(a_frag[k_tile][row_sub]) :
                        a_frag[k_tile][row_sub];
                    const hrx_q8_0_wmma_vk128_half16_vec b_use = do_b_copy ?
                        (hoist_b_copy ? b_col_use : q8_repro_copy_frag(b_frag[k_tile][col_sub])) :
                        b_col_use;
                    if constexpr (use_asm_wmma) {
                        acc[group] = q8_repro_wmma_f16_w64_asm(a_use, b_use, acc[group]);
                    } else {
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

#pragma unroll
    for (int group = 0; group < ACTIVE_GROUPS; ++group) {
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            q8_repro_bm128_contract_store_acc(contract_rsrc, rows, cols, acc, wave, group, slot, lane);
        }
    }
#pragma unroll
    for (int group = Q8_REPRO_BM128_CONTRACT_ACTIVE_GROUPS;
            group < Q8_REPRO_BM128_CONTRACT_GROUPS;
            ++group) {
#pragma unroll
        for (int slot = 0; slot < Q8_REPRO_CONTRACT_SLOTS; ++slot) {
            q8_repro_contract_store_synthetic(contract_rsrc, group, slot, lane);
        }
    }
}

template <bool use_asm_wmma, bool use_asm_inout = false>
__global__ __launch_bounds__(256, 1)
void q8_bm128_direct192_output_repro_kernel(
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
    constexpr int ACTIVE_GROUPS = 16;

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
                        sh_a_lds, wave_row * 4 + row_sub, k_tile, lane);
            }
#pragma unroll
            for (int col_sub = 0; col_sub < 4; ++col_sub) {
                b_frag[k_tile][col_sub] =
                    hrx_q8_0_wmma_vk128_load_b_frag_w64_b64asm_nowait(
                        sh_b_lds, wave_col * 4 + col_sub, k_tile, lane);
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
                    if constexpr (use_asm_inout) {
                        acc[group] = q8_repro_wmma_f16_w64_asm_inout(
                            a_frag[k_tile][row_sub],
                            b_frag[k_tile][col_sub],
                            acc[group]);
                    } else if constexpr (use_asm_wmma) {
                        acc[group] = q8_repro_wmma_f16_w64_asm(
                            a_frag[k_tile][row_sub],
                            b_frag[k_tile][col_sub],
                            acc[group]);
                    } else {
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

#pragma unroll
    for (int group = 0; group < ACTIVE_GROUPS; ++group) {
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            q8_repro_bm128_direct_raw_store_slot(
                dst_rsrc, rows, row_base, col_base, rows, cols, acc, wave, group, slot, lane);
        }
    }
}

template <
    bool use_asm_wmma,
    bool use_asm_inout = false,
    bool copy_a = false,
    bool copy_b = false,
    bool prefetch_wait51 = false>
__global__ __launch_bounds__(256, 1)
void q8_bm128_streamk_output_repro_kernel(
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
    constexpr int ACTIVE_GROUPS = 16;

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
            if constexpr (prefetch_wait51) {
                q8_repro_prefetch_k2_fragments_for_wait51(
                    sh_a_lds, sh_b_lds, wave_row, wave_col, lane);
                asm volatile("s_waitcnt lgkmcnt(51)\n" ::: "memory");
            } else {
                asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
            }
#pragma unroll
            for (int col_sub = 0; col_sub < 4; ++col_sub) {
#pragma unroll
                for (int row_sub = 0; row_sub < 4; ++row_sub) {
                    const int group = col_sub * 4 + row_sub;
                    const hrx_q8_0_wmma_vk128_half16_vec a_use = copy_a ?
                        q8_repro_copy_frag(a_frag[row_sub]) :
                        a_frag[row_sub];
                    const hrx_q8_0_wmma_vk128_half16_vec b_use = copy_b ?
                        q8_repro_copy_frag(b_frag[col_sub]) :
                        b_frag[col_sub];
                    if constexpr (use_asm_inout) {
                        acc[group] = q8_repro_wmma_f16_w64_asm_inout(
                            a_use,
                            b_use,
                            acc[group]);
                    } else if constexpr (use_asm_wmma) {
                        acc[group] = q8_repro_wmma_f16_w64_asm(
                            a_use,
                            b_use,
                            acc[group]);
                    } else {
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

#pragma unroll
    for (int group = 0; group < ACTIVE_GROUPS; ++group) {
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            q8_repro_bm128_direct_raw_store_slot(
                dst_rsrc, rows, row_base, col_base, rows, cols, acc, wave, group, slot, lane);
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

template <
    int group_base,
    int col_start,
    int col_count,
    bool copy_a,
    bool copy_b,
    bool use_asm_wmma = false>
__global__ __launch_bounds__(256, 1)
void q8_phase96_bm128_repro_kernel(
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
    static_assert(col_count >= 1 && col_count <= 2, "unexpected column group count");

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
            sh_b[c * SHARED_STRIDE + kk] = col < cols ? static_cast<_Float16>(src1[col * k + k0 + kk]) : zero;
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
            for (int col_delta = 0; col_delta < col_count; ++col_delta) {
                const int col_sub = col_start + col_delta;
#pragma unroll
                for (int row_sub = 0; row_sub < 4; ++row_sub) {
                    const int local = col_delta * 4 + row_sub;
                    const hrx_q8_0_wmma_vk128_half16_vec a_use = copy_a ?
                        q8_repro_copy_frag(a_frag[row_sub]) :
                        a_frag[row_sub];
                    const hrx_q8_0_wmma_vk128_half16_vec b_use = copy_b ?
                        q8_repro_copy_frag(b_frag[col_sub]) :
                        b_frag[col_sub];
                    if constexpr (use_asm_wmma) {
                        acc[local] = q8_repro_wmma_f16_w64_asm(a_use, b_use, acc[local]);
                    } else {
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

#pragma unroll
    for (int local = 0; local < 8; ++local) {
        const int group = group_base + local;
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            q8_repro_bm128_raw_store_slot(
                dst_rsrc, rows, row_base, col_base, rows, cols, acc, wave, group, slot, lane);
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

static float q8_repro_motif192_expected_value(int row, int col) {
    const int tile_row = row & 127;
    const int tile_col = col & 127;
    const int row_in_wave = tile_row & 63;
    const int col_in_wave = tile_col & 63;
    const int row_sub = row_in_wave >> 4;
    const int col_sub = col_in_wave >> 4;
    const int group = col_sub * 4 + row_sub;
    const int slot = (row_in_wave & 15) >> 2;
    const int row_lane = row_in_wave & 3;
    const int col_lane = col_in_wave & 15;
    const int lane = row_lane * 16 + col_lane;
    return half_bits_to_float(float_to_half_bits(q8_repro_contract_synthetic_value(group, slot, static_cast<unsigned int>(lane))));
}

static float q8_repro_motif192_wmma_expected_value(int row, int col) {
    const int row_sub = ((row & 127) & 63) >> 4;
    const int col_sub = ((col & 127) & 63) >> 4;
    return 32.0f * static_cast<float>((row_sub + 1) * (col_sub + 1));
}

static int run_motif192_synthetic_case(int rows, int cols) {
    std::vector<float> h_out(static_cast<size_t>(rows) * cols, -7777.0f);
    device_buffer<float> d_out(h_out.size());
    HIP_CHECK(hipMemcpy(d_out.ptr, h_out.data(), h_out.size() * sizeof(h_out[0]), hipMemcpyHostToDevice));

    dim3 grid((rows + 127) / 128, (cols + 127) / 128, 1);
    hipLaunchKernelGGL(q8_motif192_synthetic_store_kernel, grid, dim3(256, 1, 1), 0, 0,
        d_out.ptr, rows, cols);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_out.data(), d_out.ptr, h_out.size() * sizeof(h_out[0]), hipMemcpyDeviceToHost));

    size_t bad = 0;
    size_t nan = 0;
    size_t inf = 0;
    size_t sentinel = 0;
    float max_abs = 0.0f;
    bool have_first_bad = false;
    int first_row = -1;
    int first_col = -1;
    float first_actual = 0.0f;
    float first_expected = 0.0f;
    float first_err = 0.0f;

    for (int col = 0; col < cols; ++col) {
        for (int row = 0; row < rows; ++row) {
            const size_t index = static_cast<size_t>(col) * rows + static_cast<size_t>(row);
            const float actual = h_out[index];
            const float expected = q8_repro_motif192_expected_value(row, col);
            if (actual == -7777.0f) {
                ++sentinel;
                ++bad;
                if (!have_first_bad) {
                    have_first_bad = true;
                    first_row = row;
                    first_col = col;
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
                    first_row = row;
                    first_col = col;
                    first_actual = actual;
                    first_expected = expected;
                    first_err = INFINITY;
                }
                continue;
            }
            if (std::isinf(actual)) {
                ++inf;
                ++bad;
                if (!have_first_bad) {
                    have_first_bad = true;
                    first_row = row;
                    first_col = col;
                    first_actual = actual;
                    first_expected = expected;
                    first_err = INFINITY;
                }
                continue;
            }
            const float err = std::fabs(actual - expected);
            max_abs = std::max(max_abs, err);
            if (err > 0.0f) {
                ++bad;
                if (!have_first_bad) {
                    have_first_bad = true;
                    first_row = row;
                    first_col = col;
                    first_actual = actual;
                    first_expected = expected;
                    first_err = err;
                }
            }
        }
    }

    std::printf(
        "motif192-synth-address rows=%d cols=%d active=%zu bad=%zu nan=%zu inf=%zu sentinel=%zu max_abs=%g\n",
        rows,
        cols,
        h_out.size(),
        bad,
        nan,
        inf,
        sentinel,
        max_abs);
    if (have_first_bad) {
        std::printf(
            "  first_bad row=%d col=%d actual=%g expected=%g err=%g\n",
            first_row,
            first_col,
            first_actual,
            first_expected,
            first_err);
    }
    return bad == 0 ? 0 : 1;
}

static int run_motif192_wmma_case(
        const char * label,
        int rows,
        int cols,
        int store_mode,
        bool wait_after_load) {
    std::vector<float> h_out(static_cast<size_t>(rows) * cols, -7777.0f);
    device_buffer<float> d_out(h_out.size());
    HIP_CHECK(hipMemcpy(d_out.ptr, h_out.data(), h_out.size() * sizeof(h_out[0]), hipMemcpyHostToDevice));

    dim3 grid((rows + 127) / 128, (cols + 127) / 128, 1);
    hipLaunchKernelGGL(q8_motif192_wmma_store_kernel, grid, dim3(256, 1, 1), 0, 0,
        d_out.ptr, rows, cols, store_mode, wait_after_load);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_out.data(), d_out.ptr, h_out.size() * sizeof(h_out[0]), hipMemcpyDeviceToHost));

    size_t bad = 0;
    size_t nan = 0;
    size_t inf = 0;
    size_t sentinel = 0;
    float max_abs = 0.0f;
    bool have_first_bad = false;
    int first_row = -1;
    int first_col = -1;
    float first_actual = 0.0f;
    float first_expected = 0.0f;
    float first_err = 0.0f;

    for (int col = 0; col < cols; ++col) {
        for (int row = 0; row < rows; ++row) {
            const size_t index = static_cast<size_t>(col) * rows + static_cast<size_t>(row);
            const float actual = h_out[index];
            const float expected = q8_repro_motif192_wmma_expected_value(row, col);
            if (actual == -7777.0f) {
                ++sentinel;
                ++bad;
                if (!have_first_bad) {
                    have_first_bad = true;
                    first_row = row;
                    first_col = col;
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
                    first_row = row;
                    first_col = col;
                    first_actual = actual;
                    first_expected = expected;
                    first_err = INFINITY;
                }
                continue;
            }
            if (std::isinf(actual)) {
                ++inf;
                ++bad;
                if (!have_first_bad) {
                    have_first_bad = true;
                    first_row = row;
                    first_col = col;
                    first_actual = actual;
                    first_expected = expected;
                    first_err = INFINITY;
                }
                continue;
            }
            const float err = std::fabs(actual - expected);
            max_abs = std::max(max_abs, err);
            if (err > 0.0f) {
                ++bad;
                if (!have_first_bad) {
                    have_first_bad = true;
                    first_row = row;
                    first_col = col;
                    first_actual = actual;
                    first_expected = expected;
                    first_err = err;
                }
            }
        }
    }

    std::printf(
        "%s rows=%d cols=%d active=%zu bad=%zu nan=%zu inf=%zu sentinel=%zu max_abs=%g\n",
        label,
        rows,
        cols,
        h_out.size(),
        bad,
        nan,
        inf,
        sentinel,
        max_abs);
    if (have_first_bad) {
        std::printf(
            "  first_bad row=%d col=%d actual=%g expected=%g err=%g\n",
            first_row,
            first_col,
            first_actual,
            first_expected,
            first_err);
    }
    return bad == 0 ? 0 : 1;
}

template <bool dependent_ladder>
static int run_motif192_wmma_k2_case(
        const char * label,
        int rows,
        int cols,
        bool wait_after_load) {
    std::vector<float> h_out(static_cast<size_t>(rows) * cols, -7777.0f);
    device_buffer<float> d_out(h_out.size());
    HIP_CHECK(hipMemcpy(d_out.ptr, h_out.data(), h_out.size() * sizeof(h_out[0]), hipMemcpyHostToDevice));

    dim3 grid((rows + 127) / 128, (cols + 127) / 128, 1);
    hipLaunchKernelGGL((q8_motif192_wmma_k2_store_kernel<dependent_ladder>),
        grid,
        dim3(256, 1, 1),
        0,
        0,
        d_out.ptr,
        rows,
        cols,
        wait_after_load);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_out.data(), d_out.ptr, h_out.size() * sizeof(h_out[0]), hipMemcpyDeviceToHost));

    size_t bad = 0;
    size_t nan = 0;
    size_t inf = 0;
    size_t sentinel = 0;
    float max_abs = 0.0f;
    bool have_first_bad = false;
    int first_row = -1;
    int first_col = -1;
    float first_actual = 0.0f;
    float first_expected = 0.0f;
    float first_err = 0.0f;

    for (int col = 0; col < cols; ++col) {
        for (int row = 0; row < rows; ++row) {
            const size_t index = static_cast<size_t>(col) * rows + static_cast<size_t>(row);
            const float actual = h_out[index];
            const float expected = q8_repro_motif192_wmma_expected_value(row, col);
            if (actual == -7777.0f) {
                ++sentinel;
                ++bad;
                if (!have_first_bad) {
                    have_first_bad = true;
                    first_row = row;
                    first_col = col;
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
                    first_row = row;
                    first_col = col;
                    first_actual = actual;
                    first_expected = expected;
                    first_err = INFINITY;
                }
                continue;
            }
            if (std::isinf(actual)) {
                ++inf;
                ++bad;
                if (!have_first_bad) {
                    have_first_bad = true;
                    first_row = row;
                    first_col = col;
                    first_actual = actual;
                    first_expected = expected;
                    first_err = INFINITY;
                }
                continue;
            }
            const float err = std::fabs(actual - expected);
            max_abs = std::max(max_abs, err);
            if (err > 0.0f) {
                ++bad;
                if (!have_first_bad) {
                    have_first_bad = true;
                    first_row = row;
                    first_col = col;
                    first_actual = actual;
                    first_expected = expected;
                    first_err = err;
                }
            }
        }
    }

    std::printf(
        "%s rows=%d cols=%d active=%zu bad=%zu nan=%zu inf=%zu sentinel=%zu max_abs=%g\n",
        label,
        rows,
        cols,
        h_out.size(),
        bad,
        nan,
        inf,
        sentinel,
        max_abs);
    if (have_first_bad) {
        std::printf(
            "  first_bad row=%d col=%d actual=%g expected=%g err=%g\n",
            first_row,
            first_col,
            first_actual,
            first_expected,
            first_err);
    }
    return bad == 0 ? 0 : 1;
}

static int run_motif192_wmma_suite(const char * label, int store_mode, bool wait_after_load) {
    int status = 0;
    status |= run_motif192_wmma_case(label, 128, 128, store_mode, wait_after_load);
    status |= run_motif192_wmma_case(label, 128, 129, store_mode, wait_after_load);
    status |= run_motif192_wmma_case(label, 129, 128, store_mode, wait_after_load);
    status |= run_motif192_wmma_case(label, 1024, 512, store_mode, wait_after_load);
    status |= run_motif192_wmma_case(label, 4096, 512, store_mode, wait_after_load);
    status |= run_motif192_wmma_case(label, 4096, 513, store_mode, wait_after_load);
    return status;
}

template <bool dependent_ladder>
static int run_motif192_wmma_k2_suite(const char * label, bool wait_after_load) {
    int status = 0;
    status |= run_motif192_wmma_k2_case<dependent_ladder>(label, 128, 128, wait_after_load);
    status |= run_motif192_wmma_k2_case<dependent_ladder>(label, 128, 129, wait_after_load);
    status |= run_motif192_wmma_k2_case<dependent_ladder>(label, 129, 128, wait_after_load);
    status |= run_motif192_wmma_k2_case<dependent_ladder>(label, 1024, 512, wait_after_load);
    status |= run_motif192_wmma_k2_case<dependent_ladder>(label, 4096, 512, wait_after_load);
    status |= run_motif192_wmma_k2_case<dependent_ladder>(label, 4096, 513, wait_after_load);
    return status;
}

template <typename T>
static bool read_binary_vector(const std::string & path, std::vector<T> & out, size_t expected_count) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::fprintf(stderr, "failed to open %s\n", path.c_str());
        return false;
    }
    file.seekg(0, std::ios::end);
    const std::streamoff size = file.tellg();
    file.seekg(0, std::ios::beg);
    if (size < 0 || static_cast<size_t>(size) != expected_count * sizeof(T)) {
        std::fprintf(stderr,
            "unexpected size for %s: got %lld expected %zu\n",
            path.c_str(),
            static_cast<long long>(size),
            expected_count * sizeof(T));
        return false;
    }
    out.resize(expected_count);
    file.read(reinterpret_cast<char *>(out.data()), size);
    return static_cast<bool>(file);
}

static bool read_meta_int(const std::string & path, const std::string & key, int * value) {
    std::ifstream file(path);
    if (!file) {
        std::fprintf(stderr, "failed to open %s\n", path.c_str());
        return false;
    }
    std::string line;
    const std::string prefix = key + "=";
    while (std::getline(file, line)) {
        if (line.rfind(prefix, 0) == 0) {
            *value = std::atoi(line.c_str() + prefix.size());
            return true;
        }
    }
    std::fprintf(stderr, "missing %s in %s\n", key.c_str(), path.c_str());
    return false;
}

static float rhs_value(int col, int k_index) {
    const int raw = (col * 19 + k_index * 7 + 23) & 63;
    return (static_cast<float>(raw) - 31.0f) * 0.0015f;
}

static float deterministic_uniform_signed(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    const float u = static_cast<float>(x & 0x00ffffffu) * (1.0f / 8388607.5f);
    return u - 1.0f;
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

static void fill_q8_backend_like(std::vector<hrx_block_q8_0_wmma_vk128_lhs> & blocks, int rows, int blocks_per_row) {
    for (int row = 0; row < rows; ++row) {
        for (int block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
            float values[32];
            float amax = 0.0f;
            for (int i = 0; i < 32; ++i) {
                const uint32_t seed =
                    0x9e3779b9u ^
                    static_cast<uint32_t>(row * 0x45d9f3bu) ^
                    static_cast<uint32_t>(block_idx * 0x119de1f3u) ^
                    static_cast<uint32_t>(i * 0x3449b1u);
                values[i] = deterministic_uniform_signed(seed);
                amax = std::max(amax, std::fabs(values[i]));
            }
            const float d = amax / 127.0f;
            const float id = d != 0.0f ? 1.0f / d : 0.0f;
            hrx_block_q8_0_wmma_vk128_lhs & block =
                blocks[static_cast<size_t>(row) * blocks_per_row + block_idx];
            block.d = float_to_half_bits(d);
            for (int i = 0; i < 32; ++i) {
                block.qs[i] = static_cast<int8_t>(std::round(values[i] * id));
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

static void fill_rhs_backend_like(std::vector<float> & rhs, int k, int cols) {
    for (int col = 0; col < cols; ++col) {
        for (int kk = 0; kk < k; ++kk) {
            const uint32_t seed =
                0x243f6a88u ^
                static_cast<uint32_t>(col * 0x85ebca6bu) ^
                static_cast<uint32_t>(kk * 0xc2b2ae35u);
            rhs[static_cast<size_t>(col) * k + kk] = deterministic_uniform_signed(seed);
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

static float cpu_reference_value(
        const std::vector<hrx_block_q8_0_wmma_vk128_lhs> & blocks,
        const std::vector<float> & rhs,
        int k,
        int rows,
        int cols,
        int row,
        int col) {
    (void) rows;
    (void) cols;
    const int blocks_per_row = k / 32;
    float sum = 0.0f;
    for (int kk = 0; kk < k; ++kk) {
        sum += q8_dequant(blocks, row, kk, blocks_per_row) * rhs[static_cast<size_t>(col) * k + kk];
    }
    return sum;
}

template <bool dependent_ladder>
static int run_motif192_wmma_k2_realdata_k32_case(
        const char * label,
        int rows,
        int cols,
        bool wait_after_load) {
    constexpr int k = 32;
    const int blocks_per_row = k / 32;
    std::vector<hrx_block_q8_0_wmma_vk128_lhs> h_q8(static_cast<size_t>(rows) * blocks_per_row);
    std::vector<float> h_rhs(static_cast<size_t>(cols) * k);
    std::vector<float> h_out(static_cast<size_t>(rows) * cols, -7777.0f);
    fill_q8_backend_like(h_q8, rows, blocks_per_row);
    fill_rhs_backend_like(h_rhs, k, cols);
    const std::vector<float> ref = cpu_reference(h_q8, h_rhs, k, rows, cols);

    device_buffer<hrx_block_q8_0_wmma_vk128_lhs> d_q8(h_q8.size());
    device_buffer<float> d_rhs(h_rhs.size());
    device_buffer<float> d_out(h_out.size());
    HIP_CHECK(hipMemcpy(d_q8.ptr, h_q8.data(), h_q8.size() * sizeof(h_q8[0]), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_rhs.ptr, h_rhs.data(), h_rhs.size() * sizeof(h_rhs[0]), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_out.ptr, h_out.data(), h_out.size() * sizeof(h_out[0]), hipMemcpyHostToDevice));

    dim3 grid((rows + 127) / 128, (cols + 127) / 128, 1);
    hipLaunchKernelGGL((q8_motif192_wmma_k2_realdata_k32_store_kernel<dependent_ladder>),
        grid,
        dim3(256, 1, 1),
        0,
        0,
        d_q8.ptr,
        d_rhs.ptr,
        d_out.ptr,
        k,
        rows,
        cols,
        wait_after_load);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_out.data(), d_out.ptr, h_out.size() * sizeof(h_out[0]), hipMemcpyDeviceToHost));

    size_t bad = 0;
    size_t nan = 0;
    size_t inf = 0;
    size_t sentinel = 0;
    double sq_err = 0.0;
    double sq_ref = 0.0;
    float max_abs = 0.0f;
    bool have_first_bad = false;
    int first_row = -1;
    int first_col = -1;
    float first_actual = 0.0f;
    float first_expected = 0.0f;
    float first_err = 0.0f;

    for (int col = 0; col < cols; ++col) {
        for (int row = 0; row < rows; ++row) {
            const size_t index = static_cast<size_t>(col) * rows + static_cast<size_t>(row);
            const float actual = h_out[index];
            const float expected = ref[index];
            if (actual == -7777.0f) {
                ++sentinel;
                ++bad;
                if (!have_first_bad) {
                    have_first_bad = true;
                    first_row = row;
                    first_col = col;
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
                    first_row = row;
                    first_col = col;
                    first_actual = actual;
                    first_expected = expected;
                    first_err = INFINITY;
                }
                continue;
            }
            if (std::isinf(actual)) {
                ++inf;
                ++bad;
                if (!have_first_bad) {
                    have_first_bad = true;
                    first_row = row;
                    first_col = col;
                    first_actual = actual;
                    first_expected = expected;
                    first_err = INFINITY;
                }
                continue;
            }
            const float err = std::fabs(actual - expected);
            max_abs = std::max(max_abs, err);
            sq_err += static_cast<double>(err) * static_cast<double>(err);
            sq_ref += static_cast<double>(expected) * static_cast<double>(expected);
            const float tol = 0.35f + 0.015f * std::fabs(expected);
            if (err > tol) {
                ++bad;
                if (!have_first_bad) {
                    have_first_bad = true;
                    first_row = row;
                    first_col = col;
                    first_actual = actual;
                    first_expected = expected;
                    first_err = err;
                }
            }
        }
    }

    const double nmse = sq_ref > 0.0 ? sq_err / sq_ref : sq_err;
    std::printf(
        "%s rows=%d cols=%d k=%d active=%zu bad=%zu nan=%zu inf=%zu sentinel=%zu max_abs=%g nmse=%g\n",
        label,
        rows,
        cols,
        k,
        h_out.size(),
        bad,
        nan,
        inf,
        sentinel,
        max_abs,
        nmse);
    if (have_first_bad) {
        std::printf(
            "  first_bad row=%d col=%d actual=%g expected=%g err=%g\n",
            first_row,
            first_col,
            first_actual,
            first_expected,
            first_err);
    }
    return bad == 0 ? 0 : 1;
}

template <bool dependent_ladder>
static int run_motif192_wmma_k2_realdata_k32_suite(const char * label, bool wait_after_load) {
    int status = 0;
    status |= run_motif192_wmma_k2_realdata_k32_case<dependent_ladder>(label, 128, 128, wait_after_load);
    status |= run_motif192_wmma_k2_realdata_k32_case<dependent_ladder>(label, 128, 129, wait_after_load);
    status |= run_motif192_wmma_k2_realdata_k32_case<dependent_ladder>(label, 129, 128, wait_after_load);
    status |= run_motif192_wmma_k2_realdata_k32_case<dependent_ladder>(label, 1024, 512, wait_after_load);
    status |= run_motif192_wmma_k2_realdata_k32_case<dependent_ladder>(label, 4096, 512, wait_after_load);
    status |= run_motif192_wmma_k2_realdata_k32_case<dependent_ladder>(label, 4096, 513, wait_after_load);
    return status;
}

template <bool dependent_ladder>
static int run_motif192_wmma_k2_realdata_fullk_case(
        const char * label,
        int rows,
        int cols,
        int k,
        bool wait_after_load) {
    const int blocks_per_row = k / 32;
    std::vector<hrx_block_q8_0_wmma_vk128_lhs> h_q8(static_cast<size_t>(rows) * blocks_per_row);
    std::vector<float> h_rhs(static_cast<size_t>(cols) * k);
    std::vector<float> h_out(static_cast<size_t>(rows) * cols, -7777.0f);
    fill_q8_backend_like(h_q8, rows, blocks_per_row);
    fill_rhs_backend_like(h_rhs, k, cols);

    device_buffer<hrx_block_q8_0_wmma_vk128_lhs> d_q8(h_q8.size());
    device_buffer<float> d_rhs(h_rhs.size());
    device_buffer<float> d_out(h_out.size());
    HIP_CHECK(hipMemcpy(d_q8.ptr, h_q8.data(), h_q8.size() * sizeof(h_q8[0]), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_rhs.ptr, h_rhs.data(), h_rhs.size() * sizeof(h_rhs[0]), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_out.ptr, h_out.data(), h_out.size() * sizeof(h_out[0]), hipMemcpyHostToDevice));

    dim3 grid((rows + 127) / 128, (cols + 127) / 128, 1);
    hipLaunchKernelGGL((q8_motif192_wmma_k2_realdata_fullk_store_kernel<dependent_ladder>),
        grid,
        dim3(256, 1, 1),
        0,
        0,
        d_q8.ptr,
        d_rhs.ptr,
        d_out.ptr,
        k,
        rows,
        cols,
        wait_after_load);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_out.data(), d_out.ptr, h_out.size() * sizeof(h_out[0]), hipMemcpyDeviceToHost));

    const size_t total = h_out.size();
    const size_t max_checks = 8192;
    const size_t stride = total <= max_checks ? 1 : std::max<size_t>(1, total / max_checks);

    size_t checked = 0;
    size_t bad = 0;
    size_t nan = 0;
    size_t inf = 0;
    size_t sentinel = 0;
    double sq_err = 0.0;
    double sq_ref = 0.0;
    float max_abs = 0.0f;
    bool have_first_bad = false;
    int first_row = -1;
    int first_col = -1;
    float first_actual = 0.0f;
    float first_expected = 0.0f;
    float first_err = 0.0f;

    for (size_t index = 0; index < total; index += stride) {
        const int row = static_cast<int>(index % static_cast<size_t>(rows));
        const int col = static_cast<int>(index / static_cast<size_t>(rows));
        const float actual = h_out[index];
        const float expected = cpu_reference_value(h_q8, h_rhs, k, rows, cols, row, col);
        ++checked;
        if (actual == -7777.0f) {
            ++sentinel;
            ++bad;
            if (!have_first_bad) {
                have_first_bad = true;
                first_row = row;
                first_col = col;
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
                first_row = row;
                first_col = col;
                first_actual = actual;
                first_expected = expected;
                first_err = INFINITY;
            }
            continue;
        }
        if (std::isinf(actual)) {
            ++inf;
            ++bad;
            if (!have_first_bad) {
                have_first_bad = true;
                first_row = row;
                first_col = col;
                first_actual = actual;
                first_expected = expected;
                first_err = INFINITY;
            }
            continue;
        }
        const float err = std::fabs(actual - expected);
        max_abs = std::max(max_abs, err);
        sq_err += static_cast<double>(err) * static_cast<double>(err);
        sq_ref += static_cast<double>(expected) * static_cast<double>(expected);
        const float tol = 2.0f + 0.05f * std::fabs(expected);
        if (err > tol) {
            ++bad;
            if (!have_first_bad) {
                have_first_bad = true;
                first_row = row;
                first_col = col;
                first_actual = actual;
                first_expected = expected;
                first_err = err;
            }
        }
    }

    const double nmse = sq_ref > 0.0 ? sq_err / sq_ref : sq_err;
    std::printf(
        "%s rows=%d cols=%d k=%d active=%zu checked=%zu stride=%zu bad=%zu nan=%zu inf=%zu sentinel=%zu max_abs=%g nmse=%g\n",
        label,
        rows,
        cols,
        k,
        total,
        checked,
        stride,
        bad,
        nan,
        inf,
        sentinel,
        max_abs,
        nmse);
    if (have_first_bad) {
        std::printf(
            "  first_bad row=%d col=%d actual=%g expected=%g err=%g\n",
            first_row,
            first_col,
            first_actual,
            first_expected,
            first_err);
    }
    return bad == 0 ? 0 : 1;
}

template <bool dependent_ladder>
static int run_motif192_wmma_k2_realdata_fullk_suite(const char * label, bool wait_after_load) {
    constexpr int k = 4096;
    int status = 0;
    status |= run_motif192_wmma_k2_realdata_fullk_case<dependent_ladder>(label, 128, 128, k, wait_after_load);
    status |= run_motif192_wmma_k2_realdata_fullk_case<dependent_ladder>(label, 128, 129, k, wait_after_load);
    status |= run_motif192_wmma_k2_realdata_fullk_case<dependent_ladder>(label, 129, 128, k, wait_after_load);
    status |= run_motif192_wmma_k2_realdata_fullk_case<dependent_ladder>(label, 1024, 512, k, wait_after_load);
    status |= run_motif192_wmma_k2_realdata_fullk_case<dependent_ladder>(label, 4096, 512, k, wait_after_load);
    status |= run_motif192_wmma_k2_realdata_fullk_case<dependent_ladder>(label, 4096, 513, k, wait_after_load);
    return status;
}

static int run_motif192_wmma_k2_realdata_fullk_phase8_case(
        const char * label,
        int rows,
        int cols,
        int k,
        bool wait_after_load) {
    const int blocks_per_row = k / 32;
    std::vector<hrx_block_q8_0_wmma_vk128_lhs> h_q8(static_cast<size_t>(rows) * blocks_per_row);
    std::vector<float> h_rhs(static_cast<size_t>(cols) * k);
    std::vector<float> h_out(static_cast<size_t>(rows) * cols, -7777.0f);
    fill_q8_backend_like(h_q8, rows, blocks_per_row);
    fill_rhs_backend_like(h_rhs, k, cols);

    device_buffer<hrx_block_q8_0_wmma_vk128_lhs> d_q8(h_q8.size());
    device_buffer<float> d_rhs(h_rhs.size());
    device_buffer<float> d_out(h_out.size());
    HIP_CHECK(hipMemcpy(d_q8.ptr, h_q8.data(), h_q8.size() * sizeof(h_q8[0]), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_rhs.ptr, h_rhs.data(), h_rhs.size() * sizeof(h_rhs[0]), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_out.ptr, h_out.data(), h_out.size() * sizeof(h_out[0]), hipMemcpyHostToDevice));

    dim3 grid((rows + 127) / 128, (cols + 127) / 128, 1);
    hipLaunchKernelGGL((q8_motif192_wmma_k2_realdata_fullk_phase8_store_kernel<0, 0>),
        grid,
        dim3(256, 1, 1),
        0,
        0,
        d_q8.ptr,
        d_rhs.ptr,
        d_out.ptr,
        k,
        rows,
        cols,
        wait_after_load);
    hipLaunchKernelGGL((q8_motif192_wmma_k2_realdata_fullk_phase8_store_kernel<8, 2>),
        grid,
        dim3(256, 1, 1),
        0,
        0,
        d_q8.ptr,
        d_rhs.ptr,
        d_out.ptr,
        k,
        rows,
        cols,
        wait_after_load);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_out.data(), d_out.ptr, h_out.size() * sizeof(h_out[0]), hipMemcpyDeviceToHost));

    const size_t total = h_out.size();
    const size_t max_checks = 8192;
    const size_t stride = total <= max_checks ? 1 : std::max<size_t>(1, total / max_checks);

    size_t checked = 0;
    size_t bad = 0;
    size_t nan = 0;
    size_t inf = 0;
    size_t sentinel = 0;
    double sq_err = 0.0;
    double sq_ref = 0.0;
    float max_abs = 0.0f;
    bool have_first_bad = false;
    int first_row = -1;
    int first_col = -1;
    float first_actual = 0.0f;
    float first_expected = 0.0f;
    float first_err = 0.0f;

    for (size_t index = 0; index < total; index += stride) {
        const int row = static_cast<int>(index % static_cast<size_t>(rows));
        const int col = static_cast<int>(index / static_cast<size_t>(rows));
        const float actual = h_out[index];
        const float expected = cpu_reference_value(h_q8, h_rhs, k, rows, cols, row, col);
        ++checked;
        if (actual == -7777.0f) {
            ++sentinel;
            ++bad;
            if (!have_first_bad) {
                have_first_bad = true;
                first_row = row;
                first_col = col;
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
                first_row = row;
                first_col = col;
                first_actual = actual;
                first_expected = expected;
                first_err = INFINITY;
            }
            continue;
        }
        if (std::isinf(actual)) {
            ++inf;
            ++bad;
            if (!have_first_bad) {
                have_first_bad = true;
                first_row = row;
                first_col = col;
                first_actual = actual;
                first_expected = expected;
                first_err = INFINITY;
            }
            continue;
        }
        const float err = std::fabs(actual - expected);
        max_abs = std::max(max_abs, err);
        sq_err += static_cast<double>(err) * static_cast<double>(err);
        sq_ref += static_cast<double>(expected) * static_cast<double>(expected);
        const float tol = 2.0f + 0.05f * std::fabs(expected);
        if (err > tol) {
            ++bad;
            if (!have_first_bad) {
                have_first_bad = true;
                first_row = row;
                first_col = col;
                first_actual = actual;
                first_expected = expected;
                first_err = err;
            }
        }
    }

    const double nmse = sq_ref > 0.0 ? sq_err / sq_ref : sq_err;
    std::printf(
        "%s rows=%d cols=%d k=%d active=%zu checked=%zu stride=%zu bad=%zu nan=%zu inf=%zu sentinel=%zu max_abs=%g nmse=%g\n",
        label,
        rows,
        cols,
        k,
        total,
        checked,
        stride,
        bad,
        nan,
        inf,
        sentinel,
        max_abs,
        nmse);
    if (have_first_bad) {
        std::printf(
            "  first_bad row=%d col=%d actual=%g expected=%g err=%g\n",
            first_row,
            first_col,
            first_actual,
            first_expected,
            first_err);
    }
    return bad == 0 ? 0 : 1;
}

static int run_motif192_wmma_k2_realdata_fullk_phase8_suite(const char * label, bool wait_after_load) {
    constexpr int k = 4096;
    int status = 0;
    status |= run_motif192_wmma_k2_realdata_fullk_phase8_case(label, 128, 128, k, wait_after_load);
    status |= run_motif192_wmma_k2_realdata_fullk_phase8_case(label, 128, 129, k, wait_after_load);
    status |= run_motif192_wmma_k2_realdata_fullk_phase8_case(label, 129, 128, k, wait_after_load);
    status |= run_motif192_wmma_k2_realdata_fullk_phase8_case(label, 1024, 512, k, wait_after_load);
    status |= run_motif192_wmma_k2_realdata_fullk_phase8_case(label, 4096, 512, k, wait_after_load);
    status |= run_motif192_wmma_k2_realdata_fullk_phase8_case(label, 4096, 513, k, wait_after_load);
    return status;
}

static int run_motif192_wmma_k2_realdata_fullk_phase8seq_case(
        const char * label,
        int rows,
        int cols,
        int k,
        bool wait_after_load) {
    const int blocks_per_row = k / 32;
    std::vector<hrx_block_q8_0_wmma_vk128_lhs> h_q8(static_cast<size_t>(rows) * blocks_per_row);
    std::vector<float> h_rhs(static_cast<size_t>(cols) * k);
    std::vector<float> h_out(static_cast<size_t>(rows) * cols, -7777.0f);
    fill_q8_backend_like(h_q8, rows, blocks_per_row);
    fill_rhs_backend_like(h_rhs, k, cols);

    device_buffer<hrx_block_q8_0_wmma_vk128_lhs> d_q8(h_q8.size());
    device_buffer<float> d_rhs(h_rhs.size());
    device_buffer<float> d_out(h_out.size());
    HIP_CHECK(hipMemcpy(d_q8.ptr, h_q8.data(), h_q8.size() * sizeof(h_q8[0]), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_rhs.ptr, h_rhs.data(), h_rhs.size() * sizeof(h_rhs[0]), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_out.ptr, h_out.data(), h_out.size() * sizeof(h_out[0]), hipMemcpyHostToDevice));

    dim3 grid((rows + 127) / 128, (cols + 127) / 128, 1);
    hipLaunchKernelGGL(q8_motif192_wmma_k2_realdata_fullk_phase8seq_store_kernel,
        grid,
        dim3(256, 1, 1),
        0,
        0,
        d_q8.ptr,
        d_rhs.ptr,
        d_out.ptr,
        k,
        rows,
        cols,
        wait_after_load);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_out.data(), d_out.ptr, h_out.size() * sizeof(h_out[0]), hipMemcpyDeviceToHost));

    const size_t total = h_out.size();
    const size_t max_checks = 8192;
    const size_t stride = total <= max_checks ? 1 : std::max<size_t>(1, total / max_checks);

    size_t checked = 0;
    size_t bad = 0;
    size_t nan = 0;
    size_t inf = 0;
    size_t sentinel = 0;
    double sq_err = 0.0;
    double sq_ref = 0.0;
    float max_abs = 0.0f;
    bool have_first_bad = false;
    int first_row = -1;
    int first_col = -1;
    float first_actual = 0.0f;
    float first_expected = 0.0f;
    float first_err = 0.0f;

    for (size_t index = 0; index < total; index += stride) {
        const int row = static_cast<int>(index % static_cast<size_t>(rows));
        const int col = static_cast<int>(index / static_cast<size_t>(rows));
        const float actual = h_out[index];
        const float expected = cpu_reference_value(h_q8, h_rhs, k, rows, cols, row, col);
        ++checked;
        if (actual == -7777.0f) {
            ++sentinel;
            ++bad;
            if (!have_first_bad) {
                have_first_bad = true;
                first_row = row;
                first_col = col;
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
                first_row = row;
                first_col = col;
                first_actual = actual;
                first_expected = expected;
                first_err = INFINITY;
            }
            continue;
        }
        if (std::isinf(actual)) {
            ++inf;
            ++bad;
            if (!have_first_bad) {
                have_first_bad = true;
                first_row = row;
                first_col = col;
                first_actual = actual;
                first_expected = expected;
                first_err = INFINITY;
            }
            continue;
        }
        const float err = std::fabs(actual - expected);
        max_abs = std::max(max_abs, err);
        sq_err += static_cast<double>(err) * static_cast<double>(err);
        sq_ref += static_cast<double>(expected) * static_cast<double>(expected);
        const float tol = 2.0f + 0.05f * std::fabs(expected);
        if (err > tol) {
            ++bad;
            if (!have_first_bad) {
                have_first_bad = true;
                first_row = row;
                first_col = col;
                first_actual = actual;
                first_expected = expected;
                first_err = err;
            }
        }
    }

    const double nmse = sq_ref > 0.0 ? sq_err / sq_ref : sq_err;
    std::printf(
        "%s rows=%d cols=%d k=%d active=%zu checked=%zu stride=%zu bad=%zu nan=%zu inf=%zu sentinel=%zu max_abs=%g nmse=%g\n",
        label,
        rows,
        cols,
        k,
        total,
        checked,
        stride,
        bad,
        nan,
        inf,
        sentinel,
        max_abs,
        nmse);
    if (have_first_bad) {
        std::printf(
            "  first_bad row=%d col=%d actual=%g expected=%g err=%g\n",
            first_row,
            first_col,
            first_actual,
            first_expected,
            first_err);
    }
    return bad == 0 ? 0 : 1;
}

static int run_motif192_wmma_k2_realdata_fullk_phase8seq_suite(const char * label, bool wait_after_load) {
    constexpr int k = 4096;
    int status = 0;
    status |= run_motif192_wmma_k2_realdata_fullk_phase8seq_case(label, 128, 128, k, wait_after_load);
    status |= run_motif192_wmma_k2_realdata_fullk_phase8seq_case(label, 128, 129, k, wait_after_load);
    status |= run_motif192_wmma_k2_realdata_fullk_phase8seq_case(label, 129, 128, k, wait_after_load);
    status |= run_motif192_wmma_k2_realdata_fullk_phase8seq_case(label, 1024, 512, k, wait_after_load);
    status |= run_motif192_wmma_k2_realdata_fullk_phase8seq_case(label, 4096, 512, k, wait_after_load);
    status |= run_motif192_wmma_k2_realdata_fullk_phase8seq_case(label, 4096, 513, k, wait_after_load);
    return status;
}

static int run_motif192_wmma_k2_realdata_fullk_streamfrag_case(
        const char * label,
        int rows,
        int cols,
        int k,
        bool wait_after_load) {
    const int blocks_per_row = k / 32;
    std::vector<hrx_block_q8_0_wmma_vk128_lhs> h_q8(static_cast<size_t>(rows) * blocks_per_row);
    std::vector<float> h_rhs(static_cast<size_t>(cols) * k);
    std::vector<float> h_out(static_cast<size_t>(rows) * cols, -7777.0f);
    fill_q8_backend_like(h_q8, rows, blocks_per_row);
    fill_rhs_backend_like(h_rhs, k, cols);

    device_buffer<hrx_block_q8_0_wmma_vk128_lhs> d_q8(h_q8.size());
    device_buffer<float> d_rhs(h_rhs.size());
    device_buffer<float> d_out(h_out.size());
    HIP_CHECK(hipMemcpy(d_q8.ptr, h_q8.data(), h_q8.size() * sizeof(h_q8[0]), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_rhs.ptr, h_rhs.data(), h_rhs.size() * sizeof(h_rhs[0]), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_out.ptr, h_out.data(), h_out.size() * sizeof(h_out[0]), hipMemcpyHostToDevice));

    dim3 grid((rows + 127) / 128, (cols + 127) / 128, 1);
    hipLaunchKernelGGL(q8_motif192_wmma_k2_realdata_fullk_streamfrag_store_kernel,
        grid,
        dim3(256, 1, 1),
        0,
        0,
        d_q8.ptr,
        d_rhs.ptr,
        d_out.ptr,
        k,
        rows,
        cols,
        wait_after_load);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_out.data(), d_out.ptr, h_out.size() * sizeof(h_out[0]), hipMemcpyDeviceToHost));

    const size_t total = h_out.size();
    const size_t max_checks = 8192;
    const size_t stride = total <= max_checks ? 1 : std::max<size_t>(1, total / max_checks);

    size_t checked = 0;
    size_t bad = 0;
    size_t nan = 0;
    size_t inf = 0;
    size_t sentinel = 0;
    double sq_err = 0.0;
    double sq_ref = 0.0;
    float max_abs = 0.0f;
    bool have_first_bad = false;
    int first_row = -1;
    int first_col = -1;
    float first_actual = 0.0f;
    float first_expected = 0.0f;
    float first_err = 0.0f;

    for (size_t index = 0; index < total; index += stride) {
        const int row = static_cast<int>(index % static_cast<size_t>(rows));
        const int col = static_cast<int>(index / static_cast<size_t>(rows));
        const float actual = h_out[index];
        const float expected = cpu_reference_value(h_q8, h_rhs, k, rows, cols, row, col);
        ++checked;
        if (actual == -7777.0f) {
            ++sentinel;
            ++bad;
            if (!have_first_bad) {
                have_first_bad = true;
                first_row = row;
                first_col = col;
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
                first_row = row;
                first_col = col;
                first_actual = actual;
                first_expected = expected;
                first_err = INFINITY;
            }
            continue;
        }
        if (std::isinf(actual)) {
            ++inf;
            ++bad;
            if (!have_first_bad) {
                have_first_bad = true;
                first_row = row;
                first_col = col;
                first_actual = actual;
                first_expected = expected;
                first_err = INFINITY;
            }
            continue;
        }
        const float err = std::fabs(actual - expected);
        max_abs = std::max(max_abs, err);
        sq_err += static_cast<double>(err) * static_cast<double>(err);
        sq_ref += static_cast<double>(expected) * static_cast<double>(expected);
        const float tol = 2.0f + 0.05f * std::fabs(expected);
        if (err > tol) {
            ++bad;
            if (!have_first_bad) {
                have_first_bad = true;
                first_row = row;
                first_col = col;
                first_actual = actual;
                first_expected = expected;
                first_err = err;
            }
        }
    }

    const double nmse = sq_ref > 0.0 ? sq_err / sq_ref : sq_err;
    std::printf(
        "%s rows=%d cols=%d k=%d active=%zu checked=%zu stride=%zu bad=%zu nan=%zu inf=%zu sentinel=%zu max_abs=%g nmse=%g\n",
        label,
        rows,
        cols,
        k,
        total,
        checked,
        stride,
        bad,
        nan,
        inf,
        sentinel,
        max_abs,
        nmse);
    if (have_first_bad) {
        std::printf(
            "  first_bad row=%d col=%d actual=%g expected=%g err=%g\n",
            first_row,
            first_col,
            first_actual,
            first_expected,
            first_err);
    }
    return bad == 0 ? 0 : 1;
}

static int run_motif192_wmma_k2_realdata_fullk_streamfrag_suite(const char * label, bool wait_after_load) {
    constexpr int k = 4096;
    int status = 0;
    status |= run_motif192_wmma_k2_realdata_fullk_streamfrag_case(label, 128, 128, k, wait_after_load);
    status |= run_motif192_wmma_k2_realdata_fullk_streamfrag_case(label, 128, 129, k, wait_after_load);
    status |= run_motif192_wmma_k2_realdata_fullk_streamfrag_case(label, 129, 128, k, wait_after_load);
    status |= run_motif192_wmma_k2_realdata_fullk_streamfrag_case(label, 1024, 512, k, wait_after_load);
    status |= run_motif192_wmma_k2_realdata_fullk_streamfrag_case(label, 4096, 512, k, wait_after_load);
    status |= run_motif192_wmma_k2_realdata_fullk_streamfrag_case(label, 4096, 513, k, wait_after_load);
    return status;
}

static int run_motif192_wmma_k2_realdata_fullk_ktilefrag_case(
        const char * label,
        int rows,
        int cols,
        int k,
        bool wait_after_load,
        bool storebatch = false,
        bool storebatch4 = false) {
    const int blocks_per_row = k / 32;
    std::vector<hrx_block_q8_0_wmma_vk128_lhs> h_q8(static_cast<size_t>(rows) * blocks_per_row);
    std::vector<float> h_rhs(static_cast<size_t>(cols) * k);
    std::vector<float> h_out(static_cast<size_t>(rows) * cols, -7777.0f);
    fill_q8_backend_like(h_q8, rows, blocks_per_row);
    fill_rhs_backend_like(h_rhs, k, cols);

    device_buffer<hrx_block_q8_0_wmma_vk128_lhs> d_q8(h_q8.size());
    device_buffer<float> d_rhs(h_rhs.size());
    device_buffer<float> d_out(h_out.size());
    HIP_CHECK(hipMemcpy(d_q8.ptr, h_q8.data(), h_q8.size() * sizeof(h_q8[0]), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_rhs.ptr, h_rhs.data(), h_rhs.size() * sizeof(h_rhs[0]), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_out.ptr, h_out.data(), h_out.size() * sizeof(h_out[0]), hipMemcpyHostToDevice));

    dim3 grid((rows + 127) / 128, (cols + 127) / 128, 1);
    if (storebatch4) {
        hipLaunchKernelGGL(q8_motif192_wmma_k2_realdata_fullk_ktilefrag_storebatch4_kernel,
            grid,
            dim3(256, 1, 1),
            0,
            0,
            d_q8.ptr,
            d_rhs.ptr,
            d_out.ptr,
            k,
            rows,
            cols,
            wait_after_load);
    } else if (storebatch) {
        hipLaunchKernelGGL(q8_motif192_wmma_k2_realdata_fullk_ktilefrag_storebatch_kernel,
            grid,
            dim3(256, 1, 1),
            0,
            0,
            d_q8.ptr,
            d_rhs.ptr,
            d_out.ptr,
            k,
            rows,
            cols,
            wait_after_load);
    } else {
        hipLaunchKernelGGL(q8_motif192_wmma_k2_realdata_fullk_ktilefrag_store_kernel,
            grid,
            dim3(256, 1, 1),
            0,
            0,
            d_q8.ptr,
            d_rhs.ptr,
            d_out.ptr,
            k,
            rows,
            cols,
            wait_after_load);
    }
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_out.data(), d_out.ptr, h_out.size() * sizeof(h_out[0]), hipMemcpyDeviceToHost));

    const size_t total = h_out.size();
    const size_t max_checks = 8192;
    const size_t stride = total <= max_checks ? 1 : std::max<size_t>(1, total / max_checks);

    size_t checked = 0;
    size_t bad = 0;
    size_t nan = 0;
    size_t inf = 0;
    size_t sentinel = 0;
    double sq_err = 0.0;
    double sq_ref = 0.0;
    float max_abs = 0.0f;
    bool have_first_bad = false;
    int first_row = -1;
    int first_col = -1;
    float first_actual = 0.0f;
    float first_expected = 0.0f;
    float first_err = 0.0f;

    for (size_t index = 0; index < total; index += stride) {
        const int row = static_cast<int>(index % static_cast<size_t>(rows));
        const int col = static_cast<int>(index / static_cast<size_t>(rows));
        const float actual = h_out[index];
        const float expected = cpu_reference_value(h_q8, h_rhs, k, rows, cols, row, col);
        ++checked;
        if (actual == -7777.0f) {
            ++sentinel;
            ++bad;
            if (!have_first_bad) {
                have_first_bad = true;
                first_row = row;
                first_col = col;
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
                first_row = row;
                first_col = col;
                first_actual = actual;
                first_expected = expected;
                first_err = INFINITY;
            }
            continue;
        }
        if (std::isinf(actual)) {
            ++inf;
            ++bad;
            if (!have_first_bad) {
                have_first_bad = true;
                first_row = row;
                first_col = col;
                first_actual = actual;
                first_expected = expected;
                first_err = INFINITY;
            }
            continue;
        }
        const float err = std::fabs(actual - expected);
        max_abs = std::max(max_abs, err);
        sq_err += static_cast<double>(err) * static_cast<double>(err);
        sq_ref += static_cast<double>(expected) * static_cast<double>(expected);
        const float tol = 2.0f + 0.05f * std::fabs(expected);
        if (err > tol) {
            ++bad;
            if (!have_first_bad) {
                have_first_bad = true;
                first_row = row;
                first_col = col;
                first_actual = actual;
                first_expected = expected;
                first_err = err;
            }
        }
    }

    const double nmse = sq_ref > 0.0 ? sq_err / sq_ref : sq_err;
    std::printf(
        "%s rows=%d cols=%d k=%d active=%zu checked=%zu stride=%zu bad=%zu nan=%zu inf=%zu sentinel=%zu max_abs=%g nmse=%g\n",
        label,
        rows,
        cols,
        k,
        total,
        checked,
        stride,
        bad,
        nan,
        inf,
        sentinel,
        max_abs,
        nmse);
    if (have_first_bad) {
        std::printf(
            "  first_bad row=%d col=%d actual=%g expected=%g err=%g\n",
            first_row,
            first_col,
            first_actual,
            first_expected,
            first_err);
    }
    return bad == 0 ? 0 : 1;
}

static int run_motif192_wmma_k2_realdata_fullk_ktilefrag_suite(
        const char * label,
        bool wait_after_load,
        bool storebatch = false,
        bool storebatch4 = false) {
    constexpr int k = 4096;
    int status = 0;
    status |= run_motif192_wmma_k2_realdata_fullk_ktilefrag_case(label, 128, 128, k, wait_after_load, storebatch, storebatch4);
    status |= run_motif192_wmma_k2_realdata_fullk_ktilefrag_case(label, 128, 129, k, wait_after_load, storebatch, storebatch4);
    status |= run_motif192_wmma_k2_realdata_fullk_ktilefrag_case(label, 129, 128, k, wait_after_load, storebatch, storebatch4);
    status |= run_motif192_wmma_k2_realdata_fullk_ktilefrag_case(label, 1024, 512, k, wait_after_load, storebatch, storebatch4);
    status |= run_motif192_wmma_k2_realdata_fullk_ktilefrag_case(label, 4096, 512, k, wait_after_load, storebatch, storebatch4);
    status |= run_motif192_wmma_k2_realdata_fullk_ktilefrag_case(label, 4096, 513, k, wait_after_load, storebatch, storebatch4);
    return status;
}

static int run_motif192_wmma_k2_realdata_fullk_colpairfrag_case(
        const char * label,
        int rows,
        int cols,
        int k,
        bool wait_after_load) {
    const int blocks_per_row = k / 32;
    std::vector<hrx_block_q8_0_wmma_vk128_lhs> h_q8(static_cast<size_t>(rows) * blocks_per_row);
    std::vector<float> h_rhs(static_cast<size_t>(cols) * k);
    std::vector<float> h_out(static_cast<size_t>(rows) * cols, -7777.0f);
    fill_q8_backend_like(h_q8, rows, blocks_per_row);
    fill_rhs_backend_like(h_rhs, k, cols);

    device_buffer<hrx_block_q8_0_wmma_vk128_lhs> d_q8(h_q8.size());
    device_buffer<float> d_rhs(h_rhs.size());
    device_buffer<float> d_out(h_out.size());
    HIP_CHECK(hipMemcpy(d_q8.ptr, h_q8.data(), h_q8.size() * sizeof(h_q8[0]), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_rhs.ptr, h_rhs.data(), h_rhs.size() * sizeof(h_rhs[0]), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_out.ptr, h_out.data(), h_out.size() * sizeof(h_out[0]), hipMemcpyHostToDevice));

    dim3 grid((rows + 127) / 128, (cols + 127) / 128, 1);
    hipLaunchKernelGGL(q8_motif192_wmma_k2_realdata_fullk_colpairfrag_store_kernel,
        grid,
        dim3(256, 1, 1),
        0,
        0,
        d_q8.ptr,
        d_rhs.ptr,
        d_out.ptr,
        k,
        rows,
        cols,
        wait_after_load);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_out.data(), d_out.ptr, h_out.size() * sizeof(h_out[0]), hipMemcpyDeviceToHost));

    const size_t total = h_out.size();
    const size_t max_checks = 8192;
    const size_t stride = total <= max_checks ? 1 : std::max<size_t>(1, total / max_checks);

    size_t checked = 0;
    size_t bad = 0;
    size_t nan = 0;
    size_t inf = 0;
    size_t sentinel = 0;
    double sq_err = 0.0;
    double sq_ref = 0.0;
    float max_abs = 0.0f;
    bool have_first_bad = false;
    int first_row = -1;
    int first_col = -1;
    float first_actual = 0.0f;
    float first_expected = 0.0f;
    float first_err = 0.0f;

    for (size_t index = 0; index < total; index += stride) {
        const int row = static_cast<int>(index % static_cast<size_t>(rows));
        const int col = static_cast<int>(index / static_cast<size_t>(rows));
        const float actual = h_out[index];
        const float expected = cpu_reference_value(h_q8, h_rhs, k, rows, cols, row, col);
        ++checked;
        if (actual == -7777.0f) {
            ++sentinel;
            ++bad;
            if (!have_first_bad) {
                have_first_bad = true;
                first_row = row;
                first_col = col;
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
                first_row = row;
                first_col = col;
                first_actual = actual;
                first_expected = expected;
                first_err = INFINITY;
            }
            continue;
        }
        if (std::isinf(actual)) {
            ++inf;
            ++bad;
            if (!have_first_bad) {
                have_first_bad = true;
                first_row = row;
                first_col = col;
                first_actual = actual;
                first_expected = expected;
                first_err = INFINITY;
            }
            continue;
        }
        const float err = std::fabs(actual - expected);
        max_abs = std::max(max_abs, err);
        sq_err += static_cast<double>(err) * static_cast<double>(err);
        sq_ref += static_cast<double>(expected) * static_cast<double>(expected);
        const float tol = 2.0f + 0.05f * std::fabs(expected);
        if (err > tol) {
            ++bad;
            if (!have_first_bad) {
                have_first_bad = true;
                first_row = row;
                first_col = col;
                first_actual = actual;
                first_expected = expected;
                first_err = err;
            }
        }
    }

    const double nmse = sq_ref > 0.0 ? sq_err / sq_ref : sq_err;
    std::printf(
        "%s rows=%d cols=%d k=%d active=%zu checked=%zu stride=%zu bad=%zu nan=%zu inf=%zu sentinel=%zu max_abs=%g nmse=%g\n",
        label,
        rows,
        cols,
        k,
        total,
        checked,
        stride,
        bad,
        nan,
        inf,
        sentinel,
        max_abs,
        nmse);
    if (have_first_bad) {
        std::printf(
            "  first_bad row=%d col=%d actual=%g expected=%g err=%g\n",
            first_row,
            first_col,
            first_actual,
            first_expected,
            first_err);
    }
    return bad == 0 ? 0 : 1;
}

static int run_motif192_wmma_k2_realdata_fullk_colpairfrag_suite(const char * label, bool wait_after_load) {
    constexpr int k = 4096;
    int status = 0;
    status |= run_motif192_wmma_k2_realdata_fullk_colpairfrag_case(label, 128, 128, k, wait_after_load);
    status |= run_motif192_wmma_k2_realdata_fullk_colpairfrag_case(label, 128, 129, k, wait_after_load);
    status |= run_motif192_wmma_k2_realdata_fullk_colpairfrag_case(label, 129, 128, k, wait_after_load);
    status |= run_motif192_wmma_k2_realdata_fullk_colpairfrag_case(label, 1024, 512, k, wait_after_load);
    status |= run_motif192_wmma_k2_realdata_fullk_colpairfrag_case(label, 4096, 512, k, wait_after_load);
    status |= run_motif192_wmma_k2_realdata_fullk_colpairfrag_case(label, 4096, 513, k, wait_after_load);
    return status;
}

static int run_motif192_wmma_k2_realdata_fullk_accpark_case(
        const char * label,
        int rows,
        int cols,
        int k,
        bool wait_after_load) {
    const int blocks_per_row = k / 32;
    std::vector<hrx_block_q8_0_wmma_vk128_lhs> h_q8(static_cast<size_t>(rows) * blocks_per_row);
    std::vector<float> h_rhs(static_cast<size_t>(cols) * k);
    std::vector<float> h_out(static_cast<size_t>(rows) * cols, -7777.0f);
    fill_q8_backend_like(h_q8, rows, blocks_per_row);
    fill_rhs_backend_like(h_rhs, k, cols);

    device_buffer<hrx_block_q8_0_wmma_vk128_lhs> d_q8(h_q8.size());
    device_buffer<float> d_rhs(h_rhs.size());
    device_buffer<float> d_out(h_out.size());
    HIP_CHECK(hipMemcpy(d_q8.ptr, h_q8.data(), h_q8.size() * sizeof(h_q8[0]), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_rhs.ptr, h_rhs.data(), h_rhs.size() * sizeof(h_rhs[0]), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_out.ptr, h_out.data(), h_out.size() * sizeof(h_out[0]), hipMemcpyHostToDevice));

    dim3 grid((rows + 127) / 128, (cols + 127) / 128, 1);
    hipLaunchKernelGGL(q8_motif192_wmma_k2_realdata_fullk_accpark_store_kernel,
        grid,
        dim3(256, 1, 1),
        0,
        0,
        d_q8.ptr,
        d_rhs.ptr,
        d_out.ptr,
        k,
        rows,
        cols,
        wait_after_load);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_out.data(), d_out.ptr, h_out.size() * sizeof(h_out[0]), hipMemcpyDeviceToHost));

    const size_t total = h_out.size();
    const size_t max_checks = 8192;
    const size_t stride = total <= max_checks ? 1 : std::max<size_t>(1, total / max_checks);

    size_t checked = 0;
    size_t bad = 0;
    size_t nan = 0;
    size_t inf = 0;
    size_t sentinel = 0;
    double sq_err = 0.0;
    double sq_ref = 0.0;
    float max_abs = 0.0f;
    bool have_first_bad = false;
    int first_row = -1;
    int first_col = -1;
    float first_actual = 0.0f;
    float first_expected = 0.0f;
    float first_err = 0.0f;

    for (size_t index = 0; index < total; index += stride) {
        const int row = static_cast<int>(index % static_cast<size_t>(rows));
        const int col = static_cast<int>(index / static_cast<size_t>(rows));
        const float actual = h_out[index];
        const float expected = cpu_reference_value(h_q8, h_rhs, k, rows, cols, row, col);
        ++checked;
        if (actual == -7777.0f) {
            ++sentinel;
            ++bad;
            if (!have_first_bad) {
                have_first_bad = true;
                first_row = row;
                first_col = col;
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
                first_row = row;
                first_col = col;
                first_actual = actual;
                first_expected = expected;
                first_err = INFINITY;
            }
            continue;
        }
        if (std::isinf(actual)) {
            ++inf;
            ++bad;
            if (!have_first_bad) {
                have_first_bad = true;
                first_row = row;
                first_col = col;
                first_actual = actual;
                first_expected = expected;
                first_err = INFINITY;
            }
            continue;
        }
        const float err = std::fabs(actual - expected);
        max_abs = std::max(max_abs, err);
        sq_err += static_cast<double>(err) * static_cast<double>(err);
        sq_ref += static_cast<double>(expected) * static_cast<double>(expected);
        const float tol = 2.0f + 0.05f * std::fabs(expected);
        if (err > tol) {
            ++bad;
            if (!have_first_bad) {
                have_first_bad = true;
                first_row = row;
                first_col = col;
                first_actual = actual;
                first_expected = expected;
                first_err = err;
            }
        }
    }

    const double nmse = sq_ref > 0.0 ? sq_err / sq_ref : sq_err;
    std::printf(
        "%s rows=%d cols=%d k=%d active=%zu checked=%zu stride=%zu bad=%zu nan=%zu inf=%zu sentinel=%zu max_abs=%g nmse=%g\n",
        label,
        rows,
        cols,
        k,
        total,
        checked,
        stride,
        bad,
        nan,
        inf,
        sentinel,
        max_abs,
        nmse);
    if (have_first_bad) {
        std::printf(
            "  first_bad row=%d col=%d actual=%g expected=%g err=%g\n",
            first_row,
            first_col,
            first_actual,
            first_expected,
            first_err);
    }
    return bad == 0 ? 0 : 1;
}

static int run_motif192_wmma_k2_realdata_fullk_accpark_suite(const char * label, bool wait_after_load) {
    constexpr int k = 4096;
    int status = 0;
    status |= run_motif192_wmma_k2_realdata_fullk_accpark_case(label, 128, 128, k, wait_after_load);
    status |= run_motif192_wmma_k2_realdata_fullk_accpark_case(label, 128, 129, k, wait_after_load);
    status |= run_motif192_wmma_k2_realdata_fullk_accpark_case(label, 129, 128, k, wait_after_load);
    status |= run_motif192_wmma_k2_realdata_fullk_accpark_case(label, 1024, 512, k, wait_after_load);
    status |= run_motif192_wmma_k2_realdata_fullk_accpark_case(label, 4096, 512, k, wait_after_load);
    status |= run_motif192_wmma_k2_realdata_fullk_accpark_case(label, 4096, 513, k, wait_after_load);
    return status;
}

static int run_motif192_wmma_k2_realdata_fullk_accparkfull8_case(
        const char * label,
        int rows,
        int cols,
        int k,
        bool wait_after_load) {
    const int blocks_per_row = k / 32;
    std::vector<hrx_block_q8_0_wmma_vk128_lhs> h_q8(static_cast<size_t>(rows) * blocks_per_row);
    std::vector<float> h_rhs(static_cast<size_t>(cols) * k);
    std::vector<float> h_out(static_cast<size_t>(rows) * cols, -7777.0f);
    fill_q8_backend_like(h_q8, rows, blocks_per_row);
    fill_rhs_backend_like(h_rhs, k, cols);

    device_buffer<hrx_block_q8_0_wmma_vk128_lhs> d_q8(h_q8.size());
    device_buffer<float> d_rhs(h_rhs.size());
    device_buffer<float> d_out(h_out.size());
    HIP_CHECK(hipMemcpy(d_q8.ptr, h_q8.data(), h_q8.size() * sizeof(h_q8[0]), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_rhs.ptr, h_rhs.data(), h_rhs.size() * sizeof(h_rhs[0]), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_out.ptr, h_out.data(), h_out.size() * sizeof(h_out[0]), hipMemcpyHostToDevice));

    dim3 grid((rows + 127) / 128, (cols + 127) / 128, 1);
    hipLaunchKernelGGL(q8_motif192_wmma_k2_realdata_fullk_accparkfull8_store_kernel,
        grid,
        dim3(256, 1, 1),
        0,
        0,
        d_q8.ptr,
        d_rhs.ptr,
        d_out.ptr,
        k,
        rows,
        cols,
        wait_after_load);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_out.data(), d_out.ptr, h_out.size() * sizeof(h_out[0]), hipMemcpyDeviceToHost));

    const size_t total = h_out.size();
    const size_t max_checks = 8192;
    const size_t stride = total <= max_checks ? 1 : std::max<size_t>(1, total / max_checks);

    size_t checked = 0;
    size_t bad = 0;
    size_t nan = 0;
    size_t inf = 0;
    size_t sentinel = 0;
    double sq_err = 0.0;
    double sq_ref = 0.0;
    float max_abs = 0.0f;
    bool have_first_bad = false;
    int first_row = -1;
    int first_col = -1;
    float first_actual = 0.0f;
    float first_expected = 0.0f;
    float first_err = 0.0f;

    for (size_t index = 0; index < total; index += stride) {
        const int row = static_cast<int>(index % static_cast<size_t>(rows));
        const int col = static_cast<int>(index / static_cast<size_t>(rows));
        const float actual = h_out[index];
        const float expected = cpu_reference_value(h_q8, h_rhs, k, rows, cols, row, col);
        ++checked;
        if (actual == -7777.0f) {
            ++sentinel;
            ++bad;
            if (!have_first_bad) {
                have_first_bad = true;
                first_row = row;
                first_col = col;
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
                first_row = row;
                first_col = col;
                first_actual = actual;
                first_expected = expected;
                first_err = INFINITY;
            }
            continue;
        }
        if (std::isinf(actual)) {
            ++inf;
            ++bad;
            if (!have_first_bad) {
                have_first_bad = true;
                first_row = row;
                first_col = col;
                first_actual = actual;
                first_expected = expected;
                first_err = INFINITY;
            }
            continue;
        }
        const float err = std::fabs(actual - expected);
        max_abs = std::max(max_abs, err);
        sq_err += static_cast<double>(err) * static_cast<double>(err);
        sq_ref += static_cast<double>(expected) * static_cast<double>(expected);
        const float tol = 2.0f + 0.05f * std::fabs(expected);
        if (err > tol) {
            ++bad;
            if (!have_first_bad) {
                have_first_bad = true;
                first_row = row;
                first_col = col;
                first_actual = actual;
                first_expected = expected;
                first_err = err;
            }
        }
    }

    const double nmse = sq_ref > 0.0 ? sq_err / sq_ref : sq_err;
    std::printf(
        "%s rows=%d cols=%d k=%d active=%zu checked=%zu stride=%zu bad=%zu nan=%zu inf=%zu sentinel=%zu max_abs=%g nmse=%g\n",
        label,
        rows,
        cols,
        k,
        total,
        checked,
        stride,
        bad,
        nan,
        inf,
        sentinel,
        max_abs,
        nmse);
    if (have_first_bad) {
        std::printf(
            "  first_bad row=%d col=%d actual=%g expected=%g err=%g\n",
            first_row,
            first_col,
            first_actual,
            first_expected,
            first_err);
    }
    return bad == 0 ? 0 : 1;
}

static int run_motif192_wmma_k2_realdata_fullk_accparkfull8_suite(const char * label, bool wait_after_load) {
    constexpr int k = 4096;
    int status = 0;
    status |= run_motif192_wmma_k2_realdata_fullk_accparkfull8_case(label, 128, 128, k, wait_after_load);
    status |= run_motif192_wmma_k2_realdata_fullk_accparkfull8_case(label, 128, 129, k, wait_after_load);
    status |= run_motif192_wmma_k2_realdata_fullk_accparkfull8_case(label, 129, 128, k, wait_after_load);
    status |= run_motif192_wmma_k2_realdata_fullk_accparkfull8_case(label, 1024, 512, k, wait_after_load);
    status |= run_motif192_wmma_k2_realdata_fullk_accparkfull8_case(label, 4096, 512, k, wait_after_load);
    status |= run_motif192_wmma_k2_realdata_fullk_accparkfull8_case(label, 4096, 513, k, wait_after_load);
    return status;
}

static float time_fullk_launches(
        const hrx_block_q8_0_wmma_vk128_lhs * d_q8,
        const float * d_rhs,
        float * d_out,
        int rows,
        int cols,
        int k,
        int reps,
        bool phase8,
        bool wait_after_load) {
    dim3 grid((rows + 127) / 128, (cols + 127) / 128, 1);
    HIP_CHECK(hipDeviceSynchronize());
    const auto start = std::chrono::steady_clock::now();
    for (int rep = 0; rep < reps; ++rep) {
        if (phase8) {
            hipLaunchKernelGGL((q8_motif192_wmma_k2_realdata_fullk_phase8_store_kernel<0, 0>),
                grid, dim3(256, 1, 1), 0, 0, d_q8, d_rhs, d_out, k, rows, cols, wait_after_load);
            hipLaunchKernelGGL((q8_motif192_wmma_k2_realdata_fullk_phase8_store_kernel<8, 2>),
                grid, dim3(256, 1, 1), 0, 0, d_q8, d_rhs, d_out, k, rows, cols, wait_after_load);
        } else {
            hipLaunchKernelGGL((q8_motif192_wmma_k2_realdata_fullk_store_kernel<false>),
                grid, dim3(256, 1, 1), 0, 0, d_q8, d_rhs, d_out, k, rows, cols, wait_after_load);
        }
    }
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    const auto stop = std::chrono::steady_clock::now();
    const double elapsed_ms = std::chrono::duration<double, std::milli>(stop - start).count();
    return static_cast<float>(elapsed_ms / static_cast<double>(reps));
}

static int run_motif192_wmma_k2_realdata_fullk_timing_case(
        const char * label,
        int rows,
        int cols,
        int k,
        int reps,
        bool wait_after_load) {
    const int blocks_per_row = k / 32;
    std::vector<hrx_block_q8_0_wmma_vk128_lhs> h_q8(static_cast<size_t>(rows) * blocks_per_row);
    std::vector<float> h_rhs(static_cast<size_t>(cols) * k);
    std::vector<float> h_out(static_cast<size_t>(rows) * cols, -7777.0f);
    fill_q8_backend_like(h_q8, rows, blocks_per_row);
    fill_rhs_backend_like(h_rhs, k, cols);

    device_buffer<hrx_block_q8_0_wmma_vk128_lhs> d_q8(h_q8.size());
    device_buffer<float> d_rhs(h_rhs.size());
    device_buffer<float> d_out(h_out.size());
    HIP_CHECK(hipMemcpy(d_q8.ptr, h_q8.data(), h_q8.size() * sizeof(h_q8[0]), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_rhs.ptr, h_rhs.data(), h_rhs.size() * sizeof(h_rhs[0]), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_out.ptr, h_out.data(), h_out.size() * sizeof(h_out[0]), hipMemcpyHostToDevice));

    (void) time_fullk_launches(d_q8.ptr, d_rhs.ptr, d_out.ptr, rows, cols, k, 3, false, wait_after_load);
    (void) time_fullk_launches(d_q8.ptr, d_rhs.ptr, d_out.ptr, rows, cols, k, 3, true, wait_after_load);
    const float fullk_ms = time_fullk_launches(d_q8.ptr, d_rhs.ptr, d_out.ptr, rows, cols, k, reps, false, wait_after_load);
    const float phase8_ms = time_fullk_launches(d_q8.ptr, d_rhs.ptr, d_out.ptr, rows, cols, k, reps, true, wait_after_load);
    std::printf(
        "%s rows=%d cols=%d k=%d reps=%d fullk_ms=%g phase8_ms=%g phase8_over_fullk=%g\n",
        label,
        rows,
        cols,
        k,
        reps,
        fullk_ms,
        phase8_ms,
        phase8_ms / fullk_ms);
    return 0;
}

static int run_motif192_wmma_k2_realdata_fullk_timing_suite(const char * label, bool wait_after_load) {
    constexpr int k = 4096;
    int status = 0;
    status |= run_motif192_wmma_k2_realdata_fullk_timing_case(label, 128, 128, k, 100, wait_after_load);
    status |= run_motif192_wmma_k2_realdata_fullk_timing_case(label, 1024, 512, k, 30, wait_after_load);
    status |= run_motif192_wmma_k2_realdata_fullk_timing_case(label, 4096, 512, k, 20, wait_after_load);
    status |= run_motif192_wmma_k2_realdata_fullk_timing_case(label, 4096, 513, k, 20, wait_after_load);
    return status;
}

static float time_phase8seq_launches(
        const hrx_block_q8_0_wmma_vk128_lhs * d_q8,
        const float * d_rhs,
        float * d_out,
        int rows,
        int cols,
        int k,
        int reps,
        bool wait_after_load) {
    dim3 grid((rows + 127) / 128, (cols + 127) / 128, 1);
    HIP_CHECK(hipDeviceSynchronize());
    const auto start = std::chrono::steady_clock::now();
    for (int rep = 0; rep < reps; ++rep) {
        hipLaunchKernelGGL(q8_motif192_wmma_k2_realdata_fullk_phase8seq_store_kernel,
            grid, dim3(256, 1, 1), 0, 0, d_q8, d_rhs, d_out, k, rows, cols, wait_after_load);
    }
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    const auto stop = std::chrono::steady_clock::now();
    const double elapsed_ms = std::chrono::duration<double, std::milli>(stop - start).count();
    return static_cast<float>(elapsed_ms / static_cast<double>(reps));
}

static int run_motif192_wmma_k2_realdata_fullk_phase8seq_timing_case(
        const char * label,
        int rows,
        int cols,
        int k,
        int reps,
        bool wait_after_load) {
    const int blocks_per_row = k / 32;
    std::vector<hrx_block_q8_0_wmma_vk128_lhs> h_q8(static_cast<size_t>(rows) * blocks_per_row);
    std::vector<float> h_rhs(static_cast<size_t>(cols) * k);
    std::vector<float> h_out(static_cast<size_t>(rows) * cols, -7777.0f);
    fill_q8_backend_like(h_q8, rows, blocks_per_row);
    fill_rhs_backend_like(h_rhs, k, cols);

    device_buffer<hrx_block_q8_0_wmma_vk128_lhs> d_q8(h_q8.size());
    device_buffer<float> d_rhs(h_rhs.size());
    device_buffer<float> d_out(h_out.size());
    HIP_CHECK(hipMemcpy(d_q8.ptr, h_q8.data(), h_q8.size() * sizeof(h_q8[0]), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_rhs.ptr, h_rhs.data(), h_rhs.size() * sizeof(h_rhs[0]), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_out.ptr, h_out.data(), h_out.size() * sizeof(h_out[0]), hipMemcpyHostToDevice));

    (void) time_fullk_launches(d_q8.ptr, d_rhs.ptr, d_out.ptr, rows, cols, k, 3, false, wait_after_load);
    (void) time_phase8seq_launches(d_q8.ptr, d_rhs.ptr, d_out.ptr, rows, cols, k, 3, wait_after_load);
    const float fullk_ms = time_fullk_launches(d_q8.ptr, d_rhs.ptr, d_out.ptr, rows, cols, k, reps, false, wait_after_load);
    const float phase8seq_ms = time_phase8seq_launches(d_q8.ptr, d_rhs.ptr, d_out.ptr, rows, cols, k, reps, wait_after_load);
    std::printf(
        "%s rows=%d cols=%d k=%d reps=%d fullk_ms=%g phase8seq_ms=%g phase8seq_over_fullk=%g\n",
        label,
        rows,
        cols,
        k,
        reps,
        fullk_ms,
        phase8seq_ms,
        phase8seq_ms / fullk_ms);
    return 0;
}

static int run_motif192_wmma_k2_realdata_fullk_phase8seq_timing_suite(const char * label, bool wait_after_load) {
    constexpr int k = 4096;
    int status = 0;
    status |= run_motif192_wmma_k2_realdata_fullk_phase8seq_timing_case(label, 128, 128, k, 100, wait_after_load);
    status |= run_motif192_wmma_k2_realdata_fullk_phase8seq_timing_case(label, 1024, 512, k, 30, wait_after_load);
    status |= run_motif192_wmma_k2_realdata_fullk_phase8seq_timing_case(label, 4096, 512, k, 20, wait_after_load);
    status |= run_motif192_wmma_k2_realdata_fullk_phase8seq_timing_case(label, 4096, 513, k, 20, wait_after_load);
    return status;
}

static float time_streamfrag_launches(
        const hrx_block_q8_0_wmma_vk128_lhs * d_q8,
        const float * d_rhs,
        float * d_out,
        int rows,
        int cols,
        int k,
        int reps,
        bool wait_after_load) {
    dim3 grid((rows + 127) / 128, (cols + 127) / 128, 1);
    HIP_CHECK(hipDeviceSynchronize());
    const auto start = std::chrono::steady_clock::now();
    for (int rep = 0; rep < reps; ++rep) {
        hipLaunchKernelGGL(q8_motif192_wmma_k2_realdata_fullk_streamfrag_store_kernel,
            grid, dim3(256, 1, 1), 0, 0, d_q8, d_rhs, d_out, k, rows, cols, wait_after_load);
    }
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    const auto stop = std::chrono::steady_clock::now();
    const double elapsed_ms = std::chrono::duration<double, std::milli>(stop - start).count();
    return static_cast<float>(elapsed_ms / static_cast<double>(reps));
}

static int run_motif192_wmma_k2_realdata_fullk_streamfrag_timing_case(
        const char * label,
        int rows,
        int cols,
        int k,
        int reps,
        bool wait_after_load) {
    const int blocks_per_row = k / 32;
    std::vector<hrx_block_q8_0_wmma_vk128_lhs> h_q8(static_cast<size_t>(rows) * blocks_per_row);
    std::vector<float> h_rhs(static_cast<size_t>(cols) * k);
    std::vector<float> h_out(static_cast<size_t>(rows) * cols, -7777.0f);
    fill_q8_backend_like(h_q8, rows, blocks_per_row);
    fill_rhs_backend_like(h_rhs, k, cols);

    device_buffer<hrx_block_q8_0_wmma_vk128_lhs> d_q8(h_q8.size());
    device_buffer<float> d_rhs(h_rhs.size());
    device_buffer<float> d_out(h_out.size());
    HIP_CHECK(hipMemcpy(d_q8.ptr, h_q8.data(), h_q8.size() * sizeof(h_q8[0]), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_rhs.ptr, h_rhs.data(), h_rhs.size() * sizeof(h_rhs[0]), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_out.ptr, h_out.data(), h_out.size() * sizeof(h_out[0]), hipMemcpyHostToDevice));

    (void) time_fullk_launches(d_q8.ptr, d_rhs.ptr, d_out.ptr, rows, cols, k, 3, false, wait_after_load);
    (void) time_streamfrag_launches(d_q8.ptr, d_rhs.ptr, d_out.ptr, rows, cols, k, 3, wait_after_load);
    const float fullk_ms = time_fullk_launches(d_q8.ptr, d_rhs.ptr, d_out.ptr, rows, cols, k, reps, false, wait_after_load);
    const float streamfrag_ms = time_streamfrag_launches(d_q8.ptr, d_rhs.ptr, d_out.ptr, rows, cols, k, reps, wait_after_load);
    std::printf(
        "%s rows=%d cols=%d k=%d reps=%d fullk_ms=%g streamfrag_ms=%g streamfrag_over_fullk=%g\n",
        label,
        rows,
        cols,
        k,
        reps,
        fullk_ms,
        streamfrag_ms,
        streamfrag_ms / fullk_ms);
    return 0;
}

static int run_motif192_wmma_k2_realdata_fullk_streamfrag_timing_suite(const char * label, bool wait_after_load) {
    constexpr int k = 4096;
    int status = 0;
    status |= run_motif192_wmma_k2_realdata_fullk_streamfrag_timing_case(label, 128, 128, k, 100, wait_after_load);
    status |= run_motif192_wmma_k2_realdata_fullk_streamfrag_timing_case(label, 1024, 512, k, 30, wait_after_load);
    status |= run_motif192_wmma_k2_realdata_fullk_streamfrag_timing_case(label, 4096, 512, k, 20, wait_after_load);
    status |= run_motif192_wmma_k2_realdata_fullk_streamfrag_timing_case(label, 4096, 513, k, 20, wait_after_load);
    return status;
}

static float time_ktilefrag_launches(
        const hrx_block_q8_0_wmma_vk128_lhs * d_q8,
        const float * d_rhs,
        float * d_out,
        int rows,
        int cols,
        int k,
        int reps,
        bool wait_after_load,
        bool storebatch = false,
        bool storebatch4 = false) {
    dim3 grid((rows + 127) / 128, (cols + 127) / 128, 1);
    HIP_CHECK(hipDeviceSynchronize());
    const auto start = std::chrono::steady_clock::now();
    for (int rep = 0; rep < reps; ++rep) {
        if (storebatch4) {
            hipLaunchKernelGGL(q8_motif192_wmma_k2_realdata_fullk_ktilefrag_storebatch4_kernel,
                grid, dim3(256, 1, 1), 0, 0, d_q8, d_rhs, d_out, k, rows, cols, wait_after_load);
        } else if (storebatch) {
            hipLaunchKernelGGL(q8_motif192_wmma_k2_realdata_fullk_ktilefrag_storebatch_kernel,
                grid, dim3(256, 1, 1), 0, 0, d_q8, d_rhs, d_out, k, rows, cols, wait_after_load);
        } else {
            hipLaunchKernelGGL(q8_motif192_wmma_k2_realdata_fullk_ktilefrag_store_kernel,
                grid, dim3(256, 1, 1), 0, 0, d_q8, d_rhs, d_out, k, rows, cols, wait_after_load);
        }
    }
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    const auto stop = std::chrono::steady_clock::now();
    const double elapsed_ms = std::chrono::duration<double, std::milli>(stop - start).count();
    return static_cast<float>(elapsed_ms / static_cast<double>(reps));
}

static int run_motif192_wmma_k2_realdata_fullk_ktilefrag_timing_case(
        const char * label,
        int rows,
        int cols,
        int k,
        int reps,
        bool wait_after_load,
        bool storebatch = false,
        bool storebatch4 = false) {
    const int blocks_per_row = k / 32;
    std::vector<hrx_block_q8_0_wmma_vk128_lhs> h_q8(static_cast<size_t>(rows) * blocks_per_row);
    std::vector<float> h_rhs(static_cast<size_t>(cols) * k);
    std::vector<float> h_out(static_cast<size_t>(rows) * cols, -7777.0f);
    fill_q8_backend_like(h_q8, rows, blocks_per_row);
    fill_rhs_backend_like(h_rhs, k, cols);

    device_buffer<hrx_block_q8_0_wmma_vk128_lhs> d_q8(h_q8.size());
    device_buffer<float> d_rhs(h_rhs.size());
    device_buffer<float> d_out(h_out.size());
    HIP_CHECK(hipMemcpy(d_q8.ptr, h_q8.data(), h_q8.size() * sizeof(h_q8[0]), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_rhs.ptr, h_rhs.data(), h_rhs.size() * sizeof(h_rhs[0]), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_out.ptr, h_out.data(), h_out.size() * sizeof(h_out[0]), hipMemcpyHostToDevice));

    (void) time_fullk_launches(d_q8.ptr, d_rhs.ptr, d_out.ptr, rows, cols, k, 3, false, wait_after_load);
    (void) time_ktilefrag_launches(d_q8.ptr, d_rhs.ptr, d_out.ptr, rows, cols, k, 3, wait_after_load, storebatch, storebatch4);
    const float fullk_ms = time_fullk_launches(d_q8.ptr, d_rhs.ptr, d_out.ptr, rows, cols, k, reps, false, wait_after_load);
    const float ktilefrag_ms = time_ktilefrag_launches(d_q8.ptr, d_rhs.ptr, d_out.ptr, rows, cols, k, reps, wait_after_load, storebatch, storebatch4);
    if (storebatch4) {
        std::printf(
            "%s rows=%d cols=%d k=%d reps=%d fullk_ms=%g ktilefrag_storebatch4_ms=%g ktilefrag_storebatch4_over_fullk=%g\n",
            label,
            rows,
            cols,
            k,
            reps,
            fullk_ms,
            ktilefrag_ms,
            ktilefrag_ms / fullk_ms);
    } else if (storebatch) {
        std::printf(
            "%s rows=%d cols=%d k=%d reps=%d fullk_ms=%g ktilefrag_storebatch_ms=%g ktilefrag_storebatch_over_fullk=%g\n",
            label,
            rows,
            cols,
            k,
            reps,
            fullk_ms,
            ktilefrag_ms,
            ktilefrag_ms / fullk_ms);
    } else {
        std::printf(
            "%s rows=%d cols=%d k=%d reps=%d fullk_ms=%g ktilefrag_ms=%g ktilefrag_over_fullk=%g\n",
            label,
            rows,
            cols,
            k,
            reps,
            fullk_ms,
            ktilefrag_ms,
            ktilefrag_ms / fullk_ms);
    }
    return 0;
}

static int run_motif192_wmma_k2_realdata_fullk_ktilefrag_timing_suite(
        const char * label,
        bool wait_after_load,
        bool storebatch = false,
        bool storebatch4 = false) {
    constexpr int k = 4096;
    int status = 0;
    status |= run_motif192_wmma_k2_realdata_fullk_ktilefrag_timing_case(label, 128, 128, k, 100, wait_after_load, storebatch, storebatch4);
    status |= run_motif192_wmma_k2_realdata_fullk_ktilefrag_timing_case(label, 1024, 512, k, 30, wait_after_load, storebatch, storebatch4);
    status |= run_motif192_wmma_k2_realdata_fullk_ktilefrag_timing_case(label, 4096, 512, k, 20, wait_after_load, storebatch, storebatch4);
    status |= run_motif192_wmma_k2_realdata_fullk_ktilefrag_timing_case(label, 4096, 513, k, 20, wait_after_load, storebatch, storebatch4);
    return status;
}

static float time_colpairfrag_launches(
        const hrx_block_q8_0_wmma_vk128_lhs * d_q8,
        const float * d_rhs,
        float * d_out,
        int rows,
        int cols,
        int k,
        int reps,
        bool wait_after_load) {
    dim3 grid((rows + 127) / 128, (cols + 127) / 128, 1);
    HIP_CHECK(hipDeviceSynchronize());
    const auto start = std::chrono::steady_clock::now();
    for (int rep = 0; rep < reps; ++rep) {
        hipLaunchKernelGGL(q8_motif192_wmma_k2_realdata_fullk_colpairfrag_store_kernel,
            grid, dim3(256, 1, 1), 0, 0, d_q8, d_rhs, d_out, k, rows, cols, wait_after_load);
    }
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    const auto stop = std::chrono::steady_clock::now();
    const double elapsed_ms = std::chrono::duration<double, std::milli>(stop - start).count();
    return static_cast<float>(elapsed_ms / static_cast<double>(reps));
}

static int run_motif192_wmma_k2_realdata_fullk_colpairfrag_timing_case(
        const char * label,
        int rows,
        int cols,
        int k,
        int reps,
        bool wait_after_load) {
    const int blocks_per_row = k / 32;
    std::vector<hrx_block_q8_0_wmma_vk128_lhs> h_q8(static_cast<size_t>(rows) * blocks_per_row);
    std::vector<float> h_rhs(static_cast<size_t>(cols) * k);
    std::vector<float> h_out(static_cast<size_t>(rows) * cols, -7777.0f);
    fill_q8_backend_like(h_q8, rows, blocks_per_row);
    fill_rhs_backend_like(h_rhs, k, cols);

    device_buffer<hrx_block_q8_0_wmma_vk128_lhs> d_q8(h_q8.size());
    device_buffer<float> d_rhs(h_rhs.size());
    device_buffer<float> d_out(h_out.size());
    HIP_CHECK(hipMemcpy(d_q8.ptr, h_q8.data(), h_q8.size() * sizeof(h_q8[0]), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_rhs.ptr, h_rhs.data(), h_rhs.size() * sizeof(h_rhs[0]), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_out.ptr, h_out.data(), h_out.size() * sizeof(h_out[0]), hipMemcpyHostToDevice));

    (void) time_fullk_launches(d_q8.ptr, d_rhs.ptr, d_out.ptr, rows, cols, k, 3, false, wait_after_load);
    (void) time_colpairfrag_launches(d_q8.ptr, d_rhs.ptr, d_out.ptr, rows, cols, k, 3, wait_after_load);
    const float fullk_ms = time_fullk_launches(d_q8.ptr, d_rhs.ptr, d_out.ptr, rows, cols, k, reps, false, wait_after_load);
    const float colpairfrag_ms = time_colpairfrag_launches(d_q8.ptr, d_rhs.ptr, d_out.ptr, rows, cols, k, reps, wait_after_load);
    std::printf(
        "%s rows=%d cols=%d k=%d reps=%d fullk_ms=%g colpairfrag_ms=%g colpairfrag_over_fullk=%g\n",
        label,
        rows,
        cols,
        k,
        reps,
        fullk_ms,
        colpairfrag_ms,
        colpairfrag_ms / fullk_ms);
    return 0;
}

static int run_motif192_wmma_k2_realdata_fullk_colpairfrag_timing_suite(const char * label, bool wait_after_load) {
    constexpr int k = 4096;
    int status = 0;
    status |= run_motif192_wmma_k2_realdata_fullk_colpairfrag_timing_case(label, 128, 128, k, 100, wait_after_load);
    status |= run_motif192_wmma_k2_realdata_fullk_colpairfrag_timing_case(label, 1024, 512, k, 30, wait_after_load);
    status |= run_motif192_wmma_k2_realdata_fullk_colpairfrag_timing_case(label, 4096, 512, k, 20, wait_after_load);
    status |= run_motif192_wmma_k2_realdata_fullk_colpairfrag_timing_case(label, 4096, 513, k, 20, wait_after_load);
    return status;
}

static float time_accpark_launches(
        const hrx_block_q8_0_wmma_vk128_lhs * d_q8,
        const float * d_rhs,
        float * d_out,
        int rows,
        int cols,
        int k,
        int reps,
        bool wait_after_load) {
    dim3 grid((rows + 127) / 128, (cols + 127) / 128, 1);
    HIP_CHECK(hipDeviceSynchronize());
    const auto start = std::chrono::steady_clock::now();
    for (int rep = 0; rep < reps; ++rep) {
        hipLaunchKernelGGL(q8_motif192_wmma_k2_realdata_fullk_accpark_store_kernel,
            grid, dim3(256, 1, 1), 0, 0, d_q8, d_rhs, d_out, k, rows, cols, wait_after_load);
    }
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    const auto stop = std::chrono::steady_clock::now();
    const double elapsed_ms = std::chrono::duration<double, std::milli>(stop - start).count();
    return static_cast<float>(elapsed_ms / static_cast<double>(reps));
}

static int run_motif192_wmma_k2_realdata_fullk_accpark_timing_case(
        const char * label,
        int rows,
        int cols,
        int k,
        int reps,
        bool wait_after_load) {
    const int blocks_per_row = k / 32;
    std::vector<hrx_block_q8_0_wmma_vk128_lhs> h_q8(static_cast<size_t>(rows) * blocks_per_row);
    std::vector<float> h_rhs(static_cast<size_t>(cols) * k);
    std::vector<float> h_out(static_cast<size_t>(rows) * cols, -7777.0f);
    fill_q8_backend_like(h_q8, rows, blocks_per_row);
    fill_rhs_backend_like(h_rhs, k, cols);

    device_buffer<hrx_block_q8_0_wmma_vk128_lhs> d_q8(h_q8.size());
    device_buffer<float> d_rhs(h_rhs.size());
    device_buffer<float> d_out(h_out.size());
    HIP_CHECK(hipMemcpy(d_q8.ptr, h_q8.data(), h_q8.size() * sizeof(h_q8[0]), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_rhs.ptr, h_rhs.data(), h_rhs.size() * sizeof(h_rhs[0]), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_out.ptr, h_out.data(), h_out.size() * sizeof(h_out[0]), hipMemcpyHostToDevice));

    (void) time_fullk_launches(d_q8.ptr, d_rhs.ptr, d_out.ptr, rows, cols, k, 3, false, wait_after_load);
    (void) time_accpark_launches(d_q8.ptr, d_rhs.ptr, d_out.ptr, rows, cols, k, 3, wait_after_load);
    const float fullk_ms = time_fullk_launches(d_q8.ptr, d_rhs.ptr, d_out.ptr, rows, cols, k, reps, false, wait_after_load);
    const float accpark_ms = time_accpark_launches(d_q8.ptr, d_rhs.ptr, d_out.ptr, rows, cols, k, reps, wait_after_load);
    std::printf(
        "%s rows=%d cols=%d k=%d reps=%d fullk_ms=%g accpark_ms=%g accpark_over_fullk=%g\n",
        label,
        rows,
        cols,
        k,
        reps,
        fullk_ms,
        accpark_ms,
        accpark_ms / fullk_ms);
    return 0;
}

static int run_motif192_wmma_k2_realdata_fullk_accpark_timing_suite(const char * label, bool wait_after_load) {
    constexpr int k = 4096;
    int status = 0;
    status |= run_motif192_wmma_k2_realdata_fullk_accpark_timing_case(label, 128, 128, k, 100, wait_after_load);
    status |= run_motif192_wmma_k2_realdata_fullk_accpark_timing_case(label, 1024, 512, k, 30, wait_after_load);
    status |= run_motif192_wmma_k2_realdata_fullk_accpark_timing_case(label, 4096, 512, k, 20, wait_after_load);
    status |= run_motif192_wmma_k2_realdata_fullk_accpark_timing_case(label, 4096, 513, k, 20, wait_after_load);
    return status;
}

static float time_accparkfull8_launches(
        const hrx_block_q8_0_wmma_vk128_lhs * d_q8,
        const float * d_rhs,
        float * d_out,
        int rows,
        int cols,
        int k,
        int reps,
        bool wait_after_load) {
    dim3 grid((rows + 127) / 128, (cols + 127) / 128, 1);
    HIP_CHECK(hipDeviceSynchronize());
    const auto start = std::chrono::steady_clock::now();
    for (int rep = 0; rep < reps; ++rep) {
        hipLaunchKernelGGL(q8_motif192_wmma_k2_realdata_fullk_accparkfull8_store_kernel,
            grid, dim3(256, 1, 1), 0, 0, d_q8, d_rhs, d_out, k, rows, cols, wait_after_load);
    }
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    const auto stop = std::chrono::steady_clock::now();
    const double elapsed_ms = std::chrono::duration<double, std::milli>(stop - start).count();
    return static_cast<float>(elapsed_ms / static_cast<double>(reps));
}

static int run_motif192_wmma_k2_realdata_fullk_accparkfull8_timing_case(
        const char * label,
        int rows,
        int cols,
        int k,
        int reps,
        bool wait_after_load) {
    const int blocks_per_row = k / 32;
    std::vector<hrx_block_q8_0_wmma_vk128_lhs> h_q8(static_cast<size_t>(rows) * blocks_per_row);
    std::vector<float> h_rhs(static_cast<size_t>(cols) * k);
    std::vector<float> h_out(static_cast<size_t>(rows) * cols, -7777.0f);
    fill_q8_backend_like(h_q8, rows, blocks_per_row);
    fill_rhs_backend_like(h_rhs, k, cols);

    device_buffer<hrx_block_q8_0_wmma_vk128_lhs> d_q8(h_q8.size());
    device_buffer<float> d_rhs(h_rhs.size());
    device_buffer<float> d_out(h_out.size());
    HIP_CHECK(hipMemcpy(d_q8.ptr, h_q8.data(), h_q8.size() * sizeof(h_q8[0]), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_rhs.ptr, h_rhs.data(), h_rhs.size() * sizeof(h_rhs[0]), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_out.ptr, h_out.data(), h_out.size() * sizeof(h_out[0]), hipMemcpyHostToDevice));

    (void) time_fullk_launches(d_q8.ptr, d_rhs.ptr, d_out.ptr, rows, cols, k, 3, false, wait_after_load);
    (void) time_accparkfull8_launches(d_q8.ptr, d_rhs.ptr, d_out.ptr, rows, cols, k, 3, wait_after_load);
    const float fullk_ms = time_fullk_launches(d_q8.ptr, d_rhs.ptr, d_out.ptr, rows, cols, k, reps, false, wait_after_load);
    const float accparkfull8_ms = time_accparkfull8_launches(d_q8.ptr, d_rhs.ptr, d_out.ptr, rows, cols, k, reps, wait_after_load);
    std::printf(
        "%s rows=%d cols=%d k=%d reps=%d fullk_ms=%g accparkfull8_ms=%g accparkfull8_over_fullk=%g\n",
        label,
        rows,
        cols,
        k,
        reps,
        fullk_ms,
        accparkfull8_ms,
        accparkfull8_ms / fullk_ms);
    return 0;
}

static int run_motif192_wmma_k2_realdata_fullk_accparkfull8_timing_suite(const char * label, bool wait_after_load) {
    constexpr int k = 4096;
    int status = 0;
    status |= run_motif192_wmma_k2_realdata_fullk_accparkfull8_timing_case(label, 128, 128, k, 100, wait_after_load);
    status |= run_motif192_wmma_k2_realdata_fullk_accparkfull8_timing_case(label, 1024, 512, k, 30, wait_after_load);
    status |= run_motif192_wmma_k2_realdata_fullk_accparkfull8_timing_case(label, 4096, 512, k, 20, wait_after_load);
    status |= run_motif192_wmma_k2_realdata_fullk_accparkfull8_timing_case(label, 4096, 513, k, 20, wait_after_load);
    return status;
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
    if (mode == "single-group13-bcopy-stage-selected" ||
            mode == "single-group13-abcopy-stage-selected") {
        *group_id = 13;
        return true;
    }
    if (mode == "single-group14-bcopy-stage-selected" ||
            mode == "single-group14-abcopy-stage-selected") {
        *group_id = 14;
        return true;
    }
    if (mode == "single-group15-bcopy-stage-selected" ||
            mode == "single-group15-abcopy-stage-selected") {
        *group_id = 15;
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
            mode == "bm128-direct192-raw-output" ||
            mode == "bm128-direct192-raw-asm-output" ||
            mode == "bm128-direct192-raw-asm-inout-output" ||
            mode == "bm128-streamk-raw-output" ||
            mode == "bm128-streamk-raw-asm-output" ||
            mode == "bm128-streamk-raw-asm-inout-output" ||
            mode == "bm128-streamk-prefetch51-raw-asm-output" ||
            mode == "bm128-streamk-abcopy-output" ||
            mode == "phase96-bm128-raw" ||
            mode == "phase96-bm128-raw-asm" ||
            mode == "phase96-bm128-acopy" ||
            mode == "phase96-bm128-bcopy" ||
            mode == "phase96-bm128-abcopy" ||
            mode == "phase96-bm128-abcopy-backendlike" ||
            mode == "phase96-bm128-abcopy-asm" ||
            mode == "phase96-bm128-abcopy-asm-backendlike" ||
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

static float expected_value_for_output_sample(
        size_t index,
        int rows,
        int cols,
        const std::string & mode,
        const std::vector<hrx_block_q8_0_wmma_vk128_lhs> & blocks,
        const std::vector<float> & rhs,
        int k) {
    int compute_group = 0;
    int store_group = 0;
    int row = static_cast<int>(index % static_cast<size_t>(rows));
    int col = static_cast<int>(index / static_cast<size_t>(rows));

    if (remap_mode_groups(mode, &compute_group, &store_group) ||
            bmirror_mode_groups(mode, &compute_group, &store_group)) {
        const int row_tile_base = row & ~63;
        const int col_tile_base = col & ~63;
        const int store_row_sub = store_group & 3;
        const int store_col_sub = (store_group >> 2) & 3;
        const int compute_row_sub = compute_group & 3;
        const int compute_col_sub = (compute_group >> 2) & 3;
        const int row_inner = (row & 63) - store_row_sub * 16;
        const int col_inner = (col & 63) - store_col_sub * 16;
        row = row_tile_base + compute_row_sub * 16 + row_inner;
        col = col_tile_base + compute_col_sub * 16 + col_inner;
        if (row < 0 || row >= rows || col < 0 || col >= cols) {
            return 0.0f;
        }
    }

    return cpu_reference_value(blocks, rhs, k, rows, cols, row, col);
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
    const bool backend_like =
        mode == "phase96-bm128-abcopy-backendlike" ||
        mode == "phase96-bm128-abcopy-asm-backendlike";
    if (backend_like) {
        fill_q8_backend_like(h_q8, rows, blocks_per_row);
        fill_rhs_backend_like(h_rhs, k, cols);
    } else {
        fill_q8(h_q8, rows, blocks_per_row);
        fill_rhs(h_rhs, k, cols);
    }
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
    if (mode == "contract-direct192-raw") {
        hipLaunchKernelGGL((q8_contract_direct192_repro_kernel<false, false>), grid, dim3(256, 1, 1), 0, 0,
            d_q8.ptr, d_rhs.ptr, d_contract.ptr, k, rows, cols);
    } else if (mode == "contract-direct192-bcopy") {
        hipLaunchKernelGGL((q8_contract_direct192_repro_kernel<false, true>), grid, dim3(256, 1, 1), 0, 0,
            d_q8.ptr, d_rhs.ptr, d_contract.ptr, k, rows, cols);
    } else if (mode == "contract-direct192-bcopy-hoist") {
        hipLaunchKernelGGL((q8_contract_direct192_repro_kernel<false, true, true>), grid, dim3(256, 1, 1), 0, 0,
            d_q8.ptr, d_rhs.ptr, d_contract.ptr, k, rows, cols);
    } else if (mode == "contract-direct192-abcopy") {
        hipLaunchKernelGGL((q8_contract_direct192_repro_kernel<true, true>), grid, dim3(256, 1, 1), 0, 0,
            d_q8.ptr, d_rhs.ptr, d_contract.ptr, k, rows, cols);
    } else if (mode == "contract-direct192-abcopy-bhoist") {
        hipLaunchKernelGGL((q8_contract_direct192_repro_kernel<true, true, true>), grid, dim3(256, 1, 1), 0, 0,
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

static int run_bm128_contract_case(const std::string & mode, int rows, int cols, int k) {
    const int blocks_per_row = k / 32;
    std::vector<hrx_block_q8_0_wmma_vk128_lhs> h_q8(static_cast<size_t>(rows) * blocks_per_row);
    std::vector<float> h_rhs(static_cast<size_t>(cols) * k);
    std::vector<float> h_contract(Q8_REPRO_BM128_CONTRACT_VALUES, -7777.0f);
    fill_q8(h_q8, rows, blocks_per_row);
    fill_rhs(h_rhs, k, cols);
    const std::vector<float> ref = cpu_reference(h_q8, h_rhs, k, rows, cols);

    device_buffer<hrx_block_q8_0_wmma_vk128_lhs> d_q8(h_q8.size());
    device_buffer<float> d_rhs(h_rhs.size());
    device_buffer<float> d_contract(h_contract.size());
    HIP_CHECK(hipMemcpy(d_q8.ptr, h_q8.data(), h_q8.size() * sizeof(h_q8[0]), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_rhs.ptr, h_rhs.data(), h_rhs.size() * sizeof(h_rhs[0]), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_contract.ptr, h_contract.data(), h_contract.size() * sizeof(h_contract[0]), hipMemcpyHostToDevice));

    dim3 grid((rows + 127) / 128, (cols + 127) / 128, 1);
    if (mode == "contract-bm128-direct192-raw") {
        hipLaunchKernelGGL((q8_contract_bm128_direct192_repro_kernel<false, false>),
            grid, dim3(256, 1, 1), 0, 0, d_q8.ptr, d_rhs.ptr, d_contract.ptr, k, rows, cols);
    } else if (mode == "contract-bm128-direct192-raw-asm") {
        hipLaunchKernelGGL((q8_contract_bm128_direct192_repro_kernel<false, false, false, 0, true>),
            grid, dim3(256, 1, 1), 0, 0, d_q8.ptr, d_rhs.ptr, d_contract.ptr, k, rows, cols);
    } else if (mode == "contract-bm128-direct192-bcopy-upper") {
        hipLaunchKernelGGL((q8_contract_bm128_direct192_repro_kernel<false, true, false, 2>),
            grid, dim3(256, 1, 1), 0, 0, d_q8.ptr, d_rhs.ptr, d_contract.ptr, k, rows, cols);
    } else if (mode == "contract-bm128-direct192-bcopy-upper-asm") {
        hipLaunchKernelGGL((q8_contract_bm128_direct192_repro_kernel<false, true, false, 2, true>),
            grid, dim3(256, 1, 1), 0, 0, d_q8.ptr, d_rhs.ptr, d_contract.ptr, k, rows, cols);
    } else if (mode == "contract-bm128-direct192-bcopy-upper-hoist") {
        hipLaunchKernelGGL((q8_contract_bm128_direct192_repro_kernel<false, true, true, 2>),
            grid, dim3(256, 1, 1), 0, 0, d_q8.ptr, d_rhs.ptr, d_contract.ptr, k, rows, cols);
    } else if (mode == "contract-bm128-direct192-abcopy") {
        hipLaunchKernelGGL((q8_contract_bm128_direct192_repro_kernel<true, true>),
            grid, dim3(256, 1, 1), 0, 0, d_q8.ptr, d_rhs.ptr, d_contract.ptr, k, rows, cols);
    } else if (mode == "contract-bm128-direct192-abcopy-bhoist") {
        hipLaunchKernelGGL((q8_contract_bm128_direct192_repro_kernel<true, true, true>),
            grid, dim3(256, 1, 1), 0, 0, d_q8.ptr, d_rhs.ptr, d_contract.ptr, k, rows, cols);
    } else if (mode == "contract-bm128-direct192-abcopy-bhoist-asm") {
        hipLaunchKernelGGL((q8_contract_bm128_direct192_repro_kernel<true, true, true, 0, true>),
            grid, dim3(256, 1, 1), 0, 0, d_q8.ptr, d_rhs.ptr, d_contract.ptr, k, rows, cols);
    } else {
        std::fprintf(stderr, "unknown BM128 contract mode: %s\n", mode.c_str());
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

    for (int group = 0; group < Q8_REPRO_BM128_CONTRACT_GROUPS; ++group) {
        for (int slot = 0; slot < Q8_REPRO_CONTRACT_SLOTS; ++slot) {
            for (int lane = 0; lane < Q8_REPRO_CONTRACT_LANES; ++lane) {
                const int index = q8_repro_contract_index(group, slot, static_cast<unsigned int>(lane));
                const float actual = h_contract[static_cast<size_t>(index)];
                bool should_be_active = true;
                float expected = 0.0f;
                if (group < Q8_REPRO_BM128_CONTRACT_ACTIVE_GROUPS) {
                    const int wave = group >> 4;
                    const int local_group = group & 15;
                    const int wave_row = wave & 1;
                    const int wave_col = wave >> 1;
                    const int row = wave_row * 64 + (local_group & 3) * 16 + (lane >> 4) + slot * 4;
                    const int col = wave_col * 64 + ((local_group >> 2) & 3) * 16 + (lane & 15);
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
                const float threshold = group < Q8_REPRO_BM128_CONTRACT_ACTIVE_GROUPS ? 0.25f : 0.0f;
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

static int run_case(const std::string & mode, int rows, int cols, int k, size_t sample_stride = 0) {
    const int blocks_per_row = k / 32;
    std::vector<hrx_block_q8_0_wmma_vk128_lhs> h_q8(static_cast<size_t>(rows) * blocks_per_row);
    std::vector<float> h_rhs(static_cast<size_t>(cols) * k);
    std::vector<float> h_out(static_cast<size_t>(rows) * cols, -7777.0f);
    fill_q8(h_q8, rows, blocks_per_row);
    fill_rhs(h_rhs, k, cols);
    std::vector<float> ref;
    if (sample_stride == 0) {
        ref = cpu_reference(h_q8, h_rhs, k, rows, cols);
    }

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
    } else if (mode == "bm128-direct192-raw-output") {
        dim3 bm128_grid((rows + 127) / 128, (cols + 127) / 128, 1);
        hipLaunchKernelGGL((q8_bm128_direct192_output_repro_kernel<false>),
            bm128_grid, dim3(256, 1, 1), 0, 0, d_q8.ptr, d_rhs.ptr, d_out.ptr, k, rows, cols);
    } else if (mode == "bm128-direct192-raw-asm-output") {
        dim3 bm128_grid((rows + 127) / 128, (cols + 127) / 128, 1);
        hipLaunchKernelGGL((q8_bm128_direct192_output_repro_kernel<true>),
            bm128_grid, dim3(256, 1, 1), 0, 0, d_q8.ptr, d_rhs.ptr, d_out.ptr, k, rows, cols);
    } else if (mode == "bm128-direct192-raw-asm-inout-output") {
        dim3 bm128_grid((rows + 127) / 128, (cols + 127) / 128, 1);
        hipLaunchKernelGGL((q8_bm128_direct192_output_repro_kernel<true, true>),
            bm128_grid, dim3(256, 1, 1), 0, 0, d_q8.ptr, d_rhs.ptr, d_out.ptr, k, rows, cols);
    } else if (mode == "bm128-streamk-raw-output") {
        dim3 bm128_grid((rows + 127) / 128, (cols + 127) / 128, 1);
        hipLaunchKernelGGL((q8_bm128_streamk_output_repro_kernel<false>),
            bm128_grid, dim3(256, 1, 1), 0, 0, d_q8.ptr, d_rhs.ptr, d_out.ptr, k, rows, cols);
    } else if (mode == "bm128-streamk-raw-asm-output") {
        dim3 bm128_grid((rows + 127) / 128, (cols + 127) / 128, 1);
        hipLaunchKernelGGL((q8_bm128_streamk_output_repro_kernel<true>),
            bm128_grid, dim3(256, 1, 1), 0, 0, d_q8.ptr, d_rhs.ptr, d_out.ptr, k, rows, cols);
    } else if (mode == "bm128-streamk-raw-asm-inout-output") {
        dim3 bm128_grid((rows + 127) / 128, (cols + 127) / 128, 1);
        hipLaunchKernelGGL((q8_bm128_streamk_output_repro_kernel<true, true>),
            bm128_grid, dim3(256, 1, 1), 0, 0, d_q8.ptr, d_rhs.ptr, d_out.ptr, k, rows, cols);
    } else if (mode == "bm128-streamk-prefetch51-raw-asm-output") {
        dim3 bm128_grid((rows + 127) / 128, (cols + 127) / 128, 1);
        hipLaunchKernelGGL((q8_bm128_streamk_output_repro_kernel<true, false, false, false, true>),
            bm128_grid, dim3(256, 1, 1), 0, 0, d_q8.ptr, d_rhs.ptr, d_out.ptr, k, rows, cols);
    } else if (mode == "bm128-streamk-abcopy-output") {
        dim3 bm128_grid((rows + 127) / 128, (cols + 127) / 128, 1);
        hipLaunchKernelGGL((q8_bm128_streamk_output_repro_kernel<false, false, true, true>),
            bm128_grid, dim3(256, 1, 1), 0, 0, d_q8.ptr, d_rhs.ptr, d_out.ptr, k, rows, cols);
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
    } else if (mode == "phase96-bm128-raw") {
        dim3 bm128_grid((rows + 127) / 128, (cols + 127) / 128, 1);
        hipLaunchKernelGGL((q8_phase96_bm128_repro_kernel<0, 0, 2, false, false>),
            bm128_grid, dim3(256, 1, 1), 0, 0, d_q8.ptr, d_rhs.ptr, d_out.ptr, k, rows, cols);
        hipLaunchKernelGGL((q8_phase96_bm128_repro_kernel<8, 2, 2, false, false>),
            bm128_grid, dim3(256, 1, 1), 0, 0, d_q8.ptr, d_rhs.ptr, d_out.ptr, k, rows, cols);
    } else if (mode == "phase96-bm128-raw-asm") {
        dim3 bm128_grid((rows + 127) / 128, (cols + 127) / 128, 1);
        hipLaunchKernelGGL((q8_phase96_bm128_repro_kernel<0, 0, 2, false, false, true>),
            bm128_grid, dim3(256, 1, 1), 0, 0, d_q8.ptr, d_rhs.ptr, d_out.ptr, k, rows, cols);
        hipLaunchKernelGGL((q8_phase96_bm128_repro_kernel<8, 2, 2, false, false, true>),
            bm128_grid, dim3(256, 1, 1), 0, 0, d_q8.ptr, d_rhs.ptr, d_out.ptr, k, rows, cols);
    } else if (mode == "phase96-bm128-acopy") {
        dim3 bm128_grid((rows + 127) / 128, (cols + 127) / 128, 1);
        hipLaunchKernelGGL((q8_phase96_bm128_repro_kernel<0, 0, 2, true, false>),
            bm128_grid, dim3(256, 1, 1), 0, 0, d_q8.ptr, d_rhs.ptr, d_out.ptr, k, rows, cols);
        hipLaunchKernelGGL((q8_phase96_bm128_repro_kernel<8, 2, 2, true, false>),
            bm128_grid, dim3(256, 1, 1), 0, 0, d_q8.ptr, d_rhs.ptr, d_out.ptr, k, rows, cols);
    } else if (mode == "phase96-bm128-bcopy") {
        dim3 bm128_grid((rows + 127) / 128, (cols + 127) / 128, 1);
        hipLaunchKernelGGL((q8_phase96_bm128_repro_kernel<0, 0, 2, false, true>),
            bm128_grid, dim3(256, 1, 1), 0, 0, d_q8.ptr, d_rhs.ptr, d_out.ptr, k, rows, cols);
        hipLaunchKernelGGL((q8_phase96_bm128_repro_kernel<8, 2, 2, false, true>),
            bm128_grid, dim3(256, 1, 1), 0, 0, d_q8.ptr, d_rhs.ptr, d_out.ptr, k, rows, cols);
    } else if (mode == "phase96-bm128-abcopy" || mode == "phase96-bm128-abcopy-backendlike") {
        dim3 bm128_grid((rows + 127) / 128, (cols + 127) / 128, 1);
        hipLaunchKernelGGL((q8_phase96_bm128_repro_kernel<0, 0, 2, true, true>),
            bm128_grid, dim3(256, 1, 1), 0, 0, d_q8.ptr, d_rhs.ptr, d_out.ptr, k, rows, cols);
        hipLaunchKernelGGL((q8_phase96_bm128_repro_kernel<8, 2, 2, true, true>),
            bm128_grid, dim3(256, 1, 1), 0, 0, d_q8.ptr, d_rhs.ptr, d_out.ptr, k, rows, cols);
    } else if (mode == "phase96-bm128-abcopy-asm" ||
            mode == "phase96-bm128-abcopy-asm-backendlike") {
        dim3 bm128_grid((rows + 127) / 128, (cols + 127) / 128, 1);
        hipLaunchKernelGGL((q8_phase96_bm128_repro_kernel<0, 0, 2, true, true, true>),
            bm128_grid, dim3(256, 1, 1), 0, 0, d_q8.ptr, d_rhs.ptr, d_out.ptr, k, rows, cols);
        hipLaunchKernelGGL((q8_phase96_bm128_repro_kernel<8, 2, 2, true, true, true>),
            bm128_grid, dim3(256, 1, 1), 0, 0, d_q8.ptr, d_rhs.ptr, d_out.ptr, k, rows, cols);
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
    } else if (mode == "single-group13-bcopy-stage-selected") {
        hipLaunchKernelGGL((q8_array_fullb_phase_repro_kernel<13, 3, 1, 1, false, false, true, true, true>), grid, dim3(256, 1, 1), 0, 0,
            d_q8.ptr, d_rhs.ptr, d_out.ptr, k, rows, cols);
    } else if (mode == "single-group13-abcopy-stage-selected") {
        hipLaunchKernelGGL((q8_array_fullb_phase_repro_kernel<13, 3, 1, 1, false, true, true, true, true>), grid, dim3(256, 1, 1), 0, 0,
            d_q8.ptr, d_rhs.ptr, d_out.ptr, k, rows, cols);
    } else if (mode == "single-group14-bcopy-stage-selected") {
        hipLaunchKernelGGL((q8_array_fullb_phase_repro_kernel<14, 3, 1, 1, false, false, true, true, true>), grid, dim3(256, 1, 1), 0, 0,
            d_q8.ptr, d_rhs.ptr, d_out.ptr, k, rows, cols);
    } else if (mode == "single-group14-abcopy-stage-selected") {
        hipLaunchKernelGGL((q8_array_fullb_phase_repro_kernel<14, 3, 1, 1, false, true, true, true, true>), grid, dim3(256, 1, 1), 0, 0,
            d_q8.ptr, d_rhs.ptr, d_out.ptr, k, rows, cols);
    } else if (mode == "single-group15-bcopy-stage-selected") {
        hipLaunchKernelGGL((q8_array_fullb_phase_repro_kernel<15, 3, 1, 1, false, false, true, true, true>), grid, dim3(256, 1, 1), 0, 0,
            d_q8.ptr, d_rhs.ptr, d_out.ptr, k, rows, cols);
    } else if (mode == "single-group15-abcopy-stage-selected") {
        hipLaunchKernelGGL((q8_array_fullb_phase_repro_kernel<15, 3, 1, 1, false, true, true, true, true>), grid, dim3(256, 1, 1), 0, 0,
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
    double mse = 0.0;
    double ref_mse = 0.0;
    group_stats by_group[16];
    const size_t step = sample_stride == 0 ? 1 : sample_stride;
    size_t checked = 0;
    for (size_t i = 0; i < h_out.size(); i += step) {
        if (!output_is_active(i, rows, mode)) {
            continue;
        }
        const int group = output_group(i, rows);
        group_stats & gs = by_group[group];
        ++checked;
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
        const float expected = sample_stride == 0 ?
            expected_value_for_output(i, rows, cols, mode, ref) :
            expected_value_for_output_sample(i, rows, cols, mode, h_q8, h_rhs, k);
        const float err = std::fabs(actual - expected);
        mse += static_cast<double>(actual - expected) * static_cast<double>(actual - expected);
        ref_mse += static_cast<double>(expected) * static_cast<double>(expected);
        max_abs = std::max(max_abs, err);
        gs.max_abs = std::max(gs.max_abs, err);
        if (err > 0.25f) {
            ++bad;
            ++gs.bad;
            note_bad_sample(gs, i, rows, actual, expected, err);
        }
    }
    const double nmse_value = ref_mse != 0.0 ? mse / ref_mse : 0.0;

    std::printf(
        "%s rows=%d cols=%d k=%d active=%zu checked=%zu stride=%zu bad=%zu nan=%zu inf=%zu sentinel=%zu max_abs=%g nmse=%.9f\n",
        mode.c_str(), rows, cols, k, active, checked, step, bad, nan, inf, sentinel, max_abs, nmse_value);
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

static int run_dump_case(const std::string & mode, const std::string & dump_dir) {
    if (mode != "phase96-bm128-abcopy" && mode != "phase96-bm128-abcopy-backendlike") {
        std::fprintf(stderr, "--dump-dir replay currently supports phase96-bm128-abcopy modes only\n");
        return 2;
    }

    const std::string stem = dump_dir + "/dump-0-MUL_MAT";
    const std::string meta_path = stem + ".meta.txt";
    int k = 0;
    int rows = 0;
    int cols = 0;
    if (!read_meta_int(meta_path, "k", &k) ||
            !read_meta_int(meta_path, "rows", &rows) ||
            !read_meta_int(meta_path, "cols", &cols)) {
        return 2;
    }
    if (k <= 0 || rows <= 0 || cols <= 0 || (k % 32) != 0) {
        std::fprintf(stderr, "invalid dump shape rows=%d cols=%d k=%d\n", rows, cols, k);
        return 2;
    }

    const int blocks_per_row = k / 32;
    std::vector<hrx_block_q8_0_wmma_vk128_lhs> h_q8;
    std::vector<float> h_rhs;
    std::vector<float> ref;
    std::vector<float> h_out(static_cast<size_t>(rows) * cols, -7777.0f);
    if (!read_binary_vector(stem + ".a.q8_0.bin", h_q8, static_cast<size_t>(rows) * blocks_per_row) ||
            !read_binary_vector(stem + ".b.f32.bin", h_rhs, static_cast<size_t>(cols) * k) ||
            !read_binary_vector(stem + ".ref.f32.bin", ref, static_cast<size_t>(rows) * cols)) {
        return 2;
    }

    device_buffer<hrx_block_q8_0_wmma_vk128_lhs> d_q8(h_q8.size());
    device_buffer<float> d_rhs(h_rhs.size());
    device_buffer<float> d_out(h_out.size());
    HIP_CHECK(hipMemcpy(d_q8.ptr, h_q8.data(), h_q8.size() * sizeof(h_q8[0]), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_rhs.ptr, h_rhs.data(), h_rhs.size() * sizeof(h_rhs[0]), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_out.ptr, h_out.data(), h_out.size() * sizeof(h_out[0]), hipMemcpyHostToDevice));

    dim3 bm128_grid((rows + 127) / 128, (cols + 127) / 128, 1);
    hipLaunchKernelGGL((q8_phase96_bm128_repro_kernel<0, 0, 2, true, true>),
        bm128_grid, dim3(256, 1, 1), 0, 0, d_q8.ptr, d_rhs.ptr, d_out.ptr, k, rows, cols);
    hipLaunchKernelGGL((q8_phase96_bm128_repro_kernel<8, 2, 2, true, true>),
        bm128_grid, dim3(256, 1, 1), 0, 0, d_q8.ptr, d_rhs.ptr, d_out.ptr, k, rows, cols);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_out.data(), d_out.ptr, h_out.size() * sizeof(h_out[0]), hipMemcpyDeviceToHost));

    size_t active = 0;
    size_t nan = 0;
    size_t inf = 0;
    size_t sentinel = 0;
    size_t bad = 0;
    float max_abs = 0.0f;
    double mse = 0.0;
    double ref_mse = 0.0;
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
        mse += static_cast<double>(actual - expected) * static_cast<double>(actual - expected);
        ref_mse += static_cast<double>(expected) * static_cast<double>(expected);
        max_abs = std::max(max_abs, err);
        gs.max_abs = std::max(gs.max_abs, err);
        if (err > 0.25f) {
            ++bad;
            ++gs.bad;
            note_bad_sample(gs, i, rows, actual, expected, err);
        }
    }
    const double nmse_value = ref_mse != 0.0 ? mse / ref_mse : 0.0;

    std::printf(
        "%s dump=%s rows=%d cols=%d k=%d active=%zu bad=%zu nan=%zu inf=%zu sentinel=%zu max_abs=%g nmse=%.9f\n",
        mode.c_str(), dump_dir.c_str(), rows, cols, k, active, bad, nan, inf, sentinel, max_abs, nmse_value);
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
                "    first row=%d col=%d lane=%d slot=%d actual=%g expected=%g err=%g\n",
                gs.first_row,
                gs.first_col,
                gs.first_lane,
                gs.first_slot,
                gs.first_actual,
                gs.first_expected,
                gs.first_err);
        }
        for (int sample = 0; sample < gs.sample_count; ++sample) {
            std::printf(
                "    sample row=%d col=%d lane=%d slot=%d actual=%g expected=%g err=%g\n",
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
    std::string dump_dir;
    int custom_rows = 0;
    int custom_cols = 0;
    int custom_k = 0;
    size_t sample_stride = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            mode = argv[++i];
        } else if (std::strcmp(argv[i], "--dump-dir") == 0 && i + 1 < argc) {
            dump_dir = argv[++i];
        } else if (std::strcmp(argv[i], "--rows") == 0 && i + 1 < argc) {
            custom_rows = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--cols") == 0 && i + 1 < argc) {
            custom_cols = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--k") == 0 && i + 1 < argc) {
            custom_k = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--sample-stride") == 0 && i + 1 < argc) {
            sample_stride = static_cast<size_t>(std::strtoull(argv[++i], nullptr, 10));
        } else {
            std::fprintf(stderr, "usage: %s [--mode MODE] [--dump-dir <test-backend-ops dump dir>] [--rows N --cols N] [--k N] [--sample-stride N]\n", argv[0]);
            return 2;
        }
    }
    const bool custom_shape = custom_rows != 0 || custom_cols != 0;
    if (custom_shape && (custom_rows <= 0 || custom_cols <= 0)) {
        std::fprintf(stderr, "--rows and --cols must both be positive when either is provided\n");
        return 2;
    }
    if (custom_k < 0 || (custom_k > 0 && custom_k % 32 != 0)) {
        std::fprintf(stderr, "--k must be a positive multiple of 32\n");
        return 2;
    }

    if (!dump_dir.empty()) {
        return run_dump_case(mode, dump_dir);
    }

    int status = 0;
    const int rows = 64;
    const int k = custom_k > 0 ? custom_k : 4096;
    if (mode == "motif192-synth-address") {
        if (custom_shape) {
            status |= run_motif192_synthetic_case(custom_rows, custom_cols);
        } else {
            status |= run_motif192_synthetic_case(128, 128);
            status |= run_motif192_synthetic_case(128, 129);
            status |= run_motif192_synthetic_case(129, 128);
            status |= run_motif192_synthetic_case(1024, 512);
            status |= run_motif192_synthetic_case(4096, 512);
            status |= run_motif192_synthetic_case(4096, 513);
        }
    }
    if (mode == "motif192-wmma-address") {
        if (custom_shape) {
            status |= run_motif192_wmma_case(
                "motif192-wmma-address",
                custom_rows,
                custom_cols,
                Q8_REPRO_MOTIF192_STORE_FULL,
                false);
        } else {
            status |= run_motif192_wmma_suite(
                "motif192-wmma-address",
                Q8_REPRO_MOTIF192_STORE_FULL,
                false);
        }
    }
    if (mode == "motif192-wmma-waitload-address") {
        if (custom_shape) {
            status |= run_motif192_wmma_case(
                "motif192-wmma-waitload-address",
                custom_rows,
                custom_cols,
                Q8_REPRO_MOTIF192_STORE_FULL,
                true);
        } else {
            status |= run_motif192_wmma_suite(
                "motif192-wmma-waitload-address",
                Q8_REPRO_MOTIF192_STORE_FULL,
                true);
        }
    }
    if (mode == "motif192-wmma-k2-directwait-waitload-address") {
        if (custom_shape) {
            status |= run_motif192_wmma_k2_case<false>(
                "motif192-wmma-k2-directwait-waitload-address",
                custom_rows,
                custom_cols,
                true);
        } else {
            status |= run_motif192_wmma_k2_suite<false>(
                "motif192-wmma-k2-directwait-waitload-address",
                true);
        }
    }
    if (mode == "motif192-wmma-k2-depwait-waitload-address") {
        if (custom_shape) {
            status |= run_motif192_wmma_k2_case<true>(
                "motif192-wmma-k2-depwait-waitload-address",
                custom_rows,
                custom_cols,
                true);
        } else {
            status |= run_motif192_wmma_k2_suite<true>(
                "motif192-wmma-k2-depwait-waitload-address",
                true);
        }
    }
    if (mode == "motif192-wmma-k2-realdata-k32-directwait-waitload-address") {
        if (custom_shape) {
            status |= run_motif192_wmma_k2_realdata_k32_case<false>(
                "motif192-wmma-k2-realdata-k32-directwait-waitload-address",
                custom_rows,
                custom_cols,
                true);
        } else {
            status |= run_motif192_wmma_k2_realdata_k32_suite<false>(
                "motif192-wmma-k2-realdata-k32-directwait-waitload-address",
                true);
        }
    }
    if (mode == "motif192-wmma-k2-realdata-fullk-directwait-waitload-address") {
        if (custom_shape) {
            status |= run_motif192_wmma_k2_realdata_fullk_case<false>(
                "motif192-wmma-k2-realdata-fullk-directwait-waitload-address",
                custom_rows,
                custom_cols,
                k,
                true);
        } else {
            status |= run_motif192_wmma_k2_realdata_fullk_suite<false>(
                "motif192-wmma-k2-realdata-fullk-directwait-waitload-address",
                true);
        }
    }
    if (mode == "motif192-wmma-k2-realdata-fullk-phase8-directwait-waitload-address") {
        if (custom_shape) {
            status |= run_motif192_wmma_k2_realdata_fullk_phase8_case(
                "motif192-wmma-k2-realdata-fullk-phase8-directwait-waitload-address",
                custom_rows,
                custom_cols,
                k,
                true);
        } else {
            status |= run_motif192_wmma_k2_realdata_fullk_phase8_suite(
                "motif192-wmma-k2-realdata-fullk-phase8-directwait-waitload-address",
                true);
        }
    }
    if (mode == "motif192-wmma-k2-realdata-fullk-phase8seq-directwait-waitload-address") {
        if (custom_shape) {
            status |= run_motif192_wmma_k2_realdata_fullk_phase8seq_case(
                "motif192-wmma-k2-realdata-fullk-phase8seq-directwait-waitload-address",
                custom_rows,
                custom_cols,
                k,
                true);
        } else {
            status |= run_motif192_wmma_k2_realdata_fullk_phase8seq_suite(
                "motif192-wmma-k2-realdata-fullk-phase8seq-directwait-waitload-address",
                true);
        }
    }
    if (mode == "motif192-wmma-k2-realdata-fullk-streamfrag-directwait-waitload-address") {
        if (custom_shape) {
            status |= run_motif192_wmma_k2_realdata_fullk_streamfrag_case(
                "motif192-wmma-k2-realdata-fullk-streamfrag-directwait-waitload-address",
                custom_rows,
                custom_cols,
                k,
                true);
        } else {
            status |= run_motif192_wmma_k2_realdata_fullk_streamfrag_suite(
                "motif192-wmma-k2-realdata-fullk-streamfrag-directwait-waitload-address",
                true);
        }
    }
    if (mode == "motif192-wmma-k2-realdata-fullk-ktilefrag-directwait-waitload-address") {
        if (custom_shape) {
            status |= run_motif192_wmma_k2_realdata_fullk_ktilefrag_case(
                "motif192-wmma-k2-realdata-fullk-ktilefrag-directwait-waitload-address",
                custom_rows,
                custom_cols,
                k,
                true);
        } else {
            status |= run_motif192_wmma_k2_realdata_fullk_ktilefrag_suite(
                "motif192-wmma-k2-realdata-fullk-ktilefrag-directwait-waitload-address",
                true);
        }
    }
    if (mode == "motif192-wmma-k2-realdata-fullk-ktilefrag-storebatch-directwait-waitload-address") {
        if (custom_shape) {
            status |= run_motif192_wmma_k2_realdata_fullk_ktilefrag_case(
                "motif192-wmma-k2-realdata-fullk-ktilefrag-storebatch-directwait-waitload-address",
                custom_rows,
                custom_cols,
                k,
                true,
                true);
        } else {
            status |= run_motif192_wmma_k2_realdata_fullk_ktilefrag_suite(
                "motif192-wmma-k2-realdata-fullk-ktilefrag-storebatch-directwait-waitload-address",
                true,
                true);
        }
    }
    if (mode == "motif192-wmma-k2-realdata-fullk-ktilefrag-storebatch4-directwait-waitload-address") {
        if (custom_shape) {
            status |= run_motif192_wmma_k2_realdata_fullk_ktilefrag_case(
                "motif192-wmma-k2-realdata-fullk-ktilefrag-storebatch4-directwait-waitload-address",
                custom_rows,
                custom_cols,
                k,
                true,
                false,
                true);
        } else {
            status |= run_motif192_wmma_k2_realdata_fullk_ktilefrag_suite(
                "motif192-wmma-k2-realdata-fullk-ktilefrag-storebatch4-directwait-waitload-address",
                true,
                false,
                true);
        }
    }
    if (mode == "motif192-wmma-k2-realdata-fullk-colpairfrag-directwait-waitload-address") {
        if (custom_shape) {
            status |= run_motif192_wmma_k2_realdata_fullk_colpairfrag_case(
                "motif192-wmma-k2-realdata-fullk-colpairfrag-directwait-waitload-address",
                custom_rows,
                custom_cols,
                k,
                true);
        } else {
            status |= run_motif192_wmma_k2_realdata_fullk_colpairfrag_suite(
                "motif192-wmma-k2-realdata-fullk-colpairfrag-directwait-waitload-address",
                true);
        }
    }
    if (mode == "motif192-wmma-k2-realdata-fullk-accpark-directwait-waitload-address") {
        if (custom_shape) {
            status |= run_motif192_wmma_k2_realdata_fullk_accpark_case(
                "motif192-wmma-k2-realdata-fullk-accpark-directwait-waitload-address",
                custom_rows,
                custom_cols,
                k,
                true);
        } else {
            status |= run_motif192_wmma_k2_realdata_fullk_accpark_suite(
                "motif192-wmma-k2-realdata-fullk-accpark-directwait-waitload-address",
                true);
        }
    }
    if (mode == "motif192-wmma-k2-realdata-fullk-accparkfull8-directwait-waitload-address") {
        if (custom_shape) {
            status |= run_motif192_wmma_k2_realdata_fullk_accparkfull8_case(
                "motif192-wmma-k2-realdata-fullk-accparkfull8-directwait-waitload-address",
                custom_rows,
                custom_cols,
                k,
                true);
        } else {
            status |= run_motif192_wmma_k2_realdata_fullk_accparkfull8_suite(
                "motif192-wmma-k2-realdata-fullk-accparkfull8-directwait-waitload-address",
                true);
        }
    }
    if (mode == "motif192-wmma-k2-realdata-fullk-timing") {
        if (custom_shape) {
            status |= run_motif192_wmma_k2_realdata_fullk_timing_case(
                "motif192-wmma-k2-realdata-fullk-timing",
                custom_rows,
                custom_cols,
                4096,
                50,
                true);
        } else {
            status |= run_motif192_wmma_k2_realdata_fullk_timing_suite(
                "motif192-wmma-k2-realdata-fullk-timing",
                true);
        }
    }
    if (mode == "motif192-wmma-k2-realdata-fullk-phase8seq-timing") {
        if (custom_shape) {
            status |= run_motif192_wmma_k2_realdata_fullk_phase8seq_timing_case(
                "motif192-wmma-k2-realdata-fullk-phase8seq-timing",
                custom_rows,
                custom_cols,
                4096,
                50,
                true);
        } else {
            status |= run_motif192_wmma_k2_realdata_fullk_phase8seq_timing_suite(
                "motif192-wmma-k2-realdata-fullk-phase8seq-timing",
                true);
        }
    }
    if (mode == "motif192-wmma-k2-realdata-fullk-streamfrag-timing") {
        if (custom_shape) {
            status |= run_motif192_wmma_k2_realdata_fullk_streamfrag_timing_case(
                "motif192-wmma-k2-realdata-fullk-streamfrag-timing",
                custom_rows,
                custom_cols,
                4096,
                50,
                true);
        } else {
            status |= run_motif192_wmma_k2_realdata_fullk_streamfrag_timing_suite(
                "motif192-wmma-k2-realdata-fullk-streamfrag-timing",
                true);
        }
    }
    if (mode == "motif192-wmma-k2-realdata-fullk-ktilefrag-timing") {
        if (custom_shape) {
            status |= run_motif192_wmma_k2_realdata_fullk_ktilefrag_timing_case(
                "motif192-wmma-k2-realdata-fullk-ktilefrag-timing",
                custom_rows,
                custom_cols,
                4096,
                50,
                true);
        } else {
            status |= run_motif192_wmma_k2_realdata_fullk_ktilefrag_timing_suite(
                "motif192-wmma-k2-realdata-fullk-ktilefrag-timing",
                true);
        }
    }
    if (mode == "motif192-wmma-k2-realdata-fullk-ktilefrag-storebatch-timing") {
        if (custom_shape) {
            status |= run_motif192_wmma_k2_realdata_fullk_ktilefrag_timing_case(
                "motif192-wmma-k2-realdata-fullk-ktilefrag-storebatch-timing",
                custom_rows,
                custom_cols,
                4096,
                50,
                true,
                true);
        } else {
            status |= run_motif192_wmma_k2_realdata_fullk_ktilefrag_timing_suite(
                "motif192-wmma-k2-realdata-fullk-ktilefrag-storebatch-timing",
                true,
                true);
        }
    }
    if (mode == "motif192-wmma-k2-realdata-fullk-ktilefrag-storebatch4-timing") {
        if (custom_shape) {
            status |= run_motif192_wmma_k2_realdata_fullk_ktilefrag_timing_case(
                "motif192-wmma-k2-realdata-fullk-ktilefrag-storebatch4-timing",
                custom_rows,
                custom_cols,
                4096,
                50,
                true,
                false,
                true);
        } else {
            status |= run_motif192_wmma_k2_realdata_fullk_ktilefrag_timing_suite(
                "motif192-wmma-k2-realdata-fullk-ktilefrag-storebatch4-timing",
                true,
                false,
                true);
        }
    }
    if (mode == "motif192-wmma-k2-realdata-fullk-colpairfrag-timing") {
        if (custom_shape) {
            status |= run_motif192_wmma_k2_realdata_fullk_colpairfrag_timing_case(
                "motif192-wmma-k2-realdata-fullk-colpairfrag-timing",
                custom_rows,
                custom_cols,
                4096,
                50,
                true);
        } else {
            status |= run_motif192_wmma_k2_realdata_fullk_colpairfrag_timing_suite(
                "motif192-wmma-k2-realdata-fullk-colpairfrag-timing",
                true);
        }
    }
    if (mode == "motif192-wmma-k2-realdata-fullk-accpark-timing") {
        if (custom_shape) {
            status |= run_motif192_wmma_k2_realdata_fullk_accpark_timing_case(
                "motif192-wmma-k2-realdata-fullk-accpark-timing",
                custom_rows,
                custom_cols,
                4096,
                50,
                true);
        } else {
            status |= run_motif192_wmma_k2_realdata_fullk_accpark_timing_suite(
                "motif192-wmma-k2-realdata-fullk-accpark-timing",
                true);
        }
    }
    if (mode == "motif192-wmma-k2-realdata-fullk-accparkfull8-timing") {
        if (custom_shape) {
            status |= run_motif192_wmma_k2_realdata_fullk_accparkfull8_timing_case(
                "motif192-wmma-k2-realdata-fullk-accparkfull8-timing",
                custom_rows,
                custom_cols,
                4096,
                50,
                true);
        } else {
            status |= run_motif192_wmma_k2_realdata_fullk_accparkfull8_timing_suite(
                "motif192-wmma-k2-realdata-fullk-accparkfull8-timing",
                true);
        }
    }
    if (mode == "motif192-wmma-direct-address") {
        if (custom_shape) {
            status |= run_motif192_wmma_case(
                "motif192-wmma-direct-address",
                custom_rows,
                custom_cols,
                Q8_REPRO_MOTIF192_STORE_DIRECT,
                false);
        } else {
            status |= run_motif192_wmma_suite(
                "motif192-wmma-direct-address",
                Q8_REPRO_MOTIF192_STORE_DIRECT,
                false);
        }
    }
    if (mode == "motif192-wmma-stage16-address") {
        if (custom_shape) {
            status |= run_motif192_wmma_case(
                "motif192-wmma-stage16-address",
                custom_rows,
                custom_cols,
                Q8_REPRO_MOTIF192_STORE_STAGE16,
                false);
        } else {
            status |= run_motif192_wmma_suite(
                "motif192-wmma-stage16-address",
                Q8_REPRO_MOTIF192_STORE_STAGE16,
                false);
        }
    }
    if (mode == "motif192-wmma-stage32-address") {
        if (custom_shape) {
            status |= run_motif192_wmma_case(
                "motif192-wmma-stage32-address",
                custom_rows,
                custom_cols,
                Q8_REPRO_MOTIF192_STORE_STAGE32,
                false);
        } else {
            status |= run_motif192_wmma_suite(
                "motif192-wmma-stage32-address",
                Q8_REPRO_MOTIF192_STORE_STAGE32,
                false);
        }
    }
    if (mode == "motif192-wmma-stage16-waitload-address") {
        if (custom_shape) {
            status |= run_motif192_wmma_case(
                "motif192-wmma-stage16-waitload-address",
                custom_rows,
                custom_cols,
                Q8_REPRO_MOTIF192_STORE_STAGE16,
                true);
        } else {
            status |= run_motif192_wmma_suite(
                "motif192-wmma-stage16-waitload-address",
                Q8_REPRO_MOTIF192_STORE_STAGE16,
                true);
        }
    }
    if (mode == "motif192-wmma-stage32-waitload-address") {
        if (custom_shape) {
            status |= run_motif192_wmma_case(
                "motif192-wmma-stage32-waitload-address",
                custom_rows,
                custom_cols,
                Q8_REPRO_MOTIF192_STORE_STAGE32,
                true);
        } else {
            status |= run_motif192_wmma_suite(
                "motif192-wmma-stage32-waitload-address",
                Q8_REPRO_MOTIF192_STORE_STAGE32,
                true);
        }
    }
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
    if (mode == "bm128-direct192-raw-output" ||
            mode == "bm128-direct192-raw-asm-output" ||
            mode == "bm128-direct192-raw-asm-inout-output" ||
            mode == "bm128-streamk-raw-output" ||
            mode == "bm128-streamk-raw-asm-output" ||
            mode == "bm128-streamk-raw-asm-inout-output" ||
            mode == "bm128-streamk-prefetch51-raw-asm-output" ||
            mode == "bm128-streamk-abcopy-output") {
        if (custom_shape) {
            status |= run_case(mode, custom_rows, custom_cols, k, sample_stride);
        } else {
            status |= run_case(mode, 128, 128, k);
            status |= run_case(mode, 128, 33, k);
            status |= run_case(mode, 4096, 512, k);
            status |= run_case(mode, 4096, 513, k);
        }
    }
    if (mode == "contract-direct192-raw" ||
            mode == "contract-direct192-bcopy" ||
            mode == "contract-direct192-bcopy-hoist" ||
            mode == "contract-direct192-abcopy" ||
            mode == "contract-direct192-abcopy-bhoist" ||
            mode == "contract-phase96-abcopy") {
        if (custom_shape) {
            if (custom_rows > 64 || custom_cols > 64) {
                std::fprintf(stderr, "%s custom contract shapes are limited to one 64x64 tile\n", mode.c_str());
                return 2;
            }
            status |= run_contract_case(mode, custom_rows, custom_cols, k);
        } else {
            status |= run_contract_case(mode, rows, 64, k);
            status |= run_contract_case(mode, rows, 33, k);
        }
    }
    if (mode == "contract-bm128-direct192-raw" ||
            mode == "contract-bm128-direct192-raw-asm" ||
            mode == "contract-bm128-direct192-bcopy-upper" ||
            mode == "contract-bm128-direct192-bcopy-upper-asm" ||
            mode == "contract-bm128-direct192-bcopy-upper-hoist" ||
            mode == "contract-bm128-direct192-abcopy" ||
            mode == "contract-bm128-direct192-abcopy-bhoist" ||
            mode == "contract-bm128-direct192-abcopy-bhoist-asm") {
        if (custom_shape) {
            if (custom_rows > 128 || custom_cols > 128) {
                std::fprintf(stderr, "%s custom contract shapes are limited to one 128x128 tile\n", mode.c_str());
                return 2;
            }
            status |= run_bm128_contract_case(mode, custom_rows, custom_cols, k);
        } else {
            status |= run_bm128_contract_case(mode, 128, 128, k);
            status |= run_bm128_contract_case(mode, 128, 33, k);
        }
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
    if (mode == "phase96-bm128-raw" ||
            mode == "phase96-bm128-raw-asm" ||
            mode == "phase96-bm128-acopy" ||
            mode == "phase96-bm128-bcopy" ||
            mode == "phase96-bm128-abcopy" ||
            mode == "phase96-bm128-abcopy-asm") {
        status |= run_case(mode, 128, 128, k);
        status |= run_case(mode, 128, 129, k);
        status |= run_case(mode, 128, 33, k);
        status |= run_case(mode, 128, 128, 14336);
        status |= run_case(mode, 128, 129, 14336);
        status |= run_case(mode, 512, 128, k);
        status |= run_case(mode, 1024, 128, k);
    }
    if (mode == "phase96-bm128-abcopy-backendlike" ||
            mode == "phase96-bm128-abcopy-asm-backendlike") {
        status |= run_case(mode, 128, 128, k);
        status |= run_case(mode, 128, 512, k);
        status |= run_case(mode, 128, 128, 14336);
        status |= run_case(mode, 1024, 512, k);
        status |= run_case(mode, 4096, 512, k);
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
            mode == "single-group13-bcopy-stage-selected" ||
            mode == "single-group13-abcopy-stage-selected" ||
            mode == "single-group14-bcopy-stage-selected" ||
            mode == "single-group14-abcopy-stage-selected" ||
            mode == "single-group15-bcopy-stage-selected" ||
            mode == "single-group15-abcopy-stage-selected" ||
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
