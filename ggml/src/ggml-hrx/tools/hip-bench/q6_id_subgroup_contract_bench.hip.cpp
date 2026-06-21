#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define HIP_CHECK(expr) do { \
    hipError_t _err = (expr); \
    if (_err != hipSuccess) { \
        std::fprintf(stderr, "%s:%d: HIP error: %s\n", __FILE__, __LINE__, hipGetErrorString(_err)); \
        std::exit(2); \
    } \
} while (0)

typedef _Float16 q6id_contract_half16_vec __attribute__((ext_vector_type(16)));
typedef _Float16 q6id_contract_half8_vec __attribute__((ext_vector_type(8)));
typedef uint32_t q6id_contract_u32x4_vec __attribute__((ext_vector_type(4)));
typedef uint32_t q6id_contract_u32x8_vec __attribute__((ext_vector_type(8)));
typedef uint64_t q6id_contract_u64x4_vec __attribute__((ext_vector_type(4)));

static constexpr unsigned int HRX_Q6ID_CONTRACT_LANES = 64;
static constexpr unsigned int HRX_Q6ID_CONTRACT_GROUPS = 16;
static constexpr unsigned int HRX_Q6ID_CONTRACT_SLOTS = 2;
static constexpr unsigned int HRX_Q6ID_CONTRACT_VALUES =
    HRX_Q6ID_CONTRACT_GROUPS * HRX_Q6ID_CONTRACT_SLOTS * HRX_Q6ID_CONTRACT_LANES;
static constexpr unsigned int HRX_Q6ID_CONTRACT_PROD_ROWS = 64;
static constexpr unsigned int HRX_Q6ID_CONTRACT_PROD_COLS = 33;
static constexpr unsigned int HRX_Q6ID_CONTRACT_PROD_VALUES =
    HRX_Q6ID_CONTRACT_PROD_ROWS * HRX_Q6ID_CONTRACT_PROD_COLS;

static __host__ __device__ __forceinline__ unsigned int q6id_contract_index(
        unsigned int group,
        unsigned int slot,
        unsigned int lane) {
    return (group * HRX_Q6ID_CONTRACT_SLOTS + slot) * HRX_Q6ID_CONTRACT_LANES + lane;
}

static __host__ __device__ __forceinline__ float q6id_contract_seed(
        unsigned int group,
        unsigned int slot,
        unsigned int lane) {
    return 0.125f + static_cast<float>(group) * 0.015625f +
        static_cast<float>(slot) * 0.00390625f +
        static_cast<float>(lane) * 0.000244140625f;
}

static __device__ __forceinline__ uint64_t q6id_contract_half_pack4(float value) {
    const uint64_t bits = static_cast<uint64_t>(
        __builtin_bit_cast(uint16_t, static_cast<_Float16>(value)));
    return bits | (bits << 16) | (bits << 32) | (bits << 48);
}

static __device__ __forceinline__ uint32_t q6id_contract_half_pack2(float lo, float hi) {
    const uint32_t lo_bits = static_cast<uint32_t>(
        __builtin_bit_cast(uint16_t, static_cast<_Float16>(lo)));
    const uint32_t hi_bits = static_cast<uint32_t>(
        __builtin_bit_cast(uint16_t, static_cast<_Float16>(hi)));
    return lo_bits | (hi_bits << 16);
}

static __device__ __forceinline__ uint64_t q6id_contract_ds_load_b64_nowait(
        const __attribute__((address_space(3))) uint64_t * ptr) {
    uint64_t value = 0;
    asm volatile("ds_read_b64 %0, %1 offset:0\n"
                 : "=v"(value)
                 : "v"(ptr)
                 : "memory");
    return value;
}

static __device__ __forceinline__ q6id_contract_half16_vec q6id_contract_load_fragment(
        const __attribute__((address_space(3))) uint64_t * lds,
        unsigned int frag,
        unsigned int lane) {
    const unsigned int base = frag * 256u + lane * 4u;
    q6id_contract_u64x4_vec raw;
    raw[0] = q6id_contract_ds_load_b64_nowait(lds + base + 0u);
    raw[1] = q6id_contract_ds_load_b64_nowait(lds + base + 1u);
    raw[2] = q6id_contract_ds_load_b64_nowait(lds + base + 2u);
    raw[3] = q6id_contract_ds_load_b64_nowait(lds + base + 3u);
    return __builtin_bit_cast(q6id_contract_half16_vec, raw);
}

static __device__ __forceinline__ void q6id_contract_sink_fragment(q6id_contract_half16_vec frag) {
    const q6id_contract_u32x8_vec raw = __builtin_bit_cast(q6id_contract_u32x8_vec, frag);
    asm volatile("" :: "v"(raw[0]), "v"(raw[1]), "v"(raw[2]), "v"(raw[3]),
                       "v"(raw[4]), "v"(raw[5]), "v"(raw[6]), "v"(raw[7]) : "memory");
}

static __device__ __forceinline__ void q6id_contract_ds_write_b16(
        __attribute__((address_space(3))) uint16_t * ptr,
        uint32_t value) {
    asm volatile("ds_write_b16 %0, %1 offset:0\n"
                 :
                 : "v"(ptr), "v"(value)
                 : "memory");
}

static __device__ __forceinline__ q6id_contract_u32x4_vec q6id_contract_init_compact_acc(
        unsigned int group,
        unsigned int lane) {
    q6id_contract_u32x4_vec acc;
    acc[0] = q6id_contract_half_pack2(q6id_contract_seed(group, 0u, lane), q6id_contract_seed(group, 1u, lane));
    acc[1] = q6id_contract_half_pack2(q6id_contract_seed(group, 0u, lane), q6id_contract_seed(group, 1u, lane));
    acc[2] = q6id_contract_half_pack2(q6id_contract_seed(group, 0u, lane), q6id_contract_seed(group, 1u, lane));
    acc[3] = q6id_contract_half_pack2(q6id_contract_seed(group, 0u, lane), q6id_contract_seed(group, 1u, lane));
    return acc;
}

static __device__ __forceinline__ uint32_t q6id_contract_ds_read_u16_d16(
        const __attribute__((address_space(3))) uint16_t * ptr) {
    uint32_t value = 0;
    asm volatile("ds_read_u16_d16 %0, %1 offset:0\n"
                 : "+v"(value)
                 : "v"(ptr)
                 : "memory");
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    return value;
}

static __device__ __forceinline__ void q6id_contract_store_value(
        float * dst,
        unsigned int * counts,
        unsigned int group,
        unsigned int slot,
        unsigned int lane,
        float value) {
    const unsigned int index = q6id_contract_index(group, slot, lane);
    dst[index] = value;
    atomicAdd(counts + index, 1u);
}

static __device__ __forceinline__ float q6id_contract_prod_value(
        unsigned int group,
        unsigned int reg,
        unsigned int lane) {
    return 1.0f + static_cast<float>(group) * 0.03125f +
        static_cast<float>(reg) * 0.00390625f +
        static_cast<float>(lane) * 0.00006103515625f;
}

static __device__ __forceinline__ q6id_contract_u32x4_vec q6id_contract_init_compact_prod_acc(
        unsigned int group,
        unsigned int lane) {
    q6id_contract_u32x4_vec acc;
    acc[0] = q6id_contract_half_pack2(q6id_contract_prod_value(group, 0u, lane), q6id_contract_prod_value(group, 1u, lane));
    acc[1] = q6id_contract_half_pack2(q6id_contract_prod_value(group, 1u, lane), q6id_contract_prod_value(group, 2u, lane));
    acc[2] = q6id_contract_half_pack2(q6id_contract_prod_value(group, 2u, lane), q6id_contract_prod_value(group, 3u, lane));
    acc[3] = q6id_contract_half_pack2(q6id_contract_prod_value(group, 3u, lane), q6id_contract_prod_value(group, 0u, lane));
    return acc;
}

static __device__ __forceinline__ void q6id_contract_prod_store_group(
        float * dst,
        unsigned int * counts,
        unsigned int group,
        unsigned int lane,
        float value_bias) {
    const unsigned int row_tile = group & 3u;
    const unsigned int col_tile = (group >> 2u) & 3u;
    const unsigned int row_lane = lane >> 4u;
    const unsigned int col = col_tile * 16u + (lane & 15u);
    if (col >= HRX_Q6ID_CONTRACT_PROD_COLS) {
        return;
    }
#pragma unroll
    for (unsigned int reg = 0; reg < 4u; ++reg) {
        const unsigned int row = row_tile * 16u + row_lane + reg * 4u;
        if (row < HRX_Q6ID_CONTRACT_PROD_ROWS) {
            const unsigned int index = col * HRX_Q6ID_CONTRACT_PROD_ROWS + row;
            dst[index] = q6id_contract_prod_value(group, reg, lane) + value_bias;
            atomicAdd(counts + index, 1u);
        }
    }
}

static __device__ __forceinline__ unsigned int q6id_contract_stage_index(
        unsigned int group,
        unsigned int slot,
        unsigned int lane) {
    return (group * HRX_Q6ID_CONTRACT_SLOTS + slot) * HRX_Q6ID_CONTRACT_LANES + lane;
}

static __device__ __forceinline__ void q6id_contract_stage_compact_acc(
        uint16_t * sh_stage,
        unsigned int group,
        unsigned int lane,
        q6id_contract_u32x4_vec acc) {
    q6id_contract_ds_write_b16(
        (__attribute__((address_space(3))) uint16_t *) (sh_stage + q6id_contract_stage_index(group, 0u, lane)),
        acc[0] & 0xffffu);
    q6id_contract_ds_write_b16(
        (__attribute__((address_space(3))) uint16_t *) (sh_stage + q6id_contract_stage_index(group, 1u, lane)),
        acc[1] & 0xffffu);
}

static __device__ __forceinline__ unsigned int q6id_contract_prod_stage_index(
        unsigned int group,
        unsigned int reg,
        unsigned int lane) {
    return group * HRX_Q6ID_CONTRACT_LANES * 4u + lane * 4u + reg;
}

static __device__ __forceinline__ void q6id_contract_prod_stage_compact_acc(
        uint16_t * sh_prod_stage,
        unsigned int group,
        unsigned int lane,
        q6id_contract_u32x4_vec acc) {
#pragma unroll
    for (unsigned int reg = 0; reg < 4u; ++reg) {
        q6id_contract_ds_write_b16(
            (__attribute__((address_space(3))) uint16_t *) (sh_prod_stage + q6id_contract_prod_stage_index(group, reg, lane)),
            acc[reg] & 0xffffu);
    }
}

static __device__ __attribute__((noinline)) void q6id_contract_prod_stage_compact_acc8(
        uint16_t * sh_prod_stage,
        unsigned int group_base,
        unsigned int lane,
        q6id_contract_u32x4_vec acc0,
        q6id_contract_u32x4_vec acc1,
        q6id_contract_u32x4_vec acc2,
        q6id_contract_u32x4_vec acc3,
        q6id_contract_u32x4_vec acc4,
        q6id_contract_u32x4_vec acc5,
        q6id_contract_u32x4_vec acc6,
        q6id_contract_u32x4_vec acc7) {
    q6id_contract_prod_stage_compact_acc(sh_prod_stage, group_base + 0u, lane, acc0);
    q6id_contract_prod_stage_compact_acc(sh_prod_stage, group_base + 1u, lane, acc1);
    q6id_contract_prod_stage_compact_acc(sh_prod_stage, group_base + 2u, lane, acc2);
    q6id_contract_prod_stage_compact_acc(sh_prod_stage, group_base + 3u, lane, acc3);
    q6id_contract_prod_stage_compact_acc(sh_prod_stage, group_base + 4u, lane, acc4);
    q6id_contract_prod_stage_compact_acc(sh_prod_stage, group_base + 5u, lane, acc5);
    q6id_contract_prod_stage_compact_acc(sh_prod_stage, group_base + 6u, lane, acc6);
    q6id_contract_prod_stage_compact_acc(sh_prod_stage, group_base + 7u, lane, acc7);
}

static __device__ __forceinline__ _Float16 q6id_contract_u16_to_f16(uint32_t value) {
    union {
        uint16_t u;
        _Float16 h;
    } pack;
    pack.u = static_cast<uint16_t>(value);
    return pack.h;
}

static __device__ __forceinline__ void q6id_contract_prod_store_value(
        float * dst,
        unsigned int * counts,
        unsigned int group,
        unsigned int reg,
        unsigned int lane,
        float value) {
    const unsigned int row_tile = group & 3u;
    const unsigned int col_tile = group >> 2u;
    const unsigned int row_lane = lane >> 4u;
    const unsigned int col = col_tile * 16u + (lane & 15u);
    if (col >= HRX_Q6ID_CONTRACT_PROD_COLS) {
        return;
    }
    const unsigned int row = row_tile * 16u + row_lane + reg * 4u;
    if (row >= HRX_Q6ID_CONTRACT_PROD_ROWS) {
        return;
    }
    const unsigned int index = col * HRX_Q6ID_CONTRACT_PROD_ROWS + row;
    dst[index] = value;
    atomicAdd(counts + index, 1u);
}

static __device__ __forceinline__ void q6id_contract_init_fragments(
        uint64_t * sh_frag,
        unsigned int lane) {
#pragma unroll
    for (unsigned int frag = 0; frag < 4u; ++frag) {
        const uint64_t packed = q6id_contract_half_pack4(1.0f + static_cast<float>(frag & 1u));
#pragma unroll
        for (unsigned int item = 0; item < 4u; ++item) {
            sh_frag[frag * 256u + lane * 4u + item] = packed;
        }
    }
}

static __device__ __forceinline__ void q6id_contract_init_banked_fragments(
        uint64_t * sh_frag,
        unsigned int lane) {
#pragma unroll
    for (unsigned int frag = 0; frag < 12u; ++frag) {
        const uint64_t packed = q6id_contract_half_pack4(1.0f + static_cast<float>(frag & 3u));
#pragma unroll
        for (unsigned int item = 0; item < 4u; ++item) {
            sh_frag[frag * 256u + lane * 4u + item] = packed;
        }
    }
}

static __device__ __forceinline__ void q6id_contract_init_fragment0(
        uint64_t * sh_frag,
        unsigned int lane) {
    const uint64_t packed = q6id_contract_half_pack4(1.0f);
#pragma unroll
    for (unsigned int item = 0; item < 4u; ++item) {
        sh_frag[lane * 4u + item] = packed;
    }
}

#define HRX_Q6ID_CONTRACT_WMMA(GROUP_ID, A_FRAG, B_FRAG) do { \
    q6id_contract_half8_vec acc; \
    const unsigned int _group = (GROUP_ID); \
    _Pragma("unroll") \
    for (unsigned int _slot = 0; _slot < 8u; ++_slot) { \
        acc[_slot] = static_cast<_Float16>(q6id_contract_seed(_group, _slot & 1u, lane)); \
    } \
    acc = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64((A_FRAG), (B_FRAG), acc, false); \
    q6id_contract_ds_write_b16( \
        (__attribute__((address_space(3))) uint16_t *) (sh_stage + q6id_contract_stage_index(_group, 0u, lane)), \
        static_cast<uint32_t>(__builtin_bit_cast(uint16_t, acc[0]))); \
    q6id_contract_ds_write_b16( \
        (__attribute__((address_space(3))) uint16_t *) (sh_stage + q6id_contract_stage_index(_group, 1u, lane)), \
        static_cast<uint32_t>(__builtin_bit_cast(uint16_t, acc[2]))); \
} while (0)

#define HRX_Q6ID_CONTRACT_WMMA_COMPACT(GROUP_ID, A_FRAG, B_FRAG) do { \
    const unsigned int _group = (GROUP_ID); \
    q6id_contract_u32x4_vec acc; \
    acc[0] = q6id_contract_half_pack2(q6id_contract_seed(_group, 0u, lane), q6id_contract_seed(_group, 1u, lane)); \
    acc[1] = q6id_contract_half_pack2(q6id_contract_seed(_group, 0u, lane), q6id_contract_seed(_group, 1u, lane)); \
    acc[2] = q6id_contract_half_pack2(q6id_contract_seed(_group, 0u, lane), q6id_contract_seed(_group, 1u, lane)); \
    acc[3] = q6id_contract_half_pack2(q6id_contract_seed(_group, 0u, lane), q6id_contract_seed(_group, 1u, lane)); \
    asm volatile("v_wmma_f16_16x16x16_f16 %0, %1, %2, %0\n" \
                 : "+v"(acc) \
                 : "v"(A_FRAG), "v"(B_FRAG) \
                 : "memory"); \
    q6id_contract_ds_write_b16( \
        (__attribute__((address_space(3))) uint16_t *) (sh_stage + q6id_contract_stage_index(_group, 0u, lane)), \
        acc[0] & 0xffffu); \
    q6id_contract_ds_write_b16( \
        (__attribute__((address_space(3))) uint16_t *) (sh_stage + q6id_contract_stage_index(_group, 1u, lane)), \
        acc[1] & 0xffffu); \
} while (0)

#define HRX_Q6ID_CONTRACT_WMMA_COMPACT_ACC(ACC, A_FRAG, B_FRAG) do { \
    asm volatile("v_wmma_f16_16x16x16_f16 %0, %1, %2, %0\n" \
                 : "+v"(ACC) \
                 : "v"(A_FRAG), "v"(B_FRAG) \
                 : "memory"); \
} while (0)

#define HRX_Q6ID_CONTRACT_WMMA_COMPACT_PROD(GROUP_ID, A_FRAG, B_FRAG) do { \
    const unsigned int _group = (GROUP_ID); \
    q6id_contract_u32x4_vec acc; \
    acc[0] = q6id_contract_half_pack2(q6id_contract_prod_value(_group, 0u, lane), q6id_contract_prod_value(_group, 1u, lane)); \
    acc[1] = q6id_contract_half_pack2(q6id_contract_prod_value(_group, 1u, lane), q6id_contract_prod_value(_group, 2u, lane)); \
    acc[2] = q6id_contract_half_pack2(q6id_contract_prod_value(_group, 2u, lane), q6id_contract_prod_value(_group, 3u, lane)); \
    acc[3] = q6id_contract_half_pack2(q6id_contract_prod_value(_group, 3u, lane), q6id_contract_prod_value(_group, 0u, lane)); \
    asm volatile("v_wmma_f16_16x16x16_f16 %0, %1, %2, %0\n" \
                 : "+v"(acc) \
                 : "v"(A_FRAG), "v"(B_FRAG) \
                 : "memory"); \
    q6id_contract_ds_write_b16( \
        (__attribute__((address_space(3))) uint16_t *) (sh_prod_stage + q6id_contract_prod_stage_index(_group, 0u, lane)), \
        acc[0] & 0xffffu); \
    q6id_contract_ds_write_b16( \
        (__attribute__((address_space(3))) uint16_t *) (sh_prod_stage + q6id_contract_prod_stage_index(_group, 1u, lane)), \
        acc[1] & 0xffffu); \
    q6id_contract_ds_write_b16( \
        (__attribute__((address_space(3))) uint16_t *) (sh_prod_stage + q6id_contract_prod_stage_index(_group, 2u, lane)), \
        acc[2] & 0xffffu); \
    q6id_contract_ds_write_b16( \
        (__attribute__((address_space(3))) uint16_t *) (sh_prod_stage + q6id_contract_prod_stage_index(_group, 3u, lane)), \
        acc[3] & 0xffffu); \
} while (0)

#define HRX_Q6ID_CONTRACT_STAGE_FLUSH(GROUP_ID) do { \
    const unsigned int _group = (GROUP_ID); \
    const _Float16 v0 = __builtin_bit_cast( \
        _Float16, static_cast<uint16_t>(q6id_contract_ds_read_u16_d16( \
            (const __attribute__((address_space(3))) uint16_t *) (sh_stage + q6id_contract_stage_index(_group, 0u, lane))))); \
    const _Float16 v1 = __builtin_bit_cast( \
        _Float16, static_cast<uint16_t>(q6id_contract_ds_read_u16_d16( \
            (const __attribute__((address_space(3))) uint16_t *) (sh_stage + q6id_contract_stage_index(_group, 1u, lane))))); \
    q6id_contract_store_value(dst, counts, _group, 0u, lane, static_cast<float>(v0)); \
    q6id_contract_store_value(dst, counts, _group, 1u, lane, static_cast<float>(v1)); \
} while (0)

#define HRX_Q6ID_CONTRACT_PROD_STAGE_FLUSH(GROUP_ID) do { \
    const unsigned int _group = (GROUP_ID); \
    _Pragma("unroll") \
    for (unsigned int _reg = 0; _reg < 4u; ++_reg) { \
        const _Float16 _value = q6id_contract_u16_to_f16( \
            q6id_contract_ds_read_u16_d16( \
                (const __attribute__((address_space(3))) uint16_t *) (sh_prod_stage + q6id_contract_prod_stage_index(_group, _reg, lane)))); \
        q6id_contract_prod_store_value(dst, counts, _group, _reg, lane, static_cast<float>(_value)); \
    } \
} while (0)

#define HRX_Q6ID_CONTRACT_PROD_STAGE_COMPACT_ACC8_INLINE(GROUP_BASE, ACC0, ACC1, ACC2, ACC3, ACC4, ACC5, ACC6, ACC7) do { \
    q6id_contract_prod_stage_compact_acc(sh_prod_stage, (GROUP_BASE) + 0u, lane, (ACC0)); \
    q6id_contract_prod_stage_compact_acc(sh_prod_stage, (GROUP_BASE) + 1u, lane, (ACC1)); \
    q6id_contract_prod_stage_compact_acc(sh_prod_stage, (GROUP_BASE) + 2u, lane, (ACC2)); \
    q6id_contract_prod_stage_compact_acc(sh_prod_stage, (GROUP_BASE) + 3u, lane, (ACC3)); \
    q6id_contract_prod_stage_compact_acc(sh_prod_stage, (GROUP_BASE) + 4u, lane, (ACC4)); \
    q6id_contract_prod_stage_compact_acc(sh_prod_stage, (GROUP_BASE) + 5u, lane, (ACC5)); \
    q6id_contract_prod_stage_compact_acc(sh_prod_stage, (GROUP_BASE) + 6u, lane, (ACC6)); \
    q6id_contract_prod_stage_compact_acc(sh_prod_stage, (GROUP_BASE) + 7u, lane, (ACC7)); \
} while (0)

#define HRX_Q6ID_CONTRACT_WAIT_LGKMCNT(VALUE) \
    asm volatile("s_waitcnt lgkmcnt(" #VALUE ")\n" ::: "memory")

#define HRX_Q6ID_CONTRACT_WAIT_LGKMCNT_DEPS(VALUE, A_FRAG, B_FRAG) do { \
    const q6id_contract_u32x8_vec _a_raw = __builtin_bit_cast(q6id_contract_u32x8_vec, (A_FRAG)); \
    const q6id_contract_u32x8_vec _b_raw = __builtin_bit_cast(q6id_contract_u32x8_vec, (B_FRAG)); \
    asm volatile("s_waitcnt lgkmcnt(" #VALUE ")\n" \
                 :: "v"(_a_raw[0]), "v"(_a_raw[7]), "v"(_b_raw[0]), "v"(_b_raw[7]) \
                 : "memory"); \
} while (0)

extern "C" __global__ __launch_bounds__(256, 1)
void q6_id_subgroup_contract_direct_probe(float * dst, unsigned int * counts) {
    const unsigned int lane = __builtin_amdgcn_workitem_id_x() & 63u;
    const unsigned int wave = __builtin_amdgcn_workitem_id_x() >> 6u;
    if (wave != 0u) {
        return;
    }

#pragma unroll
    for (unsigned int group = 0; group < HRX_Q6ID_CONTRACT_GROUPS; ++group) {
        q6id_contract_store_value(dst, counts, group, 0u, lane, q6id_contract_seed(group, 0u, lane));
        q6id_contract_store_value(dst, counts, group, 1u, lane, q6id_contract_seed(group, 1u, lane));
    }
}

extern "C" __global__ __launch_bounds__(256, 1)
void q6_id_subgroup_contract_prodaddr_direct_probe(float * dst, unsigned int * counts) {
    const unsigned int lane = __builtin_amdgcn_workitem_id_x() & 63u;
    const unsigned int wave = __builtin_amdgcn_workitem_id_x() >> 6u;
    if (wave != 0u) {
        return;
    }

#pragma unroll
    for (unsigned int group = 0; group < HRX_Q6ID_CONTRACT_GROUPS; ++group) {
        q6id_contract_prod_store_group(dst, counts, group, lane, 0.0f);
    }
}

extern "C" __global__ __launch_bounds__(256, 1)
void q6_id_subgroup_contract_prodaddr_radv96_duplicate_probe(float * dst, unsigned int * counts) {
    __shared__ uint16_t sh_stage[6 * 64 * 4];

    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & 63u;
    const unsigned int wave = tid >> 6u;
    if (wave != 0u) {
        return;
    }

#pragma unroll
    for (unsigned int group = 0; group < 8u; ++group) {
        q6id_contract_prod_store_group(dst, counts, group, lane, 0.0f);
    }

#pragma unroll
    for (unsigned int group = 8u; group < 14u; ++group) {
        const unsigned int stage_base = (group - 8u) * 64u * 4u + lane * 4u;
#pragma unroll
        for (unsigned int reg = 0; reg < 4u; ++reg) {
            sh_stage[stage_base + reg] = static_cast<uint16_t>(0x1000u + group * 16u + reg);
        }
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
#pragma unroll
    for (unsigned int group = 8u; group < 14u; ++group) {
        q6id_contract_prod_store_group(dst, counts, group, lane, 0.0f);
    }

#pragma unroll
    for (unsigned int group = 14u; group < 20u; ++group) {
        const unsigned int acc_group = group < 16u ? group : group - 8u;
        const unsigned int stage_base = (group - 14u) * 64u * 4u + lane * 4u;
#pragma unroll
        for (unsigned int reg = 0; reg < 4u; ++reg) {
            sh_stage[stage_base + reg] = static_cast<uint16_t>(0x2000u + acc_group * 16u + reg);
        }
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
#pragma unroll
    for (unsigned int group = 14u; group < 20u; ++group) {
        if (group < 16u) {
            q6id_contract_prod_store_group(dst, counts, group, lane, 0.0f);
        } else {
            q6id_contract_prod_store_group(dst, counts, group - 8u, lane, 0.0f);
        }
    }

#pragma unroll
    for (unsigned int group = 20u; group < 24u; ++group) {
        const unsigned int acc_group = group - 8u;
        const unsigned int stage_base = (group - 20u) * 64u * 4u + lane * 4u;
#pragma unroll
        for (unsigned int reg = 0; reg < 4u; ++reg) {
            sh_stage[stage_base + reg] = static_cast<uint16_t>(0x3000u + acc_group * 16u + reg);
        }
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
#pragma unroll
    for (unsigned int group = 20u; group < 24u; ++group) {
        q6id_contract_prod_store_group(dst, counts, group - 8u, lane, 0.0f);
    }
}

extern "C" __global__ __launch_bounds__(256, 1)
void q6_id_subgroup_contract_banked_probe(float * dst, unsigned int * counts) {
    __shared__ uint64_t sh_frag[12 * 256];
    __shared__ uint16_t sh_stage[HRX_Q6ID_CONTRACT_VALUES];

    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & 63u;
    const unsigned int wave = tid >> 6u;

    if (wave == 0u) {
        q6id_contract_init_banked_fragments(sh_frag, lane);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (wave == 0u) {
    q6id_contract_sink_fragment(q6id_contract_load_fragment(
        (const __attribute__((address_space(3))) uint64_t *) sh_frag, 0u, lane));
    const q6id_contract_half16_vec a0 = q6id_contract_load_fragment(
        (const __attribute__((address_space(3))) uint64_t *) sh_frag, 0u, lane);
    const q6id_contract_half16_vec b0 = q6id_contract_load_fragment(
        (const __attribute__((address_space(3))) uint64_t *) sh_frag, 8u, lane);
    const q6id_contract_half16_vec b1 = q6id_contract_load_fragment(
        (const __attribute__((address_space(3))) uint64_t *) sh_frag, 9u, lane);
    const q6id_contract_half16_vec a1 = q6id_contract_load_fragment(
        (const __attribute__((address_space(3))) uint64_t *) sh_frag, 1u, lane);
    const q6id_contract_half16_vec a2 = q6id_contract_load_fragment(
        (const __attribute__((address_space(3))) uint64_t *) sh_frag, 2u, lane);
    const q6id_contract_half16_vec a3 = q6id_contract_load_fragment(
        (const __attribute__((address_space(3))) uint64_t *) sh_frag, 3u, lane);
    const q6id_contract_half16_vec a4 = q6id_contract_load_fragment(
        (const __attribute__((address_space(3))) uint64_t *) sh_frag, 4u, lane);
    const q6id_contract_half16_vec b2 = q6id_contract_load_fragment(
        (const __attribute__((address_space(3))) uint64_t *) sh_frag, 10u, lane);
    const q6id_contract_half16_vec b3 = q6id_contract_load_fragment(
        (const __attribute__((address_space(3))) uint64_t *) sh_frag, 11u, lane);
    const q6id_contract_half16_vec a5 = q6id_contract_load_fragment(
        (const __attribute__((address_space(3))) uint64_t *) sh_frag, 5u, lane);
    const q6id_contract_half16_vec a6 = q6id_contract_load_fragment(
        (const __attribute__((address_space(3))) uint64_t *) sh_frag, 6u, lane);
    const q6id_contract_half16_vec a7 = q6id_contract_load_fragment(
        (const __attribute__((address_space(3))) uint64_t *) sh_frag, 7u, lane);

    HRX_Q6ID_CONTRACT_WAIT_LGKMCNT_DEPS(40, a0, b0);
    HRX_Q6ID_CONTRACT_WMMA(0u, a0, b0);
    HRX_Q6ID_CONTRACT_WAIT_LGKMCNT_DEPS(36, a0, b1);
    HRX_Q6ID_CONTRACT_WMMA(1u, a0, b1);
    HRX_Q6ID_CONTRACT_WAIT_LGKMCNT_DEPS(32, a1, b0);
    HRX_Q6ID_CONTRACT_WMMA(2u, a1, b0);
    HRX_Q6ID_CONTRACT_WMMA(3u, a1, b1);
    HRX_Q6ID_CONTRACT_WAIT_LGKMCNT_DEPS(28, a2, b0);
    HRX_Q6ID_CONTRACT_WMMA(4u, a2, b0);
    HRX_Q6ID_CONTRACT_WMMA(5u, a2, b1);
    HRX_Q6ID_CONTRACT_WAIT_LGKMCNT_DEPS(24, a3, b1);
    HRX_Q6ID_CONTRACT_WMMA(6u, a3, b1);
    HRX_Q6ID_CONTRACT_WMMA(7u, a3, b0);
    HRX_Q6ID_CONTRACT_WAIT_LGKMCNT_DEPS(16, a4, b2);
    HRX_Q6ID_CONTRACT_WMMA(8u, a4, b2);
    HRX_Q6ID_CONTRACT_WAIT_LGKMCNT_DEPS(12, a4, b3);
    HRX_Q6ID_CONTRACT_WMMA(9u, a4, b3);
    HRX_Q6ID_CONTRACT_WAIT_LGKMCNT_DEPS(8, a5, b2);
    HRX_Q6ID_CONTRACT_WMMA(10u, a5, b2);
    HRX_Q6ID_CONTRACT_WMMA(11u, a5, b3);
    HRX_Q6ID_CONTRACT_WAIT_LGKMCNT_DEPS(4, a6, b2);
    HRX_Q6ID_CONTRACT_WMMA(12u, a6, b2);
    HRX_Q6ID_CONTRACT_WMMA(13u, a6, b3);
    HRX_Q6ID_CONTRACT_WAIT_LGKMCNT_DEPS(0, a7, b3);
    HRX_Q6ID_CONTRACT_WMMA(14u, a7, b3);
    HRX_Q6ID_CONTRACT_WMMA(15u, a7, b2);
    }

    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (wave == 0u) {
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(0u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(1u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(2u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(3u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(4u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(5u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(6u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(7u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(8u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(9u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(10u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(11u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(12u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(13u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(14u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(15u);
    }
}

extern "C" __global__ __launch_bounds__(256, 1)
void q6_id_subgroup_contract_banked_compact_probe(float * dst, unsigned int * counts) {
    __shared__ uint64_t sh_frag[12 * 256];
    __shared__ uint16_t sh_stage[HRX_Q6ID_CONTRACT_VALUES];

    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & 63u;
    const unsigned int wave = tid >> 6u;

    if (wave == 0u) {
        q6id_contract_init_banked_fragments(sh_frag, lane);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (wave == 0u) {
    q6id_contract_sink_fragment(q6id_contract_load_fragment(
        (const __attribute__((address_space(3))) uint64_t *) sh_frag, 0u, lane));
    const q6id_contract_half16_vec a0 = q6id_contract_load_fragment(
        (const __attribute__((address_space(3))) uint64_t *) sh_frag, 0u, lane);
    const q6id_contract_half16_vec b0 = q6id_contract_load_fragment(
        (const __attribute__((address_space(3))) uint64_t *) sh_frag, 8u, lane);
    const q6id_contract_half16_vec b1 = q6id_contract_load_fragment(
        (const __attribute__((address_space(3))) uint64_t *) sh_frag, 9u, lane);
    const q6id_contract_half16_vec a1 = q6id_contract_load_fragment(
        (const __attribute__((address_space(3))) uint64_t *) sh_frag, 1u, lane);
    const q6id_contract_half16_vec a2 = q6id_contract_load_fragment(
        (const __attribute__((address_space(3))) uint64_t *) sh_frag, 2u, lane);
    const q6id_contract_half16_vec a3 = q6id_contract_load_fragment(
        (const __attribute__((address_space(3))) uint64_t *) sh_frag, 3u, lane);
    const q6id_contract_half16_vec a4 = q6id_contract_load_fragment(
        (const __attribute__((address_space(3))) uint64_t *) sh_frag, 4u, lane);
    const q6id_contract_half16_vec b2 = q6id_contract_load_fragment(
        (const __attribute__((address_space(3))) uint64_t *) sh_frag, 10u, lane);
    const q6id_contract_half16_vec b3 = q6id_contract_load_fragment(
        (const __attribute__((address_space(3))) uint64_t *) sh_frag, 11u, lane);
    const q6id_contract_half16_vec a5 = q6id_contract_load_fragment(
        (const __attribute__((address_space(3))) uint64_t *) sh_frag, 5u, lane);
    const q6id_contract_half16_vec a6 = q6id_contract_load_fragment(
        (const __attribute__((address_space(3))) uint64_t *) sh_frag, 6u, lane);
    const q6id_contract_half16_vec a7 = q6id_contract_load_fragment(
        (const __attribute__((address_space(3))) uint64_t *) sh_frag, 7u, lane);

    HRX_Q6ID_CONTRACT_WAIT_LGKMCNT_DEPS(40, a0, b0);
    HRX_Q6ID_CONTRACT_WMMA_COMPACT(0u, a0, b0);
    HRX_Q6ID_CONTRACT_WAIT_LGKMCNT_DEPS(36, a0, b1);
    HRX_Q6ID_CONTRACT_WMMA_COMPACT(1u, a0, b1);
    HRX_Q6ID_CONTRACT_WAIT_LGKMCNT_DEPS(32, a1, b0);
    HRX_Q6ID_CONTRACT_WMMA_COMPACT(2u, a1, b0);
    HRX_Q6ID_CONTRACT_WMMA_COMPACT(3u, a1, b1);
    HRX_Q6ID_CONTRACT_WAIT_LGKMCNT_DEPS(28, a2, b0);
    HRX_Q6ID_CONTRACT_WMMA_COMPACT(4u, a2, b0);
    HRX_Q6ID_CONTRACT_WMMA_COMPACT(5u, a2, b1);
    HRX_Q6ID_CONTRACT_WAIT_LGKMCNT_DEPS(24, a3, b1);
    HRX_Q6ID_CONTRACT_WMMA_COMPACT(6u, a3, b1);
    HRX_Q6ID_CONTRACT_WMMA_COMPACT(7u, a3, b0);
    HRX_Q6ID_CONTRACT_WAIT_LGKMCNT_DEPS(16, a4, b2);
    HRX_Q6ID_CONTRACT_WMMA_COMPACT(8u, a4, b2);
    HRX_Q6ID_CONTRACT_WAIT_LGKMCNT_DEPS(12, a4, b3);
    HRX_Q6ID_CONTRACT_WMMA_COMPACT(9u, a4, b3);
    HRX_Q6ID_CONTRACT_WAIT_LGKMCNT_DEPS(8, a5, b2);
    HRX_Q6ID_CONTRACT_WMMA_COMPACT(10u, a5, b2);
    HRX_Q6ID_CONTRACT_WMMA_COMPACT(11u, a5, b3);
    HRX_Q6ID_CONTRACT_WAIT_LGKMCNT_DEPS(4, a6, b2);
    HRX_Q6ID_CONTRACT_WMMA_COMPACT(12u, a6, b2);
    HRX_Q6ID_CONTRACT_WMMA_COMPACT(13u, a6, b3);
    HRX_Q6ID_CONTRACT_WAIT_LGKMCNT_DEPS(0, a7, b3);
    HRX_Q6ID_CONTRACT_WMMA_COMPACT(14u, a7, b3);
    HRX_Q6ID_CONTRACT_WMMA_COMPACT(15u, a7, b2);
    }

    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (wave == 0u) {
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(0u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(1u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(2u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(3u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(4u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(5u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(6u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(7u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(8u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(9u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(10u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(11u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(12u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(13u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(14u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(15u);
    }
}

extern "C" __global__ __launch_bounds__(256, 1)
void q6_id_subgroup_contract_radv_issue_compact_probe(float * dst, unsigned int * counts) {
    __shared__ uint64_t sh_frag[12 * 256];
    __shared__ uint16_t sh_stage[HRX_Q6ID_CONTRACT_VALUES];

    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & 63u;
    const unsigned int wave = tid >> 6u;

    if (wave == 0u) {
        q6id_contract_init_banked_fragments(sh_frag, lane);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (wave == 0u) {
    q6id_contract_sink_fragment(q6id_contract_load_fragment(
        (const __attribute__((address_space(3))) uint64_t *) sh_frag, 0u, lane));
    const q6id_contract_half16_vec a0 = q6id_contract_load_fragment(
        (const __attribute__((address_space(3))) uint64_t *) sh_frag, 0u, lane);
    const q6id_contract_half16_vec b0 = q6id_contract_load_fragment(
        (const __attribute__((address_space(3))) uint64_t *) sh_frag, 8u, lane);
    const q6id_contract_half16_vec b1 = q6id_contract_load_fragment(
        (const __attribute__((address_space(3))) uint64_t *) sh_frag, 9u, lane);
    const q6id_contract_half16_vec a1 = q6id_contract_load_fragment(
        (const __attribute__((address_space(3))) uint64_t *) sh_frag, 1u, lane);
    const q6id_contract_half16_vec a2 = q6id_contract_load_fragment(
        (const __attribute__((address_space(3))) uint64_t *) sh_frag, 2u, lane);
    const q6id_contract_half16_vec a3 = q6id_contract_load_fragment(
        (const __attribute__((address_space(3))) uint64_t *) sh_frag, 3u, lane);
    const q6id_contract_half16_vec a4 = q6id_contract_load_fragment(
        (const __attribute__((address_space(3))) uint64_t *) sh_frag, 4u, lane);
    const q6id_contract_half16_vec b2 = q6id_contract_load_fragment(
        (const __attribute__((address_space(3))) uint64_t *) sh_frag, 10u, lane);
    const q6id_contract_half16_vec b3 = q6id_contract_load_fragment(
        (const __attribute__((address_space(3))) uint64_t *) sh_frag, 11u, lane);
    const q6id_contract_half16_vec a5 = q6id_contract_load_fragment(
        (const __attribute__((address_space(3))) uint64_t *) sh_frag, 5u, lane);
    const q6id_contract_half16_vec a6 = q6id_contract_load_fragment(
        (const __attribute__((address_space(3))) uint64_t *) sh_frag, 6u, lane);
    const q6id_contract_half16_vec a7 = q6id_contract_load_fragment(
        (const __attribute__((address_space(3))) uint64_t *) sh_frag, 7u, lane);

    q6id_contract_u32x4_vec acc0 = q6id_contract_init_compact_acc(0u, lane);
    q6id_contract_u32x4_vec acc1 = q6id_contract_init_compact_acc(1u, lane);
    q6id_contract_u32x4_vec acc2 = q6id_contract_init_compact_acc(2u, lane);
    q6id_contract_u32x4_vec acc3 = q6id_contract_init_compact_acc(3u, lane);
    q6id_contract_u32x4_vec acc4 = q6id_contract_init_compact_acc(4u, lane);
    q6id_contract_u32x4_vec acc5 = q6id_contract_init_compact_acc(5u, lane);
    q6id_contract_u32x4_vec acc6 = q6id_contract_init_compact_acc(6u, lane);
    q6id_contract_u32x4_vec acc7 = q6id_contract_init_compact_acc(7u, lane);

    HRX_Q6ID_CONTRACT_WAIT_LGKMCNT_DEPS(40, a0, b0);
    HRX_Q6ID_CONTRACT_WMMA_COMPACT_ACC(acc0, a0, b0);
    HRX_Q6ID_CONTRACT_WAIT_LGKMCNT_DEPS(36, a0, b1);
    HRX_Q6ID_CONTRACT_WMMA_COMPACT_ACC(acc1, a0, b1);
    HRX_Q6ID_CONTRACT_WAIT_LGKMCNT_DEPS(32, a1, b0);
    HRX_Q6ID_CONTRACT_WMMA_COMPACT_ACC(acc2, a1, b0);
    HRX_Q6ID_CONTRACT_WMMA_COMPACT_ACC(acc3, a1, b1);
    HRX_Q6ID_CONTRACT_WAIT_LGKMCNT_DEPS(28, a2, b0);
    HRX_Q6ID_CONTRACT_WMMA_COMPACT_ACC(acc4, a2, b0);
    HRX_Q6ID_CONTRACT_WMMA_COMPACT_ACC(acc5, a2, b1);
    HRX_Q6ID_CONTRACT_WAIT_LGKMCNT_DEPS(24, a3, b0);
    HRX_Q6ID_CONTRACT_WMMA_COMPACT_ACC(acc6, a3, b0);
    HRX_Q6ID_CONTRACT_WMMA_COMPACT_ACC(acc7, a3, b1);
    HRX_Q6ID_CONTRACT_WAIT_LGKMCNT_DEPS(16, a4, b2);
    HRX_Q6ID_CONTRACT_WMMA_COMPACT_ACC(acc0, a4, b2);
    HRX_Q6ID_CONTRACT_WAIT_LGKMCNT_DEPS(7, a4, b3);
    HRX_Q6ID_CONTRACT_WMMA_COMPACT_ACC(acc1, a4, b3);
    HRX_Q6ID_CONTRACT_WMMA_COMPACT_ACC(acc2, a5, b2);
    HRX_Q6ID_CONTRACT_WAIT_LGKMCNT_DEPS(6, a5, b3);
    HRX_Q6ID_CONTRACT_WMMA_COMPACT_ACC(acc3, a5, b3);
    HRX_Q6ID_CONTRACT_WAIT_LGKMCNT_DEPS(1, a6, b2);
    HRX_Q6ID_CONTRACT_WMMA_COMPACT_ACC(acc4, a6, b2);
    HRX_Q6ID_CONTRACT_WMMA_COMPACT_ACC(acc5, a6, b3);
    HRX_Q6ID_CONTRACT_WAIT_LGKMCNT_DEPS(0, a7, b3);
    HRX_Q6ID_CONTRACT_WMMA_COMPACT_ACC(acc7, a7, b3);
    HRX_Q6ID_CONTRACT_WMMA_COMPACT_ACC(acc6, a7, b2);

    q6id_contract_stage_compact_acc(sh_stage, 0u, lane, acc0);
    q6id_contract_stage_compact_acc(sh_stage, 1u, lane, acc1);
    q6id_contract_stage_compact_acc(sh_stage, 2u, lane, acc2);
    q6id_contract_stage_compact_acc(sh_stage, 3u, lane, acc3);
    q6id_contract_stage_compact_acc(sh_stage, 4u, lane, acc4);
    q6id_contract_stage_compact_acc(sh_stage, 5u, lane, acc5);
    q6id_contract_stage_compact_acc(sh_stage, 6u, lane, acc6);
    q6id_contract_stage_compact_acc(sh_stage, 7u, lane, acc7);
    q6id_contract_stage_compact_acc(sh_stage, 8u, lane, acc0);
    q6id_contract_stage_compact_acc(sh_stage, 9u, lane, acc1);
    q6id_contract_stage_compact_acc(sh_stage, 10u, lane, acc2);
    q6id_contract_stage_compact_acc(sh_stage, 11u, lane, acc3);
    q6id_contract_stage_compact_acc(sh_stage, 12u, lane, acc4);
    q6id_contract_stage_compact_acc(sh_stage, 13u, lane, acc5);
    q6id_contract_stage_compact_acc(sh_stage, 14u, lane, acc7);
    q6id_contract_stage_compact_acc(sh_stage, 15u, lane, acc6);
    }

    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (wave == 0u) {
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(0u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(1u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(2u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(3u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(4u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(5u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(6u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(7u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(8u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(9u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(10u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(11u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(12u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(13u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(14u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(15u);
    }
}

extern "C" __global__ __launch_bounds__(256, 1)
void q6_id_subgroup_contract_prodaddr_radv_issue_compact_probe(float * dst, unsigned int * counts) {
    __shared__ uint64_t sh_frag[12 * 256];
    __shared__ uint16_t sh_prod_stage[16 * 64 * 4];

    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & 63u;
    const unsigned int wave = tid >> 6u;

    if (wave == 0u) {
        q6id_contract_init_banked_fragments(sh_frag, lane);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (wave == 0u) {
        q6id_contract_sink_fragment(q6id_contract_load_fragment(
            (const __attribute__((address_space(3))) uint64_t *) sh_frag, 0u, lane));
        const q6id_contract_half16_vec a0 = q6id_contract_load_fragment(
            (const __attribute__((address_space(3))) uint64_t *) sh_frag, 0u, lane);
        const q6id_contract_half16_vec b0 = q6id_contract_load_fragment(
            (const __attribute__((address_space(3))) uint64_t *) sh_frag, 8u, lane);
        const q6id_contract_half16_vec b1 = q6id_contract_load_fragment(
            (const __attribute__((address_space(3))) uint64_t *) sh_frag, 9u, lane);
        const q6id_contract_half16_vec a1 = q6id_contract_load_fragment(
            (const __attribute__((address_space(3))) uint64_t *) sh_frag, 1u, lane);
        const q6id_contract_half16_vec a2 = q6id_contract_load_fragment(
            (const __attribute__((address_space(3))) uint64_t *) sh_frag, 2u, lane);
        const q6id_contract_half16_vec a3 = q6id_contract_load_fragment(
            (const __attribute__((address_space(3))) uint64_t *) sh_frag, 3u, lane);
        const q6id_contract_half16_vec a4 = q6id_contract_load_fragment(
            (const __attribute__((address_space(3))) uint64_t *) sh_frag, 4u, lane);
        const q6id_contract_half16_vec b2 = q6id_contract_load_fragment(
            (const __attribute__((address_space(3))) uint64_t *) sh_frag, 10u, lane);
        const q6id_contract_half16_vec b3 = q6id_contract_load_fragment(
            (const __attribute__((address_space(3))) uint64_t *) sh_frag, 11u, lane);
        const q6id_contract_half16_vec a5 = q6id_contract_load_fragment(
            (const __attribute__((address_space(3))) uint64_t *) sh_frag, 5u, lane);
        const q6id_contract_half16_vec a6 = q6id_contract_load_fragment(
            (const __attribute__((address_space(3))) uint64_t *) sh_frag, 6u, lane);
        const q6id_contract_half16_vec a7 = q6id_contract_load_fragment(
            (const __attribute__((address_space(3))) uint64_t *) sh_frag, 7u, lane);

        q6id_contract_u32x4_vec acc0 = q6id_contract_init_compact_prod_acc(0u, lane);
        q6id_contract_u32x4_vec acc1 = q6id_contract_init_compact_prod_acc(1u, lane);
        q6id_contract_u32x4_vec acc2 = q6id_contract_init_compact_prod_acc(2u, lane);
        q6id_contract_u32x4_vec acc3 = q6id_contract_init_compact_prod_acc(3u, lane);
        q6id_contract_u32x4_vec acc4 = q6id_contract_init_compact_prod_acc(4u, lane);
        q6id_contract_u32x4_vec acc5 = q6id_contract_init_compact_prod_acc(5u, lane);
        q6id_contract_u32x4_vec acc6 = q6id_contract_init_compact_prod_acc(6u, lane);
        q6id_contract_u32x4_vec acc7 = q6id_contract_init_compact_prod_acc(7u, lane);

        HRX_Q6ID_CONTRACT_WAIT_LGKMCNT_DEPS(40, a0, b0);
        HRX_Q6ID_CONTRACT_WMMA_COMPACT_ACC(acc0, a0, b0);
        HRX_Q6ID_CONTRACT_WAIT_LGKMCNT_DEPS(36, a0, b1);
        HRX_Q6ID_CONTRACT_WMMA_COMPACT_ACC(acc1, a0, b1);
        HRX_Q6ID_CONTRACT_WAIT_LGKMCNT_DEPS(32, a1, b0);
        HRX_Q6ID_CONTRACT_WMMA_COMPACT_ACC(acc2, a1, b0);
        HRX_Q6ID_CONTRACT_WMMA_COMPACT_ACC(acc3, a1, b1);
        HRX_Q6ID_CONTRACT_WAIT_LGKMCNT_DEPS(28, a2, b0);
        HRX_Q6ID_CONTRACT_WMMA_COMPACT_ACC(acc4, a2, b0);
        HRX_Q6ID_CONTRACT_WMMA_COMPACT_ACC(acc5, a2, b1);
        HRX_Q6ID_CONTRACT_WAIT_LGKMCNT_DEPS(24, a3, b0);
        HRX_Q6ID_CONTRACT_WMMA_COMPACT_ACC(acc6, a3, b0);
        HRX_Q6ID_CONTRACT_WMMA_COMPACT_ACC(acc7, a3, b1);

        q6id_contract_prod_stage_compact_acc(sh_prod_stage, 0u, lane, acc0);
        q6id_contract_prod_stage_compact_acc(sh_prod_stage, 1u, lane, acc1);
        q6id_contract_prod_stage_compact_acc(sh_prod_stage, 2u, lane, acc2);
        q6id_contract_prod_stage_compact_acc(sh_prod_stage, 3u, lane, acc3);
        q6id_contract_prod_stage_compact_acc(sh_prod_stage, 4u, lane, acc4);
        q6id_contract_prod_stage_compact_acc(sh_prod_stage, 5u, lane, acc5);
        q6id_contract_prod_stage_compact_acc(sh_prod_stage, 6u, lane, acc6);
        q6id_contract_prod_stage_compact_acc(sh_prod_stage, 7u, lane, acc7);

        acc0 = q6id_contract_init_compact_prod_acc(8u, lane);
        acc1 = q6id_contract_init_compact_prod_acc(9u, lane);
        acc2 = q6id_contract_init_compact_prod_acc(10u, lane);
        acc3 = q6id_contract_init_compact_prod_acc(11u, lane);
        acc4 = q6id_contract_init_compact_prod_acc(12u, lane);
        acc5 = q6id_contract_init_compact_prod_acc(13u, lane);
        acc7 = q6id_contract_init_compact_prod_acc(14u, lane);
        acc6 = q6id_contract_init_compact_prod_acc(15u, lane);

        HRX_Q6ID_CONTRACT_WAIT_LGKMCNT_DEPS(16, a4, b2);
        HRX_Q6ID_CONTRACT_WMMA_COMPACT_ACC(acc0, a4, b2);
        HRX_Q6ID_CONTRACT_WAIT_LGKMCNT_DEPS(7, a4, b3);
        HRX_Q6ID_CONTRACT_WMMA_COMPACT_ACC(acc1, a4, b3);
        HRX_Q6ID_CONTRACT_WMMA_COMPACT_ACC(acc2, a5, b2);
        HRX_Q6ID_CONTRACT_WAIT_LGKMCNT_DEPS(6, a5, b3);
        HRX_Q6ID_CONTRACT_WMMA_COMPACT_ACC(acc3, a5, b3);
        HRX_Q6ID_CONTRACT_WAIT_LGKMCNT_DEPS(1, a6, b2);
        HRX_Q6ID_CONTRACT_WMMA_COMPACT_ACC(acc4, a6, b2);
        HRX_Q6ID_CONTRACT_WMMA_COMPACT_ACC(acc5, a6, b3);
        HRX_Q6ID_CONTRACT_WAIT_LGKMCNT_DEPS(0, a7, b3);
        HRX_Q6ID_CONTRACT_WMMA_COMPACT_ACC(acc7, a7, b3);
        HRX_Q6ID_CONTRACT_WMMA_COMPACT_ACC(acc6, a7, b2);

        q6id_contract_prod_stage_compact_acc(sh_prod_stage, 8u, lane, acc0);
        q6id_contract_prod_stage_compact_acc(sh_prod_stage, 9u, lane, acc1);
        q6id_contract_prod_stage_compact_acc(sh_prod_stage, 10u, lane, acc2);
        q6id_contract_prod_stage_compact_acc(sh_prod_stage, 11u, lane, acc3);
        q6id_contract_prod_stage_compact_acc(sh_prod_stage, 12u, lane, acc4);
        q6id_contract_prod_stage_compact_acc(sh_prod_stage, 13u, lane, acc5);
        q6id_contract_prod_stage_compact_acc(sh_prod_stage, 14u, lane, acc7);
        q6id_contract_prod_stage_compact_acc(sh_prod_stage, 15u, lane, acc6);
    }

    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (wave == 0u) {
        HRX_Q6ID_CONTRACT_PROD_STAGE_FLUSH(0u);
        HRX_Q6ID_CONTRACT_PROD_STAGE_FLUSH(1u);
        HRX_Q6ID_CONTRACT_PROD_STAGE_FLUSH(2u);
        HRX_Q6ID_CONTRACT_PROD_STAGE_FLUSH(3u);
        HRX_Q6ID_CONTRACT_PROD_STAGE_FLUSH(4u);
        HRX_Q6ID_CONTRACT_PROD_STAGE_FLUSH(5u);
        HRX_Q6ID_CONTRACT_PROD_STAGE_FLUSH(6u);
        HRX_Q6ID_CONTRACT_PROD_STAGE_FLUSH(7u);
        HRX_Q6ID_CONTRACT_PROD_STAGE_FLUSH(8u);
        HRX_Q6ID_CONTRACT_PROD_STAGE_FLUSH(9u);
        HRX_Q6ID_CONTRACT_PROD_STAGE_FLUSH(10u);
        HRX_Q6ID_CONTRACT_PROD_STAGE_FLUSH(11u);
        HRX_Q6ID_CONTRACT_PROD_STAGE_FLUSH(12u);
        HRX_Q6ID_CONTRACT_PROD_STAGE_FLUSH(13u);
        HRX_Q6ID_CONTRACT_PROD_STAGE_FLUSH(14u);
        HRX_Q6ID_CONTRACT_PROD_STAGE_FLUSH(15u);
    }
}

extern "C" __global__ __launch_bounds__(256, 1)
void q6_id_subgroup_contract_prodaddr_radv_issue_compact_helper_probe(float * dst, unsigned int * counts) {
    __shared__ uint64_t sh_frag[12 * 256];
    __shared__ uint16_t sh_prod_stage[16 * 64 * 4];

    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & 63u;
    const unsigned int wave = tid >> 6u;

    if (wave == 0u) {
        q6id_contract_init_banked_fragments(sh_frag, lane);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (wave == 0u) {
        q6id_contract_sink_fragment(q6id_contract_load_fragment(
            (const __attribute__((address_space(3))) uint64_t *) sh_frag, 0u, lane));
        const q6id_contract_half16_vec a0 = q6id_contract_load_fragment(
            (const __attribute__((address_space(3))) uint64_t *) sh_frag, 0u, lane);
        const q6id_contract_half16_vec b0 = q6id_contract_load_fragment(
            (const __attribute__((address_space(3))) uint64_t *) sh_frag, 8u, lane);
        const q6id_contract_half16_vec b1 = q6id_contract_load_fragment(
            (const __attribute__((address_space(3))) uint64_t *) sh_frag, 9u, lane);
        const q6id_contract_half16_vec a1 = q6id_contract_load_fragment(
            (const __attribute__((address_space(3))) uint64_t *) sh_frag, 1u, lane);
        const q6id_contract_half16_vec a2 = q6id_contract_load_fragment(
            (const __attribute__((address_space(3))) uint64_t *) sh_frag, 2u, lane);
        const q6id_contract_half16_vec a3 = q6id_contract_load_fragment(
            (const __attribute__((address_space(3))) uint64_t *) sh_frag, 3u, lane);
        const q6id_contract_half16_vec a4 = q6id_contract_load_fragment(
            (const __attribute__((address_space(3))) uint64_t *) sh_frag, 4u, lane);
        const q6id_contract_half16_vec b2 = q6id_contract_load_fragment(
            (const __attribute__((address_space(3))) uint64_t *) sh_frag, 10u, lane);
        const q6id_contract_half16_vec b3 = q6id_contract_load_fragment(
            (const __attribute__((address_space(3))) uint64_t *) sh_frag, 11u, lane);
        const q6id_contract_half16_vec a5 = q6id_contract_load_fragment(
            (const __attribute__((address_space(3))) uint64_t *) sh_frag, 5u, lane);
        const q6id_contract_half16_vec a6 = q6id_contract_load_fragment(
            (const __attribute__((address_space(3))) uint64_t *) sh_frag, 6u, lane);
        const q6id_contract_half16_vec a7 = q6id_contract_load_fragment(
            (const __attribute__((address_space(3))) uint64_t *) sh_frag, 7u, lane);

        {
            q6id_contract_u32x4_vec acc0 = q6id_contract_init_compact_prod_acc(0u, lane);
            q6id_contract_u32x4_vec acc1 = q6id_contract_init_compact_prod_acc(1u, lane);
            q6id_contract_u32x4_vec acc2 = q6id_contract_init_compact_prod_acc(2u, lane);
            q6id_contract_u32x4_vec acc3 = q6id_contract_init_compact_prod_acc(3u, lane);
            q6id_contract_u32x4_vec acc4 = q6id_contract_init_compact_prod_acc(4u, lane);
            q6id_contract_u32x4_vec acc5 = q6id_contract_init_compact_prod_acc(5u, lane);
            q6id_contract_u32x4_vec acc6 = q6id_contract_init_compact_prod_acc(6u, lane);
            q6id_contract_u32x4_vec acc7 = q6id_contract_init_compact_prod_acc(7u, lane);

            HRX_Q6ID_CONTRACT_WAIT_LGKMCNT_DEPS(40, a0, b0);
            HRX_Q6ID_CONTRACT_WMMA_COMPACT_ACC(acc0, a0, b0);
            HRX_Q6ID_CONTRACT_WAIT_LGKMCNT_DEPS(36, a0, b1);
            HRX_Q6ID_CONTRACT_WMMA_COMPACT_ACC(acc1, a0, b1);
            HRX_Q6ID_CONTRACT_WAIT_LGKMCNT_DEPS(32, a1, b0);
            HRX_Q6ID_CONTRACT_WMMA_COMPACT_ACC(acc2, a1, b0);
            HRX_Q6ID_CONTRACT_WMMA_COMPACT_ACC(acc3, a1, b1);
            HRX_Q6ID_CONTRACT_WAIT_LGKMCNT_DEPS(28, a2, b0);
            HRX_Q6ID_CONTRACT_WMMA_COMPACT_ACC(acc4, a2, b0);
            HRX_Q6ID_CONTRACT_WMMA_COMPACT_ACC(acc5, a2, b1);
            HRX_Q6ID_CONTRACT_WAIT_LGKMCNT_DEPS(24, a3, b0);
            HRX_Q6ID_CONTRACT_WMMA_COMPACT_ACC(acc6, a3, b0);
            HRX_Q6ID_CONTRACT_WMMA_COMPACT_ACC(acc7, a3, b1);

            q6id_contract_prod_stage_compact_acc8(
                sh_prod_stage, 0u, lane, acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7);
        }

        {
            q6id_contract_u32x4_vec acc0 = q6id_contract_init_compact_prod_acc(8u, lane);
            q6id_contract_u32x4_vec acc1 = q6id_contract_init_compact_prod_acc(9u, lane);
            q6id_contract_u32x4_vec acc2 = q6id_contract_init_compact_prod_acc(10u, lane);
            q6id_contract_u32x4_vec acc3 = q6id_contract_init_compact_prod_acc(11u, lane);
            q6id_contract_u32x4_vec acc4 = q6id_contract_init_compact_prod_acc(12u, lane);
            q6id_contract_u32x4_vec acc5 = q6id_contract_init_compact_prod_acc(13u, lane);
            q6id_contract_u32x4_vec acc6 = q6id_contract_init_compact_prod_acc(15u, lane);
            q6id_contract_u32x4_vec acc7 = q6id_contract_init_compact_prod_acc(14u, lane);

            HRX_Q6ID_CONTRACT_WAIT_LGKMCNT_DEPS(16, a4, b2);
            HRX_Q6ID_CONTRACT_WMMA_COMPACT_ACC(acc0, a4, b2);
            HRX_Q6ID_CONTRACT_WAIT_LGKMCNT_DEPS(7, a4, b3);
            HRX_Q6ID_CONTRACT_WMMA_COMPACT_ACC(acc1, a4, b3);
            HRX_Q6ID_CONTRACT_WMMA_COMPACT_ACC(acc2, a5, b2);
            HRX_Q6ID_CONTRACT_WAIT_LGKMCNT_DEPS(6, a5, b3);
            HRX_Q6ID_CONTRACT_WMMA_COMPACT_ACC(acc3, a5, b3);
            HRX_Q6ID_CONTRACT_WAIT_LGKMCNT_DEPS(1, a6, b2);
            HRX_Q6ID_CONTRACT_WMMA_COMPACT_ACC(acc4, a6, b2);
            HRX_Q6ID_CONTRACT_WMMA_COMPACT_ACC(acc5, a6, b3);
            HRX_Q6ID_CONTRACT_WAIT_LGKMCNT_DEPS(0, a7, b3);
            HRX_Q6ID_CONTRACT_WMMA_COMPACT_ACC(acc7, a7, b3);
            HRX_Q6ID_CONTRACT_WMMA_COMPACT_ACC(acc6, a7, b2);

            q6id_contract_prod_stage_compact_acc8(
                sh_prod_stage, 8u, lane, acc0, acc1, acc2, acc3, acc4, acc5, acc7, acc6);
        }
    }

    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (wave == 0u) {
        HRX_Q6ID_CONTRACT_PROD_STAGE_FLUSH(0u);
        HRX_Q6ID_CONTRACT_PROD_STAGE_FLUSH(1u);
        HRX_Q6ID_CONTRACT_PROD_STAGE_FLUSH(2u);
        HRX_Q6ID_CONTRACT_PROD_STAGE_FLUSH(3u);
        HRX_Q6ID_CONTRACT_PROD_STAGE_FLUSH(4u);
        HRX_Q6ID_CONTRACT_PROD_STAGE_FLUSH(5u);
        HRX_Q6ID_CONTRACT_PROD_STAGE_FLUSH(6u);
        HRX_Q6ID_CONTRACT_PROD_STAGE_FLUSH(7u);
        HRX_Q6ID_CONTRACT_PROD_STAGE_FLUSH(8u);
        HRX_Q6ID_CONTRACT_PROD_STAGE_FLUSH(9u);
        HRX_Q6ID_CONTRACT_PROD_STAGE_FLUSH(10u);
        HRX_Q6ID_CONTRACT_PROD_STAGE_FLUSH(11u);
        HRX_Q6ID_CONTRACT_PROD_STAGE_FLUSH(12u);
        HRX_Q6ID_CONTRACT_PROD_STAGE_FLUSH(13u);
        HRX_Q6ID_CONTRACT_PROD_STAGE_FLUSH(14u);
        HRX_Q6ID_CONTRACT_PROD_STAGE_FLUSH(15u);
    }
}

extern "C" __global__ __launch_bounds__(256, 1)
void q6_id_subgroup_contract_prodaddr_radv_issue_compact_scoped_probe(float * dst, unsigned int * counts) {
    __shared__ uint64_t sh_frag[12 * 256];
    __shared__ uint16_t sh_prod_stage[16 * 64 * 4];

    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & 63u;
    const unsigned int wave = tid >> 6u;

    if (wave == 0u) {
        q6id_contract_init_banked_fragments(sh_frag, lane);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (wave == 0u) {
        q6id_contract_sink_fragment(q6id_contract_load_fragment(
            (const __attribute__((address_space(3))) uint64_t *) sh_frag, 0u, lane));
        const q6id_contract_half16_vec a0 = q6id_contract_load_fragment(
            (const __attribute__((address_space(3))) uint64_t *) sh_frag, 0u, lane);
        const q6id_contract_half16_vec b0 = q6id_contract_load_fragment(
            (const __attribute__((address_space(3))) uint64_t *) sh_frag, 8u, lane);
        const q6id_contract_half16_vec b1 = q6id_contract_load_fragment(
            (const __attribute__((address_space(3))) uint64_t *) sh_frag, 9u, lane);
        const q6id_contract_half16_vec a1 = q6id_contract_load_fragment(
            (const __attribute__((address_space(3))) uint64_t *) sh_frag, 1u, lane);
        const q6id_contract_half16_vec a2 = q6id_contract_load_fragment(
            (const __attribute__((address_space(3))) uint64_t *) sh_frag, 2u, lane);
        const q6id_contract_half16_vec a3 = q6id_contract_load_fragment(
            (const __attribute__((address_space(3))) uint64_t *) sh_frag, 3u, lane);
        const q6id_contract_half16_vec a4 = q6id_contract_load_fragment(
            (const __attribute__((address_space(3))) uint64_t *) sh_frag, 4u, lane);
        const q6id_contract_half16_vec b2 = q6id_contract_load_fragment(
            (const __attribute__((address_space(3))) uint64_t *) sh_frag, 10u, lane);
        const q6id_contract_half16_vec b3 = q6id_contract_load_fragment(
            (const __attribute__((address_space(3))) uint64_t *) sh_frag, 11u, lane);
        const q6id_contract_half16_vec a5 = q6id_contract_load_fragment(
            (const __attribute__((address_space(3))) uint64_t *) sh_frag, 5u, lane);
        const q6id_contract_half16_vec a6 = q6id_contract_load_fragment(
            (const __attribute__((address_space(3))) uint64_t *) sh_frag, 6u, lane);
        const q6id_contract_half16_vec a7 = q6id_contract_load_fragment(
            (const __attribute__((address_space(3))) uint64_t *) sh_frag, 7u, lane);

        {
            q6id_contract_u32x4_vec acc0 = q6id_contract_init_compact_prod_acc(0u, lane);
            q6id_contract_u32x4_vec acc1 = q6id_contract_init_compact_prod_acc(1u, lane);
            q6id_contract_u32x4_vec acc2 = q6id_contract_init_compact_prod_acc(2u, lane);
            q6id_contract_u32x4_vec acc3 = q6id_contract_init_compact_prod_acc(3u, lane);
            q6id_contract_u32x4_vec acc4 = q6id_contract_init_compact_prod_acc(4u, lane);
            q6id_contract_u32x4_vec acc5 = q6id_contract_init_compact_prod_acc(5u, lane);
            q6id_contract_u32x4_vec acc6 = q6id_contract_init_compact_prod_acc(6u, lane);
            q6id_contract_u32x4_vec acc7 = q6id_contract_init_compact_prod_acc(7u, lane);

            HRX_Q6ID_CONTRACT_WAIT_LGKMCNT_DEPS(40, a0, b0);
            HRX_Q6ID_CONTRACT_WMMA_COMPACT_ACC(acc0, a0, b0);
            HRX_Q6ID_CONTRACT_WAIT_LGKMCNT_DEPS(36, a0, b1);
            HRX_Q6ID_CONTRACT_WMMA_COMPACT_ACC(acc1, a0, b1);
            HRX_Q6ID_CONTRACT_WAIT_LGKMCNT_DEPS(32, a1, b0);
            HRX_Q6ID_CONTRACT_WMMA_COMPACT_ACC(acc2, a1, b0);
            HRX_Q6ID_CONTRACT_WMMA_COMPACT_ACC(acc3, a1, b1);
            HRX_Q6ID_CONTRACT_WAIT_LGKMCNT_DEPS(28, a2, b0);
            HRX_Q6ID_CONTRACT_WMMA_COMPACT_ACC(acc4, a2, b0);
            HRX_Q6ID_CONTRACT_WMMA_COMPACT_ACC(acc5, a2, b1);
            HRX_Q6ID_CONTRACT_WAIT_LGKMCNT_DEPS(24, a3, b0);
            HRX_Q6ID_CONTRACT_WMMA_COMPACT_ACC(acc6, a3, b0);
            HRX_Q6ID_CONTRACT_WMMA_COMPACT_ACC(acc7, a3, b1);

            HRX_Q6ID_CONTRACT_PROD_STAGE_COMPACT_ACC8_INLINE(
                0u, acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7);
        }

        {
            q6id_contract_u32x4_vec acc0 = q6id_contract_init_compact_prod_acc(8u, lane);
            q6id_contract_u32x4_vec acc1 = q6id_contract_init_compact_prod_acc(9u, lane);
            q6id_contract_u32x4_vec acc2 = q6id_contract_init_compact_prod_acc(10u, lane);
            q6id_contract_u32x4_vec acc3 = q6id_contract_init_compact_prod_acc(11u, lane);
            q6id_contract_u32x4_vec acc4 = q6id_contract_init_compact_prod_acc(12u, lane);
            q6id_contract_u32x4_vec acc5 = q6id_contract_init_compact_prod_acc(13u, lane);
            q6id_contract_u32x4_vec acc6 = q6id_contract_init_compact_prod_acc(15u, lane);
            q6id_contract_u32x4_vec acc7 = q6id_contract_init_compact_prod_acc(14u, lane);

            HRX_Q6ID_CONTRACT_WAIT_LGKMCNT_DEPS(16, a4, b2);
            HRX_Q6ID_CONTRACT_WMMA_COMPACT_ACC(acc0, a4, b2);
            HRX_Q6ID_CONTRACT_WAIT_LGKMCNT_DEPS(7, a4, b3);
            HRX_Q6ID_CONTRACT_WMMA_COMPACT_ACC(acc1, a4, b3);
            HRX_Q6ID_CONTRACT_WMMA_COMPACT_ACC(acc2, a5, b2);
            HRX_Q6ID_CONTRACT_WAIT_LGKMCNT_DEPS(6, a5, b3);
            HRX_Q6ID_CONTRACT_WMMA_COMPACT_ACC(acc3, a5, b3);
            HRX_Q6ID_CONTRACT_WAIT_LGKMCNT_DEPS(1, a6, b2);
            HRX_Q6ID_CONTRACT_WMMA_COMPACT_ACC(acc4, a6, b2);
            HRX_Q6ID_CONTRACT_WMMA_COMPACT_ACC(acc5, a6, b3);
            HRX_Q6ID_CONTRACT_WAIT_LGKMCNT_DEPS(0, a7, b3);
            HRX_Q6ID_CONTRACT_WMMA_COMPACT_ACC(acc7, a7, b3);
            HRX_Q6ID_CONTRACT_WMMA_COMPACT_ACC(acc6, a7, b2);

            HRX_Q6ID_CONTRACT_PROD_STAGE_COMPACT_ACC8_INLINE(
                8u, acc0, acc1, acc2, acc3, acc4, acc5, acc7, acc6);
        }
    }

    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (wave == 0u) {
        HRX_Q6ID_CONTRACT_PROD_STAGE_FLUSH(0u);
        HRX_Q6ID_CONTRACT_PROD_STAGE_FLUSH(1u);
        HRX_Q6ID_CONTRACT_PROD_STAGE_FLUSH(2u);
        HRX_Q6ID_CONTRACT_PROD_STAGE_FLUSH(3u);
        HRX_Q6ID_CONTRACT_PROD_STAGE_FLUSH(4u);
        HRX_Q6ID_CONTRACT_PROD_STAGE_FLUSH(5u);
        HRX_Q6ID_CONTRACT_PROD_STAGE_FLUSH(6u);
        HRX_Q6ID_CONTRACT_PROD_STAGE_FLUSH(7u);
        HRX_Q6ID_CONTRACT_PROD_STAGE_FLUSH(8u);
        HRX_Q6ID_CONTRACT_PROD_STAGE_FLUSH(9u);
        HRX_Q6ID_CONTRACT_PROD_STAGE_FLUSH(10u);
        HRX_Q6ID_CONTRACT_PROD_STAGE_FLUSH(11u);
        HRX_Q6ID_CONTRACT_PROD_STAGE_FLUSH(12u);
        HRX_Q6ID_CONTRACT_PROD_STAGE_FLUSH(13u);
        HRX_Q6ID_CONTRACT_PROD_STAGE_FLUSH(14u);
        HRX_Q6ID_CONTRACT_PROD_STAGE_FLUSH(15u);
    }
}

extern "C" __global__ __launch_bounds__(256, 1)
void q6_id_subgroup_contract_prodaddr_banked_compact_probe(float * dst, unsigned int * counts) {
    __shared__ uint64_t sh_frag[12 * 256];
    __shared__ uint16_t sh_prod_stage[16 * 64 * 4];

    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & 63u;
    const unsigned int wave = tid >> 6u;

    if (wave == 0u) {
        q6id_contract_init_banked_fragments(sh_frag, lane);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (wave == 0u) {
        q6id_contract_sink_fragment(q6id_contract_load_fragment(
            (const __attribute__((address_space(3))) uint64_t *) sh_frag, 0u, lane));
        const q6id_contract_half16_vec a0 = q6id_contract_load_fragment(
            (const __attribute__((address_space(3))) uint64_t *) sh_frag, 0u, lane);
        const q6id_contract_half16_vec b0 = q6id_contract_load_fragment(
            (const __attribute__((address_space(3))) uint64_t *) sh_frag, 8u, lane);
        const q6id_contract_half16_vec b1 = q6id_contract_load_fragment(
            (const __attribute__((address_space(3))) uint64_t *) sh_frag, 9u, lane);
        const q6id_contract_half16_vec a1 = q6id_contract_load_fragment(
            (const __attribute__((address_space(3))) uint64_t *) sh_frag, 1u, lane);
        const q6id_contract_half16_vec a2 = q6id_contract_load_fragment(
            (const __attribute__((address_space(3))) uint64_t *) sh_frag, 2u, lane);
        const q6id_contract_half16_vec a3 = q6id_contract_load_fragment(
            (const __attribute__((address_space(3))) uint64_t *) sh_frag, 3u, lane);
        const q6id_contract_half16_vec a4 = q6id_contract_load_fragment(
            (const __attribute__((address_space(3))) uint64_t *) sh_frag, 4u, lane);
        const q6id_contract_half16_vec b2 = q6id_contract_load_fragment(
            (const __attribute__((address_space(3))) uint64_t *) sh_frag, 10u, lane);
        const q6id_contract_half16_vec b3 = q6id_contract_load_fragment(
            (const __attribute__((address_space(3))) uint64_t *) sh_frag, 11u, lane);
        const q6id_contract_half16_vec a5 = q6id_contract_load_fragment(
            (const __attribute__((address_space(3))) uint64_t *) sh_frag, 5u, lane);
        const q6id_contract_half16_vec a6 = q6id_contract_load_fragment(
            (const __attribute__((address_space(3))) uint64_t *) sh_frag, 6u, lane);
        const q6id_contract_half16_vec a7 = q6id_contract_load_fragment(
            (const __attribute__((address_space(3))) uint64_t *) sh_frag, 7u, lane);

        HRX_Q6ID_CONTRACT_WAIT_LGKMCNT_DEPS(40, a0, b0);
        HRX_Q6ID_CONTRACT_WMMA_COMPACT_PROD(0u, a0, b0);
        HRX_Q6ID_CONTRACT_WAIT_LGKMCNT_DEPS(36, a0, b1);
        HRX_Q6ID_CONTRACT_WMMA_COMPACT_PROD(1u, a0, b1);
        HRX_Q6ID_CONTRACT_WAIT_LGKMCNT_DEPS(32, a1, b0);
        HRX_Q6ID_CONTRACT_WMMA_COMPACT_PROD(2u, a1, b0);
        HRX_Q6ID_CONTRACT_WMMA_COMPACT_PROD(3u, a1, b1);
        HRX_Q6ID_CONTRACT_WAIT_LGKMCNT_DEPS(28, a2, b0);
        HRX_Q6ID_CONTRACT_WMMA_COMPACT_PROD(4u, a2, b0);
        HRX_Q6ID_CONTRACT_WMMA_COMPACT_PROD(5u, a2, b1);
        HRX_Q6ID_CONTRACT_WAIT_LGKMCNT_DEPS(24, a3, b1);
        HRX_Q6ID_CONTRACT_WMMA_COMPACT_PROD(6u, a3, b1);
        HRX_Q6ID_CONTRACT_WMMA_COMPACT_PROD(7u, a3, b0);
        HRX_Q6ID_CONTRACT_WAIT_LGKMCNT_DEPS(16, a4, b2);
        HRX_Q6ID_CONTRACT_WMMA_COMPACT_PROD(8u, a4, b2);
        HRX_Q6ID_CONTRACT_WAIT_LGKMCNT_DEPS(12, a4, b3);
        HRX_Q6ID_CONTRACT_WMMA_COMPACT_PROD(9u, a4, b3);
        HRX_Q6ID_CONTRACT_WAIT_LGKMCNT_DEPS(8, a5, b2);
        HRX_Q6ID_CONTRACT_WMMA_COMPACT_PROD(10u, a5, b2);
        HRX_Q6ID_CONTRACT_WMMA_COMPACT_PROD(11u, a5, b3);
        HRX_Q6ID_CONTRACT_WAIT_LGKMCNT_DEPS(4, a6, b2);
        HRX_Q6ID_CONTRACT_WMMA_COMPACT_PROD(12u, a6, b2);
        HRX_Q6ID_CONTRACT_WMMA_COMPACT_PROD(13u, a6, b3);
        HRX_Q6ID_CONTRACT_WAIT_LGKMCNT_DEPS(0, a7, b3);
        HRX_Q6ID_CONTRACT_WMMA_COMPACT_PROD(14u, a7, b3);
        HRX_Q6ID_CONTRACT_WMMA_COMPACT_PROD(15u, a7, b2);
    }

    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (wave == 0u) {
        HRX_Q6ID_CONTRACT_PROD_STAGE_FLUSH(0u);
        HRX_Q6ID_CONTRACT_PROD_STAGE_FLUSH(1u);
        HRX_Q6ID_CONTRACT_PROD_STAGE_FLUSH(2u);
        HRX_Q6ID_CONTRACT_PROD_STAGE_FLUSH(3u);
        HRX_Q6ID_CONTRACT_PROD_STAGE_FLUSH(4u);
        HRX_Q6ID_CONTRACT_PROD_STAGE_FLUSH(5u);
        HRX_Q6ID_CONTRACT_PROD_STAGE_FLUSH(6u);
        HRX_Q6ID_CONTRACT_PROD_STAGE_FLUSH(7u);
        HRX_Q6ID_CONTRACT_PROD_STAGE_FLUSH(8u);
        HRX_Q6ID_CONTRACT_PROD_STAGE_FLUSH(9u);
        HRX_Q6ID_CONTRACT_PROD_STAGE_FLUSH(10u);
        HRX_Q6ID_CONTRACT_PROD_STAGE_FLUSH(11u);
        HRX_Q6ID_CONTRACT_PROD_STAGE_FLUSH(12u);
        HRX_Q6ID_CONTRACT_PROD_STAGE_FLUSH(13u);
        HRX_Q6ID_CONTRACT_PROD_STAGE_FLUSH(14u);
        HRX_Q6ID_CONTRACT_PROD_STAGE_FLUSH(15u);
    }
}

extern "C" __global__ __launch_bounds__(256, 1)
void q6_id_subgroup_contract_staged_probe(float * dst, unsigned int * counts) {
    __shared__ uint64_t sh_frag[4 * 256];
    __shared__ uint16_t sh_stage[HRX_Q6ID_CONTRACT_VALUES];

    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & 63u;
    const unsigned int wave = tid >> 6u;

    if (wave == 0u) {
        q6id_contract_init_fragments(sh_frag, lane);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (wave == 0u) {
    const q6id_contract_half16_vec a0 = q6id_contract_load_fragment(
        (const __attribute__((address_space(3))) uint64_t *) sh_frag, 0u, lane);
    const q6id_contract_half16_vec b0 = q6id_contract_load_fragment(
        (const __attribute__((address_space(3))) uint64_t *) sh_frag, 1u, lane);
    const q6id_contract_half16_vec a1 = q6id_contract_load_fragment(
        (const __attribute__((address_space(3))) uint64_t *) sh_frag, 2u, lane);
    const q6id_contract_half16_vec b1 = q6id_contract_load_fragment(
        (const __attribute__((address_space(3))) uint64_t *) sh_frag, 3u, lane);
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");

    HRX_Q6ID_CONTRACT_WMMA(0u, a0, b0);
    HRX_Q6ID_CONTRACT_WMMA(1u, a0, b1);
    HRX_Q6ID_CONTRACT_WMMA(2u, a1, b0);
    HRX_Q6ID_CONTRACT_WMMA(3u, a1, b1);
    HRX_Q6ID_CONTRACT_WMMA(4u, a0, b0);
    HRX_Q6ID_CONTRACT_WMMA(5u, a0, b1);
    HRX_Q6ID_CONTRACT_WMMA(6u, a1, b0);
    HRX_Q6ID_CONTRACT_WMMA(7u, a1, b1);
    HRX_Q6ID_CONTRACT_WMMA(8u, a0, b0);
    HRX_Q6ID_CONTRACT_WMMA(9u, a0, b1);
    HRX_Q6ID_CONTRACT_WMMA(10u, a1, b0);
    HRX_Q6ID_CONTRACT_WMMA(11u, a1, b1);
    HRX_Q6ID_CONTRACT_WMMA(12u, a0, b0);
    HRX_Q6ID_CONTRACT_WMMA(13u, a0, b1);
    HRX_Q6ID_CONTRACT_WMMA(14u, a1, b0);
    HRX_Q6ID_CONTRACT_WMMA(15u, a1, b1);
    }

    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (wave == 0u) {
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(0u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(1u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(2u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(3u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(4u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(5u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(6u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(7u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(8u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(9u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(10u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(11u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(12u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(13u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(14u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(15u);
    }
}

extern "C" __global__ __launch_bounds__(256, 1)
void q6_id_subgroup_contract_loaddeep_probe(float * dst, unsigned int * counts) {
    __shared__ uint64_t sh_frag[4 * 256];
    __shared__ uint16_t sh_stage[HRX_Q6ID_CONTRACT_VALUES];

    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & 63u;
    const unsigned int wave = tid >> 6u;

    if (wave == 0u) {
        q6id_contract_init_fragments(sh_frag, lane);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (wave == 0u) {
    const q6id_contract_half16_vec f0 = q6id_contract_load_fragment(
        (const __attribute__((address_space(3))) uint64_t *) sh_frag, 0u, lane);
    const q6id_contract_half16_vec f1 = q6id_contract_load_fragment(
        (const __attribute__((address_space(3))) uint64_t *) sh_frag, 1u, lane);
    const q6id_contract_half16_vec f2 = q6id_contract_load_fragment(
        (const __attribute__((address_space(3))) uint64_t *) sh_frag, 2u, lane);
    const q6id_contract_half16_vec f3 = q6id_contract_load_fragment(
        (const __attribute__((address_space(3))) uint64_t *) sh_frag, 3u, lane);
    q6id_contract_sink_fragment(q6id_contract_load_fragment(
        (const __attribute__((address_space(3))) uint64_t *) sh_frag, 0u, lane));
    q6id_contract_sink_fragment(q6id_contract_load_fragment(
        (const __attribute__((address_space(3))) uint64_t *) sh_frag, 1u, lane));
    q6id_contract_sink_fragment(q6id_contract_load_fragment(
        (const __attribute__((address_space(3))) uint64_t *) sh_frag, 2u, lane));
    q6id_contract_sink_fragment(q6id_contract_load_fragment(
        (const __attribute__((address_space(3))) uint64_t *) sh_frag, 3u, lane));
    q6id_contract_sink_fragment(q6id_contract_load_fragment(
        (const __attribute__((address_space(3))) uint64_t *) sh_frag, 0u, lane));
    q6id_contract_sink_fragment(q6id_contract_load_fragment(
        (const __attribute__((address_space(3))) uint64_t *) sh_frag, 1u, lane));
    q6id_contract_sink_fragment(q6id_contract_load_fragment(
        (const __attribute__((address_space(3))) uint64_t *) sh_frag, 2u, lane));
    q6id_contract_sink_fragment(q6id_contract_load_fragment(
        (const __attribute__((address_space(3))) uint64_t *) sh_frag, 3u, lane));
    q6id_contract_sink_fragment(q6id_contract_load_fragment(
        (const __attribute__((address_space(3))) uint64_t *) sh_frag, 0u, lane));
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");

    HRX_Q6ID_CONTRACT_WMMA(0u, f0, f1);
    HRX_Q6ID_CONTRACT_WMMA(1u, f2, f3);
    HRX_Q6ID_CONTRACT_WMMA(2u, f0, f1);
    HRX_Q6ID_CONTRACT_WMMA(3u, f2, f3);
    HRX_Q6ID_CONTRACT_WMMA(4u, f0, f1);
    HRX_Q6ID_CONTRACT_WMMA(5u, f2, f3);
    HRX_Q6ID_CONTRACT_WMMA(6u, f0, f1);
    HRX_Q6ID_CONTRACT_WMMA(7u, f1, f2);
    HRX_Q6ID_CONTRACT_WMMA(8u, f3, f0);
    HRX_Q6ID_CONTRACT_WMMA(9u, f1, f2);
    HRX_Q6ID_CONTRACT_WMMA(10u, f3, f0);
    HRX_Q6ID_CONTRACT_WMMA(11u, f1, f2);
    HRX_Q6ID_CONTRACT_WMMA(12u, f3, f0);
    HRX_Q6ID_CONTRACT_WMMA(13u, f0, f2);
    HRX_Q6ID_CONTRACT_WMMA(14u, f0, f2);
    HRX_Q6ID_CONTRACT_WMMA(15u, f0, f2);
    }

    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (wave == 0u) {
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(0u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(1u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(2u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(3u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(4u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(5u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(6u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(7u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(8u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(9u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(10u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(11u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(12u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(13u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(14u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(15u);
    }
}

extern "C" __global__ __launch_bounds__(256, 1)
void q6_id_subgroup_contract_minstore_probe(float * dst, unsigned int * counts) {
    __shared__ uint64_t sh_frag[1 * 256];
    __shared__ uint16_t sh_stage[HRX_Q6ID_CONTRACT_VALUES];
    __shared__ uint16_t sh_pad[3072];

    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & 63u;
    const unsigned int wave = tid >> 6u;

    if (wave == 0u) {
        q6id_contract_init_fragment0(sh_frag, lane);
        asm volatile("" :: "v"(sh_pad + (lane & 1u)) : "memory");
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (wave == 0u) {
    const q6id_contract_half16_vec f0 = q6id_contract_load_fragment(
        (const __attribute__((address_space(3))) uint64_t *) sh_frag, 0u, lane);
    q6id_contract_sink_fragment(q6id_contract_load_fragment(
        (const __attribute__((address_space(3))) uint64_t *) sh_frag, 0u, lane));
    q6id_contract_sink_fragment(q6id_contract_load_fragment(
        (const __attribute__((address_space(3))) uint64_t *) sh_frag, 0u, lane));
    q6id_contract_sink_fragment(q6id_contract_load_fragment(
        (const __attribute__((address_space(3))) uint64_t *) sh_frag, 0u, lane));
    q6id_contract_sink_fragment(q6id_contract_load_fragment(
        (const __attribute__((address_space(3))) uint64_t *) sh_frag, 0u, lane));
    q6id_contract_sink_fragment(q6id_contract_load_fragment(
        (const __attribute__((address_space(3))) uint64_t *) sh_frag, 0u, lane));
    q6id_contract_sink_fragment(q6id_contract_load_fragment(
        (const __attribute__((address_space(3))) uint64_t *) sh_frag, 0u, lane));
    q6id_contract_sink_fragment(q6id_contract_load_fragment(
        (const __attribute__((address_space(3))) uint64_t *) sh_frag, 0u, lane));
    q6id_contract_sink_fragment(q6id_contract_load_fragment(
        (const __attribute__((address_space(3))) uint64_t *) sh_frag, 0u, lane));
    q6id_contract_sink_fragment(q6id_contract_load_fragment(
        (const __attribute__((address_space(3))) uint64_t *) sh_frag, 0u, lane));
    q6id_contract_sink_fragment(q6id_contract_load_fragment(
        (const __attribute__((address_space(3))) uint64_t *) sh_frag, 0u, lane));
    q6id_contract_sink_fragment(q6id_contract_load_fragment(
        (const __attribute__((address_space(3))) uint64_t *) sh_frag, 0u, lane));
    q6id_contract_sink_fragment(q6id_contract_load_fragment(
        (const __attribute__((address_space(3))) uint64_t *) sh_frag, 0u, lane));
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");

    HRX_Q6ID_CONTRACT_WMMA(0u, f0, f0);
    HRX_Q6ID_CONTRACT_WMMA(1u, f0, f0);
    HRX_Q6ID_CONTRACT_WMMA(2u, f0, f0);
    HRX_Q6ID_CONTRACT_WMMA(3u, f0, f0);
    HRX_Q6ID_CONTRACT_WMMA(4u, f0, f0);
    HRX_Q6ID_CONTRACT_WMMA(5u, f0, f0);
    HRX_Q6ID_CONTRACT_WMMA(6u, f0, f0);
    HRX_Q6ID_CONTRACT_WMMA(7u, f0, f0);
    HRX_Q6ID_CONTRACT_WMMA(8u, f0, f0);
    HRX_Q6ID_CONTRACT_WMMA(9u, f0, f0);
    HRX_Q6ID_CONTRACT_WMMA(10u, f0, f0);
    HRX_Q6ID_CONTRACT_WMMA(11u, f0, f0);
    HRX_Q6ID_CONTRACT_WMMA(12u, f0, f0);
    HRX_Q6ID_CONTRACT_WMMA(13u, f0, f0);
    HRX_Q6ID_CONTRACT_WMMA(14u, f0, f0);
    HRX_Q6ID_CONTRACT_WMMA(15u, f0, f0);
    }

    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (wave == 0u) {
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(0u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(1u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(2u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(3u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(4u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(5u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(6u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(7u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(8u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(9u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(10u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(11u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(12u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(13u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(14u);
    HRX_Q6ID_CONTRACT_STAGE_FLUSH(15u);
    }
}

static double elapsed_us(const std::chrono::steady_clock::time_point & a,
        const std::chrono::steady_clock::time_point & b) {
    return std::chrono::duration<double, std::micro>(b - a).count();
}

static void clear_buffers(float * d_dst, unsigned int * d_counts) {
    HIP_CHECK(hipMemset(d_dst, 0, HRX_Q6ID_CONTRACT_VALUES * sizeof(float)));
    HIP_CHECK(hipMemset(d_counts, 0, HRX_Q6ID_CONTRACT_VALUES * sizeof(unsigned int)));
}

static void clear_prod_buffers(float * d_dst, unsigned int * d_counts) {
    HIP_CHECK(hipMemset(d_dst, 0, HRX_Q6ID_CONTRACT_PROD_VALUES * sizeof(float)));
    HIP_CHECK(hipMemset(d_counts, 0, HRX_Q6ID_CONTRACT_PROD_VALUES * sizeof(unsigned int)));
}

static double run_kernel(const std::string & name, int reps, float * d_dst, unsigned int * d_counts) {
    clear_buffers(d_dst, d_counts);
    HIP_CHECK(hipDeviceSynchronize());
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; ++i) {
        if (name == "direct") {
            q6_id_subgroup_contract_direct_probe<<<1, 256>>>(d_dst, d_counts);
        } else if (name == "banked") {
            q6_id_subgroup_contract_banked_probe<<<1, 256>>>(d_dst, d_counts);
        } else if (name == "bankedcompact") {
            q6_id_subgroup_contract_banked_compact_probe<<<1, 256>>>(d_dst, d_counts);
        } else if (name == "radvissuecompact") {
            q6_id_subgroup_contract_radv_issue_compact_probe<<<1, 256>>>(d_dst, d_counts);
        } else if (name == "loaddeep") {
            q6_id_subgroup_contract_loaddeep_probe<<<1, 256>>>(d_dst, d_counts);
        } else if (name == "minstore") {
            q6_id_subgroup_contract_minstore_probe<<<1, 256>>>(d_dst, d_counts);
        } else {
            q6_id_subgroup_contract_staged_probe<<<1, 256>>>(d_dst, d_counts);
        }
    }
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    const auto end = std::chrono::steady_clock::now();
    return elapsed_us(start, end) / static_cast<double>(reps);
}

static double run_prod_kernel(const std::string & name, int reps, float * d_dst, unsigned int * d_counts) {
    clear_prod_buffers(d_dst, d_counts);
    HIP_CHECK(hipDeviceSynchronize());
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; ++i) {
        if (name == "prodaddr-bankedcompact") {
            q6_id_subgroup_contract_prodaddr_banked_compact_probe<<<1, 256>>>(d_dst, d_counts);
        } else if (name == "prodaddr-radvissuecompact") {
            q6_id_subgroup_contract_prodaddr_radv_issue_compact_probe<<<1, 256>>>(d_dst, d_counts);
        } else if (name == "prodaddr-radvissuecompact-helper") {
            q6_id_subgroup_contract_prodaddr_radv_issue_compact_helper_probe<<<1, 256>>>(d_dst, d_counts);
        } else if (name == "prodaddr-radvissuecompact-scoped") {
            q6_id_subgroup_contract_prodaddr_radv_issue_compact_scoped_probe<<<1, 256>>>(d_dst, d_counts);
        } else if (name == "prodaddr-radv96-duplicate") {
            q6_id_subgroup_contract_prodaddr_radv96_duplicate_probe<<<1, 256>>>(d_dst, d_counts);
        } else {
            q6_id_subgroup_contract_prodaddr_direct_probe<<<1, 256>>>(d_dst, d_counts);
        }
    }
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    const auto end = std::chrono::steady_clock::now();
    return elapsed_us(start, end) / static_cast<double>(reps);
}

static bool validate(float * d_dst, unsigned int * d_counts) {
    std::vector<float> values(HRX_Q6ID_CONTRACT_VALUES);
    std::vector<unsigned int> counts(HRX_Q6ID_CONTRACT_VALUES);
    HIP_CHECK(hipMemcpy(values.data(), d_dst, values.size() * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(counts.data(), d_counts, counts.size() * sizeof(unsigned int), hipMemcpyDeviceToHost));

    bool ok = true;
    for (unsigned int group = 0; group < HRX_Q6ID_CONTRACT_GROUPS; ++group) {
        for (unsigned int slot = 0; slot < HRX_Q6ID_CONTRACT_SLOTS; ++slot) {
            for (unsigned int lane = 0; lane < HRX_Q6ID_CONTRACT_LANES; ++lane) {
                const unsigned int index = q6id_contract_index(group, slot, lane);
                if (counts[index] != 1u || !std::isfinite(values[index])) {
                    if (ok) {
                        std::fprintf(stderr,
                            "validation failure at group=%u slot=%u lane=%u count=%u value=%f\n",
                            group, slot, lane, counts[index], values[index]);
                    }
                    ok = false;
                }
            }
        }
    }
    return ok;
}

struct prod_validate_result {
    bool ok = false;
    unsigned int missed = 0;
    unsigned int duplicated = 0;
    unsigned int max_count = 0;
};

static prod_validate_result validate_prod(float * d_dst, unsigned int * d_counts, bool allow_tail_duplicates) {
    std::vector<float> values(HRX_Q6ID_CONTRACT_PROD_VALUES);
    std::vector<unsigned int> counts(HRX_Q6ID_CONTRACT_PROD_VALUES);
    HIP_CHECK(hipMemcpy(values.data(), d_dst, values.size() * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(counts.data(), d_counts, counts.size() * sizeof(unsigned int), hipMemcpyDeviceToHost));

    prod_validate_result result;
    result.ok = true;
    for (unsigned int col = 0; col < HRX_Q6ID_CONTRACT_PROD_COLS; ++col) {
        for (unsigned int row = 0; row < HRX_Q6ID_CONTRACT_PROD_ROWS; ++row) {
            const unsigned int index = col * HRX_Q6ID_CONTRACT_PROD_ROWS + row;
            const unsigned int count = counts[index];
            result.max_count = std::max(result.max_count, count);
            const bool duplicate_allowed = allow_tail_duplicates && col == 32u;
            const bool count_ok = duplicate_allowed ? (count >= 1u && count <= 2u) : (count == 1u);
            if (count == 0u) {
                ++result.missed;
            } else if (count > 1u) {
                ++result.duplicated;
            }
            if (!count_ok || !std::isfinite(values[index])) {
                if (result.ok) {
                    std::fprintf(stderr,
                        "prod validation failure at row=%u col=%u count=%u value=%f allow_tail_duplicates=%d\n",
                        row, col, count, values[index], allow_tail_duplicates ? 1 : 0);
                }
                result.ok = false;
            }
        }
    }
    return result;
}

int main(int argc, char ** argv) {
    int reps = 10000;
    if (argc > 1) {
        reps = std::max(1, std::atoi(argv[1]));
    }

    int device = 0;
    HIP_CHECK(hipGetDevice(&device));
    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, device));
    std::printf("device=%s reps=%d values=%u\n", prop.name, reps, HRX_Q6ID_CONTRACT_VALUES);

    float * d_dst = nullptr;
    unsigned int * d_counts = nullptr;
    HIP_CHECK(hipMalloc(&d_dst, HRX_Q6ID_CONTRACT_VALUES * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_counts, HRX_Q6ID_CONTRACT_VALUES * sizeof(unsigned int)));
    float * d_prod_dst = nullptr;
    unsigned int * d_prod_counts = nullptr;
    HIP_CHECK(hipMalloc(&d_prod_dst, HRX_Q6ID_CONTRACT_PROD_VALUES * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_prod_counts, HRX_Q6ID_CONTRACT_PROD_VALUES * sizeof(unsigned int)));

    clear_buffers(d_dst, d_counts);
    q6_id_subgroup_contract_direct_probe<<<1, 256>>>(d_dst, d_counts);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    const bool direct_ok = validate(d_dst, d_counts);

    clear_buffers(d_dst, d_counts);
    q6_id_subgroup_contract_staged_probe<<<1, 256>>>(d_dst, d_counts);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    const bool staged_ok = validate(d_dst, d_counts);

    clear_buffers(d_dst, d_counts);
    q6_id_subgroup_contract_loaddeep_probe<<<1, 256>>>(d_dst, d_counts);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    const bool loaddeep_ok = validate(d_dst, d_counts);

    clear_buffers(d_dst, d_counts);
    q6_id_subgroup_contract_minstore_probe<<<1, 256>>>(d_dst, d_counts);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    const bool minstore_ok = validate(d_dst, d_counts);

    clear_buffers(d_dst, d_counts);
    q6_id_subgroup_contract_banked_probe<<<1, 256>>>(d_dst, d_counts);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    const bool banked_ok = validate(d_dst, d_counts);

    clear_buffers(d_dst, d_counts);
    q6_id_subgroup_contract_banked_compact_probe<<<1, 256>>>(d_dst, d_counts);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    const bool bankedcompact_ok = validate(d_dst, d_counts);

    clear_buffers(d_dst, d_counts);
    q6_id_subgroup_contract_radv_issue_compact_probe<<<1, 256>>>(d_dst, d_counts);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    const bool radvissuecompact_ok = validate(d_dst, d_counts);

    clear_prod_buffers(d_prod_dst, d_prod_counts);
    q6_id_subgroup_contract_prodaddr_direct_probe<<<1, 256>>>(d_prod_dst, d_prod_counts);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    const prod_validate_result prod_direct = validate_prod(d_prod_dst, d_prod_counts, false);

    clear_prod_buffers(d_prod_dst, d_prod_counts);
    q6_id_subgroup_contract_prodaddr_radv96_duplicate_probe<<<1, 256>>>(d_prod_dst, d_prod_counts);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    const prod_validate_result prod_radv96_duplicate = validate_prod(d_prod_dst, d_prod_counts, true);

    clear_prod_buffers(d_prod_dst, d_prod_counts);
    q6_id_subgroup_contract_prodaddr_banked_compact_probe<<<1, 256>>>(d_prod_dst, d_prod_counts);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    const prod_validate_result prod_bankedcompact = validate_prod(d_prod_dst, d_prod_counts, false);

    clear_prod_buffers(d_prod_dst, d_prod_counts);
    q6_id_subgroup_contract_prodaddr_radv_issue_compact_probe<<<1, 256>>>(d_prod_dst, d_prod_counts);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    const prod_validate_result prod_radvissuecompact = validate_prod(d_prod_dst, d_prod_counts, false);

    clear_prod_buffers(d_prod_dst, d_prod_counts);
    q6_id_subgroup_contract_prodaddr_radv_issue_compact_helper_probe<<<1, 256>>>(d_prod_dst, d_prod_counts);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    const prod_validate_result prod_radvissuecompact_helper = validate_prod(d_prod_dst, d_prod_counts, false);

    clear_prod_buffers(d_prod_dst, d_prod_counts);
    q6_id_subgroup_contract_prodaddr_radv_issue_compact_scoped_probe<<<1, 256>>>(d_prod_dst, d_prod_counts);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    const prod_validate_result prod_radvissuecompact_scoped = validate_prod(d_prod_dst, d_prod_counts, false);

    const double direct_us = run_kernel("direct", reps, d_dst, d_counts);
    const double staged_us = run_kernel("staged", reps, d_dst, d_counts);
    const double loaddeep_us = run_kernel("loaddeep", reps, d_dst, d_counts);
    const double minstore_us = run_kernel("minstore", reps, d_dst, d_counts);
    const double banked_us = run_kernel("banked", reps, d_dst, d_counts);
    const double bankedcompact_us = run_kernel("bankedcompact", reps, d_dst, d_counts);
    const double radvissuecompact_us = run_kernel("radvissuecompact", reps, d_dst, d_counts);
    const double prod_direct_us = run_prod_kernel("prodaddr-direct", reps, d_prod_dst, d_prod_counts);
    const double prod_radv96_duplicate_us = run_prod_kernel("prodaddr-radv96-duplicate", reps, d_prod_dst, d_prod_counts);
    const double prod_bankedcompact_us = run_prod_kernel("prodaddr-bankedcompact", reps, d_prod_dst, d_prod_counts);
    const double prod_radvissuecompact_us = run_prod_kernel("prodaddr-radvissuecompact", reps, d_prod_dst, d_prod_counts);
    const double prod_radvissuecompact_helper_us = run_prod_kernel("prodaddr-radvissuecompact-helper", reps, d_prod_dst, d_prod_counts);
    const double prod_radvissuecompact_scoped_us = run_prod_kernel("prodaddr-radvissuecompact-scoped", reps, d_prod_dst, d_prod_counts);
    std::printf("kernel,valid,us\n");
    std::printf("direct,%d,%.6f\n", direct_ok ? 1 : 0, direct_us);
    std::printf("staged,%d,%.6f\n", staged_ok ? 1 : 0, staged_us);
    std::printf("loaddeep,%d,%.6f\n", loaddeep_ok ? 1 : 0, loaddeep_us);
    std::printf("minstore,%d,%.6f\n", minstore_ok ? 1 : 0, minstore_us);
    std::printf("banked,%d,%.6f\n", banked_ok ? 1 : 0, banked_us);
    std::printf("bankedcompact,%d,%.6f\n", bankedcompact_ok ? 1 : 0, bankedcompact_us);
    std::printf("radvissuecompact,%d,%.6f\n", radvissuecompact_ok ? 1 : 0, radvissuecompact_us);
    std::printf("prodaddr_direct,%d,%.6f\n", prod_direct.ok ? 1 : 0, prod_direct_us);
    std::printf("prodaddr_radv96_duplicate,%d,%.6f\n", prod_radv96_duplicate.ok ? 1 : 0, prod_radv96_duplicate_us);
    std::printf("prodaddr_bankedcompact,%d,%.6f\n", prod_bankedcompact.ok ? 1 : 0, prod_bankedcompact_us);
    std::printf("prodaddr_radvissuecompact,%d,%.6f\n", prod_radvissuecompact.ok ? 1 : 0, prod_radvissuecompact_us);
    std::printf("prodaddr_radvissuecompact_helper,%d,%.6f\n",
        prod_radvissuecompact_helper.ok ? 1 : 0,
        prod_radvissuecompact_helper_us);
    std::printf("prodaddr_radvissuecompact_scoped,%d,%.6f\n",
        prod_radvissuecompact_scoped.ok ? 1 : 0,
        prod_radvissuecompact_scoped_us);
    std::printf("prodaddr_direct,%d,missed=%u,duplicated=%u,max_count=%u\n",
        prod_direct.ok ? 1 : 0,
        prod_direct.missed,
        prod_direct.duplicated,
        prod_direct.max_count);
    std::printf("prodaddr_radv96_duplicate,%d,missed=%u,duplicated=%u,max_count=%u\n",
        prod_radv96_duplicate.ok ? 1 : 0,
        prod_radv96_duplicate.missed,
        prod_radv96_duplicate.duplicated,
        prod_radv96_duplicate.max_count);
    std::printf("prodaddr_bankedcompact,%d,missed=%u,duplicated=%u,max_count=%u\n",
        prod_bankedcompact.ok ? 1 : 0,
        prod_bankedcompact.missed,
        prod_bankedcompact.duplicated,
        prod_bankedcompact.max_count);
    std::printf("prodaddr_radvissuecompact,%d,missed=%u,duplicated=%u,max_count=%u\n",
        prod_radvissuecompact.ok ? 1 : 0,
        prod_radvissuecompact.missed,
        prod_radvissuecompact.duplicated,
        prod_radvissuecompact.max_count);
    std::printf("prodaddr_radvissuecompact_helper,%d,missed=%u,duplicated=%u,max_count=%u\n",
        prod_radvissuecompact_helper.ok ? 1 : 0,
        prod_radvissuecompact_helper.missed,
        prod_radvissuecompact_helper.duplicated,
        prod_radvissuecompact_helper.max_count);
    std::printf("prodaddr_radvissuecompact_scoped,%d,missed=%u,duplicated=%u,max_count=%u\n",
        prod_radvissuecompact_scoped.ok ? 1 : 0,
        prod_radvissuecompact_scoped.missed,
        prod_radvissuecompact_scoped.duplicated,
        prod_radvissuecompact_scoped.max_count);

    HIP_CHECK(hipFree(d_dst));
    HIP_CHECK(hipFree(d_counts));
    HIP_CHECK(hipFree(d_prod_dst));
    HIP_CHECK(hipFree(d_prod_counts));
    return (direct_ok && staged_ok && loaddeep_ok && minstore_ok && banked_ok && bankedcompact_ok &&
            radvissuecompact_ok &&
            prod_direct.ok && prod_radv96_duplicate.ok && prod_bankedcompact.ok &&
            prod_radvissuecompact.ok && prod_radvissuecompact_helper.ok &&
            prod_radvissuecompact_scoped.ok) ? 0 : 1;
}
