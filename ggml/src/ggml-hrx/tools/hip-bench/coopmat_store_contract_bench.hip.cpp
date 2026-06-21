#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cmath>
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

static constexpr unsigned int HRX_COOPSTORE_LANES = 64;
static constexpr unsigned int HRX_COOPSTORE_VALUES_PER_GROUP = 4;
static constexpr unsigned int HRX_COOPSTORE_MAX_GROUPS = 48;
static constexpr unsigned int HRX_COOPSTORE_MAX_VALUES =
    HRX_COOPSTORE_LANES * HRX_COOPSTORE_VALUES_PER_GROUP * HRX_COOPSTORE_MAX_GROUPS;

typedef _Float16 coopstore_probe_half16_vec __attribute__((ext_vector_type(16)));
typedef _Float16 coopstore_probe_half8_vec __attribute__((ext_vector_type(8)));
typedef uint32_t coopstore_probe_u32x8_vec __attribute__((ext_vector_type(8)));
typedef uint64_t coopstore_probe_u64x4_vec __attribute__((ext_vector_type(4)));

static __host__ __device__ __forceinline__ float coopstore_probe_value(
        unsigned int group,
        unsigned int slot,
        unsigned int lane) {
    const int bits = static_cast<int>((group * 1009u + slot * 131u + lane * 17u) & 0x7fffu);
    return static_cast<float>(bits - 16384) * 0.001953125f;
}

static __host__ __device__ __forceinline__ unsigned int coopstore_probe_index(
        unsigned int group,
        unsigned int slot,
        unsigned int lane) {
    return (group * HRX_COOPSTORE_VALUES_PER_GROUP + slot) * HRX_COOPSTORE_LANES + lane;
}

static __host__ __device__ __forceinline__ uint16_t coopstore_probe_stage_value(
        unsigned int group,
        unsigned int slot,
        unsigned int lane) {
    return static_cast<uint16_t>(0x2000u + ((group & 0x3fu) << 8) + ((lane & 0x3fu) << 2) + (slot & 0x03u));
}

static __host__ __device__ __forceinline__ float coopstore_probe_expected_value(
        unsigned int group,
        unsigned int slot,
        unsigned int lane) {
    if (group < 16u) {
        return coopstore_probe_value(group, slot, lane);
    }
    return static_cast<float>(coopstore_probe_stage_value(group, slot, lane));
}

static __device__ __forceinline__ unsigned int coopstore_probe_stage_index_base(
        unsigned int group,
        unsigned int slot,
        unsigned int lane,
        unsigned int base_group) {
    const unsigned int stage_group = group - base_group;
    const unsigned int row_lane = lane >> 4;
    const unsigned int col_lane = lane & 15u;
    return stage_group * 16u * 16u + col_lane * 16u + row_lane + slot * 4u;
}

static __device__ __forceinline__ unsigned int coopstore_probe_stage_index(
        unsigned int group,
        unsigned int slot,
        unsigned int lane) {
    return coopstore_probe_stage_index_base(group, slot, lane, 16u);
}

static __device__ __forceinline__ unsigned int coopstore_probe_stage_index_linear_base(
        unsigned int group,
        unsigned int slot,
        unsigned int lane,
        unsigned int base_group) {
    const unsigned int stage_group = group - base_group;
    return (stage_group * HRX_COOPSTORE_LANES + lane) * HRX_COOPSTORE_VALUES_PER_GROUP + slot;
}

static __device__ __forceinline__ __amdgpu_buffer_rsrc_t coopstore_probe_make_rsrc(
        float * dst,
        unsigned long long extent,
        unsigned int flags) {
    return __builtin_amdgcn_make_buffer_rsrc(
        dst,
        static_cast<unsigned short>(0),
        extent,
        static_cast<int>(flags));
}

static __device__ __forceinline__ void coopstore_probe_raw_store(
        __amdgpu_buffer_rsrc_t rsrc,
        unsigned int group,
        unsigned int slot,
        unsigned int lane) {
    const float value = coopstore_probe_value(group, slot, lane);
    const unsigned int index = coopstore_probe_index(group, slot, lane);
    const int byte_offset = static_cast<int>(index * sizeof(float));
    __builtin_amdgcn_raw_buffer_store_b32(__builtin_bit_cast(int, value), rsrc, byte_offset, 0, 0);
}

static __device__ __forceinline__ void coopstore_probe_raw_store_value(
        __amdgpu_buffer_rsrc_t rsrc,
        unsigned int group,
        unsigned int slot,
        unsigned int lane,
        float value) {
    const unsigned int index = coopstore_probe_index(group, slot, lane);
    const int byte_offset = static_cast<int>(index * sizeof(float));
    __builtin_amdgcn_raw_buffer_store_b32(__builtin_bit_cast(int, value), rsrc, byte_offset, 0, 0);
}

static __device__ __forceinline__ void coopstore_probe_raw_store_acc(
        __amdgpu_buffer_rsrc_t rsrc,
        unsigned int group,
        unsigned int slot,
        unsigned int lane,
        coopstore_probe_half8_vec acc) {
    coopstore_probe_raw_store_value(rsrc, group, slot, lane, static_cast<float>(acc[slot * 2u]));
}

static __device__ __forceinline__ void coopstore_probe_raw_store_acc_element(
        __amdgpu_buffer_rsrc_t rsrc,
        unsigned int group,
        unsigned int slot,
        unsigned int lane,
        coopstore_probe_half8_vec acc,
        unsigned int element) {
    coopstore_probe_raw_store_value(rsrc, group, slot, lane, static_cast<float>(acc[element]));
}

static __host__ __device__ __forceinline__ bool coopstore_probe_q6addr_index(
        unsigned int group,
        unsigned int slot,
        unsigned int lane,
        unsigned int rows,
        unsigned int cols,
        unsigned int * index) {
    const unsigned int row_tile = group & 3u;
    const unsigned int col_tile = group >> 2u;
    const unsigned int row_lane = lane >> 4u;
    const unsigned int col_lane = lane & 15u;
    const unsigned int row = row_tile * 16u + row_lane + slot * 4u;
    const unsigned int col = col_tile * 16u + col_lane;
    if (row >= rows || col >= cols) {
        return false;
    }
    *index = col * rows + row;
    return true;
}

static __device__ __forceinline__ void coopstore_probe_raw_store_q6addr_value(
        __amdgpu_buffer_rsrc_t rsrc,
        unsigned int group,
        unsigned int slot,
        unsigned int lane,
        unsigned int rows,
        unsigned int cols,
        float value) {
    unsigned int index = 0;
    if (!coopstore_probe_q6addr_index(group, slot, lane, rows, cols, &index)) {
        return;
    }
    const int byte_offset = static_cast<int>(index * sizeof(float));
    __builtin_amdgcn_raw_buffer_store_b32(__builtin_bit_cast(int, value), rsrc, byte_offset, 0, 0);
}

static __device__ __forceinline__ void coopstore_probe_raw_store_acc_q6addr(
        __amdgpu_buffer_rsrc_t rsrc,
        unsigned int group,
        unsigned int slot,
        unsigned int lane,
        unsigned int rows,
        unsigned int cols,
        coopstore_probe_half8_vec acc) {
    coopstore_probe_raw_store_q6addr_value(
        rsrc, group, slot, lane, rows, cols, static_cast<float>(acc[slot * 2u]));
}

static __device__ __forceinline__ void coopstore_probe_ds_store_u16(
        __attribute__((address_space(3))) uint16_t * ptr,
        uint32_t value) {
    asm volatile("ds_write_b16 %0, %1 offset:0\n"
                 :
                 : "v"(ptr), "v"(value)
                 : "memory");
}

static __device__ __forceinline__ uint32_t coopstore_probe_ds_load_u16_d16(
        const __attribute__((address_space(3))) uint16_t * ptr) {
    uint32_t value = 0;
    asm volatile("ds_read_u16_d16 %0, %1 offset:0\n"
                 : "=v"(value)
                 : "v"(ptr)
                 : "memory");
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    return value;
}

static __device__ __forceinline__ _Float16 coopstore_probe_u16_to_f16(uint32_t value) {
    return __builtin_bit_cast(_Float16, static_cast<uint16_t>(value));
}

static __device__ __forceinline__ uint64_t coopstore_probe_ds_load_b64_nowait(
        const __attribute__((address_space(3))) uint64_t * ptr) {
    uint64_t value = 0;
    asm volatile("ds_read_b64 %0, %1 offset:0\n"
                 : "=v"(value)
                 : "v"(ptr)
                 : "memory");
    return value;
}

static __device__ __forceinline__ uint16_t coopstore_probe_fragment_half_bits(unsigned int frag) {
    const unsigned int index = frag < 4u ? frag : frag - 4u;
    return __builtin_bit_cast(uint16_t, static_cast<_Float16>(1.0f + static_cast<float>(index)));
}

static __device__ __forceinline__ void coopstore_probe_init_lds_fragments(
        uint64_t * sh_frag,
        unsigned int lane) {
#pragma unroll
    for (unsigned int frag = 0; frag < 8u; ++frag) {
        const uint64_t half_bits = static_cast<uint64_t>(coopstore_probe_fragment_half_bits(frag));
        const uint64_t packed = half_bits | (half_bits << 16) | (half_bits << 32) | (half_bits << 48);
#pragma unroll
        for (unsigned int item = 0; item < 4u; ++item) {
            sh_frag[frag * 256u + lane * 4u + item] = packed;
        }
    }
}

static __device__ __forceinline__ void coopstore_probe_init_lds_fragments4(
        uint64_t * sh_frag,
        unsigned int lane) {
#pragma unroll
    for (unsigned int frag = 0; frag < 4u; ++frag) {
        const uint64_t half_bits = static_cast<uint64_t>(coopstore_probe_fragment_half_bits(frag));
        const uint64_t packed = half_bits | (half_bits << 16) | (half_bits << 32) | (half_bits << 48);
#pragma unroll
        for (unsigned int item = 0; item < 4u; ++item) {
            sh_frag[frag * 256u + lane * 4u + item] = packed;
        }
    }
}

static __device__ __forceinline__ coopstore_probe_half16_vec coopstore_probe_load_lds_fragment(
        const __attribute__((address_space(3))) uint64_t * lds,
        unsigned int frag,
        unsigned int lane) {
    const unsigned int base = frag * 256u + lane * 4u;
    coopstore_probe_u64x4_vec raw;
    raw[0] = coopstore_probe_ds_load_b64_nowait(lds + base + 0u);
    raw[1] = coopstore_probe_ds_load_b64_nowait(lds + base + 1u);
    raw[2] = coopstore_probe_ds_load_b64_nowait(lds + base + 2u);
    raw[3] = coopstore_probe_ds_load_b64_nowait(lds + base + 3u);
    return __builtin_bit_cast(coopstore_probe_half16_vec, raw);
}

static __device__ __forceinline__ coopstore_probe_half16_vec coopstore_probe_load_lds_fragment_ring4(
        const __attribute__((address_space(3))) uint64_t * lds,
        unsigned int frag,
        unsigned int lane) {
    return coopstore_probe_load_lds_fragment(lds, frag & 3u, lane);
}

static __device__ __forceinline__ coopstore_probe_half16_vec coopstore_probe_copy_frag(
        coopstore_probe_half16_vec frag) {
    const coopstore_probe_u32x8_vec in = __builtin_bit_cast(coopstore_probe_u32x8_vec, frag);
    coopstore_probe_u32x8_vec out;
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
    return __builtin_bit_cast(coopstore_probe_half16_vec, out);
}

#define HRX_COOPSTORE_STORE_GROUP(GROUP_ID) do { \
    coopstore_probe_raw_store(rsrc, (GROUP_ID), 0u, lane); \
    coopstore_probe_raw_store(rsrc, (GROUP_ID), 1u, lane); \
    coopstore_probe_raw_store(rsrc, (GROUP_ID), 2u, lane); \
    coopstore_probe_raw_store(rsrc, (GROUP_ID), 3u, lane); \
} while (0)

#define HRX_COOPSTORE_GROUPS_0_7() do { \
    HRX_COOPSTORE_STORE_GROUP(0u);  HRX_COOPSTORE_STORE_GROUP(1u); \
    HRX_COOPSTORE_STORE_GROUP(2u);  HRX_COOPSTORE_STORE_GROUP(3u); \
    HRX_COOPSTORE_STORE_GROUP(4u);  HRX_COOPSTORE_STORE_GROUP(5u); \
    HRX_COOPSTORE_STORE_GROUP(6u);  HRX_COOPSTORE_STORE_GROUP(7u); \
} while (0)

#define HRX_COOPSTORE_GROUPS_8_15() do { \
    HRX_COOPSTORE_STORE_GROUP(8u);  HRX_COOPSTORE_STORE_GROUP(9u); \
    HRX_COOPSTORE_STORE_GROUP(10u); HRX_COOPSTORE_STORE_GROUP(11u); \
    HRX_COOPSTORE_STORE_GROUP(12u); HRX_COOPSTORE_STORE_GROUP(13u); \
    HRX_COOPSTORE_STORE_GROUP(14u); HRX_COOPSTORE_STORE_GROUP(15u); \
} while (0)

#define HRX_COOPSTORE_GROUPS_0_15() do { \
    HRX_COOPSTORE_GROUPS_0_7(); \
    HRX_COOPSTORE_GROUPS_8_15(); \
} while (0)

#define HRX_COOPSTORE_GROUPS_16_31() do { \
    HRX_COOPSTORE_STORE_GROUP(16u); HRX_COOPSTORE_STORE_GROUP(17u); \
    HRX_COOPSTORE_STORE_GROUP(18u); HRX_COOPSTORE_STORE_GROUP(19u); \
    HRX_COOPSTORE_STORE_GROUP(20u); HRX_COOPSTORE_STORE_GROUP(21u); \
    HRX_COOPSTORE_STORE_GROUP(22u); HRX_COOPSTORE_STORE_GROUP(23u); \
    HRX_COOPSTORE_STORE_GROUP(24u); HRX_COOPSTORE_STORE_GROUP(25u); \
    HRX_COOPSTORE_STORE_GROUP(26u); HRX_COOPSTORE_STORE_GROUP(27u); \
    HRX_COOPSTORE_STORE_GROUP(28u); HRX_COOPSTORE_STORE_GROUP(29u); \
    HRX_COOPSTORE_STORE_GROUP(30u); HRX_COOPSTORE_STORE_GROUP(31u); \
} while (0)

#define HRX_COOPSTORE_GROUPS_32_47() do { \
    HRX_COOPSTORE_STORE_GROUP(32u); HRX_COOPSTORE_STORE_GROUP(33u); \
    HRX_COOPSTORE_STORE_GROUP(34u); HRX_COOPSTORE_STORE_GROUP(35u); \
    HRX_COOPSTORE_STORE_GROUP(36u); HRX_COOPSTORE_STORE_GROUP(37u); \
    HRX_COOPSTORE_STORE_GROUP(38u); HRX_COOPSTORE_STORE_GROUP(39u); \
    HRX_COOPSTORE_STORE_GROUP(40u); HRX_COOPSTORE_STORE_GROUP(41u); \
    HRX_COOPSTORE_STORE_GROUP(42u); HRX_COOPSTORE_STORE_GROUP(43u); \
    HRX_COOPSTORE_STORE_GROUP(44u); HRX_COOPSTORE_STORE_GROUP(45u); \
    HRX_COOPSTORE_STORE_GROUP(46u); HRX_COOPSTORE_STORE_GROUP(47u); \
} while (0)

extern "C" __global__ __launch_bounds__(64, 1)
void coopstore_probe_linear64(float * dst, unsigned long long extent, unsigned int flags) {
    const unsigned int lane = __builtin_amdgcn_workitem_id_x() & 63u;
    const __amdgpu_buffer_rsrc_t rsrc = coopstore_probe_make_rsrc(dst, extent, flags);
    HRX_COOPSTORE_GROUPS_0_15();
}

extern "C" __global__ __launch_bounds__(64, 1)
void coopstore_probe_linear128(float * dst, unsigned long long extent, unsigned int flags) {
    const unsigned int lane = __builtin_amdgcn_workitem_id_x() & 63u;
    const __amdgpu_buffer_rsrc_t rsrc = coopstore_probe_make_rsrc(dst, extent, flags);
    HRX_COOPSTORE_GROUPS_0_15();
    HRX_COOPSTORE_GROUPS_16_31();
}

extern "C" __global__ __launch_bounds__(64, 1)
void coopstore_probe_linear192(float * dst, unsigned long long extent, unsigned int flags) {
    const unsigned int lane = __builtin_amdgcn_workitem_id_x() & 63u;
    const __amdgpu_buffer_rsrc_t rsrc = coopstore_probe_make_rsrc(dst, extent, flags);
    HRX_COOPSTORE_GROUPS_0_15();
    HRX_COOPSTORE_GROUPS_16_31();
    HRX_COOPSTORE_GROUPS_32_47();
}

extern "C" __global__ __launch_bounds__(64, 1)
void coopstore_probe_branch192(
        float * dst,
        unsigned long long extent,
        unsigned int flags,
        unsigned int selected_group) {
    const unsigned int lane = __builtin_amdgcn_workitem_id_x() & 63u;
    const __amdgpu_buffer_rsrc_t rsrc = coopstore_probe_make_rsrc(dst, extent, flags);
    volatile unsigned int selected = selected_group;
    if (selected == 0u)  { HRX_COOPSTORE_STORE_GROUP(0u); }
    if (selected == 1u)  { HRX_COOPSTORE_STORE_GROUP(1u); }
    if (selected == 2u)  { HRX_COOPSTORE_STORE_GROUP(2u); }
    if (selected == 3u)  { HRX_COOPSTORE_STORE_GROUP(3u); }
    if (selected == 4u)  { HRX_COOPSTORE_STORE_GROUP(4u); }
    if (selected == 5u)  { HRX_COOPSTORE_STORE_GROUP(5u); }
    if (selected == 6u)  { HRX_COOPSTORE_STORE_GROUP(6u); }
    if (selected == 7u)  { HRX_COOPSTORE_STORE_GROUP(7u); }
    if (selected == 8u)  { HRX_COOPSTORE_STORE_GROUP(8u); }
    if (selected == 9u)  { HRX_COOPSTORE_STORE_GROUP(9u); }
    if (selected == 10u) { HRX_COOPSTORE_STORE_GROUP(10u); }
    if (selected == 11u) { HRX_COOPSTORE_STORE_GROUP(11u); }
    if (selected == 12u) { HRX_COOPSTORE_STORE_GROUP(12u); }
    if (selected == 13u) { HRX_COOPSTORE_STORE_GROUP(13u); }
    if (selected == 14u) { HRX_COOPSTORE_STORE_GROUP(14u); }
    if (selected == 15u) { HRX_COOPSTORE_STORE_GROUP(15u); }
    if (selected == 16u) { HRX_COOPSTORE_STORE_GROUP(16u); }
    if (selected == 17u) { HRX_COOPSTORE_STORE_GROUP(17u); }
    if (selected == 18u) { HRX_COOPSTORE_STORE_GROUP(18u); }
    if (selected == 19u) { HRX_COOPSTORE_STORE_GROUP(19u); }
    if (selected == 20u) { HRX_COOPSTORE_STORE_GROUP(20u); }
    if (selected == 21u) { HRX_COOPSTORE_STORE_GROUP(21u); }
    if (selected == 22u) { HRX_COOPSTORE_STORE_GROUP(22u); }
    if (selected == 23u) { HRX_COOPSTORE_STORE_GROUP(23u); }
    if (selected == 24u) { HRX_COOPSTORE_STORE_GROUP(24u); }
    if (selected == 25u) { HRX_COOPSTORE_STORE_GROUP(25u); }
    if (selected == 26u) { HRX_COOPSTORE_STORE_GROUP(26u); }
    if (selected == 27u) { HRX_COOPSTORE_STORE_GROUP(27u); }
    if (selected == 28u) { HRX_COOPSTORE_STORE_GROUP(28u); }
    if (selected == 29u) { HRX_COOPSTORE_STORE_GROUP(29u); }
    if (selected == 30u) { HRX_COOPSTORE_STORE_GROUP(30u); }
    if (selected == 31u) { HRX_COOPSTORE_STORE_GROUP(31u); }
    if (selected == 32u) { HRX_COOPSTORE_STORE_GROUP(32u); }
    if (selected == 33u) { HRX_COOPSTORE_STORE_GROUP(33u); }
    if (selected == 34u) { HRX_COOPSTORE_STORE_GROUP(34u); }
    if (selected == 35u) { HRX_COOPSTORE_STORE_GROUP(35u); }
    if (selected == 36u) { HRX_COOPSTORE_STORE_GROUP(36u); }
    if (selected == 37u) { HRX_COOPSTORE_STORE_GROUP(37u); }
    if (selected == 38u) { HRX_COOPSTORE_STORE_GROUP(38u); }
    if (selected == 39u) { HRX_COOPSTORE_STORE_GROUP(39u); }
    if (selected == 40u) { HRX_COOPSTORE_STORE_GROUP(40u); }
    if (selected == 41u) { HRX_COOPSTORE_STORE_GROUP(41u); }
    if (selected == 42u) { HRX_COOPSTORE_STORE_GROUP(42u); }
    if (selected == 43u) { HRX_COOPSTORE_STORE_GROUP(43u); }
    if (selected == 44u) { HRX_COOPSTORE_STORE_GROUP(44u); }
    if (selected == 45u) { HRX_COOPSTORE_STORE_GROUP(45u); }
    if (selected == 46u) { HRX_COOPSTORE_STORE_GROUP(46u); }
    if (selected == 47u) { HRX_COOPSTORE_STORE_GROUP(47u); }
}

#define HRX_COOPSTORE_STAGE_STORE(GROUP_ID, SLOT_ID) do { \
    __attribute__((address_space(3))) uint16_t * ptr = \
        (__attribute__((address_space(3))) uint16_t *) (sh_stage + coopstore_probe_stage_index((GROUP_ID), (SLOT_ID), lane)); \
    coopstore_probe_ds_store_u16(ptr, coopstore_probe_stage_value((GROUP_ID), (SLOT_ID), lane)); \
} while (0)

#define HRX_COOPSTORE_STAGE_LOAD_STORE(GROUP_ID, SLOT_ID) do { \
    const __attribute__((address_space(3))) uint16_t * ptr = \
        (const __attribute__((address_space(3))) uint16_t *) (sh_stage + coopstore_probe_stage_index((GROUP_ID), (SLOT_ID), lane)); \
    const uint32_t value = coopstore_probe_ds_load_u16_d16(ptr); \
    coopstore_probe_raw_store_value(rsrc, (GROUP_ID), (SLOT_ID), lane, static_cast<float>(value & 0xffffu)); \
} while (0)

#define HRX_COOPSTORE_STAGE_STORE_BASE(GROUP_ID, SLOT_ID, BASE_GROUP) do { \
    __attribute__((address_space(3))) uint16_t * ptr = \
        (__attribute__((address_space(3))) uint16_t *) (sh_stage + coopstore_probe_stage_index_base((GROUP_ID), (SLOT_ID), lane, (BASE_GROUP))); \
    coopstore_probe_ds_store_u16(ptr, coopstore_probe_stage_value((GROUP_ID), (SLOT_ID), lane)); \
} while (0)

#define HRX_COOPSTORE_STAGE_LOAD_STORE_BASE(GROUP_ID, SLOT_ID, BASE_GROUP) do { \
    const __attribute__((address_space(3))) uint16_t * ptr = \
        (const __attribute__((address_space(3))) uint16_t *) (sh_stage + coopstore_probe_stage_index_base((GROUP_ID), (SLOT_ID), lane, (BASE_GROUP))); \
    const uint32_t value = coopstore_probe_ds_load_u16_d16(ptr); \
    coopstore_probe_raw_store_value(rsrc, (GROUP_ID), (SLOT_ID), lane, static_cast<float>(value & 0xffffu)); \
} while (0)

#define HRX_COOPSTORE_STAGE_STORE_LINEAR_BASE(GROUP_ID, SLOT_ID, BASE_GROUP) do { \
    __attribute__((address_space(3))) uint16_t * ptr = \
        (__attribute__((address_space(3))) uint16_t *) (sh_stage + coopstore_probe_stage_index_linear_base((GROUP_ID), (SLOT_ID), lane, (BASE_GROUP))); \
    coopstore_probe_ds_store_u16(ptr, coopstore_probe_stage_value((GROUP_ID), (SLOT_ID), lane)); \
} while (0)

#define HRX_COOPSTORE_STAGE_LOAD_STORE_LINEAR_BASE(GROUP_ID, SLOT_ID, BASE_GROUP) do { \
    const __attribute__((address_space(3))) uint16_t * ptr = \
        (const __attribute__((address_space(3))) uint16_t *) (sh_stage + coopstore_probe_stage_index_linear_base((GROUP_ID), (SLOT_ID), lane, (BASE_GROUP))); \
    const uint32_t value = coopstore_probe_ds_load_u16_d16(ptr); \
    coopstore_probe_raw_store_value(rsrc, (GROUP_ID), (SLOT_ID), lane, static_cast<float>(value & 0xffffu)); \
} while (0)

#define HRX_COOPSTORE_STAGE_STORE_ACC_BASE(GROUP_ID, SLOT_ID, BASE_GROUP) do { \
    __attribute__((address_space(3))) uint16_t * ptr = \
        (__attribute__((address_space(3))) uint16_t *) (sh_stage + coopstore_probe_stage_index_base((GROUP_ID), (SLOT_ID), lane, (BASE_GROUP))); \
    coopstore_probe_ds_store_u16(ptr, static_cast<uint32_t>(__builtin_bit_cast(uint16_t, acc[(GROUP_ID)][(SLOT_ID) * 2u]))); \
} while (0)

#define HRX_COOPSTORE_STAGE_LOAD_STORE_ACC_BASE(GROUP_ID, SLOT_ID, BASE_GROUP) do { \
    const __attribute__((address_space(3))) uint16_t * ptr = \
        (const __attribute__((address_space(3))) uint16_t *) (sh_stage + coopstore_probe_stage_index_base((GROUP_ID), (SLOT_ID), lane, (BASE_GROUP))); \
    const uint32_t value = coopstore_probe_ds_load_u16_d16(ptr); \
    coopstore_probe_raw_store_value(rsrc, (GROUP_ID), (SLOT_ID), lane, static_cast<float>(coopstore_probe_u16_to_f16(value))); \
} while (0)

#define HRX_COOPSTORE_STAGE_LOAD_STORE_ACC_Q6ADDR_BASE(GROUP_ID, SLOT_ID, BASE_GROUP) do { \
    const __attribute__((address_space(3))) uint16_t * ptr = \
        (const __attribute__((address_space(3))) uint16_t *) (sh_stage + coopstore_probe_stage_index_base((GROUP_ID), (SLOT_ID), lane, (BASE_GROUP))); \
    const uint32_t value = coopstore_probe_ds_load_u16_d16(ptr); \
    coopstore_probe_raw_store_q6addr_value( \
        rsrc, (GROUP_ID), (SLOT_ID), lane, rows, cols, static_cast<float>(coopstore_probe_u16_to_f16(value))); \
} while (0)

#define HRX_COOPSTORE_STAGE_LOAD_STORE_Q6ADDR_BASE(GROUP_ID, SLOT_ID, BASE_GROUP) do { \
    const __attribute__((address_space(3))) uint16_t * ptr = \
        (const __attribute__((address_space(3))) uint16_t *) (sh_stage + coopstore_probe_stage_index_base((GROUP_ID), (SLOT_ID), lane, (BASE_GROUP))); \
    const uint32_t value = coopstore_probe_ds_load_u16_d16(ptr); \
    coopstore_probe_raw_store_q6addr_value( \
        rsrc, (GROUP_ID), (SLOT_ID), lane, rows, cols, static_cast<float>(value & 0xffffu)); \
} while (0)

#define HRX_COOPSTORE_STAGE_STORE_GROUP(GROUP_ID) do { \
    HRX_COOPSTORE_STAGE_STORE((GROUP_ID), 0u); \
    HRX_COOPSTORE_STAGE_STORE((GROUP_ID), 1u); \
    HRX_COOPSTORE_STAGE_STORE((GROUP_ID), 2u); \
    HRX_COOPSTORE_STAGE_STORE((GROUP_ID), 3u); \
} while (0)

#define HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP(GROUP_ID) do { \
    HRX_COOPSTORE_STAGE_LOAD_STORE((GROUP_ID), 0u); \
    HRX_COOPSTORE_STAGE_LOAD_STORE((GROUP_ID), 1u); \
    HRX_COOPSTORE_STAGE_LOAD_STORE((GROUP_ID), 2u); \
    HRX_COOPSTORE_STAGE_LOAD_STORE((GROUP_ID), 3u); \
} while (0)

#define HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(GROUP_ID, BASE_GROUP) do { \
    HRX_COOPSTORE_STAGE_STORE_BASE((GROUP_ID), 0u, (BASE_GROUP)); \
    HRX_COOPSTORE_STAGE_STORE_BASE((GROUP_ID), 1u, (BASE_GROUP)); \
    HRX_COOPSTORE_STAGE_STORE_BASE((GROUP_ID), 2u, (BASE_GROUP)); \
    HRX_COOPSTORE_STAGE_STORE_BASE((GROUP_ID), 3u, (BASE_GROUP)); \
} while (0)

#define HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_BASE(GROUP_ID, BASE_GROUP) do { \
    HRX_COOPSTORE_STAGE_LOAD_STORE_BASE((GROUP_ID), 0u, (BASE_GROUP)); \
    HRX_COOPSTORE_STAGE_LOAD_STORE_BASE((GROUP_ID), 1u, (BASE_GROUP)); \
    HRX_COOPSTORE_STAGE_LOAD_STORE_BASE((GROUP_ID), 2u, (BASE_GROUP)); \
    HRX_COOPSTORE_STAGE_LOAD_STORE_BASE((GROUP_ID), 3u, (BASE_GROUP)); \
} while (0)

#define HRX_COOPSTORE_STAGE_STORE_GROUP_LINEAR_BASE(GROUP_ID, BASE_GROUP) do { \
    HRX_COOPSTORE_STAGE_STORE_LINEAR_BASE((GROUP_ID), 0u, (BASE_GROUP)); \
    HRX_COOPSTORE_STAGE_STORE_LINEAR_BASE((GROUP_ID), 1u, (BASE_GROUP)); \
    HRX_COOPSTORE_STAGE_STORE_LINEAR_BASE((GROUP_ID), 2u, (BASE_GROUP)); \
    HRX_COOPSTORE_STAGE_STORE_LINEAR_BASE((GROUP_ID), 3u, (BASE_GROUP)); \
} while (0)

#define HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_LINEAR_BASE(GROUP_ID, BASE_GROUP) do { \
    HRX_COOPSTORE_STAGE_LOAD_STORE_LINEAR_BASE((GROUP_ID), 0u, (BASE_GROUP)); \
    HRX_COOPSTORE_STAGE_LOAD_STORE_LINEAR_BASE((GROUP_ID), 1u, (BASE_GROUP)); \
    HRX_COOPSTORE_STAGE_LOAD_STORE_LINEAR_BASE((GROUP_ID), 2u, (BASE_GROUP)); \
    HRX_COOPSTORE_STAGE_LOAD_STORE_LINEAR_BASE((GROUP_ID), 3u, (BASE_GROUP)); \
} while (0)

#define HRX_COOPSTORE_STAGE_STORE_TYPED_LINEAR_BASE(GROUP_ID, SLOT_ID, BASE_GROUP) do { \
    sh_stage[coopstore_probe_stage_index_linear_base((GROUP_ID), (SLOT_ID), lane, (BASE_GROUP))] = \
        coopstore_probe_stage_value((GROUP_ID), (SLOT_ID), lane); \
} while (0)

#define HRX_COOPSTORE_STAGE_LOAD_STORE_TYPED_LINEAR_BASE(GROUP_ID, SLOT_ID, BASE_GROUP) do { \
    const uint32_t value = sh_stage[coopstore_probe_stage_index_linear_base((GROUP_ID), (SLOT_ID), lane, (BASE_GROUP))]; \
    coopstore_probe_raw_store_value(rsrc, (GROUP_ID), (SLOT_ID), lane, static_cast<float>(value & 0xffffu)); \
} while (0)

#define HRX_COOPSTORE_STAGE_STORE_GROUP_TYPED_LINEAR_BASE(GROUP_ID, BASE_GROUP) do { \
    HRX_COOPSTORE_STAGE_STORE_TYPED_LINEAR_BASE((GROUP_ID), 0u, (BASE_GROUP)); \
    HRX_COOPSTORE_STAGE_STORE_TYPED_LINEAR_BASE((GROUP_ID), 1u, (BASE_GROUP)); \
    HRX_COOPSTORE_STAGE_STORE_TYPED_LINEAR_BASE((GROUP_ID), 2u, (BASE_GROUP)); \
    HRX_COOPSTORE_STAGE_STORE_TYPED_LINEAR_BASE((GROUP_ID), 3u, (BASE_GROUP)); \
} while (0)

#define HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_TYPED_LINEAR_BASE(GROUP_ID, BASE_GROUP) do { \
    HRX_COOPSTORE_STAGE_LOAD_STORE_TYPED_LINEAR_BASE((GROUP_ID), 0u, (BASE_GROUP)); \
    HRX_COOPSTORE_STAGE_LOAD_STORE_TYPED_LINEAR_BASE((GROUP_ID), 1u, (BASE_GROUP)); \
    HRX_COOPSTORE_STAGE_LOAD_STORE_TYPED_LINEAR_BASE((GROUP_ID), 2u, (BASE_GROUP)); \
    HRX_COOPSTORE_STAGE_LOAD_STORE_TYPED_LINEAR_BASE((GROUP_ID), 3u, (BASE_GROUP)); \
} while (0)

#define HRX_COOPSTORE_STAGE_STORE_ACC_GROUP_BASE(GROUP_ID, BASE_GROUP) do { \
    HRX_COOPSTORE_STAGE_STORE_ACC_BASE((GROUP_ID), 0u, (BASE_GROUP)); \
    HRX_COOPSTORE_STAGE_STORE_ACC_BASE((GROUP_ID), 1u, (BASE_GROUP)); \
    HRX_COOPSTORE_STAGE_STORE_ACC_BASE((GROUP_ID), 2u, (BASE_GROUP)); \
    HRX_COOPSTORE_STAGE_STORE_ACC_BASE((GROUP_ID), 3u, (BASE_GROUP)); \
} while (0)

#define HRX_COOPSTORE_STAGE_LOAD_STORE_ACC_GROUP_BASE(GROUP_ID, BASE_GROUP) do { \
    HRX_COOPSTORE_STAGE_LOAD_STORE_ACC_BASE((GROUP_ID), 0u, (BASE_GROUP)); \
    HRX_COOPSTORE_STAGE_LOAD_STORE_ACC_BASE((GROUP_ID), 1u, (BASE_GROUP)); \
    HRX_COOPSTORE_STAGE_LOAD_STORE_ACC_BASE((GROUP_ID), 2u, (BASE_GROUP)); \
    HRX_COOPSTORE_STAGE_LOAD_STORE_ACC_BASE((GROUP_ID), 3u, (BASE_GROUP)); \
} while (0)

#define HRX_COOPSTORE_STAGE_LOAD_STORE_ACC_Q6ADDR_GROUP_BASE(GROUP_ID, BASE_GROUP) do { \
    HRX_COOPSTORE_STAGE_LOAD_STORE_ACC_Q6ADDR_BASE((GROUP_ID), 0u, (BASE_GROUP)); \
    HRX_COOPSTORE_STAGE_LOAD_STORE_ACC_Q6ADDR_BASE((GROUP_ID), 1u, (BASE_GROUP)); \
    HRX_COOPSTORE_STAGE_LOAD_STORE_ACC_Q6ADDR_BASE((GROUP_ID), 2u, (BASE_GROUP)); \
    HRX_COOPSTORE_STAGE_LOAD_STORE_ACC_Q6ADDR_BASE((GROUP_ID), 3u, (BASE_GROUP)); \
} while (0)

#define HRX_COOPSTORE_STAGE_LOAD_STORE_Q6ADDR_GROUP_BASE(GROUP_ID, BASE_GROUP) do { \
    HRX_COOPSTORE_STAGE_LOAD_STORE_Q6ADDR_BASE((GROUP_ID), 0u, (BASE_GROUP)); \
    HRX_COOPSTORE_STAGE_LOAD_STORE_Q6ADDR_BASE((GROUP_ID), 1u, (BASE_GROUP)); \
    HRX_COOPSTORE_STAGE_LOAD_STORE_Q6ADDR_BASE((GROUP_ID), 2u, (BASE_GROUP)); \
    HRX_COOPSTORE_STAGE_LOAD_STORE_Q6ADDR_BASE((GROUP_ID), 3u, (BASE_GROUP)); \
} while (0)

#define HRX_COOPSTORE_STAGE_STORE_GROUP_BASE8(GROUP_ID) \
    HRX_COOPSTORE_STAGE_STORE_GROUP_BASE((GROUP_ID), 8u)

#define HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_BASE8(GROUP_ID) \
    HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_BASE((GROUP_ID), 8u)

#define HRX_COOPSTORE_STAGE_STORE_GROUP_BASE24(GROUP_ID) \
    HRX_COOPSTORE_STAGE_STORE_GROUP_BASE((GROUP_ID), 24u)

#define HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_BASE24(GROUP_ID) \
    HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_BASE((GROUP_ID), 24u)

#define HRX_COOPSTORE_STAGE_STORE_GROUP_BASE32(GROUP_ID) \
    HRX_COOPSTORE_STAGE_STORE_GROUP_BASE((GROUP_ID), 32u)

#define HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_BASE32(GROUP_ID) \
    HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_BASE((GROUP_ID), 32u)

#define HRX_COOPSTORE_STAGE_STORE_GROUP_BASE40(GROUP_ID) \
    HRX_COOPSTORE_STAGE_STORE_GROUP_BASE((GROUP_ID), 40u)

#define HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_BASE40(GROUP_ID) \
    HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_BASE((GROUP_ID), 40u)

#define HRX_COOPSTORE_STAGE_STORE_GROUP_LINEAR(GROUP_ID) \
    HRX_COOPSTORE_STAGE_STORE_GROUP_LINEAR_BASE((GROUP_ID), 16u)

#define HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_LINEAR(GROUP_ID) \
    HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_LINEAR_BASE((GROUP_ID), 16u)

#define HRX_COOPSTORE_STAGE_STORE_GROUP_TYPED_LINEAR(GROUP_ID) \
    HRX_COOPSTORE_STAGE_STORE_GROUP_TYPED_LINEAR_BASE((GROUP_ID), 16u)

#define HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_TYPED_LINEAR(GROUP_ID) \
    HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_TYPED_LINEAR_BASE((GROUP_ID), 16u)

#define HRX_COOPSTORE_STAGE_STORE_GROUP_TYPED_LINEAR_BASE32(GROUP_ID) \
    HRX_COOPSTORE_STAGE_STORE_GROUP_TYPED_LINEAR_BASE((GROUP_ID), 32u)

#define HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_TYPED_LINEAR_BASE32(GROUP_ID) \
    HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_TYPED_LINEAR_BASE((GROUP_ID), 32u)

#define HRX_COOPSTORE_STAGE_STORE_ACC_GROUP_BASE8(GROUP_ID) \
    HRX_COOPSTORE_STAGE_STORE_ACC_GROUP_BASE((GROUP_ID), 8u)

#define HRX_COOPSTORE_STAGE_LOAD_STORE_ACC_GROUP_BASE8(GROUP_ID) \
    HRX_COOPSTORE_STAGE_LOAD_STORE_ACC_GROUP_BASE((GROUP_ID), 8u)

#define HRX_COOPSTORE_STORE_ACC_GROUP(GROUP_ID) do { \
    coopstore_probe_raw_store_acc(rsrc, (GROUP_ID), 0u, lane, acc[(GROUP_ID)]); \
    coopstore_probe_raw_store_acc(rsrc, (GROUP_ID), 1u, lane, acc[(GROUP_ID)]); \
    coopstore_probe_raw_store_acc(rsrc, (GROUP_ID), 2u, lane, acc[(GROUP_ID)]); \
    coopstore_probe_raw_store_acc(rsrc, (GROUP_ID), 3u, lane, acc[(GROUP_ID)]); \
} while (0)

#define HRX_COOPSTORE_STORE_ACC_Q6ADDR_GROUP(GROUP_ID) do { \
    coopstore_probe_raw_store_acc_q6addr(rsrc, (GROUP_ID), 0u, lane, rows, cols, acc[(GROUP_ID)]); \
    coopstore_probe_raw_store_acc_q6addr(rsrc, (GROUP_ID), 1u, lane, rows, cols, acc[(GROUP_ID)]); \
    coopstore_probe_raw_store_acc_q6addr(rsrc, (GROUP_ID), 2u, lane, rows, cols, acc[(GROUP_ID)]); \
    coopstore_probe_raw_store_acc_q6addr(rsrc, (GROUP_ID), 3u, lane, rows, cols, acc[(GROUP_ID)]); \
} while (0)

#define HRX_COOPSTORE_STORE_ACC_ELEMENT(OUT_GROUP_ID, SLOT_ID, SRC_GROUP_ID, ELEMENT_ID) \
    coopstore_probe_raw_store_acc_element( \
        rsrc, (OUT_GROUP_ID), (SLOT_ID), lane, acc[(SRC_GROUP_ID)], (ELEMENT_ID))

#define HRX_COOPSTORE_STORE_ACC_SLOTS_GROUP(SRC_GROUP_ID) do { \
    HRX_COOPSTORE_STORE_ACC_ELEMENT((SRC_GROUP_ID), 0u, (SRC_GROUP_ID), 0u); \
    HRX_COOPSTORE_STORE_ACC_ELEMENT((SRC_GROUP_ID), 1u, (SRC_GROUP_ID), 1u); \
    HRX_COOPSTORE_STORE_ACC_ELEMENT((SRC_GROUP_ID), 2u, (SRC_GROUP_ID), 2u); \
    HRX_COOPSTORE_STORE_ACC_ELEMENT((SRC_GROUP_ID), 3u, (SRC_GROUP_ID), 3u); \
    HRX_COOPSTORE_STORE_ACC_ELEMENT((SRC_GROUP_ID) + 8u, 0u, (SRC_GROUP_ID), 4u); \
    HRX_COOPSTORE_STORE_ACC_ELEMENT((SRC_GROUP_ID) + 8u, 1u, (SRC_GROUP_ID), 5u); \
    HRX_COOPSTORE_STORE_ACC_ELEMENT((SRC_GROUP_ID) + 8u, 2u, (SRC_GROUP_ID), 6u); \
    HRX_COOPSTORE_STORE_ACC_ELEMENT((SRC_GROUP_ID) + 8u, 3u, (SRC_GROUP_ID), 7u); \
} while (0)

#define HRX_COOPSTORE_ACC_SLOTS_GROUPS_0_7() do { \
    HRX_COOPSTORE_STORE_ACC_SLOTS_GROUP(0u); HRX_COOPSTORE_STORE_ACC_SLOTS_GROUP(1u); \
    HRX_COOPSTORE_STORE_ACC_SLOTS_GROUP(2u); HRX_COOPSTORE_STORE_ACC_SLOTS_GROUP(3u); \
    HRX_COOPSTORE_STORE_ACC_SLOTS_GROUP(4u); HRX_COOPSTORE_STORE_ACC_SLOTS_GROUP(5u); \
    HRX_COOPSTORE_STORE_ACC_SLOTS_GROUP(6u); HRX_COOPSTORE_STORE_ACC_SLOTS_GROUP(7u); \
} while (0)

#define HRX_COOPSTORE_ACC_GROUPS_0_7() do { \
    HRX_COOPSTORE_STORE_ACC_GROUP(0u);  HRX_COOPSTORE_STORE_ACC_GROUP(1u); \
    HRX_COOPSTORE_STORE_ACC_GROUP(2u);  HRX_COOPSTORE_STORE_ACC_GROUP(3u); \
    HRX_COOPSTORE_STORE_ACC_GROUP(4u);  HRX_COOPSTORE_STORE_ACC_GROUP(5u); \
    HRX_COOPSTORE_STORE_ACC_GROUP(6u);  HRX_COOPSTORE_STORE_ACC_GROUP(7u); \
} while (0)

#define HRX_COOPSTORE_ACC_Q6ADDR_GROUPS_0_7() do { \
    HRX_COOPSTORE_STORE_ACC_Q6ADDR_GROUP(0u);  HRX_COOPSTORE_STORE_ACC_Q6ADDR_GROUP(1u); \
    HRX_COOPSTORE_STORE_ACC_Q6ADDR_GROUP(2u);  HRX_COOPSTORE_STORE_ACC_Q6ADDR_GROUP(3u); \
    HRX_COOPSTORE_STORE_ACC_Q6ADDR_GROUP(4u);  HRX_COOPSTORE_STORE_ACC_Q6ADDR_GROUP(5u); \
    HRX_COOPSTORE_STORE_ACC_Q6ADDR_GROUP(6u);  HRX_COOPSTORE_STORE_ACC_Q6ADDR_GROUP(7u); \
} while (0)

#define HRX_COOPSTORE_ACC_GROUPS_8_15() do { \
    HRX_COOPSTORE_STORE_ACC_GROUP(8u);  HRX_COOPSTORE_STORE_ACC_GROUP(9u); \
    HRX_COOPSTORE_STORE_ACC_GROUP(10u); HRX_COOPSTORE_STORE_ACC_GROUP(11u); \
    HRX_COOPSTORE_STORE_ACC_GROUP(12u); HRX_COOPSTORE_STORE_ACC_GROUP(13u); \
    HRX_COOPSTORE_STORE_ACC_GROUP(14u); HRX_COOPSTORE_STORE_ACC_GROUP(15u); \
} while (0)

#define HRX_COOPSTORE_ACC_GROUPS_0_15() do { \
    HRX_COOPSTORE_ACC_GROUPS_0_7(); \
    HRX_COOPSTORE_ACC_GROUPS_8_15(); \
} while (0)

#define HRX_COOPSTORE_RADV_MOTIF_GROUP(GROUP_ID) do { \
    HRX_COOPSTORE_STORE_ACC_GROUP((GROUP_ID)); \
    HRX_COOPSTORE_STAGE_STORE_GROUP((GROUP_ID) + 16u); \
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory"); \
    HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP((GROUP_ID) + 16u); \
    HRX_COOPSTORE_STAGE_STORE_GROUP_BASE32((GROUP_ID) + 32u); \
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory"); \
    HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_BASE32((GROUP_ID) + 32u); \
} while (0)

#define HRX_COOPSTORE_RADV_MOTIF_GROUPS_0_15() do { \
    HRX_COOPSTORE_RADV_MOTIF_GROUP(0u);  HRX_COOPSTORE_RADV_MOTIF_GROUP(1u); \
    HRX_COOPSTORE_RADV_MOTIF_GROUP(2u);  HRX_COOPSTORE_RADV_MOTIF_GROUP(3u); \
    HRX_COOPSTORE_RADV_MOTIF_GROUP(4u);  HRX_COOPSTORE_RADV_MOTIF_GROUP(5u); \
    HRX_COOPSTORE_RADV_MOTIF_GROUP(6u);  HRX_COOPSTORE_RADV_MOTIF_GROUP(7u); \
    HRX_COOPSTORE_RADV_MOTIF_GROUP(8u);  HRX_COOPSTORE_RADV_MOTIF_GROUP(9u); \
    HRX_COOPSTORE_RADV_MOTIF_GROUP(10u); HRX_COOPSTORE_RADV_MOTIF_GROUP(11u); \
    HRX_COOPSTORE_RADV_MOTIF_GROUP(12u); HRX_COOPSTORE_RADV_MOTIF_GROUP(13u); \
    HRX_COOPSTORE_RADV_MOTIF_GROUP(14u); HRX_COOPSTORE_RADV_MOTIF_GROUP(15u); \
} while (0)

#define HRX_COOPSTORE_RADV_MOTIF_TYPED_GROUP(GROUP_ID) do { \
    HRX_COOPSTORE_STORE_ACC_GROUP((GROUP_ID)); \
    HRX_COOPSTORE_STAGE_STORE_GROUP_TYPED_LINEAR((GROUP_ID) + 16u); \
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory"); \
    HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_TYPED_LINEAR((GROUP_ID) + 16u); \
    HRX_COOPSTORE_STAGE_STORE_GROUP_TYPED_LINEAR_BASE32((GROUP_ID) + 32u); \
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory"); \
    HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_TYPED_LINEAR_BASE32((GROUP_ID) + 32u); \
} while (0)

#define HRX_COOPSTORE_RADV_MOTIF_TYPED_GROUPS_0_15() do { \
    HRX_COOPSTORE_RADV_MOTIF_TYPED_GROUP(0u);  HRX_COOPSTORE_RADV_MOTIF_TYPED_GROUP(1u); \
    HRX_COOPSTORE_RADV_MOTIF_TYPED_GROUP(2u);  HRX_COOPSTORE_RADV_MOTIF_TYPED_GROUP(3u); \
    HRX_COOPSTORE_RADV_MOTIF_TYPED_GROUP(4u);  HRX_COOPSTORE_RADV_MOTIF_TYPED_GROUP(5u); \
    HRX_COOPSTORE_RADV_MOTIF_TYPED_GROUP(6u);  HRX_COOPSTORE_RADV_MOTIF_TYPED_GROUP(7u); \
    HRX_COOPSTORE_RADV_MOTIF_TYPED_GROUP(8u);  HRX_COOPSTORE_RADV_MOTIF_TYPED_GROUP(9u); \
    HRX_COOPSTORE_RADV_MOTIF_TYPED_GROUP(10u); HRX_COOPSTORE_RADV_MOTIF_TYPED_GROUP(11u); \
    HRX_COOPSTORE_RADV_MOTIF_TYPED_GROUP(12u); HRX_COOPSTORE_RADV_MOTIF_TYPED_GROUP(13u); \
    HRX_COOPSTORE_RADV_MOTIF_TYPED_GROUP(14u); HRX_COOPSTORE_RADV_MOTIF_TYPED_GROUP(15u); \
} while (0)

#define HRX_COOPSTORE_STAGE_GROUPS_8_23(MACRO) do { \
    MACRO(8u);  MACRO(9u);  MACRO(10u); MACRO(11u); \
    MACRO(12u); MACRO(13u); MACRO(14u); MACRO(15u); \
    MACRO(16u); MACRO(17u); MACRO(18u); MACRO(19u); \
    MACRO(20u); MACRO(21u); MACRO(22u); MACRO(23u); \
} while (0)

#define HRX_COOPSTORE_STAGE_GROUPS_8_15(MACRO) do { \
    MACRO(8u);  MACRO(9u);  MACRO(10u); MACRO(11u); \
    MACRO(12u); MACRO(13u); MACRO(14u); MACRO(15u); \
} while (0)

#define HRX_COOPSTORE_STAGE_GROUPS_16_23(MACRO) do { \
    MACRO(16u); MACRO(17u); MACRO(18u); MACRO(19u); \
    MACRO(20u); MACRO(21u); MACRO(22u); MACRO(23u); \
} while (0)

#define HRX_COOPSTORE_STAGE_GROUPS_16_31(MACRO) do { \
    MACRO(16u); MACRO(17u); MACRO(18u); MACRO(19u); \
    MACRO(20u); MACRO(21u); MACRO(22u); MACRO(23u); \
    MACRO(24u); MACRO(25u); MACRO(26u); MACRO(27u); \
    MACRO(28u); MACRO(29u); MACRO(30u); MACRO(31u); \
} while (0)

#define HRX_COOPSTORE_STAGE_GROUPS_32_39(MACRO) do { \
    MACRO(32u); MACRO(33u); MACRO(34u); MACRO(35u); \
    MACRO(36u); MACRO(37u); MACRO(38u); MACRO(39u); \
} while (0)

#define HRX_COOPSTORE_STAGE_GROUPS_40_47(MACRO) do { \
    MACRO(40u); MACRO(41u); MACRO(42u); MACRO(43u); \
    MACRO(44u); MACRO(45u); MACRO(46u); MACRO(47u); \
} while (0)

#define HRX_COOPSTORE_STAGE_GROUPS_32_47(MACRO) do { \
    MACRO(32u); MACRO(33u); MACRO(34u); MACRO(35u); \
    MACRO(36u); MACRO(37u); MACRO(38u); MACRO(39u); \
    MACRO(40u); MACRO(41u); MACRO(42u); MACRO(43u); \
    MACRO(44u); MACRO(45u); MACRO(46u); MACRO(47u); \
} while (0)

extern "C" __global__ __launch_bounds__(256, 1)
void coopstore_probe_radv_mixed96(float * dst, unsigned long long extent, unsigned int flags) {
    __shared__ uint16_t sh_stage[16 * 16 * 16];
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & 63u;
    const __amdgpu_buffer_rsrc_t rsrc = coopstore_probe_make_rsrc(dst, extent, flags);

    if (tid < 64u) {
        HRX_COOPSTORE_GROUPS_0_7();
        HRX_COOPSTORE_STAGE_GROUPS_8_23(HRX_COOPSTORE_STAGE_STORE_GROUP_BASE8);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        HRX_COOPSTORE_STAGE_GROUPS_8_23(HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_BASE8);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();
}

extern "C" __global__ __launch_bounds__(256, 1)
void coopstore_probe_radv_mixed192(float * dst, unsigned long long extent, unsigned int flags) {
    __shared__ uint16_t sh_stage[32 * 16 * 16];
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & 63u;
    const __amdgpu_buffer_rsrc_t rsrc = coopstore_probe_make_rsrc(dst, extent, flags);

    if (tid < 64u) {
        HRX_COOPSTORE_GROUPS_0_15();
        HRX_COOPSTORE_STAGE_GROUPS_16_31(HRX_COOPSTORE_STAGE_STORE_GROUP);
        HRX_COOPSTORE_STAGE_GROUPS_32_47(HRX_COOPSTORE_STAGE_STORE_GROUP);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        HRX_COOPSTORE_STAGE_GROUPS_16_31(HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP);
        HRX_COOPSTORE_STAGE_GROUPS_32_47(HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();
}

template <unsigned int direct_groups>
static __device__ __forceinline__ void coopstore_probe_make_wmma_acc(
        coopstore_probe_half8_vec (&acc)[direct_groups]) {
    coopstore_probe_half16_vec a[4];
    coopstore_probe_half16_vec b[4];
#pragma unroll
    for (unsigned int frag = 0; frag < 4u; ++frag) {
#pragma unroll
        for (unsigned int i = 0; i < 16u; ++i) {
            a[frag][i] = static_cast<_Float16>(1.0f + static_cast<float>(frag));
            b[frag][i] = static_cast<_Float16>(1.0f + static_cast<float>(frag));
        }
    }
#pragma unroll
    for (unsigned int group = 0; group < direct_groups; ++group) {
#pragma unroll
        for (unsigned int slot = 0; slot < 8u; ++slot) {
            acc[group][slot] = static_cast<_Float16>(0.0f);
        }
    }
#pragma unroll
    for (unsigned int group = 0; group < direct_groups; ++group) {
        const unsigned int row_frag = group & 3u;
        const unsigned int col_frag = (group >> 2u) & 3u;
        acc[group] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(
            a[row_frag], b[col_frag], acc[group], false);
    }
}

template <unsigned int direct_groups>
static __device__ __forceinline__ void coopstore_probe_make_wmma_acc_from_lds(
        const __attribute__((address_space(3))) uint64_t * lds,
        unsigned int lane,
        coopstore_probe_half8_vec (&acc)[direct_groups]) {
    coopstore_probe_half16_vec a[4];
    coopstore_probe_half16_vec b[4];
#pragma unroll
    for (unsigned int frag = 0; frag < 4u; ++frag) {
        a[frag] = coopstore_probe_load_lds_fragment(lds, frag, lane);
        b[frag] = coopstore_probe_load_lds_fragment(lds, frag + 4u, lane);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");

#pragma unroll
    for (unsigned int group = 0; group < direct_groups; ++group) {
#pragma unroll
        for (unsigned int slot = 0; slot < 8u; ++slot) {
            acc[group][slot] = static_cast<_Float16>(0.0f);
        }
    }
#pragma unroll
    for (unsigned int group = 0; group < direct_groups; ++group) {
        const unsigned int row_frag = group & 3u;
        const unsigned int col_frag = (group >> 2u) & 3u;
        acc[group] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(
            a[row_frag], b[col_frag], acc[group], false);
    }
}

template <unsigned int direct_groups>
static __device__ __forceinline__ void coopstore_probe_zero_acc(
        coopstore_probe_half8_vec (&acc)[direct_groups]) {
#pragma unroll
    for (unsigned int group = 0; group < direct_groups; ++group) {
#pragma unroll
        for (unsigned int slot = 0; slot < 8u; ++slot) {
            acc[group][slot] = static_cast<_Float16>(0.0f);
        }
    }
}

template <unsigned int direct_groups>
static __device__ __forceinline__ void coopstore_probe_accumulate_wmma_from_lds(
        const __attribute__((address_space(3))) uint64_t * lds,
        unsigned int lane,
        coopstore_probe_half8_vec (&acc)[direct_groups]) {
    coopstore_probe_half16_vec a[4];
    coopstore_probe_half16_vec b[4];
#pragma unroll
    for (unsigned int frag = 0; frag < 4u; ++frag) {
        a[frag] = coopstore_probe_load_lds_fragment(lds, frag, lane);
        b[frag] = coopstore_probe_load_lds_fragment(lds, frag + 4u, lane);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");

#pragma unroll
    for (unsigned int group = 0; group < direct_groups; ++group) {
        const unsigned int row_frag = group & 3u;
        const unsigned int col_frag = (group >> 2u) & 3u;
        acc[group] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(
            a[row_frag], b[col_frag], acc[group], false);
    }
}

static __device__ __forceinline__ coopstore_probe_half8_vec coopstore_probe_wmma_f16_w64_asm(
        coopstore_probe_half16_vec a_frag,
        coopstore_probe_half16_vec b_frag,
        coopstore_probe_half8_vec acc) {
    coopstore_probe_half8_vec out;
    asm volatile("v_wmma_f16_16x16x16_f16 %0, %1, %2, %3\n"
                 : "=v"(out)
                 : "v"(a_frag), "v"(b_frag), "v"(acc)
                 : "memory");
    return out;
}

template <unsigned int direct_groups>
static __device__ __forceinline__ void coopstore_probe_accumulate_wmma_from_lds_kloop(
        const __attribute__((address_space(3))) uint64_t * lds,
        unsigned int lane,
        coopstore_probe_half8_vec (&acc)[direct_groups]) {
#pragma clang loop unroll(disable)
    for (unsigned int k_iter = 0; k_iter < 2u; ++k_iter) {
        (void) k_iter;
        coopstore_probe_half16_vec a[4];
        coopstore_probe_half16_vec b[4];
#pragma unroll
        for (unsigned int frag = 0; frag < 4u; ++frag) {
            a[frag] = coopstore_probe_load_lds_fragment(lds, frag, lane);
            b[frag] = coopstore_probe_load_lds_fragment(lds, frag + 4u, lane);
        }
        asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");

#pragma unroll
        for (unsigned int group = 0; group < direct_groups; ++group) {
            const unsigned int row_frag = group & 3u;
            const unsigned int col_frag = (group >> 2u) & 3u;
            acc[group] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(
                a[row_frag], b[col_frag], acc[group], false);
        }
    }
}

template <unsigned int direct_groups>
static __device__ __forceinline__ void coopstore_probe_accumulate_wmma_from_lds_kloop_asm(
        const __attribute__((address_space(3))) uint64_t * lds,
        unsigned int lane,
        coopstore_probe_half8_vec (&acc)[direct_groups]) {
#pragma clang loop unroll(disable)
    for (unsigned int k_iter = 0; k_iter < 2u; ++k_iter) {
        (void) k_iter;
        coopstore_probe_half16_vec a[4];
        coopstore_probe_half16_vec b[4];
#pragma unroll
        for (unsigned int frag = 0; frag < 4u; ++frag) {
            a[frag] = coopstore_probe_load_lds_fragment(lds, frag, lane);
            b[frag] = coopstore_probe_load_lds_fragment(lds, frag + 4u, lane);
        }
        asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");

#pragma unroll
        for (unsigned int group = 0; group < direct_groups; ++group) {
            const unsigned int row_frag = group & 3u;
            const unsigned int col_frag = (group >> 2u) & 3u;
            acc[group] = coopstore_probe_wmma_f16_w64_asm(a[row_frag], b[col_frag], acc[group]);
        }
    }
}

template <unsigned int direct_groups>
static __device__ __forceinline__ void coopstore_probe_accumulate_wmma_from_lds_ring12(
        const __attribute__((address_space(3))) uint64_t * lds,
        unsigned int lane,
        coopstore_probe_half8_vec (&acc)[direct_groups]) {
    coopstore_probe_half16_vec a[6];
    coopstore_probe_half16_vec b[6];
#pragma unroll
    for (unsigned int frag = 0; frag < 6u; ++frag) {
        a[frag] = coopstore_probe_load_lds_fragment_ring4(lds, frag, lane);
        b[frag] = coopstore_probe_load_lds_fragment_ring4(lds, frag + 6u, lane);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");

#pragma unroll
    for (unsigned int group = 0; group < direct_groups; ++group) {
        const unsigned int row_frag = group % 6u;
        const unsigned int col_frag = (group + group / 6u) % 6u;
        acc[group] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(
            a[row_frag], b[col_frag], acc[group], false);
    }
}

template <unsigned int direct_groups, bool copy_a, bool copy_b>
static __device__ __forceinline__ void coopstore_probe_accumulate_wmma_from_lds_ring12_copy(
        const __attribute__((address_space(3))) uint64_t * lds,
        unsigned int lane,
        coopstore_probe_half8_vec (&acc)[direct_groups]) {
    coopstore_probe_half16_vec a[6];
    coopstore_probe_half16_vec b[6];
#pragma unroll
    for (unsigned int frag = 0; frag < 6u; ++frag) {
        a[frag] = coopstore_probe_load_lds_fragment_ring4(lds, frag, lane);
        b[frag] = coopstore_probe_load_lds_fragment_ring4(lds, frag + 6u, lane);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");

#pragma unroll
    for (unsigned int group = 0; group < direct_groups; ++group) {
        const unsigned int row_frag = group % 6u;
        const unsigned int col_frag = (group + group / 6u) % 6u;
        coopstore_probe_half16_vec a_use = a[row_frag];
        coopstore_probe_half16_vec b_use = b[col_frag];
        if constexpr (copy_a) {
            a_use = coopstore_probe_copy_frag(a_use);
        }
        if constexpr (copy_b) {
            b_use = coopstore_probe_copy_frag(b_use);
        }
        acc[group] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(
            a_use, b_use, acc[group], false);
    }
}

extern "C" __global__ __launch_bounds__(256, 1)
void coopstore_probe_wmma_radv_mixed96(float * dst, unsigned long long extent, unsigned int flags) {
    __shared__ uint16_t sh_stage[16 * 16 * 16];
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & 63u;
    const __amdgpu_buffer_rsrc_t rsrc = coopstore_probe_make_rsrc(dst, extent, flags);

    if (tid < 64u) {
        coopstore_probe_half8_vec acc[8];
        coopstore_probe_make_wmma_acc(acc);
        HRX_COOPSTORE_ACC_GROUPS_0_7();
        HRX_COOPSTORE_STAGE_GROUPS_8_23(HRX_COOPSTORE_STAGE_STORE_GROUP_BASE8);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        HRX_COOPSTORE_STAGE_GROUPS_8_23(HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_BASE8);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();
}

extern "C" __global__ __launch_bounds__(256, 1)
void coopstore_probe_wmma_radv_mixed192(float * dst, unsigned long long extent, unsigned int flags) {
    __shared__ uint16_t sh_stage[32 * 16 * 16];
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & 63u;
    const __amdgpu_buffer_rsrc_t rsrc = coopstore_probe_make_rsrc(dst, extent, flags);

    if (tid < 64u) {
        coopstore_probe_half8_vec acc[16];
        coopstore_probe_make_wmma_acc(acc);
        HRX_COOPSTORE_ACC_GROUPS_0_15();
        HRX_COOPSTORE_STAGE_GROUPS_16_31(HRX_COOPSTORE_STAGE_STORE_GROUP);
        HRX_COOPSTORE_STAGE_GROUPS_32_47(HRX_COOPSTORE_STAGE_STORE_GROUP);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        HRX_COOPSTORE_STAGE_GROUPS_16_31(HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP);
        HRX_COOPSTORE_STAGE_GROUPS_32_47(HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();
}

extern "C" __global__ __launch_bounds__(256, 1)
void coopstore_probe_wmma_lds_radv_mixed96(float * dst, unsigned long long extent, unsigned int flags) {
    __shared__ uint64_t sh_frag[8 * 64 * 4];
    __shared__ uint16_t sh_stage[16 * 16 * 16];
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & 63u;
    const __amdgpu_buffer_rsrc_t rsrc = coopstore_probe_make_rsrc(dst, extent, flags);

    if (tid < 64u) {
        coopstore_probe_init_lds_fragments(sh_frag, lane);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        const __attribute__((address_space(3))) uint64_t * lds =
            (const __attribute__((address_space(3))) uint64_t *) sh_frag;
        coopstore_probe_half8_vec acc[16];
        coopstore_probe_make_wmma_acc_from_lds(lds, lane, acc);
        HRX_COOPSTORE_ACC_GROUPS_0_7();
        HRX_COOPSTORE_STAGE_GROUPS_8_15(HRX_COOPSTORE_STAGE_STORE_ACC_GROUP_BASE8);
        HRX_COOPSTORE_STAGE_GROUPS_16_23(HRX_COOPSTORE_STAGE_STORE_GROUP_BASE8);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        HRX_COOPSTORE_STAGE_GROUPS_8_15(HRX_COOPSTORE_STAGE_LOAD_STORE_ACC_GROUP_BASE8);
        HRX_COOPSTORE_STAGE_GROUPS_16_23(HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_BASE8);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();
}

extern "C" __global__ __launch_bounds__(256, 1)
void coopstore_probe_wmma_lds_radv_mixed192(float * dst, unsigned long long extent, unsigned int flags) {
    __shared__ uint64_t sh_frag[8 * 64 * 4];
    __shared__ uint16_t sh_stage[32 * 16 * 16];
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & 63u;
    const __amdgpu_buffer_rsrc_t rsrc = coopstore_probe_make_rsrc(dst, extent, flags);

    if (tid < 64u) {
        coopstore_probe_init_lds_fragments(sh_frag, lane);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        const __attribute__((address_space(3))) uint64_t * lds =
            (const __attribute__((address_space(3))) uint64_t *) sh_frag;
        coopstore_probe_half8_vec acc[16];
        coopstore_probe_make_wmma_acc_from_lds(lds, lane, acc);
        HRX_COOPSTORE_ACC_GROUPS_0_15();
        HRX_COOPSTORE_STAGE_GROUPS_16_31(HRX_COOPSTORE_STAGE_STORE_GROUP);
        HRX_COOPSTORE_STAGE_GROUPS_32_47(HRX_COOPSTORE_STAGE_STORE_GROUP);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        HRX_COOPSTORE_STAGE_GROUPS_16_31(HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP);
        HRX_COOPSTORE_STAGE_GROUPS_32_47(HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();
}

extern "C" __global__ __launch_bounds__(256, 1)
void coopstore_probe_wmma_lds_k2_radv_mixed192(float * dst, unsigned long long extent, unsigned int flags) {
    __shared__ uint64_t sh_frag[8 * 64 * 4];
    __shared__ uint16_t sh_stage[32 * 16 * 16];
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & 63u;
    const __amdgpu_buffer_rsrc_t rsrc = coopstore_probe_make_rsrc(dst, extent, flags);

    if (tid < 64u) {
        coopstore_probe_init_lds_fragments(sh_frag, lane);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        const __attribute__((address_space(3))) uint64_t * lds =
            (const __attribute__((address_space(3))) uint64_t *) sh_frag;
        coopstore_probe_half8_vec acc[16];
        coopstore_probe_zero_acc(acc);
        coopstore_probe_accumulate_wmma_from_lds(lds, lane, acc);
        coopstore_probe_accumulate_wmma_from_lds(lds, lane, acc);
        HRX_COOPSTORE_ACC_GROUPS_0_15();
        HRX_COOPSTORE_STAGE_GROUPS_16_31(HRX_COOPSTORE_STAGE_STORE_GROUP);
        HRX_COOPSTORE_STAGE_GROUPS_32_47(HRX_COOPSTORE_STAGE_STORE_GROUP);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        HRX_COOPSTORE_STAGE_GROUPS_16_31(HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP);
        HRX_COOPSTORE_STAGE_GROUPS_32_47(HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();
}

extern "C" __global__ __launch_bounds__(256, 1)
void coopstore_probe_wmma_lds_k2_stagefirst_mixed192(float * dst, unsigned long long extent, unsigned int flags) {
    __shared__ uint64_t sh_frag[8 * 64 * 4];
    __shared__ uint16_t sh_stage[32 * 16 * 16];
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & 63u;
    const __amdgpu_buffer_rsrc_t rsrc = coopstore_probe_make_rsrc(dst, extent, flags);

    if (tid < 64u) {
        coopstore_probe_init_lds_fragments(sh_frag, lane);
        HRX_COOPSTORE_STAGE_GROUPS_16_31(HRX_COOPSTORE_STAGE_STORE_GROUP);
        HRX_COOPSTORE_STAGE_GROUPS_32_47(HRX_COOPSTORE_STAGE_STORE_GROUP);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        const __attribute__((address_space(3))) uint64_t * lds =
            (const __attribute__((address_space(3))) uint64_t *) sh_frag;
        coopstore_probe_half8_vec acc[16];
        coopstore_probe_zero_acc(acc);
        coopstore_probe_accumulate_wmma_from_lds(lds, lane, acc);
        coopstore_probe_accumulate_wmma_from_lds(lds, lane, acc);
        HRX_COOPSTORE_ACC_GROUPS_0_15();
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        HRX_COOPSTORE_STAGE_GROUPS_16_31(HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP);
        HRX_COOPSTORE_STAGE_GROUPS_32_47(HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();
}

extern "C" __global__ __launch_bounds__(256, 1)
void coopstore_probe_wmma_lds_k2_mixed96(float * dst, unsigned long long extent, unsigned int flags) {
    __shared__ uint64_t sh_frag[8 * 64 * 4];
    __shared__ uint16_t sh_stage[8 * 16 * 16];
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & 63u;
    const __amdgpu_buffer_rsrc_t rsrc = coopstore_probe_make_rsrc(dst, extent, flags);

    if (tid < 64u) {
        coopstore_probe_init_lds_fragments(sh_frag, lane);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        const __attribute__((address_space(3))) uint64_t * lds =
            (const __attribute__((address_space(3))) uint64_t *) sh_frag;
        coopstore_probe_half8_vec acc[16];
        coopstore_probe_zero_acc(acc);
        coopstore_probe_accumulate_wmma_from_lds(lds, lane, acc);
        coopstore_probe_accumulate_wmma_from_lds(lds, lane, acc);
        HRX_COOPSTORE_ACC_GROUPS_0_15();
        HRX_COOPSTORE_STAGE_GROUPS_16_23(HRX_COOPSTORE_STAGE_STORE_GROUP);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        HRX_COOPSTORE_STAGE_GROUPS_16_23(HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();
}

extern "C" __global__ __launch_bounds__(256, 1)
void coopstore_probe_wmma_lds_k2_typedstage_mixed96(float * dst, unsigned long long extent, unsigned int flags) {
    __shared__ uint64_t sh_frag[8 * 64 * 4];
    __shared__ uint16_t sh_stage[8 * 64 * 4];
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & 63u;
    const __amdgpu_buffer_rsrc_t rsrc = coopstore_probe_make_rsrc(dst, extent, flags);

    if (tid < 64u) {
        coopstore_probe_init_lds_fragments(sh_frag, lane);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        const __attribute__((address_space(3))) uint64_t * lds =
            (const __attribute__((address_space(3))) uint64_t *) sh_frag;
        coopstore_probe_half8_vec acc[16];
        coopstore_probe_zero_acc(acc);
        coopstore_probe_accumulate_wmma_from_lds(lds, lane, acc);
        coopstore_probe_accumulate_wmma_from_lds(lds, lane, acc);
        HRX_COOPSTORE_ACC_GROUPS_0_15();
        HRX_COOPSTORE_STAGE_GROUPS_16_23(HRX_COOPSTORE_STAGE_STORE_GROUP_TYPED_LINEAR);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        HRX_COOPSTORE_STAGE_GROUPS_16_23(HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_TYPED_LINEAR);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();
}

extern "C" __global__ __launch_bounds__(256, 1)
void coopstore_probe_wmma_lds_k2_kloop_mixed96(float * dst, unsigned long long extent, unsigned int flags) {
    __shared__ uint64_t sh_frag[8 * 64 * 4];
    __shared__ uint16_t sh_stage[8 * 16 * 16];
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & 63u;
    const __amdgpu_buffer_rsrc_t rsrc = coopstore_probe_make_rsrc(dst, extent, flags);

    if (tid < 64u) {
        coopstore_probe_init_lds_fragments(sh_frag, lane);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        const __attribute__((address_space(3))) uint64_t * lds =
            (const __attribute__((address_space(3))) uint64_t *) sh_frag;
        coopstore_probe_half8_vec acc[16];
        coopstore_probe_zero_acc(acc);
        coopstore_probe_accumulate_wmma_from_lds_kloop(lds, lane, acc);
        HRX_COOPSTORE_ACC_GROUPS_0_15();
        HRX_COOPSTORE_STAGE_GROUPS_16_23(HRX_COOPSTORE_STAGE_STORE_GROUP);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        HRX_COOPSTORE_STAGE_GROUPS_16_23(HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();
}

extern "C" __global__ __launch_bounds__(256, 1)
void coopstore_probe_wmma_lds_k2_kloop_direct64(float * dst, unsigned long long extent, unsigned int flags) {
    __shared__ uint64_t sh_frag[8 * 64 * 4];
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & 63u;
    const __amdgpu_buffer_rsrc_t rsrc = coopstore_probe_make_rsrc(dst, extent, flags);

    if (tid < 64u) {
        coopstore_probe_init_lds_fragments(sh_frag, lane);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        const __attribute__((address_space(3))) uint64_t * lds =
            (const __attribute__((address_space(3))) uint64_t *) sh_frag;
        coopstore_probe_half8_vec acc[16];
        coopstore_probe_zero_acc(acc);
        coopstore_probe_accumulate_wmma_from_lds_kloop(lds, lane, acc);
        HRX_COOPSTORE_ACC_GROUPS_0_15();
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();
}

extern "C" __global__ __launch_bounds__(256, 1)
void coopstore_probe_wmma_lds_k2_kloop_asm_mixed96(float * dst, unsigned long long extent, unsigned int flags) {
    __shared__ uint64_t sh_frag[8 * 64 * 4];
    __shared__ uint16_t sh_stage[8 * 16 * 16];
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & 63u;
    const __amdgpu_buffer_rsrc_t rsrc = coopstore_probe_make_rsrc(dst, extent, flags);

    if (tid < 64u) {
        coopstore_probe_init_lds_fragments(sh_frag, lane);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        const __attribute__((address_space(3))) uint64_t * lds =
            (const __attribute__((address_space(3))) uint64_t *) sh_frag;
        coopstore_probe_half8_vec acc[16];
        coopstore_probe_zero_acc(acc);
        coopstore_probe_accumulate_wmma_from_lds_kloop_asm(lds, lane, acc);
        HRX_COOPSTORE_ACC_GROUPS_0_15();
        HRX_COOPSTORE_STAGE_GROUPS_16_23(HRX_COOPSTORE_STAGE_STORE_GROUP);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        HRX_COOPSTORE_STAGE_GROUPS_16_23(HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();
}

extern "C" __global__ __launch_bounds__(256, 1)
void coopstore_probe_wmma_lds_k2_kloop_asm_stagefirst_mixed96(float * dst, unsigned long long extent, unsigned int flags) {
    __shared__ uint64_t sh_frag[8 * 64 * 4];
    __shared__ uint16_t sh_stage[8 * 16 * 16];
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & 63u;
    const __amdgpu_buffer_rsrc_t rsrc = coopstore_probe_make_rsrc(dst, extent, flags);

    if (tid < 64u) {
        coopstore_probe_init_lds_fragments(sh_frag, lane);
        HRX_COOPSTORE_STAGE_GROUPS_16_23(HRX_COOPSTORE_STAGE_STORE_GROUP);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        const __attribute__((address_space(3))) uint64_t * lds =
            (const __attribute__((address_space(3))) uint64_t *) sh_frag;
        coopstore_probe_half8_vec acc[16];
        coopstore_probe_zero_acc(acc);
        coopstore_probe_accumulate_wmma_from_lds_kloop_asm(lds, lane, acc);
        HRX_COOPSTORE_ACC_GROUPS_0_15();
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        HRX_COOPSTORE_STAGE_GROUPS_16_23(HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();
}

extern "C" __global__ __launch_bounds__(256, 1)
void coopstore_probe_wmma_lds_k2_kloop_asm_linearstage_mixed96(float * dst, unsigned long long extent, unsigned int flags) {
    __shared__ uint64_t sh_frag[8 * 64 * 4];
    __shared__ uint16_t sh_stage[8 * 64 * 4];
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & 63u;
    const __amdgpu_buffer_rsrc_t rsrc = coopstore_probe_make_rsrc(dst, extent, flags);

    if (tid < 64u) {
        coopstore_probe_init_lds_fragments(sh_frag, lane);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        const __attribute__((address_space(3))) uint64_t * lds =
            (const __attribute__((address_space(3))) uint64_t *) sh_frag;
        coopstore_probe_half8_vec acc[16];
        coopstore_probe_zero_acc(acc);
        coopstore_probe_accumulate_wmma_from_lds_kloop_asm(lds, lane, acc);
        HRX_COOPSTORE_ACC_GROUPS_0_15();
        HRX_COOPSTORE_STAGE_GROUPS_16_23(HRX_COOPSTORE_STAGE_STORE_GROUP_LINEAR);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        HRX_COOPSTORE_STAGE_GROUPS_16_23(HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_LINEAR);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();
}

extern "C" __global__ __launch_bounds__(256, 1)
void coopstore_probe_wmma_lds_k2_kloop_asm_typedstage_mixed96(float * dst, unsigned long long extent, unsigned int flags) {
    __shared__ uint64_t sh_frag[8 * 64 * 4];
    __shared__ uint16_t sh_stage[8 * 64 * 4];
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & 63u;
    const __amdgpu_buffer_rsrc_t rsrc = coopstore_probe_make_rsrc(dst, extent, flags);

    if (tid < 64u) {
        coopstore_probe_init_lds_fragments(sh_frag, lane);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        const __attribute__((address_space(3))) uint64_t * lds =
            (const __attribute__((address_space(3))) uint64_t *) sh_frag;
        coopstore_probe_half8_vec acc[16];
        coopstore_probe_zero_acc(acc);
        coopstore_probe_accumulate_wmma_from_lds_kloop_asm(lds, lane, acc);
        HRX_COOPSTORE_ACC_GROUPS_0_15();
        HRX_COOPSTORE_STAGE_GROUPS_16_23(HRX_COOPSTORE_STAGE_STORE_GROUP_TYPED_LINEAR);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        HRX_COOPSTORE_STAGE_GROUPS_16_23(HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_TYPED_LINEAR);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();
}

extern "C" __global__ __launch_bounds__(256, 1)
void coopstore_probe_wmma_lds_k2_kloop_asm_direct64(float * dst, unsigned long long extent, unsigned int flags) {
    __shared__ uint64_t sh_frag[8 * 64 * 4];
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & 63u;
    const __amdgpu_buffer_rsrc_t rsrc = coopstore_probe_make_rsrc(dst, extent, flags);

    if (tid < 64u) {
        coopstore_probe_init_lds_fragments(sh_frag, lane);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        const __attribute__((address_space(3))) uint64_t * lds =
            (const __attribute__((address_space(3))) uint64_t *) sh_frag;
        coopstore_probe_half8_vec acc[16];
        coopstore_probe_zero_acc(acc);
        coopstore_probe_accumulate_wmma_from_lds_kloop_asm(lds, lane, acc);
        HRX_COOPSTORE_ACC_GROUPS_0_15();
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();
}

extern "C" __global__ __launch_bounds__(256, 1)
void coopstore_probe_wmma_lds_vk64_radv96(float * dst, unsigned long long extent, unsigned int flags) {
    __shared__ uint64_t sh_frag[4 * 64 * 4];
    __shared__ uint16_t sh_stage[6 * 16 * 16];
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & 63u;
    const __amdgpu_buffer_rsrc_t rsrc = coopstore_probe_make_rsrc(dst, extent, flags);

    if (tid < 64u) {
        coopstore_probe_init_lds_fragments4(sh_frag, lane);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        const __attribute__((address_space(3))) uint64_t * lds =
            (const __attribute__((address_space(3))) uint64_t *) sh_frag;
        coopstore_probe_half8_vec acc[16];
        coopstore_probe_zero_acc(acc);
        coopstore_probe_accumulate_wmma_from_lds_ring12(lds, lane, acc);
        HRX_COOPSTORE_GROUPS_0_7();
        __attribute__((address_space(3))) uint16_t * sink =
            (__attribute__((address_space(3))) uint16_t *) sh_stage;
#pragma unroll
        for (unsigned int group = 0; group < 16u; ++group) {
            coopstore_probe_ds_store_u16(
                sink + lane + group * HRX_COOPSTORE_LANES,
                static_cast<uint32_t>(__builtin_bit_cast(uint16_t, acc[group][0])));
        }
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(8u, 8u);
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(9u, 8u);
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(10u, 8u);
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(11u, 8u);
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(12u, 8u);
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(13u, 8u);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();
    if (tid < 64u) {
        HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_BASE(8u, 8u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_BASE(9u, 8u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_BASE(10u, 8u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_BASE(11u, 8u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_BASE(12u, 8u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_BASE(13u, 8u);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(14u, 14u);
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(15u, 14u);
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(16u, 14u);
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(17u, 14u);
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(18u, 14u);
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(19u, 14u);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();
    if (tid < 64u) {
        HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_BASE(14u, 14u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_BASE(15u, 14u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_BASE(16u, 14u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_BASE(17u, 14u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_BASE(18u, 14u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_BASE(19u, 14u);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(20u, 20u);
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(21u, 20u);
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(22u, 20u);
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(23u, 20u);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();
    if (tid < 64u) {
        HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_BASE(20u, 20u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_BASE(21u, 20u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_BASE(22u, 20u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_BASE(23u, 20u);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();
}

extern "C" __global__ __launch_bounds__(256, 1)
void coopstore_probe_wmma_lds_vk64_radv96_accdirect(float * dst, unsigned long long extent, unsigned int flags) {
    __shared__ uint64_t sh_frag[4 * 64 * 4];
    __shared__ uint16_t sh_stage[6 * 16 * 16];
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & 63u;
    const __amdgpu_buffer_rsrc_t rsrc = coopstore_probe_make_rsrc(dst, extent, flags);

    if (tid < 64u) {
        coopstore_probe_init_lds_fragments4(sh_frag, lane);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        const __attribute__((address_space(3))) uint64_t * lds =
            (const __attribute__((address_space(3))) uint64_t *) sh_frag;
        coopstore_probe_half8_vec acc[16];
        coopstore_probe_zero_acc(acc);
        coopstore_probe_accumulate_wmma_from_lds_ring12(lds, lane, acc);
        HRX_COOPSTORE_ACC_GROUPS_0_7();
        __attribute__((address_space(3))) uint16_t * sink =
            (__attribute__((address_space(3))) uint16_t *) sh_stage;
#pragma unroll
        for (unsigned int group = 0; group < 16u; ++group) {
            coopstore_probe_ds_store_u16(
                sink + lane + group * HRX_COOPSTORE_LANES,
                static_cast<uint32_t>(__builtin_bit_cast(uint16_t, acc[group][0])));
        }
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(8u, 8u);
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(9u, 8u);
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(10u, 8u);
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(11u, 8u);
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(12u, 8u);
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(13u, 8u);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();
    if (tid < 64u) {
        HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_BASE(8u, 8u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_BASE(9u, 8u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_BASE(10u, 8u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_BASE(11u, 8u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_BASE(12u, 8u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_BASE(13u, 8u);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(14u, 14u);
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(15u, 14u);
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(16u, 14u);
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(17u, 14u);
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(18u, 14u);
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(19u, 14u);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();
    if (tid < 64u) {
        HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_BASE(14u, 14u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_BASE(15u, 14u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_BASE(16u, 14u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_BASE(17u, 14u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_BASE(18u, 14u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_BASE(19u, 14u);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(20u, 20u);
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(21u, 20u);
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(22u, 20u);
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(23u, 20u);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();
    if (tid < 64u) {
        HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_BASE(20u, 20u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_BASE(21u, 20u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_BASE(22u, 20u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_BASE(23u, 20u);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();
}

extern "C" __global__ __launch_bounds__(256, 1)
void coopstore_probe_wmma_lds_vk64_radv96_accslots(float * dst, unsigned long long extent, unsigned int flags) {
    __shared__ uint64_t sh_frag[4 * 64 * 4];
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & 63u;
    const __amdgpu_buffer_rsrc_t rsrc = coopstore_probe_make_rsrc(dst, extent, flags);

    if (tid < 64u) {
        coopstore_probe_init_lds_fragments4(sh_frag, lane);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        const __attribute__((address_space(3))) uint64_t * lds =
            (const __attribute__((address_space(3))) uint64_t *) sh_frag;
        coopstore_probe_half8_vec acc[16];
        coopstore_probe_zero_acc(acc);
        coopstore_probe_accumulate_wmma_from_lds_ring12(lds, lane, acc);
        HRX_COOPSTORE_ACC_SLOTS_GROUPS_0_7();
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();
}

template <bool copy_a, bool copy_b>
__global__ __launch_bounds__(256, 1)
void coopstore_probe_wmma_lds_vk64_radv96_accdirect_copy(float * dst, unsigned long long extent, unsigned int flags) {
    __shared__ uint64_t sh_frag[4 * 64 * 4];
    __shared__ uint16_t sh_stage[6 * 16 * 16];
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & 63u;
    const __amdgpu_buffer_rsrc_t rsrc = coopstore_probe_make_rsrc(dst, extent, flags);

    if (tid < 64u) {
        coopstore_probe_init_lds_fragments4(sh_frag, lane);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        const __attribute__((address_space(3))) uint64_t * lds =
            (const __attribute__((address_space(3))) uint64_t *) sh_frag;
        coopstore_probe_half8_vec acc[16];
        coopstore_probe_zero_acc(acc);
        coopstore_probe_accumulate_wmma_from_lds_ring12_copy<16, copy_a, copy_b>(lds, lane, acc);
        HRX_COOPSTORE_ACC_GROUPS_0_7();
        __attribute__((address_space(3))) uint16_t * sink =
            (__attribute__((address_space(3))) uint16_t *) sh_stage;
#pragma unroll
        for (unsigned int group = 0; group < 16u; ++group) {
            coopstore_probe_ds_store_u16(
                sink + lane + group * HRX_COOPSTORE_LANES,
                static_cast<uint32_t>(__builtin_bit_cast(uint16_t, acc[group][0])));
        }
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(8u, 8u);
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(9u, 8u);
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(10u, 8u);
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(11u, 8u);
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(12u, 8u);
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(13u, 8u);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();
    if (tid < 64u) {
        HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_BASE(8u, 8u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_BASE(9u, 8u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_BASE(10u, 8u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_BASE(11u, 8u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_BASE(12u, 8u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_BASE(13u, 8u);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(14u, 14u);
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(15u, 14u);
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(16u, 14u);
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(17u, 14u);
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(18u, 14u);
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(19u, 14u);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();
    if (tid < 64u) {
        HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_BASE(14u, 14u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_BASE(15u, 14u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_BASE(16u, 14u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_BASE(17u, 14u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_BASE(18u, 14u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_BASE(19u, 14u);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(20u, 20u);
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(21u, 20u);
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(22u, 20u);
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(23u, 20u);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();
    if (tid < 64u) {
        HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_BASE(20u, 20u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_BASE(21u, 20u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_BASE(22u, 20u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_BASE(23u, 20u);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();
}

template <bool copy_a, bool copy_b, bool sink_acc>
__global__ __launch_bounds__(256, 1)
void coopstore_probe_wmma_lds_vk64_radv96_accdirect_copy_q6addr(
        float * dst,
        unsigned long long extent,
        unsigned int flags,
        unsigned int rows,
        unsigned int cols) {
    __shared__ uint64_t sh_frag[4 * 64 * 4];
    __shared__ uint16_t sh_stage[6 * 16 * 16];
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & 63u;
    const __amdgpu_buffer_rsrc_t rsrc = coopstore_probe_make_rsrc(dst, extent, flags);

    if (tid < 64u) {
        coopstore_probe_init_lds_fragments4(sh_frag, lane);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        const __attribute__((address_space(3))) uint64_t * lds =
            (const __attribute__((address_space(3))) uint64_t *) sh_frag;
        coopstore_probe_half8_vec acc[16];
        coopstore_probe_zero_acc(acc);
        coopstore_probe_accumulate_wmma_from_lds_ring12_copy<16, copy_a, copy_b>(lds, lane, acc);
        HRX_COOPSTORE_ACC_Q6ADDR_GROUPS_0_7();
        if constexpr (sink_acc) {
        __attribute__((address_space(3))) uint16_t * sink =
            (__attribute__((address_space(3))) uint16_t *) sh_stage;
#pragma unroll
        for (unsigned int group = 0; group < 16u; ++group) {
            coopstore_probe_ds_store_u16(
                sink + lane + group * HRX_COOPSTORE_LANES,
                static_cast<uint32_t>(__builtin_bit_cast(uint16_t, acc[group][0])));
        }
        }
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(8u, 8u);
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(9u, 8u);
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(10u, 8u);
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(11u, 8u);
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(12u, 8u);
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(13u, 8u);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();
    if (tid < 64u) {
        HRX_COOPSTORE_STAGE_LOAD_STORE_Q6ADDR_GROUP_BASE(8u, 8u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_Q6ADDR_GROUP_BASE(9u, 8u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_Q6ADDR_GROUP_BASE(10u, 8u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_Q6ADDR_GROUP_BASE(11u, 8u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_Q6ADDR_GROUP_BASE(12u, 8u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_Q6ADDR_GROUP_BASE(13u, 8u);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(14u, 14u);
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(15u, 14u);
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(16u, 14u);
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(17u, 14u);
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(18u, 14u);
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(19u, 14u);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();
    if (tid < 64u) {
        HRX_COOPSTORE_STAGE_LOAD_STORE_Q6ADDR_GROUP_BASE(14u, 14u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_Q6ADDR_GROUP_BASE(15u, 14u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_Q6ADDR_GROUP_BASE(16u, 14u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_Q6ADDR_GROUP_BASE(17u, 14u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_Q6ADDR_GROUP_BASE(18u, 14u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_Q6ADDR_GROUP_BASE(19u, 14u);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(20u, 20u);
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(21u, 20u);
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(22u, 20u);
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(23u, 20u);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();
    if (tid < 64u) {
        HRX_COOPSTORE_STAGE_LOAD_STORE_Q6ADDR_GROUP_BASE(20u, 20u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_Q6ADDR_GROUP_BASE(21u, 20u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_Q6ADDR_GROUP_BASE(22u, 20u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_Q6ADDR_GROUP_BASE(23u, 20u);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();
}

template <bool copy_a, bool copy_b>
__global__ __launch_bounds__(256, 1)
void coopstore_probe_wmma_lds_vk64_radv96_accdirect_copy_q6addr_upperreal(
        float * dst,
        unsigned long long extent,
        unsigned int flags,
        unsigned int rows,
        unsigned int cols) {
    __shared__ uint64_t sh_frag[4 * 64 * 4];
    __shared__ uint16_t sh_stage[6 * 16 * 16];
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & 63u;
    const __amdgpu_buffer_rsrc_t rsrc = coopstore_probe_make_rsrc(dst, extent, flags);
    coopstore_probe_half8_vec acc[16];

    if (tid < 64u) {
        coopstore_probe_init_lds_fragments4(sh_frag, lane);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        const __attribute__((address_space(3))) uint64_t * lds =
            (const __attribute__((address_space(3))) uint64_t *) sh_frag;
        coopstore_probe_zero_acc(acc);
        coopstore_probe_accumulate_wmma_from_lds_ring12_copy<16, copy_a, copy_b>(lds, lane, acc);
        HRX_COOPSTORE_ACC_Q6ADDR_GROUPS_0_7();

        HRX_COOPSTORE_STAGE_STORE_ACC_GROUP_BASE(8u, 8u);
        HRX_COOPSTORE_STAGE_STORE_ACC_GROUP_BASE(9u, 8u);
        HRX_COOPSTORE_STAGE_STORE_ACC_GROUP_BASE(10u, 8u);
        HRX_COOPSTORE_STAGE_STORE_ACC_GROUP_BASE(11u, 8u);
        HRX_COOPSTORE_STAGE_STORE_ACC_GROUP_BASE(12u, 8u);
        HRX_COOPSTORE_STAGE_STORE_ACC_GROUP_BASE(13u, 8u);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();
    if (tid < 64u) {
        HRX_COOPSTORE_STAGE_LOAD_STORE_ACC_Q6ADDR_GROUP_BASE(8u, 8u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_ACC_Q6ADDR_GROUP_BASE(9u, 8u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_ACC_Q6ADDR_GROUP_BASE(10u, 8u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_ACC_Q6ADDR_GROUP_BASE(11u, 8u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_ACC_Q6ADDR_GROUP_BASE(12u, 8u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_ACC_Q6ADDR_GROUP_BASE(13u, 8u);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        HRX_COOPSTORE_STAGE_STORE_ACC_GROUP_BASE(14u, 14u);
        HRX_COOPSTORE_STAGE_STORE_ACC_GROUP_BASE(15u, 14u);
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(16u, 14u);
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(17u, 14u);
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(18u, 14u);
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(19u, 14u);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();
    if (tid < 64u) {
        HRX_COOPSTORE_STAGE_LOAD_STORE_ACC_Q6ADDR_GROUP_BASE(14u, 14u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_ACC_Q6ADDR_GROUP_BASE(15u, 14u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_Q6ADDR_GROUP_BASE(16u, 14u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_Q6ADDR_GROUP_BASE(17u, 14u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_Q6ADDR_GROUP_BASE(18u, 14u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_Q6ADDR_GROUP_BASE(19u, 14u);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(20u, 20u);
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(21u, 20u);
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(22u, 20u);
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(23u, 20u);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();
    if (tid < 64u) {
        HRX_COOPSTORE_STAGE_LOAD_STORE_Q6ADDR_GROUP_BASE(20u, 20u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_Q6ADDR_GROUP_BASE(21u, 20u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_Q6ADDR_GROUP_BASE(22u, 20u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_Q6ADDR_GROUP_BASE(23u, 20u);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();
}

template <bool copy_a, bool copy_b>
__global__ __launch_bounds__(256, 1)
void coopstore_probe_wmma_lds_vk64_radv96_accdirect_copy_q6addr_upperwait(
        float * dst,
        unsigned long long extent,
        unsigned int flags,
        unsigned int rows,
        unsigned int cols) {
    __shared__ uint64_t sh_frag[4 * 64 * 4];
    __shared__ uint16_t sh_stage[6 * 16 * 16];
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & 63u;
    const __amdgpu_buffer_rsrc_t rsrc = coopstore_probe_make_rsrc(dst, extent, flags);
    coopstore_probe_half8_vec acc[16];

    if (tid < 64u) {
        coopstore_probe_init_lds_fragments4(sh_frag, lane);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        const __attribute__((address_space(3))) uint64_t * lds =
            (const __attribute__((address_space(3))) uint64_t *) sh_frag;
        coopstore_probe_zero_acc(acc);
        coopstore_probe_accumulate_wmma_from_lds_ring12_copy<16, copy_a, copy_b>(lds, lane, acc);
        HRX_COOPSTORE_ACC_Q6ADDR_GROUPS_0_7();

        HRX_COOPSTORE_STAGE_STORE_ACC_GROUP_BASE(8u, 8u);
        HRX_COOPSTORE_STAGE_STORE_ACC_GROUP_BASE(9u, 8u);
        HRX_COOPSTORE_STAGE_STORE_ACC_GROUP_BASE(10u, 8u);
        HRX_COOPSTORE_STAGE_STORE_ACC_GROUP_BASE(11u, 8u);
        HRX_COOPSTORE_STAGE_STORE_ACC_GROUP_BASE(12u, 8u);
        HRX_COOPSTORE_STAGE_STORE_ACC_GROUP_BASE(13u, 8u);
        asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
        HRX_COOPSTORE_STAGE_LOAD_STORE_ACC_Q6ADDR_GROUP_BASE(8u, 8u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_ACC_Q6ADDR_GROUP_BASE(9u, 8u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_ACC_Q6ADDR_GROUP_BASE(10u, 8u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_ACC_Q6ADDR_GROUP_BASE(11u, 8u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_ACC_Q6ADDR_GROUP_BASE(12u, 8u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_ACC_Q6ADDR_GROUP_BASE(13u, 8u);

        HRX_COOPSTORE_STAGE_STORE_ACC_GROUP_BASE(14u, 14u);
        HRX_COOPSTORE_STAGE_STORE_ACC_GROUP_BASE(15u, 14u);
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(16u, 14u);
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(17u, 14u);
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(18u, 14u);
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(19u, 14u);
        asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
        HRX_COOPSTORE_STAGE_LOAD_STORE_ACC_Q6ADDR_GROUP_BASE(14u, 14u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_ACC_Q6ADDR_GROUP_BASE(15u, 14u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_Q6ADDR_GROUP_BASE(16u, 14u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_Q6ADDR_GROUP_BASE(17u, 14u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_Q6ADDR_GROUP_BASE(18u, 14u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_Q6ADDR_GROUP_BASE(19u, 14u);

        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(20u, 20u);
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(21u, 20u);
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(22u, 20u);
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(23u, 20u);
        asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
        HRX_COOPSTORE_STAGE_LOAD_STORE_Q6ADDR_GROUP_BASE(20u, 20u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_Q6ADDR_GROUP_BASE(21u, 20u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_Q6ADDR_GROUP_BASE(22u, 20u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_Q6ADDR_GROUP_BASE(23u, 20u);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
}

template <bool copy_a, bool copy_b>
__global__ __launch_bounds__(256, 1)
void coopstore_probe_wmma_lds_vk64_radv96_accdirect_copy_q6addr_upperwide(
        float * dst,
        unsigned long long extent,
        unsigned int flags,
        unsigned int rows,
        unsigned int cols) {
    __shared__ uint64_t sh_frag[4 * 64 * 4];
    __shared__ uint16_t sh_stage[16 * 16 * 16];
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & 63u;
    const __amdgpu_buffer_rsrc_t rsrc = coopstore_probe_make_rsrc(dst, extent, flags);
    coopstore_probe_half8_vec acc[16];

    if (tid < 64u) {
        coopstore_probe_init_lds_fragments4(sh_frag, lane);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        const __attribute__((address_space(3))) uint64_t * lds =
            (const __attribute__((address_space(3))) uint64_t *) sh_frag;
        coopstore_probe_zero_acc(acc);
        coopstore_probe_accumulate_wmma_from_lds_ring12_copy<16, copy_a, copy_b>(lds, lane, acc);
        HRX_COOPSTORE_ACC_Q6ADDR_GROUPS_0_7();

        HRX_COOPSTORE_STAGE_STORE_ACC_GROUP_BASE(8u, 8u);
        HRX_COOPSTORE_STAGE_STORE_ACC_GROUP_BASE(9u, 8u);
        HRX_COOPSTORE_STAGE_STORE_ACC_GROUP_BASE(10u, 8u);
        HRX_COOPSTORE_STAGE_STORE_ACC_GROUP_BASE(11u, 8u);
        HRX_COOPSTORE_STAGE_STORE_ACC_GROUP_BASE(12u, 8u);
        HRX_COOPSTORE_STAGE_STORE_ACC_GROUP_BASE(13u, 8u);
        HRX_COOPSTORE_STAGE_STORE_ACC_GROUP_BASE(14u, 8u);
        HRX_COOPSTORE_STAGE_STORE_ACC_GROUP_BASE(15u, 8u);
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(16u, 8u);
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(17u, 8u);
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(18u, 8u);
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(19u, 8u);
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(20u, 8u);
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(21u, 8u);
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(22u, 8u);
        HRX_COOPSTORE_STAGE_STORE_GROUP_BASE(23u, 8u);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();
    if (tid < 64u) {
        HRX_COOPSTORE_STAGE_LOAD_STORE_ACC_Q6ADDR_GROUP_BASE(8u, 8u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_ACC_Q6ADDR_GROUP_BASE(9u, 8u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_ACC_Q6ADDR_GROUP_BASE(10u, 8u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_ACC_Q6ADDR_GROUP_BASE(11u, 8u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_ACC_Q6ADDR_GROUP_BASE(12u, 8u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_ACC_Q6ADDR_GROUP_BASE(13u, 8u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_ACC_Q6ADDR_GROUP_BASE(14u, 8u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_ACC_Q6ADDR_GROUP_BASE(15u, 8u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_Q6ADDR_GROUP_BASE(16u, 8u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_Q6ADDR_GROUP_BASE(17u, 8u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_Q6ADDR_GROUP_BASE(18u, 8u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_Q6ADDR_GROUP_BASE(19u, 8u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_Q6ADDR_GROUP_BASE(20u, 8u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_Q6ADDR_GROUP_BASE(21u, 8u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_Q6ADDR_GROUP_BASE(22u, 8u);
        HRX_COOPSTORE_STAGE_LOAD_STORE_Q6ADDR_GROUP_BASE(23u, 8u);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();
}

extern "C" __global__ __launch_bounds__(256, 1)
void coopstore_probe_wmma_lds_k2_mixed128(float * dst, unsigned long long extent, unsigned int flags) {
    __shared__ uint64_t sh_frag[8 * 64 * 4];
    __shared__ uint16_t sh_stage[16 * 16 * 16];
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & 63u;
    const __amdgpu_buffer_rsrc_t rsrc = coopstore_probe_make_rsrc(dst, extent, flags);

    if (tid < 64u) {
        coopstore_probe_init_lds_fragments(sh_frag, lane);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        const __attribute__((address_space(3))) uint64_t * lds =
            (const __attribute__((address_space(3))) uint64_t *) sh_frag;
        coopstore_probe_half8_vec acc[16];
        coopstore_probe_zero_acc(acc);
        coopstore_probe_accumulate_wmma_from_lds(lds, lane, acc);
        coopstore_probe_accumulate_wmma_from_lds(lds, lane, acc);
        HRX_COOPSTORE_ACC_GROUPS_0_15();
        HRX_COOPSTORE_STAGE_GROUPS_16_31(HRX_COOPSTORE_STAGE_STORE_GROUP);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        HRX_COOPSTORE_STAGE_GROUPS_16_31(HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();
}

extern "C" __global__ __launch_bounds__(256, 1)
void coopstore_probe_wmma_lds_k2_mixed128_padded32(float * dst, unsigned long long extent, unsigned int flags) {
    __shared__ uint64_t sh_frag[8 * 64 * 4];
    __shared__ uint16_t sh_stage[32 * 16 * 16];
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & 63u;
    const __amdgpu_buffer_rsrc_t rsrc = coopstore_probe_make_rsrc(dst, extent, flags);

    if (tid < 64u) {
        coopstore_probe_init_lds_fragments(sh_frag, lane);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        const __attribute__((address_space(3))) uint64_t * lds =
            (const __attribute__((address_space(3))) uint64_t *) sh_frag;
        coopstore_probe_half8_vec acc[16];
        coopstore_probe_zero_acc(acc);
        coopstore_probe_accumulate_wmma_from_lds(lds, lane, acc);
        coopstore_probe_accumulate_wmma_from_lds(lds, lane, acc);
        HRX_COOPSTORE_ACC_GROUPS_0_15();
        HRX_COOPSTORE_STAGE_GROUPS_16_31(HRX_COOPSTORE_STAGE_STORE_GROUP);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        HRX_COOPSTORE_STAGE_GROUPS_16_31(HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();
}

extern "C" __global__ __launch_bounds__(256, 1)
void coopstore_probe_wmma_lds_k2_mixed160_lo(float * dst, unsigned long long extent, unsigned int flags) {
    __shared__ uint64_t sh_frag[8 * 64 * 4];
    __shared__ uint16_t sh_stage[32 * 16 * 16];
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & 63u;
    const __amdgpu_buffer_rsrc_t rsrc = coopstore_probe_make_rsrc(dst, extent, flags);

    if (tid < 64u) {
        coopstore_probe_init_lds_fragments(sh_frag, lane);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        const __attribute__((address_space(3))) uint64_t * lds =
            (const __attribute__((address_space(3))) uint64_t *) sh_frag;
        coopstore_probe_half8_vec acc[16];
        coopstore_probe_zero_acc(acc);
        coopstore_probe_accumulate_wmma_from_lds(lds, lane, acc);
        coopstore_probe_accumulate_wmma_from_lds(lds, lane, acc);
        HRX_COOPSTORE_ACC_GROUPS_0_15();
        HRX_COOPSTORE_STAGE_GROUPS_16_31(HRX_COOPSTORE_STAGE_STORE_GROUP);
        HRX_COOPSTORE_STAGE_GROUPS_32_39(HRX_COOPSTORE_STAGE_STORE_GROUP);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        HRX_COOPSTORE_STAGE_GROUPS_16_31(HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP);
        HRX_COOPSTORE_STAGE_GROUPS_32_39(HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();
}

extern "C" __global__ __launch_bounds__(256, 1)
void coopstore_probe_wmma_lds_k2_mixed160_hi(float * dst, unsigned long long extent, unsigned int flags) {
    __shared__ uint64_t sh_frag[8 * 64 * 4];
    __shared__ uint16_t sh_stage[32 * 16 * 16];
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & 63u;
    const __amdgpu_buffer_rsrc_t rsrc = coopstore_probe_make_rsrc(dst, extent, flags);

    if (tid < 64u) {
        coopstore_probe_init_lds_fragments(sh_frag, lane);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        const __attribute__((address_space(3))) uint64_t * lds =
            (const __attribute__((address_space(3))) uint64_t *) sh_frag;
        coopstore_probe_half8_vec acc[16];
        coopstore_probe_zero_acc(acc);
        coopstore_probe_accumulate_wmma_from_lds(lds, lane, acc);
        coopstore_probe_accumulate_wmma_from_lds(lds, lane, acc);
        HRX_COOPSTORE_ACC_GROUPS_0_15();
        HRX_COOPSTORE_STAGE_GROUPS_16_31(HRX_COOPSTORE_STAGE_STORE_GROUP);
        HRX_COOPSTORE_STAGE_GROUPS_40_47(HRX_COOPSTORE_STAGE_STORE_GROUP);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        HRX_COOPSTORE_STAGE_GROUPS_16_31(HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP);
        HRX_COOPSTORE_STAGE_GROUPS_40_47(HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();
}

extern "C" __global__ __launch_bounds__(256, 1)
void coopstore_probe_wmma_lds_k2_mixed160_lo_tight(float * dst, unsigned long long extent, unsigned int flags) {
    __shared__ uint64_t sh_frag[8 * 64 * 4];
    __shared__ uint16_t sh_stage[24 * 16 * 16];
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & 63u;
    const __amdgpu_buffer_rsrc_t rsrc = coopstore_probe_make_rsrc(dst, extent, flags);

    if (tid < 64u) {
        coopstore_probe_init_lds_fragments(sh_frag, lane);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        const __attribute__((address_space(3))) uint64_t * lds =
            (const __attribute__((address_space(3))) uint64_t *) sh_frag;
        coopstore_probe_half8_vec acc[16];
        coopstore_probe_zero_acc(acc);
        coopstore_probe_accumulate_wmma_from_lds(lds, lane, acc);
        coopstore_probe_accumulate_wmma_from_lds(lds, lane, acc);
        HRX_COOPSTORE_ACC_GROUPS_0_15();
        HRX_COOPSTORE_STAGE_GROUPS_16_31(HRX_COOPSTORE_STAGE_STORE_GROUP);
        HRX_COOPSTORE_STAGE_GROUPS_32_39(HRX_COOPSTORE_STAGE_STORE_GROUP);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        HRX_COOPSTORE_STAGE_GROUPS_16_31(HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP);
        HRX_COOPSTORE_STAGE_GROUPS_32_39(HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();
}

extern "C" __global__ __launch_bounds__(256, 1)
void coopstore_probe_wmma_lds_k2_mixed160_hi_tight(float * dst, unsigned long long extent, unsigned int flags) {
    __shared__ uint64_t sh_frag[8 * 64 * 4];
    __shared__ uint16_t sh_stage[24 * 16 * 16];
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & 63u;
    const __amdgpu_buffer_rsrc_t rsrc = coopstore_probe_make_rsrc(dst, extent, flags);

    if (tid < 64u) {
        coopstore_probe_init_lds_fragments(sh_frag, lane);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        const __attribute__((address_space(3))) uint64_t * lds =
            (const __attribute__((address_space(3))) uint64_t *) sh_frag;
        coopstore_probe_half8_vec acc[16];
        coopstore_probe_zero_acc(acc);
        coopstore_probe_accumulate_wmma_from_lds(lds, lane, acc);
        coopstore_probe_accumulate_wmma_from_lds(lds, lane, acc);
        HRX_COOPSTORE_ACC_GROUPS_0_15();
        HRX_COOPSTORE_STAGE_GROUPS_16_31(HRX_COOPSTORE_STAGE_STORE_GROUP);
        HRX_COOPSTORE_STAGE_GROUPS_40_47(HRX_COOPSTORE_STAGE_STORE_GROUP_BASE24);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        HRX_COOPSTORE_STAGE_GROUPS_16_31(HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP);
        HRX_COOPSTORE_STAGE_GROUPS_40_47(HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_BASE24);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();
}

extern "C" __global__ __launch_bounds__(256, 1)
void coopstore_probe_wmma_lds_k2_stage96_accsink(float * dst, unsigned long long extent, unsigned int flags) {
    __shared__ uint64_t sh_frag[8 * 64 * 4];
    __shared__ uint16_t sh_stage[24 * 16 * 16];
    __shared__ uint16_t sh_sink[4 * 64];
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & 63u;
    const __amdgpu_buffer_rsrc_t rsrc = coopstore_probe_make_rsrc(dst, extent, flags);

    if (tid < 64u) {
        coopstore_probe_init_lds_fragments(sh_frag, lane);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        const __attribute__((address_space(3))) uint64_t * lds =
            (const __attribute__((address_space(3))) uint64_t *) sh_frag;
        coopstore_probe_half8_vec acc[16];
        coopstore_probe_zero_acc(acc);
        coopstore_probe_accumulate_wmma_from_lds(lds, lane, acc);
        coopstore_probe_accumulate_wmma_from_lds(lds, lane, acc);
        __attribute__((address_space(3))) uint16_t * sink =
            (__attribute__((address_space(3))) uint16_t *) sh_sink;
        coopstore_probe_ds_store_u16(
            sink + lane + 0u * HRX_COOPSTORE_LANES,
            static_cast<uint32_t>(__builtin_bit_cast(uint16_t, acc[0][0])));
        coopstore_probe_ds_store_u16(
            sink + lane + 1u * HRX_COOPSTORE_LANES,
            static_cast<uint32_t>(__builtin_bit_cast(uint16_t, acc[0][2])));
        coopstore_probe_ds_store_u16(
            sink + lane + 2u * HRX_COOPSTORE_LANES,
            static_cast<uint32_t>(__builtin_bit_cast(uint16_t, acc[0][4])));
        coopstore_probe_ds_store_u16(
            sink + lane + 3u * HRX_COOPSTORE_LANES,
            static_cast<uint32_t>(__builtin_bit_cast(uint16_t, acc[0][6])));
        HRX_COOPSTORE_STAGE_GROUPS_16_31(HRX_COOPSTORE_STAGE_STORE_GROUP);
        HRX_COOPSTORE_STAGE_GROUPS_32_39(HRX_COOPSTORE_STAGE_STORE_GROUP);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        HRX_COOPSTORE_STAGE_GROUPS_16_31(HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP);
        HRX_COOPSTORE_STAGE_GROUPS_32_39(HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();
}

extern "C" __global__ __launch_bounds__(256, 1)
void coopstore_probe_wmma_lds_k2_mixed160_linearstage(float * dst, unsigned long long extent, unsigned int flags) {
    __shared__ uint64_t sh_frag[8 * 64 * 4];
    __shared__ uint16_t sh_stage[24 * 64 * 4];
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & 63u;
    const __amdgpu_buffer_rsrc_t rsrc = coopstore_probe_make_rsrc(dst, extent, flags);

    if (tid < 64u) {
        coopstore_probe_init_lds_fragments(sh_frag, lane);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        const __attribute__((address_space(3))) uint64_t * lds =
            (const __attribute__((address_space(3))) uint64_t *) sh_frag;
        coopstore_probe_half8_vec acc[16];
        coopstore_probe_zero_acc(acc);
        coopstore_probe_accumulate_wmma_from_lds(lds, lane, acc);
        coopstore_probe_accumulate_wmma_from_lds(lds, lane, acc);
        HRX_COOPSTORE_ACC_GROUPS_0_15();
        HRX_COOPSTORE_STAGE_GROUPS_16_31(HRX_COOPSTORE_STAGE_STORE_GROUP_LINEAR);
        HRX_COOPSTORE_STAGE_GROUPS_32_39(HRX_COOPSTORE_STAGE_STORE_GROUP_LINEAR);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        HRX_COOPSTORE_STAGE_GROUPS_16_31(HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_LINEAR);
        HRX_COOPSTORE_STAGE_GROUPS_32_39(HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_LINEAR);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();
}

extern "C" __global__ __launch_bounds__(256, 1)
void coopstore_probe_wmma_lds_k2_mixed160_splitstage(float * dst, unsigned long long extent, unsigned int flags) {
    __shared__ uint64_t sh_frag[8 * 64 * 4];
    __shared__ uint16_t sh_stage[16 * 16 * 16];
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & 63u;
    const __amdgpu_buffer_rsrc_t rsrc = coopstore_probe_make_rsrc(dst, extent, flags);

    if (tid < 64u) {
        coopstore_probe_init_lds_fragments(sh_frag, lane);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        const __attribute__((address_space(3))) uint64_t * lds =
            (const __attribute__((address_space(3))) uint64_t *) sh_frag;
        coopstore_probe_half8_vec acc[16];
        coopstore_probe_zero_acc(acc);
        coopstore_probe_accumulate_wmma_from_lds(lds, lane, acc);
        coopstore_probe_accumulate_wmma_from_lds(lds, lane, acc);
        HRX_COOPSTORE_ACC_GROUPS_0_15();
        HRX_COOPSTORE_STAGE_GROUPS_16_31(HRX_COOPSTORE_STAGE_STORE_GROUP);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        HRX_COOPSTORE_STAGE_GROUPS_16_31(HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        HRX_COOPSTORE_STAGE_GROUPS_32_39(HRX_COOPSTORE_STAGE_STORE_GROUP_BASE32);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        HRX_COOPSTORE_STAGE_GROUPS_32_39(HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_BASE32);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();
}

extern "C" __global__ __launch_bounds__(256, 1)
void coopstore_probe_wmma_lds_k2_mixed192_splitstage(float * dst, unsigned long long extent, unsigned int flags) {
    __shared__ uint64_t sh_frag[8 * 64 * 4];
    __shared__ uint16_t sh_stage[16 * 16 * 16];
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & 63u;
    const __amdgpu_buffer_rsrc_t rsrc = coopstore_probe_make_rsrc(dst, extent, flags);

    if (tid < 64u) {
        coopstore_probe_init_lds_fragments(sh_frag, lane);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        const __attribute__((address_space(3))) uint64_t * lds =
            (const __attribute__((address_space(3))) uint64_t *) sh_frag;
        coopstore_probe_half8_vec acc[16];
        coopstore_probe_zero_acc(acc);
        coopstore_probe_accumulate_wmma_from_lds(lds, lane, acc);
        coopstore_probe_accumulate_wmma_from_lds(lds, lane, acc);
        HRX_COOPSTORE_ACC_GROUPS_0_15();
        HRX_COOPSTORE_STAGE_GROUPS_16_31(HRX_COOPSTORE_STAGE_STORE_GROUP);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        HRX_COOPSTORE_STAGE_GROUPS_16_31(HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        HRX_COOPSTORE_STAGE_GROUPS_32_47(HRX_COOPSTORE_STAGE_STORE_GROUP_BASE32);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        HRX_COOPSTORE_STAGE_GROUPS_32_47(HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_BASE32);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();
}

extern "C" __global__ __launch_bounds__(256, 1)
void coopstore_probe_wmma_lds_k2_mixed192_splitstage8(float * dst, unsigned long long extent, unsigned int flags) {
    __shared__ uint64_t sh_frag[8 * 64 * 4];
    __shared__ uint16_t sh_stage[16 * 16 * 16];
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & 63u;
    const __amdgpu_buffer_rsrc_t rsrc = coopstore_probe_make_rsrc(dst, extent, flags);

    if (tid < 64u) {
        coopstore_probe_init_lds_fragments(sh_frag, lane);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        const __attribute__((address_space(3))) uint64_t * lds =
            (const __attribute__((address_space(3))) uint64_t *) sh_frag;
        coopstore_probe_half8_vec acc[16];
        coopstore_probe_zero_acc(acc);
        coopstore_probe_accumulate_wmma_from_lds(lds, lane, acc);
        coopstore_probe_accumulate_wmma_from_lds(lds, lane, acc);
        HRX_COOPSTORE_ACC_GROUPS_0_15();
        HRX_COOPSTORE_STAGE_GROUPS_16_31(HRX_COOPSTORE_STAGE_STORE_GROUP);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        HRX_COOPSTORE_STAGE_GROUPS_16_31(HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        HRX_COOPSTORE_STAGE_GROUPS_32_39(HRX_COOPSTORE_STAGE_STORE_GROUP_BASE32);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        HRX_COOPSTORE_STAGE_GROUPS_32_39(HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_BASE32);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        HRX_COOPSTORE_STAGE_GROUPS_40_47(HRX_COOPSTORE_STAGE_STORE_GROUP_BASE40);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        HRX_COOPSTORE_STAGE_GROUPS_40_47(HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_BASE40);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();
}

extern "C" __global__ __launch_bounds__(256, 1)
void coopstore_probe_wmma_lds_k2_radv_motif192(float * dst, unsigned long long extent, unsigned int flags) {
    __shared__ uint64_t sh_frag[8 * 64 * 4];
    __shared__ uint16_t sh_stage[16 * 16 * 16];
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & 63u;
    const __amdgpu_buffer_rsrc_t rsrc = coopstore_probe_make_rsrc(dst, extent, flags);

    if (tid < 64u) {
        coopstore_probe_init_lds_fragments(sh_frag, lane);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        const __attribute__((address_space(3))) uint64_t * lds =
            (const __attribute__((address_space(3))) uint64_t *) sh_frag;
        coopstore_probe_half8_vec acc[16];
        coopstore_probe_zero_acc(acc);
        coopstore_probe_accumulate_wmma_from_lds(lds, lane, acc);
        coopstore_probe_accumulate_wmma_from_lds(lds, lane, acc);
        HRX_COOPSTORE_RADV_MOTIF_GROUPS_0_15();
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();
}

extern "C" __global__ __launch_bounds__(256, 1)
void coopstore_probe_wmma_lds_k2_radv_motif192_typedstage(float * dst, unsigned long long extent, unsigned int flags) {
    __shared__ uint64_t sh_frag[8 * 64 * 4];
    __shared__ uint16_t sh_stage[16 * 16 * 16];
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & 63u;
    const __amdgpu_buffer_rsrc_t rsrc = coopstore_probe_make_rsrc(dst, extent, flags);

    if (tid < 64u) {
        coopstore_probe_init_lds_fragments(sh_frag, lane);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        const __attribute__((address_space(3))) uint64_t * lds =
            (const __attribute__((address_space(3))) uint64_t *) sh_frag;
        coopstore_probe_half8_vec acc[16];
        coopstore_probe_zero_acc(acc);
        coopstore_probe_accumulate_wmma_from_lds(lds, lane, acc);
        coopstore_probe_accumulate_wmma_from_lds(lds, lane, acc);
        HRX_COOPSTORE_RADV_MOTIF_TYPED_GROUPS_0_15();
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();
}

template <bool direct_acc_groups>
__global__ __launch_bounds__(256, 1)
void coopstore_probe_wmma_lds_k2_stage_one18(float * dst, unsigned long long extent, unsigned int flags) {
    __shared__ uint64_t sh_frag[8 * 64 * 4];
    __shared__ uint16_t sh_stage[16 * 16 * 16];
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & 63u;
    const __amdgpu_buffer_rsrc_t rsrc = coopstore_probe_make_rsrc(dst, extent, flags);

    if (tid < 64u) {
        coopstore_probe_init_lds_fragments(sh_frag, lane);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        const __attribute__((address_space(3))) uint64_t * lds =
            (const __attribute__((address_space(3))) uint64_t *) sh_frag;
        coopstore_probe_half8_vec acc[16];
        coopstore_probe_zero_acc(acc);
        coopstore_probe_accumulate_wmma_from_lds(lds, lane, acc);
        coopstore_probe_accumulate_wmma_from_lds(lds, lane, acc);
        if constexpr (direct_acc_groups) {
            HRX_COOPSTORE_ACC_GROUPS_0_15();
        }
        HRX_COOPSTORE_STAGE_STORE_GROUP(18u);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP(18u);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();
}

extern "C" __global__ __launch_bounds__(256, 1)
void coopstore_probe_wmma_lds_k2_direct160_raw(float * dst, unsigned long long extent, unsigned int flags) {
    __shared__ uint64_t sh_frag[8 * 64 * 4];
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & 63u;
    const __amdgpu_buffer_rsrc_t rsrc = coopstore_probe_make_rsrc(dst, extent, flags);

    if (tid < 64u) {
        coopstore_probe_init_lds_fragments(sh_frag, lane);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        const __attribute__((address_space(3))) uint64_t * lds =
            (const __attribute__((address_space(3))) uint64_t *) sh_frag;
        coopstore_probe_half8_vec acc[16];
        coopstore_probe_zero_acc(acc);
        coopstore_probe_accumulate_wmma_from_lds(lds, lane, acc);
        coopstore_probe_accumulate_wmma_from_lds(lds, lane, acc);
        HRX_COOPSTORE_ACC_GROUPS_0_15();
        HRX_COOPSTORE_GROUPS_16_31();
        HRX_COOPSTORE_STORE_GROUP(32u); HRX_COOPSTORE_STORE_GROUP(33u);
        HRX_COOPSTORE_STORE_GROUP(34u); HRX_COOPSTORE_STORE_GROUP(35u);
        HRX_COOPSTORE_STORE_GROUP(36u); HRX_COOPSTORE_STORE_GROUP(37u);
        HRX_COOPSTORE_STORE_GROUP(38u); HRX_COOPSTORE_STORE_GROUP(39u);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();
}

extern "C" __global__ __launch_bounds__(256, 1)
void coopstore_probe_wmma_lds_k2_direct192_raw(float * dst, unsigned long long extent, unsigned int flags) {
    __shared__ uint64_t sh_frag[8 * 64 * 4];
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & 63u;
    const __amdgpu_buffer_rsrc_t rsrc = coopstore_probe_make_rsrc(dst, extent, flags);

    if (tid < 64u) {
        coopstore_probe_init_lds_fragments(sh_frag, lane);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        const __attribute__((address_space(3))) uint64_t * lds =
            (const __attribute__((address_space(3))) uint64_t *) sh_frag;
        coopstore_probe_half8_vec acc[16];
        coopstore_probe_zero_acc(acc);
        coopstore_probe_accumulate_wmma_from_lds(lds, lane, acc);
        coopstore_probe_accumulate_wmma_from_lds(lds, lane, acc);
        HRX_COOPSTORE_ACC_GROUPS_0_15();
        HRX_COOPSTORE_GROUPS_16_31();
        HRX_COOPSTORE_GROUPS_32_47();
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();
}

extern "C" __global__ __launch_bounds__(256, 1)
void coopstore_probe_wmma_lds_k2_accslots_raw192(float * dst, unsigned long long extent, unsigned int flags) {
    __shared__ uint64_t sh_frag[8 * 64 * 4];
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & 63u;
    const __amdgpu_buffer_rsrc_t rsrc = coopstore_probe_make_rsrc(dst, extent, flags);

    if (tid < 64u) {
        coopstore_probe_init_lds_fragments(sh_frag, lane);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        const __attribute__((address_space(3))) uint64_t * lds =
            (const __attribute__((address_space(3))) uint64_t *) sh_frag;
        coopstore_probe_half8_vec acc[8];
        coopstore_probe_zero_acc(acc);
        coopstore_probe_accumulate_wmma_from_lds(lds, lane, acc);
        coopstore_probe_accumulate_wmma_from_lds(lds, lane, acc);
        HRX_COOPSTORE_ACC_SLOTS_GROUPS_0_7();
        HRX_COOPSTORE_GROUPS_16_31();
        HRX_COOPSTORE_GROUPS_32_47();
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();
}

extern "C" __global__ __launch_bounds__(256, 1)
void coopstore_probe_wmma_lds_k2_direct64(float * dst, unsigned long long extent, unsigned int flags) {
    __shared__ uint64_t sh_frag[8 * 64 * 4];
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & 63u;
    const __amdgpu_buffer_rsrc_t rsrc = coopstore_probe_make_rsrc(dst, extent, flags);

    if (tid < 64u) {
        coopstore_probe_init_lds_fragments(sh_frag, lane);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        const __attribute__((address_space(3))) uint64_t * lds =
            (const __attribute__((address_space(3))) uint64_t *) sh_frag;
        coopstore_probe_half8_vec acc[16];
        coopstore_probe_zero_acc(acc);
        coopstore_probe_accumulate_wmma_from_lds(lds, lane, acc);
        coopstore_probe_accumulate_wmma_from_lds(lds, lane, acc);
        HRX_COOPSTORE_ACC_GROUPS_0_15();
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();
}

extern "C" __global__ __launch_bounds__(256, 1)
void coopstore_probe_wmma_lds_k2_radv_branch64(
        float * dst,
        unsigned long long extent,
        unsigned int flags,
        unsigned int selected_path) {
    __shared__ uint64_t sh_frag[8 * 64 * 4];
    __shared__ uint16_t sh_stage[16 * 16 * 16];
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & 63u;
    const __amdgpu_buffer_rsrc_t rsrc = coopstore_probe_make_rsrc(dst, extent, flags);

    if (tid < 64u) {
        coopstore_probe_init_lds_fragments(sh_frag, lane);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        const __attribute__((address_space(3))) uint64_t * lds =
            (const __attribute__((address_space(3))) uint64_t *) sh_frag;
        coopstore_probe_half8_vec acc[16];
        coopstore_probe_zero_acc(acc);
        coopstore_probe_accumulate_wmma_from_lds(lds, lane, acc);
        coopstore_probe_accumulate_wmma_from_lds(lds, lane, acc);

        if (selected_path == 0u) {
            HRX_COOPSTORE_ACC_GROUPS_0_15();
        } else if (selected_path == 1u) {
            HRX_COOPSTORE_STAGE_GROUPS_16_31(HRX_COOPSTORE_STAGE_STORE_GROUP);
            asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
            HRX_COOPSTORE_STAGE_GROUPS_16_31(HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP);
        } else if (selected_path == 2u) {
            HRX_COOPSTORE_STAGE_GROUPS_32_47(HRX_COOPSTORE_STAGE_STORE_GROUP_BASE32);
            asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
            HRX_COOPSTORE_STAGE_GROUPS_32_47(HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_BASE32);
        }
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    __syncthreads();
}

#undef HRX_COOPSTORE_ACC_GROUPS_0_15
#undef HRX_COOPSTORE_ACC_GROUPS_8_15
#undef HRX_COOPSTORE_ACC_GROUPS_0_7
#undef HRX_COOPSTORE_ACC_SLOTS_GROUPS_0_7
#undef HRX_COOPSTORE_STORE_ACC_SLOTS_GROUP
#undef HRX_COOPSTORE_STORE_ACC_ELEMENT
#undef HRX_COOPSTORE_RADV_MOTIF_TYPED_GROUPS_0_15
#undef HRX_COOPSTORE_RADV_MOTIF_TYPED_GROUP
#undef HRX_COOPSTORE_RADV_MOTIF_GROUPS_0_15
#undef HRX_COOPSTORE_RADV_MOTIF_GROUP
#undef HRX_COOPSTORE_STORE_ACC_GROUP

#undef HRX_COOPSTORE_STAGE_GROUPS_32_47
#undef HRX_COOPSTORE_STAGE_GROUPS_40_47
#undef HRX_COOPSTORE_STAGE_GROUPS_32_39
#undef HRX_COOPSTORE_STAGE_GROUPS_16_31
#undef HRX_COOPSTORE_STAGE_GROUPS_16_23
#undef HRX_COOPSTORE_STAGE_GROUPS_8_15
#undef HRX_COOPSTORE_STAGE_GROUPS_8_23
#undef HRX_COOPSTORE_STAGE_LOAD_STORE_ACC_GROUP_BASE8
#undef HRX_COOPSTORE_STAGE_STORE_ACC_GROUP_BASE8
#undef HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_BASE8
#undef HRX_COOPSTORE_STAGE_STORE_GROUP_BASE8
#undef HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_BASE24
#undef HRX_COOPSTORE_STAGE_STORE_GROUP_BASE24
#undef HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_BASE32
#undef HRX_COOPSTORE_STAGE_STORE_GROUP_BASE32
#undef HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_BASE40
#undef HRX_COOPSTORE_STAGE_STORE_GROUP_BASE40
#undef HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_LINEAR
#undef HRX_COOPSTORE_STAGE_STORE_GROUP_LINEAR
#undef HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_TYPED_LINEAR_BASE32
#undef HRX_COOPSTORE_STAGE_STORE_GROUP_TYPED_LINEAR_BASE32
#undef HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_TYPED_LINEAR
#undef HRX_COOPSTORE_STAGE_STORE_GROUP_TYPED_LINEAR
#undef HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_TYPED_LINEAR_BASE
#undef HRX_COOPSTORE_STAGE_STORE_GROUP_TYPED_LINEAR_BASE
#undef HRX_COOPSTORE_STAGE_LOAD_STORE_TYPED_LINEAR_BASE
#undef HRX_COOPSTORE_STAGE_STORE_TYPED_LINEAR_BASE
#undef HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_LINEAR_BASE
#undef HRX_COOPSTORE_STAGE_STORE_GROUP_LINEAR_BASE
#undef HRX_COOPSTORE_STAGE_LOAD_STORE_LINEAR_BASE
#undef HRX_COOPSTORE_STAGE_STORE_LINEAR_BASE
#undef HRX_COOPSTORE_STAGE_LOAD_STORE_ACC_GROUP_BASE
#undef HRX_COOPSTORE_STAGE_STORE_ACC_GROUP_BASE
#undef HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP_BASE
#undef HRX_COOPSTORE_STAGE_STORE_GROUP_BASE
#undef HRX_COOPSTORE_STAGE_LOAD_STORE_ACC_BASE
#undef HRX_COOPSTORE_STAGE_STORE_ACC_BASE
#undef HRX_COOPSTORE_STAGE_LOAD_STORE_BASE
#undef HRX_COOPSTORE_STAGE_STORE_BASE
#undef HRX_COOPSTORE_STAGE_LOAD_STORE_GROUP
#undef HRX_COOPSTORE_STAGE_STORE_GROUP
#undef HRX_COOPSTORE_STAGE_LOAD_STORE
#undef HRX_COOPSTORE_STAGE_STORE

#undef HRX_COOPSTORE_GROUPS_32_47
#undef HRX_COOPSTORE_GROUPS_16_31
#undef HRX_COOPSTORE_GROUPS_8_15
#undef HRX_COOPSTORE_GROUPS_0_7
#undef HRX_COOPSTORE_GROUPS_0_15
#undef HRX_COOPSTORE_STORE_GROUP

struct options {
    std::string mode = "branch192";
    unsigned int group = 17;
    unsigned int flags = 0x31004000u;
    unsigned int timing_iters = 10000;
    unsigned int rows = 64;
    unsigned int cols = 96;
};

static unsigned int parse_u32(const char * value) {
    char * end = nullptr;
    const unsigned long parsed = std::strtoul(value, &end, 0);
    if (end == value || *end != '\0' || parsed > 0xfffffffful) {
        std::fprintf(stderr, "invalid u32: %s\n", value);
        std::exit(2);
    }
    return static_cast<unsigned int>(parsed);
}

static options parse_options(int argc, char ** argv) {
    options opts;
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], "--mode=", 7) == 0) {
            opts.mode = argv[i] + 7;
        } else if (std::strncmp(argv[i], "--group=", 8) == 0) {
            opts.group = std::min(HRX_COOPSTORE_MAX_GROUPS - 1u, parse_u32(argv[i] + 8));
        } else if (std::strncmp(argv[i], "--flags=", 8) == 0) {
            opts.flags = parse_u32(argv[i] + 8);
        } else if (std::strncmp(argv[i], "--timing-iters=", 15) == 0) {
            opts.timing_iters = std::max(1u, parse_u32(argv[i] + 15));
        } else if (std::strncmp(argv[i], "--rows=", 7) == 0) {
            opts.rows = parse_u32(argv[i] + 7);
        } else if (std::strncmp(argv[i], "--cols=", 7) == 0) {
            opts.cols = parse_u32(argv[i] + 7);
        } else {
            std::fprintf(stderr,
                "usage: %s [--mode=linear64|linear128|linear192|branch192|radv-mixed96|radv-mixed192|wmma-radv-mixed96|wmma-radv-mixed192|wmma-lds-radv-mixed96|wmma-lds-radv-mixed192|wmma-lds-k2-radv-mixed192|wmma-lds-k2-radv-motif192|wmma-lds-k2-radv-motif192-typedstage|wmma-lds-k2-stage-one18|wmma-lds-k2-stage-one18-nodirect|wmma-lds-k2-stagefirst-mixed192|wmma-lds-k2-mixed96|wmma-lds-k2-typedstage-mixed96|wmma-lds-k2-kloop-mixed96|wmma-lds-k2-kloop-direct64|wmma-lds-k2-kloop-asm-mixed96|wmma-lds-k2-kloop-asm-stagefirst-mixed96|wmma-lds-k2-kloop-asm-linearstage-mixed96|wmma-lds-k2-kloop-asm-typedstage-mixed96|wmma-lds-k2-kloop-asm-direct64|wmma-lds-vk64-radv96|wmma-lds-vk64-radv96-accdirect|wmma-lds-vk64-radv96-accslots|wmma-lds-vk64-radv96-accdirect-copya|wmma-lds-vk64-radv96-accdirect-copyb|wmma-lds-vk64-radv96-accdirect-copyab|wmma-lds-vk64-radv96-accdirect-copyab-q6addr|wmma-lds-vk64-radv96-accdirect-copyab-q6addr-nosink|wmma-lds-vk64-radv96-accdirect-copyab-q6addr-upperreal|wmma-lds-vk64-radv96-accdirect-copyab-q6addr-upperwait|wmma-lds-vk64-radv96-accdirect-copyab-q6addr-upperwide|wmma-lds-k2-mixed128|wmma-lds-k2-mixed128-padded32|wmma-lds-k2-mixed160-lo|wmma-lds-k2-mixed160-hi|wmma-lds-k2-mixed160-lo-tight|wmma-lds-k2-mixed160-hi-tight|wmma-lds-k2-stage96-accsink|wmma-lds-k2-mixed160-linearstage|wmma-lds-k2-mixed160-splitstage|wmma-lds-k2-mixed192-splitstage|wmma-lds-k2-mixed192-splitstage8|wmma-lds-k2-direct160-raw|wmma-lds-k2-direct192-raw|wmma-lds-k2-accslots-raw192|wmma-lds-k2-direct64|wmma-lds-k2-radv-branch64|timing] [--group=N] [--flags=0x31004000] [--timing-iters=N] [--rows=N] [--cols=N]\n",
                argv[0]);
            std::exit(2);
        }
    }
    return opts;
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

static void compare_outputs(
        const std::vector<float> & actual,
        const std::vector<float> & expected,
        const char * label) {
    double max_abs = 0.0;
    size_t bad_count = 0;
    size_t first_bad = 0;
    size_t nan_count = 0;
    size_t inf_count = 0;
    size_t bad_groups[HRX_COOPSTORE_MAX_GROUPS] = {};
    size_t bad_group_slots[HRX_COOPSTORE_MAX_GROUPS][HRX_COOPSTORE_VALUES_PER_GROUP] = {};
    for (size_t i = 0; i < actual.size(); ++i) {
        const double diff = std::abs(static_cast<double>(actual[i]) - static_cast<double>(expected[i]));
        if (std::isnan(static_cast<double>(actual[i]))) {
            ++nan_count;
        } else if (std::isinf(static_cast<double>(actual[i]))) {
            ++inf_count;
        }
        if (diff > max_abs) {
            max_abs = diff;
        }
        if (diff != 0.0) {
            if (bad_count == 0) {
                first_bad = i;
            }
            const unsigned int bad_group =
                static_cast<unsigned int>(i / (HRX_COOPSTORE_LANES * HRX_COOPSTORE_VALUES_PER_GROUP));
            const unsigned int bad_slot =
                static_cast<unsigned int>((i / HRX_COOPSTORE_LANES) % HRX_COOPSTORE_VALUES_PER_GROUP);
            if (bad_group < HRX_COOPSTORE_MAX_GROUPS) {
                ++bad_groups[bad_group];
                if (bad_slot < HRX_COOPSTORE_VALUES_PER_GROUP) {
                    ++bad_group_slots[bad_group][bad_slot];
                }
            }
            ++bad_count;
        }
    }

    std::printf("%s: elements=%zu bad=%zu max_abs=%g", label, actual.size(), bad_count, max_abs);
    if (bad_count != 0) {
        const unsigned int first_lane = first_bad % HRX_COOPSTORE_LANES;
        const unsigned int first_slot = (first_bad / HRX_COOPSTORE_LANES) % HRX_COOPSTORE_VALUES_PER_GROUP;
        const unsigned int first_group =
            first_bad / (HRX_COOPSTORE_LANES * HRX_COOPSTORE_VALUES_PER_GROUP);
        std::printf(" first_bad=%zu group=%u slot=%u lane=%u actual=%g expected=%g",
            first_bad, first_group, first_slot, first_lane, actual[first_bad], expected[first_bad]);
        if (nan_count != 0 || inf_count != 0) {
            std::printf(" nan=%zu inf=%zu", nan_count, inf_count);
        }
        std::printf(" bad_groups=");
        bool first_group_printed = true;
        for (unsigned int group = 0; group < HRX_COOPSTORE_MAX_GROUPS; ++group) {
            if (bad_groups[group] == 0) {
                continue;
            }
            if (!first_group_printed) {
                std::printf(",");
            }
            std::printf("%u:%zu", group, bad_groups[group]);
            first_group_printed = false;
        }
        std::printf(" bad_slots=");
        bool first_slot_printed = true;
        for (unsigned int group = 0; group < HRX_COOPSTORE_MAX_GROUPS; ++group) {
            for (unsigned int slot = 0; slot < HRX_COOPSTORE_VALUES_PER_GROUP; ++slot) {
                if (bad_group_slots[group][slot] == 0) {
                    continue;
                }
                if (!first_slot_printed) {
                    std::printf(",");
                }
                std::printf("%u/%u:%zu", group, slot, bad_group_slots[group][slot]);
                first_slot_printed = false;
            }
        }
        std::printf(" samples=");
        size_t printed = 0;
        for (size_t i = 0; i < actual.size() && printed < 12; ++i) {
            const double diff = std::abs(static_cast<double>(actual[i]) - static_cast<double>(expected[i]));
            if (diff == 0.0) {
                continue;
            }
            const unsigned int lane = i % HRX_COOPSTORE_LANES;
            const unsigned int slot = (i / HRX_COOPSTORE_LANES) % HRX_COOPSTORE_VALUES_PER_GROUP;
            const unsigned int group = i / (HRX_COOPSTORE_LANES * HRX_COOPSTORE_VALUES_PER_GROUP);
            if (printed != 0) {
                std::printf(",");
            }
            std::printf("%u/%u/%u:%g!=%g", group, slot, lane, actual[i], expected[i]);
            ++printed;
        }
    }
    std::printf("\n");

    if (bad_count != 0) {
        std::exit(1);
    }
}

static float coopstore_probe_wmma_expected_value(unsigned int group) {
    const unsigned int row_frag = group & 3u;
    const unsigned int col_frag = (group >> 2u) & 3u;
    return static_cast<float>(16u * (1u + row_frag) * (1u + col_frag));
}

static float coopstore_probe_wmma_ring12_expected_value(unsigned int group) {
    const unsigned int row_frag = group % 6u;
    const unsigned int col_frag = (group + group / 6u) % 6u;
    return static_cast<float>(16u * (1u + (row_frag & 3u)) * (1u + ((col_frag + 6u) & 3u)));
}

static void run_timing_suite(
        float * d_out,
        size_t count,
        unsigned long long byte_extent,
        unsigned int flags,
        unsigned int timing_iters) {
    constexpr unsigned int warmup_iters = 128;

    std::printf("coopmat-store-timing-suite iters=%u warmup=%u flags=0x%x bytes=%llu\n",
        timing_iters, warmup_iters, flags, byte_extent);

#define HRX_COOPSTORE_TIME_CASE(MODE_NAME, KERNEL_NAME) do { \
        HIP_CHECK(hipMemset(d_out, 0, count * sizeof(float))); \
        for (unsigned int iter = 0; iter < warmup_iters; ++iter) { \
            hipLaunchKernelGGL(KERNEL_NAME, dim3(1), dim3(256), 0, 0, d_out, byte_extent, flags); \
        } \
        HIP_CHECK(hipGetLastError()); \
        HIP_CHECK(hipDeviceSynchronize()); \
        hipEvent_t start_event = nullptr; \
        hipEvent_t stop_event = nullptr; \
        HIP_CHECK(hipEventCreate(&start_event)); \
        HIP_CHECK(hipEventCreate(&stop_event)); \
        const auto host_start = std::chrono::steady_clock::now(); \
        HIP_CHECK(hipEventRecord(start_event, 0)); \
        for (unsigned int iter = 0; iter < timing_iters; ++iter) { \
            hipLaunchKernelGGL(KERNEL_NAME, dim3(1), dim3(256), 0, 0, d_out, byte_extent, flags); \
        } \
        HIP_CHECK(hipGetLastError()); \
        HIP_CHECK(hipEventRecord(stop_event, 0)); \
        HIP_CHECK(hipEventSynchronize(stop_event)); \
        const auto host_stop = std::chrono::steady_clock::now(); \
        float event_ms = 0.0f; \
        HIP_CHECK(hipEventElapsedTime(&event_ms, start_event, stop_event)); \
        HIP_CHECK(hipEventDestroy(start_event)); \
        HIP_CHECK(hipEventDestroy(stop_event)); \
        const double host_us = std::chrono::duration<double, std::micro>(host_stop - host_start).count(); \
        std::printf("coopmat-store-timing mode=%s iters=%u event_total_us=%.3f event_avg_us=%.6f host_total_us=%.3f host_avg_us=%.6f\n", \
            (MODE_NAME), timing_iters, \
            static_cast<double>(event_ms) * 1000.0, \
            static_cast<double>(event_ms) * 1000.0 / static_cast<double>(timing_iters), \
            host_us, host_us / static_cast<double>(timing_iters)); \
    } while (0)

    HRX_COOPSTORE_TIME_CASE("wmma-lds-k2-direct64", coopstore_probe_wmma_lds_k2_direct64);
    HRX_COOPSTORE_TIME_CASE("wmma-lds-k2-direct160-raw", coopstore_probe_wmma_lds_k2_direct160_raw);
    HRX_COOPSTORE_TIME_CASE("wmma-lds-k2-direct192-raw", coopstore_probe_wmma_lds_k2_direct192_raw);
    HRX_COOPSTORE_TIME_CASE("wmma-lds-k2-accslots-raw192", coopstore_probe_wmma_lds_k2_accslots_raw192);
    HRX_COOPSTORE_TIME_CASE("wmma-lds-k2-radv-motif192", coopstore_probe_wmma_lds_k2_radv_motif192);
    HRX_COOPSTORE_TIME_CASE("radv-mixed192", coopstore_probe_radv_mixed192);

#undef HRX_COOPSTORE_TIME_CASE
}

int main(int argc, char ** argv) {
    const options opts = parse_options(argc, argv);
    const size_t count = HRX_COOPSTORE_MAX_VALUES;
    const unsigned long long byte_extent = static_cast<unsigned long long>(count * sizeof(float));
    if (opts.rows == 0 || opts.cols == 0 ||
            static_cast<unsigned long long>(opts.rows) * static_cast<unsigned long long>(opts.cols) >
                static_cast<unsigned long long>(count)) {
        std::fprintf(stderr, "invalid rows/cols for fixed output buffer: rows=%u cols=%u max=%zu\n",
            opts.rows, opts.cols, count);
        return 2;
    }

    device_buffer<float> d_out(count);
    std::vector<float> h_out(count, 0.0f);
    std::vector<float> h_expected(count, 0.0f);
    HIP_CHECK(hipMemset(d_out.ptr, 0, count * sizeof(float)));

    if (opts.mode == "timing") {
        run_timing_suite(d_out.ptr, count, byte_extent, opts.flags, opts.timing_iters);
        return 0;
    } else if (opts.mode == "linear64") {
        hipLaunchKernelGGL(coopstore_probe_linear64, dim3(1), dim3(64), 0, 0, d_out.ptr, byte_extent, opts.flags);
        for (unsigned int group = 0; group < 16; ++group) {
            for (unsigned int slot = 0; slot < HRX_COOPSTORE_VALUES_PER_GROUP; ++slot) {
                for (unsigned int lane = 0; lane < HRX_COOPSTORE_LANES; ++lane) {
                    h_expected[coopstore_probe_index(group, slot, lane)] = coopstore_probe_value(group, slot, lane);
                }
            }
        }
    } else if (opts.mode == "linear128") {
        hipLaunchKernelGGL(coopstore_probe_linear128, dim3(1), dim3(64), 0, 0, d_out.ptr, byte_extent, opts.flags);
        for (unsigned int group = 0; group < 32; ++group) {
            for (unsigned int slot = 0; slot < HRX_COOPSTORE_VALUES_PER_GROUP; ++slot) {
                for (unsigned int lane = 0; lane < HRX_COOPSTORE_LANES; ++lane) {
                    h_expected[coopstore_probe_index(group, slot, lane)] = coopstore_probe_value(group, slot, lane);
                }
            }
        }
    } else if (opts.mode == "linear192") {
        hipLaunchKernelGGL(coopstore_probe_linear192, dim3(1), dim3(64), 0, 0, d_out.ptr, byte_extent, opts.flags);
        for (unsigned int group = 0; group < HRX_COOPSTORE_MAX_GROUPS; ++group) {
            for (unsigned int slot = 0; slot < HRX_COOPSTORE_VALUES_PER_GROUP; ++slot) {
                for (unsigned int lane = 0; lane < HRX_COOPSTORE_LANES; ++lane) {
                    h_expected[coopstore_probe_index(group, slot, lane)] = coopstore_probe_value(group, slot, lane);
                }
            }
        }
    } else if (opts.mode == "branch192") {
        hipLaunchKernelGGL(coopstore_probe_branch192, dim3(1), dim3(64), 0, 0,
            d_out.ptr, byte_extent, opts.flags, opts.group);
        for (unsigned int slot = 0; slot < HRX_COOPSTORE_VALUES_PER_GROUP; ++slot) {
            for (unsigned int lane = 0; lane < HRX_COOPSTORE_LANES; ++lane) {
                h_expected[coopstore_probe_index(opts.group, slot, lane)] =
                    coopstore_probe_value(opts.group, slot, lane);
            }
        }
    } else if (opts.mode == "radv-mixed96") {
        hipLaunchKernelGGL(coopstore_probe_radv_mixed96, dim3(1), dim3(256), 0, 0,
            d_out.ptr, byte_extent, opts.flags);
        for (unsigned int group = 0; group < 24u; ++group) {
            for (unsigned int slot = 0; slot < HRX_COOPSTORE_VALUES_PER_GROUP; ++slot) {
                for (unsigned int lane = 0; lane < HRX_COOPSTORE_LANES; ++lane) {
                    h_expected[coopstore_probe_index(group, slot, lane)] =
                        group < 8u ? coopstore_probe_value(group, slot, lane) :
                                     static_cast<float>(coopstore_probe_stage_value(group, slot, lane));
                }
            }
        }
    } else if (opts.mode == "radv-mixed192") {
        hipLaunchKernelGGL(coopstore_probe_radv_mixed192, dim3(1), dim3(256), 0, 0,
            d_out.ptr, byte_extent, opts.flags);
        for (unsigned int group = 0; group < HRX_COOPSTORE_MAX_GROUPS; ++group) {
            for (unsigned int slot = 0; slot < HRX_COOPSTORE_VALUES_PER_GROUP; ++slot) {
                for (unsigned int lane = 0; lane < HRX_COOPSTORE_LANES; ++lane) {
                    h_expected[coopstore_probe_index(group, slot, lane)] =
                        coopstore_probe_expected_value(group, slot, lane);
                }
            }
        }
    } else if (opts.mode == "wmma-radv-mixed96") {
        hipLaunchKernelGGL(coopstore_probe_wmma_radv_mixed96, dim3(1), dim3(256), 0, 0,
            d_out.ptr, byte_extent, opts.flags);
        for (unsigned int group = 0; group < 24u; ++group) {
            for (unsigned int slot = 0; slot < HRX_COOPSTORE_VALUES_PER_GROUP; ++slot) {
                for (unsigned int lane = 0; lane < HRX_COOPSTORE_LANES; ++lane) {
                    h_expected[coopstore_probe_index(group, slot, lane)] =
                        group < 16u ? coopstore_probe_wmma_expected_value(group) :
                                      static_cast<float>(coopstore_probe_stage_value(group, slot, lane));
                }
            }
        }
    } else if (opts.mode == "wmma-radv-mixed192") {
        hipLaunchKernelGGL(coopstore_probe_wmma_radv_mixed192, dim3(1), dim3(256), 0, 0,
            d_out.ptr, byte_extent, opts.flags);
        for (unsigned int group = 0; group < HRX_COOPSTORE_MAX_GROUPS; ++group) {
            for (unsigned int slot = 0; slot < HRX_COOPSTORE_VALUES_PER_GROUP; ++slot) {
                for (unsigned int lane = 0; lane < HRX_COOPSTORE_LANES; ++lane) {
                    h_expected[coopstore_probe_index(group, slot, lane)] =
                        group < 16u ? coopstore_probe_wmma_expected_value(group) :
                                      static_cast<float>(coopstore_probe_stage_value(group, slot, lane));
                }
            }
        }
    } else if (opts.mode == "wmma-lds-radv-mixed96") {
        hipLaunchKernelGGL(coopstore_probe_wmma_lds_radv_mixed96, dim3(1), dim3(256), 0, 0,
            d_out.ptr, byte_extent, opts.flags);
        for (unsigned int group = 0; group < 24u; ++group) {
            for (unsigned int slot = 0; slot < HRX_COOPSTORE_VALUES_PER_GROUP; ++slot) {
                for (unsigned int lane = 0; lane < HRX_COOPSTORE_LANES; ++lane) {
                    h_expected[coopstore_probe_index(group, slot, lane)] =
                        group < 16u ? coopstore_probe_wmma_expected_value(group) :
                                      static_cast<float>(coopstore_probe_stage_value(group, slot, lane));
                }
            }
        }
    } else if (opts.mode == "wmma-lds-radv-mixed192") {
        hipLaunchKernelGGL(coopstore_probe_wmma_lds_radv_mixed192, dim3(1), dim3(256), 0, 0,
            d_out.ptr, byte_extent, opts.flags);
        for (unsigned int group = 0; group < HRX_COOPSTORE_MAX_GROUPS; ++group) {
            for (unsigned int slot = 0; slot < HRX_COOPSTORE_VALUES_PER_GROUP; ++slot) {
                for (unsigned int lane = 0; lane < HRX_COOPSTORE_LANES; ++lane) {
                    h_expected[coopstore_probe_index(group, slot, lane)] =
                        group < 16u ? coopstore_probe_wmma_expected_value(group) :
                                      static_cast<float>(coopstore_probe_stage_value(group, slot, lane));
                }
            }
        }
    } else if (opts.mode == "wmma-lds-k2-radv-mixed192") {
        hipLaunchKernelGGL(coopstore_probe_wmma_lds_k2_radv_mixed192, dim3(1), dim3(256), 0, 0,
            d_out.ptr, byte_extent, opts.flags);
        for (unsigned int group = 0; group < HRX_COOPSTORE_MAX_GROUPS; ++group) {
            for (unsigned int slot = 0; slot < HRX_COOPSTORE_VALUES_PER_GROUP; ++slot) {
                for (unsigned int lane = 0; lane < HRX_COOPSTORE_LANES; ++lane) {
                    h_expected[coopstore_probe_index(group, slot, lane)] =
                        group < 16u ? 2.0f * coopstore_probe_wmma_expected_value(group) :
                                      static_cast<float>(coopstore_probe_stage_value(group, slot, lane));
                }
            }
        }
    } else if (opts.mode == "wmma-lds-k2-radv-motif192") {
        hipLaunchKernelGGL(coopstore_probe_wmma_lds_k2_radv_motif192, dim3(1), dim3(256), 0, 0,
            d_out.ptr, byte_extent, opts.flags);
        for (unsigned int group = 0; group < HRX_COOPSTORE_MAX_GROUPS; ++group) {
            for (unsigned int slot = 0; slot < HRX_COOPSTORE_VALUES_PER_GROUP; ++slot) {
                for (unsigned int lane = 0; lane < HRX_COOPSTORE_LANES; ++lane) {
                    h_expected[coopstore_probe_index(group, slot, lane)] =
                        group < 16u ? 2.0f * coopstore_probe_wmma_expected_value(group) :
                                      static_cast<float>(coopstore_probe_stage_value(group, slot, lane));
                }
            }
        }
    } else if (opts.mode == "wmma-lds-k2-radv-motif192-typedstage") {
        hipLaunchKernelGGL(coopstore_probe_wmma_lds_k2_radv_motif192_typedstage, dim3(1), dim3(256), 0, 0,
            d_out.ptr, byte_extent, opts.flags);
        for (unsigned int group = 0; group < HRX_COOPSTORE_MAX_GROUPS; ++group) {
            for (unsigned int slot = 0; slot < HRX_COOPSTORE_VALUES_PER_GROUP; ++slot) {
                for (unsigned int lane = 0; lane < HRX_COOPSTORE_LANES; ++lane) {
                    h_expected[coopstore_probe_index(group, slot, lane)] =
                        group < 16u ? 2.0f * coopstore_probe_wmma_expected_value(group) :
                                      static_cast<float>(coopstore_probe_stage_value(group, slot, lane));
                }
            }
        }
    } else if (opts.mode == "wmma-lds-k2-stage-one18") {
        hipLaunchKernelGGL((coopstore_probe_wmma_lds_k2_stage_one18<true>), dim3(1), dim3(256), 0, 0,
            d_out.ptr, byte_extent, opts.flags);
        for (unsigned int group = 0; group < 19u; ++group) {
            if (group > 15u && group != 18u) {
                continue;
            }
            for (unsigned int slot = 0; slot < HRX_COOPSTORE_VALUES_PER_GROUP; ++slot) {
                for (unsigned int lane = 0; lane < HRX_COOPSTORE_LANES; ++lane) {
                    h_expected[coopstore_probe_index(group, slot, lane)] =
                        group < 16u ? 2.0f * coopstore_probe_wmma_expected_value(group) :
                                      static_cast<float>(coopstore_probe_stage_value(group, slot, lane));
                }
            }
        }
    } else if (opts.mode == "wmma-lds-k2-stage-one18-nodirect") {
        hipLaunchKernelGGL((coopstore_probe_wmma_lds_k2_stage_one18<false>), dim3(1), dim3(256), 0, 0,
            d_out.ptr, byte_extent, opts.flags);
        for (unsigned int slot = 0; slot < HRX_COOPSTORE_VALUES_PER_GROUP; ++slot) {
            for (unsigned int lane = 0; lane < HRX_COOPSTORE_LANES; ++lane) {
                h_expected[coopstore_probe_index(18u, slot, lane)] =
                    static_cast<float>(coopstore_probe_stage_value(18u, slot, lane));
            }
        }
    } else if (opts.mode == "wmma-lds-k2-stagefirst-mixed192") {
        hipLaunchKernelGGL(coopstore_probe_wmma_lds_k2_stagefirst_mixed192, dim3(1), dim3(256), 0, 0,
            d_out.ptr, byte_extent, opts.flags);
        for (unsigned int group = 0; group < HRX_COOPSTORE_MAX_GROUPS; ++group) {
            for (unsigned int slot = 0; slot < HRX_COOPSTORE_VALUES_PER_GROUP; ++slot) {
                for (unsigned int lane = 0; lane < HRX_COOPSTORE_LANES; ++lane) {
                    h_expected[coopstore_probe_index(group, slot, lane)] =
                        group < 16u ? 2.0f * coopstore_probe_wmma_expected_value(group) :
                                      static_cast<float>(coopstore_probe_stage_value(group, slot, lane));
                }
            }
        }
    } else if (opts.mode == "wmma-lds-k2-mixed96") {
        hipLaunchKernelGGL(coopstore_probe_wmma_lds_k2_mixed96, dim3(1), dim3(256), 0, 0,
            d_out.ptr, byte_extent, opts.flags);
        for (unsigned int group = 0; group < 24u; ++group) {
            for (unsigned int slot = 0; slot < HRX_COOPSTORE_VALUES_PER_GROUP; ++slot) {
                for (unsigned int lane = 0; lane < HRX_COOPSTORE_LANES; ++lane) {
                    h_expected[coopstore_probe_index(group, slot, lane)] =
                        group < 16u ? 2.0f * coopstore_probe_wmma_expected_value(group) :
                                      static_cast<float>(coopstore_probe_stage_value(group, slot, lane));
                }
            }
        }
    } else if (opts.mode == "wmma-lds-k2-typedstage-mixed96") {
        hipLaunchKernelGGL(coopstore_probe_wmma_lds_k2_typedstage_mixed96, dim3(1), dim3(256), 0, 0,
            d_out.ptr, byte_extent, opts.flags);
        for (unsigned int group = 0; group < 24u; ++group) {
            for (unsigned int slot = 0; slot < HRX_COOPSTORE_VALUES_PER_GROUP; ++slot) {
                for (unsigned int lane = 0; lane < HRX_COOPSTORE_LANES; ++lane) {
                    h_expected[coopstore_probe_index(group, slot, lane)] =
                        group < 16u ? 2.0f * coopstore_probe_wmma_expected_value(group) :
                                      static_cast<float>(coopstore_probe_stage_value(group, slot, lane));
                }
            }
        }
    } else if (opts.mode == "wmma-lds-k2-kloop-mixed96") {
        hipLaunchKernelGGL(coopstore_probe_wmma_lds_k2_kloop_mixed96, dim3(1), dim3(256), 0, 0,
            d_out.ptr, byte_extent, opts.flags);
        for (unsigned int group = 0; group < 24u; ++group) {
            for (unsigned int slot = 0; slot < HRX_COOPSTORE_VALUES_PER_GROUP; ++slot) {
                for (unsigned int lane = 0; lane < HRX_COOPSTORE_LANES; ++lane) {
                    h_expected[coopstore_probe_index(group, slot, lane)] =
                        group < 16u ? 2.0f * coopstore_probe_wmma_expected_value(group) :
                                      static_cast<float>(coopstore_probe_stage_value(group, slot, lane));
                }
            }
        }
    } else if (opts.mode == "wmma-lds-k2-kloop-direct64") {
        hipLaunchKernelGGL(coopstore_probe_wmma_lds_k2_kloop_direct64, dim3(1), dim3(256), 0, 0,
            d_out.ptr, byte_extent, opts.flags);
        for (unsigned int group = 0; group < 16u; ++group) {
            for (unsigned int slot = 0; slot < HRX_COOPSTORE_VALUES_PER_GROUP; ++slot) {
                for (unsigned int lane = 0; lane < HRX_COOPSTORE_LANES; ++lane) {
                    h_expected[coopstore_probe_index(group, slot, lane)] =
                        2.0f * coopstore_probe_wmma_expected_value(group);
                }
            }
        }
    } else if (opts.mode == "wmma-lds-k2-kloop-asm-mixed96") {
        hipLaunchKernelGGL(coopstore_probe_wmma_lds_k2_kloop_asm_mixed96, dim3(1), dim3(256), 0, 0,
            d_out.ptr, byte_extent, opts.flags);
        for (unsigned int group = 0; group < 24u; ++group) {
            for (unsigned int slot = 0; slot < HRX_COOPSTORE_VALUES_PER_GROUP; ++slot) {
                for (unsigned int lane = 0; lane < HRX_COOPSTORE_LANES; ++lane) {
                    h_expected[coopstore_probe_index(group, slot, lane)] =
                        group < 16u ? 2.0f * coopstore_probe_wmma_expected_value(group) :
                                      static_cast<float>(coopstore_probe_stage_value(group, slot, lane));
                }
            }
        }
    } else if (opts.mode == "wmma-lds-k2-kloop-asm-stagefirst-mixed96") {
        hipLaunchKernelGGL(coopstore_probe_wmma_lds_k2_kloop_asm_stagefirst_mixed96, dim3(1), dim3(256), 0, 0,
            d_out.ptr, byte_extent, opts.flags);
        for (unsigned int group = 0; group < 24u; ++group) {
            for (unsigned int slot = 0; slot < HRX_COOPSTORE_VALUES_PER_GROUP; ++slot) {
                for (unsigned int lane = 0; lane < HRX_COOPSTORE_LANES; ++lane) {
                    h_expected[coopstore_probe_index(group, slot, lane)] =
                        group < 16u ? 2.0f * coopstore_probe_wmma_expected_value(group) :
                                      static_cast<float>(coopstore_probe_stage_value(group, slot, lane));
                }
            }
        }
    } else if (opts.mode == "wmma-lds-k2-kloop-asm-linearstage-mixed96") {
        hipLaunchKernelGGL(coopstore_probe_wmma_lds_k2_kloop_asm_linearstage_mixed96, dim3(1), dim3(256), 0, 0,
            d_out.ptr, byte_extent, opts.flags);
        for (unsigned int group = 0; group < 24u; ++group) {
            for (unsigned int slot = 0; slot < HRX_COOPSTORE_VALUES_PER_GROUP; ++slot) {
                for (unsigned int lane = 0; lane < HRX_COOPSTORE_LANES; ++lane) {
                    h_expected[coopstore_probe_index(group, slot, lane)] =
                        group < 16u ? 2.0f * coopstore_probe_wmma_expected_value(group) :
                                      static_cast<float>(coopstore_probe_stage_value(group, slot, lane));
                }
            }
        }
    } else if (opts.mode == "wmma-lds-k2-kloop-asm-typedstage-mixed96") {
        hipLaunchKernelGGL(coopstore_probe_wmma_lds_k2_kloop_asm_typedstage_mixed96, dim3(1), dim3(256), 0, 0,
            d_out.ptr, byte_extent, opts.flags);
        for (unsigned int group = 0; group < 24u; ++group) {
            for (unsigned int slot = 0; slot < HRX_COOPSTORE_VALUES_PER_GROUP; ++slot) {
                for (unsigned int lane = 0; lane < HRX_COOPSTORE_LANES; ++lane) {
                    h_expected[coopstore_probe_index(group, slot, lane)] =
                        group < 16u ? 2.0f * coopstore_probe_wmma_expected_value(group) :
                                      static_cast<float>(coopstore_probe_stage_value(group, slot, lane));
                }
            }
        }
    } else if (opts.mode == "wmma-lds-k2-kloop-asm-direct64") {
        hipLaunchKernelGGL(coopstore_probe_wmma_lds_k2_kloop_asm_direct64, dim3(1), dim3(256), 0, 0,
            d_out.ptr, byte_extent, opts.flags);
        for (unsigned int group = 0; group < 16u; ++group) {
            for (unsigned int slot = 0; slot < HRX_COOPSTORE_VALUES_PER_GROUP; ++slot) {
                for (unsigned int lane = 0; lane < HRX_COOPSTORE_LANES; ++lane) {
                    h_expected[coopstore_probe_index(group, slot, lane)] =
                        2.0f * coopstore_probe_wmma_expected_value(group);
                }
            }
        }
    } else if (opts.mode == "wmma-lds-vk64-radv96") {
        hipLaunchKernelGGL(coopstore_probe_wmma_lds_vk64_radv96, dim3(1), dim3(256), 0, 0,
            d_out.ptr, byte_extent, opts.flags);
        for (unsigned int group = 0; group < 24u; ++group) {
            for (unsigned int slot = 0; slot < HRX_COOPSTORE_VALUES_PER_GROUP; ++slot) {
                for (unsigned int lane = 0; lane < HRX_COOPSTORE_LANES; ++lane) {
                    h_expected[coopstore_probe_index(group, slot, lane)] =
                        group < 8u ? coopstore_probe_value(group, slot, lane) :
                                     static_cast<float>(coopstore_probe_stage_value(group, slot, lane));
                }
            }
        }
    } else if (opts.mode == "wmma-lds-vk64-radv96-accdirect") {
        hipLaunchKernelGGL(coopstore_probe_wmma_lds_vk64_radv96_accdirect, dim3(1), dim3(256), 0, 0,
            d_out.ptr, byte_extent, opts.flags);
        for (unsigned int group = 0; group < 24u; ++group) {
            for (unsigned int slot = 0; slot < HRX_COOPSTORE_VALUES_PER_GROUP; ++slot) {
                for (unsigned int lane = 0; lane < HRX_COOPSTORE_LANES; ++lane) {
                    h_expected[coopstore_probe_index(group, slot, lane)] =
                        group < 8u ? coopstore_probe_wmma_ring12_expected_value(group) :
                                     static_cast<float>(coopstore_probe_stage_value(group, slot, lane));
                }
            }
        }
    } else if (opts.mode == "wmma-lds-vk64-radv96-accslots") {
        hipLaunchKernelGGL(coopstore_probe_wmma_lds_vk64_radv96_accslots, dim3(1), dim3(256), 0, 0,
            d_out.ptr, byte_extent, opts.flags);
        for (unsigned int group = 0; group < 16u; ++group) {
            const unsigned int src_group = group < 8u ? group : group - 8u;
            for (unsigned int slot = 0; slot < HRX_COOPSTORE_VALUES_PER_GROUP; ++slot) {
                for (unsigned int lane = 0; lane < HRX_COOPSTORE_LANES; ++lane) {
                    h_expected[coopstore_probe_index(group, slot, lane)] =
                        coopstore_probe_wmma_ring12_expected_value(src_group);
                }
            }
        }
    } else if (opts.mode == "wmma-lds-vk64-radv96-accdirect-copya") {
        hipLaunchKernelGGL((coopstore_probe_wmma_lds_vk64_radv96_accdirect_copy<true, false>), dim3(1), dim3(256), 0, 0,
            d_out.ptr, byte_extent, opts.flags);
        for (unsigned int group = 0; group < 24u; ++group) {
            for (unsigned int slot = 0; slot < HRX_COOPSTORE_VALUES_PER_GROUP; ++slot) {
                for (unsigned int lane = 0; lane < HRX_COOPSTORE_LANES; ++lane) {
                    h_expected[coopstore_probe_index(group, slot, lane)] =
                        group < 8u ? coopstore_probe_wmma_ring12_expected_value(group) :
                                     static_cast<float>(coopstore_probe_stage_value(group, slot, lane));
                }
            }
        }
    } else if (opts.mode == "wmma-lds-vk64-radv96-accdirect-copyb") {
        hipLaunchKernelGGL((coopstore_probe_wmma_lds_vk64_radv96_accdirect_copy<false, true>), dim3(1), dim3(256), 0, 0,
            d_out.ptr, byte_extent, opts.flags);
        for (unsigned int group = 0; group < 24u; ++group) {
            for (unsigned int slot = 0; slot < HRX_COOPSTORE_VALUES_PER_GROUP; ++slot) {
                for (unsigned int lane = 0; lane < HRX_COOPSTORE_LANES; ++lane) {
                    h_expected[coopstore_probe_index(group, slot, lane)] =
                        group < 8u ? coopstore_probe_wmma_ring12_expected_value(group) :
                                     static_cast<float>(coopstore_probe_stage_value(group, slot, lane));
                }
            }
        }
    } else if (opts.mode == "wmma-lds-vk64-radv96-accdirect-copyab") {
        hipLaunchKernelGGL((coopstore_probe_wmma_lds_vk64_radv96_accdirect_copy<true, true>), dim3(1), dim3(256), 0, 0,
            d_out.ptr, byte_extent, opts.flags);
        for (unsigned int group = 0; group < 24u; ++group) {
            for (unsigned int slot = 0; slot < HRX_COOPSTORE_VALUES_PER_GROUP; ++slot) {
                for (unsigned int lane = 0; lane < HRX_COOPSTORE_LANES; ++lane) {
                    h_expected[coopstore_probe_index(group, slot, lane)] =
                        group < 8u ? coopstore_probe_wmma_ring12_expected_value(group) :
                                     static_cast<float>(coopstore_probe_stage_value(group, slot, lane));
                }
            }
        }
    } else if (opts.mode == "wmma-lds-vk64-radv96-accdirect-copyab-q6addr") {
        hipLaunchKernelGGL((coopstore_probe_wmma_lds_vk64_radv96_accdirect_copy_q6addr<true, true, true>), dim3(1), dim3(256), 0, 0,
            d_out.ptr, byte_extent, opts.flags, opts.rows, opts.cols);
        for (unsigned int group = 0; group < 24u; ++group) {
            for (unsigned int slot = 0; slot < HRX_COOPSTORE_VALUES_PER_GROUP; ++slot) {
                for (unsigned int lane = 0; lane < HRX_COOPSTORE_LANES; ++lane) {
                    unsigned int index = 0;
                    if (coopstore_probe_q6addr_index(group, slot, lane, opts.rows, opts.cols, &index)) {
                        h_expected[index] =
                            group < 8u ? coopstore_probe_wmma_ring12_expected_value(group) :
                                         static_cast<float>(coopstore_probe_stage_value(group, slot, lane));
                    }
                }
            }
        }
    } else if (opts.mode == "wmma-lds-vk64-radv96-accdirect-copyab-q6addr-nosink") {
        hipLaunchKernelGGL((coopstore_probe_wmma_lds_vk64_radv96_accdirect_copy_q6addr<true, true, false>), dim3(1), dim3(256), 0, 0,
            d_out.ptr, byte_extent, opts.flags, opts.rows, opts.cols);
        for (unsigned int group = 0; group < 24u; ++group) {
            for (unsigned int slot = 0; slot < HRX_COOPSTORE_VALUES_PER_GROUP; ++slot) {
                for (unsigned int lane = 0; lane < HRX_COOPSTORE_LANES; ++lane) {
                    unsigned int index = 0;
                    if (coopstore_probe_q6addr_index(group, slot, lane, opts.rows, opts.cols, &index)) {
                        h_expected[index] =
                            group < 8u ? coopstore_probe_wmma_ring12_expected_value(group) :
                                         static_cast<float>(coopstore_probe_stage_value(group, slot, lane));
                    }
                }
            }
        }
    } else if (opts.mode == "wmma-lds-vk64-radv96-accdirect-copyab-q6addr-upperreal") {
        hipLaunchKernelGGL((coopstore_probe_wmma_lds_vk64_radv96_accdirect_copy_q6addr_upperreal<true, true>), dim3(1), dim3(256), 0, 0,
            d_out.ptr, byte_extent, opts.flags, opts.rows, opts.cols);
        for (unsigned int group = 0; group < 24u; ++group) {
            for (unsigned int slot = 0; slot < HRX_COOPSTORE_VALUES_PER_GROUP; ++slot) {
                for (unsigned int lane = 0; lane < HRX_COOPSTORE_LANES; ++lane) {
                    unsigned int index = 0;
                    if (coopstore_probe_q6addr_index(group, slot, lane, opts.rows, opts.cols, &index)) {
                        h_expected[index] =
                            group < 16u ? coopstore_probe_wmma_ring12_expected_value(group) :
                                          static_cast<float>(coopstore_probe_stage_value(group, slot, lane));
                    }
                }
            }
        }
    } else if (opts.mode == "wmma-lds-vk64-radv96-accdirect-copyab-q6addr-upperwait") {
        hipLaunchKernelGGL((coopstore_probe_wmma_lds_vk64_radv96_accdirect_copy_q6addr_upperwait<true, true>), dim3(1), dim3(256), 0, 0,
            d_out.ptr, byte_extent, opts.flags, opts.rows, opts.cols);
        for (unsigned int group = 0; group < 24u; ++group) {
            for (unsigned int slot = 0; slot < HRX_COOPSTORE_VALUES_PER_GROUP; ++slot) {
                for (unsigned int lane = 0; lane < HRX_COOPSTORE_LANES; ++lane) {
                    unsigned int index = 0;
                    if (coopstore_probe_q6addr_index(group, slot, lane, opts.rows, opts.cols, &index)) {
                        h_expected[index] =
                            group < 16u ? coopstore_probe_wmma_ring12_expected_value(group) :
                                          static_cast<float>(coopstore_probe_stage_value(group, slot, lane));
                    }
                }
            }
        }
    } else if (opts.mode == "wmma-lds-vk64-radv96-accdirect-copyab-q6addr-upperwide") {
        hipLaunchKernelGGL((coopstore_probe_wmma_lds_vk64_radv96_accdirect_copy_q6addr_upperwide<true, true>), dim3(1), dim3(256), 0, 0,
            d_out.ptr, byte_extent, opts.flags, opts.rows, opts.cols);
        for (unsigned int group = 0; group < 24u; ++group) {
            for (unsigned int slot = 0; slot < HRX_COOPSTORE_VALUES_PER_GROUP; ++slot) {
                for (unsigned int lane = 0; lane < HRX_COOPSTORE_LANES; ++lane) {
                    unsigned int index = 0;
                    if (coopstore_probe_q6addr_index(group, slot, lane, opts.rows, opts.cols, &index)) {
                        h_expected[index] =
                            group < 16u ? coopstore_probe_wmma_ring12_expected_value(group) :
                                          static_cast<float>(coopstore_probe_stage_value(group, slot, lane));
                    }
                }
            }
        }
    } else if (opts.mode == "wmma-lds-k2-mixed128") {
        hipLaunchKernelGGL(coopstore_probe_wmma_lds_k2_mixed128, dim3(1), dim3(256), 0, 0,
            d_out.ptr, byte_extent, opts.flags);
        for (unsigned int group = 0; group < 32u; ++group) {
            for (unsigned int slot = 0; slot < HRX_COOPSTORE_VALUES_PER_GROUP; ++slot) {
                for (unsigned int lane = 0; lane < HRX_COOPSTORE_LANES; ++lane) {
                    h_expected[coopstore_probe_index(group, slot, lane)] =
                        group < 16u ? 2.0f * coopstore_probe_wmma_expected_value(group) :
                                      static_cast<float>(coopstore_probe_stage_value(group, slot, lane));
                }
            }
        }
    } else if (opts.mode == "wmma-lds-k2-mixed128-padded32") {
        hipLaunchKernelGGL(coopstore_probe_wmma_lds_k2_mixed128_padded32, dim3(1), dim3(256), 0, 0,
            d_out.ptr, byte_extent, opts.flags);
        for (unsigned int group = 0; group < 32u; ++group) {
            for (unsigned int slot = 0; slot < HRX_COOPSTORE_VALUES_PER_GROUP; ++slot) {
                for (unsigned int lane = 0; lane < HRX_COOPSTORE_LANES; ++lane) {
                    h_expected[coopstore_probe_index(group, slot, lane)] =
                        group < 16u ? 2.0f * coopstore_probe_wmma_expected_value(group) :
                                      static_cast<float>(coopstore_probe_stage_value(group, slot, lane));
                }
            }
        }
    } else if (opts.mode == "wmma-lds-k2-mixed160-lo") {
        hipLaunchKernelGGL(coopstore_probe_wmma_lds_k2_mixed160_lo, dim3(1), dim3(256), 0, 0,
            d_out.ptr, byte_extent, opts.flags);
        for (unsigned int group = 0; group < 40u; ++group) {
            for (unsigned int slot = 0; slot < HRX_COOPSTORE_VALUES_PER_GROUP; ++slot) {
                for (unsigned int lane = 0; lane < HRX_COOPSTORE_LANES; ++lane) {
                    h_expected[coopstore_probe_index(group, slot, lane)] =
                        group < 16u ? 2.0f * coopstore_probe_wmma_expected_value(group) :
                                      static_cast<float>(coopstore_probe_stage_value(group, slot, lane));
                }
            }
        }
    } else if (opts.mode == "wmma-lds-k2-mixed160-hi") {
        hipLaunchKernelGGL(coopstore_probe_wmma_lds_k2_mixed160_hi, dim3(1), dim3(256), 0, 0,
            d_out.ptr, byte_extent, opts.flags);
        for (unsigned int group = 0; group < 32u; ++group) {
            for (unsigned int slot = 0; slot < HRX_COOPSTORE_VALUES_PER_GROUP; ++slot) {
                for (unsigned int lane = 0; lane < HRX_COOPSTORE_LANES; ++lane) {
                    h_expected[coopstore_probe_index(group, slot, lane)] =
                        group < 16u ? 2.0f * coopstore_probe_wmma_expected_value(group) :
                                      static_cast<float>(coopstore_probe_stage_value(group, slot, lane));
                }
            }
        }
        for (unsigned int group = 40u; group < HRX_COOPSTORE_MAX_GROUPS; ++group) {
            for (unsigned int slot = 0; slot < HRX_COOPSTORE_VALUES_PER_GROUP; ++slot) {
                for (unsigned int lane = 0; lane < HRX_COOPSTORE_LANES; ++lane) {
                    h_expected[coopstore_probe_index(group, slot, lane)] =
                        static_cast<float>(coopstore_probe_stage_value(group, slot, lane));
                }
            }
        }
    } else if (opts.mode == "wmma-lds-k2-mixed160-lo-tight") {
        hipLaunchKernelGGL(coopstore_probe_wmma_lds_k2_mixed160_lo_tight, dim3(1), dim3(256), 0, 0,
            d_out.ptr, byte_extent, opts.flags);
        for (unsigned int group = 0; group < 40u; ++group) {
            for (unsigned int slot = 0; slot < HRX_COOPSTORE_VALUES_PER_GROUP; ++slot) {
                for (unsigned int lane = 0; lane < HRX_COOPSTORE_LANES; ++lane) {
                    h_expected[coopstore_probe_index(group, slot, lane)] =
                        group < 16u ? 2.0f * coopstore_probe_wmma_expected_value(group) :
                                      static_cast<float>(coopstore_probe_stage_value(group, slot, lane));
                }
            }
        }
    } else if (opts.mode == "wmma-lds-k2-mixed160-hi-tight") {
        hipLaunchKernelGGL(coopstore_probe_wmma_lds_k2_mixed160_hi_tight, dim3(1), dim3(256), 0, 0,
            d_out.ptr, byte_extent, opts.flags);
        for (unsigned int group = 0; group < 32u; ++group) {
            for (unsigned int slot = 0; slot < HRX_COOPSTORE_VALUES_PER_GROUP; ++slot) {
                for (unsigned int lane = 0; lane < HRX_COOPSTORE_LANES; ++lane) {
                    h_expected[coopstore_probe_index(group, slot, lane)] =
                        group < 16u ? 2.0f * coopstore_probe_wmma_expected_value(group) :
                                      static_cast<float>(coopstore_probe_stage_value(group, slot, lane));
                }
            }
        }
        for (unsigned int group = 40u; group < HRX_COOPSTORE_MAX_GROUPS; ++group) {
            for (unsigned int slot = 0; slot < HRX_COOPSTORE_VALUES_PER_GROUP; ++slot) {
                for (unsigned int lane = 0; lane < HRX_COOPSTORE_LANES; ++lane) {
                    h_expected[coopstore_probe_index(group, slot, lane)] =
                        static_cast<float>(coopstore_probe_stage_value(group, slot, lane));
                }
            }
        }
    } else if (opts.mode == "wmma-lds-k2-stage96-accsink") {
        hipLaunchKernelGGL(coopstore_probe_wmma_lds_k2_stage96_accsink, dim3(1), dim3(256), 0, 0,
            d_out.ptr, byte_extent, opts.flags);
        for (unsigned int group = 16u; group < 40u; ++group) {
            for (unsigned int slot = 0; slot < HRX_COOPSTORE_VALUES_PER_GROUP; ++slot) {
                for (unsigned int lane = 0; lane < HRX_COOPSTORE_LANES; ++lane) {
                    h_expected[coopstore_probe_index(group, slot, lane)] =
                        static_cast<float>(coopstore_probe_stage_value(group, slot, lane));
                }
            }
        }
    } else if (opts.mode == "wmma-lds-k2-mixed160-linearstage") {
        hipLaunchKernelGGL(coopstore_probe_wmma_lds_k2_mixed160_linearstage, dim3(1), dim3(256), 0, 0,
            d_out.ptr, byte_extent, opts.flags);
        for (unsigned int group = 0; group < 40u; ++group) {
            for (unsigned int slot = 0; slot < HRX_COOPSTORE_VALUES_PER_GROUP; ++slot) {
                for (unsigned int lane = 0; lane < HRX_COOPSTORE_LANES; ++lane) {
                    h_expected[coopstore_probe_index(group, slot, lane)] =
                        group < 16u ? 2.0f * coopstore_probe_wmma_expected_value(group) :
                                      static_cast<float>(coopstore_probe_stage_value(group, slot, lane));
                }
            }
        }
    } else if (opts.mode == "wmma-lds-k2-mixed160-splitstage") {
        hipLaunchKernelGGL(coopstore_probe_wmma_lds_k2_mixed160_splitstage, dim3(1), dim3(256), 0, 0,
            d_out.ptr, byte_extent, opts.flags);
        for (unsigned int group = 0; group < 40u; ++group) {
            for (unsigned int slot = 0; slot < HRX_COOPSTORE_VALUES_PER_GROUP; ++slot) {
                for (unsigned int lane = 0; lane < HRX_COOPSTORE_LANES; ++lane) {
                    h_expected[coopstore_probe_index(group, slot, lane)] =
                        group < 16u ? 2.0f * coopstore_probe_wmma_expected_value(group) :
                                      static_cast<float>(coopstore_probe_stage_value(group, slot, lane));
                }
            }
        }
    } else if (opts.mode == "wmma-lds-k2-mixed192-splitstage") {
        hipLaunchKernelGGL(coopstore_probe_wmma_lds_k2_mixed192_splitstage, dim3(1), dim3(256), 0, 0,
            d_out.ptr, byte_extent, opts.flags);
        for (unsigned int group = 0; group < HRX_COOPSTORE_MAX_GROUPS; ++group) {
            for (unsigned int slot = 0; slot < HRX_COOPSTORE_VALUES_PER_GROUP; ++slot) {
                for (unsigned int lane = 0; lane < HRX_COOPSTORE_LANES; ++lane) {
                    h_expected[coopstore_probe_index(group, slot, lane)] =
                        group < 16u ? 2.0f * coopstore_probe_wmma_expected_value(group) :
                                      static_cast<float>(coopstore_probe_stage_value(group, slot, lane));
                }
            }
        }
    } else if (opts.mode == "wmma-lds-k2-mixed192-splitstage8") {
        hipLaunchKernelGGL(coopstore_probe_wmma_lds_k2_mixed192_splitstage8, dim3(1), dim3(256), 0, 0,
            d_out.ptr, byte_extent, opts.flags);
        for (unsigned int group = 0; group < HRX_COOPSTORE_MAX_GROUPS; ++group) {
            for (unsigned int slot = 0; slot < HRX_COOPSTORE_VALUES_PER_GROUP; ++slot) {
                for (unsigned int lane = 0; lane < HRX_COOPSTORE_LANES; ++lane) {
                    h_expected[coopstore_probe_index(group, slot, lane)] =
                        group < 16u ? 2.0f * coopstore_probe_wmma_expected_value(group) :
                                      static_cast<float>(coopstore_probe_stage_value(group, slot, lane));
                }
            }
        }
    } else if (opts.mode == "wmma-lds-k2-direct160-raw") {
        hipLaunchKernelGGL(coopstore_probe_wmma_lds_k2_direct160_raw, dim3(1), dim3(256), 0, 0,
            d_out.ptr, byte_extent, opts.flags);
        for (unsigned int group = 0; group < 40u; ++group) {
            for (unsigned int slot = 0; slot < HRX_COOPSTORE_VALUES_PER_GROUP; ++slot) {
                for (unsigned int lane = 0; lane < HRX_COOPSTORE_LANES; ++lane) {
                    h_expected[coopstore_probe_index(group, slot, lane)] =
                        group < 16u ? 2.0f * coopstore_probe_wmma_expected_value(group) :
                                      coopstore_probe_value(group, slot, lane);
                }
            }
        }
    } else if (opts.mode == "wmma-lds-k2-direct192-raw") {
        hipLaunchKernelGGL(coopstore_probe_wmma_lds_k2_direct192_raw, dim3(1), dim3(256), 0, 0,
            d_out.ptr, byte_extent, opts.flags);
        for (unsigned int group = 0; group < HRX_COOPSTORE_MAX_GROUPS; ++group) {
            for (unsigned int slot = 0; slot < HRX_COOPSTORE_VALUES_PER_GROUP; ++slot) {
                for (unsigned int lane = 0; lane < HRX_COOPSTORE_LANES; ++lane) {
                    h_expected[coopstore_probe_index(group, slot, lane)] =
                        group < 16u ? 2.0f * coopstore_probe_wmma_expected_value(group) :
                                      coopstore_probe_value(group, slot, lane);
                }
            }
        }
    } else if (opts.mode == "wmma-lds-k2-accslots-raw192") {
        hipLaunchKernelGGL(coopstore_probe_wmma_lds_k2_accslots_raw192, dim3(1), dim3(256), 0, 0,
            d_out.ptr, byte_extent, opts.flags);
        for (unsigned int group = 0; group < HRX_COOPSTORE_MAX_GROUPS; ++group) {
            const unsigned int src_group = group < 8u ? group : group - 8u;
            for (unsigned int slot = 0; slot < HRX_COOPSTORE_VALUES_PER_GROUP; ++slot) {
                for (unsigned int lane = 0; lane < HRX_COOPSTORE_LANES; ++lane) {
                    h_expected[coopstore_probe_index(group, slot, lane)] =
                        group < 16u ? 2.0f * coopstore_probe_wmma_expected_value(src_group) :
                                      coopstore_probe_value(group, slot, lane);
                }
            }
        }
    } else if (opts.mode == "wmma-lds-k2-direct64") {
        hipLaunchKernelGGL(coopstore_probe_wmma_lds_k2_direct64, dim3(1), dim3(256), 0, 0,
            d_out.ptr, byte_extent, opts.flags);
        for (unsigned int group = 0; group < 16u; ++group) {
            for (unsigned int slot = 0; slot < HRX_COOPSTORE_VALUES_PER_GROUP; ++slot) {
                for (unsigned int lane = 0; lane < HRX_COOPSTORE_LANES; ++lane) {
                    h_expected[coopstore_probe_index(group, slot, lane)] =
                        2.0f * coopstore_probe_wmma_expected_value(group);
                }
            }
        }
    } else if (opts.mode == "wmma-lds-k2-radv-branch64") {
        hipLaunchKernelGGL(coopstore_probe_wmma_lds_k2_radv_branch64, dim3(1), dim3(256), 0, 0,
            d_out.ptr, byte_extent, opts.flags, opts.group);
        const unsigned int first_group = opts.group == 0u ? 0u : (opts.group == 1u ? 16u : 32u);
        const unsigned int last_group = first_group + 16u;
        for (unsigned int group = first_group; group < last_group; ++group) {
            for (unsigned int slot = 0; slot < HRX_COOPSTORE_VALUES_PER_GROUP; ++slot) {
                for (unsigned int lane = 0; lane < HRX_COOPSTORE_LANES; ++lane) {
                    h_expected[coopstore_probe_index(group, slot, lane)] =
                        opts.group == 0u ? 2.0f * coopstore_probe_wmma_expected_value(group) :
                                           static_cast<float>(coopstore_probe_stage_value(group, slot, lane));
                }
            }
        }
    } else {
        std::fprintf(stderr, "unknown mode: %s\n", opts.mode.c_str());
        return 2;
    }
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_out.data(), d_out.ptr, count * sizeof(float), hipMemcpyDeviceToHost));

    std::printf("coopmat-store-contract mode=%s group=%u flags=0x%x rows=%u cols=%u bytes=%llu\n",
        opts.mode.c_str(), opts.group, opts.flags, opts.rows, opts.cols, byte_extent);
    compare_outputs(h_out, h_expected, "check");
    return 0;
}
