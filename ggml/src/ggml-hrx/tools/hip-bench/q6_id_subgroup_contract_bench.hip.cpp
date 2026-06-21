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
typedef uint32_t q6id_contract_u32x8_vec __attribute__((ext_vector_type(8)));
typedef uint64_t q6id_contract_u64x4_vec __attribute__((ext_vector_type(4)));

static constexpr unsigned int HRX_Q6ID_CONTRACT_LANES = 64;
static constexpr unsigned int HRX_Q6ID_CONTRACT_GROUPS = 16;
static constexpr unsigned int HRX_Q6ID_CONTRACT_SLOTS = 2;
static constexpr unsigned int HRX_Q6ID_CONTRACT_VALUES =
    HRX_Q6ID_CONTRACT_GROUPS * HRX_Q6ID_CONTRACT_SLOTS * HRX_Q6ID_CONTRACT_LANES;

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

static __device__ __forceinline__ void q6id_contract_ds_write_b16(
        __attribute__((address_space(3))) uint16_t * ptr,
        uint32_t value) {
    asm volatile("ds_write_b16 %0, %1 offset:0\n"
                 :
                 : "v"(ptr), "v"(value)
                 : "memory");
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

static __device__ __forceinline__ unsigned int q6id_contract_stage_index(
        unsigned int group,
        unsigned int slot,
        unsigned int lane) {
    return (group * HRX_Q6ID_CONTRACT_SLOTS + slot) * HRX_Q6ID_CONTRACT_LANES + lane;
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

static double elapsed_us(const std::chrono::steady_clock::time_point & a,
        const std::chrono::steady_clock::time_point & b) {
    return std::chrono::duration<double, std::micro>(b - a).count();
}

static void clear_buffers(float * d_dst, unsigned int * d_counts) {
    HIP_CHECK(hipMemset(d_dst, 0, HRX_Q6ID_CONTRACT_VALUES * sizeof(float)));
    HIP_CHECK(hipMemset(d_counts, 0, HRX_Q6ID_CONTRACT_VALUES * sizeof(unsigned int)));
}

static double run_kernel(const std::string & name, int reps, float * d_dst, unsigned int * d_counts) {
    clear_buffers(d_dst, d_counts);
    HIP_CHECK(hipDeviceSynchronize());
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; ++i) {
        if (name == "direct") {
            q6_id_subgroup_contract_direct_probe<<<1, 256>>>(d_dst, d_counts);
        } else {
            q6_id_subgroup_contract_staged_probe<<<1, 256>>>(d_dst, d_counts);
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

    const double direct_us = run_kernel("direct", reps, d_dst, d_counts);
    const double staged_us = run_kernel("staged", reps, d_dst, d_counts);
    std::printf("kernel,valid,us\n");
    std::printf("direct,%d,%.6f\n", direct_ok ? 1 : 0, direct_us);
    std::printf("staged,%d,%.6f\n", staged_ok ? 1 : 0, staged_us);

    HIP_CHECK(hipFree(d_dst));
    HIP_CHECK(hipFree(d_counts));
    return (direct_ok && staged_ok) ? 0 : 1;
}
