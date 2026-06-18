#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define HIP_CHECK(expr) do { \
    hipError_t _err = (expr); \
    if (_err != hipSuccess) { \
        std::fprintf(stderr, "%s:%d: HIP error: %s\n", __FILE__, __LINE__, hipGetErrorString(_err)); \
        std::exit(2); \
    } \
} while (0)

typedef _Float16 wmma_lane_map_half16_vec __attribute__((ext_vector_type(16)));
typedef _Float16 wmma_lane_map_half8_vec __attribute__((ext_vector_type(8)));

static constexpr unsigned int HRX_WMMA_LANE_MAP_LANES = 64;
static constexpr unsigned int HRX_WMMA_LANE_MAP_ACC_SLOTS = 8;
static constexpr unsigned int HRX_WMMA_LANE_MAP_OPSELS = 2;
static constexpr float HRX_WMMA_LANE_MAP_DOT = 16.0f;

static __host__ __device__ __forceinline__ unsigned int wmma_lane_map_index(
        unsigned int opsel,
        unsigned int lane,
        unsigned int slot) {
    return (opsel * HRX_WMMA_LANE_MAP_LANES + lane) * HRX_WMMA_LANE_MAP_ACC_SLOTS + slot;
}

static __host__ __device__ __forceinline__ float wmma_lane_map_sentinel(
        unsigned int lane,
        unsigned int slot) {
    return -static_cast<float>((lane + 1u) * 16u + slot);
}

template <bool op_sel>
__global__ __launch_bounds__(64, 1)
void wmma_f16_lane_map_probe(float * dst) {
    const unsigned int lane = __builtin_amdgcn_workitem_id_x() & 63u;

    wmma_lane_map_half16_vec a;
    wmma_lane_map_half16_vec b;
    wmma_lane_map_half8_vec acc;

#pragma unroll
    for (int i = 0; i < 16; ++i) {
        a[i] = static_cast<_Float16>(1.0f);
        b[i] = static_cast<_Float16>(1.0f);
    }

#pragma unroll
    for (int i = 0; i < 8; ++i) {
        acc[i] = static_cast<_Float16>(wmma_lane_map_sentinel(lane, static_cast<unsigned int>(i)));
    }

    acc = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(a, b, acc, op_sel);

#pragma unroll
    for (int i = 0; i < 8; ++i) {
        dst[wmma_lane_map_index(op_sel ? 1u : 0u, lane, static_cast<unsigned int>(i))] =
            static_cast<float>(acc[i]);
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

static bool close_enough(float actual, float expected) {
    return std::fabs(actual - expected) <= 0.01f;
}

static void compare_outputs(const std::vector<float> & actual) {
    size_t bad_count = 0;
    size_t first_bad = 0;
    float first_expected = 0.0f;

    for (unsigned int opsel = 0; opsel < HRX_WMMA_LANE_MAP_OPSELS; ++opsel) {
        for (unsigned int lane = 0; lane < HRX_WMMA_LANE_MAP_LANES; ++lane) {
            for (unsigned int slot = 0; slot < HRX_WMMA_LANE_MAP_ACC_SLOTS; ++slot) {
                const float sentinel = wmma_lane_map_sentinel(lane, slot);
                const bool selected = (slot & 1u) == opsel;
                const float expected = selected ? sentinel + HRX_WMMA_LANE_MAP_DOT : sentinel;
                const size_t idx = wmma_lane_map_index(opsel, lane, slot);
                if (!close_enough(actual[idx], expected)) {
                    if (bad_count == 0) {
                        first_bad = idx;
                        first_expected = expected;
                    }
                    ++bad_count;
                }
            }
        }
    }

    std::printf("check: elements=%zu bad=%zu", actual.size(), bad_count);
    if (bad_count != 0) {
        std::printf(" first_bad=%zu actual=%f expected=%f",
            first_bad,
            static_cast<double>(actual[first_bad]),
            static_cast<double>(first_expected));
    }
    std::printf("\n");

    if (bad_count != 0) {
        std::exit(1);
    }
}

static void print_summary(const std::vector<float> & actual) {
    for (unsigned int opsel = 0; opsel < HRX_WMMA_LANE_MAP_OPSELS; ++opsel) {
        unsigned int changed_even = 0;
        unsigned int changed_odd = 0;
        for (unsigned int lane = 0; lane < HRX_WMMA_LANE_MAP_LANES; ++lane) {
            for (unsigned int slot = 0; slot < HRX_WMMA_LANE_MAP_ACC_SLOTS; ++slot) {
                const float sentinel = wmma_lane_map_sentinel(lane, slot);
                const size_t idx = wmma_lane_map_index(opsel, lane, slot);
                if (!close_enough(actual[idx], sentinel)) {
                    if ((slot & 1u) == 0u) {
                        ++changed_even;
                    } else {
                        ++changed_odd;
                    }
                }
            }
        }
        std::printf("opsel=%u changed_even=%u changed_odd=%u\n", opsel, changed_even, changed_odd);
    }
}

int main() {
    const size_t count =
        HRX_WMMA_LANE_MAP_OPSELS * HRX_WMMA_LANE_MAP_LANES * HRX_WMMA_LANE_MAP_ACC_SLOTS;
    device_buffer<float> d_out(count);
    std::vector<float> h_out(count, 0.0f);

    HIP_CHECK(hipMemset(d_out.ptr, 0, count * sizeof(float)));
    hipLaunchKernelGGL(wmma_f16_lane_map_probe<false>, dim3(1), dim3(64), 0, 0, d_out.ptr);
    HIP_CHECK(hipGetLastError());
    hipLaunchKernelGGL(wmma_f16_lane_map_probe<true>, dim3(1), dim3(64), 0, 0, d_out.ptr);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_out.data(), d_out.ptr, count * sizeof(float), hipMemcpyDeviceToHost));

    std::printf("wmma-f16-lane-map op=v_wmma_f16_16x16x16_f16_w64 lanes=%u slots=%u\n",
        HRX_WMMA_LANE_MAP_LANES,
        HRX_WMMA_LANE_MAP_ACC_SLOTS);
    print_summary(h_out);
    compare_outputs(h_out);
    return 0;
}
