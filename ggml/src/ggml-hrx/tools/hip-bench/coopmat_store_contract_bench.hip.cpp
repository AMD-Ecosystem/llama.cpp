#include <hip/hip_runtime.h>

#include <algorithm>
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

#define HRX_COOPSTORE_STORE_GROUP(GROUP_ID) do { \
    coopstore_probe_raw_store(rsrc, (GROUP_ID), 0u, lane); \
    coopstore_probe_raw_store(rsrc, (GROUP_ID), 1u, lane); \
    coopstore_probe_raw_store(rsrc, (GROUP_ID), 2u, lane); \
    coopstore_probe_raw_store(rsrc, (GROUP_ID), 3u, lane); \
} while (0)

#define HRX_COOPSTORE_GROUPS_0_15() do { \
    HRX_COOPSTORE_STORE_GROUP(0u);  HRX_COOPSTORE_STORE_GROUP(1u); \
    HRX_COOPSTORE_STORE_GROUP(2u);  HRX_COOPSTORE_STORE_GROUP(3u); \
    HRX_COOPSTORE_STORE_GROUP(4u);  HRX_COOPSTORE_STORE_GROUP(5u); \
    HRX_COOPSTORE_STORE_GROUP(6u);  HRX_COOPSTORE_STORE_GROUP(7u); \
    HRX_COOPSTORE_STORE_GROUP(8u);  HRX_COOPSTORE_STORE_GROUP(9u); \
    HRX_COOPSTORE_STORE_GROUP(10u); HRX_COOPSTORE_STORE_GROUP(11u); \
    HRX_COOPSTORE_STORE_GROUP(12u); HRX_COOPSTORE_STORE_GROUP(13u); \
    HRX_COOPSTORE_STORE_GROUP(14u); HRX_COOPSTORE_STORE_GROUP(15u); \
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

#undef HRX_COOPSTORE_GROUPS_32_47
#undef HRX_COOPSTORE_GROUPS_16_31
#undef HRX_COOPSTORE_GROUPS_0_15
#undef HRX_COOPSTORE_STORE_GROUP

struct options {
    std::string mode = "branch192";
    unsigned int group = 17;
    unsigned int flags = 0x31004000u;
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
        } else {
            std::fprintf(stderr,
                "usage: %s [--mode=linear64|linear128|linear192|branch192] [--group=N] [--flags=0x31004000]\n",
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
    for (size_t i = 0; i < actual.size(); ++i) {
        const double diff = std::abs(static_cast<double>(actual[i]) - static_cast<double>(expected[i]));
        if (diff > max_abs) {
            max_abs = diff;
        }
        if (diff != 0.0) {
            if (bad_count == 0) {
                first_bad = i;
            }
            ++bad_count;
        }
    }

    std::printf("%s: elements=%zu bad=%zu max_abs=%g", label, actual.size(), bad_count, max_abs);
    if (bad_count != 0) {
        std::printf(" first_bad=%zu actual=%g expected=%g", first_bad, actual[first_bad], expected[first_bad]);
    }
    std::printf("\n");

    if (bad_count != 0) {
        std::exit(1);
    }
}

int main(int argc, char ** argv) {
    const options opts = parse_options(argc, argv);
    const size_t count = HRX_COOPSTORE_MAX_VALUES;
    const unsigned long long byte_extent = static_cast<unsigned long long>(count * sizeof(float));

    device_buffer<float> d_out(count);
    std::vector<float> h_out(count, 0.0f);
    std::vector<float> h_expected(count, 0.0f);
    HIP_CHECK(hipMemset(d_out.ptr, 0, count * sizeof(float)));

    if (opts.mode == "linear64") {
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
    } else {
        std::fprintf(stderr, "unknown mode: %s\n", opts.mode.c_str());
        return 2;
    }
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_out.data(), d_out.ptr, count * sizeof(float), hipMemcpyDeviceToHost));

    std::printf("coopmat-store-contract mode=%s group=%u flags=0x%x bytes=%llu\n",
        opts.mode.c_str(), opts.group, opts.flags, byte_extent);
    compare_outputs(h_out, h_expected, "check");
    return 0;
}
