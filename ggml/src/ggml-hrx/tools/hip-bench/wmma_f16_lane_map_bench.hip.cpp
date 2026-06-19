#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

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

typedef _Float16 wmma_lane_map_half16_vec __attribute__((ext_vector_type(16)));
typedef _Float16 wmma_lane_map_half8_vec __attribute__((ext_vector_type(8)));
typedef uint64_t wmma_lane_map_u64x4_vec __attribute__((ext_vector_type(4)));

static constexpr unsigned int HRX_WMMA_LANE_MAP_LANES = 64;
static constexpr unsigned int HRX_WMMA_LANE_MAP_ACC_SLOTS = 8;
static constexpr unsigned int HRX_WMMA_LANE_MAP_OPSELS = 2;
static constexpr unsigned int HRX_WMMA_LANE_MAP_D_ROWS = 16;
static constexpr unsigned int HRX_WMMA_LANE_MAP_D_COLS = 16;
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

static __host__ __device__ __forceinline__ unsigned int wmma_lane_map_fulltile_index(
        unsigned int tile,
        unsigned int lane,
        unsigned int slot) {
    return (tile * HRX_WMMA_LANE_MAP_LANES + lane) * HRX_WMMA_LANE_MAP_ACC_SLOTS + slot;
}

static __host__ __device__ __forceinline__ unsigned int wmma_lane_map_d_row(
        unsigned int lane,
        unsigned int slot) {
    // AMD Matrix Instruction Calculator, gfx1151/RDNA3, wave64:
    // D[i][j] GPR=floor(i/4), lane=((16*i)%64)+j.
    return ((slot >> 1u) * 4u) + (lane >> 4u);
}

static __host__ __device__ __forceinline__ unsigned int wmma_lane_map_d_col(unsigned int lane) {
    return lane & 15u;
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

static __device__ __forceinline__ uint64_t wmma_lane_map_ds_read_b64_nowait(
        const __attribute__((address_space(3))) uint64_t * ptr) {
    uint64_t value = 0;
    asm volatile("ds_read_b64 %0, %1 offset:0\n"
                 : "=v"(value)
                 : "v"(ptr)
                 : "memory");
    return value;
}

static __device__ __forceinline__ wmma_lane_map_half16_vec wmma_lane_map_load_lds_fragment(
        const __attribute__((address_space(3))) uint64_t * base,
        unsigned int lane,
        unsigned int frag) {
    const unsigned int index = frag * 256u + lane * 4u;
    wmma_lane_map_u64x4_vec raw;
    raw[0] = wmma_lane_map_ds_read_b64_nowait(base + index + 0u);
    raw[1] = wmma_lane_map_ds_read_b64_nowait(base + index + 1u);
    raw[2] = wmma_lane_map_ds_read_b64_nowait(base + index + 2u);
    raw[3] = wmma_lane_map_ds_read_b64_nowait(base + index + 3u);
    return __builtin_bit_cast(wmma_lane_map_half16_vec, raw);
}

template <bool use_lds_fragments>
__global__ __launch_bounds__(64, 1)
void wmma_f16_fulltile_probe(float * dst) {
    __shared__ uint64_t sh[16 * 64 * 4];
    const unsigned int lane = __builtin_amdgcn_workitem_id_x() & 63u;

    wmma_lane_map_half16_vec a[4];
    wmma_lane_map_half16_vec b[4];
    wmma_lane_map_half8_vec acc[16];

    if constexpr (use_lds_fragments) {
#pragma unroll
        for (unsigned int frag = 0; frag < 8u; ++frag) {
#pragma unroll
            for (unsigned int item = 0; item < 4u; ++item) {
                const uint64_t half_bits = static_cast<uint64_t>(0x3c00u + ((frag + item + lane) & 7u));
                const uint64_t packed = half_bits | (half_bits << 16) | (half_bits << 32) | (half_bits << 48);
                sh[frag * 256u + lane * 4u + item] = packed;
            }
        }
        asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
        __syncthreads();

        const __attribute__((address_space(3))) uint64_t * lds =
            (const __attribute__((address_space(3))) uint64_t *) sh;
#pragma unroll
        for (unsigned int i = 0; i < 4u; ++i) {
            a[i] = wmma_lane_map_load_lds_fragment(lds, lane, i);
            b[i] = wmma_lane_map_load_lds_fragment(lds, lane, i + 4u);
        }
        asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    } else {
#pragma unroll
        for (int frag = 0; frag < 4; ++frag) {
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                a[frag][i] = static_cast<_Float16>(1.0f);
                b[frag][i] = static_cast<_Float16>(1.0f);
            }
        }
    }

#pragma unroll
    for (int tile = 0; tile < 16; ++tile) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            acc[tile][i] = static_cast<_Float16>(0.0f);
        }
    }

#pragma unroll
    for (int col = 0; col < 4; ++col) {
#pragma unroll
        for (int row = 0; row < 4; ++row) {
            const int tile = col * 4 + row;
            acc[tile] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(a[row], b[col], acc[tile], false);
        }
    }

#pragma unroll
    for (int tile = 0; tile < 16; ++tile) {
#pragma unroll
        for (int slot = 0; slot < 8; ++slot) {
            dst[wmma_lane_map_fulltile_index(static_cast<unsigned int>(tile), lane, static_cast<unsigned int>(slot))] =
                static_cast<float>(acc[tile][slot]);
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

static bool close_enough(float actual, float expected) {
    return std::fabs(actual - expected) <= 0.01f;
}

static void compare_outputs(const std::vector<float> & actual) {
    size_t bad_count = 0;
    size_t first_bad = 0;
    float first_expected = 0.0f;
    unsigned int coord_counts[HRX_WMMA_LANE_MAP_OPSELS][HRX_WMMA_LANE_MAP_D_ROWS][HRX_WMMA_LANE_MAP_D_COLS] = {};
    unsigned int coord_bad = 0;

    for (unsigned int opsel = 0; opsel < HRX_WMMA_LANE_MAP_OPSELS; ++opsel) {
        for (unsigned int lane = 0; lane < HRX_WMMA_LANE_MAP_LANES; ++lane) {
            for (unsigned int slot = 0; slot < HRX_WMMA_LANE_MAP_ACC_SLOTS; ++slot) {
                const float sentinel = wmma_lane_map_sentinel(lane, slot);
                const bool selected = (slot & 1u) == opsel;
                const float expected = selected ? sentinel + HRX_WMMA_LANE_MAP_DOT : sentinel;
                const size_t idx = wmma_lane_map_index(opsel, lane, slot);
                if (selected) {
                    const unsigned int row = wmma_lane_map_d_row(lane, slot);
                    const unsigned int col = wmma_lane_map_d_col(lane);
                    if (row < HRX_WMMA_LANE_MAP_D_ROWS && col < HRX_WMMA_LANE_MAP_D_COLS) {
                        ++coord_counts[opsel][row][col];
                    } else {
                        ++coord_bad;
                    }
                }
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

    for (unsigned int opsel = 0; opsel < HRX_WMMA_LANE_MAP_OPSELS; ++opsel) {
        for (unsigned int row = 0; row < HRX_WMMA_LANE_MAP_D_ROWS; ++row) {
            for (unsigned int col = 0; col < HRX_WMMA_LANE_MAP_D_COLS; ++col) {
                if (coord_counts[opsel][row][col] != 1u) {
                    ++coord_bad;
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
    std::printf(" coord_bad=%u", coord_bad);
    std::printf("\n");

    if (bad_count != 0 || coord_bad != 0) {
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

static void print_coord_map(const std::vector<float> & actual) {
    std::printf("coord_map_source=amd-matrix-instruction-calculator arch=gfx1151 instruction=v_wmma_f16_16x16x16_f16 wavefront=64 D-matrix matrix-layout\n");
    std::printf("coord_map_note=opsel0_updates_low_half_even_slots opsel4_updates_high_half_odd_slots same_D_coordinates\n");
    std::printf("coord_map_csv_begin\n");
    std::printf("opsel_bit,opsel_field,lane,slot,vgpr,half,row,col,actual,expected\n");
    for (unsigned int opsel = 0; opsel < HRX_WMMA_LANE_MAP_OPSELS; ++opsel) {
        for (unsigned int lane = 0; lane < HRX_WMMA_LANE_MAP_LANES; ++lane) {
            for (unsigned int slot = 0; slot < HRX_WMMA_LANE_MAP_ACC_SLOTS; ++slot) {
                if ((slot & 1u) != opsel) {
                    continue;
                }
                const size_t idx = wmma_lane_map_index(opsel, lane, slot);
                const float expected = wmma_lane_map_sentinel(lane, slot) + HRX_WMMA_LANE_MAP_DOT;
                std::printf("%u,%u,%u,%u,%u,%s,%u,%u,%.6f,%.6f\n",
                    opsel,
                    opsel == 0u ? 0u : 4u,
                    lane,
                    slot,
                    slot >> 1u,
                    (slot & 1u) == 0u ? "lo" : "hi",
                    wmma_lane_map_d_row(lane, slot),
                    wmma_lane_map_d_col(lane),
                    static_cast<double>(actual[idx]),
                    static_cast<double>(expected));
            }
        }
    }
    std::printf("coord_map_csv_end\n");
}

static bool is_nan(float value) {
    return !(value == value);
}

static int run_basic_lane_map() {
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
    print_coord_map(h_out);
    compare_outputs(h_out);
    return 0;
}

template <bool use_lds_fragments>
static int run_fulltile_probe(const char * mode) {
    const size_t count = 16u * HRX_WMMA_LANE_MAP_LANES * HRX_WMMA_LANE_MAP_ACC_SLOTS;
    device_buffer<float> d_out(count);
    std::vector<float> h_out(count, 0.0f);

    HIP_CHECK(hipMemset(d_out.ptr, 0, count * sizeof(float)));
    hipLaunchKernelGGL((wmma_f16_fulltile_probe<use_lds_fragments>), dim3(1), dim3(64), 0, 0, d_out.ptr);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_out.data(), d_out.ptr, count * sizeof(float), hipMemcpyDeviceToHost));

    size_t nan_count = 0;
    size_t nan_even_slots = 0;
    size_t nan_odd_slots = 0;
    size_t bad_ones_count = 0;
    size_t first_nan = count;
    size_t first_bad = count;
    for (unsigned int tile = 0; tile < 16u; ++tile) {
        for (unsigned int lane = 0; lane < HRX_WMMA_LANE_MAP_LANES; ++lane) {
            for (unsigned int slot = 0; slot < HRX_WMMA_LANE_MAP_ACC_SLOTS; ++slot) {
                const size_t idx = wmma_lane_map_fulltile_index(tile, lane, slot);
                const float value = h_out[idx];
                if (is_nan(value)) {
                    if (first_nan == count) {
                        first_nan = idx;
                    }
                    if ((slot & 1u) == 0u) {
                        ++nan_even_slots;
                    } else {
                        ++nan_odd_slots;
                    }
                    ++nan_count;
                    continue;
                }
                if constexpr (!use_lds_fragments) {
                    const float expected = (slot & 1u) == 0u ? HRX_WMMA_LANE_MAP_DOT : 0.0f;
                    if (!close_enough(value, expected)) {
                        if (first_bad == count) {
                            first_bad = idx;
                        }
                        ++bad_ones_count;
                    }
                }
            }
        }
    }

    std::printf("wmma-f16-fulltile mode=%s elements=%zu nan=%zu nan_even=%zu nan_odd=%zu",
        mode, h_out.size(), nan_count, nan_even_slots, nan_odd_slots);
    if (first_nan != count) {
        const unsigned int slot = first_nan % HRX_WMMA_LANE_MAP_ACC_SLOTS;
        const unsigned int lane = (first_nan / HRX_WMMA_LANE_MAP_ACC_SLOTS) % HRX_WMMA_LANE_MAP_LANES;
        const unsigned int tile = first_nan / (HRX_WMMA_LANE_MAP_ACC_SLOTS * HRX_WMMA_LANE_MAP_LANES);
        std::printf(" first_nan_tile=%u first_nan_lane=%u first_nan_slot=%u",
            tile, lane, slot);
    }
    if constexpr (!use_lds_fragments) {
        std::printf(" bad_ones=%zu", bad_ones_count);
        if (first_bad != count) {
            const unsigned int slot = first_bad % HRX_WMMA_LANE_MAP_ACC_SLOTS;
            const unsigned int lane = (first_bad / HRX_WMMA_LANE_MAP_ACC_SLOTS) % HRX_WMMA_LANE_MAP_LANES;
            const unsigned int tile = first_bad / (HRX_WMMA_LANE_MAP_ACC_SLOTS * HRX_WMMA_LANE_MAP_LANES);
            const float expected = (slot & 1u) == 0u ? HRX_WMMA_LANE_MAP_DOT : 0.0f;
            std::printf(" first_bad_tile=%u first_bad_lane=%u first_bad_slot=%u actual=%f expected=%f",
                tile, lane, slot, static_cast<double>(h_out[first_bad]), static_cast<double>(expected));
        }
    }
    std::printf("\n");

    if (nan_count != 0 || bad_ones_count != 0) {
        return 1;
    }
    return 0;
}

int main(int argc, char ** argv) {
    std::string mode = "basic";
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], "--mode=", 7) == 0) {
            mode = argv[i] + 7;
        } else {
            std::fprintf(stderr, "usage: %s [--mode=basic|fulltile-ones|fulltile-lds|all]\n", argv[0]);
            return 2;
        }
    }

    if (mode == "basic") {
        return run_basic_lane_map();
    }
    if (mode == "fulltile-ones") {
        return run_fulltile_probe<false>(mode.c_str());
    }
    if (mode == "fulltile-lds") {
        return run_fulltile_probe<true>(mode.c_str());
    }
    if (mode == "all") {
        int status = run_basic_lane_map();
        status |= run_fulltile_probe<false>("fulltile-ones");
        status |= run_fulltile_probe<true>("fulltile-lds");
        return status;
    }

    std::fprintf(stderr, "unknown mode: %s\n", mode.c_str());
    return 2;
}
