#include <hip/hip_runtime.h>

#include <algorithm>
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

static __host__ __device__ __forceinline__ uint16_t lds_halfword_probe_value(
        unsigned int tile,
        unsigned int lane,
        unsigned int reg) {
    return static_cast<uint16_t>(0x1000u + ((tile & 0x1fu) << 10) + ((lane & 0x3fu) << 4) + (reg & 0x0fu));
}

static __device__ __forceinline__ void lds_halfword_probe_ds_store_u16(
        __attribute__((address_space(3))) uint16_t * ptr,
        uint32_t value) {
    asm volatile("ds_write_b16 %0, %1 offset:0\n"
                 :
                 : "v"(ptr), "v"(value)
                 : "memory");
}

static __device__ __forceinline__ uint32_t lds_halfword_probe_ds_load_u16_d16(
        const __attribute__((address_space(3))) uint16_t * ptr) {
    uint32_t value = 0;
    asm volatile("ds_read_u16_d16 %0, %1 offset:0\n"
                 : "=v"(value)
                 : "v"(ptr)
                 : "memory");
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    return value;
}

extern "C" __global__ __launch_bounds__(256, 1)
void lds_halfword_stage_probe_typed(uint16_t * dst, unsigned int tiles) {
    __shared__ volatile uint16_t sh[8 * 16 * 16];
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int wave = tid >> 6;
    const unsigned int lane = tid & 63u;
    const unsigned int row_lane = lane >> 4;
    const unsigned int col_lane = lane & 15u;

    for (unsigned int tile = 0; tile < tiles; ++tile) {
        const unsigned int tile_base = wave * 16u * 16u;
        const unsigned int col_major_base = tile_base + col_lane * 16u + row_lane;
#pragma unroll
        for (unsigned int reg = 0; reg < 4; ++reg) {
            sh[col_major_base + reg * 4u] = lds_halfword_probe_value(tile, lane, reg);
        }
        __syncthreads();

#pragma unroll
        for (unsigned int reg = 0; reg < 4; ++reg) {
            const unsigned int out_idx = (((tile * 256u + wave * 64u + lane) * 4u) + reg);
            dst[out_idx] = sh[col_major_base + reg * 4u];
        }
        __syncthreads();
    }
}

extern "C" __global__ __launch_bounds__(256, 1)
void lds_halfword_stage_probe_asm(uint16_t * dst, unsigned int tiles) {
    __shared__ uint16_t sh[8 * 16 * 16];
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int wave = tid >> 6;
    const unsigned int lane = tid & 63u;
    const unsigned int row_lane = lane >> 4;
    const unsigned int col_lane = lane & 15u;

    for (unsigned int tile = 0; tile < tiles; ++tile) {
        const unsigned int tile_base = wave * 16u * 16u;
        const unsigned int col_major_base = tile_base + col_lane * 16u + row_lane;
#pragma unroll
        for (unsigned int reg = 0; reg < 4; ++reg) {
            __attribute__((address_space(3))) uint16_t * ptr =
                (__attribute__((address_space(3))) uint16_t *) (sh + col_major_base + reg * 4u);
            lds_halfword_probe_ds_store_u16(ptr, lds_halfword_probe_value(tile, lane, reg));
        }
        asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");

#pragma unroll
        for (unsigned int reg = 0; reg < 4; ++reg) {
            const __attribute__((address_space(3))) uint16_t * ptr =
                (const __attribute__((address_space(3))) uint16_t *) (sh + col_major_base + reg * 4u);
            const uint32_t value = lds_halfword_probe_ds_load_u16_d16(ptr);
            const unsigned int out_idx = (((tile * 256u + wave * 64u + lane) * 4u) + reg);
            dst[out_idx] = static_cast<uint16_t>(value);
        }
        asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    }
}

struct options {
    unsigned int tiles = 16;
    std::string mode = "asm";
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
        if (std::strncmp(argv[i], "--tiles=", 8) == 0) {
            opts.tiles = std::max(1u, parse_u32(argv[i] + 8));
        } else if (std::strncmp(argv[i], "--mode=", 7) == 0) {
            opts.mode = argv[i] + 7;
        } else {
            std::fprintf(stderr, "usage: %s [--tiles=N] [--mode=typed|asm]\n", argv[0]);
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

static void compare_outputs(const std::vector<uint16_t> & actual, unsigned int tiles) {
    size_t bad_count = 0;
    size_t first_bad = 0;
    uint16_t first_expected = 0;
    for (unsigned int tile = 0; tile < tiles; ++tile) {
        for (unsigned int lane = 0; lane < 256; ++lane) {
            for (unsigned int reg = 0; reg < 4; ++reg) {
                const size_t idx = static_cast<size_t>((tile * 256u + lane) * 4u + reg);
                const uint16_t expected = lds_halfword_probe_value(tile, lane & 63u, reg);
                if (actual[idx] != expected) {
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
        std::printf(" first_bad=%zu actual=0x%04x expected=0x%04x",
            first_bad,
            static_cast<unsigned int>(actual[first_bad]),
            static_cast<unsigned int>(first_expected));
    }
    std::printf("\n");

    if (bad_count != 0) {
        std::exit(1);
    }
}

int main(int argc, char ** argv) {
    const options opts = parse_options(argc, argv);
    const size_t count = static_cast<size_t>(opts.tiles) * 256u * 4u;
    device_buffer<uint16_t> d_out(count);
    std::vector<uint16_t> h_out(count);

    HIP_CHECK(hipMemset(d_out.ptr, 0, count * sizeof(uint16_t)));
    if (opts.mode == "typed") {
        hipLaunchKernelGGL(lds_halfword_stage_probe_typed, dim3(1), dim3(256), 0, 0, d_out.ptr, opts.tiles);
    } else if (opts.mode == "asm") {
        hipLaunchKernelGGL(lds_halfword_stage_probe_asm, dim3(1), dim3(256), 0, 0, d_out.ptr, opts.tiles);
    } else {
        std::fprintf(stderr, "unknown mode: %s\n", opts.mode.c_str());
        return 2;
    }
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_out.data(), d_out.ptr, count * sizeof(uint16_t), hipMemcpyDeviceToHost));

    std::printf("lds-halfword-stage mode=%s tiles=%u bytes=%zu\n",
        opts.mode.c_str(), opts.tiles, count * sizeof(uint16_t));
    compare_outputs(h_out, opts.tiles);
    return 0;
}
