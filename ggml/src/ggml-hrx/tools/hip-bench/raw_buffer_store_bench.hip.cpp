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

static __device__ __forceinline__ float raw_store_probe_value(long long row, long long col) {
    const int bits = static_cast<int>((row * 17 + col * 131) & 0x7fff);
    return static_cast<float>(bits - 16384) * 0.001953125f;
}

static __device__ __forceinline__ __amdgpu_buffer_rsrc_t raw_store_probe_make_rsrc(
        float * dst,
        unsigned long long extent,
        unsigned int flags) {
    return __builtin_amdgcn_make_buffer_rsrc(
        dst,
        static_cast<unsigned short>(0),
        extent,
        static_cast<int>(flags));
}

static __device__ __forceinline__ __amdgpu_buffer_rsrc_t raw_store_probe_manual_rsrc(
        float * dst,
        unsigned int extent,
        unsigned int flags) {
    const unsigned long long ptr = reinterpret_cast<unsigned long long>(dst);
    unsigned int words[4] = {
        static_cast<unsigned int>(ptr),
        static_cast<unsigned int>(ptr >> 32),
        extent,
        flags,
    };
    return *reinterpret_cast<__amdgpu_buffer_rsrc_t *>(words);
}

extern "C" __global__ void raw_store_probe_global_store(
        float * dst,
        long long rows,
        long long cols) {
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long n = rows * cols;
    if (idx >= n) {
        return;
    }
    const long long row = idx % rows;
    const long long col = idx / rows;
    dst[col * rows + row] = raw_store_probe_value(row, col);
}

extern "C" __global__ void raw_store_probe_make_store(
        float * dst,
        long long rows,
        long long cols,
        unsigned long long extent,
        unsigned int flags) {
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long n = rows * cols;
    if (idx >= n) {
        return;
    }
    const long long row = idx % rows;
    const long long col = idx / rows;
    const float value = raw_store_probe_value(row, col);
    const int byte_offset = static_cast<int>((col * rows + row) * static_cast<long long>(sizeof(float)));
    const __amdgpu_buffer_rsrc_t rsrc = raw_store_probe_make_rsrc(dst, extent, flags);
    __builtin_amdgcn_raw_buffer_store_b32(__builtin_bit_cast(int, value), rsrc, byte_offset, 0, 0);
}

extern "C" __global__ void raw_store_probe_manual_store(
        float * dst,
        long long rows,
        long long cols,
        unsigned int extent,
        unsigned int flags) {
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long n = rows * cols;
    if (idx >= n) {
        return;
    }
    const long long row = idx % rows;
    const long long col = idx / rows;
    const float value = raw_store_probe_value(row, col);
    const int byte_offset = static_cast<int>((col * rows + row) * static_cast<long long>(sizeof(float)));
    const __amdgpu_buffer_rsrc_t rsrc = raw_store_probe_manual_rsrc(dst, extent, flags);
    __builtin_amdgcn_raw_buffer_store_b32(__builtin_bit_cast(int, value), rsrc, byte_offset, 0, 0);
}

struct options {
    long long rows = 14336;
    long long cols = 512;
    std::string mode = "make-max";
    unsigned int flags = 0x31004000u;
};

static long long parse_ll(const char * value) {
    char * end = nullptr;
    const long long parsed = std::strtoll(value, &end, 0);
    if (end == value || *end != '\0') {
        std::fprintf(stderr, "invalid integer: %s\n", value);
        std::exit(2);
    }
    return parsed;
}

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
        if (std::strncmp(argv[i], "--rows=", 7) == 0) {
            opts.rows = std::max(1LL, parse_ll(argv[i] + 7));
        } else if (std::strncmp(argv[i], "--cols=", 7) == 0) {
            opts.cols = std::max(1LL, parse_ll(argv[i] + 7));
        } else if (std::strncmp(argv[i], "--mode=", 7) == 0) {
            opts.mode = argv[i] + 7;
        } else if (std::strncmp(argv[i], "--flags=", 8) == 0) {
            opts.flags = parse_u32(argv[i] + 8);
        } else {
            std::fprintf(stderr,
                "usage: %s [--rows=N] [--cols=N] [--mode=make-max|make-bytes|manual-max|manual-bytes] [--flags=0x31004000]\n",
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

static void compare_outputs(const std::vector<float> & actual, const std::vector<float> & expected) {
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

    std::printf("check: elements=%zu bad=%zu max_abs=%g", actual.size(), bad_count, max_abs);
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
    const size_t count = static_cast<size_t>(opts.rows) * static_cast<size_t>(opts.cols);
    const unsigned long long byte_extent = static_cast<unsigned long long>(count * sizeof(float));
    if (byte_extent > 0xffffffffull) {
        std::fprintf(stderr, "byte extent exceeds raw buffer i32 offset range: %llu\n", byte_extent);
        return 2;
    }

    device_buffer<float> d_ref(count);
    device_buffer<float> d_raw(count);
    std::vector<float> h_ref(count);
    std::vector<float> h_raw(count);

    const int block = 256;
    const int grid = static_cast<int>((count + block - 1) / block);
    hipLaunchKernelGGL(raw_store_probe_global_store, dim3(grid), dim3(block), 0, 0, d_ref.ptr, opts.rows, opts.cols);
    HIP_CHECK(hipGetLastError());

    if (opts.mode == "make-max") {
        hipLaunchKernelGGL(raw_store_probe_make_store, dim3(grid), dim3(block), 0, 0,
            d_raw.ptr, opts.rows, opts.cols, 0xffffffffull, opts.flags);
    } else if (opts.mode == "make-bytes") {
        hipLaunchKernelGGL(raw_store_probe_make_store, dim3(grid), dim3(block), 0, 0,
            d_raw.ptr, opts.rows, opts.cols, byte_extent, opts.flags);
    } else if (opts.mode == "manual-max") {
        hipLaunchKernelGGL(raw_store_probe_manual_store, dim3(grid), dim3(block), 0, 0,
            d_raw.ptr, opts.rows, opts.cols, 0xffffffffu, opts.flags);
    } else if (opts.mode == "manual-bytes") {
        hipLaunchKernelGGL(raw_store_probe_manual_store, dim3(grid), dim3(block), 0, 0,
            d_raw.ptr, opts.rows, opts.cols, static_cast<unsigned int>(byte_extent), opts.flags);
    } else {
        std::fprintf(stderr, "unknown mode: %s\n", opts.mode.c_str());
        return 2;
    }
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(h_ref.data(), d_ref.ptr, count * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_raw.data(), d_raw.ptr, count * sizeof(float), hipMemcpyDeviceToHost));

    std::printf("raw-buffer-store rows=%lld cols=%lld mode=%s flags=0x%x bytes=%llu\n",
        opts.rows, opts.cols, opts.mode.c_str(), opts.flags, byte_extent);
    compare_outputs(h_raw, h_ref);
    return 0;
}
