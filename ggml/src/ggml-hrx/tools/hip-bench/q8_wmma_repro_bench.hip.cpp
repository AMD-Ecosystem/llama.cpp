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

static __device__ __forceinline__ void q8_repro_consume_frag(
        hrx_q8_0_wmma_vk128_half16_vec frag) {
    asm volatile("" :: "v"(frag[0]), "v"(frag[4]), "v"(frag[8]), "v"(frag[12]) : "memory");
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

template <int group_base, int col_start, int col_count, int active_groups, bool consume_unused_b>
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
                        acc[local] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(
                            a_frag[k_tile][row_sub],
                            b_frag[k_tile][col_sub],
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
            for (int slot = 0; slot < 4; ++slot) {
                q8_repro_raw_store_slot(dst_rsrc, rows, row_base, col_base, rows, cols, acc, local, group, slot, lane);
            }
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

static bool output_is_active(size_t index, int rows, const std::string & mode) {
    const int row = static_cast<int>(index % static_cast<size_t>(rows));
    const int col = static_cast<int>(index / static_cast<size_t>(rows));
    const int row_local = row & 63;
    const int col_local = col & 63;
    const int group = (col_local >> 4) * 4 + (row_local >> 4);
    if (mode == "array8-fullb-2phase" || mode == "batched4" ||
            mode == "array8-fullb-2phase-consume" || mode == "batched4-consume") {
        return true;
    }
    return group < 8;
}

struct group_stats {
    size_t active = 0;
    size_t bad = 0;
    size_t nan = 0;
    size_t inf = 0;
    size_t sentinel = 0;
    float max_abs = 0.0f;
};

static int output_group(size_t index, int rows) {
    const int row = static_cast<int>(index % static_cast<size_t>(rows));
    const int col = static_cast<int>(index / static_cast<size_t>(rows));
    const int row_local = row & 63;
    const int col_local = col & 63;
    return (col_local >> 4) * 4 + (row_local >> 4);
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
        }
        if (std::isnan(actual)) {
            ++nan;
            ++bad;
            ++gs.nan;
            ++gs.bad;
            continue;
        }
        if (std::isinf(actual)) {
            ++inf;
            ++bad;
            ++gs.inf;
            ++gs.bad;
            continue;
        }
        const float err = std::fabs(actual - ref[i]);
        max_abs = std::max(max_abs, err);
        gs.max_abs = std::max(gs.max_abs, err);
        if (err > 0.25f) {
            ++bad;
            ++gs.bad;
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
    }
    return (nan == 0 && inf == 0 && sentinel == 0) ? 0 : 1;
}

int main(int argc, char ** argv) {
    std::string mode = "all";
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            mode = argv[++i];
        } else {
            std::fprintf(stderr, "usage: %s [--mode array8-fullb|array8-b2|array8-fullb-2phase|array8-fullb-2phase-consume|batched4|batched4-consume|all]\n", argv[0]);
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
    if (mode == "all" || mode == "batched4") {
        status |= run_case("batched4", rows, 64, k);
        status |= run_case("batched4", rows, 33, k);
    }
    if (mode == "all" || mode == "batched4-consume") {
        status |= run_case("batched4-consume", rows, 64, k);
        status |= run_case("batched4-consume", rows, 33, k);
    }
    return status;
}
