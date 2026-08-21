#include "common.cuh"
#include "mmq.cuh"
#include "quantize.cuh"
#include "mmid.cuh"

#include <cstdint>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#if defined(GGML_RDNA35_Q4K_ADAPTIVE_Q4_ACTIVATION)
static constexpr float adaptive_q4_energy_epsilon = 1.0e-20f;

static float get_adaptive_q4_score_limit() {
    const char * value = std::getenv("GGML_CUDA_Q4_ACTIVATION_NMSE_LIMIT");
    if (value == nullptr || value[0] == '\0') {
        return GGML_RDNA35_Q4K_NMSE_LIMIT;
    }

    char * end = nullptr;
    const float limit = std::strtof(value, &end);
    return end != value && std::isfinite(limit) && limit >= 0.0f ? limit : GGML_RDNA35_Q4K_NMSE_LIMIT;
}

static __device__ __forceinline__ void get_q4_k_scale_min(
        const int group, const uint8_t * scales, uint8_t & scale, uint8_t & minimum) {
    if (group < 4) {
        scale = scales[group] & 63;
        minimum = scales[group + 4] & 63;
    } else {
        scale = (scales[group + 4] & 0x0F) | ((scales[group - 4] >> 6) << 4);
        minimum = (scales[group + 4] >> 4) | ((scales[group] >> 6) << 4);
    }
}

static __global__ void compute_q4_k_weight_features(
        const block_q4_K * __restrict__ weights, float * __restrict__ tile_energies,
        float * __restrict__ matrix_energies, const int nrows, const int ncols,
        const int n_m_tiles, const int n_k_windows,
        const int64_t row_stride, const int64_t channel_stride, const int64_t sample_stride,
        const int nchannels) {
    const int k_block = blockIdx.x;
    const int m_tile = blockIdx.y;
    const int channel = blockIdx.z % nchannels;
    const int sample = blockIdx.z / nchannels;
    const int64_t plane_offset = int64_t(channel)*channel_stride + int64_t(sample)*sample_stride;
    const int row = 128*m_tile + threadIdx.x;

    float weight_energy[8] = {};
    if (row < nrows) {
        const block_q4_K & block = weights[plane_offset + int64_t(row)*row_stride + k_block];
        const float base_scale = __low2half(block.dm);
        const float base_minimum = __high2half(block.dm);
#pragma unroll
        for (int group = 0; group < 8; ++group) {
            uint8_t scale;
            uint8_t minimum;
            get_q4_k_scale_min(group, block.scales, scale, minimum);
            const float d = base_scale*scale;
            const float m = base_minimum*minimum;
            weight_energy[group] = d*d + m*m;
        }
    }

    __shared__ float energy_reduction[8][128];
#pragma unroll
    for (int group = 0; group < 8; ++group) {
        energy_reduction[group][threadIdx.x] = weight_energy[group];
    }
    __syncthreads();
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
#pragma unroll
            for (int group = 0; group < 8; ++group) {
                energy_reduction[group][threadIdx.x] += energy_reduction[group][threadIdx.x + stride];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
#pragma unroll
        for (int group = 0; group < 8; ++group) {
            const int k_window = 8*k_block + group;
            const int feature_index =
                (blockIdx.z*n_m_tiles + m_tile)*n_k_windows + k_window;
            tile_energies[feature_index] = energy_reduction[group][0];
            atomicAdd(&matrix_energies[blockIdx.z], energy_reduction[group][0]);
        }
    }
    GGML_UNUSED(ncols);
}

static __global__ void compute_adaptive_q4_fragment_flags(
        const float * __restrict__ values, uint32_t * __restrict__ flags,
        const float * __restrict__ weight_tile_energies, const float * __restrict__ weight_matrix_energies,
        const int64_t ncols, const int nrows,
        const int64_t row_stride, const int64_t channel_stride, const int64_t sample_stride,
        const int nchannels, const int weight_nrows, const int weight_ncols,
        const int n_m_tiles, const int n_k_windows, const int n_weight_channels,
        const int n_weight_samples, const float score_limit) {
    GGML_UNUSED(weight_ncols);
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    const int k0 = 32*blockIdx.x;
    const int channel = blockIdx.z % nchannels;
    const int sample = blockIdx.z / nchannels;
    const int64_t plane_offset = int64_t(channel)*channel_stride + int64_t(sample)*sample_stride;
    const int n_groups = (nrows + 15)/16;

    __shared__ float window_error[16];
    __shared__ float window_energy[16];
    __shared__ float unified_nmse;

    if (threadIdx.x == 0) {
        unified_nmse = 0.0f;
    }
    __syncthreads();

    for (int n_group = 0; n_group < n_groups; ++n_group) {
        const int row0 = 16*n_group;
#pragma unroll
        for (int row_phase = 0; row_phase < 2; ++row_phase) {
            const int row = row0 + row_phase*8 + warp;
            float value = 0.0f;
            if (row < nrows && k0 + lane < ncols) {
                value = values[plane_offset + int64_t(row)*row_stride + k0 + lane];
            }

            float amax = fabsf(value);
#pragma unroll
            for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
                amax = fmaxf(amax, __shfl_down_sync(0xFFFFFFFFFFFFFFFFull, amax, offset, WARP_SIZE));
            }
            amax = __shfl_sync(0xFFFFFFFFFFFFFFFFull, amax, 0, WARP_SIZE);

            float error = 0.0f;
            float energy = 0.0f;
            if (row < nrows && amax > 0.0f) {
                const float d8 = amax/127.0f;
                const float d4 = amax/7.0f;
                const int q8 = max(-127, min(127, __float2int_rn(value/d8)));
                const int q4 = max(-7, min(7, __float2int_rn(value/d4)));
                const float q8_value = q8*d8;
                const float delta = q8_value - q4*d4;
                error = delta*delta;
                energy = q8_value*q8_value;
            }
#pragma unroll
            for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
                error += __shfl_down_sync(0xFFFFFFFFFFFFFFFFull, error, offset, WARP_SIZE);
                energy += __shfl_down_sync(0xFFFFFFFFFFFFFFFFull, energy, offset, WARP_SIZE);
            }
            if (lane == 0) {
                window_error[row_phase*8 + warp] = error;
                window_energy[row_phase*8 + warp] = energy;
            }
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            float group_error = 0.0f;
            float group_energy = 0.0f;
#pragma unroll
            for (int row = 0; row < 16; ++row) {
                group_error += window_error[row];
                group_energy += window_energy[row];
            }
            const float group_nmse = group_error/fmaxf(group_energy, adaptive_q4_energy_epsilon);
            unified_nmse = fmaxf(unified_nmse, group_nmse);
        }
        __syncthreads();
    }

    const float activation_nmse = unified_nmse;
    if (threadIdx.x < n_m_tiles) {
        const int weight_channel = channel % n_weight_channels;
        const int weight_sample = sample % n_weight_samples;
        const int weight_plane = weight_sample*n_weight_channels + weight_channel;
        const float matrix_energy = weight_matrix_energies[weight_plane];
        const float matrix_mean =
            matrix_energy/fmaxf(float(weight_nrows)*n_k_windows, 1.0f);
        const int tile_rows = min(128, weight_nrows);

        const int m_tile = threadIdx.x;
        const int rows_in_tile = min(tile_rows, weight_nrows - 128*m_tile);
        const int feature_index = (weight_plane*n_m_tiles + m_tile)*n_k_windows + blockIdx.x;
        const float tile_mean =
            weight_tile_energies[feature_index]/fmaxf(float(rows_in_tile), 1.0f);
        const float relative_weight_rms = matrix_mean > 0.0f ? sqrtf(tile_mean/matrix_mean) : 0.0f;
        const bool selected = relative_weight_rms*activation_nmse <= score_limit;
        if (selected) {
            for (int n_group = 0; n_group < n_groups; ++n_group) {
                const int flag =
                    ((blockIdx.z*n_m_tiles + m_tile)*n_groups + n_group)*n_k_windows + blockIdx.x;
                atomicOr(&flags[flag/32], 1U << (flag%32));
            }
        }
    }
}

static __global__ void coarsen_adaptive_q4_flags_k256(
        uint32_t * __restrict__ flags, const int n_flag_words) {
    const int word = blockIdx.x*blockDim.x + threadIdx.x;
    if (word >= n_flag_words) {
        return;
    }

    const uint32_t v = flags[word];
    uint32_t out = 0;
#pragma unroll
    for (int b = 0; b < 4; ++b) {
        const uint32_t bits = (v >> (8*b)) & 0xFFu;
        if (bits == 0xFFu) {
            out |= 0xFFu << (8*b);
        }
    }
    flags[word] = out;
}

static __global__ void count_adaptive_q4_fragments(
        const uint32_t * __restrict__ flags, unsigned int * __restrict__ any_selected,
        unsigned long long * __restrict__ counts, const int n_flags) {
    const int fragment = blockIdx.x*blockDim.x + threadIdx.x;
    if (fragment >= n_flags) {
        return;
    }

    const bool selected = (flags[fragment/32] & (1U << (fragment%32))) != 0;
    atomicOr(any_selected, selected ? 1U : 2U);
    if (counts != nullptr) {
        atomicAdd(&counts[0], 1ULL);
        if (selected) {
            atomicAdd(&counts[1], 1ULL);
        }
    }
}

static __global__ void force_adaptive_q4_k256_flags(
        uint32_t * __restrict__ flags, unsigned int * __restrict__ any_selected,
        const int n_flag_words, const int q4_percent, const int n_k_windows) {
    const int word = blockIdx.x*blockDim.x + threadIdx.x;
    if (threadIdx.x == 0 && blockIdx.x == 0 && any_selected != nullptr) {
        unsigned any = 2u;
        if (q4_percent >= 100) {
            any = 1u;
        } else if (q4_percent > 0) {
            any = 3u;
        }
        *any_selected = any;
    }
    if (word >= n_flag_words) {
        return;
    }

    uint32_t out = 0;
    const int n_k_blocks = n_k_windows / 8;
    if (n_k_blocks > 0 && q4_percent > 0 && n_k_windows > 0) {
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            const int k_window = (word*32 + b*8) % n_k_windows;
            const int k_block = k_window / 8;
            if (int64_t(k_block)*100 < int64_t(q4_percent)*n_k_blocks) {
                out |= 0xFFu << (8*b);
            }
        }
    }
    flags[word] = out;
}
#endif

static void ggml_cuda_mul_mat_q_switch_type(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) {
    switch (args.type_x) {
        case GGML_TYPE_Q1_0:
            mul_mat_q_case<GGML_TYPE_Q1_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_0:
            mul_mat_q_case<GGML_TYPE_Q4_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_1:
            mul_mat_q_case<GGML_TYPE_Q4_1>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_q_case<GGML_TYPE_Q5_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_q_case<GGML_TYPE_Q5_1>(ctx, args, stream);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_q_case<GGML_TYPE_Q8_0>(ctx, args, stream);
            break;
// -----------------------------------------------------------------------
        case GGML_TYPE_Q2_K:
            mul_mat_q_case<GGML_TYPE_Q2_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_q_case<GGML_TYPE_Q3_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_q_case<GGML_TYPE_Q4_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_q_case<GGML_TYPE_Q5_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_q_case<GGML_TYPE_Q6_K>(ctx, args, stream);
            break;
// -----------------------------------------------------------------------
        case GGML_TYPE_IQ1_S:
            mul_mat_q_case<GGML_TYPE_IQ1_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            mul_mat_q_case<GGML_TYPE_IQ2_XXS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            mul_mat_q_case<GGML_TYPE_IQ2_XS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_S:
            mul_mat_q_case<GGML_TYPE_IQ2_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            mul_mat_q_case<GGML_TYPE_IQ3_XXS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_S:
            mul_mat_q_case<GGML_TYPE_IQ3_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_q_case<GGML_TYPE_IQ4_XS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            mul_mat_q_case<GGML_TYPE_IQ4_NL>(ctx, args, stream);
            break;
// -----------------------------------------------------------------------
        case GGML_TYPE_MXFP4:
            mul_mat_q_case<GGML_TYPE_MXFP4>(ctx, args, stream);
            break;
        case GGML_TYPE_NVFP4:
            mul_mat_q_case<GGML_TYPE_NVFP4>(ctx, args, stream);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

void ggml_cuda_mul_mat_q(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst) {
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(        dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(!ids || ids->type  == GGML_TYPE_I32); // Optional, used for batched GGML_MUL_MAT_ID.

    GGML_TENSOR_BINARY_OP_LOCALS;

    cudaStream_t stream = ctx.stream();
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(        nb00       == ts_src0);
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(        nb0        == ts_dst);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));

    const char  * src0_d = (const char  *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float       *  dst_d = (float       *)  dst->data;

    // If src0 is a temporary compute buffer, clear any potential padding.
    if (ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        const size_t size_data  = ggml_nbytes(src0);
        const size_t size_alloc = ggml_backend_buffer_get_alloc_size(src0->buffer, src0);
        if (size_alloc > size_data) {
            GGML_ASSERT(ggml_is_contiguously_allocated(src0));
            GGML_ASSERT(!src0->view_src);
            CUDA_CHECK(cudaMemsetAsync((char *) src0->data + size_data, 0, size_alloc - size_data, stream));
        }
    }

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    const bool fallback = ne01 % 128 != 0;

    const bool use_native_fp4 = blackwell_mma_available(cc) && (src0->type == GGML_TYPE_MXFP4 || src0->type == GGML_TYPE_NVFP4);
#if defined(GGML_RDNA35_Q4K_Q4_ACTIVATION_BENCH)
    const bool use_q4_activation = GGML_CUDA_CC_IS_RDNA3_5(cc) && src0->type == GGML_TYPE_Q4_K && !ids;
#else
    const bool use_q4_activation = false;
#endif
#if defined(GGML_RDNA35_Q4K_ADAPTIVE_Q4_ACTIVATION)
    static constexpr int64_t adaptive_q4_min_rows = 64;
    const bool use_adaptive_q4 =
        GGML_CUDA_CC_IS_RDNA3_5(cc) && src0->type == GGML_TYPE_Q4_K && ne11 >= adaptive_q4_min_rows && !ids;
    const char * force_mix_env = use_adaptive_q4 ? std::getenv("GGML_CUDA_Q4_ACTIVATION_FORCE_MIX") : nullptr;
    int force_q4_percent = 70;
    unsigned host_q4_any = 0;
    int kb0_q4_end = -1;
    if (force_mix_env != nullptr) {
        const char * force_percent = std::getenv("GGML_CUDA_Q4_ACTIVATION_FORCE_MIX_PERCENT");
        force_q4_percent = force_percent != nullptr ?
            max(0, min(100, std::atoi(force_percent))) : 70;
        host_q4_any = force_q4_percent >= 100 ? 1u : (force_q4_percent > 0 ? 3u : 2u);
        const int n_k256 = int(ne00 / 256);
        kb0_q4_end = 0;
        while (kb0_q4_end < n_k256 &&
               int64_t(kb0_q4_end)*100 < int64_t(force_q4_percent)*n_k256) {
            kb0_q4_end++;
        }
    }
#else
    const bool use_adaptive_q4 = false;
    const char * force_mix_env = nullptr;
    const int force_q4_percent = 70;
    const unsigned host_q4_any = 0;
    const int kb0_q4_end = -1;
#endif
    const size_t y_block_size       = use_native_fp4 ? sizeof(block_fp4_mmq) :
                                      use_q4_activation ? sizeof(block_q4_1_mmq) : sizeof(block_q8_1_mmq);
    const size_t y_values_per_block = use_native_fp4 ? QK_FP4_MMQ            : QK8_1_MMQ;

    if (!ids) {
        const size_t nbytes_src1_q8_1 = ne13*ne12 * ne11*ne10_padded * y_block_size/y_values_per_block +
            ggml_cuda_mmq_get_J_max(src0->type, fallback, cc, ne11) * sizeof(block_q8_1_mmq);
        ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_src1_q8_1);
        ggml_cuda_pool_alloc<char> src1_q4_1(ctx.pool());
        ggml_cuda_pool_alloc<uint32_t> src1_q4_flags(ctx.pool());
        ggml_cuda_pool_alloc<unsigned int> src1_q4_any(ctx.pool());
        ggml_cuda_pool_alloc<float> weight_features(ctx.pool());
        ggml_cuda_pool_alloc<float> weight_feature_totals(ctx.pool());
        if (use_adaptive_q4) {
            const size_t n_q8_blocks = ne13*ne12*ne11*ne10_padded/QK8_1_MMQ;
            const int n_m_tiles = (ne01 + 127)/128;
            const int n_n_tiles = (ne11 + 15)/16;
            const int n_k_windows = ne10/32;
            const size_t n_q4_flags = size_t(ne12)*ne13*n_m_tiles*n_n_tiles*n_k_windows;
            const size_t n_q4_flag_words = (n_q4_flags + 31)/32;
            const size_t nbytes_src1_q4_1 =
                n_q8_blocks*sizeof(block_q4_1_mmq) +
                ggml_cuda_mmq_get_J_max(src0->type, fallback, cc, ne11)*sizeof(block_q4_1_mmq);
            src1_q4_1.alloc(nbytes_src1_q4_1);
            src1_q4_flags.alloc(n_q4_flag_words);
            src1_q4_any.alloc(1);
            if (force_mix_env == nullptr) {
                weight_features.alloc(size_t(ne02)*ne03*n_m_tiles*n_k_windows);
                weight_feature_totals.alloc(ne02*ne03);
                CUDA_CHECK(cudaMemsetAsync(weight_feature_totals.ptr, 0, ne02*ne03*sizeof(float), stream));
                CUDA_CHECK(cudaMemsetAsync(src1_q4_flags.ptr, 0, n_q4_flag_words*sizeof(uint32_t), stream));
                CUDA_CHECK(cudaMemsetAsync(src1_q4_any.ptr, 0, sizeof(unsigned int), stream));
            }
        }
        ggml_cuda_pool_alloc<float> src1_scale(ctx.pool());
        if (src0->type == GGML_TYPE_NVFP4 && use_native_fp4) {
            src1_scale.alloc(ne13*ne12*ne11);
        }

        {
            const int64_t s11 = src1->nb[1] / ts_src1;
            const int64_t s12 = src1->nb[2] / ts_src1;
            const int64_t s13 = src1->nb[3] / ts_src1;
            if (use_adaptive_q4) {
                const int n_m_tiles = (ne01 + 127)/128;
                const int n_k_windows = ne10/32;
                const int n_n_tiles = (ne11 + 15)/16;
                const int n_flags = ne12*ne13*n_m_tiles*n_n_tiles*n_k_windows;
                const int n_flag_words = (n_flags + 31)/32;
                if (force_mix_env == nullptr) {
                    const dim3 weight_feature_grid(n_k_windows/8, n_m_tiles, ne02*ne03);
                    compute_q4_k_weight_features<<<weight_feature_grid, 128, 0, stream>>>(
                        (const block_q4_K *) src0_d, weight_features.ptr, weight_feature_totals.ptr,
                        ne01, ne00, n_m_tiles, n_k_windows, s01, s02, s03, ne02);
                    const dim3 fragment_grid(n_k_windows, 1, ne12*ne13);
                    compute_adaptive_q4_fragment_flags<<<fragment_grid, 256, 0, stream>>>(
                        src1_d, src1_q4_flags.ptr, weight_features.ptr, weight_feature_totals.ptr,
                        ne10, ne11, s11, s12, s13, ne12, ne01, ne00, n_m_tiles, n_k_windows,
                        ne02, ne03, get_adaptive_q4_score_limit());
                    coarsen_adaptive_q4_flags_k256<<<(n_flag_words + 255)/256, 256, 0, stream>>>(
                        src1_q4_flags.ptr, n_flag_words);
                    count_adaptive_q4_fragments<<<(n_flags + 255)/256, 256, 0, stream>>>(
                        src1_q4_flags.ptr, src1_q4_any.ptr, nullptr, n_flags);
                }
                quantize_mmq_q8_q4_1_cuda(
                    src1_d, src1_q8_1.ptr, src1_q4_1.ptr, src1_q4_any.ptr,
                    ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream, host_q4_any);
            } else if (use_native_fp4) {
                static constexpr size_t align_float8 = 32;
                const bool use_aligned_float8 = ggml_cuda_is_aligned(src1, align_float8);
                static_assert(sizeof(block_fp4_mmq) == 4 * sizeof(block_q8_1));
                quantize_mmq_fp4_cuda(src1_d, nullptr, src1_q8_1.get(), src1_scale.ptr, src0->type, use_aligned_float8, ne10, s11, s12, s13, ne10_padded,
                                        ne11, ne12, ne13, stream);

            } else {
                quantize_mmq_q8_1_cuda(src1_d, nullptr, src1_q8_1.get(), nullptr, src0->type, ne10, s11, s12, s13, ne10_padded,
                                       ne11, ne12, ne13, stream);
            }
            CUDA_CHECK(cudaGetLastError());
        }

#if defined(GGML_RDNA35_Q4K_ADAPTIVE_Q4_ACTIVATION)
        if (use_adaptive_q4) {
            const int n_k_windows = ne10/32;
            const int n_m_tiles = (ne01 + 127)/128;
            const int n_n_tiles = (ne11 + 15)/16;
            const int n_fragments = ne12*ne13*n_m_tiles*n_n_tiles*n_k_windows;
            constexpr int stats_block_size = 256;
            const bool collect_stats = std::getenv("GGML_CUDA_Q4_ACTIVATION_STATS") != nullptr;
            ggml_cuda_pool_alloc<unsigned long long> q4_counts(ctx.pool());
            if (collect_stats) {
                q4_counts.alloc(2);
                CUDA_CHECK(cudaMemsetAsync(q4_counts.ptr, 0, 2*sizeof(unsigned long long), stream));
                count_adaptive_q4_fragments<<<(n_fragments + stats_block_size - 1)/stats_block_size, stats_block_size, 0, stream>>>(
                    src1_q4_flags.ptr, src1_q4_any.ptr, q4_counts.ptr, n_fragments);
                unsigned long long host_counts[2];
                CUDA_CHECK(cudaMemcpyAsync(host_counts, q4_counts.ptr, sizeof(host_counts), cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
                std::fprintf(stderr, "adaptive_q4: K=%" PRId64 " M=%" PRId64 " N=%" PRId64 " limit=%.6g selected=%llu total=%llu rate=%.6f\n",
                    ne10, ne01, ne11, get_adaptive_q4_score_limit(), host_counts[1], host_counts[0],
                    host_counts[0] ? double(host_counts[1])/double(host_counts[0]) : 0.0);
            }
        }
#endif

        // Stride depends on quantization format
        const int64_t s12 = ne11 * ne10_padded * y_block_size / (y_values_per_block * sizeof(int));
        const int64_t s13 = ne12*s12;

        const mmq_args args = {
            src0_d, src0->type, (const int *) src1_q8_1.ptr, (const int *) src1_q4_1.ptr,
            src1_q4_flags.ptr, src1_q4_any.ptr,
            nullptr, nullptr, dst_d,
            src0->type == GGML_TYPE_NVFP4 && use_native_fp4 ? src1_scale.ptr : nullptr,
            ne00, ne01, ne1, s01, ne11, s1,
            ne02, ne12, s02, s12, s2,
            ne03, ne13, s03, s13, s3,
            ne1, host_q4_any, kb0_q4_end};
        ggml_cuda_mul_mat_q_switch_type(ctx, args, stream);
        return;
    }

    GGML_ASSERT(ne13 == 1);
    GGML_ASSERT(nb12 % nb11 == 0);
    GGML_ASSERT(nb2  % nb1  == 0);

    const int64_t n_expert_used = ids->ne[0];
    const int64_t ne_get_rows = ne12 * n_expert_used;
    GGML_ASSERT(ne1 == n_expert_used);

    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), ne_get_rows);
    ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool(), ne_get_rows);
    ggml_cuda_pool_alloc<int32_t> expert_bounds(ctx.pool(), ne02 + 1);

    // gate/up activations are broadcast across experts (ne11 == 1): quantize each token once and
    // scatter to its slots. ids_src1 then holds the inverse map (token slot -> compact row).
    const bool dedup_bcast = ne11 == 1 && n_expert_used > 1;

    {
        GGML_ASSERT(ids->nb[0] == ggml_element_size(ids));
        const int si1  = ids->nb[1] / ggml_element_size(ids);
        const int sis1 = nb12 / nb11;

        ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, ids_src1.get(), ids_dst.get(), expert_bounds.get(),
            ne02, ne12, n_expert_used, ne11, si1, sis1, /*write_inverse =*/ dedup_bcast, stream);
        CUDA_CHECK(cudaGetLastError());
    }

    const size_t nbytes_src1_q8_1 = ne12*n_expert_used*ne10_padded * y_block_size/y_values_per_block +
        ggml_cuda_mmq_get_J_max(src0->type, fallback, cc, ne11) * sizeof(block_q8_1_mmq);
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_src1_q8_1);
    ggml_cuda_pool_alloc<float> src1_scale(ctx.pool());
    if (src0->type == GGML_TYPE_NVFP4 && use_native_fp4) {
        src1_scale.alloc(ne12*n_expert_used);
    }

    const int64_t ne11_flat = ne12*n_expert_used;
    const int64_t ne12_flat = 1;
    const int64_t ne13_flat = 1;

    {
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;

        if (use_native_fp4) {
            static constexpr size_t align_float8 = 32;
            const bool use_aligned_float8 = ggml_cuda_is_aligned(src1, align_float8);
            if (dedup_bcast) {
                quantize_scatter_mmq_fp4_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src1_scale.ptr, src0->type, use_aligned_float8, ne10,
                                        /*stride_token=*/s12, ne10_padded, ne12, ne11_flat, n_expert_used, stream);
            } else {
                quantize_mmq_fp4_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src1_scale.ptr, src0->type, use_aligned_float8, ne10, s11, s12, s13,
                                        ne10_padded, ne11_flat, ne12_flat, ne13_flat, stream);
            }
        } else if (dedup_bcast) {
            quantize_scatter_mmq_q8_1_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src0->type, ne10,
                                    /*stride_token=*/s12, ne10_padded, ne12, ne11_flat, n_expert_used, stream);
        } else {
            quantize_mmq_q8_1_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), nullptr, src0->type, ne10, s11, s12, s13,
                                   ne10_padded, ne11_flat, ne12_flat, ne13_flat, stream);
        }
        CUDA_CHECK(cudaGetLastError());
    }

    static_assert(QK_FP4_MMQ == 8 * QK_MXFP4, "QK_FP4_MMQ needs to be 8 * QK_MXFP4");
    const int64_t s12 = use_native_fp4 ? ne11 * ne10_padded * sizeof(block_fp4_mmq) / (QK_FP4_MMQ * sizeof(int)) :
                                         ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
    const int64_t s13 = ne12*s12;

    // Note that ne02 is used instead of ne12 because the number of y channels determines the z dimension of the CUDA grid.
    const mmq_args args = {
        src0_d, src0->type, (const int *) src1_q8_1.get(), nullptr, nullptr, nullptr,
        ids_dst.get(), expert_bounds.get(), dst_d,
        src1_scale.ptr,
        ne00, ne01, ne_get_rows, s01, ne_get_rows, s1,
        ne02, ne02, s02, s12, s2,
        ne03, ne13, s03, s13, s3,
        ne12};

    ggml_cuda_mul_mat_q_switch_type(ctx, args, stream);
}

bool ggml_cuda_should_use_mmq(enum ggml_type type, int cc, int64_t ne11, int64_t n_experts) {
#ifdef GGML_CUDA_FORCE_CUBLAS
    return false;
#endif // GGML_CUDA_FORCE_CUBLAS

    bool mmq_supported;

    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
// -------------------------------------------------
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
// -------------------------------------------------
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL:
// -------------------------------------------------
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
            mmq_supported = true;
            break;
        default:
            mmq_supported = false;
            break;
    }

    if (!mmq_supported) {
        return false;
    }

    if (turing_mma_available(cc)) {
        return true;
    }

    if (ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_DP4A) {
        return false;
    }

#ifdef GGML_CUDA_FORCE_MMQ
    return true;
#endif //GGML_CUDA_FORCE_MMQ

    if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
        return !fp16_mma_hardware_available(cc) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
    }

    if (amd_mfma_available(cc)) {
        // As of ROCM 7.0 rocblas/tensile performs very poorly on CDNA3 and hipblaslt (via ROCBLAS_USE_HIPBLASLT)
        // performs better but is currently suffering from a crash on this architecture.
        // TODO: Revisit when hipblaslt is fixed on CDNA3
        if (GGML_CUDA_CC_IS_CDNA3(cc)) {
            return true;
        }
        if (n_experts > 64 || ne11 <= 128) {
            return true;
        }
        if (type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_1 || type == GGML_TYPE_Q5_0 || type == GGML_TYPE_Q5_1) {
            return true;
        }
        if (ne11 <= 256 && (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K)) {
            return true;
        }
        return false;
    }

    if (amd_wmma_available(cc)) {
        if (GGML_CUDA_CC_IS_RDNA3(cc)) {
            // High expert counts are almost always better on MMQ due to
            //     the synchronization overhead in the cuBLAS/hipBLAS path:
            // https://github.com/ggml-org/llama.cpp/pull/18202
            if (n_experts >= 64) {
                return true;
            }

            // For some quantization types MMQ can have lower peak TOPS than hipBLAS
            //     so it's only faster for sufficiently small batch sizes:
            switch (type) {
                case GGML_TYPE_Q2_K:
                    return ne11 <= 128;
                case GGML_TYPE_Q6_K:
                    return ne11 <= (GGML_CUDA_CC_IS_RDNA3_0(cc) ? 128 : 256);
                case GGML_TYPE_IQ2_XS:
                case GGML_TYPE_IQ2_S:
                    return GGML_CUDA_CC_IS_RDNA3_5(cc) || ne11 <= 128;
                default:
                    return true;
            }
        }

        // For RDNA4 MMQ is consistently faster than dequantization + hipBLAS:
        // https://github.com/ggml-org/llama.cpp/pull/18537#issuecomment-3706422301
        return true;
    }

    // gfx900 (Vega 10) lacks native dp4a, loses to dequant + hipBLAS
    // for dense matrices; keep MMQ only for MoE, where the
    // hipBLAS path is much slower.
    if (cc == GGML_CUDA_CC_VEGA) {
        return n_experts > 0;
    }

    return (!GGML_CUDA_CC_IS_CDNA(cc)) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
}
