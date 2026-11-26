#include "ggml-cuda/common.cuh"
#include "ggml.h"
#include "topk-moe.cuh"

#include <cmath>
#include <initializer_list>

// Kernel config struct - passed by value to CUDA kernel
struct topk_moe_config {
    bool use_sigmoid;
    bool use_sqrt_softplus;
    bool with_norm;
    bool delayed_softmax;
    int  n_groups;
    int  n_group_used;
};

// Warp-local softmax used for both the pre-top-k logits and the post-top-k delayed path.
template <int experts_per_thread, bool use_limit>
__device__ void softmax_warp_inplace(float (&vals)[experts_per_thread], const int limit, const int lane) {
    float max_val = -INFINITY;

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        if (active) {
            max_val = max(max_val, vals[i]);
        }
    }

    max_val = warp_reduce_max(max_val);

    float sum = 0.f;

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        if (active) {
            const float val = expf(vals[i] - max_val);
            vals[i]         = val;
            sum += val;
        } else {
            vals[i] = 0.f;
        }
    }

    sum = warp_reduce_sum(sum);

    const float inv_sum = 1.0f / sum;

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        if (active) {
            vals[i] *= inv_sum;
        }
    }
}

template <int experts_per_thread, bool use_limit>
__device__ void sigmoid_warp_inplace(float (&vals)[experts_per_thread], const int limit, const int lane) {
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        vals[i]           = active ? 1.f / (1.f + expf(-vals[i])) : -INFINITY;
    }
}

template <int experts_per_thread, bool use_limit>
__device__ void sqrt_softplus_warp_inplace(float (&vals)[experts_per_thread], const int limit, const int lane) {
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        vals[i]           = active ? sqrtf(vals[i] > 20.0f ? vals[i] : log1pf(expf(vals[i]))) : -INFINITY;
    }
}

#if !defined(GGML_USE_HIP) || HIP_VERSION >= 60300000
// Set -inf for experts outside the top n_group_used groups. Group score is the sum of its top-2 experts (DeepSeek V3).
template <int n_experts, int experts_per_thread>
__device__ void group_mask_warp_inplace(float (&vals)[experts_per_thread], const int n_groups, const int n_group_used, const int lane) {
    const int n_per_group = n_experts / n_groups;

    float score = -INFINITY; // lane g holds the score of group g
    for (int g = 0; g < n_groups; g++) {
        float top1 = -INFINITY;
        float top2 = -INFINITY;
#pragma unroll
        for (int i = 0; i < experts_per_thread; i++) {
            if ((i * WARP_SIZE + lane) / n_per_group == g) {
                top2 = fmaxf(top2, fminf(top1, vals[i]));
                top1 = fmaxf(top1, vals[i]);
            }
        }
#pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            const float o1 = __shfl_xor_sync(0xFFFFFFFF, top1, offset, WARP_SIZE);
            const float o2 = __shfl_xor_sync(0xFFFFFFFF, top2, offset, WARP_SIZE);
            top2 = fmaxf(fminf(top1, o1), fmaxf(top2, o2));
            top1 = fmaxf(top1, o1);
        }
        if (lane == g) {
            score = top1 + top2;
        }
    }

    // rank of each group, lower index wins ties
    int rank = 0;
    for (int g = 0; g < n_groups; g++) {
        const float other = __shfl_sync(0xFFFFFFFF, score, g, WARP_SIZE);
        rank += other > score || (other == score && g < lane);
    }
    const auto keep = __ballot_sync(0xFFFFFFFFULL, lane < n_groups && rank < n_group_used);

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int g = (i * WARP_SIZE + lane) / n_per_group;
        if (g >= n_groups || !((keep >> g) & 1)) {
            vals[i] = -INFINITY;
        }
    }
}
#endif // !defined(GGML_USE_HIP) || HIP_VERSION >= 60300000

// The top-k selection of topk_moe_cuda for one row, by one warp (lane threadIdx.x)
template <int n_experts, bool has_bias>
static __device__ __forceinline__ void topk_moe_row(const float *         logits,
                                                    float *               weights,
                                                    int32_t *             ids,
                                                    float *               bias,
                                                    const int             n_expert_used,
                                                    const float           clamp_val,
                                                    const float           scale_val,
                                                    const topk_moe_config config) {
    constexpr int experts_per_thread = (n_experts > WARP_SIZE) ? n_experts / WARP_SIZE : 1;

    float wt[experts_per_thread];

    // Initialize all slots to -INFINITY
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        wt[i] = -INFINITY;
    }

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int i = 0; i < n_experts; i += WARP_SIZE) {
        const int expert  = i + threadIdx.x;
        wt[i / WARP_SIZE] = (n_experts % WARP_SIZE == 0 || expert < n_experts) ? logits[expert] : -INFINITY;
    }

    // Weights and IDs can alias logits, so wait until every row in the block reads its logits.
    __syncthreads();

    if (!config.delayed_softmax) {
        if (config.use_sigmoid) {
           sigmoid_warp_inplace<experts_per_thread, false>(wt, n_experts, threadIdx.x);
        } else if (config.use_sqrt_softplus) {
           sqrt_softplus_warp_inplace<experts_per_thread, false>(wt, n_experts, threadIdx.x);
        } else {
           softmax_warp_inplace<experts_per_thread, false>(wt, n_experts, threadIdx.x);
        }
    }

    // Sanitize NaN to -FLT_MAX so the iterative argmax produces unique expert IDs.
    // NaN comparisons always return false, which would cause the same expert to be
    // selected repeatedly. -FLT_MAX compares normally and is still excluded by the
    // -INFINITY sentinel used after each selection round.
    // More relevant for the cuBLAS path. See https://github.com/ggml-org/llama.cpp/issues/19659
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        if (__isnanf(wt[i])) {
            wt[i] = -FLT_MAX;
        }
    }

    // selection_wt is only needed when bias is present (selection uses wt + bias)
    // when no bias, we use wt directly for both selection and weight values
    [[maybe_unused]] float selection_wt[has_bias ? experts_per_thread : 1];

    if constexpr (has_bias) {
#pragma unroll
        for (int i = 0; i < experts_per_thread; i++) {
            selection_wt[i] = -INFINITY;
        }
#pragma unroll
        for (int i = 0; i < n_experts; i += WARP_SIZE) {
            const int expert = i + threadIdx.x;
            selection_wt[i / WARP_SIZE] =
                (n_experts % WARP_SIZE == 0 || expert < n_experts) ? wt[i / WARP_SIZE] + bias[expert] : -INFINITY;
            if (__isnanf(selection_wt[i / WARP_SIZE])) {
                selection_wt[i / WARP_SIZE] = -FLT_MAX;
            }
        }
    }

#if !defined(GGML_USE_HIP) || HIP_VERSION >= 60300000
    // without bias, masking wt is safe: the matcher ensures the kept groups hold all used experts
    if (config.n_groups > 0) {
        if constexpr (has_bias) {
            group_mask_warp_inplace<n_experts>(selection_wt, config.n_groups, config.n_group_used, threadIdx.x);
        } else {
            group_mask_warp_inplace<n_experts>(wt, config.n_groups, config.n_group_used, threadIdx.x);
        }
    }
#endif // !defined(GGML_USE_HIP) || HIP_VERSION >= 60300000

    //at this point, each thread holds either a portion of the softmax distribution
    //or the raw logits. We do the argmax reduce over n_expert_used, each time marking
    //the expert weight as -inf to exclude from the next iteration

    float wt_sum = 0.f;

    float output_weights[experts_per_thread];

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        output_weights[i] = 0.f;
    }

    ggml_cuda_pdl_lc();
    for (int k = 0; k < n_expert_used; k++) {
        float max_val    = wt[0];
        int   max_expert = threadIdx.x;

        if constexpr (has_bias) {
            float max_val_s = selection_wt[0];

#pragma unroll
            for (int i = 1; i < experts_per_thread; i++) {
                const int expert = threadIdx.x + i * WARP_SIZE;
                if ((n_experts % WARP_SIZE == 0 || expert < n_experts) && selection_wt[i] > max_val_s) {
                    max_val    = wt[i];
                    max_val_s  = selection_wt[i];
                    max_expert = expert;
                }
            }

#pragma unroll
            for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
                const float val    = __shfl_xor_sync(0xFFFFFFFF, max_val, mask, WARP_SIZE);
                const float val_s  = __shfl_xor_sync(0xFFFFFFFF, max_val_s, mask, WARP_SIZE);
                const int   expert = __shfl_xor_sync(0xFFFFFFFF, max_expert, mask, WARP_SIZE);
                if (val_s > max_val_s || (val_s == max_val_s && expert < max_expert)) {
                    max_val    = val;
                    max_val_s  = val_s;
                    max_expert = expert;
                }
            }

            if ((max_expert & (WARP_SIZE - 1)) == threadIdx.x) {
                selection_wt[max_expert / WARP_SIZE] = -INFINITY;
            }
        } else {
#pragma unroll
            for (int i = 1; i < experts_per_thread; i++) {
                const int expert = threadIdx.x + i * WARP_SIZE;
                if ((n_experts % WARP_SIZE == 0 || expert < n_experts) && wt[i] > max_val) {
                    max_val    = wt[i];
                    max_expert = expert;
                }
            }

#pragma unroll
            for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
                const float val    = __shfl_xor_sync(0xFFFFFFFF, max_val, mask, WARP_SIZE);
                const int   expert = __shfl_xor_sync(0xFFFFFFFF, max_expert, mask, WARP_SIZE);
                if (val > max_val || (val == max_val && expert < max_expert)) {
                    max_val    = val;
                    max_expert = expert;
                }
            }

            if ((max_expert & (WARP_SIZE - 1)) == threadIdx.x) {
                wt[max_expert / WARP_SIZE] = -INFINITY;
            }
        }

        if ((k & (WARP_SIZE - 1)) == threadIdx.x) {
            output_weights[k / WARP_SIZE] = max_val;
        }

        if ((max_expert & (WARP_SIZE - 1)) == threadIdx.x) {
            ids[k] = max_expert;
            if (config.with_norm) {
                wt_sum += max_val;
            }
        }
    }

    if (config.with_norm) {
        wt_sum              = warp_reduce_sum(wt_sum);
        wt_sum              = max(wt_sum, clamp_val);
        const float inv_sum = 1.0f / wt_sum;

        for (int i = 0; i < experts_per_thread; i++) {
            output_weights[i] *= inv_sum;
        }
    }

    if (config.delayed_softmax) {
        softmax_warp_inplace<experts_per_thread, true>(output_weights, n_expert_used, threadIdx.x);
    }

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int idx = i * WARP_SIZE + threadIdx.x;
        if (idx < n_expert_used) {
            weights[idx] = output_weights[i] * scale_val;
        }
    }
}

/*
    This kernel does the following:
    1. optionally softmax over the logits per token [n_experts, n_tokens]
    2. argmax reduce over the top-k (n_experts_used) logits
    3. write weights + ids to global memory
    4. optionally normalize the weights or apply softmax over the selected logits

    It is intended as fusion of softmax->top-k->get_rows pipeline for MoE models
*/
template <int n_experts, bool has_bias>
__launch_bounds__(TOPK_MOE_ROWS_PER_BLOCK * WARP_SIZE, 1)
__global__ void topk_moe_cuda(const float *         logits,
                              float *               weights,
                              int32_t *             ids,
                              float *               bias,
                              const int             n_rows,
                              const int             n_expert_used,
                              const float           clamp_val,
                              const float           scale_val,
                              const topk_moe_config config) {
#if defined(GGML_USE_MUSA)
    // MUSA: every warp of a partially filled block must reach the barrier below.
    const int row = MIN(blockIdx.x * blockDim.y + threadIdx.y, n_rows - 1);
#else
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
#endif // defined(GGML_USE_MUSA)
    if (row >= n_rows) {
        return;
    }

    logits += n_experts * row;
    weights += n_expert_used * row;
    ids += n_experts * row;

    topk_moe_row<n_experts, has_bias>(logits, weights, ids, bias, n_expert_used, clamp_val, scale_val, config);
}

template<bool has_bias>
static void launch_topk_moe_cuda(ggml_backend_cuda_context & ctx,
                                 const float *               logits,
                                 float *                     weights,
                                 int32_t *                   ids,
                                 float *                     bias,
                                 const int                   n_rows,
                                 const int                   n_expert,
                                 const int                   n_expert_used,
                                 const float                 clamp_val,
                                 const float                 scale_val,
                                 const topk_moe_config       config) {
    GGML_ASSERT(!(config.with_norm && config.delayed_softmax) &&
                "delayed softmax is not supported with weight normalization");
    const int    rows_per_block = TOPK_MOE_ROWS_PER_BLOCK;
    dim3         grid_dims((n_rows + rows_per_block - 1) / rows_per_block, 1, 1);
    dim3         block_dims(WARP_SIZE, rows_per_block, 1);
    cudaStream_t stream = ctx.stream();
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);

#if defined(GGML_USE_HIP) && HIP_VERSION < 60300000
    GGML_ASSERT(config.n_groups == 0 && "expert groups fusion is not supported on this HIP version");
#endif

    switch (n_expert) {
        case 1:
            ggml_cuda_kernel_launch(topk_moe_cuda<1, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 2:
            ggml_cuda_kernel_launch(topk_moe_cuda<2, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 4:
            ggml_cuda_kernel_launch(topk_moe_cuda<4, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 8:
            ggml_cuda_kernel_launch(topk_moe_cuda<8, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 16:
            ggml_cuda_kernel_launch(topk_moe_cuda<16, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 32:
            ggml_cuda_kernel_launch(topk_moe_cuda<32, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 64:
            ggml_cuda_kernel_launch(topk_moe_cuda<64, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 128:
            ggml_cuda_kernel_launch(topk_moe_cuda<128, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 256:
            ggml_cuda_kernel_launch(topk_moe_cuda<256, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 288: // StepFun 3.7
            ggml_cuda_kernel_launch(topk_moe_cuda<288, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 384: // DeepSeek V4.1
            ggml_cuda_kernel_launch(topk_moe_cuda<384, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 512:
            ggml_cuda_kernel_launch(topk_moe_cuda<512, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 576:
            ggml_cuda_kernel_launch(topk_moe_cuda<576, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        default:
            GGML_ASSERT(false && "fatal error");
    }
}

void ggml_cuda_op_topk_moe(ggml_backend_cuda_context &     ctx,
                           const ggml_tensor *             logits,
                           ggml_tensor *                   weights,
                           ggml_tensor *                   ids,
                           const ggml_tensor *             clamp,
                           const ggml_tensor *             scale,
                           const ggml_tensor *             bias,
                           const ggml_cuda_topk_moe_args & args) {
    GGML_ASSERT(logits->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    const int n_experts = logits->ne[0];
    const int n_rows    = logits->ne[1];

    const float * logits_d  = (const float *) logits->data;
    float *       weights_d = (float *) weights->data;
    int32_t *     ids_d     = (int32_t *) ids->data;
    float *       bias_d    = bias ? (float *) bias->data : nullptr;

    float scale_val = scale ? ggml_get_op_params_f32(scale, 0) : 1.0f;

    GGML_ASSERT(ids->nb[1] / ggml_type_size(ids->type) == (size_t) n_experts);

    const int n_expert_used = weights->ne[1];

    const bool with_norm = clamp != nullptr;

    float clamp_val = -INFINITY;
    if (clamp) {
        clamp_val = ggml_get_op_params_f32(clamp, 0);
    }

    topk_moe_config config;
    config.use_sigmoid       = args.sigmoid;
    config.use_sqrt_softplus = args.sqrt_softplus;
    config.with_norm         = with_norm;
    config.delayed_softmax   = args.delayed_softmax;
    config.n_groups          = args.n_groups;
    config.n_group_used      = args.n_group_used;

    if (bias) {
        launch_topk_moe_cuda<true>(ctx, logits_d, weights_d, ids_d, bias_d, n_rows, n_experts, n_expert_used, clamp_val,
                             scale_val, config);
    } else {
        launch_topk_moe_cuda<false>(ctx, logits_d, weights_d, ids_d, bias_d, n_rows, n_experts, n_expert_used, clamp_val,
                             scale_val, config);
    }
}

// The F32 router product of one token as mul_mat_vec_f<float, float, 1, 256> computes it, block row computing logit
// row, and the top-k selection of topk_moe_cuda on the logits by the last block to finish (counted in counter, which it
// resets).
template <int n_experts, bool has_bias>
__launch_bounds__(256, 1)
static __global__ void router_topk_moe_cuda(const float *         x,
                                            const float *         y,
                                            float *               logits,
                                            const int             ncols2,
                                            const int             stride_row,
                                            float *               weights,
                                            int32_t *             ids,
                                            float *               bias,
                                            const int             n_expert_used,
                                            const float           clamp_val,
                                            const float           scale_val,
                                            const topk_moe_config config,
                                            int *                 counter) {
    constexpr int block_size = 256;
    const int row = blockIdx.x;
    const int tid = threadIdx.y*WARP_SIZE + threadIdx.x;

    __shared__ float buf_iw[WARP_SIZE];
    if (threadIdx.y == 0) {
        buf_iw[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    float sumf = 0.0f;
    const float2 * x2 = (const float2 *) (x + (int64_t) row*stride_row);
    const float2 * y2 = (const float2 *) y;
    for (int col2 = tid; col2 < ncols2; col2 += block_size) {
        const float2 tmpx = x2[col2];
        const float2 tmpy = y2[col2];
        ggml_cuda_mad(sumf, tmpx.x, tmpy.x);
        ggml_cuda_mad(sumf, tmpx.y, tmpy.y);
    }
    sumf = warp_reduce_sum<WARP_SIZE>(sumf);
    buf_iw[threadIdx.y] = sumf;
    __syncthreads();
    if (threadIdx.y != 0) {
        return;
    }
    sumf = buf_iw[threadIdx.x];
    sumf = warp_reduce_sum<WARP_SIZE>(sumf);
    if (threadIdx.x == 0) {
        logits[row] = sumf;
    }

    __threadfence();
    int arrived = 0;
    if (threadIdx.x == 0) {
        arrived = atomicAdd(counter, 1);
    }
    arrived = __shfl_sync(0xFFFFFFFF, arrived, 0, WARP_SIZE);
    if (arrived != (int) gridDim.x - 1) {
        return;
    }
    __threadfence();
    if (threadIdx.x == 0) {
        *counter = 0;
    }
    topk_moe_row<n_experts, has_bias>(logits, weights, ids, bias, n_expert_used, clamp_val, scale_val, config);
}

static __global__ void router_weights_f32_to_bf16(const float * x, uint16_t * dst, const int64_t n) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) {
        // round to nearest even, as SnowLLM stores the router
        const uint32_t u = __float_as_uint(x[i]);
        dst[i] = (u + 0x7FFF + ((u >> 16) & 1)) >> 16;
    }
}

void ggml_cuda_router_weights_to_bf16(const float * x, void * dst, const int64_t n, cudaStream_t stream) {
    const int block_size = 256;
    router_weights_f32_to_bf16<<<(n + block_size - 1) / block_size, block_size, 0, stream>>>(x, (uint16_t *) dst, n);
}

// The router product of one token with BF16 weights (the F32 ones rounded, see ggml_cuda_router_weights_to_bf16), block
// row computing logit row with 4 weights per thread and step, and the top-k selection as router_topk_moe_cuda does it.
template <int n_experts, bool has_bias>
__launch_bounds__(256, 1)
static __global__ void router_topk_moe_bf16_cuda(const uint16_t *      x,
                                                 const float *         y,
                                                 float *               logits,
                                                 const int             ncols4,
                                                 const int             stride_row,
                                                 float *               weights,
                                                 int32_t *             ids,
                                                 float *               bias,
                                                 const int             n_expert_used,
                                                 const float           clamp_val,
                                                 const float           scale_val,
                                                 const topk_moe_config config,
                                                 int *                 counter) {
    constexpr int block_size = 256;
    const int row = blockIdx.x;
    const int tid = threadIdx.y*WARP_SIZE + threadIdx.x;

    __shared__ float buf_iw[WARP_SIZE];
    if (threadIdx.y == 0) {
        buf_iw[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    float sumf = 0.0f;
    const uint2  * x4 = (const uint2  *) (x + (int64_t) row*stride_row);
    const float4 * y4 = (const float4 *) y;
    for (int col4 = tid; col4 < ncols4; col4 += block_size) {
        const uint2  w  = x4[col4];
        const float4 yv = y4[col4];
        sumf += __uint_as_float(w.x << 16)         * yv.x;
        sumf += __uint_as_float(w.x & 0xFFFF0000u) * yv.y;
        sumf += __uint_as_float(w.y << 16)         * yv.z;
        sumf += __uint_as_float(w.y & 0xFFFF0000u) * yv.w;
    }
    sumf = warp_reduce_sum<WARP_SIZE>(sumf);
    buf_iw[threadIdx.y] = sumf;
    __syncthreads();
    if (threadIdx.y != 0) {
        return;
    }
    sumf = buf_iw[threadIdx.x];
    sumf = warp_reduce_sum<WARP_SIZE>(sumf);
    if (threadIdx.x == 0) {
        logits[row] = sumf;
    }

    __threadfence();
    int arrived = 0;
    if (threadIdx.x == 0) {
        arrived = atomicAdd(counter, 1);
    }
    arrived = __shfl_sync(0xFFFFFFFF, arrived, 0, WARP_SIZE);
    if (arrived != (int) gridDim.x - 1) {
        return;
    }
    __threadfence();
    if (threadIdx.x == 0) {
        *counter = 0;
    }
    topk_moe_row<n_experts, has_bias>(logits, weights, ids, bias, n_expert_used, clamp_val, scale_val, config);
}

bool ggml_cuda_router_topk_moe_supported(const ggml_tensor * mm, const ggml_tensor * logits) {
    const ggml_tensor * x = mm->src[0];
    const ggml_tensor * y = mm->src[1];
    if (logits != mm || x->type != GGML_TYPE_F32 || y->type != GGML_TYPE_F32 || mm->type != GGML_TYPE_F32 ||
            mm->ne[0] != 512 || ggml_nrows(mm) != 1 || ggml_nrows(y) != 1 || x->ne[1] != mm->ne[0] || ggml_nrows(x) != x->ne[1] ||
            x->nb[0] != sizeof(float) || y->nb[0] != sizeof(float) || !ggml_is_contiguous(mm) ||
            x->ne[0] % 2 != 0 || (x->nb[1] / sizeof(float)) % 2 != 0 || ((uintptr_t) x->data & 7) != 0 ||
            ((uintptr_t) y->data & 7) != 0 || ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size != WARP_SIZE) {
        return false;
    }
    // the block size ggml_cuda_mul_mat_vec_f picks
    const int64_t ncols      = x->ne[0];
    const int     warp_size  = WARP_SIZE;
    const int     cc         = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    int64_t block_size_best = warp_size;
    int64_t niter_best      = (ncols + 2*warp_size - 1) / (2*warp_size);
    const int64_t max_block_size = cc > GGML_CUDA_CC_OFFSET_AMD && cc < GGML_CUDA_CC_RDNA1 ? 128 : 256;
    for (int64_t block_size = 2*warp_size; block_size <= max_block_size; block_size += warp_size) {
        const int64_t niter = (ncols + 2*block_size - 1) / (2*block_size);
        if (niter < niter_best) {
            niter_best      = niter;
            block_size_best = block_size;
        }
    }
    return block_size_best == 256;
}

void ggml_cuda_op_router_topk_moe(ggml_backend_cuda_context &     ctx,
                                  const ggml_tensor *             mm,
                                  ggml_tensor *                   weights,
                                  ggml_tensor *                   ids,
                                  const ggml_tensor *             clamp,
                                  const ggml_tensor *             scale,
                                  const ggml_tensor *             bias,
                                  const ggml_cuda_topk_moe_args & args,
                                  int *                           counter) {
    const ggml_tensor * x = mm->src[0];
    const ggml_tensor * y = mm->src[1];
    const auto it_bf16 = ctx.router_bf16.find(x->data);
    const uint16_t * x_bf16 = it_bf16 != ctx.router_bf16.end() && x->ne[0] % 4 == 0 && ((uintptr_t) y->data & 15) == 0 ?
        (const uint16_t *) it_bf16->second : nullptr;

    GGML_ASSERT(!(clamp != nullptr && args.delayed_softmax) && args.n_groups == 0);
    GGML_ASSERT(ids->nb[1] / ggml_type_size(ids->type) == (size_t) mm->ne[0]);

    const float scale_val = scale ? ggml_get_op_params_f32(scale, 0) : 1.0f;
    const float clamp_val = clamp ? ggml_get_op_params_f32(clamp, 0) : -INFINITY;

    topk_moe_config config;
    config.use_sigmoid       = args.sigmoid;
    config.use_sqrt_softplus = args.sqrt_softplus;
    config.with_norm         = clamp != nullptr;
    config.delayed_softmax   = args.delayed_softmax;
    config.n_groups          = args.n_groups;
    config.n_group_used      = args.n_group_used;

    const int n_expert_used = weights->ne[1];
    const dim3 block_dims(WARP_SIZE, 256/WARP_SIZE, 1);
    const dim3 grid_dims(mm->ne[0], 1, 1);
    if (x_bf16) {
        auto kernel_bf16 = bias ? router_topk_moe_bf16_cuda<512, true> : router_topk_moe_bf16_cuda<512, false>;
        kernel_bf16<<<grid_dims, block_dims, 0, ctx.stream()>>>(
            x_bf16, (const float *) y->data, (float *) mm->data, x->ne[0] / 4, x->nb[1] / sizeof(float),
            (float *) weights->data, (int32_t *) ids->data, bias ? (float *) bias->data : nullptr, n_expert_used, clamp_val,
            scale_val, config, counter);
        return;
    }
    auto kernel = bias ? router_topk_moe_cuda<512, true> : router_topk_moe_cuda<512, false>;
    kernel<<<grid_dims, block_dims, 0, ctx.stream()>>>(
        (const float *) x->data, (const float *) y->data, (float *) mm->data, x->ne[0] / 2, x->nb[1] / sizeof(float),
        (float *) weights->data, (int32_t *) ids->data, bias ? (float *) bias->data : nullptr, n_expert_used, clamp_val,
        scale_val, config, counter);
}

bool ggml_cuda_should_use_topk_moe(const ggml_tensor * gating_op,
                                   const ggml_tensor * weights,
                                   const ggml_tensor * logits,
                                   const ggml_tensor * ids) {
    // must match an instantiation of launch_topk_moe_cuda: a power of 2 up to 512,
    // or one of the non-power-of-2 expert counts of supported models
    const int n_expert = ids->nb[1] / ids->nb[0];
    if (((n_expert & (n_expert - 1)) != 0 || n_expert > 512) && n_expert != 288 && n_expert != 384 && n_expert != 576) {
        return false;
    }

    if (!ggml_is_contiguous(weights) || !ggml_is_contiguous(logits)) {
        return false;
    }

    if (gating_op->op == GGML_OP_SOFT_MAX) {
        const ggml_tensor * softmax  = gating_op;
        float               scale    = 1.0f;
        float               max_bias = 0.0f;

        memcpy(&scale, (const float *) softmax->op_params + 0, sizeof(float));
        memcpy(&max_bias, (const float *) softmax->op_params + 1, sizeof(float));

        if (!ggml_is_contiguous(softmax->src[0])) {
            return false;
        }

        if (scale != 1.0f || max_bias != 0.0f) {
            return false;
        }

        // don't fuse when masks or sinks are present
        if (softmax->src[1] || softmax->src[2]) {
            return false;
        }
    } else if (gating_op->op == GGML_OP_UNARY) {
        ggml_unary_op op = ggml_get_unary_op(gating_op);

        if (op != GGML_UNARY_OP_SIGMOID && op != GGML_UNARY_OP_SOFTPLUS) {
            return false;
        }
    }

    return true;
}
