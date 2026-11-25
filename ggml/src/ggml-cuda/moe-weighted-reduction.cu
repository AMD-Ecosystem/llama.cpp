#include "moe-weighted-reduction.cuh"

static __global__ void moe_weighted_reduction_f32(const float * __restrict__ experts,
                                                  const float * __restrict__ expert_scale,
                                                  const float * __restrict__ weights,
                                                  float * __restrict__ dst,
                                                  const int64_t n_embd,
                                                  const int     n_expert_used) {
    const int64_t token = blockIdx.x;
    const int64_t col   = (int64_t) blockIdx.y * blockDim.x + threadIdx.x;
    if (col >= n_embd) {
        return;
    }

    const uint64_t first_row   = (uint64_t) token * n_expert_used;
    const float    first_scale = expert_scale != nullptr ? expert_scale[first_row] : 1.0f;
    float          sum         = (experts[first_row * n_embd + col] * first_scale) * weights[first_row];

    for (int expert = 1; expert < n_expert_used; ++expert) {
        const uint64_t row   = first_row + expert;
        const float   scale = expert_scale != nullptr ? expert_scale[row] : 1.0f;
        sum += (experts[row * n_embd + col] * scale) * weights[row];
    }
    dst[token * n_embd + col] = sum;
}

// same sums as moe_weighted_reduction_f32, with all loads of a thread issued before the sum and 4 columns per thread
template <int n_expert_used>
static __global__ void moe_weighted_reduction_f32_x4(const float4 * __restrict__ experts,
                                                     const float * __restrict__  expert_scale,
                                                     const float * __restrict__  weights,
                                                     float4 * __restrict__       dst,
                                                     const int64_t               n_embd4) {
    const int64_t token = blockIdx.x;
    const int64_t col   = (int64_t) blockIdx.y * blockDim.x + threadIdx.x;
    if (col >= n_embd4) {
        return;
    }

    const uint64_t first_row = (uint64_t) token * n_expert_used;
    float4 x[n_expert_used];
    float  w[n_expert_used];
#pragma unroll
    for (int expert = 0; expert < n_expert_used; ++expert) {
        x[expert] = experts[(first_row + expert) * n_embd4 + col];
        w[expert] = weights[first_row + expert];
    }
    float s[n_expert_used];
#pragma unroll
    for (int expert = 0; expert < n_expert_used; ++expert) {
        s[expert] = expert_scale != nullptr ? expert_scale[first_row + expert] : 1.0f;
    }
    float4 sum;
    sum.x = (x[0].x * s[0]) * w[0];
    sum.y = (x[0].y * s[0]) * w[0];
    sum.z = (x[0].z * s[0]) * w[0];
    sum.w = (x[0].w * s[0]) * w[0];
#pragma unroll
    for (int expert = 1; expert < n_expert_used; ++expert) {
        sum.x += (x[expert].x * s[expert]) * w[expert];
        sum.y += (x[expert].y * s[expert]) * w[expert];
        sum.z += (x[expert].z * s[expert]) * w[expert];
        sum.w += (x[expert].w * s[expert]) * w[expert];
    }
    dst[token * n_embd4 + col] = sum;
}

static void launch_moe_weighted_reduction(const float * experts,
                                          const float * expert_scale,
                                          const float * weights,
                                          float *       dst,
                                          int64_t       n_embd,
                                          int64_t       n_tokens,
                                          int           n_expert_used,
                                          cudaStream_t  stream) {
    // opt-in: one kernel per common expert count, the generic kernel is latency bound with large n_tokens
    static const bool use_x4 = getenv("GGML_CUDA_MOE_REDUCE_X4") != nullptr && atoi(getenv("GGML_CUDA_MOE_REDUCE_X4")) != 0;
    if (use_x4 && n_embd % 4 == 0 && (n_expert_used == 8 || n_expert_used == 4) &&
            (uintptr_t) experts % 16 == 0 && (uintptr_t) dst % 16 == 0) {
        constexpr int threads = 128;
        const int64_t n_embd4 = n_embd / 4;
        const dim3 blocks(n_tokens, (n_embd4 + threads - 1) / threads, 1);
        if (n_expert_used == 8) {
            moe_weighted_reduction_f32_x4<8><<<blocks, threads, 0, stream>>>((const float4 *) experts, expert_scale, weights, (float4 *) dst, n_embd4);
        } else {
            moe_weighted_reduction_f32_x4<4><<<blocks, threads, 0, stream>>>((const float4 *) experts, expert_scale, weights, (float4 *) dst, n_embd4);
        }
        return;
    }
    constexpr int threads = 256;
    const dim3 blocks(n_tokens, (n_embd + threads - 1) / threads, 1);
    moe_weighted_reduction_f32
        <<<blocks, threads, 0, stream>>>(experts, expert_scale, weights, dst, n_embd, n_expert_used);
}

void ggml_cuda_op_moe_weighted_reduction(ggml_backend_cuda_context & ctx,
                                         const ggml_tensor *         experts,
                                         const ggml_tensor *         expert_scale,
                                         const ggml_tensor *         weights,
                                         ggml_tensor *               dst) {
    GGML_ASSERT(experts->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(expert_scale == nullptr || expert_scale->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(experts));
    GGML_ASSERT(ggml_is_contiguous(weights));
    GGML_ASSERT(expert_scale == nullptr || ggml_is_contiguous(expert_scale));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int64_t n_embd        = experts->ne[0];
    const int64_t n_expert_used = experts->ne[1];
    const int64_t n_tokens      = experts->ne[2] * experts->ne[3];
    cudaStream_t  stream        = ctx.stream();

    launch_moe_weighted_reduction((const float *) experts->data,
                                  expert_scale ? (const float *) expert_scale->data : nullptr,
                                  (const float *) weights->data,
                                  (float *) dst->data, n_embd, n_tokens, (int) n_expert_used, stream);
    CUDA_CHECK(cudaGetLastError());
}
