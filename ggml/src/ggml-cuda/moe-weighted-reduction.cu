#include "moe-weighted-reduction.cuh"

static __global__ void moe_weighted_reduction_f32(const float * __restrict__ experts,
                                                  const float * __restrict__ expert_scale,
                                                  const float * __restrict__ weights,
                                                  float * __restrict__ dst,
                                                  const int64_t n_embd,
                                                  const int     n_expert_used,
                                                  block_q8_1 * __restrict__ dst_q8_1) {
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

    if (dst_q8_1) {
        // MMVQ's q8_1 copy of dst as quantize_q8_1 makes it, one 32-value block per warp
        const float amax = warp_reduce_max<QK8_1>(fabsf(sum));
        const float d    = amax / 127.0f;
        const int8_t q   = amax == 0.0f ? 0 : roundf(sum / d);
        const int    s   = warp_reduce_sum<QK8_1>((int) q);
        block_q8_1 * y = dst_q8_1 + (token * n_embd + col) / QK8_1;
        y->qs[col % QK8_1] = q;
        if (col % QK8_1 == 0) {
            y->ds = make_half2(d, d * s);
        }
    }
}

// Prompt batches: a thread sums 4 consecutive columns with 16 byte loads, all the experts' loads in flight at once.
// The expressions and their order are those of moe_weighted_reduction_f32, so the sums are bit for bit the same.
template <int N_EXPERT>
static __global__ void __launch_bounds__(128) moe_weighted_reduction_f32_x4(const float * __restrict__ experts,
                                                                           const float * __restrict__ expert_scale,
                                                                           const float * __restrict__ weights,
                                                                           float * __restrict__ dst,
                                                                           const int64_t n_embd) {
    const int64_t token = blockIdx.y;
    const int64_t col   = ((int64_t) blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (col >= n_embd) {
        return;
    }

    const uint64_t first_row = (uint64_t) token * N_EXPERT;
    const float4 * src       = (const float4 *) (experts + first_row * n_embd + col);
    const int64_t  stride4   = n_embd / 4;

    float4 v[N_EXPERT];
#pragma unroll
    for (int expert = 0; expert < N_EXPERT; ++expert) {
        v[expert] = src[expert * stride4];
    }

    // sum_0 = (e_0 * s_0) * w_0 with both products rounded, sum_i = fma(e_i * s_i, w_i, sum_{i-1}): what the compiler makes
    // of the scalar kernel's loop. Spelled out because the compiler may otherwise pick another product to fuse here.
    const float first_scale = expert_scale != nullptr ? expert_scale[first_row] : 1.0f;
    const float w0          = weights[first_row];
    float4 sum;
    sum.x = __fmul_rn(__fmul_rn(v[0].x, first_scale), w0);
    sum.y = __fmul_rn(__fmul_rn(v[0].y, first_scale), w0);
    sum.z = __fmul_rn(__fmul_rn(v[0].z, first_scale), w0);
    sum.w = __fmul_rn(__fmul_rn(v[0].w, first_scale), w0);

#pragma unroll
    for (int expert = 1; expert < N_EXPERT; ++expert) {
        const uint64_t row   = first_row + expert;
        const float   scale = expert_scale != nullptr ? expert_scale[row] : 1.0f;
        const float   w     = weights[row];
        sum.x = fmaf(__fmul_rn(v[expert].x, scale), w, sum.x);
        sum.y = fmaf(__fmul_rn(v[expert].y, scale), w, sum.y);
        sum.z = fmaf(__fmul_rn(v[expert].z, scale), w, sum.z);
        sum.w = fmaf(__fmul_rn(v[expert].w, scale), w, sum.w);
    }
    *(float4 *) (dst + token * n_embd + col) = sum;
}

// the longest reduction the x4 kernel is instantiated for (the graph matcher accepts at most 15 experts)
#define MOE_WEIGHTED_REDUCTION_MAX_EXPERTS_X4 15

template <int N_EXPERT>
static bool launch_moe_weighted_reduction_x4(const float * experts, const float * expert_scale, const float * weights,
                                             float * dst, int64_t n_embd, int64_t n_tokens, int n_expert_used,
                                             cudaStream_t stream) {
    if constexpr (N_EXPERT > 1) {
        if (n_expert_used != N_EXPERT) {
            return launch_moe_weighted_reduction_x4<N_EXPERT - 1>(
                experts, expert_scale, weights, dst, n_embd, n_tokens, n_expert_used, stream);
        }
        constexpr int threads = 128;
        const dim3 blocks((n_embd / 4 + threads - 1) / threads, n_tokens, 1);
        moe_weighted_reduction_f32_x4<N_EXPERT>
            <<<blocks, threads, 0, stream>>>(experts, expert_scale, weights, dst, n_embd);
        return true;
    }
    return false;
}

static void launch_moe_weighted_reduction(const float * experts,
                                          const float * expert_scale,
                                          const float * weights,
                                          float *       dst,
                                          int64_t       n_embd,
                                          int64_t       n_tokens,
                                          int           n_expert_used,
                                          block_q8_1 *  dst_q8_1,
                                          cudaStream_t  stream) {
    constexpr int threads = 256;
    const dim3 blocks(n_tokens, (n_embd + threads - 1) / threads, 1);
    moe_weighted_reduction_f32
        <<<blocks, threads, 0, stream>>>(experts, expert_scale, weights, dst, n_embd, n_expert_used, dst_q8_1);
}

bool ggml_cuda_op_moe_weighted_reduction(ggml_backend_cuda_context & ctx,
                                         const ggml_tensor *         experts,
                                         const ggml_tensor *         expert_scale,
                                         const ggml_tensor *         weights,
                                         ggml_tensor *               dst,
                                         block_q8_1 *                dst_q8_1) {
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

    // the q8_1 copy needs whole, unpadded 32-value blocks per row and 32-lane warps
    const bool q8_1 = dst_q8_1 && n_embd % MATRIX_ROW_PADDING == 0 && ggml_cuda_info().devices[ctx.device].warp_size == QK8_1;

    // RDNA3.5 prompt batches without the q8_1 copy: 4 columns per thread (GGML_HIP_MOE_REDUCE_X4=0 turns it off)
    static const int env_x4 = ggml_cuda_env_switch("GGML_HIP_MOE_REDUCE_X4");
    if (!q8_1 && n_tokens >= 16 && n_embd % 4 == 0 && n_expert_used <= MOE_WEIGHTED_REDUCTION_MAX_EXPERTS_X4 &&
            ggml_cuda_switch_or_rdna3_5(env_x4, ggml_cuda_info().devices[ctx.device].cc) &&
            (uintptr_t) experts->data % 16 == 0 && (uintptr_t) dst->data % 16 == 0 &&
            launch_moe_weighted_reduction_x4<MOE_WEIGHTED_REDUCTION_MAX_EXPERTS_X4>(
                (const float *) experts->data, expert_scale ? (const float *) expert_scale->data : nullptr,
                (const float *) weights->data, (float *) dst->data, n_embd, n_tokens, (int) n_expert_used, stream)) {
        CUDA_CHECK(cudaGetLastError());
        return false;
    }

    launch_moe_weighted_reduction((const float *) experts->data,
                                  expert_scale ? (const float *) expert_scale->data : nullptr,
                                  (const float *) weights->data,
                                  (float *) dst->data, n_embd, n_tokens, (int) n_expert_used, q8_1 ? dst_q8_1 : nullptr, stream);
    CUDA_CHECK(cudaGetLastError());
    return q8_1;
}
