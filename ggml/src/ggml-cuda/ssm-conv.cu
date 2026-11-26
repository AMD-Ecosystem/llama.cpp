#include "common.cuh"
#include "ssm-conv.cuh"
#include "unary.cuh"

template <bool apply_silu, size_t split_d_inner, size_t d_conv>
static __global__ void ssm_conv_f32(const float * src0_ptr, const float * src1_ptr,
                                    const float * bias_ptr,
                                    const int src0_nb0, const int src0_nb1, const int src0_nb2, const int src1_nb1,
                                    float * dst_ptr, const int dst_nb0, const int dst_nb1, const int dst_nb2,
                                    const int64_t n_t) {
    ggml_cuda_pdl_lc();
    const float * GGML_CUDA_RESTRICT src0 = src0_ptr;
    const float * GGML_CUDA_RESTRICT src1 = src1_ptr;
    const float * GGML_CUDA_RESTRICT bias = bias_ptr;
    float       * GGML_CUDA_RESTRICT dst  = dst_ptr;
    GGML_UNUSED(src0_nb0);
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block = (float *) ((char *) dst + bidx * dst_nb2 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    float x[d_conv] = { 0.0f };
    float w[d_conv] = { 0.0f };

    ggml_cuda_pdl_sync();
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    for (int64_t i = 0; i < n_t; i++) {
        float sumf = 0.0f;

        if (i == 0) {
            for (size_t j = 0; j < d_conv; j++) {
                x[j] = x_block[tid * stride_x + j];
            }
        } else {
            x[(i - 1) % d_conv] = x_block[tid * stride_x + i + d_conv - 1];
        }

#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += x[(i + j) % d_conv] * w[j];
        }
        sumf += b;
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

template <bool apply_silu, size_t split_d_inner, size_t d_conv, int64_t split_n_t>
static __global__ void ssm_conv_long_token_f32(const float * __restrict__ src0, const float * __restrict__ src1,
                                               const float * __restrict__ bias,
                                               const int src0_nb0, const int src0_nb1, const int src0_nb2,
                                               const int src1_nb1, float * __restrict__ dst, const int dst_nb0,
                                               const int dst_nb1, const int dst_nb2, const int64_t n_t) {
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const int bidz = blockIdx.z;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1 +
                                             bidz * split_n_t * src0_nb0);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block =
        (float *) ((char *) dst + bidx * dst_nb2 + bidz * split_n_t * dst_nb1 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    const int64_t local_n_t = min(split_n_t, n_t - bidz * split_n_t);
    const int     n_cols    = d_conv - 1 + split_n_t;

    extern __shared__ float smem[];

    constexpr int load_cols   = d_conv - 1 + split_n_t;
    constexpr int total_elems = split_d_inner * load_cols;
    int row = tid / load_cols;
    int col = tid % load_cols;
#pragma unroll
    for (int idx = 0; idx < total_elems; idx += split_d_inner) {
        if (row < (int)split_d_inner) {
            smem[row * n_cols + col] = x_block[row * stride_x + col];
        }

        col += split_d_inner;
        row += col / load_cols;
        col  = col % load_cols;
        if (idx >= total_elems - tid - split_d_inner) {
            break;
        }
    }
    __syncthreads();

    // Load weights into registers (done once, small)
    float w[d_conv] = { 0.0f };
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    // Compute from shared memory
    for (int64_t i = 0; i < local_n_t; i++) {
        float sumf = 0.0f;
#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += smem[tid * n_cols + i + j] * w[j];
        }
        sumf += b;
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

template <bool apply_silu>
static void ssm_conv_f32_cuda(const float * src0, const float * src1, const float * bias, const int src0_nb0, const int src0_nb1,
                              const int src0_nb2, const int src1_nb1, float * dst, const int dst_nb0, const int dst_nb1,
                              const int dst_nb2, const int64_t nc, const int64_t nr, const int64_t n_t,
                              const int64_t n_s, cudaStream_t stream) {
    const int threads = 128;
    GGML_ASSERT(nr % threads == 0);

    auto launch_kernel = [&](auto NC) {
        constexpr int kNC = decltype(NC)::value;
        if (n_t <= 32) {
            const dim3 blocks(n_s, (nr + threads - 1) / threads, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks, threads, 0, stream);
            ggml_cuda_kernel_launch(ssm_conv_f32<apply_silu, threads, kNC>, launch_params, src0, src1, bias, src0_nb0, src0_nb1,
                                                                        src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        } else {
            const int64_t split_n_t = 32;
            dim3          blocks(n_s, (nr + threads - 1) / threads, (n_t + split_n_t - 1) / split_n_t);
            const size_t  smem_size = threads * (kNC - 1 + split_n_t) * sizeof(float);
            ssm_conv_long_token_f32<apply_silu, threads, kNC, split_n_t><<<blocks, threads, smem_size, stream>>>(
                src0, src1, bias, src0_nb0, src0_nb1, src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        }
    };

    switch (nc) {
        case 3:  launch_kernel(std::integral_constant<int, 3 >{}); break;
        case 4:  launch_kernel(std::integral_constant<int, 4 >{}); break;
        case 5:  launch_kernel(std::integral_constant<int, 5 >{}); break;
        case 7:  launch_kernel(std::integral_constant<int, 7 >{}); break;
        case 9:  launch_kernel(std::integral_constant<int, 9 >{}); break;
        case 15: launch_kernel(std::integral_constant<int, 15>{}); break;
        default: GGML_ABORT("Only support kernel sizes 3, 4, 5, 7, 9, 15 right now.");
    }
}

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node, ggml_tensor * silu_dst) {
    const struct ggml_tensor * src0 = dst->src[0];  // conv_x
    const struct ggml_tensor * src1 = dst->src[1];  // conv1d.weight
    const bool fuse_bias = bias_add_node != nullptr;
    const bool fuse_silu = silu_dst != nullptr;

    // bias always comes with silu.
    GGML_ASSERT(!fuse_bias || fuse_silu);

    // The bias (when fused) is the non-conv operand of the ADD node.
    const struct ggml_tensor * bias = fuse_bias ? (bias_add_node->src[0] == dst ? bias_add_node->src[1] : bias_add_node->src[0]) : nullptr;

    // When fusing, write to silu_dst (the node downstream references).
    const struct ggml_tensor * out = fuse_silu ? silu_dst : dst;

    const int64_t nc  = src1->ne[0];                // d_conv
    const int64_t nr  = src0->ne[1];                // d_inner
    const int64_t n_t = out->ne[1];                 // tokens per sequence
    const int64_t n_s = out->ne[2];                 // number of sequences in the batch

    GGML_ASSERT(out->ne[0] == nr);
    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(src0->nb[1] == src0->ne[0] * sizeof(float));

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    const float * bias_d = fuse_bias ? (const float *) bias->data : nullptr;
    float *       dst_d  = (float *) out->data;
    cudaStream_t  stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(out->type == GGML_TYPE_F32);
    if (fuse_bias) {
        GGML_ASSERT(bias->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(bias));
        GGML_ASSERT(ggml_nelements(bias) == nr);
    }

    if (fuse_silu) {
        ssm_conv_f32_cuda<true>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream);
    } else {
        ssm_conv_f32_cuda<false>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream);
    }
}

// RMS_NORM + SCALE of the rows of 128 channels [c0[k], c1[k]) of the convolution step's result into dst[k]
struct ssm_conv_step_norm {
    float * dst[2];
    int     c0[2];
    int     c1[2];
    float   eps[2];
    float   scale[2];
};

// One token of a short convolution with its state: channel c convolves its d_conv - 1 state values, read in place in
// the cache row rows[0] indexes, and x[c] with the weights as ssm_conv_f32 does (then SiLU), and writes the window's
// last d_conv - 1 values back as the new state. Each thread reads and writes the state of its own channel only.
// With has_norm, the blocks of 128 channels in a range of norm also normalize them as
// rms_norm_f32<WARP_SIZE, 1, true, false, false, true> does a row.
template <int d_conv, bool has_norm>
static __global__ void ssm_conv_step_f32(const float * __restrict__ cache, const int32_t * __restrict__ rows,
                                         const int64_t row_stride, const float * __restrict__ x,
                                         const float * __restrict__ w, const float * __restrict__ bias,
                                         float * __restrict__ state_out, float * __restrict__ dst, const int n_channels,
                                         const ssm_conv_step_norm norm) {
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= n_channels) {
        return;
    }
    const float * state = cache + rows[0]*row_stride + (int64_t) c*(d_conv - 1);

    float xw[d_conv];
#pragma unroll
    for (int j = 0; j < d_conv - 1; ++j) {
        xw[j] = state[j];
    }
    xw[d_conv - 1] = x[c];

    float wc[d_conv];
#pragma unroll
    for (int j = 0; j < d_conv; ++j) {
        wc[j] = w[c*d_conv + j];
    }
    const float b = bias != nullptr ? bias[c] : 0.0f;

    float sumf = 0.0f;
#pragma unroll
    for (int j = 0; j < d_conv; j++) {
        sumf += xw[j] * wc[j];
    }
    sumf += b;
    const float y = ggml_cuda_op_silu_single(sumf);
    dst[c] = y;

    if constexpr (has_norm) {
        constexpr int ncols = 128;
        const int c_block = blockIdx.x*blockDim.x;
        int k = -1;
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            if (norm.dst[kk] != nullptr && c_block >= norm.c0[kk] && c_block < norm.c1[kk]) {
                k = kk;
            }
        }
        if (k >= 0) {
            __shared__ float s_y[ncols];
            s_y[threadIdx.x] = y;
            __syncthreads();
            if (threadIdx.x < WARP_SIZE) {
                const float4 xs = ((const float4 *) s_y)[threadIdx.x];
                float tmp = 0.0f;
                tmp += xs.x*xs.x + xs.y*xs.y + xs.z*xs.z + xs.w*xs.w;
                tmp = warp_reduce_sum(tmp);
                const float scale = rsqrtf(tmp / ncols + norm.eps[k]);
                float4 r = make_float4(scale*xs.x, scale*xs.y, scale*xs.z, scale*xs.w);
                r.x *= norm.scale[k];
                r.y *= norm.scale[k];
                r.z *= norm.scale[k];
                r.w *= norm.scale[k];
                ((float4 *) (norm.dst[k] + (c_block - norm.c0[k])))[threadIdx.x] = r;
            }
        }
    }

#pragma unroll
    for (int j = 0; j < d_conv - 1; ++j) {
        state_out[(int64_t) c*(d_conv - 1) + j] = xw[j + 1];
    }
}

void ggml_cuda_op_ssm_conv_step(ggml_backend_cuda_context & ctx, const ggml_tensor * state_rows, const ggml_tensor * x,
                                const ggml_tensor * conv_w, ggml_tensor * state_out, ggml_tensor * dst,
                                const ggml_tensor * const * norm, const ggml_tensor * const * norm_scale) {
    const int d_conv     = conv_w->ne[0];
    const int n_channels = conv_w->ne[1];
    GGML_ASSERT(d_conv == 4);
    const int threads = 128;

    ssm_conv_step_norm args = {};
    bool has_norm = false;
    for (int k = 0; norm != nullptr && k < 2; ++k) {
        if (norm[k] == nullptr) {
            continue;
        }
        const ggml_tensor * view = norm[k]->src[0];
        GGML_ASSERT(n_channels % threads == 0 && view->ne[0] == threads && view->view_offs % (threads*sizeof(float)) == 0);
        args.dst[k]   = (float *) norm_scale[k]->data;
        args.c0[k]    = view->view_offs / sizeof(float);
        args.c1[k]    = args.c0[k] + ggml_nelements(view);
        args.eps[k]   = ggml_get_op_params_f32(norm[k], 0);
        args.scale[k] = ggml_get_op_params_f32(norm_scale[k], 0);
        has_norm = true;
    }

    auto kernel = has_norm ? ssm_conv_step_f32<4, true> : ssm_conv_step_f32<4, false>;
    kernel<<<(n_channels + threads - 1) / threads, threads, 0, ctx.stream()>>>(
        (const float *) state_rows->src[0]->data, (const int32_t *) state_rows->src[1]->data,
        state_rows->src[0]->nb[1] / sizeof(float), (const float *) x->data, (const float *) conv_w->data, nullptr,
        (float *) state_out->data, (float *) dst->data, n_channels, args);
}
