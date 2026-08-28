#include "ggml.h"
#include "common.cuh"
#include "mul_mat_f32_tall_skinny.cuh"

#if defined(GGML_USE_HIP)

// Wave32 sum reduction, result in lane 0. On RDNA3.5 (the only arch this kernel runs on)
// use row-shift DPP to tree-add within each 16-lane row, then one permlanex16 to add the
// two rows. Those builtins need gfx10+ instructions, so other archs (compiled but never
// launched) fall back to a portable shuffle reduction.
#if defined(RDNA3_5)
#define DPP_ROW_SHIFT_ADD(v, ctrl) (v) += __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), ctrl, 0xf, 0xf, 1))
static __device__ __forceinline__ float permute_xor16(float v) {
    return __int_as_float(__builtin_amdgcn_permlanex16(__float_as_int(v), __float_as_int(v),
                                                       0x76543210u, 0xFEDCBA98u, false, false));
}
static __device__ __forceinline__ float warp_reduce_sum(float v) {
    DPP_ROW_SHIFT_ADD(v, 0x108); DPP_ROW_SHIFT_ADD(v, 0x104); DPP_ROW_SHIFT_ADD(v, 0x102); DPP_ROW_SHIFT_ADD(v, 0x101);
    v += permute_xor16(v);
    return v;
}
#else
static __device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_xor_sync(0xffffffff, v, offset, 32);
    return v;
}
#endif // defined(RDNA3_5)

// Software-pipelined weight stream shared by both kernels. Two alternating register
// sets (cur/nxt) keep DEPTH*WROWS b128 loads in flight across the loop backedge; the
// sched_barriers pin the load-then-compute order so the counter does not drain to zero.
// Each lane reduces a K/32 stripe into acc[WROWS][ACOLS]; act_cols_lds holds ACOLS
// activation columns (float4 per K step). STEP indices count 32-wide K steps.
#define TSN_PIPE_STAGE(cur, nxt, load_step, use_step) do {                                       \
    _Pragma("unroll")                                                                            \
    for (int d = 0; d < DEPTH; ++d) {                                                            \
        const int load_k = lane + ((load_step) + d) * 32;                                        \
        _Pragma("unroll")                                                                        \
        for (int c = 0; c < WROWS; ++c) nxt[d][c] = weight_row[c][load_k];                       \
        __builtin_amdgcn_sched_barrier(0);                                                       \
        const int use_k = lane + ((use_step) + d) * 32;                                          \
        _Pragma("unroll")                                                                        \
        for (int c = 0; c < WROWS; ++c) { const float4 w = cur[d][c];                            \
            _Pragma("unroll")                                                                    \
            for (int e = 0; e < ACOLS; ++e) { const float4 a = act_cols_lds[e * k4 + use_k];     \
                acc[c][e] += w.x*a.x + w.y*a.y + w.z*a.z + w.w*a.w; } }                          \
        __builtin_amdgcn_sched_barrier(0);                                                       \
    } } while (0)
#define TSN_PIPE_DRAIN(cur, use_step) do {                                                       \
    _Pragma("unroll")                                                                            \
    for (int d = 0; d < DEPTH; ++d) {                                                            \
        const int use_k = lane + ((use_step) + d) * 32;                                          \
        _Pragma("unroll")                                                                        \
        for (int c = 0; c < WROWS; ++c) { const float4 w = cur[d][c];                            \
            _Pragma("unroll")                                                                    \
            for (int e = 0; e < ACOLS; ++e) { const float4 a = act_cols_lds[e * k4 + use_k];     \
                acc[c][e] += w.x*a.x + w.y*a.y + w.z*a.z + w.w*a.w; } }                          \
    } } while (0)

// Runs the DEPTH-deep prologue, the 2*DEPTH-unrolled steady loop and the epilogue over
// the K dimension for the WROWS x ACOLS output tile owned by this warp, then writes it.
// Requires n_ksteps % (2*DEPTH) == 0.
#define TSN_STREAM_AND_STORE(n_lo, n_hi) do {                                                    \
    _Pragma("unroll")                                                                            \
    for (int c = 0; c < WROWS; ++c) { const int row = min(nb + c, N - 1);                        \
        weight_row[c] = (const float4 *) __builtin_assume_aligned(x + (int64_t) row * ldx, 16); }\
    float acc[WROWS][ACOLS];                                                                     \
    _Pragma("unroll")                                                                            \
    for (int c = 0; c < WROWS; ++c) _Pragma("unroll") for (int e = 0; e < ACOLS; ++e) acc[c][e] = 0.f; \
    float4 w_cur[DEPTH][WROWS], w_nxt[DEPTH][WROWS];                                             \
    _Pragma("unroll")                                                                            \
    for (int d = 0; d < DEPTH; ++d) _Pragma("unroll") for (int c = 0; c < WROWS; ++c) w_cur[d][c] = weight_row[c][lane + d * 32]; \
    int step = DEPTH;                                                                            \
    for (; step + 2 * DEPTH <= n_ksteps; step += 2 * DEPTH) {                                    \
        TSN_PIPE_STAGE(w_cur, w_nxt, step,         step - DEPTH);                                \
        TSN_PIPE_STAGE(w_nxt, w_cur, step + DEPTH, step);                                        \
    }                                                                                            \
    TSN_PIPE_STAGE(w_cur, w_nxt, step, step - DEPTH);                                            \
    TSN_PIPE_DRAIN(w_nxt, step);                                                                 \
    _Pragma("unroll")                                                                            \
    for (int c = 0; c < WROWS; ++c) _Pragma("unroll") for (int e = 0; e < ACOLS; ++e) {          \
        const float sum = warp_reduce_sum(acc[c][e]);                                            \
        if (lane == 0 && (nb + c) < (n_hi) && (mb + e) < M) dst[(int64_t)(mb + e) * N + (nb + c)] = sum; } \
    } while (0)

// dst[N,M] = weight[K,N]^T . activation[K,M]. The activation is staged resident in LDS
// and the weight is streamed from global memory. Persistent over M tiles: while the
// weight stream of the current M tile is computed, the next tile's activation is
// prefetched into registers and stored into the other of two LDS buffers, overlapping
// the activation load with the weight compute. vec_per_thread = (K/4)/blockDim.x.
template<int WROWS, int ACOLS, int DEPTH, int VEC_PER_THREAD>
static __global__ void __launch_bounds__(512)
mul_mat_f32_tall_skinny_double_buffered(const float * __restrict__ x, const float * __restrict__ y,
        float * __restrict__ dst, const int K, const int N, const int M, const int ldx) {
    const int block_size = blockDim.x, k4 = K / 4, n_ksteps = k4 / 32;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, nwarps = block_size >> 5;
    const int m_stride = gridDim.x * ACOLS;
    extern __shared__ float smem[];
    float4 * act_buf0 = (float4 *) smem;
    float4 * act_buf1 = act_buf0 + (int64_t) ACOLS * k4;
    float4 prefetch[ACOLS][VEC_PER_THREAD > 0 ? VEC_PER_THREAD : 1];
    {   const int mb = blockIdx.x * ACOLS;
        #pragma unroll
        for (int e = 0; e < ACOLS; ++e) { const int col = min(mb + e, M - 1);
            const float4 * yr = (const float4 *) __builtin_assume_aligned(y + (int64_t) col * K, 16);
            #pragma unroll
            for (int p = 0; p < VEC_PER_THREAD; ++p) { const int j = tid + p * block_size;
                prefetch[e][p] = mb < M ? yr[j] : make_float4(0, 0, 0, 0); } } }
    int buf = 0;
    for (int mb = blockIdx.x * ACOLS; mb < M; mb += m_stride) {
        float4 * act_cols_lds = buf ? act_buf1 : act_buf0;
        #pragma unroll
        for (int e = 0; e < ACOLS; ++e)
            #pragma unroll
            for (int p = 0; p < VEC_PER_THREAD; ++p) { const int j = tid + p * block_size; act_cols_lds[e * k4 + j] = prefetch[e][p]; }
        __syncthreads();
        const int next_mb = mb + m_stride;
        #pragma unroll
        for (int e = 0; e < ACOLS; ++e) { const int col = min(next_mb + e, M - 1);
            const float4 * yr = (const float4 *) __builtin_assume_aligned(y + (int64_t) col * K, 16);
            #pragma unroll
            for (int p = 0; p < VEC_PER_THREAD; ++p) { const int j = tid + p * block_size;
                prefetch[e][p] = next_mb < M ? yr[j] : make_float4(0, 0, 0, 0); } }
        for (int nb = warp * WROWS; nb < N; nb += nwarps * WROWS) {
            const float4 * weight_row[WROWS];
            TSN_STREAM_AND_STORE(0, N);
        }
        __syncthreads();
        buf ^= 1;
    }
}

// Same weight-streamed kernel, but the N (weight-row) dimension is split across gridDim.y
// so more blocks run concurrently (higher occupancy) without extra accumulators or a
// cross-block reduction, since each split writes a disjoint N range of dst.
template<int WROWS, int ACOLS, int DEPTH>
static __global__ void __launch_bounds__(512, 1)
mul_mat_f32_tall_skinny_split_n(const float * __restrict__ x, const float * __restrict__ y,
        float * __restrict__ dst, const int K, const int N, const int M, const int ldx) {
    const int block_size = blockDim.x, k4 = K / 4, n_ksteps = k4 / 32;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, nwarps = block_size >> 5;
    const int n_splits = gridDim.y, split = blockIdx.y;
    const int n_lo = (int) (((int64_t) N * split) / n_splits);
    const int n_hi = (int) (((int64_t) N * (split + 1)) / n_splits);
    extern __shared__ float smem[];
    float4 * act_cols_lds = (float4 *) smem;
    for (int mb = blockIdx.x * ACOLS; mb < M; mb += gridDim.x * ACOLS) {
        #pragma unroll
        for (int e = 0; e < ACOLS; ++e) { const int col = min(mb + e, M - 1);
            const float4 * yr = (const float4 *) __builtin_assume_aligned(y + (int64_t) col * K, 16);
            for (int p = tid; p < k4; p += block_size) act_cols_lds[e * k4 + p] = yr[p]; }
        __syncthreads();
        for (int nb = n_lo + warp * WROWS; nb < n_hi; nb += nwarps * WROWS) {
            const float4 * weight_row[WROWS];
            TSN_STREAM_AND_STORE(n_lo, n_hi);
        }
        __syncthreads();
    }
}

#undef TSN_STREAM_AND_STORE
#undef TSN_PIPE_STAGE
#undef TSN_PIPE_DRAIN
#undef DPP_ROW_SHIFT_ADD

#endif // defined(GGML_USE_HIP)

bool ggml_cuda_should_use_mul_mat_f32_tall_skinny(const ggml_tensor * src0, const ggml_tensor * src1,
        const ggml_tensor * dst, int cc) {
#if defined(GGML_USE_HIP)
    if (!GGML_CUDA_CC_IS_RDNA3_5(cc)) {
        return false;
    }
    if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        return false;
    }
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) {
        return false;
    }
    const int64_t K = src0->ne[0], N = src0->ne[1];
    return K == 2048 && N == 32 && src1->ne[0] == K;
#else
    GGML_UNUSED(src0); GGML_UNUSED(src1); GGML_UNUSED(dst); GGML_UNUSED(cc);
    return false;
#endif
}

void ggml_cuda_mul_mat_f32_tall_skinny(ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
#if defined(GGML_USE_HIP)
    const int K   = (int) src0->ne[0];
    const int N   = (int) src0->ne[1];
    const int M   = (int) src1->ne[1];
    const int ldx = (int) (src0->nb[1] / sizeof(float));
    const float * x = (const float *) src0->data;
    const float * y = (const float *) src1->data;
    float       * d = (float       *) dst->data;
    cudaStream_t stream = ctx.stream();

    constexpr int block_size = 256;
    constexpr int vec_per_thread = (2048 / 4) / block_size;
    // Per-M configuration: small batches use the double-buffered kernel, mid batches the
    // split-N kernel (128 blocks up to M=256, then 64), large batches the double-buffered
    // kernel again with a wider activation tile.
    if (M <= 128) {
        constexpr int act_cols = 2;
        const dim3 grid(64, 1, 1), block(block_size, 1, 1);
        const size_t nbytes_shared = (size_t) 2 * act_cols * K * sizeof(float);
        mul_mat_f32_tall_skinny_double_buffered<4, act_cols, 2, vec_per_thread>
            <<<grid, block, nbytes_shared, stream>>>(x, y, d, K, N, M, ldx);
    } else if (M <= 1024) {
        constexpr int act_cols = 4;
        const int grid_x = M <= 256 ? 128 : 64;
        const dim3 grid(grid_x, 1, 1), block(block_size, 1, 1);
        const size_t nbytes_shared = (size_t) act_cols * K * sizeof(float);
        mul_mat_f32_tall_skinny_split_n<4, act_cols, 1>
            <<<grid, block, nbytes_shared, stream>>>(x, y, d, K, N, M, ldx);
    } else {
        constexpr int act_cols = 4;
        const dim3 grid(40, 1, 1), block(block_size, 1, 1);
        const size_t nbytes_shared = (size_t) 2 * act_cols * K * sizeof(float);
        mul_mat_f32_tall_skinny_double_buffered<4, act_cols, 1, vec_per_thread>
            <<<grid, block, nbytes_shared, stream>>>(x, y, d, K, N, M, ldx);
    }
#else
    GGML_UNUSED(ctx); GGML_UNUSED(src0); GGML_UNUSED(src1); GGML_UNUSED(dst);
    GGML_ABORT("ggml_cuda_mul_mat_f32_tall_skinny is HIP-only");
#endif
}
