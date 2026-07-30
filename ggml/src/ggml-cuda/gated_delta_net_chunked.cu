#include "gated_delta_net_chunked.cuh"

// Chunked delta rule (WY representation / UT transform) for the scalar-gate gated delta net.
//
// The sequence is cut into chunks of GDN_CHUNK tokens and the rank-1 state updates inside a chunk
// are folded into (I + A)^-1, so the recurrence's inner loops become dense matmuls over LDS. One
// block walks the chunks of one (head, sequence), which keeps the per-chunk intermediates in LDS
// and the state in registers.
//
// Per head and chunk, rows are tokens and indices are (row, col):
//   g_cs        = inclusive cumsum of the log decay within the chunk, g_last = g_cs[GDN_CHUNK-1]
//   A[i][j]     = beta[i] * dot(K_i, K_j) * exp(g_cs[i] - g_cs[j])          for j <  i, else 0
//   Tinv        = (I + A)^-1, unit lower triangular
//   qk[i][j]    = dot(Q_i, K_j) * exp(g_cs[i] - g_cs[j])                    for j <= i, else 0
//   Y[r][c]     = V[r][c]*beta[r] - sum_a K[r][a]*beta[r]*exp(g_cs[r]) * S[a][c]
//   v_new[t][c] = sum_{r<=t} Tinv[t][r] * Y[r][c]
//   out[i][c]   = sum_a S[a][c]*Q[i][a]*exp(g_cs[i]) + sum_t v_new[t][c]*qk[i][t]
//   S[a][c]     = S[a][c]*exp(g_last) + sum_t K[t][a]*exp(g_last-g_cs[t]) * v_new[t][c]
//
// Everything after Tinv is independent per state column, which is what lets the state live in
// registers spread across the block.
//
// Two constraints keep a whole chunk inside the 64 KB LDS budget:
//   - GDN_CHUNK is 32. At 64 the chunk's U, W and qk alone need 80 KB.
//   - U and W are never materialised. v_new = U - W S, with U = Tinv (V beta) and
//     W = Tinv (K beta exp(g_cs)), is rewritten as v_new = Tinv (V beta - (K beta exp(g_cs)) S),
//     so one [GDN_CHUNK][GDN_HEAD_DIM] buffer holds V beta, becomes Y, then becomes v_new in place.

static constexpr int GDN_CHUNK      = 32;
static constexpr int GDN_HEAD_DIM   = 128;   // the only head size this path handles
static constexpr int GDN_BLOCK_SIZE = 1024;

// +1 breaks the power-of-two stride so that a column walk spreads over the LDS banks
static constexpr int GDN_HEAD_PITCH  = GDN_HEAD_DIM + 1;
static constexpr int GDN_CHUNK_PITCH = GDN_CHUNK + 1;

// Lanes cooperating on one adjacent pair of state columns. Sixteen of them fit inside a wave32, so
// their reductions stay cross-lane with no LDS traffic and no barrier, and holding a pair rather
// than a single column lets every staged K/Q element feed two FMAs.
static constexpr int GDN_COL_LANES     = 16;
static constexpr int GDN_ROWS_PER_LANE = GDN_HEAD_DIM / GDN_COL_LANES;

#if defined(GGML_USE_HIP) && defined(RDNA3_5)
template <int N>
static __device__ __forceinline__ float gdn_dpp_shl_add(float x) {
    // row_shl:N makes lane i read lane i+N within its row of 16, and the move folds into the add
    const int y = __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, x), 0x100 | N, 0xf, 0xf, true);
    return x + __builtin_bit_cast(float, y);
}
#endif

// Sum x across the GDN_COL_LANES lanes of a column pair; only lane 0 of the group ends up with the
// result, and the callers write from that lane. On RDNA the DPP form keeps the reduction on the
// VALU, where the cross-lane move is a free modifier on the add, while the portable shuffle lowers
// to ds_bpermute_b32 and would contend with the staged tiles for the LDS pipe.
static __device__ __forceinline__ float gdn_sum_to_lane0(float x) {
#if defined(GGML_USE_HIP) && defined(RDNA3_5)
    static_assert(GDN_COL_LANES == 16, "the row_shl chain below covers exactly 16 lanes");
    x = gdn_dpp_shl_add<1>(x);
    x = gdn_dpp_shl_add<2>(x);
    x = gdn_dpp_shl_add<4>(x);
    x = gdn_dpp_shl_add<8>(x);
    return x;
#else
#pragma unroll
    for (int mask = 1; mask < GDN_COL_LANES; mask <<= 1) {
        x += __shfl_xor_sync(0xffffffff, x, mask, WARP_SIZE);
    }
    return x;
#endif
}

// The decay cumsum is converted to base 2 once per chunk, so every decay afterwards is a single
// v_exp_f32 with no scaling multiply. The raw instruction also skips the denormal rescue that OCML
// wraps around expf, which costs nothing here: an exponent that far negative means a fully decayed
// state, and flushing it to zero is the intended result.
static constexpr float GDN_LOG2E = 1.44269504088896340736f;

static __device__ __forceinline__ float gdn_exp2(float x) {
#if defined(GGML_USE_HIP)
    return __builtin_amdgcn_exp2f(x);
#else
    return exp2f(x);
#endif
}

bool ggml_cuda_gdn_chunked_supported(bool kda, bool keep_rs, int64_t S_v, int64_t n_tokens) {
    static const bool enabled = []() {
        const char * env = getenv("GGML_CUDA_GDN_CHUNKED");
        return env == nullptr || std::atoi(env) != 0;
    }();

    if (!enabled || kda || keep_rs || S_v != GDN_HEAD_DIM || n_tokens <= GDN_CHUNK) {
        return false;
    }

    // The chunk size, the lane mapping, the 16-lane reduction, the LDS pitches and the occupancy
    // target are all sized for RDNA3.5: wave32, 64 LDS banks and 1536 VGPRs per SIMD.
    return GGML_CUDA_CC_IS_RDNA3_5(ggml_cuda_info().devices[ggml_cuda_get_device()].cc);
}

// On RDNA3.5 a block of GDN_BLOCK_SIZE threads is 32 waves, and two blocks per WGP come to 16 waves
// per SIMD; requesting that caps the register allocation at 1536/16 = 96, which is what keeps both
// blocks resident. The kernel is still compiled for every target in the build, and on those the
// request would be unreachable and -Wpass-failed would turn it into a build error.
#if defined(RDNA3_5)
#define GDN_MIN_WAVES_PER_SIMD 16
#else
#define GDN_MIN_WAVES_PER_SIMD 1
#endif

__global__ void __launch_bounds__(GDN_BLOCK_SIZE, GDN_MIN_WAVES_PER_SIMD)
gdn_chunked_f32(const float * __restrict__ q,
                const float * __restrict__ k,
                const float * __restrict__ v,
                const float * __restrict__ g,
                const float * __restrict__ beta,
                const float * __restrict__ state_in,
                float       * __restrict__ dst,
                float       * __restrict__ state_out,
                const int64_t n_tokens,
                const int64_t n_heads,
                const int64_t sq1, const int64_t sq2, const int64_t sq3,
                const int64_t sv1, const int64_t sv2, const int64_t sv3,
                const int64_t sb1, const int64_t sb2, const int64_t sb3,
                const uint3   neqk1_magic,
                const uint3   rq3_magic,
                const float   scale) {
    constexpr int CHUNK     = GDN_CHUNK;
    constexpr int HEAD_DIM  = GDN_HEAD_DIM;
    constexpr int COL_LANES = GDN_COL_LANES;
    constexpr int ROWS      = GDN_ROWS_PER_LANE;

    const int head     = blockIdx.x;
    const int sequence = blockIdx.y;
    const int tid      = threadIdx.x;

    const uint32_t head_qk = fastmodulo(head, neqk1_magic);
    const uint32_t seq_qk  = fastdiv(sequence, rq3_magic);

    const int col_pair = tid / COL_LANES;
    const int col_lane = tid - col_pair * COL_LANES;
    const int col0     = col_pair * 2;
    const int col1     = col0 + 1;

    __shared__ float s_k     [CHUNK][GDN_HEAD_PITCH];
    __shared__ float s_q     [CHUNK][GDN_HEAD_PITCH];
    __shared__ float s_y     [CHUNK][GDN_HEAD_PITCH];    // V*beta, then Y, then v_new
    __shared__ float s_tinv  [CHUNK][GDN_CHUNK_PITCH];   // A, then Tinv
    __shared__ float s_qk    [CHUNK][GDN_CHUNK_PITCH];
    __shared__ float s_g_cs  [CHUNK];
    __shared__ float s_beta  [CHUNK];
    __shared__ float s_beta_decay[CHUNK];                // beta[r] * exp(g_cs[r])
    __shared__ float s_decay     [CHUNK];                // exp(g_cs[i])
    __shared__ float s_decay_end [CHUNK];                // exp(g_last - g_cs[t])

    // state rows owned by this thread: row = u*COL_LANES + col_lane, which keeps the lanes of a
    // reduction on consecutive LDS banks
    float state0[ROWS];
    float state1[ROWS];

    const int64_t state_in_off = ((int64_t) sequence * n_heads + head) * HEAD_DIM * HEAD_DIM;
#pragma unroll
    for (int u = 0; u < ROWS; u++) {
        state0[u] = state_in[state_in_off + (int64_t) col0 * HEAD_DIM + u * COL_LANES + col_lane];
        state1[u] = state_in[state_in_off + (int64_t) col1 * HEAD_DIM + u * COL_LANES + col_lane];
    }

    // Only the lower triangles are written below: the packed loop covers j <= i and the inversion
    // stores zeros above the diagonal, so one clear here holds for every chunk.
    for (int idx = tid; idx < CHUNK * GDN_CHUNK_PITCH; idx += GDN_BLOCK_SIZE) {
        s_tinv[idx / GDN_CHUNK_PITCH][idx % GDN_CHUNK_PITCH] = 0.0f;
        s_qk  [idx / GDN_CHUNK_PITCH][idx % GDN_CHUNK_PITCH] = 0.0f;
    }

    const int64_t n_chunks = (n_tokens + CHUNK - 1) / CHUNK;

    for (int64_t chunk = 0; chunk < n_chunks; chunk++) {
        const int64_t tok0    = chunk * CHUNK;
        const int     n_valid = (int) (n_tokens - tok0 < CHUNK ? n_tokens - tok0 : CHUNK);

        __syncthreads();

        if (tid < CHUNK) {
            const int64_t gb_off = sequence * sb3 + (tok0 + tid) * sb2 + head * sb1;
            s_g_cs[tid] = tid < n_valid ? g[gb_off]    : 0.0f;
            s_beta[tid] = tid < n_valid ? beta[gb_off] : 0.0f;
        }
        __syncthreads();

        // CHUNK == WARP_SIZE, so the cumsum is one shuffle scan in wave 0, overlapped with the
        // K/Q/V staging done by the rest of the block. The scan result is still in registers here,
        // so the per-token decay tables cost no extra barrier.
        if (tid < WARP_SIZE) {
            float g_cs = s_g_cs[tid] * GDN_LOG2E;
#pragma unroll
            for (int off = 1; off < CHUNK; off <<= 1) {
                const float prev = __shfl_up_sync(0xffffffff, g_cs, off, WARP_SIZE);
                if (tid >= off) {
                    g_cs += prev;
                }
            }
            const float g_last = __shfl_sync(0xffffffff, g_cs, CHUNK - 1, WARP_SIZE);

            s_g_cs      [tid] = g_cs;
            s_beta_decay[tid] = s_beta[tid] * gdn_exp2(g_cs);
            s_decay     [tid] = gdn_exp2(g_cs);
            s_decay_end [tid] = gdn_exp2(g_last - g_cs);
        }

        for (int idx = tid; idx < CHUNK * HEAD_DIM; idx += GDN_BLOCK_SIZE) {
            const int  t     = idx / HEAD_DIM;
            const int  s     = idx - t * HEAD_DIM;
            const bool valid = t < n_valid;

            s_k[t][s] = valid ? k[seq_qk * sq3 + (tok0 + t) * sq2 + head_qk * sq1 + s] : 0.0f;
            s_q[t][s] = valid ? q[seq_qk * sq3 + (tok0 + t) * sq2 + head_qk * sq1 + s] * scale : 0.0f;
            s_y[t][s] = valid ? v[sequence * sv3 + (tok0 + t) * sv2 + head * sv1 + s] : 0.0f;
        }
        __syncthreads();

        const float g_last = s_g_cs[CHUNK - 1];

        // A and qk, both triangular. Mapping (i,j) to (tid/CHUNK, tid%CHUNK) would leave half the
        // machine idle: a wave covers one row i and still issues the whole dot product with only
        // its j < i lanes unmasked. The two triangles hold 496 + 496 + 32 entries, exactly the
        // block size, so a triangle-to-rectangle fold gives every lane one entry to compute.
        {
            constexpr int n_strict = CHUNK * (CHUNK - 1) / 2;

            int  row_i, col_j;
            bool is_qk;
            if (tid < 2 * n_strict) {
                const int tri = tid < n_strict ? tid : tid - n_strict;

                is_qk = tid >= n_strict;

                const int blk = tri / (CHUNK / 2);
                const int off = tri - blk * (CHUNK / 2);
                if (off <= blk) {
                    row_i = blk + 1;
                    col_j = off;
                } else {
                    row_i = CHUNK - 1 - blk;
                    col_j = CHUNK - 1 - off;
                }
            } else {
                row_i = col_j = tid - 2 * n_strict;
                is_qk = true;
            }

            const float * __restrict__ lhs = is_qk ? &s_q[row_i][0] : &s_k[row_i][0];

            float acc0 = 0.0f;
            float acc1 = 0.0f;
            for (int s = 0; s < HEAD_DIM; s += 2) {
                acc0 += lhs[s]     * s_k[col_j][s];
                acc1 += lhs[s + 1] * s_k[col_j][s + 1];
            }
            const float dot_decayed = (acc0 + acc1) * gdn_exp2(s_g_cs[row_i] - s_g_cs[col_j]);

            if (is_qk) {
                s_qk[row_i][col_j] = dot_decayed;
            } else {
                s_tinv[row_i][col_j] = s_beta[row_i] * dot_decayed;
            }
        }
        __syncthreads();

        // Tinv = (I+A)^-1 by right-looking forward substitution: once x[i] is final, subtract
        // A[r][i]*x[i] from the rows below it. Lane r owns row r, so the update is a plain FMA with
        // no cross-lane reduction. One column per wave, and A is read-only until the store.
        {
            const int lane = tid & (WARP_SIZE - 1);
            const int col  = tid / WARP_SIZE;

            float x = lane == col ? 1.0f : 0.0f;
            for (int i = col; i < CHUNK; i++) {
                const float x_i = __shfl_sync(0xffffffff, x, i, WARP_SIZE);
                if (lane > i) {
                    x -= s_tinv[lane][i] * x_i;
                }
            }
            __syncthreads();
            s_tinv[lane][col] = x;
        }
        __syncthreads();

        // Y[r][c] = V[r][c]*beta[r] - sum_a K[r][a]*beta[r]*exp(g_cs[r]) * S[a][c]
        for (int r = 0; r < CHUNK; r++) {
            float ks0 = 0.0f;
            float ks1 = 0.0f;
#pragma unroll
            for (int u = 0; u < ROWS; u++) {
                const float k_ra = s_k[r][u * COL_LANES + col_lane];
                ks0 += k_ra * state0[u];
                ks1 += k_ra * state1[u];
            }
            ks0 = gdn_sum_to_lane0(ks0);
            ks1 = gdn_sum_to_lane0(ks1);
            if (col_lane == 0) {
                s_y[r][col0] = s_y[r][col0] * s_beta[r] - s_beta_decay[r] * ks0;
                s_y[r][col1] = s_y[r][col1] * s_beta[r] - s_beta_decay[r] * ks1;
            }
        }
        __syncthreads();

        // v_new = Tinv * Y, in place. Tinv is lower triangular, so row t only reads Y[0..t]:
        // accumulating the upper half into registers before storing it leaves the lower half
        // untouched for the second pass, and no second [CHUNK][HEAD_DIM] buffer is needed.
        {
            const int row_hi = CHUNK / 2 + col_lane;
            float     hi0    = 0.0f;
            float     hi1    = 0.0f;
            for (int r = 0; r <= row_hi; r++) {
                const float tinv = s_tinv[row_hi][r];
                hi0 += tinv * s_y[r][col0];
                hi1 += tinv * s_y[r][col1];
            }
            __syncthreads();
            s_y[row_hi][col0] = hi0;
            s_y[row_hi][col1] = hi1;

            const int row_lo = col_lane;
            float     lo0    = 0.0f;
            float     lo1    = 0.0f;
            for (int r = 0; r <= row_lo; r++) {
                const float tinv = s_tinv[row_lo][r];
                lo0 += tinv * s_y[r][col0];
                lo1 += tinv * s_y[r][col1];
            }
            __syncthreads();
            s_y[row_lo][col0] = lo0;
            s_y[row_lo][col1] = lo1;
        }
        __syncthreads();

        // out[i][c] = sum_a S[a][c]*Q[i][a]*exp(g_cs[i]) + sum_t v_new[t][c]*qk[i][t]
        for (int i = 0; i < CHUNK; i++) {
            float qs0 = 0.0f;
            float qs1 = 0.0f;
#pragma unroll
            for (int u = 0; u < ROWS; u++) {
                const float q_ia = s_q[i][u * COL_LANES + col_lane];
                qs0 += q_ia * state0[u];
                qs1 += q_ia * state1[u];
            }

            float vk0 = 0.0f;
            float vk1 = 0.0f;
#pragma unroll
            for (int w = 0; w < CHUNK / COL_LANES; w++) {
                const int   t     = col_lane * (CHUNK / COL_LANES) + w;
                const float qk_it = s_qk[i][t];
                vk0 += s_y[t][col0] * qk_it;
                vk1 += s_y[t][col1] * qk_it;
            }

            // s_decay[i] is lane-uniform and the reduction is linear, so applying it first leaves
            // one value per column to reduce instead of two.
            float out0 = qs0 * s_decay[i] + vk0;
            float out1 = qs1 * s_decay[i] + vk1;
            out0 = gdn_sum_to_lane0(out0);
            out1 = gdn_sum_to_lane0(out1);

            if (col_lane == 0 && i < n_valid) {
                const int64_t dst_off = ((int64_t) sequence * n_tokens + tok0 + i) * n_heads * HEAD_DIM
                                      + (int64_t) head * HEAD_DIM;
                // col0 is even and HEAD_DIM is even, so the pair is 8-byte aligned
                *(float2 *) &dst[dst_off + col0] = make_float2(out0, out1);
            }
        }

        // S[a][c] = S[a][c]*exp(g_last) + sum_t K[t][a]*exp(g_last-g_cs[t]) * v_new[t][c]
        const float chunk_decay = gdn_exp2(g_last);
#pragma unroll
        for (int u = 0; u < ROWS; u++) {
            state0[u] *= chunk_decay;
            state1[u] *= chunk_decay;
        }

        for (int t = 0; t < CHUNK; t++) {
            const float decayed0 = s_decay_end[t] * s_y[t][col0];
            const float decayed1 = s_decay_end[t] * s_y[t][col1];
#pragma unroll
            for (int u = 0; u < ROWS; u++) {
                const float k_ta = s_k[t][u * COL_LANES + col_lane];
                state0[u] += k_ta * decayed0;
                state1[u] += k_ta * decayed1;
            }
        }
    }

    const int64_t state_out_off = ((int64_t) sequence * n_heads + head) * HEAD_DIM * HEAD_DIM;
#pragma unroll
    for (int u = 0; u < ROWS; u++) {
        state_out[state_out_off + (int64_t) col0 * HEAD_DIM + u * COL_LANES + col_lane] = state0[u];
        state_out[state_out_off + (int64_t) col1 * HEAD_DIM + u * COL_LANES + col_lane] = state1[u];
    }
}

void ggml_cuda_gdn_chunked(ggml_backend_cuda_context & ctx, const ggml_cuda_gdn_chunked_args & args) {
    GGML_ASSERT(args.S_v == GDN_HEAD_DIM);

    const uint3 neqk1_magic = init_fastdiv_values(args.neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(args.rq3);

    const dim3 block_nums(args.H, args.n_seqs, 1);
    const dim3 block_dims(GDN_BLOCK_SIZE, 1, 1);

    const ggml_cuda_kernel_launch_params launch_params(block_nums, block_dims, 0, ctx.stream());

    ggml_cuda_kernel_launch(gdn_chunked_f32, launch_params,
        args.q, args.k, args.v, args.g, args.beta, args.state_in, args.dst, args.state_out,
        args.n_tokens, args.H,
        args.sq1, args.sq2, args.sq3, args.sv1, args.sv2, args.sv3, args.sb1, args.sb2, args.sb3,
        neqk1_magic, rq3_magic, args.scale);
}
