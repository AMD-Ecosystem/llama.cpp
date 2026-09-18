#include "mmvdq.cuh"
#include "vecdotdq.cuh"
#include "unary.cuh"

#include <cstdlib>

// -1 = unset (use arch default), 0 = force off, 1 = force on.
static int dq_env_override(const char * name) {
    const char * v = getenv(name);
    if (!v) return -1;
    return (v[0] == '0' && v[1] == '\0') ? 0 : 1;
}

static bool ggml_cuda_dq_mmv_enabled(bool arch_default) {
    static const int ov = dq_env_override("GGML_CUDA_DQ_MMV");
    return ov < 0 ? arch_default : (bool) ov;
}

static bool ggml_cuda_dq_q6k_enabled(bool arch_default) {
    static const int ov = dq_env_override("GGML_CUDA_DQ_Q6K");
    return ov < 0 ? arch_default : (bool) ov;
}

// Rows-per-block tuning knob. Only 1/2/4/8 are instantiated; anything else
// warns once and falls back to 1. Cached so we don't re-read the env per matvec.
static int dq_num_rows_init() {
    const char * v = getenv("GGML_CUDA_DQ_ROWS");
    if (!v) return 1;
    const int r = atoi(v);
    if (r == 1 || r == 2 || r == 4 || r == 8) return r;
    GGML_LOG_WARN("%s: unsupported GGML_CUDA_DQ_ROWS=%s (expected 1/2/4/8), using 1\n", __func__, v);
    return 1;
}

static int dq_num_rows() {
    static const int rows = dq_num_rows_init();
    return rows;
}

// ---- Q4_K geometry ----
struct dq_geom_q4_K {
    int itid, ix, v_im, q_offset, y_offset;
};
static __device__ __forceinline__ dq_geom_q4_K dq_setup_q4_K(int tid) {
    const int itid = tid % 16;
    const int ix   = tid / 16;
    const int il   = itid / 4;
    const int ir   = itid % 4;
    const int v_im = il / 2;
    const int v_in = il % 2;
    const int l0   = 4 * (2 * ir + v_in);
    return { itid, ix, v_im, 32 * v_im + l0, 64 * v_im + l0 };
}

// mul_mat_vec_q4_k shader: 16 threads process one super-block, each block
// computes NUM_ROWS output rows, activations are loaded once as float4 and
// reused across rows. No q8_1 activation pass (unlike mul_mat_vec_q).
template <int warp_size, int num_rows>
static __global__ void mul_mat_vec_dq_q4_K(
        const void * __restrict__ vx, const float * __restrict__ y, float * __restrict__ dst,
        const int ncols_x, const int nrows_x,
        const int32_t * __restrict__ ids, const int64_t stride_channel_x, const int64_t stride_channel_dst,
        const int64_t stride_channel_y, const int nchannels_y) {
    const int first_row = num_rows * blockIdx.x;
    const int nblocks   = ncols_x / QK_K;
    const int it_size   = warp_size / 16;

    // MoE (ids): blockIdx.y selects the expert slot; ids maps slot -> expert matrix.
    // channel_y picks this slot's activation column (per-expert ffn_down) or 0 when the
    // activation is broadcast/shared (gate/up, dense). Non-ids launches use grid.y=1,
    // ids=nullptr, nchannels_y=1 => every offset below collapses to 0.
    const int channel   = blockIdx.y;
    const int expert    = ids ? ids[channel] : 0;
    const int channel_y = nchannels_y > 1 ? channel % nchannels_y : 0;
    const dq_geom_q4_K g = dq_setup_q4_K(threadIdx.x);
    const block_q4_K * x = (const block_q4_K *) vx + expert * stride_channel_x;
    const float * yc = y + (int64_t) channel_y * stride_channel_y;
    float * dst_c = dst + channel * stride_channel_dst;

    float sumf[num_rows];
#pragma unroll
    for (int n = 0; n < num_rows; ++n) sumf[n] = 0.0f;

    for (int i = g.ix; i < nblocks; i += it_size) {
        const float * yb = yc + (int64_t) i * QK_K;
        const float4 by10  = *(const float4 *) (yb + g.y_offset      );
        const float4 by132 = *(const float4 *) (yb + g.y_offset +  32);
        const float4 by20  = *(const float4 *) (yb + g.y_offset + 128);
        const float4 by232 = *(const float4 *) (yb + g.y_offset + 160);

        const float sum10 = by10.x  + by10.y  + by10.z  + by10.w;
        const float sum32 = by132.x + by132.y + by132.z + by132.w;
        const float sum20 = by20.x  + by20.y  + by20.z  + by20.w;
        const float sum42 = by232.x + by232.y + by232.z + by232.w;

#pragma unroll
        for (int n = 0; n < num_rows; ++n) {
            const int row = min(first_row + n, nrows_x - 1);
            const block_q4_K * b = &x[(int64_t) row * nblocks + i];
            sumf[n] += dq_dot_q4_K(b, g.q_offset, g.v_im, by10, by132, by20, by232, sum10, sum32, sum20, sum42);
        }
    }

#pragma unroll
    for (int n = 0; n < num_rows; ++n) {
        const float total = warp_reduce_sum<warp_size>(sumf[n]);
        if (threadIdx.x == 0 && first_row + n < nrows_x) dst_c[first_row + n] = total;
    }
}

// Fused gate+up SwiGLU dequant matvec for Q4_K: computes up and gate matvecs
// from the shared activation in one pass, writes silu(gate)*up.
template <int warp_size, int num_rows>
static __global__ void mul_mat_vec_dq_glu_q4_K(
        const void * __restrict__ vx_up, const void * __restrict__ vx_gate,
        const float * __restrict__ y, float * __restrict__ dst,
        const int ncols_x, const int nrows_x) {
    const int first_row = num_rows * blockIdx.x;
    const int nblocks   = ncols_x / QK_K;
    const int it_size   = warp_size / 16;

    const dq_geom_q4_K g = dq_setup_q4_K(threadIdx.x);
    const block_q4_K * xu = (const block_q4_K *) vx_up;
    const block_q4_K * xg = (const block_q4_K *) vx_gate;

    if (num_rows >= 8) {
        // Two-pass: one accumulator array at a time keeps register pressure at
        // plain-kernel levels. up is reduced and stashed in dst, then pass 2
        // computes gate and combines. Weights are still read once each; only the
        // small activation vector y is re-read. Avoids the single-pass VGPR spill.
        for (int pass = 0; pass < 2; ++pass) {
            const block_q4_K * xw = pass == 0 ? xu : xg;
            float acc[num_rows];
#pragma unroll
            for (int n = 0; n < num_rows; ++n) acc[n] = 0.0f;

            for (int i = g.ix; i < nblocks; i += it_size) {
                const float * yb = y + (int64_t) i * QK_K;
                const float4 by10  = *(const float4 *) (yb + g.y_offset      );
                const float4 by132 = *(const float4 *) (yb + g.y_offset +  32);
                const float4 by20  = *(const float4 *) (yb + g.y_offset + 128);
                const float4 by232 = *(const float4 *) (yb + g.y_offset + 160);

                const float sum10 = by10.x  + by10.y  + by10.z  + by10.w;
                const float sum32 = by132.x + by132.y + by132.z + by132.w;
                const float sum20 = by20.x  + by20.y  + by20.z  + by20.w;
                const float sum42 = by232.x + by232.y + by232.z + by232.w;

#pragma unroll
                for (int n = 0; n < num_rows; ++n) {
                    const int row = min(first_row + n, nrows_x - 1);
                    acc[n] += dq_dot_q4_K(&xw[(int64_t) row * nblocks + i], g.q_offset, g.v_im, by10, by132, by20, by232, sum10, sum32, sum20, sum42);
                }
            }

#pragma unroll
            for (int n = 0; n < num_rows; ++n) {
                const float r = warp_reduce_sum<warp_size>(acc[n]);
                if (threadIdx.x == 0 && first_row + n < nrows_x) {
                    if (pass == 0) dst[first_row + n] = r;
                    else           dst[first_row + n] = ggml_cuda_op_silu_single(r) * dst[first_row + n];
                }
            }
        }
        return;
    }

    float up[num_rows], gate[num_rows];
#pragma unroll
    for (int n = 0; n < num_rows; ++n) { up[n] = 0.0f; gate[n] = 0.0f; }

    for (int i = g.ix; i < nblocks; i += it_size) {
        const float * yb = y + (int64_t) i * QK_K;
        const float4 by10  = *(const float4 *) (yb + g.y_offset      );
        const float4 by132 = *(const float4 *) (yb + g.y_offset +  32);
        const float4 by20  = *(const float4 *) (yb + g.y_offset + 128);
        const float4 by232 = *(const float4 *) (yb + g.y_offset + 160);

        const float sum10 = by10.x  + by10.y  + by10.z  + by10.w;
        const float sum32 = by132.x + by132.y + by132.z + by132.w;
        const float sum20 = by20.x  + by20.y  + by20.z  + by20.w;
        const float sum42 = by232.x + by232.y + by232.z + by232.w;

#pragma unroll
        for (int n = 0; n < num_rows; ++n) {
            const int row = min(first_row + n, nrows_x - 1);
            const int64_t off = (int64_t) row * nblocks + i;
            up[n]   += dq_dot_q4_K(&xu[off], g.q_offset, g.v_im, by10, by132, by20, by232, sum10, sum32, sum20, sum42);
            gate[n] += dq_dot_q4_K(&xg[off], g.q_offset, g.v_im, by10, by132, by20, by232, sum10, sum32, sum20, sum42);
        }
    }

#pragma unroll
    for (int n = 0; n < num_rows; ++n) {
        const float u = warp_reduce_sum<warp_size>(up[n]);
        const float gt = warp_reduce_sum<warp_size>(gate[n]);
        if (threadIdx.x == 0 && first_row + n < nrows_x) dst[first_row + n] = ggml_cuda_op_silu_single(gt) * u;
    }
}

// ---- Q5_K geometry ----
struct dq_geom_q5_K {
    int ix, l0, v_im, q_offset, y_offset;
};
static __device__ __forceinline__ dq_geom_q5_K dq_setup_q5_K(int tid) {
    const int itid = tid % 16;
    const int ix   = tid / 16;
    const int il   = itid / 4;
    const int ir   = itid % 4;
    const int v_im = il / 2;
    const int v_in = il % 2;
    const int l0   = 4 * ir + 2 * v_in;
    return { ix, l0, v_im, 32 * v_im + l0, 64 * v_im + l0 };
}

#define DQ_Q5_K_LOAD_ACT()                                                          \
    const float2 by10  = *(const float2 *) (yb + g.y_offset      );                 \
    const float2 by116 = *(const float2 *) (yb + g.y_offset +  16);                 \
    const float2 by132 = *(const float2 *) (yb + g.y_offset +  32);                 \
    const float2 by148 = *(const float2 *) (yb + g.y_offset +  48);                 \
    const float2 by20  = *(const float2 *) (yb + g.y_offset + 128);                 \
    const float2 by216 = *(const float2 *) (yb + g.y_offset + 144);                 \
    const float2 by232 = *(const float2 *) (yb + g.y_offset + 160);                 \
    const float2 by248 = *(const float2 *) (yb + g.y_offset + 176);                 \
    const float smin_x = by10.x  + by10.y  + by116.x + by116.y;                     \
    const float smin_y = by132.x + by132.y + by148.x + by148.y;                     \
    const float smin_z = by20.x  + by20.y  + by216.x + by216.y;                     \
    const float smin_w = by232.x + by232.y + by248.x + by248.y

#define DQ_Q5_K_ARGS by10, by116, by132, by148, by20, by216, by232, by248, smin_x, smin_y, smin_z, smin_w

template <int warp_size, int num_rows>
static __global__ void mul_mat_vec_dq_q5_K(
        const void * __restrict__ vx, const float * __restrict__ y, float * __restrict__ dst,
        const int ncols_x, const int nrows_x,
        const int32_t * __restrict__ ids, const int64_t stride_channel_x, const int64_t stride_channel_dst,
        const int64_t stride_channel_y, const int nchannels_y) {
    const int first_row = num_rows * blockIdx.x;
    const int nblocks   = ncols_x / QK_K;
    const int it_size   = warp_size / 16;

    const int channel   = blockIdx.y;
    const int expert    = ids ? ids[channel] : 0;
    const int channel_y = nchannels_y > 1 ? channel % nchannels_y : 0;
    const dq_geom_q5_K g = dq_setup_q5_K(threadIdx.x);
    const block_q5_K * x = (const block_q5_K *) vx + expert * stride_channel_x;
    const float * yc = y + (int64_t) channel_y * stride_channel_y;
    float * dst_c = dst + channel * stride_channel_dst;

    float sumf[num_rows];
#pragma unroll
    for (int n = 0; n < num_rows; ++n) sumf[n] = 0.0f;

    for (int i = g.ix; i < nblocks; i += it_size) {
        const float * yb = yc + (int64_t) i * QK_K;
        DQ_Q5_K_LOAD_ACT();

#pragma unroll
        for (int n = 0; n < num_rows; ++n) {
            const int row = min(first_row + n, nrows_x - 1);
            const block_q5_K * b = &x[(int64_t) row * nblocks + i];
            sumf[n] += dq_dot_q5_K(b, g.q_offset, g.l0, g.v_im, DQ_Q5_K_ARGS);
        }
    }

#pragma unroll
    for (int n = 0; n < num_rows; ++n) {
        const float total = warp_reduce_sum<warp_size>(sumf[n]);
        if (threadIdx.x == 0 && first_row + n < nrows_x) dst_c[first_row + n] = total;
    }
}

template <int warp_size, int num_rows>
static __global__ void mul_mat_vec_dq_glu_q5_K(
        const void * __restrict__ vx_up, const void * __restrict__ vx_gate,
        const float * __restrict__ y, float * __restrict__ dst,
        const int ncols_x, const int nrows_x) {
    const int first_row = num_rows * blockIdx.x;
    const int nblocks   = ncols_x / QK_K;
    const int it_size   = warp_size / 16;

    const dq_geom_q5_K g = dq_setup_q5_K(threadIdx.x);
    const block_q5_K * xu = (const block_q5_K *) vx_up;
    const block_q5_K * xg = (const block_q5_K *) vx_gate;

    if (num_rows >= 8) {
        // Two-pass: one accumulator array at a time keeps register pressure at
        // plain-kernel levels. up is reduced and stashed in dst, then pass 2
        // computes gate and combines. Weights are still read once each; only the
        // small activation vector y is re-read. Avoids the single-pass VGPR spill.
        for (int pass = 0; pass < 2; ++pass) {
            const block_q5_K * xw = pass == 0 ? xu : xg;
            float acc[num_rows];
#pragma unroll
            for (int n = 0; n < num_rows; ++n) acc[n] = 0.0f;

            for (int i = g.ix; i < nblocks; i += it_size) {
                const float * yb = y + (int64_t) i * QK_K;
                DQ_Q5_K_LOAD_ACT();

#pragma unroll
                for (int n = 0; n < num_rows; ++n) {
                    const int row = min(first_row + n, nrows_x - 1);
                    acc[n] += dq_dot_q5_K(&xw[(int64_t) row * nblocks + i], g.q_offset, g.l0, g.v_im, DQ_Q5_K_ARGS);
                }
            }

#pragma unroll
            for (int n = 0; n < num_rows; ++n) {
                const float r = warp_reduce_sum<warp_size>(acc[n]);
                if (threadIdx.x == 0 && first_row + n < nrows_x) {
                    if (pass == 0) dst[first_row + n] = r;
                    else           dst[first_row + n] = ggml_cuda_op_silu_single(r) * dst[first_row + n];
                }
            }
        }
        return;
    }

    float up[num_rows], gate[num_rows];
#pragma unroll
    for (int n = 0; n < num_rows; ++n) { up[n] = 0.0f; gate[n] = 0.0f; }

    for (int i = g.ix; i < nblocks; i += it_size) {
        const float * yb = y + (int64_t) i * QK_K;
        DQ_Q5_K_LOAD_ACT();

#pragma unroll
        for (int n = 0; n < num_rows; ++n) {
            const int row = min(first_row + n, nrows_x - 1);
            const int64_t off = (int64_t) row * nblocks + i;
            up[n]   += dq_dot_q5_K(&xu[off], g.q_offset, g.l0, g.v_im, DQ_Q5_K_ARGS);
            gate[n] += dq_dot_q5_K(&xg[off], g.q_offset, g.l0, g.v_im, DQ_Q5_K_ARGS);
        }
    }

#pragma unroll
    for (int n = 0; n < num_rows; ++n) {
        const float u = warp_reduce_sum<warp_size>(up[n]);
        const float gt = warp_reduce_sum<warp_size>(gate[n]);
        if (threadIdx.x == 0 && first_row + n < nrows_x) dst[first_row + n] = ggml_cuda_op_silu_single(gt) * u;
    }
}

// ---- Q6_K geometry ----
struct dq_geom_q6_K {
    int ix, ql_offset, qh_offset, s_offset, y_offset;
};
static __device__ __forceinline__ dq_geom_q6_K dq_setup_q6_K(int tid) {
    const int itid = tid % 16;
    const int ix   = tid / 16;
    const int v_im = itid / 8;
    const int v_in = itid % 8;
    const int l0   = 4 * v_in;
    const int is   = v_in / 4;
    return { ix, 64 * v_im + l0, 32 * v_im + l0, 8 * v_im + is, 128 * v_im + l0 };
}

template <int warp_size, int num_rows>
static __global__ void mul_mat_vec_dq_q6_K(
        const void * __restrict__ vx, const float * __restrict__ y, float * __restrict__ dst,
        const int ncols_x, const int nrows_x,
        const int32_t * __restrict__ ids, const int64_t stride_channel_x, const int64_t stride_channel_dst,
        const int64_t stride_channel_y, const int nchannels_y) {
    const int first_row = num_rows * blockIdx.x;
    const int nblocks   = ncols_x / QK_K;
    const int it_size   = warp_size / 16;

    const int channel   = blockIdx.y;
    const int expert    = ids ? ids[channel] : 0;
    const int channel_y = nchannels_y > 1 ? channel % nchannels_y : 0;
    const dq_geom_q6_K g = dq_setup_q6_K(threadIdx.x);
    const block_q6_K * x = (const block_q6_K *) vx + expert * stride_channel_x;
    const float * yc = y + (int64_t) channel_y * stride_channel_y;
    float * dst_c = dst + channel * stride_channel_dst;

    float sumf[num_rows];
#pragma unroll
    for (int n = 0; n < num_rows; ++n) sumf[n] = 0.0f;

    for (int i = g.ix; i < nblocks; i += it_size) {
        const float * yb = yc + (int64_t) i * QK_K;
        const float4 by0  = *(const float4 *) (yb + g.y_offset      );
        const float4 by32 = *(const float4 *) (yb + g.y_offset +  32);
        const float4 by64 = *(const float4 *) (yb + g.y_offset +  64);
        const float4 by96 = *(const float4 *) (yb + g.y_offset +  96);

#pragma unroll
        for (int n = 0; n < num_rows; ++n) {
            const int row = min(first_row + n, nrows_x - 1);
            const block_q6_K * b = &x[(int64_t) row * nblocks + i];
            sumf[n] += dq_dot_q6_K(b, g.ql_offset, g.qh_offset, g.s_offset, by0, by32, by64, by96);
        }
    }

#pragma unroll
    for (int n = 0; n < num_rows; ++n) {
        const float total = warp_reduce_sum<warp_size>(sumf[n]);
        if (threadIdx.x == 0 && first_row + n < nrows_x) dst_c[first_row + n] = total;
    }
}

template <int warp_size, int num_rows>
static __global__ void mul_mat_vec_dq_glu_q6_K(
        const void * __restrict__ vx_up, const void * __restrict__ vx_gate,
        const float * __restrict__ y, float * __restrict__ dst,
        const int ncols_x, const int nrows_x) {
    const int first_row = num_rows * blockIdx.x;
    const int nblocks   = ncols_x / QK_K;
    const int it_size   = warp_size / 16;

    const dq_geom_q6_K g = dq_setup_q6_K(threadIdx.x);
    const block_q6_K * xu = (const block_q6_K *) vx_up;
    const block_q6_K * xg = (const block_q6_K *) vx_gate;

    if (num_rows >= 8) {
        // Two-pass: one accumulator array at a time keeps register pressure at
        // plain-kernel levels. up is reduced and stashed in dst, then pass 2
        // computes gate and combines. Weights are still read once each; only the
        // small activation vector y is re-read. Avoids the single-pass VGPR spill.
        for (int pass = 0; pass < 2; ++pass) {
            const block_q6_K * xw = pass == 0 ? xu : xg;
            float acc[num_rows];
#pragma unroll
            for (int n = 0; n < num_rows; ++n) acc[n] = 0.0f;

            for (int i = g.ix; i < nblocks; i += it_size) {
                const float * yb = y + (int64_t) i * QK_K;
                const float4 by0  = *(const float4 *) (yb + g.y_offset      );
                const float4 by32 = *(const float4 *) (yb + g.y_offset +  32);
                const float4 by64 = *(const float4 *) (yb + g.y_offset +  64);
                const float4 by96 = *(const float4 *) (yb + g.y_offset +  96);

#pragma unroll
                for (int n = 0; n < num_rows; ++n) {
                    const int row = min(first_row + n, nrows_x - 1);
                    acc[n] += dq_dot_q6_K(&xw[(int64_t) row * nblocks + i], g.ql_offset, g.qh_offset, g.s_offset, by0, by32, by64, by96);
                }
            }

#pragma unroll
            for (int n = 0; n < num_rows; ++n) {
                const float r = warp_reduce_sum<warp_size>(acc[n]);
                if (threadIdx.x == 0 && first_row + n < nrows_x) {
                    if (pass == 0) dst[first_row + n] = r;
                    else           dst[first_row + n] = ggml_cuda_op_silu_single(r) * dst[first_row + n];
                }
            }
        }
        return;
    }

    float up[num_rows], gate[num_rows];
#pragma unroll
    for (int n = 0; n < num_rows; ++n) { up[n] = 0.0f; gate[n] = 0.0f; }

    for (int i = g.ix; i < nblocks; i += it_size) {
        const float * yb = y + (int64_t) i * QK_K;
        const float4 by0  = *(const float4 *) (yb + g.y_offset      );
        const float4 by32 = *(const float4 *) (yb + g.y_offset +  32);
        const float4 by64 = *(const float4 *) (yb + g.y_offset +  64);
        const float4 by96 = *(const float4 *) (yb + g.y_offset +  96);

#pragma unroll
        for (int n = 0; n < num_rows; ++n) {
            const int row = min(first_row + n, nrows_x - 1);
            const int64_t off = (int64_t) row * nblocks + i;
            up[n]   += dq_dot_q6_K(&xu[off], g.ql_offset, g.qh_offset, g.s_offset, by0, by32, by64, by96);
            gate[n] += dq_dot_q6_K(&xg[off], g.ql_offset, g.qh_offset, g.s_offset, by0, by32, by64, by96);
        }
    }

#pragma unroll
    for (int n = 0; n < num_rows; ++n) {
        const float u = warp_reduce_sum<warp_size>(up[n]);
        const float gt = warp_reduce_sum<warp_size>(gate[n]);
        if (threadIdx.x == 0 && first_row + n < nrows_x) dst[first_row + n] = ggml_cuda_op_silu_single(gt) * u;
    }
}

// ---- launchers ----
#define DQ_LAUNCH_PLAIN(KERN, NR)                                                                 \
    do {                                                                                          \
        const dim3 bn((nrows_x + (NR) - 1) / (NR), nchannels_dst, 1);                             \
        const dim3 bd(warp_size, 1, 1);                                                           \
        if (warp_size == 64) KERN<64, NR><<<bn, bd, 0, stream>>>(vx, y, d, ncols_x, nrows_x, ids, stride_channel_x, stride_channel_dst, stride_channel_y, nchannels_y); \
        else                 KERN<32, NR><<<bn, bd, 0, stream>>>(vx, y, d, ncols_x, nrows_x, ids, stride_channel_x, stride_channel_dst, stride_channel_y, nchannels_y); \
    } while (0)

#define DQ_LAUNCH_GLU(KERN, NR)                                                                         \
    do {                                                                                                \
        const dim3 bn((nrows_x + (NR) - 1) / (NR), 1, 1);                                               \
        const dim3 bd(warp_size, 1, 1);                                                                 \
        if (warp_size == 64) KERN<64, NR><<<bn, bd, 0, stream>>>(vx_up, vx_gate, y, d, ncols_x, nrows_x); \
        else                 KERN<32, NR><<<bn, bd, 0, stream>>>(vx_up, vx_gate, y, d, ncols_x, nrows_x); \
    } while (0)

template <int num_rows>
static void launch_dq_q4_K(const void * vx, const float * y, float * d, int ncols_x, int nrows_x,
        const int32_t * ids, int64_t stride_channel_x, int64_t stride_channel_dst,
        int64_t stride_channel_y, int nchannels_y, int nchannels_dst, int warp_size, cudaStream_t stream) {
    DQ_LAUNCH_PLAIN(mul_mat_vec_dq_q4_K, num_rows);
}
template <int num_rows>
static void launch_dq_q5_K(const void * vx, const float * y, float * d, int ncols_x, int nrows_x,
        const int32_t * ids, int64_t stride_channel_x, int64_t stride_channel_dst,
        int64_t stride_channel_y, int nchannels_y, int nchannels_dst, int warp_size, cudaStream_t stream) {
    DQ_LAUNCH_PLAIN(mul_mat_vec_dq_q5_K, num_rows);
}
template <int num_rows>
static void launch_dq_q6_K(const void * vx, const float * y, float * d, int ncols_x, int nrows_x,
        const int32_t * ids, int64_t stride_channel_x, int64_t stride_channel_dst,
        int64_t stride_channel_y, int nchannels_y, int nchannels_dst, int warp_size, cudaStream_t stream) {
    DQ_LAUNCH_PLAIN(mul_mat_vec_dq_q6_K, num_rows);
}
template <int num_rows>
static void launch_dq_glu_q4_K(const void * vx_up, const void * vx_gate, const float * y, float * d, int ncols_x, int nrows_x, int warp_size, cudaStream_t stream) {
    DQ_LAUNCH_GLU(mul_mat_vec_dq_glu_q4_K, num_rows);
}
template <int num_rows>
static void launch_dq_glu_q5_K(const void * vx_up, const void * vx_gate, const float * y, float * d, int ncols_x, int nrows_x, int warp_size, cudaStream_t stream) {
    DQ_LAUNCH_GLU(mul_mat_vec_dq_glu_q5_K, num_rows);
}
template <int num_rows>
static void launch_dq_glu_q6_K(const void * vx_up, const void * vx_gate, const float * y, float * d, int ncols_x, int nrows_x, int warp_size, cudaStream_t stream) {
    DQ_LAUNCH_GLU(mul_mat_vec_dq_glu_q6_K, num_rows);
}

#define DQ_DISPATCH_ROWS(LAUNCH, ...)                          \
    switch (dq_num_rows()) {                                   \
        case 2:  LAUNCH<2>(__VA_ARGS__); break;               \
        case 4:  LAUNCH<4>(__VA_ARGS__); break;               \
        case 8:  LAUNCH<8>(__VA_ARGS__); break;               \
        default: LAUNCH<1>(__VA_ARGS__); break;               \
    }

// MoE (ids) strides: src0 is [K, N, n_expert]; the expert-axis stride (in block
// elements), the per-expert-slot dst column stride (in floats), and the activation
// column stride+count mirror the q8_1 MUL_MAT_ID layout in ggml_cuda_mul_mat_vec_q.
// nchannels_y = src1->ne[1]: ==1 when the activation is broadcast/shared (gate/up,
// dense), ==nchannels_dst for per-expert activation (ffn_down). Non-ids callers pass
// ids=nullptr, nchannels_dst=1, nchannels_y=1 => every extra axis collapses.
struct dq_moe_dims {
    const int32_t * ids;
    int64_t stride_channel_x;
    int64_t stride_channel_dst;
    int64_t stride_channel_y;
    int     nchannels_y;
    int     nchannels_dst;
};
static dq_moe_dims dq_moe_setup(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, const ggml_tensor * dst) {
    if (!ids) {
        return { nullptr, 0, 0, 0, 1, 1 };
    }
    return {
        (const int32_t *) ids->data,
        (int64_t) (src0->nb[2] / ggml_type_size(src0->type)),
        (int64_t) (dst->nb[1]  / ggml_type_size(dst->type)),
        (int64_t) (src1->nb[1] / ggml_type_size(src1->type)),
        (int) src1->ne[1],
        (int) dst->ne[1],
    };
}

static void ggml_cuda_mul_mat_vec_dq_q4_K(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
        const ggml_tensor * ids_t, ggml_tensor * dst) {
    const int ncols_x = src0->ne[0];
    const int nrows   = src0->ne[1];
    cudaStream_t stream = ctx.stream();
    const int warp_size = ggml_cuda_info().devices[ctx.device].warp_size;
    const void  * vx = src0->data;
    const float * y  = (const float *) src1->data;
    float       * d  = (float *) dst->data;
    const dq_moe_dims m = dq_moe_setup(src0, src1, ids_t, dst);
    const int32_t * ids = m.ids;
    const int64_t stride_channel_x   = m.stride_channel_x;
    const int64_t stride_channel_dst = m.stride_channel_dst;
    const int64_t stride_channel_y   = m.stride_channel_y;
    const int     nchannels_y        = m.nchannels_y;
    const int     nchannels_dst      = m.nchannels_dst;
    DQ_DISPATCH_ROWS(launch_dq_q4_K, vx, y, d, ncols_x, nrows, ids, stride_channel_x, stride_channel_dst, stride_channel_y, nchannels_y, nchannels_dst, warp_size, stream);
}

static void ggml_cuda_mul_mat_vec_dq_q5_K(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
        const ggml_tensor * ids_t, ggml_tensor * dst) {
    const int ncols_x = src0->ne[0];
    const int nrows   = src0->ne[1];
    cudaStream_t stream = ctx.stream();
    const int warp_size = ggml_cuda_info().devices[ctx.device].warp_size;
    const void  * vx = src0->data;
    const float * y  = (const float *) src1->data;
    float       * d  = (float *) dst->data;
    const dq_moe_dims m = dq_moe_setup(src0, src1, ids_t, dst);
    const int32_t * ids = m.ids;
    const int64_t stride_channel_x   = m.stride_channel_x;
    const int64_t stride_channel_dst = m.stride_channel_dst;
    const int64_t stride_channel_y   = m.stride_channel_y;
    const int     nchannels_y        = m.nchannels_y;
    const int     nchannels_dst      = m.nchannels_dst;
    DQ_DISPATCH_ROWS(launch_dq_q5_K, vx, y, d, ncols_x, nrows, ids, stride_channel_x, stride_channel_dst, stride_channel_y, nchannels_y, nchannels_dst, warp_size, stream);
}

static void ggml_cuda_mul_mat_vec_dq_q6_K(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
        const ggml_tensor * ids_t, ggml_tensor * dst) {
    const int ncols_x = src0->ne[0];
    const int nrows   = src0->ne[1];
    cudaStream_t stream = ctx.stream();
    const int warp_size = ggml_cuda_info().devices[ctx.device].warp_size;
    const void  * vx = src0->data;
    const float * y  = (const float *) src1->data;
    float       * d  = (float *) dst->data;
    const dq_moe_dims m = dq_moe_setup(src0, src1, ids_t, dst);
    const int32_t * ids = m.ids;
    const int64_t stride_channel_x   = m.stride_channel_x;
    const int64_t stride_channel_dst = m.stride_channel_dst;
    const int64_t stride_channel_y   = m.stride_channel_y;
    const int     nchannels_y        = m.nchannels_y;
    const int     nchannels_dst      = m.nchannels_dst;
    DQ_DISPATCH_ROWS(launch_dq_q6_K, vx, y, d, ncols_x, nrows, ids, stride_channel_x, stride_channel_dst, stride_channel_y, nchannels_y, nchannels_dst, warp_size, stream);
}

static void ggml_cuda_mul_mat_vec_dq_glu(
        ggml_backend_cuda_context & ctx, const ggml_tensor * up, const ggml_tensor * gate,
        const ggml_tensor * src1, ggml_tensor * dst) {
    const int ncols_x = up->ne[0];
    const int nrows   = up->ne[1];
    cudaStream_t stream = ctx.stream();
    const int warp_size = ggml_cuda_info().devices[ctx.device].warp_size;
    const void  * vx_up   = up->data;
    const void  * vx_gate = gate->data;
    const float * y = (const float *) src1->data;
    float       * d = (float *) dst->data;
    switch (up->type) {
        case GGML_TYPE_Q4_K: DQ_DISPATCH_ROWS(launch_dq_glu_q4_K, vx_up, vx_gate, y, d, ncols_x, nrows, warp_size, stream); break;
        case GGML_TYPE_Q5_K: DQ_DISPATCH_ROWS(launch_dq_glu_q5_K, vx_up, vx_gate, y, d, ncols_x, nrows, warp_size, stream); break;
        case GGML_TYPE_Q6_K: DQ_DISPATCH_ROWS(launch_dq_glu_q6_K, vx_up, vx_gate, y, d, ncols_x, nrows, warp_size, stream); break;
        default: GGML_ABORT("mul_mat_vec_dq_glu: unsupported type %s (should_use_mmv_dq must gate this)",
                            ggml_type_name(up->type));
    }
}

// ---- dispatch: dq as an internal variant of ggml_cuda_mul_mat_vec_q ----

bool ggml_cuda_should_use_mmv_dq(
        const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
        const ggml_tensor * dst, int cc, const ggml_cuda_mm_fusion_args_host * fusion) {
    const bool dq_default = GGML_CUDA_CC_IS_RDNA3_5(cc);
    if (!ggml_cuda_dq_mmv_enabled(dq_default)) {
        return false;
    }

    // MoE (ids): the plain (non-fused) per-expert matvec for a single decode token.
    // src0 is [K, N, n_expert]; the expert axis is selected via ids and becomes the
    // kernel's grid.y. The activation may be broadcast (gate/up: src1->ne[1]==1) or
    // per-expert (ffn_down: src1->ne[1]==n_expert_used==dst->ne[1]) — both handled by
    // channel_y in the kernel; the guard below is enforced further down. Fused GLU + ids
    // and multi-token (dst->ne[2]>1) remain deferred.
    if (ids != nullptr) {
        if (ids->type != GGML_TYPE_I32) {
            return false;
        }
        if (fusion != nullptr) {   // fused GLU + ids not implemented yet
            return false;
        }
        if (dst->ne[2] != 1) {     // single decode token: ncols_dst == 1
            return false;
        }
    }

    const ggml_type type = src0->type;
    if (!(type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K
            || (type == GGML_TYPE_Q6_K && ggml_cuda_dq_q6k_enabled(dq_default)))) {
        return false;
    }

    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }

    // Activation columns (src1->ne[1]). Non-ids: single-vector only (>1 is the Phase-2
    // MTP token-batch seam). ids: either broadcast (==1, gate/up) or per-expert
    // (==dst->ne[1]==n_expert_used, ffn_down); the kernel's channel_y handles both.
    if (ids == nullptr) {
        if (src1->ne[1] != 1) {
            return false;
        }
    } else {
        if (src1->ne[1] != 1 && src1->ne[1] != dst->ne[1]) {
            return false;
        }
    }

    if (src0->ne[0] % QK_K != 0) {
        return false;
    }
    if (src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) {
        return false;
    }
    // Non-ids requires a single 2D weight matrix; for ids, src0->ne[2] is the expert axis.
    if (ids == nullptr && src0->ne[2] != 1) {
        return false;
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        return false;
    }

    // Fusion inspection: dq only implements SwiGLU gate fusion (up = src0,
    // gate = fusion->gate, shared activation). Any bias/scale or other GLU op
    // falls back to the q8_1 mmvq path.
    if (fusion) {
        if (fusion->glu_op != GGML_GLU_OP_SWIGLU || fusion->gate == nullptr) {
            return false;
        }
        if (fusion->x_bias || fusion->gate_bias || fusion->x_scale || fusion->gate_scale) {
            return false;
        }
        if (fusion->gate->type != type || !ggml_are_same_shape(src0, fusion->gate)
                || !ggml_is_contiguous(fusion->gate)) {
            return false;
        }
    }

    return true;
}

void ggml_cuda_mul_mat_vec_dq(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
        const ggml_tensor * ids, ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion) {
    if (fusion && fusion->gate) {
        // Fused GLU + ids is not implemented; should_use_mmv_dq rejects fusion when ids is set.
        ggml_cuda_mul_mat_vec_dq_glu(ctx, src0, fusion->gate, src1, dst);
        return;
    }

    switch (src0->type) {
        case GGML_TYPE_Q4_K: ggml_cuda_mul_mat_vec_dq_q4_K(ctx, src0, src1, ids, dst); break;
        case GGML_TYPE_Q5_K: ggml_cuda_mul_mat_vec_dq_q5_K(ctx, src0, src1, ids, dst); break;
        case GGML_TYPE_Q6_K: ggml_cuda_mul_mat_vec_dq_q6_K(ctx, src0, src1, ids, dst); break;
        default: GGML_ABORT("mul_mat_vec_dq: unsupported type %s (should_use_mmv_dq must gate this)",
                            ggml_type_name(src0->type));
    }
}
