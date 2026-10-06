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

// ---- dq device-class table ----
//
// Self-contained mirror of mmvq.cu's get_device_table_id() classifier, kept dq-local so
// mmvq.cu (upstream's file) is untouched and this lives entirely in the file we own.
// dq only needs the subset that maps to a distinct kernel config; gfx1151 (RDNA3.5) is the
// current target and gets its own class. Adding a device class later = a new enum entry here
// plus a branch in dq_calc_warps_per_block; an unlisted class falls through to GENERIC.
enum dq_parameter_table_id {
    DQ_PARAMETERS_GENERIC = 0,
    DQ_PARAMETERS_RDNA3_5,
    DQ_PARAMETERS_RDNA4,
};

static constexpr __device__ dq_parameter_table_id dq_get_device_table_id() {
#if defined(RDNA4)
    return DQ_PARAMETERS_RDNA4;
#elif defined(RDNA3_5)
    return DQ_PARAMETERS_RDNA3_5;
#else
    return DQ_PARAMETERS_GENERIC;
#endif
}

static __host__ dq_parameter_table_id dq_get_device_table_id(int cc) {
    if (GGML_CUDA_CC_IS_RDNA4(cc)) {
        return DQ_PARAMETERS_RDNA4;
    }
    if (GGML_CUDA_CC_IS_RDNA3_5(cc)) {
        return DQ_PARAMETERS_RDNA3_5;
    }
    return DQ_PARAMETERS_GENERIC;
}

// Warps per block for the dq matvec. One warp per super-block is the proven gfx1151 config; every class returns
// 1 for now.
static constexpr __host__ __device__ int dq_calc_warps_per_block(dq_parameter_table_id table_id) {
    return 1;
}

// ---- generic dq matvec  ----
// Per-type numerics live in a Traits struct (block_t/geom/act + setup/load/dot); adding
// a type is a new Traits, not a new kernel. dq_block_reduce is the single reduction
// seam (warp-only today; a multi-warp variant would slot in here).
struct dq_fusion_args {
    const void *  gate      = nullptr;
    const float * x_bias    = nullptr;   // up/plain-proj bias (len nrows_x; per-expert stride for ids)
    const float * gate_bias = nullptr;   // gate-proj bias (only valid alongside gate)
    ggml_glu_op   glu_op    = GGML_GLU_OP_SWIGLU;
    float         glu_limit = 0.0f;
};

template <int warp_size, int warps_per_block>
static __device__ __forceinline__ float dq_block_reduce(float x) {
    static_assert(warps_per_block == 1, "dq multi-warp reduction not implemented (seam only)");
    return warp_reduce_sum<warp_size>(x);
}

// Per-type numerics are specializations of dq_traits<type>, keyed on the same ggml_type
// template arg mmvq's mul_mat_vec_q uses. The primary template is intentionally undefined;
// each supported type defines a specialization below (block_t/geom/act + setup/load/dot).
template <ggml_type type> struct dq_traits;

// ncols_dst is the token-batch width (mmvq's kernel axis); dq pins it to 1 for now(single
// decode token) and does not read it in the body — it labels the axis and keeps the
// signature aligned with mmvq's mul_mat_vec_q<type, ncols_dst, ...>.
template <ggml_type type, int warp_size, int ncols_dst, bool has_gate, bool has_prologue = false>
__launch_bounds__(warp_size * dq_calc_warps_per_block(dq_get_device_table_id()), 1)
static __global__ void mul_mat_vec_dq(
        const void * GGML_CUDA_RESTRICT vx, const float * GGML_CUDA_RESTRICT y, float * GGML_CUDA_RESTRICT dst,
        const int ncols_x, const int nrows_x,
        const int32_t * GGML_CUDA_RESTRICT ids, const int64_t stride_channel_x, const int64_t stride_channel_dst,
        const int64_t stride_channel_y, const int nchannels_y, [[maybe_unused]] const dq_fusion_args fusion) {
    using Traits  = dq_traits<type>;
    using block_t = typename Traits::block_t;
    static_assert(!has_prologue, "has_prologue is a Phase-B seam; not implemented in Phase A");
    static_assert(ncols_dst == 1, "dq is single decode token; ncols_dst>1 is a deferred seam");
    constexpr int warps_per_block = dq_calc_warps_per_block(dq_get_device_table_id());

    const int row     = blockIdx.x;   // one block per weight row
    const int nblocks = ncols_x / QK_K;
    const int it_size = warp_size / 16;

    // MoE (ids): blockIdx.y selects the expert slot; ids maps slot -> expert matrix.
    // channel_y picks this slot's activation column (per-expert ffn_down) or 0 when the
    // activation is broadcast/shared (gate/up, dense). Non-ids launches use grid.y=1,
    // ids=nullptr, nchannels_y=1 => every offset below collapses to 0.
    const int channel   = blockIdx.y;
    const int expert    = ids ? ids[channel] : 0;
    const int channel_y = nchannels_y > 1 ? channel % nchannels_y : 0;
    const typename Traits::geom g = Traits::setup(threadIdx.x);
    const block_t * xu = (const block_t *) vx + expert * stride_channel_x;
    const float * yc = y + (int64_t) channel_y * stride_channel_y;
    float * dst_c = dst + channel * stride_channel_dst;

    if constexpr (has_gate) {
        const block_t * xg = (const block_t *) fusion.gate + expert * stride_channel_x;

        float up = 0.0f, gate = 0.0f;
        for (int i = g.ix; i < nblocks; i += it_size) {
            const typename Traits::act a = Traits::load(yc + (int64_t) i * QK_K, g);
            const int64_t off = (int64_t) row * nblocks + i;
            up   += Traits::dot(&xu[off], g, a);
            gate += Traits::dot(&xg[off], g, a);
        }

        const float u  = dq_block_reduce<warp_size, warps_per_block>(up);
        const float gt = dq_block_reduce<warp_size, warps_per_block>(gate);
        if (threadIdx.x == 0) {
            // Bias + GLU-variant epilogue, mirroring mmvq's fused path (mmvq.cu:806-834) for
            // numeric parity. bias_off indexes per-output-row; for ids (MoE) each expert owns a
            // contiguous nrows_x-stride slice, so expert*nrows_x + row (expert==0 for dense).
            const int64_t bias_off = (int64_t) expert * nrows_x + row;
            const float uu = u  + (fusion.x_bias    ? fusion.x_bias[bias_off]    : 0.0f);
            const float gg = gt + (fusion.gate_bias ? fusion.gate_bias[bias_off] : 0.0f);
            float r;
            switch (fusion.glu_op) {
                case GGML_GLU_OP_GEGLU:        r = ggml_cuda_op_gelu_single(gg) * uu;                       break;
                case GGML_GLU_OP_SWIGLU_OAI:   r = ggml_cuda_op_swiglu_oai_single(gg, uu);                  break;
                case GGML_GLU_OP_SWIGLU_CLAMP: r = ggml_cuda_op_swiglu_clamp_single(gg, uu, fusion.glu_limit); break;
                case GGML_GLU_OP_SWIGLU:       r = ggml_cuda_op_silu_single(gg) * uu;                       break;
                default:                       r = uu * gg;                                                 break;
            }
            dst_c[row] = r;
        }
    } else {
        float sumf = 0.0f;
        for (int i = g.ix; i < nblocks; i += it_size) {
            const typename Traits::act a = Traits::load(yc + (int64_t) i * QK_K, g);
            sumf += Traits::dot(&xu[(int64_t) row * nblocks + i], g, a);
        }

        const float total = dq_block_reduce<warp_size, warps_per_block>(sumf);
        if (threadIdx.x == 0) {
            const int64_t bias_off = (int64_t) expert * nrows_x + row;
            dst_c[row] = total + (fusion.x_bias ? fusion.x_bias[bias_off] : 0.0f);
        }
    }
}

// Innermost host launcher (mmvq's launch tier): picks the 32- vs 64-wide
// instantiation, sets grid (one block per weight row x nchannels_dst) and block,
// and forwards the runtime fusion args. has_gate is compile-time (plain vs fused);
// numeric-only glu choices ride in fusion.
template <ggml_type type, int ncols_dst, bool has_gate>
static void launch_dq(const void * vx, const float * y, float * d, int ncols_x, int nrows_x,
        const int32_t * ids, int64_t stride_channel_x, int64_t stride_channel_dst,
        int64_t stride_channel_y, int nchannels_y, int nchannels_dst, int warp_size, int cc,
        const dq_fusion_args & fusion, cudaStream_t stream) {
    const int warps_per_block = dq_calc_warps_per_block(dq_get_device_table_id(cc));
    const dim3 bn(nrows_x, nchannels_dst, 1);
    const dim3 bd(warp_size * warps_per_block, 1, 1);
    const ggml_cuda_kernel_launch_params params(bn, bd, 0, stream);
    if (warp_size == 64) {
        ggml_cuda_kernel_launch(mul_mat_vec_dq<type, 64, ncols_dst, has_gate>, params,
            vx, y, d, ncols_x, nrows_x, ids, stride_channel_x, stride_channel_dst, stride_channel_y, nchannels_y, fusion);
    } else {
        ggml_cuda_kernel_launch(mul_mat_vec_dq<type, 32, ncols_dst, has_gate>, params,
            vx, y, d, ncols_x, nrows_x, ids, stride_channel_x, stride_channel_dst, stride_channel_y, nchannels_y, fusion);
    }
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

// Q4_K numerics for the generic kernel: 16 threads process one super-block, the
// activation is loaded once as four float4s (+ their lane sums) and reused across rows.
// No q8_1 activation pass (unlike mul_mat_vec_q).
template <> struct dq_traits<GGML_TYPE_Q4_K> {
    using block_t = block_q4_K;
    using geom    = dq_geom_q4_K;
    struct act { float4 by10, by132, by20, by232; float sum10, sum32, sum20, sum42; };

    static __device__ __forceinline__ geom setup(int tid) { return dq_setup_q4_K(tid); }

    static __device__ __forceinline__ act load(const float * yb, const geom & g) {
        act a;
        a.by10  = *(const float4 *) (yb + g.y_offset      );
        a.by132 = *(const float4 *) (yb + g.y_offset +  32);
        a.by20  = *(const float4 *) (yb + g.y_offset + 128);
        a.by232 = *(const float4 *) (yb + g.y_offset + 160);
        a.sum10 = a.by10.x  + a.by10.y  + a.by10.z  + a.by10.w;
        a.sum32 = a.by132.x + a.by132.y + a.by132.z + a.by132.w;
        a.sum20 = a.by20.x  + a.by20.y  + a.by20.z  + a.by20.w;
        a.sum42 = a.by232.x + a.by232.y + a.by232.z + a.by232.w;
        return a;
    }

    static __device__ __forceinline__ float dot(const block_t * b, const geom & g, const act & a) {
        return dq_dot_q4_K(b, g.q_offset, g.v_im, a.by10, a.by132, a.by20, a.by232, a.sum10, a.sum32, a.sum20, a.sum42);
    }
};

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

// Q5_K numerics for the generic kernel: activation loaded once as eight float2s plus
// their four smin lane-sums, reused across rows. Mirrors dq_traits<GGML_TYPE_Q4_K>.
template <> struct dq_traits<GGML_TYPE_Q5_K> {
    using block_t = block_q5_K;
    using geom    = dq_geom_q5_K;
    struct act {
        float2 by10, by116, by132, by148, by20, by216, by232, by248;
        float  smin_x, smin_y, smin_z, smin_w;
    };

    static __device__ __forceinline__ geom setup(int tid) { return dq_setup_q5_K(tid); }

    static __device__ __forceinline__ act load(const float * yb, const geom & g) {
        act a;
        a.by10  = *(const float2 *) (yb + g.y_offset      );
        a.by116 = *(const float2 *) (yb + g.y_offset +  16);
        a.by132 = *(const float2 *) (yb + g.y_offset +  32);
        a.by148 = *(const float2 *) (yb + g.y_offset +  48);
        a.by20  = *(const float2 *) (yb + g.y_offset + 128);
        a.by216 = *(const float2 *) (yb + g.y_offset + 144);
        a.by232 = *(const float2 *) (yb + g.y_offset + 160);
        a.by248 = *(const float2 *) (yb + g.y_offset + 176);
        a.smin_x = a.by10.x  + a.by10.y  + a.by116.x + a.by116.y;
        a.smin_y = a.by132.x + a.by132.y + a.by148.x + a.by148.y;
        a.smin_z = a.by20.x  + a.by20.y  + a.by216.x + a.by216.y;
        a.smin_w = a.by232.x + a.by232.y + a.by248.x + a.by248.y;
        return a;
    }

    static __device__ __forceinline__ float dot(const block_t * b, const geom & g, const act & a) {
        return dq_dot_q5_K(b, g.q_offset, g.l0, g.v_im,
            a.by10, a.by116, a.by132, a.by148, a.by20, a.by216, a.by232, a.by248,
            a.smin_x, a.smin_y, a.smin_z, a.smin_w);
    }
};

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

template <> struct dq_traits<GGML_TYPE_Q6_K> {
    using block_t = block_q6_K;
    using geom    = dq_geom_q6_K;
    struct act { float4 by0, by32, by64, by96; };
    static __device__ __forceinline__ geom setup(int tid) { return dq_setup_q6_K(tid); }
    static __device__ __forceinline__ act load(const float * yb, const geom & g) {
        act a;
        a.by0  = *(const float4 *) (yb + g.y_offset      );
        a.by32 = *(const float4 *) (yb + g.y_offset +  32);
        a.by64 = *(const float4 *) (yb + g.y_offset +  64);
        a.by96 = *(const float4 *) (yb + g.y_offset +  96);
        return a;
    }
    static __device__ __forceinline__ float dot(const block_t * b, const geom & g, const act & a) {
        return dq_dot_q6_K(b, g.ql_offset, g.qh_offset, g.s_offset, a.by0, a.by32, a.by64, a.by96);
    }
};

// ---- launchers ----

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

// Args threaded down the tower unchanged; grouped so each dispatch layer forwards
// one struct instead of a dozen positional params.
struct dq_launch_args {
    const void *    vx;
    const float *   y;
    float *         d;
    int             ncols_x;
    int             nrows_x;
    const int32_t * ids;
    int64_t         stride_channel_x;
    int64_t         stride_channel_dst;
    int64_t         stride_channel_y;
    int             nchannels_y;
    int             nchannels_dst;
    int             warp_size;
    int             cc;
    dq_fusion_args  fusion;
    cudaStream_t    stream;
};

// Tier 3 (mmvq's switch_fusion): runtime gate pointer -> compile-time has_gate. vx is
// the "up" weight; fusion.gate is the gate weight (null for plain). The kernel branches
// on has_gate internally, so plain and fused SwiGLU share one body.
template <ggml_type type, int ncols_dst>
static void dq_switch_fusion(const dq_launch_args & a) {
    if (a.fusion.gate) {
        launch_dq<type, ncols_dst, true>(a.vx, a.y, a.d, a.ncols_x, a.nrows_x, a.ids,
            a.stride_channel_x, a.stride_channel_dst, a.stride_channel_y, a.nchannels_y,
            a.nchannels_dst, a.warp_size, a.cc, a.fusion, a.stream);
    } else {
        launch_dq<type, ncols_dst, false>(a.vx, a.y, a.d, a.ncols_x, a.nrows_x, a.ids,
            a.stride_channel_x, a.stride_channel_dst, a.stride_channel_y, a.nchannels_y,
            a.nchannels_dst, a.warp_size, a.cc, a.fusion, a.stream);
    }
}

// Tier 2 (mmvq's switch_ncols_dst, the MIDDLE layer): runtime token-batch width ->
// compile-time ncols_dst. dq handles a single decode token, so only case 1 exists.
// mmvq additionally forks `has_ids && ncols_dst > 1` to a batched moe_launch here and
// derives rows-per-block via calc_rows_per_block; dq defers both (gated off by
// dst->ne[2]==1 in should_use_mmv_dq). Build the >1 case only on a measured dq win
// over the q8_1 mmvq path.
template <ggml_type type>
static void dq_switch_ncols_dst(const dq_launch_args & a, int ncols_dst) {
    switch (ncols_dst) {
        case 1:  dq_switch_fusion<type, 1>(a); break;
        default: GGML_ABORT("mul_mat_vec_dq: ncols_dst=%d unsupported (single decode token only)", ncols_dst);
    }
}

// Tier 1 (mmvq's switch_type): runtime ggml_type -> compile-time type. Per-type
// numerics live in dq_traits<type>; adding a type is a new case here plus a Traits.
static void dq_switch_type(ggml_type type, const dq_launch_args & a, int ncols_dst) {
    switch (type) {
        case GGML_TYPE_Q4_K: dq_switch_ncols_dst<GGML_TYPE_Q4_K>(a, ncols_dst); break;
        case GGML_TYPE_Q5_K: dq_switch_ncols_dst<GGML_TYPE_Q5_K>(a, ncols_dst); break;
        case GGML_TYPE_Q6_K: dq_switch_ncols_dst<GGML_TYPE_Q6_K>(a, ncols_dst); break;
        default: GGML_ABORT("mul_mat_vec_dq: unsupported type %s (should_use_mmv_dq must gate this)",
                            ggml_type_name(type));
    }
}

// ---- dispatch: dq as an internal variant of ggml_cuda_mul_mat_vec_q ----

bool ggml_cuda_should_use_mmv_dq(
        const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
        const ggml_tensor * dst, int cc, const ggml_cuda_mm_fusion_args_host * fusion) {
    const bool dq_default = GGML_CUDA_CC_IS_RDNA3_5(cc);
    if (!ggml_cuda_dq_mmv_enabled(dq_default)) {
        return false;   // dq globally disabled
    }

    // MoE (ids): per-expert matvec for a single decode token, plain or fused
    // gate+up SwiGLU. src0 is [K, N, n_expert]; the expert axis is selected via ids and
    // becomes the kernel's grid.y. The activation may be broadcast (gate/up: src1->ne[1]==1)
    // or per-expert (ffn_down: src1->ne[1]==n_expert_used==dst->ne[1]) — both handled by
    // channel_y in the kernel; the guard below is enforced further down. The fusion block
    // lower down validates SwiGLU/no-bias/same-shape and works for the 3D expert stacks.
    // Multi-token (dst->ne[2]>1) remains deferred.
    if (ids != nullptr) {
        if (ids->type != GGML_TYPE_I32) {
            return false;
        }
        if (dst->ne[2] != 1) {     // single decode token: ncols_dst == 1
            return false;
        }
    }

    const ggml_type type = src0->type;
    if (!(type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K || type == GGML_TYPE_Q6_K)) {
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

    // Fusion inspection: dq implements all gated GLU variants (SwiGLU/GEGLU/OAI/CLAMP, gate =
    // fusion->gate, shared activation) plus x/gate bias. Scale stays mmvq-only (NVFP4, dead for
    // K-quants).
    if (fusion) {
        if (fusion->x_scale || fusion->gate_scale) {
            return false;   // scale is NVFP4-only, dead for K-quants
        }
        if (fusion->gate && (fusion->gate->type != type || !ggml_are_same_shape(src0, fusion->gate)
                || !ggml_is_contiguous(fusion->gate))) {
            return false;
        }
    }

    return true;
}

void ggml_cuda_mul_mat_vec_dq(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
        const ggml_tensor * ids, ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion) {
    // src0 is the "up" weight; a fused gate (SwiGLU) rides in fusion.gate and the kernel's
    // has_gate branch handles both. ids/strides are identical for plain and fused.
    const dq_moe_dims m = dq_moe_setup(src0, src1, ids, dst);

    dq_fusion_args fargs;
    if (fusion) {
        if (fusion->gate)      fargs.gate      = fusion->gate->data;
        if (fusion->x_bias)    fargs.x_bias    = (const float *) fusion->x_bias->data;
        if (fusion->gate_bias) fargs.gate_bias = (const float *) fusion->gate_bias->data;
        fargs.glu_op    = fusion->glu_op;
        fargs.glu_limit = fusion->glu_limit;
    }

    const dq_launch_args a = {
        /* vx                 */ src0->data,
        /* y                  */ (const float *) src1->data,
        /* d                  */ (float *) dst->data,
        /* ncols_x            */ (int) src0->ne[0],
        /* nrows_x            */ (int) src0->ne[1],
        /* ids                */ m.ids,
        /* stride_channel_x   */ m.stride_channel_x,
        /* stride_channel_dst */ m.stride_channel_dst,
        /* stride_channel_y   */ m.stride_channel_y,
        /* nchannels_y        */ m.nchannels_y,
        /* nchannels_dst      */ m.nchannels_dst,
        /* warp_size          */ ggml_cuda_info().devices[ctx.device].warp_size,
        /* cc                 */ ggml_cuda_info().devices[ctx.device].cc,
        /* fusion             */ fargs,
        /* stream             */ ctx.stream(),
    };

    const int ncols_dst = 1;   // single decode token; dst->ne[2]==1 gated upstream
    dq_switch_type(src0->type, a, ncols_dst);
}
