#include "ggml-hrx-ops.h"

#include <climits>
#include <string>

bool ggml_hrx_supports_gated_delta_net(const ggml_hrx_device_context * dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);
    const ggml_tensor * q     = op->src[0];
    const ggml_tensor * k     = op->src[1];
    const ggml_tensor * v     = op->src[2];
    const ggml_tensor * g     = op->src[3];
    const ggml_tensor * beta  = op->src[4];
    const ggml_tensor * state = op->src[5];
    if (q == nullptr || k == nullptr || v == nullptr || g == nullptr || beta == nullptr || state == nullptr) {
        return false;
    }
    for (const ggml_tensor * t : { op, q, k, v, g, beta, state }) {
        if (t->type != GGML_TYPE_F32) {
            return false;
        }
    }

    const int64_t S        = v->ne[0];
    const int64_t H        = v->ne[1];
    const int64_t n_tokens = v->ne[2];
    const int64_t n_seqs   = v->ne[3];
    const int64_t K        = ggml_get_op_params_i32(op, 0);

    // 4 or more lanes share one state column. Each lane keeps at most S/4 rows in registers.
    if (S % 4 != 0 || S < 4 || S > 128 || n_tokens < 1 || K < 1) {
        return false;
    }
    if (q->ne[0] != S || k->ne[0] != S || q->ne[2] != n_tokens || k->ne[2] != n_tokens) {
        return false;
    }
    if (q->ne[1] < 1 || k->ne[1] < 1 || q->ne[3] < 1 || k->ne[3] < 1) {
        return false;
    }
    if (H % q->ne[1] != 0 || H % k->ne[1] != 0 || n_seqs % q->ne[3] != 0 || n_seqs % k->ne[3] != 0) {
        return false;
    }
    if (q->nb[0] != sizeof(float) || k->nb[0] != sizeof(float) || v->nb[0] != sizeof(float)) {
        return false;
    }
    if ((g->ne[0] != 1 && g->ne[0] != S) || g->ne[1] != H || g->ne[2] != n_tokens || g->ne[3] != n_seqs) {
        return false;
    }
    if (beta->ne[0] != 1 || beta->ne[1] != H || beta->ne[2] != n_tokens || beta->ne[3] != n_seqs) {
        return false;
    }
    if (state->ne[0] != S || state->ne[1] != S || state->ne[2] != H || state->ne[3] != n_seqs) {
        return false;
    }
    if (!ggml_is_contiguous(g) || !ggml_is_contiguous(beta) || !ggml_is_contiguous(state) || !ggml_is_contiguous(op)) {
        return false;
    }
    if (op->ne[0] != S * H || op->ne[1] != n_tokens * n_seqs + K * S * n_seqs) {
        return false;
    }
    int64_t s[GGML_MAX_DIMS];
    if (!ggml_hrx_elem_strides(q, s) || !ggml_hrx_elem_strides(k, s) || !ggml_hrx_elem_strides(v, s)) {
        return false;
    }
    // Indices and spans are 32-bit launch scalars.
    return ggml_nelements(op) < INT_MAX && ggml_nbytes(q) / 4 < INT_MAX && ggml_nbytes(k) / 4 < INT_MAX &&
           ggml_nbytes(v) / 4 < INT_MAX && ggml_nelements(g) < INT_MAX && ggml_nelements(state) < INT_MAX;
}

bool ggml_hrx_op_gated_delta_net(ggml_backend_hrx_context * ctx, const ggml_tensor * node) {
    const ggml_tensor * q     = node->src[0];
    const ggml_tensor * k     = node->src[1];
    const ggml_tensor * v     = node->src[2];
    const ggml_tensor * g     = node->src[3];
    const ggml_tensor * beta  = node->src[4];
    const ggml_tensor * state = node->src[5];

    int64_t qs[GGML_MAX_DIMS];
    int64_t ks[GGML_MAX_DIMS];
    int64_t vs[GGML_MAX_DIMS];
    ggml_hrx_elem_strides(q, qs);
    ggml_hrx_elem_strides(k, ks);
    ggml_hrx_elem_strides(v, vs);

    const int64_t S        = v->ne[0];
    const int64_t H        = v->ne[1];
    const int64_t n_tokens = v->ne[2];
    const int64_t n_seqs   = v->ne[3];
    const int64_t K        = ggml_get_op_params_i32(node, 0);

    // S is a multiple of 4. Thus the loop ends at 4 lanes or more.
    int64_t lanes = ctx->device->tune->gated_delta_net_lanes;
    while (S % lanes != 0) {
        lanes /= 2;
    }
    const int64_t cols = 64 / lanes;

    ggml_hrx_kernel_desc desc;
    desc.file   = "gated_delta_net/gated_delta_net_f32.loom";
    desc.symbol = "ggml_gated_delta_net_f32";
    desc.libs   = { "motifs/reduce.loom" };
    desc.config = {
        { "ggml.gated_delta_net.head_width", std::to_string(S) },
        { "ggml.gated_delta_net.gate_per_channel", g->ne[0] == S ? "1" : "0" },
        { "ggml.gated_delta_net.chains", std::to_string(ctx->device->tune->gated_delta_net_chains) },
        { "ggml.gated_delta_net.lanes", std::to_string(lanes) },
    };
    ggml_hrx_kernel * kernel = ggml_hrx_get_kernel(ctx->device, desc);

    // Each workgroup of 64 lanes does cols state columns.
    return ggml_hrx_dispatch(ctx, kernel, { (S + cols - 1) / cols, H, n_seqs },
        {
            H, n_tokens, n_seqs, K,
            q->ne[1], n_seqs / q->ne[3], qs[1], qs[2], qs[3], (int64_t) (ggml_nbytes(q) / 4),
            k->ne[1], n_seqs / k->ne[3], ks[1], ks[2], ks[3], (int64_t) (ggml_nbytes(k) / 4),
            vs[1], vs[2], vs[3], (int64_t) (ggml_nbytes(v) / 4),
            ggml_nelements(g), ggml_nelements(beta), ggml_nelements(state), ggml_nelements(node),
        },
        { q, k, v, g, beta, state, node });
}
