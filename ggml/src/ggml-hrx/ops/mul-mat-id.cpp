#include "ggml-hrx-ops.h"

#include <algorithm>
#include <climits>
#include <string>

// Weight addressing and arch helpers. ops/mul-mat.cpp defines them.
int     ggml_hrx_mul_mat_weight_type(ggml_type type);
bool    ggml_hrx_mul_mat_weight_strides(const ggml_tensor * w, int64_t s[GGML_MAX_DIMS]);
int64_t ggml_hrx_mul_mat_weight_span(const ggml_tensor * w);
bool    ggml_hrx_mul_mat_wmma_arch(const ggml_hrx_device_context * dev);

bool ggml_hrx_supports_mul_mat_id(const ggml_hrx_device_context * dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);
    const ggml_tensor * w   = op->src[0];
    const ggml_tensor * x   = op->src[1];
    const ggml_tensor * ids = op->src[2];
    if (ggml_hrx_mul_mat_weight_type(w->type) < 0 || x->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32 ||
        op->type != GGML_TYPE_F32) {
        return false;
    }
    int64_t s[GGML_MAX_DIMS];
    if (!ggml_hrx_mul_mat_weight_strides(w, s) || !ggml_hrx_elem_strides(x, s) || !ggml_hrx_elem_strides(ids, s) ||
        !ggml_hrx_elem_strides(op, s)) {
        return false;
    }
    if (x->view_offs % sizeof(float) != 0 || ids->view_offs % sizeof(int32_t) != 0 || op->nb[0] != sizeof(float)) {
        return false;
    }
    // An empty tensor cannot be a buffer binding.
    if (ggml_is_empty(w) || ggml_is_empty(x) || ggml_is_empty(ids) || ggml_is_empty(op)) {
        return false;
    }
    // Indices, strides and spans are 32-bit launch scalars.
    return ggml_nbytes(w) <= INT_MAX && ggml_nbytes(x) <= INT_MAX && ggml_nbytes(ids) <= INT_MAX &&
           ggml_nbytes(op) <= INT_MAX;
}

// The matrix-op kernel needs K in steps of 32 and contiguous input rows.
// F32 and BF16 weights use the f32 kernel. Rounding them to f16 loses range or precision.
// The kernel converts src1 to F16. The src1 precision request must allow this.
static bool ggml_hrx_mul_mat_id_use_wmma(const ggml_tensor * node, const ggml_tensor * w, const int64_t xs[GGML_MAX_DIMS]) {
    const ggml_prec prec_src1 = ggml_prec_get_src(node, 1);
    if (prec_src1 != GGML_PREC_UNDEFINED && prec_src1 < GGML_PREC_F16) {
        return false;
    }
    switch (w->type) {
        case GGML_TYPE_F16:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q8_0:
            break;
        default:
            return false;
    }
    return w->ne[0] % 32 == 0 && xs[0] == 1;
}

bool ggml_hrx_op_mul_mat_id(ggml_backend_hrx_context * ctx, const ggml_tensor * node) {
    const ggml_tensor * w   = node->src[0];
    const ggml_tensor * x   = node->src[1];
    const ggml_tensor * ids = node->src[2];

    int64_t ws[GGML_MAX_DIMS];
    int64_t xs[GGML_MAX_DIMS];
    int64_t ids_s[GGML_MAX_DIMS];
    int64_t ds[GGML_MAX_DIMS];
    ggml_hrx_mul_mat_weight_strides(w, ws);
    ggml_hrx_elem_strides(x, xs);
    ggml_hrx_elem_strides(ids, ids_s);
    ggml_hrx_elem_strides(node, ds);

    const int64_t n_rows   = w->ne[1];
    const int64_t n_expert = w->ne[2];
    const int64_t n_used   = ids->ne[0];
    const int64_t n_tokens = ids->ne[1];
    const int64_t n_pairs  = n_used * n_tokens;
    // The vector K-quant kernel reads 16-byte packets and addresses whole 256-value blocks.
    const size_t  w_block  = ggml_type_size(w->type);
    const size_t  x_block  = ggml_blck_size(w->type) * sizeof(float);
    const bool    kq_rows  = (w->type == GGML_TYPE_Q4_K || w->type == GGML_TYPE_Q5_K || w->type == GGML_TYPE_Q6_K) &&
                             w->nb[1] % w_block == 0 && w->nb[2] % w_block == 0 && x->nb[0] == sizeof(float) &&
                             x->nb[1] % x_block == 0 && x->nb[2] % x_block == 0;
    // Above max_pairs, the tiled kernels read the weights of each expert one time for all its pairs.
    // With few pairs per expert, the vector K-quant kernel is as fast as the matrix-op kernel or faster.
    const ggml_hrx_mul_mat_id_vec_tune & iv = ctx->device->tune->mul_mat_id_vec[w->type];
    const int64_t max_pairs    = iv.max_pairs;
    const bool    vec_kq   = kq_rows && n_pairs <= std::max(max_pairs, iv.kq_pairs_per_expert * n_expert);
    const bool    vec      = vec_kq || n_pairs <= max_pairs;
    const bool    wmma     = !vec && ggml_hrx_mul_mat_wmma_arch(ctx->device) && ggml_hrx_mul_mat_id_use_wmma(node, w, xs);

    ggml_hrx_kernel_desc desc;
    if (wmma) {
        desc.file   = "matmul/mul_mat_id_wmma.loom";
        desc.symbol = "ggml_mul_mat_id_wmma";
    } else {
        desc.file   = "matmul/mul_mat_id.loom";
        desc.symbol = vec ? "ggml_mul_mat_id_vec" : "ggml_mul_mat_id_tiled";
    }
    desc.libs   = { "motifs/common.loom", "motifs/float.loom", "motifs/kq.loom", "motifs/q6_k.loom", "motifs/q8_0.loom",
                    "motifs/reduce.loom", "matmul/lib/mm_core.loom" };
    if (wmma) {
        desc.libs.push_back("matmul/lib/mm_wmma.loom");
        desc.libs.push_back("motifs/wmma.loom");
    }
    desc.config = { { "ggml.mul_mat_id.weight_type", std::to_string(ggml_hrx_mul_mat_weight_type(w->type)) } };
    if (vec_kq) {
        desc.symbol = "ggml_mul_mat_id_vec_kq";
        const int64_t n_blocks = w->ne[0] / ggml_blck_size(w->type);
        int64_t       slots    = iv.kq_slots;
        while (n_blocks % slots != 0) {
            slots /= 2;
        }
        desc.config.push_back({ "ggml.mul_mat_id.kq_slots", std::to_string(slots) });
    }
    if (wmma) {
        const ggml_hrx_mul_mat_id_wmma_tune & iw = ctx->device->tune->mul_mat_id_wmma[w->type];
        // Skewed routing gives some experts many chunks.
        desc.config.push_back({ "ggml.mul_mat_id.chunk_slots", std::to_string(iw.chunk_slots) });
        desc.config.push_back({ "ggml.mul_mat_id.row_groups", std::to_string(iw.row_groups) });
    }
    ggml_hrx_kernel * kernel = ggml_hrx_get_kernel(ctx->device, desc);

    std::vector<int64_t> workload;
    if (vec) {
        workload = { n_rows, n_used, n_tokens };
    } else {
        workload = { n_rows, n_expert };
    }

    return ggml_hrx_dispatch(ctx, kernel, workload,
        {
            w->ne[0], n_rows, n_expert, n_used, n_tokens, x->ne[1],
            ws[1], ws[2], ggml_hrx_mul_mat_weight_span(w),
            xs[0], xs[1], xs[2], (int64_t) (ggml_nbytes(x) / sizeof(float)),
            ids_s[0], ids_s[1], (int64_t) (ggml_nbytes(ids) / sizeof(int32_t)),
            ds[1], ds[2], (int64_t) (ggml_nbytes(node) / sizeof(float)),
        },
        { w, x, ids, node });
}
