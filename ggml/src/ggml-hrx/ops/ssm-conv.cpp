#include "ggml-hrx-ops.h"

#include <climits>

bool ggml_hrx_supports_ssm_conv(const ggml_hrx_device_context * dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);
    const ggml_tensor * sx = op->src[0];
    const ggml_tensor * c  = op->src[1];
    if (sx == nullptr || c == nullptr) {
        return false;
    }
    if (op->type != GGML_TYPE_F32 || sx->type != GGML_TYPE_F32 || c->type != GGML_TYPE_F32) {
        return false;
    }
    const int64_t d_conv  = c->ne[0];
    const int64_t d_inner = c->ne[1];
    if (sx->ne[0] != d_conv - 1 + op->ne[1] || sx->ne[1] != d_inner || sx->ne[2] != op->ne[2] || sx->ne[3] != 1 ||
        op->ne[0] != d_inner || op->ne[3] != 1 || !ggml_is_contiguous(op)) {
        return false;
    }
    int64_t s[GGML_MAX_DIMS];
    if (!ggml_hrx_elem_strides(sx, s) || !ggml_hrx_elem_strides(c, s)) {
        return false;
    }
    // Indices and spans are 32-bit launch scalars.
    return ggml_nelements(op) < INT_MAX && ggml_nbytes(sx) / 4 < INT_MAX && ggml_nbytes(c) / 4 < INT_MAX;
}

bool ggml_hrx_op_ssm_conv(ggml_backend_hrx_context * ctx, const ggml_tensor * node) {
    const ggml_tensor * sx = node->src[0];
    const ggml_tensor * c  = node->src[1];

    int64_t sxs[GGML_MAX_DIMS];
    int64_t cs[GGML_MAX_DIMS];
    ggml_hrx_elem_strides(sx, sxs);
    ggml_hrx_elem_strides(c, cs);

    // The tiled kernel makes reads and writes coalesce. Its workgroup memory holds at most 9 taps.
    const bool tiled = sxs[0] == 1 && c->ne[0] <= 9 && node->ne[1] > 1;

    ggml_hrx_kernel_desc desc;
    desc.file   = "ssm_conv/ssm_conv_f32.loom";
    desc.symbol = tiled ? "ggml_ssm_conv_f32_tiled" : "ggml_ssm_conv_f32";
    desc.config.push_back(ggml_hrx_elementwise_wg_config(ctx->device));
    ggml_hrx_kernel * kernel = ggml_hrx_get_kernel(ctx->device, desc);

    const int64_t n = ggml_nelements(node);
    const std::vector<int64_t> workload = tiled ?
        std::vector<int64_t>{ (node->ne[0] + 31) / 32, (node->ne[1] + 31) / 32, node->ne[2] } :
        std::vector<int64_t>{ n };
    return ggml_hrx_dispatch(ctx, kernel, workload,
        {
            n, node->ne[0], node->ne[1], c->ne[0],
            sxs[0], sxs[1], sxs[2], (int64_t) (ggml_nbytes(sx) / 4),
            cs[0], cs[1], (int64_t) (ggml_nbytes(c) / 4),
        },
        { sx, c, node });
}
