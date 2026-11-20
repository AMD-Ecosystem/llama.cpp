#include "ggml-hrx-ops.h"

#include <climits>
#include <string>

// Must match the dst type codes in copy/cpy_f32.loom.
static int ggml_hrx_cpy_dst_code(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32: return 0;
        case GGML_TYPE_F16: return 1;
        case GGML_TYPE_I32: return 2;
        default:            return -1;
    }
}

// Indices and spans are 32-bit launch scalars.
static bool ggml_hrx_fits_i32(const ggml_tensor * t) {
    return ggml_nelements(t) < INT_MAX && ggml_nbytes(t) / ggml_type_size(t->type) < INT_MAX;
}

// The kernel does not use the stride of a size-1 dim. Set it to 0.
static bool ggml_hrx_launch_strides(const ggml_tensor * t, int64_t s[GGML_MAX_DIMS]) {
    if (!ggml_hrx_elem_strides(t, s)) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (t->ne[i] == 1) {
            s[i] = 0;
        }
        if (s[i] >= INT_MAX) {
            return false;
        }
    }
    return true;
}

static bool ggml_hrx_can_cpy(const ggml_tensor * src, const ggml_tensor * dst) {
    if (src->type != GGML_TYPE_F32 || ggml_hrx_cpy_dst_code(dst->type) < 0 ||
        ggml_nelements(src) != ggml_nelements(dst)) {
        return false;
    }
    int64_t s[GGML_MAX_DIMS];
    if (!ggml_hrx_launch_strides(src, s) || !ggml_hrx_launch_strides(dst, s)) {
        return false;
    }
    return ggml_hrx_fits_i32(src) && ggml_hrx_fits_i32(dst);
}

static bool ggml_hrx_dispatch_cpy(ggml_backend_hrx_context * ctx, const ggml_tensor * src, const ggml_tensor * dst) {
    int64_t ss[GGML_MAX_DIMS];
    int64_t ts[GGML_MAX_DIMS];
    ggml_hrx_launch_strides(src, ss);
    ggml_hrx_launch_strides(dst, ts);

    const int64_t n      = ggml_nelements(dst);
    const int64_t s_span = (int64_t) (ggml_nbytes(src) / ggml_type_size(src->type));
    const int64_t t_span = (int64_t) (ggml_nbytes(dst) / ggml_type_size(dst->type));
    const int64_t groups = dst->type == GGML_TYPE_F32 && n % 4 == 0 ? ggml_hrx_stream_groups(ctx->device, n) : 0;
    const bool    contig = ggml_is_contiguous(src) && ggml_is_contiguous(dst);
    // The rows kernel needs all indices below 2^31 - 8.
    const bool    rows   = ggml_are_same_shape(src, dst) && src->ne[0] % 4 == 0 && ss[0] == 1 && ts[0] == 1 &&
                           s_span <= INT_MAX - 4096 && t_span <= INT_MAX - 4096;
    if (groups > 0 && (contig || rows)) {
        ggml_hrx_kernel_desc desc;
        desc.file   = "copy/cpy_f32.loom";
        desc.symbol = contig ? "ggml_cpy_f32_stream" : "ggml_cpy_f32_stream_rows";
        desc.config.push_back(ggml_hrx_elementwise_wg_config(ctx->device));
        ggml_hrx_kernel * kernel = ggml_hrx_get_kernel(ctx->device, desc);
        if (contig) {
            return ggml_hrx_dispatch(ctx, kernel, { groups }, { n, groups }, { src, dst });
        }
        return ggml_hrx_dispatch(ctx, kernel, { groups },
            { n, groups, src->ne[0], src->ne[1], src->ne[2], ss[1], ss[2], ss[3], s_span, ts[1], ts[2], ts[3], t_span },
            { src, dst });
    }

    ggml_hrx_kernel_desc desc;
    desc.file   = "copy/cpy_f32.loom";
    desc.symbol = "ggml_cpy_f32";
    desc.config = { { "ggml.cpy.dst_type", std::to_string(ggml_hrx_cpy_dst_code(dst->type)) } };
    desc.config.push_back(ggml_hrx_elementwise_wg_config(ctx->device));
    ggml_hrx_kernel * kernel = ggml_hrx_get_kernel(ctx->device, desc);

    // src and dst can have different shapes. Both use logical element order.
    return ggml_hrx_dispatch(ctx, kernel, { n },
        {
            n,
            src->ne[0], src->ne[1], src->ne[2], ss[0], ss[1], ss[2], ss[3], s_span,
            dst->ne[0], dst->ne[1], dst->ne[2], ts[0], ts[1], ts[2], ts[3], t_span,
        },
        { src, dst });
}

bool ggml_hrx_supports_cpy(const ggml_hrx_device_context * dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);
    const ggml_tensor * src = op->src[0];
    return src != nullptr && ggml_hrx_can_cpy(src, op);
}

bool ggml_hrx_op_cpy(ggml_backend_hrx_context * ctx, const ggml_tensor * node) {
    return ggml_hrx_dispatch_cpy(ctx, node->src[0], node);
}

bool ggml_hrx_supports_concat(const ggml_hrx_device_context * dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);
    const ggml_tensor * a   = op->src[0];
    const ggml_tensor * b   = op->src[1];
    const int32_t       dim = ggml_get_op_params_i32(op, 0);
    if (a == nullptr || b == nullptr || dim < 0 || dim >= GGML_MAX_DIMS) {
        return false;
    }
    if (op->type != GGML_TYPE_F32 || a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32 || !ggml_is_contiguous(op)) {
        return false;
    }
    // An empty source has no data range to bind. Thus copy the other source.
    if (ggml_is_empty(a) || ggml_is_empty(b)) {
        const ggml_tensor * src = ggml_is_empty(a) ? b : a;
        return ggml_is_empty(op) || (ggml_are_same_shape(src, op) && ggml_hrx_can_cpy(src, op));
    }
    int64_t s[GGML_MAX_DIMS];
    if (!ggml_hrx_launch_strides(a, s) || !ggml_hrx_launch_strides(b, s)) {
        return false;
    }
    return ggml_hrx_fits_i32(op) && ggml_hrx_fits_i32(a) && ggml_hrx_fits_i32(b);
}

bool ggml_hrx_op_concat(ggml_backend_hrx_context * ctx, const ggml_tensor * node) {
    const ggml_tensor * a   = node->src[0];
    const ggml_tensor * b   = node->src[1];
    const int32_t       dim = ggml_get_op_params_i32(node, 0);

    if (ggml_is_empty(a) || ggml_is_empty(b)) {
        return ggml_hrx_dispatch_cpy(ctx, ggml_is_empty(a) ? b : a, node);
    }

    int64_t as[GGML_MAX_DIMS];
    int64_t bs[GGML_MAX_DIMS];
    ggml_hrx_launch_strides(a, as);
    ggml_hrx_launch_strides(b, bs);

    int64_t b_off[GGML_MAX_DIMS] = { 0, 0, 0, 0 };
    b_off[dim] = a->ne[dim];

    // A transposed source goes through a square tile. This makes reads and writes coalesce.
    const bool tiled = node->ne[1] > 1 && ((as[0] > 1 && as[1] == 1) || (bs[0] > 1 && bs[1] == 1));

    ggml_hrx_kernel_desc desc;
    desc.file   = "copy/concat_f32.loom";
    desc.symbol = tiled ? "ggml_concat_f32_tiled" : "ggml_concat_f32";
    const int64_t tile = ctx->device->tune->concat_tile;
    desc.config.push_back(ggml_hrx_elementwise_wg_config(ctx->device));
    desc.config.push_back({ "ggml.concat.tile", std::to_string(tile) });
    ggml_hrx_kernel * kernel = ggml_hrx_get_kernel(ctx->device, desc);

    const int64_t n = ggml_nelements(node);
    const std::vector<int64_t> workload = tiled ?
        std::vector<int64_t>{ (node->ne[1] + tile - 1) / tile, (node->ne[0] + tile - 1) / tile, node->ne[2] * node->ne[3] } :
        std::vector<int64_t>{ n };
    return ggml_hrx_dispatch(ctx, kernel, workload,
        {
            n, node->ne[0], node->ne[1], node->ne[2],
            a->ne[0], a->ne[1], a->ne[2], a->ne[3], as[0], as[1], as[2], as[3], (int64_t) (ggml_nbytes(a) / 4),
            b_off[0], b_off[1], b_off[2], b_off[3], bs[0], bs[1], bs[2], bs[3], (int64_t) (ggml_nbytes(b) / 4),
        },
        { a, b, node });
}

bool ggml_hrx_supports_pad(const ggml_hrx_device_context * dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);
    const ggml_tensor * src = op->src[0];
    if (src == nullptr || op->type != GGML_TYPE_F32 || src->type != GGML_TYPE_F32 || !ggml_is_contiguous(op)) {
        return false;
    }
    // Circular padding is not supported.
    if (ggml_get_op_params_i32(op, 8) != 0) {
        return false;
    }
    for (int i = 0; i < 2 * GGML_MAX_DIMS; ++i) {
        if (ggml_get_op_params_i32(op, i) < 0) {
            return false;
        }
    }
    int64_t s[GGML_MAX_DIMS];
    if (!ggml_is_empty(src) && (!ggml_hrx_launch_strides(src, s) || !ggml_hrx_fits_i32(src))) {
        return false;
    }
    return ggml_hrx_fits_i32(op);
}

bool ggml_hrx_op_pad(ggml_backend_hrx_context * ctx, const ggml_tensor * node) {
    const ggml_tensor * src = node->src[0];

    // An empty src has no data range to bind. Bind dst instead. The kernel does not read it.
    const bool          empty = ggml_is_empty(src);
    const ggml_tensor * bound = empty ? node : src;
    int64_t ss[GGML_MAX_DIMS] = { 0, 0, 0, 0 };
    if (!empty) {
        ggml_hrx_launch_strides(src, ss);
    }

    ggml_hrx_kernel_desc desc;
    desc.file   = "copy/pad_f32.loom";
    desc.symbol = "ggml_pad_f32";
    desc.config.push_back(ggml_hrx_elementwise_wg_config(ctx->device));
    ggml_hrx_kernel * kernel = ggml_hrx_get_kernel(ctx->device, desc);

    const int64_t n = ggml_nelements(node);
    return ggml_hrx_dispatch(ctx, kernel, { n },
        {
            n, node->ne[0], node->ne[1], node->ne[2],
            src->ne[0], src->ne[1], src->ne[2], src->ne[3],
            ggml_get_op_params_i32(node, 0), ggml_get_op_params_i32(node, 2),
            ggml_get_op_params_i32(node, 4), ggml_get_op_params_i32(node, 6),
            ss[0], ss[1], ss[2], ss[3], (int64_t) (ggml_nbytes(bound) / sizeof(float)),
        },
        { bound, node });
}
