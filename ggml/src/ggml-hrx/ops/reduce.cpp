#include "ggml-hrx-ops.h"

#include <climits>
#include <string>

bool ggml_hrx_supports_argsort(const ggml_hrx_device_context * dev, const ggml_tensor * op) {
    const ggml_tensor * src = op->src[0];
    if (src->type != GGML_TYPE_F32 || op->type != GGML_TYPE_I32) {
        return false;
    }
    const int32_t order = ggml_get_op_params_i32(op, 0);
    if (order != GGML_SORT_ORDER_ASC && order != GGML_SORT_ORDER_DESC) {
        return false;
    }
    if (!ggml_is_contiguous(src) || !ggml_is_contiguous(op)) {
        return false;
    }
    return src->ne[0] <= dev->tune->argsort_max_row && ggml_nelements(src) < INT_MAX;
}

bool ggml_hrx_op_argsort(ggml_backend_hrx_context * ctx, const ggml_tensor * node) {
    const ggml_tensor * src = node->src[0];
    if (ggml_nelements(src) == 0) {
        return true;
    }

    ggml_hrx_kernel_desc desc;
    desc.file   = "reduce/argsort_f32.loom";
    desc.symbol = "ggml_argsort_f32";
    desc.config = {
        { "ggml.argsort.order", ggml_get_op_params_i32(node, 0) == GGML_SORT_ORDER_DESC ? "1" : "0" },
        { "ggml.argsort.parts", std::to_string(ctx->device->tune->argsort_parts) },
    };
    ggml_hrx_kernel * kernel = ggml_hrx_get_kernel(ctx->device, desc);

    const int64_t rows = ggml_nrows(src);
    return ggml_hrx_dispatch(ctx, kernel, { rows, src->ne[0] }, { rows, src->ne[0], ggml_nelements(src) }, { src, node });
}

bool ggml_hrx_supports_sum_rows(const ggml_hrx_device_context * dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);
    const ggml_tensor * src = op->src[0];
    if (src->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
        return false;
    }
    int64_t ss[GGML_MAX_DIMS];
    int64_t ds[GGML_MAX_DIMS];
    if (!ggml_hrx_elem_strides(src, ss) || !ggml_hrx_elem_strides(op, ds) || ss[0] != 1) {
        return false;
    }
    // Sizes, counts and spans are 32-bit launch scalars.
    return ggml_nrows(src) < INT_MAX && ggml_nbytes(src) / 4 < INT_MAX && ggml_nbytes(op) / 4 < INT_MAX;
}

bool ggml_hrx_op_sum_rows(ggml_backend_hrx_context * ctx, const ggml_tensor * node) {
    const ggml_tensor * src = node->src[0];
    if (ggml_nelements(src) == 0) {
        return true;
    }

    int64_t ss[GGML_MAX_DIMS];
    int64_t ds[GGML_MAX_DIMS];
    ggml_hrx_elem_strides(src, ss);
    ggml_hrx_elem_strides(node, ds);

    ggml_hrx_kernel_desc desc;
    desc.file   = "reduce/sum_rows_f32.loom";
    desc.symbol = "ggml_sum_rows_f32";
    desc.config = { { "ggml.norm.wg_size", std::to_string(ctx->device->tune->norm_wg_size) } };
    ggml_hrx_kernel * kernel = ggml_hrx_get_kernel(ctx->device, desc);

    const int64_t rows = ggml_nrows(src);
    return ggml_hrx_dispatch(ctx, kernel, { rows },
        {
            rows, src->ne[0], src->ne[1], src->ne[2],
            ss[1], ss[2], ss[3], (int64_t) (ggml_nbytes(src) / 4),
            ds[1], ds[2], ds[3], (int64_t) (ggml_nbytes(node) / 4),
        },
        { src, node });
}
