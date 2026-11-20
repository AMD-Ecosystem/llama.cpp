#include "ggml-hrx-ops.h"

#include <climits>
#include <cstring>
#include <string>
#include <vector>

static bool ggml_hrx_supports_row_norm(const ggml_tensor * op, bool elem_strided) {
    const ggml_tensor * src = op->src[0];
    if (src->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
        return false;
    }
    int64_t ss[GGML_MAX_DIMS];
    int64_t ds[GGML_MAX_DIMS];
    if (!ggml_hrx_elem_strides(src, ss) || !ggml_hrx_elem_strides(op, ds)) {
        return false;
    }
    if (!elem_strided && (ss[0] != 1 || ds[0] != 1)) {
        return false;
    }
    float eps;
    memcpy(&eps, op->op_params, sizeof(float));
    if (!(eps >= 0.0f)) {
        return false;
    }
    // Sizes, counts, strides and spans are 32-bit launch scalars.
    return ggml_nrows(op) < INT_MAX && ss[0] < INT_MAX && ds[0] < INT_MAX &&
           ggml_nbytes(src) / 4 < INT_MAX && ggml_nbytes(op) / 4 < INT_MAX;
}

static bool ggml_hrx_row_norm(ggml_backend_hrx_context * ctx, const ggml_tensor * node, const char * file,
                              const char * symbol, bool elem_strided) {
    const ggml_tensor * src = node->src[0];
    if (ggml_nelements(node) == 0) {
        return true;
    }

    int64_t ss[GGML_MAX_DIMS];
    int64_t ds[GGML_MAX_DIMS];
    ggml_hrx_elem_strides(src, ss);
    ggml_hrx_elem_strides(node, ds);

    uint32_t eps_bits;
    memcpy(&eps_bits, node->op_params, sizeof(uint32_t));

    const int64_t rows = ggml_nrows(node);
    const int64_t ne0  = src->ne[0];
    std::vector<int64_t> workload = { rows };
    ggml_hrx_kernel_desc desc;
    desc.file   = file;
    desc.symbol = symbol;
    desc.config = { { "ggml.norm.wg_size", std::to_string(ctx->device->tune->norm_wg_size) } };
    // The rows kernel needs all offsets below 2^31 - 8.
    const int64_t max_elems = INT_MAX - 1024;
    if (!elem_strided && ne0 % 4 == 0 && ne0 <= ctx->device->tune->rms_norm_rows_max && rows <= max_elems &&
        (int64_t) (ggml_nbytes(src) / 4) <= max_elems && (int64_t) (ggml_nbytes(node) / 4) <= max_elems) {
        int64_t lanes = 16;
        while (lanes < ctx->device->tune->rms_norm_lanes_max && 4 * lanes < ne0) {
            lanes *= 2;
        }
        const int64_t vecs  = (ne0 + 4 * lanes - 1) / (4 * lanes);
        const int64_t rows_per_group = ctx->device->tune->norm_wg_size / lanes;
        desc.symbol = "ggml_rms_norm_f32_rows";
        desc.config.push_back({ "ggml.rms_norm.lanes", std::to_string(lanes) });
        desc.config.push_back({ "ggml.rms_norm.vecs", std::to_string(vecs) });
        desc.libs   = { "motifs/reduce.loom" };
        workload    = { (rows + rows_per_group - 1) / rows_per_group };
    }
    ggml_hrx_kernel * kernel = ggml_hrx_get_kernel(ctx->device, desc);

    std::vector<int64_t> scalars = {
        rows, src->ne[0], src->ne[1], src->ne[2],
        ss[1], ss[2], ss[3], (int64_t) (ggml_nbytes(src) / 4),
        ds[1], ds[2], ds[3], (int64_t) (ggml_nbytes(node) / 4),
        (int64_t) eps_bits,
    };
    if (elem_strided) {
        scalars.push_back(ss[0]);
        scalars.push_back(ds[0]);
    }
    return ggml_hrx_dispatch(ctx, kernel, workload, scalars, { src, node });
}

bool ggml_hrx_supports_rms_norm(const ggml_hrx_device_context * dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);
    return ggml_hrx_supports_row_norm(op, false);
}

bool ggml_hrx_op_rms_norm(ggml_backend_hrx_context * ctx, const ggml_tensor * node) {
    return ggml_hrx_row_norm(ctx, node, "norm/rms_norm_f32.loom", "ggml_rms_norm_f32", false);
}

bool ggml_hrx_supports_l2_norm(const ggml_hrx_device_context * dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);
    return ggml_hrx_supports_row_norm(op, true);
}

bool ggml_hrx_op_l2_norm(ggml_backend_hrx_context * ctx, const ggml_tensor * node) {
    return ggml_hrx_row_norm(ctx, node, "norm/l2_norm_f32.loom", "ggml_l2_norm_f32", true);
}
