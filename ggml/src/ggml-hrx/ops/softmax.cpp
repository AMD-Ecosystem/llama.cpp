#include "ggml-hrx-ops.h"

#include <climits>
#include <cstring>

bool ggml_hrx_supports_soft_max(const ggml_hrx_device_context * dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);
    const ggml_tensor * src  = op->src[0];
    const ggml_tensor * mask = op->src[1];
    if (src->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
        return false;
    }
    // Sinks and ALiBi (max_bias > 0) are not supported.
    float max_bias;
    memcpy(&max_bias, (const float *) op->op_params + 1, sizeof(float));
    if (op->src[2] != nullptr || max_bias != 0.0f) {
        return false;
    }
    int64_t ss[GGML_MAX_DIMS];
    int64_t ds[GGML_MAX_DIMS];
    if (!ggml_hrx_elem_strides(src, ss) || !ggml_hrx_elem_strides(op, ds) || ss[0] != 1 || ds[0] != 1) {
        return false;
    }
    if (mask != nullptr) {
        int64_t ms[GGML_MAX_DIMS];
        if (mask->type != GGML_TYPE_F32 && mask->type != GGML_TYPE_F16) {
            return false;
        }
        if (!ggml_hrx_elem_strides(mask, ms) || ms[0] != 1 || mask->ne[0] != src->ne[0] || mask->ne[1] < src->ne[1] ||
            src->ne[2] % mask->ne[2] != 0 || src->ne[3] % mask->ne[3] != 0) {
            return false;
        }
        if (ggml_nbytes(mask) / ggml_type_size(mask->type) >= INT_MAX) {
            return false;
        }
    }
    // Sizes, counts and spans are 32-bit launch scalars.
    return ggml_nrows(op) < INT_MAX && ggml_nbytes(src) / 4 < INT_MAX && ggml_nbytes(op) / 4 < INT_MAX;
}

bool ggml_hrx_op_soft_max(ggml_backend_hrx_context * ctx, const ggml_tensor * node) {
    const ggml_tensor * src  = node->src[0];
    const ggml_tensor * mask = node->src[1];
    if (ggml_nelements(node) == 0) {
        return true;
    }

    uint32_t scale_bits;
    memcpy(&scale_bits, node->op_params, sizeof(uint32_t));

    int64_t ss[GGML_MAX_DIMS];
    int64_t ds[GGML_MAX_DIMS];
    int64_t ms[GGML_MAX_DIMS] = { 0, 0, 0, 0 };
    ggml_hrx_elem_strides(src, ss);
    ggml_hrx_elem_strides(node, ds);

    // Without a mask, the kernel does not read this binding. Bind src instead.
    const ggml_tensor * mask_buf = mask != nullptr ? mask : src;
    int64_t m_ne2  = 1;
    int64_t m_ne3  = 1;
    int64_t m_span = 1;
    const char * mask_type = "0";
    if (mask != nullptr) {
        ggml_hrx_elem_strides(mask, ms);
        m_ne2     = mask->ne[2];
        m_ne3     = mask->ne[3];
        m_span    = (int64_t) (ggml_nbytes(mask) / ggml_type_size(mask->type));
        mask_type = mask->type == GGML_TYPE_F16 ? "2" : "1";
    }

    ggml_hrx_kernel_desc desc;
    desc.file   = "softmax/soft_max_f32.loom";
    desc.symbol = "ggml_soft_max_f32";
    desc.config = { { "ggml.soft_max.mask", mask_type }, { "ggml.norm.wg_size", std::to_string(ctx->device->tune->norm_wg_size) } };
    ggml_hrx_kernel * kernel = ggml_hrx_get_kernel(ctx->device, desc);

    const int64_t rows = ggml_nrows(node);
    return ggml_hrx_dispatch(ctx, kernel, { rows },
        {
            rows, src->ne[0], src->ne[1], src->ne[2],
            ss[1], ss[2], ss[3], (int64_t) (ggml_nbytes(src) / 4),
            m_ne2, m_ne3, ms[1], ms[2], ms[3], m_span,
            ds[1], ds[2], ds[3], (int64_t) (ggml_nbytes(node) / 4),
            (int64_t) scale_bits,
        },
        { src, mask_buf, node });
}
