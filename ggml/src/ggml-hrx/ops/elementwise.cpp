#include "ggml-hrx-ops.h"

#include <climits>
#include <cstring>
#include <string>

// Must match the op codes in elementwise/unary_f32.loom.
enum ggml_hrx_unary_code {
    GGML_HRX_UNARY_SCALE    = 0,
    GGML_HRX_UNARY_CLAMP    = 1,
    GGML_HRX_UNARY_NEG      = 2,
    GGML_HRX_UNARY_ABS      = 3,
    GGML_HRX_UNARY_RELU     = 4,
    GGML_HRX_UNARY_SQR      = 5,
    GGML_HRX_UNARY_EXP      = 6,
    GGML_HRX_UNARY_SIGMOID  = 7,
    GGML_HRX_UNARY_SILU     = 8,
    GGML_HRX_UNARY_SOFTPLUS = 9,
};

static int ggml_hrx_unary_code(ggml_unary_op op) {
    switch (op) {
        case GGML_UNARY_OP_NEG:      return GGML_HRX_UNARY_NEG;
        case GGML_UNARY_OP_ABS:      return GGML_HRX_UNARY_ABS;
        case GGML_UNARY_OP_RELU:     return GGML_HRX_UNARY_RELU;
        case GGML_UNARY_OP_EXP:      return GGML_HRX_UNARY_EXP;
        case GGML_UNARY_OP_SIGMOID:  return GGML_HRX_UNARY_SIGMOID;
        case GGML_UNARY_OP_SILU:     return GGML_HRX_UNARY_SILU;
        case GGML_UNARY_OP_SOFTPLUS: return GGML_HRX_UNARY_SOFTPLUS;
        default:                     return -1;
    }
}

// Float params go as bits in a launch scalar. Thus a new value does not cause a recompile.
static int64_t ggml_hrx_f32_bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    return u;
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

// The last workgroup can go 4092 elements past n (1024 lanes). All offsets must stay below 2^31 - 8.
static bool ggml_hrx_row_stride(const ggml_tensor * t, int64_t * rs) {
    int64_t s[GGML_MAX_DIMS];
    if (!ggml_hrx_elem_strides(t, s) || s[0] != 1 || ggml_nbytes(t) / 4 > (size_t) INT_MAX - 4096) {
        return false;
    }
    *rs = 0;
    int64_t next = -1;
    for (int i = 1; i < GGML_MAX_DIMS; ++i) {
        if (t->ne[i] == 1) {
            continue;
        }
        if (next >= 0 && s[i] != next) {
            return false;
        }
        if (next < 0) {
            *rs = s[i];
        }
        next = s[i] * t->ne[i];
    }
    return *rs < INT_MAX;
}

static bool ggml_hrx_supports_unary_f32(const ggml_tensor * op) {
    const ggml_tensor * src = op->src[0];
    if (src == nullptr || src->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32 || !ggml_are_same_shape(src, op)) {
        return false;
    }
    int64_t s[GGML_MAX_DIMS];
    if (!ggml_hrx_launch_strides(src, s) || !ggml_hrx_launch_strides(op, s)) {
        return false;
    }
    return ggml_hrx_fits_i32(src) && ggml_hrx_fits_i32(op);
}

static bool ggml_hrx_dispatch_unary_f32(ggml_backend_hrx_context * ctx, const ggml_tensor * node, int code, float p0, float p1) {
    const ggml_tensor * src = node->src[0];
    const int64_t n = ggml_nelements(node);
    if (n == 0) {
        return true;
    }

    int64_t s_rs;
    int64_t t_rs;
    if (node->ne[0] % 4 == 0 && ggml_hrx_row_stride(src, &s_rs) && ggml_hrx_row_stride(node, &t_rs)) {
        ggml_hrx_kernel_desc desc;
        desc.file   = "elementwise/unary_f32.loom";
        desc.symbol = "ggml_unary_f32_rows";
        desc.config = { { "ggml.unary.op", std::to_string(code) } };
        desc.config.push_back(ggml_hrx_elementwise_wg_config(ctx->device));
        ggml_hrx_kernel * kernel = ggml_hrx_get_kernel(ctx->device, desc);

        return ggml_hrx_dispatch(ctx, kernel, { n },
            {
                n, node->ne[0], s_rs, (int64_t) (ggml_nbytes(src) / 4), t_rs, (int64_t) (ggml_nbytes(node) / 4),
                ggml_hrx_f32_bits(p0), ggml_hrx_f32_bits(p1),
            },
            { src, node });
    }

    int64_t ss[GGML_MAX_DIMS];
    int64_t ts[GGML_MAX_DIMS];
    ggml_hrx_launch_strides(src, ss);
    ggml_hrx_launch_strides(node, ts);

    ggml_hrx_kernel_desc desc;
    desc.file   = "elementwise/unary_f32.loom";
    desc.symbol = "ggml_unary_f32";
    desc.config = { { "ggml.unary.op", std::to_string(code) } };
    desc.config.push_back(ggml_hrx_elementwise_wg_config(ctx->device));
    ggml_hrx_kernel * kernel = ggml_hrx_get_kernel(ctx->device, desc);

    return ggml_hrx_dispatch(ctx, kernel, { n },
        {
            n, node->ne[0], node->ne[1], node->ne[2],
            ss[0], ss[1], ss[2], ss[3], (int64_t) (ggml_nbytes(src) / 4),
            ts[0], ts[1], ts[2], ts[3], (int64_t) (ggml_nbytes(node) / 4),
            ggml_hrx_f32_bits(p0), ggml_hrx_f32_bits(p1),
        },
        { src, node });
}

bool ggml_hrx_supports_scale(const ggml_hrx_device_context * dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);
    return ggml_hrx_supports_unary_f32(op);
}

bool ggml_hrx_op_scale(ggml_backend_hrx_context * ctx, const ggml_tensor * node) {
    return ggml_hrx_dispatch_unary_f32(ctx, node, GGML_HRX_UNARY_SCALE,
        ggml_get_op_params_f32(node, 0), ggml_get_op_params_f32(node, 1));
}

bool ggml_hrx_supports_clamp(const ggml_hrx_device_context * dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);
    return ggml_hrx_supports_unary_f32(op);
}

bool ggml_hrx_op_clamp(ggml_backend_hrx_context * ctx, const ggml_tensor * node) {
    return ggml_hrx_dispatch_unary_f32(ctx, node, GGML_HRX_UNARY_CLAMP,
        ggml_get_op_params_f32(node, 0), ggml_get_op_params_f32(node, 1));
}

bool ggml_hrx_supports_unary(const ggml_hrx_device_context * dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);
    return ggml_hrx_unary_code(ggml_get_unary_op(op)) >= 0 && ggml_hrx_supports_unary_f32(op);
}

bool ggml_hrx_op_unary(ggml_backend_hrx_context * ctx, const ggml_tensor * node) {
    return ggml_hrx_dispatch_unary_f32(ctx, node, ggml_hrx_unary_code(ggml_get_unary_op(node)), 0.0f, 0.0f);
}

bool ggml_hrx_supports_glu(const ggml_hrx_device_context * dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);
    const ggml_tensor * a = op->src[0];
    const ggml_tensor * b = op->src[1];
    if (ggml_get_glu_op(op) != GGML_GLU_OP_SWIGLU || a == nullptr) {
        return false;
    }
    if (op->type != GGML_TYPE_F32 || a->type != GGML_TYPE_F32 || (b != nullptr && b->type != GGML_TYPE_F32)) {
        return false;
    }
    if (!ggml_is_contiguous(op)) {
        return false;
    }
    if (b != nullptr) {
        if (!ggml_are_same_shape(a, op) || !ggml_are_same_shape(b, op)) {
            return false;
        }
    } else if ((a->ne[0] != 2 * op->ne[0] && a->ne[0] != 2 * op->ne[0] + 1) || a->ne[1] != op->ne[1] ||
               a->ne[2] != op->ne[2] || a->ne[3] != op->ne[3]) {
        // With an odd row length, the last element is not used. The CPU backend does the same.
        return false;
    }
    int64_t s[GGML_MAX_DIMS];
    if (!ggml_hrx_launch_strides(a, s) || (b != nullptr && !ggml_hrx_launch_strides(b, s))) {
        return false;
    }
    return ggml_hrx_fits_i32(op) && ggml_hrx_fits_i32(a) && (b == nullptr || ggml_hrx_fits_i32(b));
}

bool ggml_hrx_op_glu(ggml_backend_hrx_context * ctx, const ggml_tensor * node) {
    const ggml_tensor * a = node->src[0];
    const ggml_tensor * b = node->src[1];

    int64_t as[GGML_MAX_DIMS];
    int64_t bs[GGML_MAX_DIMS];
    ggml_hrx_launch_strides(a, as);
    int64_t a_off = 0;
    int64_t b_off = 0;
    if (b != nullptr) {
        // The split form ignores swap. src0 is always the activated input. The CPU backend does the same.
        ggml_hrx_launch_strides(b, bs);
    } else {
        b = a;
        std::memcpy(bs, as, sizeof(bs));
        const int64_t half = node->ne[0] * (int64_t) (a->nb[0] / sizeof(float));
        if (ggml_get_op_params_i32(node, 1) != 0) {
            a_off = half;
        } else {
            b_off = half;
        }
    }

    const int64_t n = ggml_nelements(node);
    if (n == 0) {
        return true;
    }

    int64_t a_rs;
    int64_t b_rs;
    if (node->ne[0] % 4 == 0 && ggml_hrx_row_stride(a, &a_rs) && ggml_hrx_row_stride(b, &b_rs)) {
        ggml_hrx_kernel_desc desc;
        desc.file   = "elementwise/swiglu_f32.loom";
        desc.symbol = "ggml_swiglu_f32_rows";
        desc.config.push_back(ggml_hrx_elementwise_wg_config(ctx->device));
        ggml_hrx_kernel * kernel = ggml_hrx_get_kernel(ctx->device, desc);

        return ggml_hrx_dispatch(ctx, kernel, { n },
            {
                n, node->ne[0],
                a_off, a_rs, (int64_t) (ggml_nbytes(a) / 4),
                b_off, b_rs, (int64_t) (ggml_nbytes(b) / 4),
            },
            { a, b, node });
    }

    ggml_hrx_kernel_desc desc;
    desc.file   = "elementwise/swiglu_f32.loom";
    desc.symbol = "ggml_swiglu_f32";
    desc.config.push_back(ggml_hrx_elementwise_wg_config(ctx->device));
    ggml_hrx_kernel * kernel = ggml_hrx_get_kernel(ctx->device, desc);

    return ggml_hrx_dispatch(ctx, kernel, { n },
        {
            n, node->ne[0], node->ne[1], node->ne[2],
            a_off, as[0], as[1], as[2], as[3], (int64_t) (ggml_nbytes(a) / 4),
            b_off, bs[0], bs[1], bs[2], bs[3], (int64_t) (ggml_nbytes(b) / 4),
        },
        { a, b, node });
}

bool ggml_hrx_supports_arange(const ggml_hrx_device_context * dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);
    return op->type == GGML_TYPE_F32 && ggml_is_contiguous(op) && ggml_hrx_fits_i32(op);
}

bool ggml_hrx_op_arange(ggml_backend_hrx_context * ctx, const ggml_tensor * node) {
    ggml_hrx_kernel_desc desc;
    desc.file   = "elementwise/arange_f32.loom";
    desc.symbol = "ggml_arange_f32";
    desc.config.push_back(ggml_hrx_elementwise_wg_config(ctx->device));
    ggml_hrx_kernel * kernel = ggml_hrx_get_kernel(ctx->device, desc);

    const int64_t n = ggml_nelements(node);
    return ggml_hrx_dispatch(ctx, kernel, { n },
        { n, ggml_hrx_f32_bits(ggml_get_op_params_f32(node, 0)), ggml_hrx_f32_bits(ggml_get_op_params_f32(node, 2)) },
        { node });
}
