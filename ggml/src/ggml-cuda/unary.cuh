#pragma once
#include "common.cuh"

#define CUDA_NEG_BLOCK_SIZE 256
#define CUDA_STEP_BLOCK_SIZE 256
#define CUDA_GELU_BLOCK_SIZE 256
#define CUDA_SILU_BLOCK_SIZE 256
#define CUDA_SILU_BACK_BLOCK_SIZE 256
#define CUDA_TANH_BLOCK_SIZE 256
#define CUDA_RELU_BLOCK_SIZE 256
#define CUDA_SIGMOID_BLOCK_SIZE 256
#define CUDA_HARDSIGMOID_BLOCK_SIZE 256
#define CUDA_EXP_BLOCK_SIZE 256
#define CUDA_HARDSWISH_BLOCK_SIZE 256
#define CUDA_SQR_BLOCK_SIZE 256
#define CUDA_SQRT_BLOCK_SIZE 256
#define CUDA_SIN_BLOCK_SIZE 256
#define CUDA_COS_BLOCK_SIZE 256
#define CUDA_GLU_BLOCK_SIZE 256
#define CUDA_XIELU_BLOCK_SIZE 256

void ggml_cuda_op_abs(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_sgn(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_neg(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_step(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_gelu(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_silu(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_silu_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_gelu_erf(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_gelu_quick(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_tanh(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_relu(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_sigmoid(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_hardsigmoid(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_exp(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_hardswish(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_leaky_relu(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_sqr(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_sqrt(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_sin(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_cos(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_log(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_expm1(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_softplus(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_elu(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_floor(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_ceil(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_round(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_trunc(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_reglu(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_geglu(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_swiglu(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_swiglu_oai(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_swiglu_clamp(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_geglu_erf(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_geglu_quick(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_xielu(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// returns whether it also wrote q8_1, MMVQ's q8_1 copy of mul_node (when q8_1 is not nullptr and the layout allows)
bool ggml_cuda_op_unary_mul(ggml_backend_cuda_context & ctx, ggml_tensor * unary_node, ggml_tensor * mul_node,
                            block_q8_1 * q8_1 = nullptr);

// unary(scale(x)); returns whether it also wrote q8_1, MMVQ's q8_1 copy of unary_node with its rows padded to
// MATRIX_ROW_PADDING (when q8_1 is not nullptr)
bool ggml_cuda_op_scale_unary(ggml_backend_cuda_context & ctx, ggml_tensor * scale_node, ggml_tensor * unary_node,
                              block_q8_1 * q8_1 = nullptr);

// add_node = a + x*unary(g), mul_node = x*unary(g) with one value of unary(g) per row of x
void ggml_cuda_op_unary_mul_add(ggml_backend_cuda_context & ctx, ggml_tensor * unary_node, ggml_tensor * mul_node,
                                ggml_tensor * add_node);

// gate = softplus(alpha_mm + dt)*a and beta = sigmoid(beta_mm) for one token, the alpha and beta products of a gated
// delta net layer (F32 weights) included, in one kernel
void ggml_cuda_op_gdn_alpha_beta(ggml_backend_cuda_context & ctx, const ggml_tensor * alpha_mm, const ggml_tensor * dt,
                                 const ggml_tensor * a, ggml_tensor * gate, const ggml_tensor * beta_mm, ggml_tensor * beta);

void ggml_cuda_op_relu_sqr(ggml_backend_cuda_context & ctx, ggml_tensor * relu_node, ggml_tensor * sqr_node);

__device__ __forceinline__ float ggml_cuda_op_silu_single(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float ggml_cuda_op_gelu_single(float x) {
    const float GELU_COEF_A    = 0.044715f;
    const float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;

    return 0.5f * x * (1.0f + tanhf(SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x)));
}

__device__ __forceinline__ float ggml_cuda_op_swiglu_oai_single(float x, float g, float alpha = 1.702f, float limit = 7.0f) {
    x = fminf(x, limit);
    g = fmaxf(fminf(g, limit), -limit);

    float out_glu = x / (1.0f + expf(-x * alpha));
    out_glu = out_glu * (1.0f + g);
    return out_glu;
}

__device__ __forceinline__ float ggml_cuda_op_swiglu_clamp_single(float gate, float up, float limit) {
    gate = fminf(gate, limit);
    up = fmaxf(fminf(up, limit), -limit);

    return ggml_cuda_op_silu_single(gate) * up;
}

__device__ __forceinline__ float ggml_cuda_op_glu_single(ggml_glu_op op, float gate, float up, float limit) {
    switch (op) {
        case GGML_GLU_OP_SWIGLU:
            return up * ggml_cuda_op_silu_single(gate);
        case GGML_GLU_OP_GEGLU:
            return up * ggml_cuda_op_gelu_single(gate);
        case GGML_GLU_OP_SWIGLU_OAI:
            return ggml_cuda_op_swiglu_oai_single(gate, up);
        case GGML_GLU_OP_SWIGLU_CLAMP:
            return ggml_cuda_op_swiglu_clamp_single(gate, up, limit);
        default:
            return up * gate;
    }
}
