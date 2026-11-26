#include "common.cuh"

void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_group_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// q8_1, if not null: also write MMVQ's q8_1 copy of the result there; returns whether it was written
bool ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor,
                                 block_q8_1 * q8_1 = nullptr);

void ggml_cuda_op_rms_norm_scale_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * scale_tensor);

void ggml_cuda_op_rms_norm_fused_add(ggml_backend_cuda_context & ctx,
                                     ggml_tensor *               dst,
                                     ggml_tensor *               mul_tensor,
                                     ggml_tensor *               add_tensor);

// q8_1, if not null: also write MMVQ's q8_1 copy of mul2 there; returns whether it was written
bool ggml_cuda_op_rms_norm_fused_add_rms(ggml_backend_cuda_context & ctx, ggml_tensor * rms1, ggml_tensor * mul1,
                                         ggml_tensor * add, ggml_tensor * rms2, ggml_tensor * mul2,
                                         block_q8_1 * q8_1 = nullptr);

void ggml_cuda_op_rms_norm_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// RMSNorm+mul of [2560, 4, T] and the 4-row F32 inject that reads it, one pass over the normalized values.
void ggml_cuda_op_rms_norm_hc_inject(ggml_backend_cuda_context & ctx,
        const ggml_tensor * rms, const ggml_tensor * mul, const ggml_tensor * mm, const ggml_tensor * scale);

void ggml_cuda_op_l2_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
