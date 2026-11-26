#include "common.cuh"

void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_group_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// q8_1, if not null: also write MMVQ's q8_1 copy of the result there; returns whether it was written
bool ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor,
                                 block_q8_1 * q8_1 = nullptr);

// the gated identity-comb DSV4_HC_POST (see ggml_cuda_op_dsv4_hc_post) of one token followed by RMS_NORM + MUL with
// MMVQ's q8_1 copy, in one kernel
bool ggml_cuda_hc_post_rms_norm_supported(const ggml_tensor * post, const ggml_tensor * gate_scale_in,
                                          const float * post_data, const ggml_tensor * rms, const ggml_tensor * mul);
void ggml_cuda_op_hc_post_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * post,
                                   const ggml_tensor * gate_scale_in, const ggml_tensor * gate_scale_out,
                                   const float * post_data, const ggml_tensor * rms, ggml_tensor * mul,
                                   block_q8_1 * q8_1);

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

void ggml_cuda_op_l2_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
