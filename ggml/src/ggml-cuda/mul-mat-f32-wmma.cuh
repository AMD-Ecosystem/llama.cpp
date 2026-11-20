#include "common.cuh"

// F32 x F32 -> F32 matrix product of a few hundred outputs for a prompt batch (the router of a MoE layer) on RDNA3 WMMA
bool ggml_cuda_should_use_mul_mat_f32_wmma(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc);

void ggml_cuda_mul_mat_f32_wmma(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
