#pragma once

#include "common.cuh"

// Prefill matmul for K-quant weights (dense Q4_K/Q6_K, MoE Q4_K/Q5_K/Q6_K) on RDNA3/RDNA3.5: fp16 activations, weights dequantized to fp16 in the kernel, fp16 WMMA.
// Off by default. Env:
//   GGML_CUDA_MMDQ           non-zero = enable
//   GGML_CUDA_MMDQ_MIN_BATCH minimum src1 columns (tokens) to use it, default 64 for Q4_K, 32 for Q6_K
//   GGML_CUDA_MMDQ_NO_Q6K    non-zero = keep Q6_K on the old path
//   GGML_CUDA_MMDQ_Q8_0      non-zero = also use it for dense Q8_0 (needs GGML_CUDA_MMDQ), GGML_CUDA_MMDQ_Q8_0_MIN_BATCH default 576, only for >= 512 rows
//   GGML_CUDA_MMDQ_ID        non-zero = also use it for MUL_MAT_ID (MoE experts, Q4_K/Q5_K/Q6_K), default off (slower than mmq)
//   GGML_CUDA_MMDQ_ID_MIN_BATCH minimum number of tokens for MUL_MAT_ID, default 32
//   GGML_CUDA_MMDQ_GLU       non-zero = fuse gate + up + SWIGLU + down when all three use it (f16 hidden state, no f32 gate/up), dense or MoE (needs GGML_CUDA_MMDQ_MOE)
//   GGML_CUDA_MMDQ_FUSE_ADD  non-zero = fuse a following bias add and/or residual add into the dense matmul
//   GGML_CUDA_MMDQ_MOE       non-zero = MUL_MAT_ID (Q4_K/Q5_K/Q6_K/Q5_1) with the weight streaming kernel, independent of GGML_CUDA_MMDQ
//   GGML_CUDA_MMDQ_MOE_MIN_BATCH minimum number of tokens for it, default 16; used only with >= 16 tokens per expert on average,
//                            >= 32 for K-quant rows shorter than 2048, GGML_CUDA_MMDQ_MOE_ALL non-zero = for every shape (tests)

bool ggml_cuda_should_use_mmdq(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc);

// bias: [ne01] added to every column, res: same shape as dst added to dst (may alias dst)
void ggml_cuda_mul_mat_dq(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
    const float * bias = nullptr, const ggml_tensor * res = nullptr);

bool ggml_cuda_should_use_mmdq_id(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, const ggml_tensor * dst, int cc);

void ggml_cuda_mul_mat_id_dq(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst);

// glu: SWIGLU of two MUL_MAT with the same src1, down: MUL_MAT of glu
bool ggml_cuda_should_use_mmdq_glu(const ggml_tensor * glu, const ggml_tensor * down, int cc);

// dst: down, or the add of down and res (dense only)
void ggml_cuda_mul_mat_dq_glu(ggml_backend_cuda_context & ctx, const ggml_tensor * glu, const ggml_tensor * down, ggml_tensor * dst, const ggml_tensor * res = nullptr);
