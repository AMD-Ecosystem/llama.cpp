#pragma once

#include "common.cuh"

// Dequantize-to-float matvec for K-quant weights (decode). Skips the q8_1
// activation quantization pass that mul_mat_vec_q requires.
// On by default where the arch default applies (RDNA3.5); env override:
// GGML_CUDA_DQ_MMV  (unset = arch default, 0 = force off, non-zero = force on)

// Returns true when mul_mat_vec_q should route this op to the dq kernels instead
// of the q8_1 path. Encapsulates arch/env enable, type/shape guards, and fusion
// inspection: all gated GLU variants (SwiGLU/GEGLU/OAI/CLAMP) plus x/gate bias are
// accepted; only scale (NVFP4) falls back to the q8_1 mmvq path. MoE (ids != nullptr) is supported
// for both the plain per-expert matvec and the fused gate+up SwiGLU at a single decode
// token: src0 is [K, N, n_expert], the expert axis is selected via ids, and the
// activation may be broadcast (src1->ne[1]==1, gate/up) or per-expert
// (src1->ne[1]==dst->ne[1], ffn_down), selected in-kernel via channel_y. Multi-token MoE
// (dst->ne[2]>1) is still deferred. In the non-ids path src1->ne[1] must be 1; >1 is the
// reserved MTP (token-batch) seam.
bool ggml_cuda_should_use_mmv_dq(
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
    const ggml_tensor * dst, int cc, const ggml_cuda_mm_fusion_args_host * fusion);

// Single dq entry point, signature mirroring ggml_cuda_mul_mat_vec_q. Dispatches
// to the plain or fused-SwiGLU dq kernels based on fusion->gate. Callers must have
// checked ggml_cuda_should_use_mmv_dq first.
void ggml_cuda_mul_mat_vec_dq(
    ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
    const ggml_tensor * ids, ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion);
