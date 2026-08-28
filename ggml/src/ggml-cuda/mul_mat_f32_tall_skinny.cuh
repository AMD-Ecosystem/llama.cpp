#include "common.cuh"

// f32 tall-skinny GEMM for gfx1151 / RDNA3.5: the weight [K, N] is tall (large K)
// and skinny (small N). Computes ggml_mul_mat(src0=weight[K,N], src1=activation[K,M])
// -> dst[N,M] for K=2048, N=32. The activation is staged resident in LDS while the
// weight is streamed from global memory with a backedge software pipeline; the K
// reduction uses a wave32 DPP tree. AMD-only: on non-HIP builds the predicate returns
// false and this path is never taken.

bool ggml_cuda_should_use_mul_mat_f32_tall_skinny(
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc);

void ggml_cuda_mul_mat_f32_tall_skinny(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
