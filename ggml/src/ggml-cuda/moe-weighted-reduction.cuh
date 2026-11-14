#include "common.cuh"

// dst_q8_1, if not null: also write MMVQ's q8_1 copy of dst there; returns whether it was written
bool ggml_cuda_op_moe_weighted_reduction(ggml_backend_cuda_context & ctx,
                                         const ggml_tensor *         experts,
                                         const ggml_tensor *         expert_scale,
                                         const ggml_tensor *         weights,
                                         ggml_tensor *               dst,
                                         block_q8_1 *                dst_q8_1 = nullptr);
