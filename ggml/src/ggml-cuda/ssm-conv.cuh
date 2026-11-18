#include "common.cuh"

// fuse_concat: dst->src[0] is a dim 0 CONCAT that was not computed, the kernel reads its two inputs
void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node = nullptr, ggml_tensor * silu_dst = nullptr,
                           bool fuse_concat = false);
