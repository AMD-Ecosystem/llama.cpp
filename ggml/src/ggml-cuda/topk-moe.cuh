#include "common.cuh"
#include "ggml.h"

#include <initializer_list>

// Rows that one CUDA block handles.
#define TOPK_MOE_ROWS_PER_BLOCK 8

struct ggml_cuda_topk_moe_args {
    bool sigmoid{};
    bool sqrt_softplus{};
    bool softmax{};
    bool delayed_softmax{};
    bool prob_bias{};
    bool norm{};
    bool scale{};
    int  n_groups{};     // > 0: pick experts only from the top n_group_used expert groups
    int  n_group_used{};
};

void ggml_cuda_op_topk_moe(ggml_backend_cuda_context &     ctx,
                           const ggml_tensor *             logits,
                           ggml_tensor *                   weights,
                           ggml_tensor *                   ids,
                           const ggml_tensor *             clamp,
                           const ggml_tensor *             scale,
                           const ggml_tensor *             bias,
                           const ggml_cuda_topk_moe_args & args);

// the F32 router MUL_MAT mm of one token and the top-k MoE selection on its result in one kernel; counter: an int that
// is zero between the launches
bool ggml_cuda_router_topk_moe_supported(const ggml_tensor * mm, const ggml_tensor * logits);
// the BF16 copy of F32 router weights that ggml_cuda_op_router_topk_moe reads when in ctx.router_bf16
void ggml_cuda_router_weights_to_bf16(const float * x, void * dst, const int64_t n, cudaStream_t stream);
void ggml_cuda_op_router_topk_moe(ggml_backend_cuda_context &     ctx,
                                  const ggml_tensor *             mm,
                                  ggml_tensor *                   weights,
                                  ggml_tensor *                   ids,
                                  const ggml_tensor *             clamp,
                                  const ggml_tensor *             scale,
                                  const ggml_tensor *             bias,
                                  const ggml_cuda_topk_moe_args & args,
                                  int *                           counter);

bool ggml_cuda_should_use_topk_moe(const ggml_tensor * gating_op,
                                   const ggml_tensor * weights,
                                   const ggml_tensor * logits,
                                   const ggml_tensor * ids);
