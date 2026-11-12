#pragma once

#include "common.cuh"

// opt-in (GGML_CUDA_FATTN_VISION=1): RDNA3.5 WMMA flash attention for vision encoders (no mask, no GQA, D 64/72/80)
bool ggml_cuda_flash_attn_ext_vision_supported(int cc, const ggml_tensor * dst);

void ggml_cuda_flash_attn_ext_vision(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// opt-in (GGML_CUDA_FATTN_PREFILL=1): same kernel with an f16 mask and GQA for LLM prefill (D 64/128/256)
bool ggml_cuda_flash_attn_ext_prefill_supported(int cc, const ggml_tensor * dst);

void ggml_cuda_flash_attn_ext_prefill(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

