#include "common.cuh"

// opt-in (GGML_CUDA_FATTN_VISION=1): RDNA3 WMMA flash attention for vision encoders (no mask, no GQA, D 64/72/80)
bool ggml_cuda_flash_attn_ext_vision_supported(int cc, const ggml_tensor * dst);

void ggml_cuda_flash_attn_ext_vision(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
