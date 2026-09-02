// RDNA3.5 (gfx1151) MMQ config overlay.
//
// This file holds ONLY the rocmfp4 CASE rows. rocmfp4 (Q4_0_ROCMFP4 dual-scale and
// _FAST single-scale) is an AMD-only, gfx1151-targeted quantization, so its MMQ config
// lives here rather than polluting the shared rdna4 table. Every other type falls
// through to ggml_cuda_mmq_get_config_rdna4() below.
//
// The dual type mirrors NVFP4 (SRAM layout NVFP4, Q8_0_16 vec_dot); the fast type mirrors
// MXFP4 (SRAM layout Q8_1). nthreads/I here are placeholders — ggml_cuda_mmq_get_config_rdna35
// in mmq.cuh force-overrides them to 128/64 for the fork's RDNA3.5 kernels.
static constexpr __host__ __device__ ggml_cuda_mmq_config ggml_cuda_mmq_get_config_rdna3_5(ggml_type type, int J, bool fallback) {
    // Q4_0_ROCMFP4 (dual-scale) — modeled on NVFP4.
    CASE(GGML_TYPE_Q4_0_ROCMFP4, 128, 2,  64,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_NVFP4, MMQ_ITER_K, false, true);
    CASE(GGML_TYPE_Q4_0_ROCMFP4, 128, 2,  64,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_NVFP4, MMQ_ITER_K, false, true);
    CASE(GGML_TYPE_Q4_0_ROCMFP4, 128, 2,  64,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_NVFP4, MMQ_ITER_K, false, true);
    CASE(GGML_TYPE_Q4_0_ROCMFP4, 256, 2, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_NVFP4, MMQ_ITER_K, false, true);
    CASE(GGML_TYPE_Q4_0_ROCMFP4, 128, 2,  64,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_NVFP4, MMQ_ITER_K, false, false);
    CASE(GGML_TYPE_Q4_0_ROCMFP4, 128, 2,  64,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_NVFP4, MMQ_ITER_K, false, false);
    CASE(GGML_TYPE_Q4_0_ROCMFP4, 128, 2,  64,  48, GGML_CUDA_MMQ_SRAM_LAYOUT_NVFP4, MMQ_ITER_K, false, false);
    CASE(GGML_TYPE_Q4_0_ROCMFP4, 128, 2,  64,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_NVFP4, MMQ_ITER_K, false, false);
    CASE(GGML_TYPE_Q4_0_ROCMFP4, 256, 2, 128,  80, GGML_CUDA_MMQ_SRAM_LAYOUT_NVFP4, MMQ_ITER_K, false, false);
    CASE(GGML_TYPE_Q4_0_ROCMFP4, 256, 2, 128,  96, GGML_CUDA_MMQ_SRAM_LAYOUT_NVFP4, MMQ_ITER_K, false, false);
    CASE(GGML_TYPE_Q4_0_ROCMFP4, 256, 2, 128, 112, GGML_CUDA_MMQ_SRAM_LAYOUT_NVFP4, MMQ_ITER_K, false, false);
    CASE(GGML_TYPE_Q4_0_ROCMFP4, 256, 2, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_NVFP4, MMQ_ITER_K, false, false);

    // Q4_0_ROCMFP4_FAST (single-scale) — modeled on MXFP4.
    CASE(GGML_TYPE_Q4_0_ROCMFP4_FAST, 128, 2,  64,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, false, true);
    CASE(GGML_TYPE_Q4_0_ROCMFP4_FAST, 128, 2,  64,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, false, true);
    CASE(GGML_TYPE_Q4_0_ROCMFP4_FAST, 128, 2,  64,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, false, true);
    CASE(GGML_TYPE_Q4_0_ROCMFP4_FAST, 256, 2, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, false, true);
    CASE(GGML_TYPE_Q4_0_ROCMFP4_FAST, 128, 2,  64,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, false, false);
    CASE(GGML_TYPE_Q4_0_ROCMFP4_FAST, 128, 2,  64,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, false, false);
    CASE(GGML_TYPE_Q4_0_ROCMFP4_FAST, 128, 2,  64,  48, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, false, false);
    CASE(GGML_TYPE_Q4_0_ROCMFP4_FAST, 128, 2,  64,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, false, false);
    CASE(GGML_TYPE_Q4_0_ROCMFP4_FAST, 256, 2, 128,  80, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, false, false);
    CASE(GGML_TYPE_Q4_0_ROCMFP4_FAST, 256, 2, 128,  96, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, false, false);
    CASE(GGML_TYPE_Q4_0_ROCMFP4_FAST, 256, 2, 128, 112, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, false, false);
    CASE(GGML_TYPE_Q4_0_ROCMFP4_FAST, 256, 2, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, false, false);

    return ggml_cuda_mmq_get_config_rdna4(type, J, fallback);
}
