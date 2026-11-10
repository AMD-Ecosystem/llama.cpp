# RDNA3.5 optimization ablation refresh

Fresh measurements collected November 9-10, 2026. Final measured report.

This report measures the optimization branch against ROCm llama.cpp `gfx11`, then switches individual optimizations off and on within the same branch. All latency numbers come from this experiment. Previous reports supplied no timing results. Existing input fixtures and model configurations were frozen for reproducibility.

Coverage: 27/27 full-branch comparisons; 136/136 model/optimization comparisons across 64 optimization or tuning directions. Each completed comparison has two process starts and ten measured samples per side.

Extra-warmup confirmations: 4/4.

## What the numbers mean

A full-branch comparison answers how much faster this tuned branch is than current gfx11 under the same workload settings. A within-branch ablation answers what changes when one control is disabled while the other listed settings remain fixed. Current gfx11 also has upstream changes after our historical fork, so these are different measurements.

Positive saved time means enabling the optimization was faster. Saved % = 100 * (off - on) / off. Speedup = off / on. Gains from different rows must not be added: parent fusions contain sub-fusions, and the newer MMDQ and attention paths can bypass older optimizations.

The combined branch column uses explicit opt-in flags from [profiles.json](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/profiles.json). It is not a claim about launching the binary with every environment variable unset. Existing source defaults are preserved by the new controls.

## Full branch versus current gfx11

### LLM prefill

| Model | Input positions | gfx11 ms | Optimized ms | Saved | Speedup | Consistency |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-0.5B LLM | 128 | 11.99 | 9.20 | 23.3% | 1.30x | gain |
| Qwen2.5-0.5B LLM | 3968 | 272.44 | 136.75 | 49.8% | 1.99x | gain |
| Qwen2.5-3B LLM | 128 | 41.78 | 31.90 | 23.6% | 1.31x | gain |
| Qwen2.5-3B LLM | 3968 | 1,125.94 | 808.80 | 28.2% | 1.39x | gain |
| Qwen2.5-7B LLM | 128 | 78.78 | 57.77 | 26.7% | 1.36x | gain |
| Qwen2.5-7B LLM | 3968 | 2,249.50 | 1,629.30 | 27.6% | 1.38x | gain |
| Qwen3-4B LLM | 128 | 54.07 | 41.04 | 24.1% | 1.32x | gain |
| Qwen3-4B LLM | 3968 | 1,556.17 | 1,148.86 | 26.2% | 1.35x | gain |
| Qwen3-8B LLM | 128 | 92.27 | 71.07 | 23.0% | 1.30x | gain |
| Qwen3-8B LLM | 3968 | 2,622.06 | 2,021.97 | 22.9% | 1.30x | gain |
| Llama-2-7B LLM | 128 | 85.78 | 60.99 | 28.9% | 1.41x | gain |
| Llama-2-7B LLM | 3968 | 2,782.93 | 1,907.03 | 31.5% | 1.46x | gain |
| Gemma-2B LLM | 128 | 30.36 | 25.72 | 15.3% | 1.18x | gain |
| Gemma-2B LLM | 3968 | 855.62 | 603.73 | 29.4% | 1.42x | gain |
| Gemma-3-12B LLM | 128 | 130.77 | 105.28 | 19.5% | 1.24x | gain |
| Gemma-3-12B LLM | 3968 | 4,038.98 | 2,919.04 | 27.7% | 1.38x | gain |
| Qwen3.6-27B LLM | 128 | 298.81 | 237.47 | 20.5% | 1.26x | gain |
| Qwen3.6-27B LLM | 3968 | 8,616.15 | 6,971.95 | 19.1% | 1.24x | gain |


Geometric mean speedup for LLM 128: **1.296x** across 9 completed models. This is the representative test set, not an all-model or vLLM comparison.

Geometric mean speedup for LLM 3968: **1.422x** across 9 completed models. This is the representative test set, not an all-model or vLLM comparison.

### VLM HTTP TTFT

| Model | Input positions | gfx11 ms | Optimized ms | Saved | Speedup | Consistency |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-VL-3B | 1606 | 2,192.05 | 667.01 | 69.6% | 3.29x | gain |
| Qwen2.5-VL-7B | 1606 | 2,661.54 | 1,059.46 | 60.2% | 2.51x | gain |
| Qwen3-VL-4B VLM | 1322 | 1,060.21 | 682.83 | 35.6% | 1.55x | gain |
| Gemma-3-4B VLM | 779 | 1,088.70 | 507.38 | 53.4% | 2.15x | gain |
| Gemma-3-12B VLM | 367 | 1,209.82 | 602.83 | 50.2% | 2.01x | gain |
| Gemma-4-26B-A4B VLM | 779 | 833.63 | 594.06 | 28.7% | 1.40x | gain |
| Qwen3.5-35B-A3B VLM | 414 | 408.21 | 345.11 | 15.5% | 1.18x | gain |
| Gemma-4-E2B (BF16 projector) VLM | 775 | 344.72 | 259.66 | 24.7% | 1.33x | gain |
| Qwen3-Omni-30B-A3B (BF16 projector) VLM | 1322 | 1,262.39 | 766.10 | 39.3% | 1.65x | gain |


Geometric mean speedup for VLM: **1.802x** across 9 completed models. This is the representative test set, not an all-model or vLLM comparison.

## Isolated gains worth noticing

| Optimization | Representative workload | Off ms | On ms | Saved ms | Saved % |
| --- | --- | --- | --- | --- | --- |
| Vision graph: Batch equal-sized vision windows | Qwen2.5-VL-3B | 1,884.79 | 676.61 | 1,208.18 | 64.1% |
| Vision graph: Batch equal-sized vision windows | Qwen2.5-VL-7B | 2,286.80 | 1,072.09 | 1,214.70 | 53.1% |
| Attention: Pad D72 to the existing D80 WMMA tile | Gemma-3-4B VLM | 900.41 | 529.34 | 371.07 | 41.2% |
| Attention: Pad D72 to the existing D80 WMMA tile | Gemma-3-12B VLM | 986.78 | 616.51 | 370.27 | 37.5% |
| GEMM: Q5_0 MMDQ support | Qwen2.5-0.5B LLM (3968 tokens) | 218.21 | 139.18 | 79.03 | 36.2% |
| Attention: Dedicated prefill attention and mask classification | Qwen2.5-0.5B LLM (3968 tokens) | 189.04 | 143.57 | 45.47 | 24.1% |
| Attention: Enable non-GQA D64/D80 WMMA | Qwen3-VL-4B VLM | 907.85 | 732.95 | 174.90 | 19.3% |
| GEMM: MMDQ gate/up, activation and down fusion | Qwen2.5-3B LLM (3968 tokens) | 978.11 | 821.15 | 156.96 | 16.0% |
| Attention: Enable non-GQA D64/D80 WMMA | Qwen2.5-VL-3B | 792.24 | 678.24 | 114.00 | 14.4% |
| GEMM: Extend MMDQ fusion to GEGLU | Gemma-3-12B LLM (3968 tokens) | 3,606.33 | 3,101.84 | 504.49 | 14.0% |
| Fusion and traffic: FP16 BLAS producer and output fusions | Qwen2.5-VL-3B | 789.83 | 679.63 | 110.21 | 14.0% |
| GEMM: Extend MMDQ fusion to GEGLU | Gemma-2B LLM (3968 tokens) | 709.73 | 614.18 | 95.55 | 13.5% |


These are isolated contributions in the contexts documented below, not additive pieces of the full-branch total. The complete tables include neutral and slower outcomes as well.

## Slower or unresolved results

| Optimization | Workload | Off ms | On ms | Saved % | Primary result | Extra-warmup confirmation |
| --- | --- | --- | --- | --- | --- | --- |
| mmdq_swizzle | Qwen3.6-27B LLM (3968 tokens) | 7,385.55 | 8,965.21 | -21.39% | slower | -20.39%; slower |
| rms_vector | Gemma-3-12B LLM (3968 tokens) | 3,037.63 | 3,219.36 | -5.98% | within variation | -2.24%; within variation |
| mmdq_q5_k | Qwen3.6-27B LLM (3968 tokens) | 8,381.66 | 8,598.47 | -2.59% | within variation | +6.89%; within variation |
| gdn_cu | Qwen3.6-27B LLM (3968 tokens) | 8,802.64 | 8,901.17 | -1.12% | within variation | - |


A negative number is retained even if it disagrees with an earlier expectation. Large variable cases receive the separately reported extra-warmup confirmation. When that confirmation does not reproduce a primary slowdown, the primary result alone does not establish a steady-state regression. Values below the consistency threshold do not establish a regression or a gain.

For the tested Qwen3.6-27B 3968-token workload, prefer `GGML_CUDA_MMDQ_SWIZZLE=0`: the extra-warmup comparison also found swizzle=2 slower (7,351.72 vs 8,850.52 ms). The full-branch table retains its originally frozen profile, including swizzle=2, so its numbers are not silently replaced by this new recommendation.

Swizzling changes the order of matrix tiles, not their shapes, precision or arithmetic count. It trades activation reuse against weight reuse. This result shows that the tradeoff is unfavorable for this tested workload; no cache-miss counters were collected here to establish the exact hardware cause.

## Controls and how to use them

Runtime controls follow Jim's environment-variable style. Start a new process after changing a value because settings are cached. The added `GGML_CUDA_OPT_ENABLED(name, default_value)` macro gives automatic backend optimizations the same on/off behavior. Explicit `0` selects the fallback and `1` permits the optimized path; hardware, tensor shape, layout and graph-consumer checks still apply. Host graph controls use the equivalent cached environment check.

Three choices are compile-time macros because they affect compiled tile storage or kernel code: `GGML_CUDA_MMQ_Q6_K_SCALE_REUSE`, `GGML_CUDA_MMQ_Q8_0_J112_BARRIERS`, and `GGML_CUDA_FATTN_D80_FULL_TILE`. All default to 1. Their off binaries were built from the same source with the corresponding `-D...=0`. Despite its name, `GGML_CUDA_MMQ_Q8_0_J112_BARRIERS=1` means redundant-barrier removal is enabled.

Numeric tuning controls are exceptions to Boolean 0/1: image threads compare 1 with 8; D256 query groups compare 4 with 8; MMDQ swizzle compares 0 with 2; RoPE compares the stated block settings. `GGML_CUDA_MMDQ_NO_Q6K` is inverse: 0 enables Q6_K MMDQ and 1 disables it.

The additional umbrella `GGML_CUDA_FUSE_CUBLAS_16BIT` defaults to 1 and disables the whole 16-bit BLAS producer-fusion framework at 0. Its FP16/BF16 and subpattern gates are measured below; the umbrella is not another independent gain.

For example, `GGML_CUDA_MMDQ=1 GGML_CUDA_MMDQ_GLU=0 llama-bench ...` selects MMDQ without its FFN fusion; changing GLU to 1 permits that fusion. Keep the model, batch, and all other flags identical. The supplied suite sets the full profile and prerequisites automatically.

## Optimization index

This index summarizes all directions, including those with no established gain. Counts refer to tested model/workload pairs. The gain range includes only rows classified as gains; it is not a prediction for untested models. Follow the detailed section for exact controls and prerequisites.

| Direction | Family | Completed pairs | Measured saving range | Gain | Within variation | Slower |
| --- | --- | --- | --- | --- | --- | --- |
| mmdq | GEMM | 4 | 3.5% to 6.9% | 3 | 1 | 0 |
| mmdq_glu | GEMM | 2 | 8.5% to 16.0% | 2 | 0 | 0 |
| mmdq_geglu | GEMM | 2 | 13.5% to 14.0% | 2 | 0 | 0 |
| mmdq_norm | GEMM | 3 | - | 0 | 3 | 0 |
| mmdq_bias | GEMM | 2 | 1.1% to 1.9% | 2 | 0 | 0 |
| mmdq_share | GEMM | 2 | 1.1% | 1 | 1 | 0 |
| mmdq_q5_0 | GEMM | 1 | 36.2% | 1 | 0 | 0 |
| mmdq_q5_k | GEMM | 1 | - | 0 | 1 | 0 |
| mmdq_q8_0 | GEMM | 4 | 3.1% | 1 | 3 | 0 |
| mmdq_moe | GEMM | 3 | 4.0% to 6.6% | 2 | 1 | 0 |
| mmdq_swizzle | GEMM | 2 | - | 0 | 1 | 1 |
| causal_attention | Attention | 4 | 2.4% to 24.1% | 4 | 0 | 0 |
| d256_sxh | Attention | 1 | 1.4% | 1 | 0 | 0 |
| mask_prefix | Attention | 2 | 2.9% to 3.5% | 2 | 0 | 0 |
| gqa4_wide | Attention | 2 | - | 0 | 2 | 0 |
| gqa8_vgpr | Attention | 1 | - | 0 | 1 | 0 |
| d512_wmma | Attention | 1 | - | 0 | 1 | 0 |
| vision_attention | Attention | 3 | 1.1% to 9.7% | 3 | 0 | 0 |
| vision_wmma | Attention | 2 | 14.4% to 19.3% | 2 | 0 | 0 |
| d72_wmma | Attention | 2 | 37.5% to 41.2% | 2 | 0 | 0 |
| window_batch | Vision graph | 2 | 53.1% to 64.1% | 2 | 0 | 0 |
| window_cast | Vision graph | 2 | - | 0 | 2 | 0 |
| vision_qkv | Vision graph | 3 | - | 0 | 3 | 0 |
| vision_cast | Vision graph | 2 | 3.7% | 1 | 1 | 0 |
| attn_set | Fusion and traffic | 2 | - | 0 | 2 | 0 |
| fp16_fusions | Fusion and traffic | 2 | 10.2% to 14.0% | 2 | 0 | 0 |
| bf16_fusions | Fusion and traffic | 4 | 7.1% to 9.9% | 2 | 2 | 0 |
| ffn_fusions | Fusion and traffic | 4 | 4.2% to 6.4% | 4 | 0 | 0 |
| ffn_norm | Fusion and traffic | 2 | 1.2% | 1 | 1 | 0 |
| bias_fusion | Fusion and traffic | 2 | 6.5% to 6.9% | 2 | 0 | 0 |
| residual_fusion | Fusion and traffic | 2 | 1.1% to 1.1% | 2 | 0 | 0 |
| clamp_fusion | Fusion and traffic | 2 | 9.5% to 9.8% | 2 | 0 | 0 |
| layernorm_affine | Fusion and traffic | 2 | 1.0% to 2.1% | 2 | 0 | 0 |
| rms_post_scale | Fusion and traffic | 1 | - | 0 | 1 | 0 |
| rms_scale | Fusion and traffic | 2 | - | 0 | 2 | 0 |
| gelu_mul | Fusion and traffic | 2 | - | 0 | 2 | 0 |
| norm_pack | Fusion and traffic | 2 | 1.2% | 1 | 1 | 0 |
| glu_pack | Fusion and traffic | 2 | 1.1% to 2.4% | 2 | 0 | 0 |
| glu_16bit | Fusion and traffic | 2 | 2.2% | 1 | 1 | 0 |
| q8_interleave | Fusion and traffic | 2 | 1.3% | 1 | 1 | 0 |
| copy_vec8 | Fusion and traffic | 2 | 1.2% to 1.6% | 2 | 0 | 0 |
| rms_vector | Normalization and RoPE | 3 | 1.3% | 1 | 2 | 0 |
| norm_small | Normalization and RoPE | 2 | - | 0 | 2 | 0 |
| norm_rope_small | Normalization and RoPE | 1 | - | 0 | 1 | 0 |
| norm_rope_warp | Normalization and RoPE | 2 | 1.3% | 1 | 1 | 0 |
| norm_rope_2axis | Normalization and RoPE | 1 | 1.7% | 1 | 0 | 0 |
| rope_blocks | Normalization and RoPE | 3 | 2.1% | 1 | 2 | 0 |
| bias_vector | Fusion and traffic | 2 | - | 0 | 2 | 0 |
| gelu_vector | Fusion and traffic | 2 | - | 0 | 2 | 0 |
| moe_reduce | MoE and graph | 2 | 1.4% | 1 | 1 | 0 |
| topk_alloc | MoE and graph | 2 | - | 0 | 2 | 0 |
| gdn_cu | MoE and graph | 2 | - | 0 | 2 | 0 |
| crop_last | MoE and graph | 2 | - | 0 | 2 | 0 |
| image_threads | CPU preprocessing | 2 | 1.1% | 1 | 1 | 0 |
| tokenizer | CPU preprocessing | 2 | - | 0 | 2 | 0 |
| q6_scale | MMQ fallback | 4 | 2.4% to 3.5% | 2 | 2 | 0 |
| q8_barriers | MMQ fallback | 1 | - | 0 | 1 | 0 |
| d80_bounds | Attention | 1 | - | 0 | 1 | 0 |
| convert_fp16_vector | Fusion and traffic | 2 | 1.1% | 1 | 1 | 0 |
| convert_clamp_vector | Fusion and traffic | 2 | - | 0 | 2 | 0 |
| prefill_mask_cache | Attention | 2 | - | 0 | 2 | 0 |
| d256_groups | Attention | 2 | 2.4% | 1 | 1 | 0 |
| plain_ffn | Fusion and traffic | 2 | 5.7% to 6.7% | 2 | 0 | 0 |
| mmdq_q6_k | GEMM | 2 | - | 0 | 2 | 0 |


## Every optimization: mechanism, target and fresh result

### GEMM

#### mmdq: FP16 WMMA on quantized weights

Dequantize small weight tiles inside an FP16 WMMA matrix multiply. This avoids a full dequantized weight buffer and the MMQ Q8 activation-packing path. The saved packing and faster matrix arithmetic must outweigh local unpacking and FP16 traffic.

Target: Dense Q4_K and Q6_K projections with enough input tokens and output rows. It retains FP32 graph interfaces; this is not a global FP16-activation mode.

| Control | Type | Unset default | Off | On | Source |
| --- | --- | --- | --- | --- | --- |
| `GGML_CUDA_MMDQ` | runtime | 0 | 0 | 1 | [mmdq.cu](../../ggml/src/ggml-cuda/mmdq.cu#L932) |


Held fixed for this ablation: `GGML_CUDA_MMDQ_FUSE_NORM=0`, `GGML_CUDA_MMDQ_GLU=0`. Other settings come from the model profile.

| Workload | Off ms | On ms | Saved ms | Saved % | Result | Process spread off/on | Raw example |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-3B LLM (3968 tokens) | 985.91 | 982.61 | 3.30 | +0.33% | within variation | 0.09% / 0.16% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen25_3b/xconucstrhalo05_llm_mmdq_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen25_3b/xconucstrhalo05_llm_mmdq_on_1/summary.json) |
| Qwen3-4B LLM (3968 tokens) | 1,371.39 | 1,323.89 | 47.50 | +3.46% | gain | 0.31% / 0.13% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen3_4b/xconucstrhalo05_llm_mmdq_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen3_4b/xconucstrhalo05_llm_mmdq_on_1/summary.json) |
| Llama-2-7B LLM (3968 tokens) | 2,252.31 | 2,134.34 | 117.97 | +5.24% | gain | 0.28% / 0.62% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/llama2_7b/xconucstrhalo05_llm_mmdq_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/llama2_7b/xconucstrhalo05_llm_mmdq_on_1/summary.json) |
| Gemma-3-12B LLM (3968 tokens) | 3,592.50 | 3,344.64 | 247.86 | +6.90% | gain | 0.36% / 0.81% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma3_12b/xconucstrhalo05_llm_mmdq_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma3_12b/xconucstrhalo05_llm_mmdq_on_1/summary.json) |


#### mmdq_glu: MMDQ gate/up, activation and down fusion

Compute gate and up projections together, apply SwiGLU, and write the down-projection input in FP16. The down projection remains a separate matrix multiply. This avoids separate gate/up launches, intermediate FP32 writes, and repacking.

Target: Dense SwiGLU FFNs, including Qwen2.5. Requires MMDQ and compatible gate/up weights.

| Control | Type | Unset default | Off | On | Source |
| --- | --- | --- | --- | --- | --- |
| `GGML_CUDA_MMDQ_GLU` | runtime | 0 | 0 | 1 | [mmdq.cu](../../ggml/src/ggml-cuda/mmdq.cu#L1103) |


Held fixed for this ablation: `GGML_CUDA_MMDQ_FUSE_NORM=0`. Other settings come from the model profile.

| Workload | Off ms | On ms | Saved ms | Saved % | Result | Process spread off/on | Raw example |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-3B LLM (3968 tokens) | 978.11 | 821.15 | 156.96 | +16.05% | gain | 0.11% / 0.19% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen25_3b/xconucstrhalo05_llm_mmdq_glu_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen25_3b/xconucstrhalo05_llm_mmdq_glu_on_1/summary.json) |
| Qwen2.5-7B LLM (3968 tokens) | 1,820.33 | 1,664.94 | 155.38 | +8.54% | gain | 0.24% / 0.21% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen25_7b/xconucstrhalo05_llm_mmdq_glu_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen25_7b/xconucstrhalo05_llm_mmdq_glu_on_1/summary.json) |


#### mmdq_geglu: Extend MMDQ fusion to GEGLU

Extend the MMDQ fused FFN path from SwiGLU to the GELU-based GEGLU activation.

Target: Gemma GEGLU language FFNs. Requires MMDQ_GLU; this does not describe the plain GELU vision FFN.

| Control | Type | Unset default | Off | On | Source |
| --- | --- | --- | --- | --- | --- |
| `GGML_CUDA_MMDQ_GEGLU` | runtime | 0 | 0 | 1 | [mmdq.cu](../../ggml/src/ggml-cuda/mmdq.cu#L1104) |


Held fixed for this ablation: `GGML_CUDA_MMDQ_FUSE_NORM=0`, `GGML_CUDA_MMDQ_GLU=1`. Other settings come from the model profile.

| Workload | Off ms | On ms | Saved ms | Saved % | Result | Process spread off/on | Raw example |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Gemma-2B LLM (3968 tokens) | 709.73 | 614.18 | 95.55 | +13.46% | gain | 0.06% / 0.10% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma_2b/xconucstrhalo05_llm_mmdq_geglu_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma_2b/xconucstrhalo05_llm_mmdq_geglu_on_1/summary.json) |
| Gemma-3-12B LLM (3968 tokens) | 3,606.33 | 3,101.84 | 504.49 | +13.99% | gain | 4.74% / 9.15% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma3_12b/xconucstrhalo05_llm_mmdq_geglu_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma3_12b/xconucstrhalo05_llm_mmdq_geglu_on_1/summary.json) |


#### mmdq_norm: Normalize directly into MMDQ input

Combine RMSNorm, learned scaling and FP16 input production with the following fused MMDQ FFN. It removes the normalized FP32 intermediate and its separate conversion.

Target: A contiguous RMSNorm -> multiply -> gate/up matmuls -> GLU -> down-matmul pattern. Requires MMDQ_GLU and exclusive compatible graph consumers; this is not a generic fusion for every normalized projection.

| Control | Type | Unset default | Off | On | Source |
| --- | --- | --- | --- | --- | --- |
| `GGML_CUDA_MMDQ_FUSE_NORM` | runtime | 0 | 0 | 1 | [ggml-cuda.cu](../../ggml/src/ggml-cuda/ggml-cuda.cu#L3967) |


Context: the combined model profile; only the listed control changes.

| Workload | Off ms | On ms | Saved ms | Saved % | Result | Process spread off/on | Raw example |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-3B LLM (3968 tokens) | 822.69 | 823.29 | -0.59 | -0.07% | within variation | 0.10% / 0.05% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen25_3b/xconucstrhalo05_llm_mmdq_norm_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen25_3b/xconucstrhalo05_llm_mmdq_norm_on_1/summary.json) |
| Qwen2.5-VL-7B | 1,067.59 | 1,066.71 | 0.88 | +0.08% | within variation | 0.28% / 0.27% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen25_7b/xconucstrhalo41_vlm_mmdq_norm_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen25_7b/xconucstrhalo41_vlm_mmdq_norm_on_1/summary.json) |
| Qwen3-VL-4B VLM | 693.03 | 692.35 | 0.68 | +0.10% | within variation | 0.25% / 0.17% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen3vl/xconucstrhalo41_vlm_mmdq_norm_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen3vl/xconucstrhalo41_vlm_mmdq_norm_on_1/summary.json) |


#### mmdq_bias: MMDQ output bias fusion

Add the bias while the MMDQ kernel writes its output.

Target: Eligible biased Qwen projections. Bias-free matrices cannot benefit.

| Control | Type | Unset default | Off | On | Source |
| --- | --- | --- | --- | --- | --- |
| `GGML_CUDA_MMDQ_FUSE_BIAS` | runtime | 0 | 0 | 1 | [ggml-cuda.cu](../../ggml/src/ggml-cuda/ggml-cuda.cu#L3951) |


Context: the combined model profile; only the listed control changes.

| Workload | Off ms | On ms | Saved ms | Saved % | Result | Process spread off/on | Raw example |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-0.5B LLM (3968 tokens) | 145.69 | 142.98 | 2.71 | +1.86% | gain | 0.05% / 1.03% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen25_05b/xconucstrhalo05_llm_mmdq_bias_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen25_05b/xconucstrhalo05_llm_mmdq_bias_on_1/summary.json) |
| Qwen2.5-3B LLM (3968 tokens) | 832.50 | 823.38 | 9.12 | +1.10% | gain | 0.01% / 0.20% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen25_3b/xconucstrhalo05_llm_mmdq_bias_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen25_3b/xconucstrhalo05_llm_mmdq_bias_on_1/summary.json) |


#### mmdq_share: Reuse converted Q/K/V input

Keep the converted FP16 activation available when consecutive projections share the same source tensor.

Target: Separate Q/K/V or gate/up projections with the same input. A parent projection/FFN fusion can remove the need for this reuse.

| Control | Type | Unset default | Off | On | Source |
| --- | --- | --- | --- | --- | --- |
| `GGML_CUDA_MMDQ_SHARE_SRC1` | runtime | 0 | 0 | 1 | [common.cuh](../../ggml/src/ggml-cuda/common.cuh#L1592) |


Context: the combined model profile; only the listed control changes.

| Workload | Off ms | On ms | Saved ms | Saved % | Result | Process spread off/on | Raw example |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-3B LLM (3968 tokens) | 832.16 | 824.74 | 7.41 | +0.89% | within variation | 0.08% / 0.04% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen25_3b/xconucstrhalo05_llm_mmdq_share_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen25_3b/xconucstrhalo05_llm_mmdq_share_on_1/summary.json) |
| Qwen3-4B LLM (3968 tokens) | 1,185.79 | 1,172.87 | 12.92 | +1.09% | gain | 0.21% / 0.01% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen3_4b/xconucstrhalo05_llm_mmdq_share_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen3_4b/xconucstrhalo05_llm_mmdq_share_on_1/summary.json) |


#### mmdq_q5_0: Q5_0 MMDQ support

Add direct Q5_0 weight decoding to the MMDQ path.

Target: Q5_0 matrices inside mixed-quantization GGUFs. The Qwen2.5-0.5B file is named Q4_K_M but contains Q5_0 tensors.

| Control | Type | Unset default | Off | On | Source |
| --- | --- | --- | --- | --- | --- |
| `GGML_CUDA_MMDQ_Q5_0` | runtime | 0 | 0 | 1 | [mmdq.cu](../../ggml/src/ggml-cuda/mmdq.cu#L948) |


Context: the combined model profile; only the listed control changes.

| Workload | Off ms | On ms | Saved ms | Saved % | Result | Process spread off/on | Raw example |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-0.5B LLM (3968 tokens) | 218.21 | 139.18 | 79.03 | +36.22% | gain | 0.46% / 1.07% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen25_05b/xconucstrhalo05_llm_mmdq_q5_0_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen25_05b/xconucstrhalo05_llm_mmdq_q5_0_on_1/summary.json) |


#### mmdq_q5_k: Q5_K MMDQ support

Add direct Q5_K weight decoding to MMDQ.

Target: Eligible Q5_K matrices in mixed-quantization models. A model with no such matrices has no applicable work.

| Control | Type | Unset default | Off | On | Source |
| --- | --- | --- | --- | --- | --- |
| `GGML_CUDA_MMDQ_Q5_K` | runtime | 0 | 0 | 1 | [mmdq.cu](../../ggml/src/ggml-cuda/mmdq.cu#L947) |


Context: the combined model profile; only the listed control changes.

| Workload | Off ms | On ms | Saved ms | Saved % | Result | Process spread off/on | Raw example |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3.6-27B LLM (3968 tokens) | 8,381.66 | 8,598.47 | -216.81 | -2.59% | within variation | 11.59% / 2.62% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen36_27b/xconucstrhalo05_llm_mmdq_q5_k_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen36_27b/xconucstrhalo05_llm_mmdq_q5_k_on_1/summary.json) |


#### mmdq_q8_0: Q8_0 MMDQ support

Use the FP16 WMMA path for selected Q8_0 weight matrices.

Target: Large enough dense Q8_0 matrices; the default minimum token batch is 576 and there are output-row thresholds. A 512-token microbatch can make this inactive.

Execution-path note: The native VLM chunks in this experiment are at most 512 tokens and cannot pass the default 576-token threshold. The added 3968-token text-only rows use microbatch 1024 on the same language weights to exercise this control.

| Control | Type | Unset default | Off | On | Source |
| --- | --- | --- | --- | --- | --- |
| `GGML_CUDA_MMDQ_Q8_0` | runtime | 0 | 0 | 1 | [mmdq.cu](../../ggml/src/ggml-cuda/mmdq.cu#L941) |


Context: the combined model profile; only the listed control changes.

| Workload | Off ms | On ms | Saved ms | Saved % | Result | Process spread off/on | Raw example |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3.5-35B-A3B VLM | 345.86 | 345.84 | 0.02 | +0.01% | within variation | 0.15% / 0.13% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen35/xconucstrhalo41_vlm_mmdq_q8_0_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen35/xconucstrhalo41_vlm_mmdq_q8_0_on_1/summary.json) |
| Gemma-4-26B-A4B VLM | 594.76 | 594.62 | 0.14 | +0.02% | within variation | 0.29% / 0.18% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma4/xconucstrhalo41_vlm_mmdq_q8_0_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma4/xconucstrhalo41_vlm_mmdq_q8_0_on_1/summary.json) |
| Qwen3.5-35B-A3B LLM (3968 tokens) | 1,521.05 | 1,531.61 | -10.55 | -0.69% | within variation | 0.20% / 0.09% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen35/xconucstrhalo05_llm_mmdq_q8_0_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen35/xconucstrhalo05_llm_mmdq_q8_0_on_1/summary.json) |
| Gemma-4-26B-A4B LLM (3968 tokens) | 1,538.72 | 1,491.32 | 47.40 | +3.08% | gain | 0.31% / 0.25% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma4/xconucstrhalo05_llm_mmdq_q8_0_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma4/xconucstrhalo05_llm_mmdq_q8_0_on_1/summary.json) |


#### mmdq_moe: MMDQ expert kernels

Group routed token rows by expert and stream quantized expert weights through a specialized WMMA kernel.

Target: MoE expert matmuls with sufficient tokens per expert. Gemma-4 and Qwen3.5 test routing and mixed weight types; small expert batches retain the fallback.

Execution-path note: Qwen3.5 has 256 experts and routes each token to 8. The tested VLM has only 414 total positions: even one unsplit batch averages fewer than 16 tokens per expert, so the expert path is inactive. The added 3968-token text-only row uses microbatch 1024 and can pass the gate/up and down thresholds on full microbatches. Gemma-4 has 128 experts and is eligible in this VLM workload.

| Control | Type | Unset default | Off | On | Source |
| --- | --- | --- | --- | --- | --- |
| `GGML_CUDA_MMDQ_MOE` | runtime | 0 | 0 | 1 | [mmdq.cu](../../ggml/src/ggml-cuda/mmdq.cu#L974) |


Context: the combined model profile; only the listed control changes.

| Workload | Off ms | On ms | Saved ms | Saved % | Result | Process spread off/on | Raw example |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3.5-35B-A3B VLM | 347.14 | 346.32 | 0.82 | +0.24% | within variation | 0.23% / 0.04% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen35/xconucstrhalo41_vlm_mmdq_moe_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen35/xconucstrhalo41_vlm_mmdq_moe_on_1/summary.json) |
| Gemma-4-26B-A4B VLM | 617.99 | 593.50 | 24.49 | +3.96% | gain | 0.20% / 0.51% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma4/xconucstrhalo41_vlm_mmdq_moe_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma4/xconucstrhalo41_vlm_mmdq_moe_on_1/summary.json) |
| Qwen3.5-35B-A3B LLM (3968 tokens) | 1,701.47 | 1,589.40 | 112.08 | +6.59% | gain | 0.09% / 4.88% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen35/xconucstrhalo05_llm_mmdq_moe_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen35/xconucstrhalo05_llm_mmdq_moe_on_1/summary.json) |


#### mmdq_swizzle: Group token tiles for cache reuse

Reorder the logical grid into groups of two token tiles across output-row tiles. This can keep a smaller activation slice in cache, but trades against reuse of weight tiles across many token tiles. It does not change GEMM shapes, dtypes or the arithmetic count.

Target: Dense MMDQ when the FP16 activation matrix exceeds 8 MiB. The balance between activation and weight reuse depends on matrix shape; this is a tuning choice, not a universally faster algorithm.

| Control | Type | Unset default | Off | On | Source |
| --- | --- | --- | --- | --- | --- |
| `GGML_CUDA_MMDQ_SWIZZLE` | runtime | 0 | 0 | 2 | [mmdq.cu](../../ggml/src/ggml-cuda/mmdq.cu#L806) |


Context: the combined model profile; only the listed control changes.

| Workload | Off ms | On ms | Saved ms | Saved % | Result | Process spread off/on | Raw example |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-4B LLM (3968 tokens) | 1,177.08 | 1,170.77 | 6.31 | +0.54% | within variation | 0.09% / 0.10% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen3_4b/xconucstrhalo05_llm_mmdq_swizzle_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen3_4b/xconucstrhalo05_llm_mmdq_swizzle_on_1/summary.json) |
| Qwen3.6-27B LLM (3968 tokens) | 7,385.55 | 8,965.21 | -1,579.66 | -21.39% | slower | 5.61% / 0.24% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen36_27b/xconucstrhalo05_llm_mmdq_swizzle_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen36_27b/xconucstrhalo05_llm_mmdq_swizzle_on_1/summary.json) |


#### mmdq_q6_k: Q6_K MMDQ support

Allow Q6_K matrices to use MMDQ instead of the fallback MMQ/BLAS dispatch.

Target: Q6_K down projections and other higher-precision tensors in Q4_K_M files. The NO_Q6K flag is inverse: 0 enables this path and 1 disables it.

| Control | Type | Unset default | Off | On | Source |
| --- | --- | --- | --- | --- | --- |
| `GGML_CUDA_MMDQ_NO_Q6K` | runtime | 0 | 1 | 0 | [mmdq.cu](../../ggml/src/ggml-cuda/mmdq.cu#L937) |


Held fixed for this ablation: `GGML_CUDA_MMDQ_FUSE_NORM=0`, `GGML_CUDA_MMDQ_GLU=0`. Other settings come from the model profile.

| Workload | Off ms | On ms | Saved ms | Saved % | Result | Process spread off/on | Raw example |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Gemma-3-12B LLM (3968 tokens) | 3,403.35 | 3,398.45 | 4.91 | +0.14% | within variation | 0.21% / 0.00% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma3_12b/xconucstrhalo05_llm_mmdq_q6_k_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma3_12b/xconucstrhalo05_llm_mmdq_q6_k_on_1/summary.json) |
| Gemma-3-4B VLM | 510.52 | 506.65 | 3.87 | +0.76% | within variation | 0.21% / 0.19% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma3/xconucstrhalo05_vlm_mmdq_q6_k_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma3/xconucstrhalo05_vlm_mmdq_q6_k_on_1/summary.json) |


### Attention

#### causal_attention: Dedicated prefill attention and mask classification

Use a dedicated prefill WMMA attention kernel. Classify mask tiles so forbidden tiles can be skipped and fully allowed tiles need less mask work.

Target: Supported D64/D128/D256 prefill attention, subject to shape/mask eligibility. Includes the kernel and its mask classification, so the row is a family-level ablation.

| Control | Type | Unset default | Off | On | Source |
| --- | --- | --- | --- | --- | --- |
| `GGML_CUDA_FATTN_PREFILL` | runtime | 0 | 0 | 1 | [common.cuh](../../ggml/src/ggml-cuda/common.cuh#L1599) |


Context: the combined model profile; only the listed control changes.

| Workload | Off ms | On ms | Saved ms | Saved % | Result | Process spread off/on | Raw example |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-0.5B LLM (3968 tokens) | 189.04 | 143.57 | 45.47 | +24.05% | gain | 1.29% / 0.47% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen25_05b/xconucstrhalo05_llm_causal_attention_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen25_05b/xconucstrhalo05_llm_causal_attention_on_1/summary.json) |
| Qwen2.5-3B LLM (3968 tokens) | 843.13 | 823.24 | 19.89 | +2.36% | gain | 0.13% / 0.03% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen25_3b/xconucstrhalo05_llm_causal_attention_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen25_3b/xconucstrhalo05_llm_causal_attention_on_1/summary.json) |
| Qwen3-4B LLM (3968 tokens) | 1,214.98 | 1,172.38 | 42.60 | +3.51% | gain | 0.11% / 0.15% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen3_4b/xconucstrhalo05_llm_causal_attention_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen3_4b/xconucstrhalo05_llm_causal_attention_on_1/summary.json) |
| Gemma-2B LLM (3968 tokens) | 649.84 | 611.84 | 38.00 | +5.85% | gain | 0.11% / 0.23% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma_2b/xconucstrhalo05_llm_causal_attention_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma_2b/xconucstrhalo05_llm_causal_attention_on_1/summary.json) |


#### d256_sxh: D256 FP16 score exchange

Exchange D256 attention scores in FP16 in the specialized prefill path, reducing the storage/traffic used for score exchange.

Target: Eligible D256 attention, represented by Gemma-2B. This changes an intermediate precision and requires tolerance validation.

| Control | Type | Unset default | Off | On | Source |
| --- | --- | --- | --- | --- | --- |
| `GGML_CUDA_FATTN_PREFILL_D256_SXH` | runtime | 0 | 0 | 1 | [fattn-vision.cu](../../ggml/src/ggml-cuda/fattn-vision.cu#L705) |


Held fixed for this ablation: `GGML_CUDA_FATTN_PREFILL=1`. Other settings come from the model profile.

| Workload | Off ms | On ms | Saved ms | Saved % | Result | Process spread off/on | Raw example |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Gemma-2B LLM (3968 tokens) | 619.16 | 610.32 | 8.84 | +1.43% | gain | 0.30% / 0.05% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma_2b/xconucstrhalo05_llm_d256_sxh_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma_2b/xconucstrhalo05_llm_d256_sxh_on_1/summary.json) |


#### mask_prefix: Skip verified causal mask loads

Recognize a verified causal-prefix mask and avoid reading each mask value in the inner attention loop.

Target: Legacy D128 GQA8 or GQA4 tiles, FP16 KV, and supported prompt/KV lengths. The dedicated prefill kernel is disabled for isolation.

| Control | Type | Unset default | Off | On | Source |
| --- | --- | --- | --- | --- | --- |
| `GGML_CUDA_FATTN_MASK_PREFIX` | runtime | 1 | 0 | 1 | [fattn-common.cuh](../../ggml/src/ggml-cuda/fattn-common.cuh#L1177) |


Held fixed for this ablation: `GGML_CUDA_FATTN_PREFILL=0`. Other settings come from the model profile.

| Workload | Off ms | On ms | Saved ms | Saved % | Result | Process spread off/on | Raw example |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-3B LLM (3968 tokens) | 868.40 | 843.31 | 25.10 | +2.89% | gain | 0.23% / 0.07% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen25_3b/xconucstrhalo05_llm_mask_prefix_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen25_3b/xconucstrhalo05_llm_mask_prefix_on_1/summary.json) |
| Qwen3-4B LLM (3968 tokens) | 1,257.18 | 1,212.93 | 44.25 | +3.52% | gain | 0.02% / 0.06% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen3_4b/xconucstrhalo05_llm_mask_prefix_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen3_4b/xconucstrhalo05_llm_mask_prefix_on_1/summary.json) |


#### gqa4_wide: Wider GQA4 attention query tile

Use a wider query tile so a block reuses K/V work across more GQA4 queries.

Target: Legacy D128, four query heads per KV head, supported aligned prompt sizes. Larger tiles trade register pressure for reuse.

| Control | Type | Unset default | Off | On | Source |
| --- | --- | --- | --- | --- | --- |
| `GGML_CUDA_FATTN_GQA4_WIDE` | runtime | 1 | 0 | 1 | [fattn-mma-f16.cuh](../../ggml/src/ggml-cuda/fattn-mma-f16.cuh#L2167) |


Held fixed for this ablation: `GGML_CUDA_FATTN_PREFILL=0`. Other settings come from the model profile.

| Workload | Off ms | On ms | Saved ms | Saved % | Result | Process spread off/on | Raw example |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-4B LLM (3968 tokens) | 1,204.97 | 1,214.79 | -9.82 | -0.82% | within variation | 0.15% / 0.02% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen3_4b/xconucstrhalo05_llm_gqa4_wide_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen3_4b/xconucstrhalo05_llm_gqa4_wide_on_1/summary.json) |
| Qwen3-8B LLM (3968 tokens) | 2,090.34 | 2,096.44 | -6.09 | -0.29% | within variation | 0.26% / 0.00% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen3_8b/xconucstrhalo05_llm_gqa4_wide_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen3_8b/xconucstrhalo05_llm_gqa4_wide_on_1/summary.json) |


#### gqa8_vgpr: Limit GQA8 attention registers

Select a register-limited specialization to improve wave residency.

Target: Legacy D128 GQA8 attention. It can help when register allocation limits occupancy but may be neutral for other bottlenecks.

| Control | Type | Unset default | Off | On | Source |
| --- | --- | --- | --- | --- | --- |
| `GGML_CUDA_FATTN_GQA8_VGPR` | runtime | 1 | 0 | 1 | [fattn-mma-f16.cuh](../../ggml/src/ggml-cuda/fattn-mma-f16.cuh#L2201) |


Held fixed for this ablation: `GGML_CUDA_FATTN_PREFILL=0`. Other settings come from the model profile.

| Workload | Off ms | On ms | Saved ms | Saved % | Result | Process spread off/on | Raw example |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-3B LLM (3968 tokens) | 847.82 | 845.73 | 2.09 | +0.25% | within variation | 0.12% / 0.16% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen25_3b/xconucstrhalo05_llm_gqa8_vgpr_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen25_3b/xconucstrhalo05_llm_gqa8_vgpr_on_1/summary.json) |


#### d512_wmma: D512 language attention WMMA

Dispatch supported D512 attention to the existing 4 x 8 WMMA tile that fits RDNA shared memory.

Target: Gemma-4 global language attention with GQA8 and FP16 KV. Gemma-3 uses D256 language attention and is not this target.

| Control | Type | Unset default | Off | On | Source |
| --- | --- | --- | --- | --- | --- |
| `GGML_CUDA_FATTN_WMMA_D512` | runtime | 1 | 0 | 1 | [fattn.cu](../../ggml/src/ggml-cuda/fattn.cu#L753) |


Context: the combined model profile; only the listed control changes.

| Workload | Off ms | On ms | Saved ms | Saved % | Result | Process spread off/on | Raw example |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Gemma-4-26B-A4B VLM | 597.75 | 595.33 | 2.42 | +0.40% | within variation | 0.27% / 0.30% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma4/xconucstrhalo41_vlm_d512_wmma_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma4/xconucstrhalo41_vlm_d512_wmma_on_1/summary.json) |


#### vision_attention: Dedicated D64/D72/D80 vision attention

Use dedicated noncausal vision attention kernels for D64/D72/D80, handling attention tiles without a large explicit score matrix.

Target: Qwen and Gemma vision self-attention. The fallback can already use WMMA; this measures the newer kernel over that fallback.

| Control | Type | Unset default | Off | On | Source |
| --- | --- | --- | --- | --- | --- |
| `GGML_CUDA_FATTN_VISION` | runtime | 0 | 0 | 1 | [fattn-vision.cuh](../../ggml/src/ggml-cuda/fattn-vision.cuh#L3) |


Context: the combined model profile; only the listed control changes.

| Workload | Off ms | On ms | Saved ms | Saved % | Result | Process spread off/on | Raw example |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-VL-3B | 675.74 | 668.19 | 7.55 | +1.12% | gain | 0.53% / 0.06% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen25_3b/xconucstrhalo41_vlm_vision_attention_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen25_3b/xconucstrhalo41_vlm_vision_attention_on_1/summary.json) |
| Qwen3-VL-4B VLM | 731.19 | 693.63 | 37.57 | +5.14% | gain | 0.54% / 0.38% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen3vl/xconucstrhalo41_vlm_vision_attention_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen3vl/xconucstrhalo41_vlm_vision_attention_on_1/summary.json) |
| Gemma-3-4B VLM | 526.54 | 475.41 | 51.13 | +9.71% | gain | 0.11% / 0.00% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma3/xconucstrhalo05_vlm_vision_attention_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma3/xconucstrhalo05_vlm_vision_attention_on_1/summary.json) |


#### vision_wmma: Enable non-GQA D64/D80 WMMA

Allow the existing WMMA attention path for unmasked self-attention with no grouped-query sharing.

Target: D64/D80 vision heads, equal Q/K sequence lengths, FP16 K/V. Dedicated vision attention is disabled for this test.

| Control | Type | Unset default | Off | On | Source |
| --- | --- | --- | --- | --- | --- |
| `GGML_CUDA_FATTN_WMMA_NO_GQA` | runtime | 1 | 0 | 1 | [fattn.cu](../../ggml/src/ggml-cuda/fattn.cu#L733) |


Held fixed for this ablation: `GGML_CUDA_FATTN_VISION=0`. Other settings come from the model profile.

| Workload | Off ms | On ms | Saved ms | Saved % | Result | Process spread off/on | Raw example |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-VL-3B | 792.24 | 678.24 | 114.00 | +14.39% | gain | 0.27% / 0.55% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen25_3b/xconucstrhalo41_vlm_vision_wmma_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen25_3b/xconucstrhalo41_vlm_vision_wmma_on_1/summary.json) |
| Qwen3-VL-4B VLM | 907.85 | 732.95 | 174.90 | +19.27% | gain | 0.11% / 0.06% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen3vl/xconucstrhalo41_vlm_vision_wmma_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen3vl/xconucstrhalo41_vlm_vision_wmma_on_1/summary.json) |


#### d72_wmma: Pad D72 to the existing D80 WMMA tile

Pad a D72 head inside an existing D80 WMMA tile, replacing the slower generic path without changing the logical 72-channel result.

Target: Gemma-3 vision attention with sufficient tokens. Dedicated vision attention is disabled for this test.

| Control | Type | Unset default | Off | On | Source |
| --- | --- | --- | --- | --- | --- |
| `GGML_CUDA_FATTN_WMMA_D72` | runtime | 1 | 0 | 1 | [fattn.cu](../../ggml/src/ggml-cuda/fattn.cu#L742) |


Held fixed for this ablation: `GGML_CUDA_FATTN_VISION=0`. Other settings come from the model profile.

| Workload | Off ms | On ms | Saved ms | Saved % | Result | Process spread off/on | Raw example |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Gemma-3-4B VLM | 900.41 | 529.34 | 371.07 | +41.21% | gain | 0.14% / 0.12% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma3/xconucstrhalo05_vlm_d72_wmma_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma3/xconucstrhalo05_vlm_d72_wmma_on_1/summary.json) |
| Gemma-3-12B VLM | 986.78 | 616.51 | 370.27 | +37.52% | gain | 0.75% / 0.17% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma3_12b/xconucstrhalo05_vlm_d72_wmma_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma3_12b/xconucstrhalo05_vlm_d72_wmma_on_1/summary.json) |


#### d80_bounds: Remove full-tile D80 bounds checks

Remove per-element bounds checks for full D80 K/V tiles. Retain bounds checks for the final partial tile.

Target: Legacy dense D80, 64-query, single-head WMMA specialization. Dedicated vision attention is disabled.

| Control | Type | Unset default | Off | On | Source |
| --- | --- | --- | --- | --- | --- |
| `GGML_CUDA_FATTN_D80_FULL_TILE` | compile-time | 1 | 0 | 1 | [fattn-mma-f16.cuh](../../ggml/src/ggml-cuda/fattn-mma-f16.cuh#L128) |


Held fixed for this ablation: `GGML_CUDA_FATTN_VISION=0`. Other settings come from the model profile.

| Workload | Off ms | On ms | Saved ms | Saved % | Result | Process spread off/on | Raw example |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-VL-3B | 692.28 | 687.81 | 4.47 | +0.65% | within variation | 0.19% / 0.32% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen25_3b/xconucstrhalo41_vlm_d80_bounds_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen25_3b/xconucstrhalo41_vlm_d80_bounds_on_1/summary.json) |


#### prefill_mask_cache: Reuse attention mask classification

Reuse mask classification across eligible attention nodes in the same graph instead of launching another classifier for each node.

Target: The dedicated prefill kernel, identical mask pointer/layout, and compatible pool/stream state. The cache is bounded to the graph lifecycle.

| Control | Type | Unset default | Off | On | Source |
| --- | --- | --- | --- | --- | --- |
| `GGML_CUDA_FATTN_PREFILL_MASK_CACHE` | runtime | 1 | 0 | 1 | [fattn-vision.cu](../../ggml/src/ggml-cuda/fattn-vision.cu#L671) |


Context: the combined model profile; only the listed control changes.

| Workload | Off ms | On ms | Saved ms | Saved % | Result | Process spread off/on | Raw example |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-3B LLM (3968 tokens) | 833.47 | 827.68 | 5.80 | +0.70% | within variation | 0.11% / 0.01% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen25_3b/xconucstrhalo05_llm_prefill_mask_cache_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen25_3b/xconucstrhalo05_llm_prefill_mask_cache_on_1/summary.json) |
| Gemma-2B LLM (3968 tokens) | 611.30 | 610.98 | 0.32 | +0.05% | within variation | 0.17% / 0.05% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma_2b/xconucstrhalo05_llm_prefill_mask_cache_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma_2b/xconucstrhalo05_llm_prefill_mask_cache_on_1/summary.json) |


#### d256_groups: Increase query groups in D256 attention

Process eight query groups instead of four in the D256 kernel.

Target: D256 attention with the FP16 score-exchange variant disabled for this comparison. More reuse can cost more registers or reduce occupancy.

| Control | Type | Unset default | Off | On | Source |
| --- | --- | --- | --- | --- | --- |
| `GGML_CUDA_FATTN_PREFILL_D256_NG` | runtime | 8 | 4 | 8 | [fattn-vision.cu](../../ggml/src/ggml-cuda/fattn-vision.cu#L657) |


Held fixed for this ablation: `GGML_CUDA_FATTN_PREFILL_D256_SXH=0`. Other settings come from the model profile.

| Workload | Off ms | On ms | Saved ms | Saved % | Result | Process spread off/on | Raw example |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Gemma-2B LLM (3968 tokens) | 634.73 | 619.25 | 15.48 | +2.44% | gain | 0.17% / 0.37% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma_2b/xconucstrhalo05_llm_d256_groups_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma_2b/xconucstrhalo05_llm_d256_groups_on_1/summary.json) |
| Qwen3.6-27B LLM (3968 tokens) | 8,949.62 | 8,970.39 | -20.78 | -0.23% | within variation | 0.35% / 0.56% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen36_27b/xconucstrhalo05_llm_d256_groups_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen36_27b/xconucstrhalo05_llm_d256_groups_on_1/summary.json) |


### Vision graph

#### window_batch: Batch equal-sized vision windows

Represent independent equal-sized image windows as batch entries. Attention only computes within each small window, replacing one large masked attention problem.

Target: Qwen2.5-VL window-attention layers. Full-attention layers are unchanged. Disabling restores construction and casting of the large block-diagonal mask.

| Control | Type | Unset default | Off | On | Source |
| --- | --- | --- | --- | --- | --- |
| `MTMD_QWEN25VL_WINDOW_BATCH` | runtime | 1 | 0 | 1 | [qwen2vl.cpp](../../tools/mtmd/models/qwen2vl.cpp#L40) |


Context: the combined model profile; only the listed control changes.

| Workload | Off ms | On ms | Saved ms | Saved % | Result | Process spread off/on | Raw example |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-VL-3B | 1,884.79 | 676.61 | 1,208.18 | +64.10% | gain | 0.16% / 0.36% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen25_3b/xconucstrhalo41_vlm_window_batch_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen25_3b/xconucstrhalo41_vlm_window_batch_on_1/summary.json) |
| Qwen2.5-VL-7B | 2,286.80 | 1,072.09 | 1,214.70 | +53.12% | gain | 0.10% / 0.17% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen25_7b/xconucstrhalo41_vlm_window_batch_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen25_7b/xconucstrhalo41_vlm_window_batch_on_1/summary.json) |


#### window_cast: Cast full window K/V before splitting

Convert a complete K/V tensor once before splitting it into window views.

Target: Batched Qwen2.5-VL windows. Avoids many small conversions and strided copies. General early K/V conversion is disabled in this comparison.

| Control | Type | Unset default | Off | On | Source |
| --- | --- | --- | --- | --- | --- |
| `MTMD_QWEN25VL_KV_CAST_EARLY` | runtime | 1 | 0 | 1 | [qwen2vl.cpp](../../tools/mtmd/models/qwen2vl.cpp#L41) |


Held fixed for this ablation: `MTMD_VISION_KV_CAST_EARLY=0`. Other settings come from the model profile.

| Workload | Off ms | On ms | Saved ms | Saved % | Result | Process spread off/on | Raw example |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-VL-3B | 678.55 | 677.85 | 0.70 | +0.10% | within variation | 0.61% / 0.05% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen25_3b/xconucstrhalo41_vlm_window_cast_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen25_3b/xconucstrhalo41_vlm_window_cast_on_1/summary.json) |
| Qwen2.5-VL-7B | 1,075.15 | 1,077.23 | -2.07 | -0.19% | within variation | 0.14% / 0.22% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen25_7b/xconucstrhalo41_vlm_window_cast_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen25_7b/xconucstrhalo41_vlm_window_cast_on_1/summary.json) |


#### vision_qkv: Combine vision Q/K/V weight projections

Concatenate separate Q/K/V weights at model load and perform one larger projection during inference.

Target: The optional loader path targets Gemma-3. Qwen towers already carrying combined QKV weights are applicability controls and are not expected to benefit from this switch.

Execution-path note: The Qwen2.5 and Qwen3 towers already store combined QKV weights, so this loader switch has no projection-combining work to do there. The separate-projection Gemma-3 tower is the active target.

| Control | Type | Unset default | Off | On | Source |
| --- | --- | --- | --- | --- | --- |
| `MTMD_VISION_FUSE_QKV` | runtime | 0 | 0 | 1 | [clip.cpp](../../tools/mtmd/clip.cpp#L3681) |


Context: the combined model profile; only the listed control changes.

| Workload | Off ms | On ms | Saved ms | Saved % | Result | Process spread off/on | Raw example |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-VL-3B | 677.25 | 676.65 | 0.60 | +0.09% | within variation | 0.08% / 0.15% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen25_3b/xconucstrhalo41_vlm_vision_qkv_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen25_3b/xconucstrhalo41_vlm_vision_qkv_on_1/summary.json) |
| Gemma-3-4B VLM | 481.79 | 477.19 | 4.60 | +0.96% | within variation | 0.07% / 0.07% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma3/xconucstrhalo05_vlm_vision_qkv_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma3/xconucstrhalo05_vlm_vision_qkv_on_1/summary.json) |
| Qwen3-VL-4B VLM | 697.53 | 696.91 | 0.62 | +0.09% | within variation | 0.16% / 0.10% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen3vl/xconucstrhalo41_vlm_vision_qkv_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen3vl/xconucstrhalo41_vlm_vision_qkv_on_1/summary.json) |


#### vision_cast: Cast vision K/V before permutation

Cast contiguous K/V tensors before permuting the tensor views for attention.

Target: FP32-produced vision K/V. Permuting a view is cheap, but casting after permutation can require a strided memory pass.

| Control | Type | Unset default | Off | On | Source |
| --- | --- | --- | --- | --- | --- |
| `MTMD_VISION_KV_CAST_EARLY` | runtime | 0 | 0 | 1 | [clip.cpp](../../tools/mtmd/clip.cpp#L794) |


Context: the combined model profile; only the listed control changes.

| Workload | Off ms | On ms | Saved ms | Saved % | Result | Process spread off/on | Raw example |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Gemma-3-4B VLM | 478.56 | 478.00 | 0.57 | +0.12% | within variation | 0.01% / 0.05% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma3/xconucstrhalo05_vlm_vision_cast_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma3/xconucstrhalo05_vlm_vision_cast_on_1/summary.json) |
| Qwen3-VL-4B VLM | 714.78 | 688.34 | 26.43 | +3.70% | gain | 0.86% / 2.36% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen3vl/xconucstrhalo41_vlm_vision_cast_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen3vl/xconucstrhalo41_vlm_vision_cast_on_1/summary.json) |


### Fusion and traffic

#### attn_set: Write attention output into its joined buffer

Have attention write directly into the correct slice of the joined output tensor.

Target: FLASH_ATTN_EXT -> RESHAPE -> SET graphs from window batching. Saves intermediate writes and a separate join/copy operation.

| Control | Type | Unset default | Off | On | Source |
| --- | --- | --- | --- | --- | --- |
| `GGML_CUDA_FUSE_ATTN_SET` | runtime | 1 | 0 | 1 | [ggml-cuda.cu](../../ggml/src/ggml-cuda/ggml-cuda.cu#L4015) |


Context: the combined model profile; only the listed control changes.

| Workload | Off ms | On ms | Saved ms | Saved % | Result | Process spread off/on | Raw example |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-VL-3B | 679.25 | 683.42 | -4.17 | -0.61% | within variation | 0.21% / 2.12% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen25_3b/xconucstrhalo41_vlm_attn_set_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen25_3b/xconucstrhalo41_vlm_attn_set_on_1/summary.json) |
| Qwen2.5-VL-7B | 1,078.67 | 1,078.06 | 0.60 | +0.06% | within variation | 0.05% / 0.35% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen25_7b/xconucstrhalo41_vlm_attn_set_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen25_7b/xconucstrhalo41_vlm_attn_set_on_1/summary.json) |


#### fp16_fusions: FP16 BLAS producer and output fusions

Let FP16-weight BLAS producer/output fusions keep eligible intermediates in FP16 and combine adjacent conversions and pointwise work.

Target: FP16 vision matmuls with compatible graph consumers. This is a precision-family gate and overlaps the FFN and bias gates.

| Control | Type | Unset default | Off | On | Source |
| --- | --- | --- | --- | --- | --- |
| `GGML_CUDA_FUSE_CUBLAS_FP16` | runtime | 1 | 0 | 1 | [ggml-cuda.cu](../../ggml/src/ggml-cuda/ggml-cuda.cu#L3655) |


Context: the combined model profile; only the listed control changes.

| Workload | Off ms | On ms | Saved ms | Saved % | Result | Process spread off/on | Raw example |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-VL-3B | 789.83 | 679.63 | 110.21 | +13.95% | gain | 0.00% / 0.10% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen25_3b/xconucstrhalo41_vlm_fp16_fusions_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen25_3b/xconucstrhalo41_vlm_fp16_fusions_on_1/summary.json) |
| Gemma-3-12B VLM | 633.54 | 568.94 | 64.59 | +10.20% | gain | 0.41% / 0.37% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma3_12b/xconucstrhalo05_vlm_fp16_fusions_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma3_12b/xconucstrhalo05_vlm_fp16_fusions_on_1/summary.json) |


#### bf16_fusions: BF16 BLAS producer and output fusions

Apply the same producer/output fusion framework to BF16 weights and BF16 temporary storage.

Target: Eligible BF16-weight BLAS matmuls, represented here by BF16 vision towers. The control can also affect eligible BF16 language matrices; these are end-to-end family timings, not vision-only kernel timings.

Execution-path note: The original Gemma-3 and Qwen3.5 projector files contain FP16 weights, so their BF16 rows are inactive controls. The added E2B and Omni files contain BF16 vision matrices and test the actual BF16 path.

| Control | Type | Unset default | Off | On | Source |
| --- | --- | --- | --- | --- | --- |
| `GGML_CUDA_FUSE_CUBLAS_BF16` | runtime | 1 | 0 | 1 | [ggml-cuda.cu](../../ggml/src/ggml-cuda/ggml-cuda.cu#L3656) |


Context: the combined model profile; only the listed control changes.

| Workload | Off ms | On ms | Saved ms | Saved % | Result | Process spread off/on | Raw example |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Gemma-3-4B VLM | 478.95 | 477.49 | 1.46 | +0.31% | within variation | 0.32% / 0.26% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma3/xconucstrhalo05_vlm_bf16_fusions_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma3/xconucstrhalo05_vlm_bf16_fusions_on_1/summary.json) |
| Qwen3.5-35B-A3B VLM | 346.89 | 347.31 | -0.42 | -0.12% | within variation | 0.02% / 0.01% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen35/xconucstrhalo41_vlm_bf16_fusions_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen35/xconucstrhalo41_vlm_bf16_fusions_on_1/summary.json) |
| Gemma-4-E2B (BF16 projector) VLM | 288.53 | 259.86 | 28.66 | +9.93% | gain | 0.22% / 0.31% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma4_e2b/xconucstrhalo05_vlm_bf16_fusions_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma4_e2b/xconucstrhalo05_vlm_bf16_fusions_on_1/summary.json) |
| Qwen3-Omni-30B-A3B (BF16 projector) VLM | 827.57 | 768.94 | 58.63 | +7.09% | gain | 0.21% / 0.05% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/omni/xconucstrhalo05_vlm_bf16_fusions_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/omni/xconucstrhalo05_vlm_bf16_fusions_on_1/summary.json) |


#### ffn_fusions: Keep BLAS FFN intermediates in 16-bit storage

Keep the intermediate of an eligible BLAS FFN in 16-bit storage and fuse its activation/conversion steps.

Target: FP16/BF16 vision FFNs, including gated, plain-GELU and clamped patterns. A parent gate for several narrower rows.

| Control | Type | Unset default | Off | On | Source |
| --- | --- | --- | --- | --- | --- |
| `GGML_CUDA_FUSE_CUBLAS_FFN` | runtime | 1 | 0 | 1 | [ggml-cuda.cu](../../ggml/src/ggml-cuda/ggml-cuda.cu#L3710) |


Context: the combined model profile; only the listed control changes.

| Workload | Off ms | On ms | Saved ms | Saved % | Result | Process spread off/on | Raw example |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-VL-3B | 720.47 | 677.34 | 43.13 | +5.99% | gain | 0.00% / 0.25% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen25_3b/xconucstrhalo41_vlm_ffn_fusions_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen25_3b/xconucstrhalo41_vlm_ffn_fusions_on_1/summary.json) |
| Gemma-3-4B VLM | 511.56 | 478.58 | 32.99 | +6.45% | gain | 0.25% / 0.22% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma3/xconucstrhalo05_vlm_ffn_fusions_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma3/xconucstrhalo05_vlm_ffn_fusions_on_1/summary.json) |
| Gemma-4-26B-A4B VLM | 623.58 | 597.54 | 26.05 | +4.18% | gain | 0.18% / 0.44% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma4/xconucstrhalo41_vlm_ffn_fusions_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma4/xconucstrhalo41_vlm_ffn_fusions_on_1/summary.json) |
| Gemma-4-E2B (BF16 projector) VLM | 271.21 | 259.90 | 11.31 | +4.17% | gain | 0.11% / 0.77% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma4_e2b/xconucstrhalo05_vlm_ffn_fusions_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma4_e2b/xconucstrhalo05_vlm_ffn_fusions_on_1/summary.json) |


#### ffn_norm: Include RMSNorm in the 16-bit FFN path

Include an immediately preceding RMSNorm in the fused 16-bit FFN producer.

Target: Compatible normalized gated FFNs. Requires the parent FFN and precision-family gates.

| Control | Type | Unset default | Off | On | Source |
| --- | --- | --- | --- | --- | --- |
| `GGML_CUDA_FUSE_CUBLAS_FFN_NORM` | runtime | 1 | 0 | 1 | [ggml-cuda.cu](../../ggml/src/ggml-cuda/ggml-cuda.cu#L3711) |


Context: the combined model profile; only the listed control changes.

| Workload | Off ms | On ms | Saved ms | Saved % | Result | Process spread off/on | Raw example |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-VL-3B | 684.85 | 676.97 | 7.89 | +1.15% | gain | 0.27% / 0.04% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen25_3b/xconucstrhalo41_vlm_ffn_norm_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen25_3b/xconucstrhalo41_vlm_ffn_norm_on_1/summary.json) |
| Qwen2.5-VL-7B | 1,081.36 | 1,079.23 | 2.14 | +0.20% | within variation | 0.31% / 0.44% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen25_7b/xconucstrhalo41_vlm_ffn_norm_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen25_7b/xconucstrhalo41_vlm_ffn_norm_on_1/summary.json) |


#### bias_fusion: Combine BLAS conversion and bias

Add a BLAS output bias during conversion back to the graph output type.

Target: Standalone biased matmuls. Parent FFN fusion is disabled in this comparison so the standalone pattern remains visible.

| Control | Type | Unset default | Off | On | Source |
| --- | --- | --- | --- | --- | --- |
| `GGML_CUDA_FUSE_CUBLAS_BIAS` | runtime | 1 | 0 | 1 | [ggml-cuda.cu](../../ggml/src/ggml-cuda/ggml-cuda.cu#L4119) |


Held fixed for this ablation: `GGML_CUDA_FUSE_CUBLAS_FFN=0`. Other settings come from the model profile.

| Workload | Off ms | On ms | Saved ms | Saved % | Result | Process spread off/on | Raw example |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-VL-3B | 775.01 | 721.87 | 53.13 | +6.86% | gain | 0.04% / 0.44% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen25_3b/xconucstrhalo41_vlm_bias_fusion_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen25_3b/xconucstrhalo41_vlm_bias_fusion_on_1/summary.json) |
| Gemma-3-4B VLM | 547.01 | 511.66 | 35.34 | +6.46% | gain | 0.43% / 0.02% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma3/xconucstrhalo05_vlm_bias_fusion_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma3/xconucstrhalo05_vlm_bias_fusion_on_1/summary.json) |


#### residual_fusion: Include residual addition in BLAS output conversion

Include residual addition in an eligible BLAS bias/output-conversion epilogue.

Target: Compatible matmul -> bias -> residual graphs. This gate does not disable every residual operation embedded in every parent FFN pattern.

| Control | Type | Unset default | Off | On | Source |
| --- | --- | --- | --- | --- | --- |
| `GGML_CUDA_FUSE_CUBLAS_RESIDUAL` | runtime | 1 | 0 | 1 | [ggml-cuda.cu](../../ggml/src/ggml-cuda/ggml-cuda.cu#L3694) |


Context: the combined model profile; only the listed control changes.

| Workload | Off ms | On ms | Saved ms | Saved % | Result | Process spread off/on | Raw example |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Gemma-3-4B VLM | 484.27 | 478.76 | 5.52 | +1.14% | gain | 0.05% / 0.09% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma3/xconucstrhalo05_vlm_residual_fusion_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma3/xconucstrhalo05_vlm_residual_fusion_on_1/summary.json) |
| Qwen2.5-VL-3B | 687.47 | 680.18 | 7.28 | +1.06% | gain | 0.69% / 0.29% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen25_3b/xconucstrhalo41_vlm_residual_fusion_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen25_3b/xconucstrhalo41_vlm_residual_fusion_on_1/summary.json) |


#### clamp_fusion: Combine Gemma vision clamps and BLAS conversions

Combine Gemma vision clamps with BLAS input/output conversions and clamped FFN patterns.

Target: Gemma-4 clamped vision projections. Saves full-tensor memory passes while retaining clamp bounds.

| Control | Type | Unset default | Off | On | Source |
| --- | --- | --- | --- | --- | --- |
| `GGML_CUDA_FUSE_CUBLAS_CLAMP` | runtime | 1 | 0 | 1 | [ggml-cuda.cu](../../ggml/src/ggml-cuda/ggml-cuda.cu#L3715) |


Context: the combined model profile; only the listed control changes.

| Workload | Off ms | On ms | Saved ms | Saved % | Result | Process spread off/on | Raw example |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Gemma-4-26B-A4B VLM | 661.44 | 598.33 | 63.11 | +9.54% | gain | 0.01% / 0.17% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma4/xconucstrhalo41_vlm_clamp_fusion_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma4/xconucstrhalo41_vlm_clamp_fusion_on_1/summary.json) |
| Gemma-4-E2B (BF16 projector) VLM | 287.33 | 259.07 | 28.27 | +9.84% | gain | 0.24% / 0.63% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma4_e2b/xconucstrhalo05_vlm_clamp_fusion_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma4_e2b/xconucstrhalo05_vlm_clamp_fusion_on_1/summary.json) |


#### layernorm_affine: Fuse LayerNorm, weight and bias

Compute LayerNorm and its learned weight/bias in one kernel.

Target: Vision LayerNorm -> multiply -> add patterns. Avoids two extra memory passes and launches.

| Control | Type | Unset default | Off | On | Source |
| --- | --- | --- | --- | --- | --- |
| `GGML_CUDA_FUSE_NORM_MUL_ADD` | runtime | 1 | 0 | 1 | [ggml-cuda.cu](../../ggml/src/ggml-cuda/ggml-cuda.cu#L4074) |


Context: the combined model profile; only the listed control changes.

| Workload | Off ms | On ms | Saved ms | Saved % | Result | Process spread off/on | Raw example |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Gemma-3-4B VLM | 489.40 | 478.93 | 10.47 | +2.14% | gain | 0.11% / 0.33% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma3/xconucstrhalo05_vlm_layernorm_affine_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma3/xconucstrhalo05_vlm_layernorm_affine_on_1/summary.json) |
| Qwen3-VL-4B VLM | 701.70 | 694.57 | 7.13 | +1.02% | gain | 0.69% / 0.02% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen3vl/xconucstrhalo41_vlm_layernorm_affine_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen3vl/xconucstrhalo41_vlm_layernorm_affine_on_1/summary.json) |


#### rms_post_scale: Fuse post-residual RMSNorm scale

Fuse a post-residual RMSNorm with its following scaling operation.

Target: Matching Gemma-4 normalization graphs. Pattern and consumer checks must succeed.

| Control | Type | Unset default | Off | On | Source |
| --- | --- | --- | --- | --- | --- |
| `GGML_CUDA_FUSE_RMS_POST_SCALE` | runtime | 1 | 0 | 1 | [ggml-cuda.cu](../../ggml/src/ggml-cuda/ggml-cuda.cu#L4178) |


Context: the combined model profile; only the listed control changes.

| Workload | Off ms | On ms | Saved ms | Saved % | Result | Process spread off/on | Raw example |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Gemma-4-26B-A4B VLM | 599.42 | 597.67 | 1.76 | +0.29% | within variation | 0.39% / 0.10% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma4/xconucstrhalo41_vlm_rms_post_scale_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma4/xconucstrhalo41_vlm_rms_post_scale_on_1/summary.json) |


#### rms_scale: Fuse RMSNorm and scalar scale

Include a scalar scale in the existing RMSNorm fusion.

Target: Matching RMSNorm/scale graphs in dense or MoE models. May overlap broader normalization fusions.

| Control | Type | Unset default | Off | On | Source |
| --- | --- | --- | --- | --- | --- |
| `GGML_CUDA_FUSE_RMS_SCALE` | runtime | 1 | 0 | 1 | [ggml-cuda.cu](../../ggml/src/ggml-cuda/ggml-cuda.cu#L3518) |


Context: the combined model profile; only the listed control changes.

| Workload | Off ms | On ms | Saved ms | Saved % | Result | Process spread off/on | Raw example |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3.5-35B-A3B VLM | 348.25 | 347.05 | 1.20 | +0.34% | within variation | 0.23% / 0.23% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen35/xconucstrhalo41_vlm_rms_scale_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen35/xconucstrhalo41_vlm_rms_scale_on_1/summary.json) |
| Gemma-4-26B-A4B VLM | 600.85 | 597.83 | 3.02 | +0.50% | within variation | 0.04% / 0.32% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma4/xconucstrhalo41_vlm_rms_scale_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma4/xconucstrhalo41_vlm_rms_scale_on_1/summary.json) |


#### gelu_mul: Fuse separate GELU and multiplication

Combine separate GELU and multiplication nodes into a GLU operation.

Target: Gemma GEGLU language graphs. MMDQ is disabled so its parent FFN fusion does not hide this path.

| Control | Type | Unset default | Off | On | Source |
| --- | --- | --- | --- | --- | --- |
| `GGML_CUDA_FUSE_GELU_MUL` | runtime | 1 | 0 | 1 | [ggml-cuda.cu](../../ggml/src/ggml-cuda/ggml-cuda.cu#L4157) |


Held fixed for this ablation: `GGML_CUDA_MMDQ=0`, `GGML_CUDA_MMDQ_FUSE_NORM=0`, `GGML_CUDA_MMDQ_GLU=0`, `GGML_CUDA_MMDQ_MOE=0`. Other settings come from the model profile.

| Workload | Off ms | On ms | Saved ms | Saved % | Result | Process spread off/on | Raw example |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Gemma-2B LLM (3968 tokens) | 723.35 | 724.83 | -1.48 | -0.20% | within variation | 0.30% / 0.24% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma_2b/xconucstrhalo05_llm_gelu_mul_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma_2b/xconucstrhalo05_llm_gelu_mul_on_1/summary.json) |
| Gemma-3-12B LLM (3968 tokens) | 3,637.69 | 3,642.52 | -4.84 | -0.13% | within variation | 0.59% / 0.08% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma3_12b/xconucstrhalo05_llm_gelu_mul_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma3_12b/xconucstrhalo05_llm_gelu_mul_on_1/summary.json) |


#### norm_pack: Normalize and share Q8 packed FFN input

Fuse RMSNorm/scale with Q8 activation packing and share the packed input across eligible FFN matmuls.

Target: The MMQ execution path. Saves normalized FP32 traffic and duplicate packing; MMDQ is disabled.

| Control | Type | Unset default | Off | On | Source |
| --- | --- | --- | --- | --- | --- |
| `GGML_CUDA_FUSE_NORM_PACK` | runtime | 1 | 0 | 1 | [ggml-cuda.cu](../../ggml/src/ggml-cuda/ggml-cuda.cu#L4195) |


Held fixed for this ablation: `GGML_CUDA_MMDQ=0`, `GGML_CUDA_MMDQ_FUSE_NORM=0`, `GGML_CUDA_MMDQ_GLU=0`, `GGML_CUDA_MMDQ_MOE=0`. Other settings come from the model profile.

| Workload | Off ms | On ms | Saved ms | Saved % | Result | Process spread off/on | Raw example |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-3B LLM (3968 tokens) | 1,002.59 | 990.18 | 12.41 | +1.24% | gain | 0.25% / 0.02% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen25_3b/xconucstrhalo05_llm_norm_pack_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen25_3b/xconucstrhalo05_llm_norm_pack_on_1/summary.json) |
| Qwen3-4B LLM (3968 tokens) | 1,382.79 | 1,371.49 | 11.30 | +0.82% | within variation | 0.01% / 0.30% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen3_4b/xconucstrhalo05_llm_norm_pack_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen3_4b/xconucstrhalo05_llm_norm_pack_on_1/summary.json) |


#### glu_pack: Fuse GLU and Q8 activation packing

Produce MMQ Q8 input directly from GLU output rather than first writing the full FP32 activation.

Target: MMQ down projections after SwiGLU/GEGLU; supports the required Q4_K/Q6_K packed layouts.

| Control | Type | Unset default | Off | On | Source |
| --- | --- | --- | --- | --- | --- |
| `GGML_CUDA_FUSE_GLU_PACK` | runtime | 1 | 0 | 1 | [ggml-cuda.cu](../../ggml/src/ggml-cuda/ggml-cuda.cu#L4141) |


Held fixed for this ablation: `GGML_CUDA_MMDQ=0`, `GGML_CUDA_MMDQ_FUSE_NORM=0`, `GGML_CUDA_MMDQ_GLU=0`, `GGML_CUDA_MMDQ_MOE=0`. Other settings come from the model profile.

| Workload | Off ms | On ms | Saved ms | Saved % | Result | Process spread off/on | Raw example |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-3B LLM (3968 tokens) | 1,014.83 | 990.70 | 24.13 | +2.38% | gain | 0.12% / 0.24% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen25_3b/xconucstrhalo05_llm_glu_pack_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen25_3b/xconucstrhalo05_llm_glu_pack_on_1/summary.json) |
| Gemma-3-12B LLM (3968 tokens) | 3,685.17 | 3,642.82 | 42.36 | +1.15% | gain | 0.24% / 0.25% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma3_12b/xconucstrhalo05_llm_glu_pack_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma3_12b/xconucstrhalo05_llm_glu_pack_on_1/summary.json) |


#### glu_16bit: Fuse standalone GLU and BLAS input conversion

Apply a standalone GLU while converting its result to the BLAS input storage type.

Target: Eligible FP16/BF16 or selected Q6_K BLAS down projections. Both MMDQ and parent FFN fusion are disabled for isolation.

| Control | Type | Unset default | Off | On | Source |
| --- | --- | --- | --- | --- | --- |
| `GGML_CUDA_FUSE_GLU_16BIT` | runtime | 1 | 0 | 1 | [ggml-cuda.cu](../../ggml/src/ggml-cuda/ggml-cuda.cu#L4150) |


Held fixed for this ablation: `GGML_CUDA_FUSE_CUBLAS_FFN=0`, `GGML_CUDA_MMDQ=0`, `GGML_CUDA_MMDQ_FUSE_NORM=0`, `GGML_CUDA_MMDQ_GLU=0`, `GGML_CUDA_MMDQ_MOE=0`. Other settings come from the model profile.

| Workload | Off ms | On ms | Saved ms | Saved % | Result | Process spread off/on | Raw example |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-VL-3B | 814.87 | 797.22 | 17.65 | +2.17% | gain | 0.15% / 0.01% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen25_3b/xconucstrhalo41_vlm_glu_16bit_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen25_3b/xconucstrhalo41_vlm_glu_16bit_on_1/summary.json) |
| Gemma-3-12B VLM | 668.37 | 670.83 | -2.46 | -0.37% | within variation | 0.25% / 0.02% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma3_12b/xconucstrhalo05_vlm_glu_16bit_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma3_12b/xconucstrhalo05_vlm_glu_16bit_on_1/summary.json) |


#### q8_interleave: Interleave Q4_K activation packing grid

Interleave the Q8 packing grid over rows/chunks to improve access scheduling.

Target: Aligned Q4_K MMQ activation packing. Fused packing and MMDQ are disabled so the standalone quantizer is exercised.

| Control | Type | Unset default | Off | On | Source |
| --- | --- | --- | --- | --- | --- |
| `GGML_CUDA_QUANTIZE_K_INTERLEAVE` | runtime | 1 | 0 | 1 | [quantize.cu](../../ggml/src/ggml-cuda/quantize.cu#L698) |


Held fixed for this ablation: `GGML_CUDA_FUSE_GLU_PACK=0`, `GGML_CUDA_FUSE_NORM_PACK=0`, `GGML_CUDA_MMDQ=0`, `GGML_CUDA_MMDQ_FUSE_NORM=0`, `GGML_CUDA_MMDQ_GLU=0`, `GGML_CUDA_MMDQ_MOE=0`. Other settings come from the model profile.

| Workload | Off ms | On ms | Saved ms | Saved % | Result | Process spread off/on | Raw example |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-3B LLM (3968 tokens) | 1,042.31 | 1,028.87 | 13.44 | +1.29% | gain | 0.04% / 0.01% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen25_3b/xconucstrhalo05_llm_q8_interleave_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen25_3b/xconucstrhalo05_llm_q8_interleave_on_1/summary.json) |
| Qwen3-4B LLM (3968 tokens) | 1,406.93 | 1,407.69 | -0.76 | -0.05% | within variation | 0.02% / 0.12% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen3_4b/xconucstrhalo05_llm_q8_interleave_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen3_4b/xconucstrhalo05_llm_q8_interleave_on_1/summary.json) |


#### copy_vec8: Vectorize strided FP32 to FP16 copies

Read eight contiguous FP32 values at a time during an eligible strided FP32-to-FP16 copy.

Target: Aligned large vision K/V copies with a contiguous destination. Early K/V casting is disabled to retain this copy pattern.

| Control | Type | Unset default | Off | On | Source |
| --- | --- | --- | --- | --- | --- |
| `GGML_CUDA_CPY_F32_F16_VEC8` | runtime | 1 | 0 | 1 | [cpy.cu](../../ggml/src/ggml-cuda/cpy.cu#L519) |


Held fixed for this ablation: `MTMD_QWEN25VL_KV_CAST_EARLY=0`, `MTMD_VISION_KV_CAST_EARLY=0`. Other settings come from the model profile.

| Workload | Off ms | On ms | Saved ms | Saved % | Result | Process spread off/on | Raw example |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Gemma-3-4B VLM | 486.17 | 480.26 | 5.92 | +1.22% | gain | 0.06% / 0.44% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma3/xconucstrhalo05_vlm_copy_vec8_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma3/xconucstrhalo05_vlm_copy_vec8_on_1/summary.json) |
| Qwen2.5-VL-3B | 692.50 | 681.23 | 11.27 | +1.63% | gain | 0.07% / 0.04% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen25_3b/xconucstrhalo41_vlm_copy_vec8_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen25_3b/xconucstrhalo41_vlm_copy_vec8_on_1/summary.json) |


#### bias_vector: Vectorize bias/output conversion

Vectorize eligible bias and output-conversion work.

Target: Surviving bias/conversion epilogues. A model dominated by fused FFNs may leave little work for this switch.

| Control | Type | Unset default | Off | On | Source |
| --- | --- | --- | --- | --- | --- |
| `GGML_CUDA_BIAS_VEC` | runtime | 1 | 0 | 1 | [convert.cu](../../ggml/src/ggml-cuda/convert.cu#L597) |


Context: the combined model profile; only the listed control changes.

| Workload | Off ms | On ms | Saved ms | Saved % | Result | Process spread off/on | Raw example |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Gemma-3-4B VLM | 480.34 | 477.84 | 2.50 | +0.52% | within variation | 0.41% / 0.16% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma3/xconucstrhalo05_vlm_bias_vector_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma3/xconucstrhalo05_vlm_bias_vector_on_1/summary.json) |
| Qwen2.5-VL-3B | 679.63 | 678.88 | 0.75 | +0.11% | within variation | 0.25% / 0.18% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen25_3b/xconucstrhalo41_vlm_bias_vector_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen25_3b/xconucstrhalo41_vlm_bias_vector_on_1/summary.json) |


#### gelu_vector: Vectorize GELU/bias/input conversion

Vectorize GELU/bias and input conversion.

Target: Eligible GELU vision FFN producer paths with aligned vectors.

| Control | Type | Unset default | Off | On | Source |
| --- | --- | --- | --- | --- | --- |
| `GGML_CUDA_GELU_BIAS_VEC` | runtime | 1 | 0 | 1 | [unary.cu](../../ggml/src/ggml-cuda/unary.cu#L485) |


Context: the combined model profile; only the listed control changes.

| Workload | Off ms | On ms | Saved ms | Saved % | Result | Process spread off/on | Raw example |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Gemma-3-4B VLM | 480.52 | 479.37 | 1.15 | +0.24% | within variation | 0.34% / 0.20% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma3/xconucstrhalo05_vlm_gelu_vector_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma3/xconucstrhalo05_vlm_gelu_vector_on_1/summary.json) |
| Gemma-3-12B VLM | 573.71 | 571.55 | 2.16 | +0.38% | within variation | 0.21% / 0.37% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma3_12b/xconucstrhalo05_vlm_gelu_vector_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma3_12b/xconucstrhalo05_vlm_gelu_vector_on_1/summary.json) |


#### convert_fp16_vector: Pair FP32/FP16 conversions

Convert FP32/FP16 values in aligned pairs.

Target: Eligible contiguous conversions on gfx1151. The benefit depends on how many conversions the larger fusions leave.

| Control | Type | Unset default | Off | On | Source |
| --- | --- | --- | --- | --- | --- |
| `GGML_CUDA_CONVERT_FP16_VEC2` | runtime | 1 | 0 | 1 | [convert.cu](../../ggml/src/ggml-cuda/convert.cu#L661) |


Context: the combined model profile; only the listed control changes.

| Workload | Off ms | On ms | Saved ms | Saved % | Result | Process spread off/on | Raw example |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-VL-3B | 686.62 | 680.17 | 6.46 | +0.94% | within variation | 0.08% / 0.18% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen25_3b/xconucstrhalo41_vlm_convert_fp16_vector_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen25_3b/xconucstrhalo41_vlm_convert_fp16_vector_on_1/summary.json) |
| Gemma-3-12B VLM | 574.91 | 568.85 | 6.07 | +1.06% | gain | 0.27% / 0.02% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma3_12b/xconucstrhalo05_vlm_convert_fp16_vector_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma3_12b/xconucstrhalo05_vlm_convert_fp16_vector_on_1/summary.json) |


#### convert_clamp_vector: Pair clamped 16-bit conversions

Convert and clamp aligned pairs of values per operation.

Target: Clamped 16-bit Gemma-4 conversion paths. This is narrower than enabling the whole clamp fusion.

| Control | Type | Unset default | Off | On | Source |
| --- | --- | --- | --- | --- | --- |
| `GGML_CUDA_CONVERT_CLAMP_VEC2` | runtime | 1 | 0 | 1 | [convert.cu](../../ggml/src/ggml-cuda/convert.cu#L627) |


Context: the combined model profile; only the listed control changes.

| Workload | Off ms | On ms | Saved ms | Saved % | Result | Process spread off/on | Raw example |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Gemma-4-26B-A4B VLM | 602.33 | 598.81 | 3.53 | +0.59% | within variation | 0.29% / 0.03% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma4/xconucstrhalo41_vlm_convert_clamp_vector_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma4/xconucstrhalo41_vlm_convert_clamp_vector_on_1/summary.json) |
| Gemma-4-E2B (BF16 projector) VLM | 259.17 | 259.40 | -0.22 | -0.09% | within variation | 0.03% / 0.53% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma4_e2b/xconucstrhalo05_vlm_convert_clamp_vector_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma4_e2b/xconucstrhalo05_vlm_convert_clamp_vector_on_1/summary.json) |


#### plain_ffn: Fuse plain GELU vision FFN

Recognize a plain GELU FFN with bias and optional residual, not just a two-branch gated FFN.

Target: Gemma-3 vision FFNs. Keeps the hidden activation in the BLAS storage type between the two matrix multiplies.

| Control | Type | Unset default | Off | On | Source |
| --- | --- | --- | --- | --- | --- |
| `GGML_CUDA_FUSE_CUBLAS_FFN_ACT` | runtime | 1 | 0 | 1 | [ggml-cuda.cu](../../ggml/src/ggml-cuda/ggml-cuda.cu#L3797) |


Context: the combined model profile; only the listed control changes.

| Workload | Off ms | On ms | Saved ms | Saved % | Result | Process spread off/on | Raw example |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Gemma-3-4B VLM | 513.75 | 479.40 | 34.35 | +6.69% | gain | 0.02% / 0.12% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma3/xconucstrhalo05_vlm_plain_ffn_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma3/xconucstrhalo05_vlm_plain_ffn_on_1/summary.json) |
| Gemma-3-12B VLM | 605.28 | 570.98 | 34.30 | +5.67% | gain | 0.08% / 0.26% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma3_12b/xconucstrhalo05_vlm_plain_ffn_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma3_12b/xconucstrhalo05_vlm_plain_ffn_on_1/summary.json) |


### Normalization and RoPE

#### rms_vector: Vector RMSNorm kernel

Use a vectorized RMSNorm implementation to reduce scalar load/store and indexing overhead.

Target: Language or vision RMSNorm with supported width/alignment. A fused normalization producer can bypass standalone RMSNorm.

| Control | Type | Unset default | Off | On | Source |
| --- | --- | --- | --- | --- | --- |
| `GGML_CUDA_RMS_NORM_VEC` | runtime | 1 | 0 | 1 | [norm.cu](../../ggml/src/ggml-cuda/norm.cu#L421) |


Context: the combined model profile; only the listed control changes.

| Workload | Off ms | On ms | Saved ms | Saved % | Result | Process spread off/on | Raw example |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-4B LLM (3968 tokens) | 1,176.82 | 1,172.60 | 4.22 | +0.36% | within variation | 0.05% / 0.15% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen3_4b/xconucstrhalo05_llm_rms_vector_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen3_4b/xconucstrhalo05_llm_rms_vector_on_1/summary.json) |
| Gemma-3-12B LLM (3968 tokens) | 3,037.63 | 3,219.36 | -181.73 | -5.98% | within variation | 0.80% / 10.41% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma3_12b/xconucstrhalo05_llm_rms_vector_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma3_12b/xconucstrhalo05_llm_rms_vector_on_1/summary.json) |
| Qwen3.5-35B-A3B VLM | 351.57 | 346.98 | 4.59 | +1.30% | gain | 0.12% / 0.03% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen35/xconucstrhalo41_vlm_rms_vector_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen35/xconucstrhalo41_vlm_rms_vector_on_1/summary.json) |


#### norm_small: Tune normalization block sizes

Use smaller thread blocks for selected normalization widths.

Target: RDNA3.5 small-row normalization. The vector RMSNorm path is disabled in this comparison to expose block sizing.

| Control | Type | Unset default | Off | On | Source |
| --- | --- | --- | --- | --- | --- |
| `GGML_CUDA_NORM_SMALL_BLOCK` | runtime | 1 | 0 | 1 | [quantize.cu](../../ggml/src/ggml-cuda/quantize.cu#L618) |


Held fixed for this ablation: `GGML_CUDA_RMS_NORM_VEC=0`. Other settings come from the model profile.

| Workload | Off ms | On ms | Saved ms | Saved % | Result | Process spread off/on | Raw example |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Gemma-3-4B VLM | 485.98 | 481.12 | 4.85 | +1.00% | within variation | 0.02% / 0.12% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma3/xconucstrhalo05_vlm_norm_small_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma3/xconucstrhalo05_vlm_norm_small_on_1/summary.json) |
| Qwen2.5-VL-3B | 687.60 | 683.44 | 4.16 | +0.60% | within variation | 0.29% / 0.09% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen25_3b/xconucstrhalo41_vlm_norm_small_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen25_3b/xconucstrhalo41_vlm_norm_small_on_1/summary.json) |


#### norm_rope_small: Tune fused norm-RoPE blocks

Use a smaller block for supported fused RMSNorm/scale/RoPE rows.

Target: D128 with enough channels. One-warp norm/RoPE is disabled because it overrides this block policy.

| Control | Type | Unset default | Off | On | Source |
| --- | --- | --- | --- | --- | --- |
| `GGML_CUDA_NORM_ROPE_SMALL_BLOCK` | runtime | 1 | 0 | 1 | [rope.cu](../../ggml/src/ggml-cuda/rope.cu#L871) |


Held fixed for this ablation: `GGML_CUDA_NORM_ROPE_WARP=0`. Other settings come from the model profile.

| Workload | Off ms | On ms | Saved ms | Saved % | Result | Process spread off/on | Raw example |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-4B LLM (3968 tokens) | 1,194.52 | 1,185.68 | 8.84 | +0.74% | within variation | 0.11% / 0.03% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen3_4b/xconucstrhalo05_llm_norm_rope_small_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen3_4b/xconucstrhalo05_llm_norm_rope_small_on_1/summary.json) |


#### norm_rope_warp: Use one warp for small fused norm-RoPE

Assign one warp to a short fused normalization/RoPE row.

Target: Small Q/K head dimensions, represented by Qwen3 D128. Reduces idle threads and synchronization overhead.

| Control | Type | Unset default | Off | On | Source |
| --- | --- | --- | --- | --- | --- |
| `GGML_CUDA_NORM_ROPE_WARP` | runtime | 0 | 0 | 1 | [rope.cu](../../ggml/src/ggml-cuda/rope.cu#L865) |


Context: the combined model profile; only the listed control changes.

| Workload | Off ms | On ms | Saved ms | Saved % | Result | Process spread off/on | Raw example |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-4B LLM (3968 tokens) | 1,186.90 | 1,171.80 | 15.10 | +1.27% | gain | 0.09% / 0.05% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen3_4b/xconucstrhalo05_llm_norm_rope_warp_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen3_4b/xconucstrhalo05_llm_norm_rope_warp_on_1/summary.json) |
| Qwen3-8B LLM (3968 tokens) | 2,076.13 | 2,058.72 | 17.41 | +0.84% | within variation | 0.62% / 0.12% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen3_8b/xconucstrhalo05_llm_norm_rope_warp_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen3_8b/xconucstrhalo05_llm_norm_rope_warp_on_1/summary.json) |


#### norm_rope_2axis: Fuse both Gemma vision RoPE axes

Fuse normalization and the two sequential vision RoPE axes.

Target: Gemma vision RMSNorm -> scale -> RoPE -> RoPE patterns. Preserves both coordinate transformations.

| Control | Type | Unset default | Off | On | Source |
| --- | --- | --- | --- | --- | --- |
| `GGML_CUDA_FUSE_NORM_ROPE_2AXIS` | runtime | 1 | 0 | 1 | [ggml-cuda.cu](../../ggml/src/ggml-cuda/ggml-cuda.cu#L4050) |


Context: the combined model profile; only the listed control changes.

| Workload | Off ms | On ms | Saved ms | Saved % | Result | Process spread off/on | Raw example |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Gemma-4-26B-A4B VLM | 610.73 | 600.56 | 10.17 | +1.67% | gain | 0.06% / 0.08% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma4/xconucstrhalo41_vlm_norm_rope_2axis_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma4/xconucstrhalo41_vlm_norm_rope_2axis_on_1/summary.json) |


#### rope_blocks: Tune standalone RoPE blocks

Reduce standalone RoPE block size for selected small head dimensions.

Target: Language NeoX and vision/multiaxis RoPE. The explicit ROPE_BLOCK override affects its supported paths; the small-block gate also controls vision/multiaxis defaults.

| Control | Type | Unset default | Off | On | Source |
| --- | --- | --- | --- | --- | --- |
| `GGML_CUDA_ROPE_SMALL_BLOCK` | runtime | 1 | 0 | 1 | [rope.cu](../../ggml/src/ggml-cuda/rope.cu#L437) |
| `GGML_CUDA_ROPE_BLOCK` | runtime | automatic (256 generic; 64 for selected RDNA3.5 heads) | 256 | 64 | [rope.cu](../../ggml/src/ggml-cuda/rope.cu#L384) |


Context: the combined model profile; only the listed control changes.

| Workload | Off ms | On ms | Saved ms | Saved % | Result | Process spread off/on | Raw example |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-3B LLM (3968 tokens) | 830.85 | 827.87 | 2.98 | +0.36% | within variation | 0.11% / 0.27% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen25_3b/xconucstrhalo05_llm_rope_blocks_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen25_3b/xconucstrhalo05_llm_rope_blocks_on_1/summary.json) |
| Qwen2.5-VL-3B | 694.20 | 679.70 | 14.50 | +2.09% | gain | 0.09% / 0.57% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen25_3b/xconucstrhalo41_vlm_rope_blocks_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen25_3b/xconucstrhalo41_vlm_rope_blocks_on_1/summary.json) |
| Gemma-4-26B-A4B VLM | 600.48 | 601.47 | -0.99 | -0.16% | within variation | 0.25% / 0.43% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma4/xconucstrhalo41_vlm_rope_blocks_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma4/xconucstrhalo41_vlm_rope_blocks_on_1/summary.json) |


### MoE and graph

#### moe_reduce: Vector expert weighted reduction

Reduce expert outputs with four-value vector operations.

Target: Weighted sum after MoE expert computation, with supported alignment. Does not change expert selection.

| Control | Type | Unset default | Off | On | Source |
| --- | --- | --- | --- | --- | --- |
| `GGML_CUDA_MOE_REDUCE_X4` | runtime | 0 | 0 | 1 | [moe-weighted-reduction.cu](../../ggml/src/ggml-cuda/moe-weighted-reduction.cu#L77) |


Context: the combined model profile; only the listed control changes.

| Workload | Off ms | On ms | Saved ms | Saved % | Result | Process spread off/on | Raw example |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3.5-35B-A3B VLM | 352.22 | 347.17 | 5.05 | +1.43% | gain | 0.16% / 0.01% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen35/xconucstrhalo41_vlm_moe_reduce_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen35/xconucstrhalo41_vlm_moe_reduce_on_1/summary.json) |
| Gemma-4-26B-A4B VLM | 601.91 | 600.02 | 1.89 | +0.31% | within variation | 0.08% / 0.09% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma4/xconucstrhalo41_vlm_moe_reduce_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma4/xconucstrhalo41_vlm_moe_reduce_on_1/summary.json) |


#### topk_alloc: Allocate fused top-k temporaries once

Allocate temporary buffers for fused top-k once rather than repeatedly during graph execution.

Target: MoE routing graphs. Reduces allocation/launch setup overhead, not the number of experts evaluated.

| Control | Type | Unset default | Off | On | Source |
| --- | --- | --- | --- | --- | --- |
| `GGML_CUDA_TOPK_ALLOC` | runtime | 1 | 0 | 1 | [ggml-cuda.cu](../../ggml/src/ggml-cuda/ggml-cuda.cu#L5393) |


Context: the combined model profile; only the listed control changes.

| Workload | Off ms | On ms | Saved ms | Saved % | Result | Process spread off/on | Raw example |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3.5-35B-A3B VLM | 349.80 | 347.20 | 2.59 | +0.74% | within variation | 0.10% / 0.13% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen35/xconucstrhalo41_vlm_topk_alloc_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen35/xconucstrhalo41_vlm_topk_alloc_on_1/summary.json) |
| Gemma-4-26B-A4B VLM | 600.08 | 598.77 | 1.32 | +0.22% | within variation | 0.14% / 0.12% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma4/xconucstrhalo41_vlm_topk_alloc_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma4/xconucstrhalo41_vlm_topk_alloc_on_1/summary.json) |


#### gdn_cu: Schedule GDN blocks in CU mode

Use CU-mode scheduling for gated-delta-net blocks when enough independent blocks exist.

Target: Qwen3.5/3.6 hybrid recurrent layers. The existing block-count threshold is retained; undersized grids still use WGP mode.

| Control | Type | Unset default | Off | On | Source |
| --- | --- | --- | --- | --- | --- |
| `GGML_CUDA_GDN_CU_MODE` | runtime | 1 | 0 | 1 | [gated_delta_net_chunked.cu](../../ggml/src/ggml-cuda/gated_delta_net_chunked.cu#L437) |


Context: the combined model profile; only the listed control changes.

| Workload | Off ms | On ms | Saved ms | Saved % | Result | Process spread off/on | Raw example |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3.6-27B LLM (3968 tokens) | 8,802.64 | 8,901.17 | -98.53 | -1.12% | within variation | 2.67% / 0.70% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen36_27b/xconucstrhalo05_llm_gdn_cu_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen36_27b/xconucstrhalo05_llm_gdn_cu_on_1/summary.json) |
| Qwen3.5-35B-A3B VLM | 349.51 | 347.64 | 1.88 | +0.54% | within variation | 0.05% / 0.18% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen35/xconucstrhalo41_vlm_gdn_cu_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen35/xconucstrhalo41_vlm_gdn_cu_on_1/summary.json) |


#### crop_last: Crop final Qwen3.5 layer to requested output rows

After the last attention layer, retain only rows requested for output before computing the final FFN.

Target: Qwen3.5 dense/MoE graphs. Full sequence rows remain available where recurrent/MTP requirements demand them. Off moves row selection after the final FFN.

| Control | Type | Unset default | Off | On | Source |
| --- | --- | --- | --- | --- | --- |
| `LLAMA_QWEN35_CROP_LAST` | runtime | 1 | 0 | 1 | [qwen35.cpp](../../src/models/qwen35.cpp#L157) |


Context: the combined model profile; only the listed control changes.

| Workload | Off ms | On ms | Saved ms | Saved % | Result | Process spread off/on | Raw example |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3.6-27B LLM (3968 tokens) | 8,861.85 | 8,941.34 | -79.48 | -0.90% | within variation | 0.47% / 1.27% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen36_27b/xconucstrhalo05_llm_crop_last_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen36_27b/xconucstrhalo05_llm_crop_last_on_1/summary.json) |
| Qwen3.5-35B-A3B VLM | 349.58 | 347.55 | 2.03 | +0.58% | within variation | 0.14% / 0.03% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen35/xconucstrhalo41_vlm_crop_last_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen35/xconucstrhalo41_vlm_crop_last_on_1/summary.json) |


### CPU preprocessing

#### image_threads: Parallelize image preprocessing

Split image resize/filter output rows across CPU threads.

Target: CPU preprocessing before the vision tower. Tests use eight threads versus one with identical image bytes and resize settings.

| Control | Type | Unset default | Off | On | Source |
| --- | --- | --- | --- | --- | --- |
| `MTMD_IMAGE_THREADS` | runtime | 1 | 1 | 8 | [mtmd-image.cpp](../../tools/mtmd/mtmd-image.cpp#L363) |


Context: the combined model profile; only the listed control changes.

| Workload | Off ms | On ms | Saved ms | Saved % | Result | Process spread off/on | Raw example |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Gemma-3-4B VLM | 482.68 | 478.77 | 3.92 | +0.81% | within variation | 0.02% / 0.03% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma3/xconucstrhalo05_vlm_image_threads_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma3/xconucstrhalo05_vlm_image_threads_on_1/summary.json) |
| Qwen2.5-VL-3B | 684.83 | 677.50 | 7.33 | +1.07% | gain | 0.09% / 0.09% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen25_3b/xconucstrhalo41_vlm_image_threads_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen25_3b/xconucstrhalo41_vlm_image_threads_on_1/summary.json) |


#### tokenizer: Fast SentencePiece reverse merge

Use a faster reverse-merge search in SentencePiece tokenization.

Target: Gemma SentencePiece text prompts. A short VLM prompt may spend too little time here for a measurable TTFT change.

| Control | Type | Unset default | Off | On | Source |
| --- | --- | --- | --- | --- | --- |
| `LLAMA_SPM_FAST_REV_MERGE` | runtime | 0 | 0 | 1 | [llama-vocab.cpp](../../src/llama-vocab.cpp#L247) |


Context: the combined model profile; only the listed control changes.

| Workload | Off ms | On ms | Saved ms | Saved % | Result | Process spread off/on | Raw example |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Gemma-3-4B VLM | 478.56 | 478.81 | -0.25 | -0.05% | within variation | 0.37% / 0.50% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma3/xconucstrhalo05_vlm_tokenizer_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma3/xconucstrhalo05_vlm_tokenizer_on_1/summary.json) |
| Gemma-3-12B VLM | 571.50 | 570.95 | 0.55 | +0.10% | within variation | 0.35% / 0.34% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma3_12b/xconucstrhalo05_vlm_tokenizer_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma3_12b/xconucstrhalo05_vlm_tokenizer_on_1/summary.json) |


### MMQ fallback

#### q6_scale: Reuse Q6_K scales inside J64 MMQ tiles

Precompute Q6_K subblock scale products while loading the weight tile and reuse them across the J64 matrix operations.

Target: RDNA3.5 Q6_K MMQ J64 tiles. It spends more shared memory to remove repeated scale conversion/multiplication. MMDQ is disabled for this measurement. Dense MMQ dispatch requires batch <= 256 on gfx1151; both 3968-token and additional 128-token prompt tests are retained.

Execution-path note: The 3968-token rows have microbatch 1024 and select BLAS for dense Q6_K when MMDQ is disabled. They do not measure the scale-reuse kernel. The additional 128-token rows use the eligible Q6_K MMQ path.

| Control | Type | Unset default | Off | On | Source |
| --- | --- | --- | --- | --- | --- |
| `GGML_CUDA_MMQ_Q6_K_SCALE_REUSE` | compile-time | 1 | 0 | 1 | [mmq-load-tiles.cuh](../../ggml/src/ggml-cuda/mmq-load-tiles.cuh#L1517) |


Held fixed for this ablation: `GGML_CUDA_MMDQ=0`, `GGML_CUDA_MMDQ_FUSE_NORM=0`, `GGML_CUDA_MMDQ_GLU=0`, `GGML_CUDA_MMDQ_MOE=0`. Other settings come from the model profile.

| Workload | Off ms | On ms | Saved ms | Saved % | Result | Process spread off/on | Raw example |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Gemma-3-12B LLM (3968 tokens) | 3,640.71 | 3,640.22 | 0.49 | +0.01% | within variation | 0.83% / 0.11% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma3_12b/xconucstrhalo05_llm_q6_scale_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma3_12b/xconucstrhalo05_llm_q6_scale_on_1/summary.json) |
| Gemma-3-12B LLM (128 tokens) | 131.15 | 126.54 | 4.61 | +3.52% | gain | 0.14% / 1.56% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma3_12b/xconucstrhalo05_llm_q6_scale_p128_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma3_12b/xconucstrhalo05_llm_q6_scale_p128_on_1/summary.json) |
| Qwen2.5-7B LLM (3968 tokens) | 2,086.35 | 2,083.62 | 2.73 | +0.13% | within variation | 0.39% / 0.08% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen25_7b/xconucstrhalo05_llm_q6_scale_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen25_7b/xconucstrhalo05_llm_q6_scale_on_1/summary.json) |
| Qwen2.5-7B LLM (128 tokens) | 77.55 | 75.66 | 1.89 | +2.44% | gain | 1.25% / 1.84% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen25_7b/xconucstrhalo05_llm_q6_scale_p128_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen25_7b/xconucstrhalo05_llm_q6_scale_p128_on_1/summary.json) |


#### q8_barriers: Remove redundant Q8_0 J112 MMQ barriers

Remove redundant tile barriers in the Q8_0 J112 MMQ loop while keeping the required synchronization.

Target: Q8_0 J112 MMQ. The macro name is historical: value 1 enables barrier removal; 0 restores the extra barriers.

| Control | Type | Unset default | Off | On | Source |
| --- | --- | --- | --- | --- | --- |
| `GGML_CUDA_MMQ_Q8_0_J112_BARRIERS` | compile-time | 1 | 0 | 1 | [mmq.cuh](../../ggml/src/ggml-cuda/mmq.cuh#L129) |


Held fixed for this ablation: `GGML_CUDA_MMDQ=0`, `GGML_CUDA_MMDQ_FUSE_NORM=0`, `GGML_CUDA_MMDQ_GLU=0`, `GGML_CUDA_MMDQ_MOE=0`, `GGML_CUDA_MMDQ_Q8_0=0`. Other settings come from the model profile.

| Workload | Off ms | On ms | Saved ms | Saved % | Result | Process spread off/on | Raw example |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3.5-35B-A3B VLM | 348.46 | 347.92 | 0.54 | +0.15% | within variation | 0.10% / 0.01% | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen35/xconucstrhalo41_vlm_q8_barriers_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen35/xconucstrhalo41_vlm_q8_barriers_on_1/summary.json) |


## Targeted confirmation after extra warmup

The large Gemma-3-12B and Qwen3.6-27B runs showed material variation. Read-only telemetry captures observed automatic clocks varying; temperatures reached 93 C during GPU load in the confirmation capture. No device settings were changed, and these observations do not establish the cause of the timing drift. These confirmation runs add five untimed full-prompt repetitions in each process before retaining the next five samples. Both states receive the same treatment. The original rows remain above; the extra repetitions are saved in raw results and are not selected according to their latency.

| Optimization | Workload | Off ms | On ms | Saved % | Result | Raw example |
| --- | --- | --- | --- | --- | --- | --- |
| mmdq_q5_k | Qwen3.6-27B LLM (3968 tokens) | 7,876.91 | 7,334.36 | +6.89% | within variation | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen36_27b/xconucstrhalo05_llm_confirm_mmdq_q5_k_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen36_27b/xconucstrhalo05_llm_confirm_mmdq_q5_k_on_1/summary.json) |
| mmdq_geglu | Gemma-3-12B LLM (3968 tokens) | 3,316.94 | 2,945.15 | +11.21% | gain | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma3_12b/xconucstrhalo05_llm_confirm_mmdq_geglu_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma3_12b/xconucstrhalo05_llm_confirm_mmdq_geglu_on_1/summary.json) |
| rms_vector | Gemma-3-12B LLM (3968 tokens) | 3,275.59 | 3,348.97 | -2.24% | within variation | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma3_12b/xconucstrhalo05_llm_confirm_rms_vector_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/gemma3_12b/xconucstrhalo05_llm_confirm_rms_vector_on_1/summary.json) |
| mmdq_swizzle | Qwen3.6-27B LLM (3968 tokens) | 7,351.72 | 8,850.52 | -20.39% | slower | [off](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen36_27b/xconucstrhalo05_llm_confirm_mmdq_swizzle_off_0/summary.json), [on](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs/qwen36_27b/xconucstrhalo05_llm_confirm_mmdq_swizzle_on_1/summary.json) |


## Existing tuning and diagnostic switches

The following existing switches select variants inside an already measured family or enable unselected experiments. They remain available, but this refresh does not claim a separate gain for every possible parameter value.

| Switches | Purpose and treatment |
| --- | --- |
| `GGML_CUDA_MMDQ_CFG`, `GGML_CUDA_MMDQ_ID_CFG`, `GGML_CUDA_MMDQ_MOE_CFG` | Select compiled matrix tile shapes. Automatic/default selection is part of the MMDQ-family on/off result; arbitrary tile IDs are not independent optimizations. |
| `GGML_CUDA_MMDQ_MIN_BATCH`, `GGML_CUDA_MMDQ_Q8_0_MIN_BATCH`, `GGML_CUDA_MMDQ_ID_MIN_BATCH`, `GGML_CUDA_MMDQ_MOE_MIN_BATCH` | Eligibility thresholds. Defaults are retained; qualifying Q6_K and Q8_0 workloads were added instead of forcing low-batch dispatch. |
| `GGML_CUDA_MMDQ_ID` | Older opt-in generic expert path. The measured combined profile uses the newer MMDQ_MOE path. It is already gated, but no separate deployed gain is claimed for this unselected alternative. |
| `GGML_CUDA_MMDQ_MOE_ALL` | Test override that bypasses expert shape heuristics. Left off. |
| `GGML_CUDA_DISABLE_FUSION`, `GGML_CUDA_CUBLAS_COMPUTE_TYPE` | Global diagnostic/precision overrides. Left at normal settings; changing all fusions or BLAS precision would confound individual attribution. |
| `MTMD_QWEN25VL_PAD_MODE`, `MTMD_GEMMA4_PAD_MODE` | Image shape/padding policy, fixed by the workload profile. Not counted as a kernel speedup. |


## Measurement conditions and model coverage

LLMs: `llama-bench`, 128/3968-token full-branch comparisons, 3968-token per-control ablations plus targeted 128-token Q6_K ablations, zero prior context, no generated tokens, full-prompt warmup, eight CPU threads, full GPU offload, flash attention, FP16 K/V, logical batch 4096. Microbatch is fixed per model on both sides. Reported time is prompt processing, not HTTP serving TTFT.

VLMs: `llama-server`, one fixed streaming HTTP client, one request at a time, two warmups per process, five measured requests per process, context 4096, output cap 32, checkpoint limit **0**, no prefix reuse, native image/text order and boundaries, FP16 K/V. TTFT ends at the first nonempty content chunk. The response is drained before the next request. Gemma-4 image maximum is fixed at 280 on both sides; Gemma-3 keeps the same image/crop policy. Image resizing is never counted as a kernel speedup.

Each pair uses off/on/on/off process order (gfx11/optimized/optimized/gfx11 for branch comparisons). The reported value is the median of ten retained samples per state. A gain/slower label requires a difference greater than both 1% and each state's process-median spread, with nonoverlapping process-median ranges. Otherwise the result is "within variation". This is a conservative consistency heuristic, not a formal confidence interval. Full samples and the two process medians are in [analysis.json](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/analysis.json).

Halo05 and Halo41 both have AMD Radeon 8060S / gfx1151 GPUs. Each comparison stays on one board. VLM ablations are split by model: Gemma-3-4B/12B and the two BF16 additions use Halo05; Qwen2.5-VL, Qwen3-VL, Gemma-4-26B and Qwen3.5 use Halo41. The branch baseline and the independent ablation for a model may use different boards, but no off/on pair crosses boards. Workers use a board lock, check foreign GPU/CPU activity, and exclude contended measurements. Some Halo41 runs loaded weights from NFS while local scratch was full; model loading is excluded, and GPU offload and warmup finish before timing. Board/kernel details are saved in [environment](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/environment).

| Model/workload | Microbatch | Language matrix storage | Vision matrix storage | Original image pixels | Profile ID |
| --- | --- | --- | --- | --- | --- |
| Qwen2.5-0.5B LLM | 2048 | Q4_K, Q5_0, Q6_K, Q8_0 | - | - | llm_qwen25_05b |
| Qwen2.5-3B LLM | 2048 | Q4_K, Q6_K | - | - | llm_qwen25_3b |
| Qwen2.5-7B LLM | 1024 | Q4_K, Q6_K | - | - | llm_qwen25_7b |
| Qwen3-4B LLM | 1024 | Q4_K, Q6_K | - | - | llm_qwen3_4b |
| Qwen3-8B LLM | 1024 | Q4_K, Q6_K | - | - | llm_qwen3_8b |
| Llama-2-7B LLM | 1024 | Q4_K, Q6_K | - | - | llm_llama2_7b |
| Gemma-2B LLM | 1024 | Q4_K, Q6_K | - | - | llm_gemma_2b |
| Gemma-3-12B LLM | 1024 | Q4_K, Q6_K | - | - | llm_gemma3_12b |
| Qwen3.6-27B LLM | 2048 | F32, Q4_K, Q5_K, Q6_K | - | - | llm_qwen36_27b |
| Qwen2.5-VL-3B | 2048 | Q4_K, Q6_K | F16 | 1024 x 800 | vlm_qwen25_3b |
| Qwen2.5-VL-7B | 2048 | Q4_K, Q6_K | F16 | 1024 x 800 | vlm_qwen25_7b |
| Qwen3-VL-4B VLM | 512 | Q4_K, Q6_K | F16 | 1024 x 800 | vlm_qwen3vl |
| Gemma-3-4B VLM | 512 | Q4_K, Q6_K | F16 | 1024 x 800 | vlm_gemma3 |
| Gemma-3-12B VLM | 512 | Q4_K, Q6_K | F16 | 640 x 480 | vlm_gemma3_12b |
| Gemma-4-26B-A4B VLM | 512 | F32, Q4_K, Q5_1, Q8_0 | F16 | 1024 x 800 | vlm_gemma4 |
| Qwen3.5-35B-A3B VLM | 512 | F32, Q4_K, Q5_K, Q6_K, Q8_0 | F16 | 640 x 480 | vlm_qwen35 |
| Gemma-4-E2B (BF16 projector) VLM | 512 | BF16, Q4_0, Q8_0 | BF16 | 1024 x 800 | vlm_gemma4_e2b |
| Qwen3-Omni-30B-A3B (BF16 projector) VLM | 512 | F32, Q4_K, Q6_K | BF16 | 1024 x 800 | vlm_omni |
| Qwen3.5-35B-A3B LLM | 1024 | F32, Q4_K, Q5_K, Q6_K, Q8_0 | - | - | llm_qwen35 |
| Gemma-4-26B-A4B LLM | 1024 | F32, Q4_K, Q5_1, Q8_0 | - | - | llm_gemma4 |


GGUF tensor headers were audited directly: the original seven projectors use FP16. Actual BF16 coverage comes from the added Gemma-4-E2B and Qwen3-Omni projectors. The Gemma-E2B request uses the same frozen image/text fixture as Gemma-4-26B. See `tensor_audit.json` for matrix types and shapes. The sample covers dense and MoE models, recurrent hybrid layers, Q4_0/Q4_K/Q5_0/Q5_1/Q5_K/Q6_K/Q8_0 matrices, FP16/BF16 vision weights, D64/D72/D80 vision heads, and D64/D128/D256/D512 language heads. The exact GGUF/projector paths, model hashes and input hashes are in [models.json](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/models.json), [commands.json](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/commands.json), [model_cache](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/model_cache) and [request_audit.json](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/request_audit.json). `commands.json` is the frozen input template; the effective command in each run's metadata includes the checkpoint, microbatch and runtime overrides.

## Limits, overlap and accuracy

An optimization can be valid but show little total-time gain because the model has no eligible tensors, a newer parent fusion already removed the work, the remaining operation is small, or its savings are below run variation. In particular, optional QKV concatenation is a Gemma-3 loader path; the Qwen towers already have combined QKV weights. Q8_0 MMDQ has a default 576-token batch threshold, so the 512-token native VLM chunks bypass it. Additional 3968-token text-only ablations on the same Qwen3.5 and Gemma-4 language weights use microbatch 1024 to measure that eligible path. These two diagnostic workloads have no extra gfx11 baseline rows. The tables report measured totals without inventing a kernel speedup for these cases.

Q6_K MMQ dispatch on gfx1151 is limited to dense batches of at most 256 tokens. The long-prompt Q6_K scale-reuse rows can therefore be inactive when MMDQ is off and large batches use BLAS; the additional 128-token rows test the eligible MMQ path. MMQ packing/Q6_K and legacy-attention rows intentionally select their older execution paths. Their savings are useful for workloads that still use those paths; they are not extra savings to add on top of the MMDQ/dedicated-attention profile. The previously developed Q4_K pipeline and inherited MMQ tile policies are already part of the gfx11 ancestry and are not credited again as new branch gains.

Correctness checks reuse `test-backend-ops`: enabled and fallback composite configurations each passed 281/281 cases. Targeted Q6_K cases passed 2/2 per build; the targeted Q8_0 case passed 1/1 per build. Targeted D72/D80/D512 attention checks passed in the legacy, dedicated and D80-off configurations for supported combinations; unsupported combinations were skipped. The raw logs identify the executed shapes.

The tests establish agreement within existing operation tolerances, not bitwise equality or model-wide accuracy. Some fusions change intermediate rounding. Generated text differed between branch baselines for: Qwen2.5-VL-7B, Qwen3-VL-4B VLM, Gemma-4-26B-A4B VLM, Qwen3.5-35B-A3B VLM, Gemma-4-E2B (BF16 projector) VLM, Qwen3-Omni-30B-A3B (BF16 projector) VLM. Per-ablation output agreement is recorded in `analysis.json`; a few deterministic short responses are not an accuracy benchmark. Prompt-position equality is checked separately from output equality.

The completed-run audit passed for 167 comparisons / 668 process starts / 3340 retained samples. It checks board identity, model hashes, input-request hashes, prompt positions, effective command arguments, expected optimization values, off/on/on/off order, sample counts, warmup policy and process cleanup. The source patch also matches the measured build snapshot. See [run_audit.json](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/run_audit.json).

Padding/budget changes, checkpoint changes, experimental mixed text/image batching, and model correctness fixes are not counted as isolated kernel optimizations in this report. The tested workload controls remain fixed. Other GPUs and arbitrary shapes may select existing fallbacks; these measurements are specifically for gfx1151.

Halo41 initially had a full scratch disk. During the sweep, 85.5 GB of old inactive model-cache copies were reclaimed after checking manifests and unchanged originals. Subsequent task-local copies are SHA-256 verified; loading remains outside the measured interval. See [scratch cleanup record](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/environment/scratch_reclaim_applied.json).

## Reproduction and provenance

| Reference | Revision |
| --- | --- |
| Optimized branch before controls | `d60e0ce79f45362ab0a59432fa278c0e1a8ab6e4` |
| ROCm llama.cpp current gfx11 | `f70aba1cb60547b13a6d2f1ef70c158dd59ca51e` |
| Historical fork used for inventory | `0363d99d3b276a5cfb8d2ab2940cd1a9a7e1d46b` |
| Jim control-style reference | `75e205b0e0400359fa113d41eac0b00b6f3d31ad` |
| Control patch SHA-256 | `83327685b732e27ec6cfc2d571053827dbdded164d03a89b73292001db2f8468` |


Measurements used the pre-control revision above plus the saved control patch on `lichang.rdna35-prefill-quant-rope`. The experiment retains [controls.patch](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/controls.patch), source state, configure commands, build logs and binary hashes. Build settings: Release HIP, gfx1151, VMM off, unsafe math off, ROCm SDK `therock-dist-linux-gfx1151-tests-10.1.0a20260731`. See [controlled build manifest](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/builds/controlled/manifest.json) and corresponding variant manifests for exact commands.

Run a focused comparison on its assigned board. Use a new label prefix to collect fresh samples; omitting the prefix reuses completed experiment runs:

```bash
python3 /proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/suite.py llm --stage ablation --only llm_qwen25_3b --optimization mmdq_glu --label-prefix manual1_
python3 /proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/suite.py vlm --stage ablation --only vlm_gemma3 --optimization vision_attention --label-prefix manual1_
```

Compile-time fallback example:

```bash
python3 /proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/build.py q6_off /proj/gdba/lichang/llama.cpp-prefill-quant-rope --hip-flags=-DGGML_CUDA_MMQ_Q6_K_SCALE_REUSE=0
```

All effective settings and raw samples are in [runs](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/runs). The machine-readable control definitions are [optimizations.json](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/optimizations.json) and [control_catalog.json](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/control_catalog.json). Analysis is generated solely from fresh completed summaries by [analyze.py](/proj/gdba/lichang/regression_results/rdna35_ablation_refresh_20261109/analyze.py). Failed/contended attempts are kept separate. Task-owned model caches are capped at 50 GB; other users' files and processes are left untouched.
