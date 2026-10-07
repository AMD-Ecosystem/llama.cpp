# RDNA3.5 optimization notes

The active implementation is `lichang.rdna35-prefill-quant-rope`. Qwen vision window batching, D64/D80 vision attention, D128 language attention, D512 language attention, FP16/FP32 conversion, RMSNorm, activation quantization, and RoPE changes share this source tree. Earlier experimental worktrees are reference snapshots; changes should be developed and validated here.

## Dispatch and tradeoffs

| Direction | Shared implementation and eligibility | Why it is selective |
| --- | --- | --- |
| Qwen2.5-VL window attention | The existing vision graph groups adjacent windows of equal length and calls the common attention backend. The non-flash path retains the mask. | Window independence comes from the model graph. Gemma global attention cannot be partitioned into independent windows without changing the model. |
| D64/D80 vision attention | One RDNA3.5 rule selects the existing 64-query WMMA kernel for unmasked FP16 self-attention, GQA1, at least 64 queries, and no bias or softcap. | Short windows retain the generic tile. The same kernel supports both head widths; no model-name dispatch is needed. |
| D128 language attention | Existing causal-mask validation, tile tuning, and pending GQA4 wide-tile work remain in place. | The geometry and resource use differ from D256/D512. Mask-prefix shortcuts still validate the actual mask before skipping loads. |
| D512 language attention | Existing 4-query x 8-head WMMA tile, with query data in shared memory on RDNA3.5. Dispatch requires FP16 K/V, GQA8, a compatible padded mask, no softcap, and at least 32 queries. | Enabling WMMA without resource tuning caused register spills and was slower. Shared-query staging reduces spills. The setting is scoped in both host and device selection; other RDNA GPUs retain their configuration. |
| FP16/FP32 conversion | One packed conversion implementation handles two adjacent values per thread, with original casts and a scalar tail. | Requires RDNA3.5, contiguous conversion, alignment, and at least 256 elements. Wider vectors did not consistently help; cold-memory gains are smaller than hot-buffer gains. |
| RMSNorm | The existing 256-thread kernel is selected at width1536 and at least 64 rows on RDNA3.5, including multiply/add variants. | Tiny batches favor the original 1024-thread block. Boundary measurements found the previous 32-row threshold too low for plain RMSNorm. |
| Activation quantization | The existing Q4_K activation packer interleaves feature chunks in its launch grid. Arithmetic and packed output layout stay unchanged. | Row strides divisible by1024 floats showed large gains. An earlier unrestricted extension slowed Qwen7B quantization. Width1536 passed byte comparisons but showed no useful end-to-end gain in the follow-up trial, so it retains ordinary traversal. |
| RoPE | Existing shape-guarded launch tuning remains shared. | The selected D128 geometry does not imply a benefit for Gemma's D256/D512 heads. More blocks or threads can trade scheduling overhead against occupancy. |
| Dense GEMM provider | `ROCBLAS_USE_HIPBLASLT=1` is a runtime choice, using the same source and weights. | It helps the measured E2B workload but is not uniformly faster across models and prompt shapes. It is not hardcoded globally. |

Correctness eligibility and performance eligibility are different. A shape can produce the right answer with both kernels and still be slower with the candidate. Dispatch should depend on hardware, types, layouts, shapes, and dependencies. Model-name branches are unnecessary for these backend changes.

## Why launch interleaving does not transfer uniformly

At 258 rows, the ordinary width1536 quantizer took 21.08 us and the interleaved version took 20.84 us. Width2048 took 40.92 us and 26.33 us respectively. Counting the same logical input/output bytes, ordinary width1536 already achieves about96 GB/s, close to the interleaved width2048 result of103 GB/s; ordinary width2048 achieves about66 GB/s. These are effective rates, not physical DRAM bandwidth.

The width1536-only extension changed E2B median TTFT from 367.514 ms to 367.490 ms in the paired trial (20 samples per configuration). A 0.024 ms difference is below the observed variation. The extension is not enabled.

Earlier hardware counters on the slow aligned shape showed fewer memory-wait cycles after interleaving, with the same arithmetic and nearly unchanged request counts. This supports an access-order-sensitive memory bottleneck. It does not identify a specific DRAM bank, channel, or cache mechanism, and no such cause is claimed for width1536. The existing ordering at that width simply has much less measured inefficiency to remove.

## Fusion results, 2026-10-07

The clamp/conversion, activation/packing, and small elementwise fusions are now implemented in this tree. They extend the existing conversion, quantization, unary, and normalization kernels. The full gate/up matrix fusion was prototyped and measured; it is not enabled because it did not give a consistent gain on the measured workload.

| Direction | Implementation and eligibility | Measured result and tradeoff |
| --- | --- | --- |
| Vision clamp + conversion | Fold FP32 input clamp into conversion to FP16, and output clamp into conversion from the FP16 GEMM result to FP32. Require RDNA3.5, contiguous tensors, and the same default-precision FP16 BLAS dispatch. Consumer checks preserve externally used intermediates. | Removes 226 launches per E2B request. Separate paired stages saved 8.55 ms for output and 5.53 ms for input. The original FP16 GEMM result and clamp/cast order are retained. Precision hints and explicit compute-type overrides fall back. |
| RMSNorm + shared gate/up packing | Normalize, multiply by the norm weight, and write Q8 activation blocks once. Pass that temporary to both existing MMQ projections. Require RDNA3.5, Q4_K/Q5_K weights with the DS4 activation layout, aligned contiguous F32 input, 64-65535 rows, padded width at most 4096, and no extra consumers. | Removes 210 launches across 105 pairs. The separate paired stage saved 2.48 ms. No persistent cache or checkpoint change is involved. Reduction order and FP32 producer rounding are preserved. |
| GEGLU + activation packing | Reuse existing GELU/gate math and write directly to DS4 Q8 for Q4_K/Q5_K, D4 Q8 for Q6_K MMQ, or FP16 for the existing Q6_K/F16 BLAS path. Require compatible contiguous F32 inputs and the existing matrix dispatch. | Removes 105 launches: 54 DS4, 17 D4, and 34 FP16 conversions. The first DS4/FP16 paired stage saved 4.87 ms. The D4 extension covers the actual 252-row Q6_K batch; 258-row batches already use the FP16 route. |
| Small GELU + multiply | Extend the existing unary-multiply implementation with F32 GELU. Permit one intervening VIEW and safe exact in-place reuse; reject extra consumers or partial/strided overlap. | Removes 204 launches. Six GELU calls remain because other real work separates those operations. |
| RMSNorm + weight + residual + scalar multiply | Extend the existing fused normalization kernel with an optional final scalar multiplication. Reuse its type/layout checks and validate output aliases. | Removes 210 launches. This and the small GELU fusion together saved 0.72 ms in their paired stage, close to ordinary run variation. The reduction in launches is directly confirmed. |

Fusion can expose arithmetic to compiler contraction across what used to be a memory boundary. The RMSNorm/GEGLU packing trials initially changed packed bytes. A register-only compiler barrier now materializes the original FP32 value before packing or conversion. It adds no memory round trip or GPU synchronization. The retained paths pass the bitwise producer/packing comparisons.

Output buffer reuse is checked according to the operation. The BLAS route consumes its inputs before the final conversion writes the output. Elementwise fusion permits identical contiguous input/output regions, but not shifted overlap. For in-place normalization, the row reduction reads the row before any element is overwritten. These rules enabled actual model fusions that an overly conservative blanket overlap rejection had prevented.

The small F32 elementwise extensions are generic. The new packing and BLAS fusions use RDNA3.5, type, layout, shape, and dependency guards; none use a model-name branch. Other shapes keep the existing route. Cross-compilation is not a substitute for runtime validation on other GPUs.

## Full gate/up matrix fusion experiment

The prototype reuses the production MMQ tile math, keeps gate and up accumulators live together, and applies GEGLU before writeback. It was tested with production launch bounds, tile widths 32/64/96/128, the actual 252/258-row batches, and warm/cold buffers. The comparison already includes shared activation packing and GEGLU packing in the separate-matrix implementation.

There was no consistent gain. At 258 rows with warm buffers, separate kernels took 492.10 us with FP16 output versus 495.83 us for the best fused tile; DS4 took 488.37 versus 498.98 us. Cold FP16 improved from 527.29 to 511.28 us, but the other cold outputs improved by only 1.86-2.75 us. At 252 rows, the warm fused variants were slower and only cold FP16 improved slightly. This leaves less than about 1 ms of favorable whole-request budget, with offsets from the losing cases.

The resource tradeoff is concrete: tile width 64 increases VGPRs from 193 to 225, and width 96 from 225 to 250. Width 128 reaches 256 VGPRs with 232 bytes of private storage, while the original has 244 VGPRs and no private storage. Narrower tiles avoid spills but sacrifice matrix efficiency. This rejects the tested prototype on this workload; it does not establish that all full matrix fusion designs are slower.

The prototype and measurements remain in the results directory, outside maintained backend source. Checkpoint changes remain paused.

## End-to-end validation

The reference already contains the earlier consolidated attention, conversion, RMSNorm, quantization, and RoPE optimizations. These numbers measure the additional fusion work, not stock upstream llama.cpp.

| Model | Before TTFT | After TTFT | Interpretation |
| --- | ---: | ---: | --- |
| Gemma-4-E2B | 366.21 ms | 342.97 ms | 23.24 ms / 6.35% lower latency with the final `complete` build. |
| Qwen2.5-VL-3B | 963.41 ms | 958.96 ms | Small difference; no material regression. |
| Qwen2.5-VL-7B | 1445.68 ms | 1443.95 ms | Essentially unchanged. |

Each paired result uses ABBA process order, five warmups per process, and 20 measured requests per configuration. The Qwen paired runs used `candidate`, before the final Q6_K GEGLU D4 extension. The final `complete` build also passed one ten-sample run for each Qwen model, at 952.14 and 1435.04 ms respectively. These later single-process runs are build/generation checks, not another paired speedup claim.

Gemma's server-reported prompt-processing median fell from 343.65 to 320.29 ms. The final profile records 6437 -> 5482 kernel launches per request, a reduction of 955 (14.84%). Profiled timings are excluded from the latency table. Individual stage savings come from separate paired experiments and should not be added as an exact decomposition of the final 23.24 ms.

Input: the same 1024x800 JPEG and 2634-character text prompt, 775 total prompt positions for Gemma, `--image-max-tokens 280`, and 1606 positions for Qwen. Gemma's 775 count includes image positions; it is not a text-token count. The models use Q4_K_M GGUF weights and F16 vision projectors. Timing requests disable prompt cache reuse, use temperature 0, and generate one token. Gemma thinking is disabled.

All final paired generation checks and both final-build Qwen checks match the reference's 32 generated token strings, bytes, and chosen-token log probabilities. The validation includes 190 fusion/normalization, 195 dense quantized matrix, and 167 MoE matrix test executions, followed by 34 focused tests after the D4 extension; all passed. This is 586 executions and 541 distinct printed parameter sets. Another 182 conversion/packing cases passed bitwise comparison. Changed backend units compiled for gfx1100 and gfx942; runtime tests ran on gfx1151 only. NVIDIA compilation/runtime and full model-quality evaluations were not run.

The older vLLM reference is 250.84 ms. It was not rerun for this fusion round and uses different weight representation/precision. Relative to that historical measurement, the remaining TTFT difference is about 92.13 ms; this round removes about 20% of the previous 115.37 ms difference. It is not an identical-arithmetic comparison.

## Additional Gemma model tests

The unchanged final fusion build was also compared with the same saved pre-fusion baseline on Gemma-3-4B-IT and Gemma-4-26B-A4B-IT. Each comparison used ABBA process order and 20 measured requests per configuration on Halo41.

| Model | Total prompt positions | Before TTFT | After TTFT | Saving |
| --- | ---: | ---: | ---: | ---: |
| Gemma-3-4B-IT, Q4_K_M | 781 | 1147.18 ms | 1134.11 ms | 13.07 ms / 1.14% |
| Gemma-4-26B-A4B-IT, UD-Q4_K_M | 779 | 942.67 ms | 908.61 ms | 34.07 ms / 3.61% |

Both use the same 1024x800 image and descriptive prompt content. Gemma-3 reuses the previously matched template and newline framing; Gemma-4 uses its native template, thinking disabled, and an image-token limit of 280. Both use F16 vision projectors and `ROCBLAS_USE_HIPBLASLT=1`. The 32-token generation checks match text, token bytes, and chosen-token log probabilities in all eight paired runs. Model loading and profile overhead are excluded from these timing results.

Gemma-3 removes 297 launches through shared norm packing and GEGLU packing/conversion, but its D72 vision attention still takes about 612 ms in summed profile kernel time. Both models have D72 vision heads, outside the existing D64/D80 WMMA dispatch. The 26B profile removes 560 launches: 380 from vision clamp/conversion and 180 from residual scaling. Its UD quantization has Q8_0 shared FFN projections and Q4_K routed gate/up projections, so the new dense norm/GEGLU packing fusions do not execute for this model. Its D72 vision attention remains about 225 ms in summed profile kernel time. See `/proj/gdba/lichang/regression_results/rdna35_fusion_20261007/TRANSFER_REPORT.md` for the final profile attribution, exact weights, raw results, and limitations.

## Build and measurements

The local gfx1151 build is `build-rdna35-gfx1151`. It includes `llama-server`, `llama-bench`, and `test-backend-ops`; no experimental library preloads are required for normal use. The measured E2B setup uses `ROCBLAS_USE_HIPBLASLT=1`.

Consolidation measurements are in `/proj/gdba/lichang/regression_results/rdna35_consolidation_20261007`. The fusion report, exact requests, source/library hashes, per-stage variants, tests, profiles, and raw samples are in `/proj/gdba/lichang/regression_results/rdna35_fusion_20261007/REPORT.md` and its adjacent artifacts. The paired comparison uses the same built server and an immutable saved HIP library for each configuration. The normal build matches the final `complete` snapshot and needs no preload. GPU tests run serially on Halo41; samples overlapping another KFD process group are excluded. No overlap was detected in the final runs. Raw outliers and exploratory numerical differences are retained and described in the report.
