# RDNA3.5 optimization notes

The active implementation is `lichang.rdna35-prefill-quant-rope`. Qwen vision window batching, D64/D72/D80 vision attention, D128 language attention, D512 language attention, FP16/FP32 conversion, RMSNorm, activation quantization, and RoPE changes share this source tree. Earlier experimental worktrees are reference snapshots; changes should be developed and validated here.

## Commit-by-commit record

This record covers all eight source-changing commits on `lichang.rdna35-prefill-quant-rope` after the inherited gfx11 base [`0363d99d3`](https://github.com/AMD-Ecosystem/llama.cpp/commit/0363d99d3b276a5cfb8d2ab2940cd1a9a7e1d46b). It excludes inherited upstream changes. The documentation-only update titled `docs: record RDNA3.5 optimization targets and measured gains` adds this audit and has no runtime performance effect.

Percentages below are latency reductions, `(before - after) / before`, not throughput increases. **Fresh prefill** measures `llama_decode` through GPU completion with fresh KV; it excludes HTTP, tokenization and image processing. **HTTP TTFT** measures request submission through the first content token, including image work for VLMs. GPU kernel sums are diagnostic measurements and are not interchangeable with either latency metric. Results from different baselines, boards, image budgets or microbatches must not be added together.

Unless stated otherwise, the measurements use Halo41 / gfx1151, Q4_K_M weights and FP16 KV; VLMs use F16 projectors. Early text-only tests use 3968 saved token IDs, context and logical batch 4096, and the stated microbatch. VLM and October 13 server tests retain context checkpoints; none of these commits disables checkpoints. Historical reports named below are relative to `/proj/gdba/lichang/regression_results/`. These local artifacts are not shipped in this repository; their key results and limitations are recorded here.

| Commit | Optimization | Measured target and headline result |
| --- | --- | --- |
| [`9de63b7c5`](https://github.com/AMD-Ecosystem/llama.cpp/commit/9de63b7c55f2aecbf3b8a05b54f6e1205440e93d) | Q8 packing traversal and standalone NEOX RoPE launch | Qwen2.5-3B: 1.75% less fresh-prefill latency at ub2048; 2.99% at ub4096. |
| [`a49aaf478`](https://github.com/AMD-Ecosystem/llama.cpp/commit/a49aaf478466f9e2ca34e091a429de0abaa5f75c) | D128 attention register cap, tighter key bounds and verified causal-mask shortcut | Qwen2.5-3B: another 1.99% at ub2048; ub4096 mean improvement not established. |
| [`8e0181d66`](https://github.com/AMD-Ecosystem/llama.cpp/commit/8e0181d66e8a59e65c7cd2e66cba4251bba072ef) | Restrict unused mask-scanner template instantiations | HIP build fix; active gfx1151 machine code is identical. No speedup claimed. |
| [`f840324b3`](https://github.com/AMD-Ecosystem/llama.cpp/commit/f840324b3eb92899b107a402346031bdc874d6ec) | Extend verified causal attention to 16-query x 4-head tiles | Qwen3-4B: 39.23 ms / 2.39% less fresh-prefill latency at ub2048. |
| [`6c4d3b436`](https://github.com/AMD-Ecosystem/llama.cpp/commit/6c4d3b436ac51e4a01329847a37e81cdb98d0361) | Generalize Q8 packing traversal to aligned row strides | Qwen3-4B: 0.89%; Qwen3-8B: 3.30%; Llama-3.1-8B: 2.86% less fresh-prefill latency. |
| [`5aa5aee1e`](https://github.com/AMD-Ecosystem/llama.cpp/commit/5aa5aee1e64312b83a1b74c67a2ff4bdb5ef97ba) | Consolidate window attention, WMMA, wider D128 tiles, conversion, RMSNorm and graph fusions | Multiple experiments, detailed below. Original Qwen2.5-VL-3B vision package: 57.0% lower HTTP TTFT; final E2B fusion stage: another 6.35%. |
| [`1ee95c533`](https://github.com/AMD-Ecosystem/llama.cpp/commit/1ee95c533fbe93930e62a6e148df1937e1ea255c) | Adapt D72 vision attention to the existing D80 WMMA kernel | Gemma-3-4B-IT: 371.76 ms / 32.57% lower HTTP TTFT. |
| [`a68d7c3ed`](https://github.com/AMD-Ecosystem/llama.cpp/commit/a68d7c3ed306ff9ee4ed57dfcced87274f4eece5) | Extend producer fusion to SwiGLU and tune fused Q/K norm-RoPE | At 3968 tokens: Qwen2.5-3B saves 53.04 ms / 4.76%; Qwen3-4B saves 59.39 ms / 3.85% HTTP TTFT. |

### 9de63b7c5: Q8 quantization and standalone RoPE launches

The activation packer schedules all feature chunks of a row next to one another instead of traversing many rows before returning to the next feature chunk. The initial guard selects RDNA3.5 dense Q4_K projections with exactly 2048 input/padded features, no IDs and one channel/sequence plane. This targets Qwen2.5-3B's hidden projections. The arithmetic, packed activation layout and downstream matrix operation stay the same. Separately, standalone NEOX RoPE at head width 128 uses 64 threads instead of 256, avoiding inactive work. This also describes compatible Qwen2.5-7B heads, but the measurements here are for 3B; it does not target normal RoPE or D64/D256 heads.

| Qwen2.5-3B, 3968 tokens | Gfx11 base mean | This commit mean | Saving |
| --- | ---: | ---: | ---: |
| Fresh prefill, ub2048 | 1148.819 ms | 1128.751 ms | 20.069 ms / 1.75% |
| Fresh prefill, ub4096 | 1197.910 ms | 1162.089 ms | 35.821 ms / 2.99% |

This is the direct comparison with parent `0363d99d3`, without an attention preload. Earlier isolated experiments on a different attention baseline reduced width-2048 packing from 83.631 to 42.791 ms and RoPE from 17.444 to 14.237 ms in summed GPU time; these explain the mechanism but must not be added to the table. Final logits matched bitwise in the direct comparison. Evidence: `qwen3b_quant_rope_20261001/gfx11_rebase_20261001/report.md` and `qwen3b_quant_rope_20261001/report.md`.

### a49aaf478: D128 attention bounds and verified causal masks

This commit reuses the existing WMMA attention implementation with a dedicated 240-VGPR entry point for D128, 8-query x 8-head, no-softcap tiles. Mask preparation uses 64-key bounds and validates the actual mask before replacing mask loads with a causal-prefix calculation. Every value must be zero or negative infinity, each allowed region must be a contiguous prefix, and successive query endpoints must advance by one within the required tile boundary. Masks with holes, finite biases or incompatible endpoints keep their supplied values. This reduces mask traffic and masked work without assuming that every mask is causal.

The initial prefix guard targets RDNA3.5, FP16 K/V, D128, GQA8, at least 512 query positions divisible by eight, padded KV length at most 4096, and compatible mask batches. Qwen2.5-3B matches this structure; Gemma D256/D512 attention does not. Against parent `9de63b7c5`, Qwen2.5-3B's 3968-token fresh-prefill mean improves **1129.508 -> 1107.082 ms at ub2048**, saving **22.426 ms / 1.99%**. At ub4096, the mean changes **1157.689 -> 1166.098 ms**, a **0.73% regression**, while the median improves; intermittent candidate stalls remain unexplained in that experiment. No ub4096 average-latency gain is claimed.

Ten direct cases, including irregular masks and fallback shapes, matched the parent bitwise across 10,635,264 output floats. Evidence: `qwen3b_quant_rope_20261001/attention_commit_20261001/report.md`.

### 8e0181d66: mask-scanner build fix

A runtime condition did not prevent the compiler from instantiating unused RDNA mask scanners for other tile widths. Those unused instances triggered unroll warnings promoted to errors in the HIP quality build. The fix explicitly instantiates the eight-row scanner already required by the active fast path. It serves all builds compiling these attention templates, including gfx942; it does not target a new model structure or alter runtime eligibility.

**Performance: no speedup claimed and no additional latency benchmark.** The entire checked gfx1151 attention `.text` section, active scanner and scanner resource descriptor are byte-for-byte identical before and after. Targeted gfx942/gfx1151 compilation checks pass; full remote CI was not run. Later commits deliberately add the needed 16- and 32-row specializations. Evidence: `qwen3b_quant_rope_20261001/hip_quality_fix_20261001/report.md`.

### f840324b3: causal shortcut for 16x4 attention

The same mask validation and prefix decoding now cover D128 16-query x 4-head tiles as well as 8x8 tiles. Removing the exact-GQA8 restriction also admits compatible partial or multiple head groups. The key tile stays 64; arbitrary-mask fallbacks remain. Qwen3-4B has 32 query heads and eight KV heads, so its GQA4 attention benefits. D64/D256/D512 and short-query paths do not acquire this shortcut.

For Qwen3-4B Q4_K_M at 3968 tokens, paired median fresh prefill improves **1641.72 -> 1602.50 ms at ub2048**, saving **39.23 ms / 2.39%**. Attention plus mask preparation falls **335.620 -> 300.478 ms** in separate profiles. At ub4096 the median improves **1732.57 -> 1722.97 ms**, only **9.60 ms / 0.55%**; retained stalls mean the unfiltered mean does not improve. Both comparison backends already contain the earlier generalized guards/quantizer experiment, so this comparison isolates the 16x4 addition rather than every line relative to its Git parent.

The earlier broader-GQA study measured Qwen2.5-7B **2321.19 -> 2304.61 ms** mean fresh prefill, saving **16.58 ms / 0.71%**. That study tested guard and traversal changes together; the final stride guard kept 7B's quantizer on its old path, and profiles attributed the saving mainly to attention. It is supporting evidence for broader head-ratio eligibility, not a second independent parent-to-commit result to add to the Qwen3 result. Evidence: `qwen3_4b_attention_opt_20261005/report.md` and `qwen3b_quant_rope_20261001/guard_generalization_20261002/report.md`.

### 6c4d3b436: quantizer interleaving for aligned strides

The fixed four-chunk mapping becomes a general grid with adjacent feature chunks per row. Eligibility requires RDNA3.5, Q4_K, more than one 512-feature chunk, actual row stride divisible by 1024 floats, no IDs, one channel/sequence plane and at most 65535 rows. Qwen3-4B gains its 4096-wide attention-output projection; Qwen3-8B gains 4096/12288 widths; Llama-3.1-8B gains 4096/14336 widths. Qwen2.5-3B's 2048-wide path was already optimized, while its 11008-wide FFN remains outside this stride guard. Q6_K projections retain their existing path.

| Model, 3968 tokens / ub2048 | Parent f840324b3 median fresh prefill | This commit | Saving |
| --- | ---: | ---: | ---: |
| Qwen3-4B | 1605.47 ms | 1591.25 ms | 14.22 ms / 0.89% |
| Qwen3-8B | 2768.65 ms | 2677.34 ms | 91.30 ms / 3.30% |
| Llama-3.1-8B | 2734.18 ms | 2655.95 ms | 78.22 ms / 2.86% |
| Qwen2.5-3B | 1149.46 ms | 1149.87 ms | -0.41 ms / -0.04%; neutral within variation |

Only the quantizer object differs in this isolated ABBA comparison. Direct aligned-width kernel reductions are about 43-48%; Llama's full-request quantizer sum falls **181.174 -> 94.043 ms**. All 54 direct byte comparisons and captured model logits match. An earlier unrestricted traversal made Qwen2.5-7B's unaligned quantizer slower, which is why the stride guard is retained. Evidence: `quant_guard_ab_20261005/report.md`.

### 5aa5aee1e: consolidated attention, conversion and fusion work

This commit brings several separately tested directions into one maintained implementation. **There is no single measured parent-to-commit gain for the whole commit.** The table records each direction's own baseline and eligible structure. Source consolidation also includes graph-capacity handling, dispatch guards, shared helpers and backend tests; those supporting edits have no separately measured speedup.

| Direction | Model and structure targeted; what changes | Measured benefit and baseline |
| --- | --- | --- |
| Independent vision windows | Qwen2.5-VL-3B/7B windowed vision graphs with flash attention. Batch adjacent equal-size windows, join outputs and project once; avoid constructing/scanning the dense window mask. Global-attention layers retain all patch interactions. | Original Halo40 3B HTTP experiment, 1606 prompt positions: **2251.038 -> 1073.904 ms** for windows alone, **1177.134 ms / 52.29%**. This is the original vision experiment, not the complete commit comparison. |
| D80 vision WMMA | Qwen2.5-VL D80 unmasked self-attention, GQA1, FP16 K/V, at least 64 queries and no bias/softcap. Enable the existing 64-query matrix kernel for eligible windows and global layers. | Same 3B experiment: windows-only **1073.904 -> 968.160 ms**, another **105.744 ms / 9.85%**. Windows plus WMMA total **57.0%** lower TTFT than its original baseline. Short edge windows retain the generic kernel. |
| D64 vision WMMA | Gemma-4-E2B vision: 12 heads x D64, unmasked global self-attention. Use the same WMMA family with tail handling; no window partitioning. | At 775 prompt positions / 2268 patches: HTTP TTFT **424.50 -> 389.93 ms**, **34.57 ms / 8.14%**. At the separate default budget, 880 positions / 3213 patches: **557.97 -> 494.57 ms**, **63.41 ms / 11.36%** using unrounded samples. Each comparison keeps image resolution fixed. |
| D128 wider query tile | Qwen3-4B and compatible D128 GQA4 prefill. Expand 16x4 to 32x4, retain the 64-key tile and reuse each loaded KV tile across twice as many queries. Require at least 512 queries divisible by 32, FP16 KV, compatible mask/KV padding, KV <=4096, no softcap and an occupancy check that keeps both tile choices off Stream-K. | Halo06, against `6c4d3b436`, 3968-token fresh prefill: **1533.35 -> 1490.69 ms**, **42.66 ms / 2.78%** at ub2048; **1651.02 -> 1502.13 ms**, **148.89 ms / 9.02%** at ub4096. Do not combine Halo06 times with Halo41 results. |
| D512 query staging | Gemma-4-E2B's seven global language-attention layers, D512/GQA8/FP16 KV. Enable the existing 4x8 WMMA tile and stage Q in shared memory to reduce spills. Guard short passes, mask layout and softcap. | On the earlier E2B vision-optimized baseline: isolated HTTP TTFT **390.18 -> 387.45 ms**, **2.73 ms / 0.70%**. Adding it after conversion/RMS/BLAS tuning: **368.78 -> 367.49 ms**, **1.29 ms / 0.35%**. Merely enabling WMMA with Q in registers regressed by 2.05 ms and was rejected. |
| Packed FP16/FP32 conversion | E2B vision and any eligible contiguous aligned RDNA3.5 conversion of at least 256 elements. Process two values per thread with the original casts and scalar tail. | Isolated E2B HTTP TTFT **392.28 -> 388.79 ms**, **3.49 ms / 0.89%**. Hot-buffer kernel time roughly 35 -> 18 us; the cold-buffer advantage is smaller. |
| RMSNorm launch tuning | E2B width-1536 language activations, including weight/add variants. Select 256 instead of 1024 threads for at least 64 rows; preserve tiny-batch behavior. | At 258 rows, final plain-RMS microbenchmark **11.67 -> 7.96 us**, about **31.8%**. Earlier guarded RMS added to BLAS/conversion saved **1.66 ms** HTTP TTFT; the final 64-row threshold has no separate isolated HTTP measurement. Enabling the smaller block for tiny batches did not help. |
| Clamp/conversion, norm/packing, GEGLU/packing and elementwise fusions | E2B/Gemma graphs: clamp next to FP16 vision GEMM; RMSNorm + weight feeding dense gate/up projections; GEGLU feeding Q4_K/Q5_K DS4, Q6_K D4 or the existing FP16 BLAS route; GELU x gate; residual scaling. Reuse shared kernels and enforce consumers, aliases, precision and rounding. | Against the already consolidated pre-fusion baseline, E2B HTTP TTFT **366.21 -> 342.97 ms**, **23.24 ms / 6.35%**; **955 launches removed**. Gemma-3-4B saves **13.07 ms / 1.14%**; Gemma-4-26B-A4B saves **34.07 ms / 3.61%**. See the per-fusion and transfer tables below for individual stages and actual dispatch. |

The original Qwen vision experiment used a 1024x800 source image resized to 1036x812, yielding 4292 patches, 80 windows grouped into 15 runs, and 1606 language prompt positions. It ran on Halo40, with ten timed requests per variant. Gemma E2B's image-limit-280 setup uses the same source dimensions, resized to 864x672, yielding 2268 patches, 252 image embeddings and 775 total prompt positions. Its paired comparisons generally use 20 timed requests per configuration on Halo41. Gemma global vision attention cannot use independent-window partitioning without changing valid token interactions.

The fusions target operations, not model names. Qwen2.5-VL-3B/7B transfer checks show only **963.41 -> 958.96 ms** and **1445.68 -> 1443.95 ms** HTTP TTFT for the fusion stage, so there is no material extra Qwen gain from that stage. Gemma-4-26B-A4B's tested UD-Q4_K_M structure gains clamp/conversion and residual-scaling fusions; its Q8_0 shared and routed MoE projections do not take the new dense norm/GEGLU packing route. The Gemma-3 and 26B fusion measurements precede the D72 adapter.

Consolidation itself was checked against the earlier optimized VLM packages: Qwen2.5-VL-3B **986.116 -> 967.534 ms**, Qwen2.5-VL-7B **1479.729 -> 1453.681 ms**, and E2B **367.529 -> 368.607 ms** median HTTP TTFT. These are integration comparisons, not isolated kernel gains. The earlier E2B combination **390.00 -> 367.49 ms** also includes `ROCBLAS_USE_HIPBLASLT=1`; that environment choice is not a code optimization in this commit and is not hardcoded. The rejected width-1536 quantizer extension and full gate/up matrix fusion remain disabled for the measured reasons discussed below.

Vision WMMA can change floating-point outputs; original 32-token greedy responses match, but embeddings/log probabilities need not match bitwise. The retained producer-packing fusions preserve tested producer bytes through an FP32 materialization barrier. Neither targeted checks nor compilation on other architectures establishes broad model quality or cross-GPU speedups.

Evidence: `qwen25vl3b_attention_opt_20261005/report.md`, `gemma4_e2b_vision_opt_20261006/report.md`, `qwen3_4b_attn_pipeline_20261006/report.md`, `gemma4_kernel_investigation_20261006/report.md`, `rdna35_consolidation_20261007/report.md`, `rdna35_fusion_20261007/REPORT.md` and `rdna35_fusion_20261007/TRANSFER_REPORT.md`.

### 1ee95c533: D72 vision adapter

Gemma-3-4B-IT has 27 global vision layers with 16 heads x 72 channels. The adapter zero-pads Q/K/V to D80, calls the existing WMMA kernel, then extracts 72 output channels while retaining the original `1/sqrt(72)` scale. The actual eligibility is structural: RDNA3.5, unmasked equal-length GQA1 self-attention, FP32 Q, FP16 K/V, compatible layouts, no sinks/bias/softcap, at least 256 positions and 4096 query rows across heads/batches. Small shapes keep the generic tile to avoid padding overhead.

Against parent `5aa5aee1e`, fresh Halo41 ABBA testing gives Gemma-3-4B HTTP TTFT **1141.47 -> 769.71 ms**, saving **371.76 ms / 32.57%**. The 1024x800 image becomes 896x896 with 4096 vision patch positions; the language prompt has 781 positions, including 256 image embeddings. There are 20 timed requests per configuration and checkpoints remain at 32. Padding uses 60 MiB of temporary storage for this shape. Gemma-4-26B-A4B also has D72 vision heads, but this commit has **no measured end-to-end gain for that model**; structural similarity is not a performance result.

The 909 D72 and 224 D64/D80 backend cases pass. Generated token IDs match for the tested 32-token response, while selected-token log probabilities differ; see the [D72 follow-up](#d72-vision-attention-follow-up-2026-10-13). Evidence: `gemma3_d72_verify_20261013/REPORT.md` and `gemma3_d72_attention_20261009/REPORT.md`.

### a68d7c3ed: SwiGLU producer fusion and fused norm-RoPE tuning

This extends the existing GEGLU implementation to SwiGLU, writing activation/gate results directly into the Q8 layout consumed by Q4_K/Q5_K/Q6_K MMQ, or into FP16 for the already selected BLAS route. It avoids the intermediate FP32 store/load and a separate packing/conversion launch. Padded rows, including Qwen2.5-3B's 11008 -> 11264 case, receive zero tails and correctly sized allocations. Existing hardware, matrix-dispatch, precision, layout and dependency guards remain. Compatible dense SwiGLU FFNs in other models may use this path, but only the two Qwen models below have end-to-end measurements for this commit.

The second change selects 128 instead of 256 threads in the existing fused RMSNorm + weight + RoPE kernel for RDNA3.5 D128 heads and at least 128 token positions. Qwen3-4B has this Q/K normalization structure; Qwen2.5-3B does not. This is distinct from the standalone 64-thread NEOX RoPE change in `9de63b7c5`.

Against parent `1ee95c533`, at 3968 tokens / ub2048, HTTP TTFT improves **1113.82 -> 1060.78 ms for Qwen2.5-3B**, saving **53.04 ms / 4.76%**, and **1543.96 -> 1484.58 ms for Qwen3-4B**, saving **59.39 ms / 3.85%**. At 128 tokens, the respective gains are **1.18 ms / 2.56%** and **1.02 ms / 1.77%**. Qwen3's paired long-input SwiGLU-only stage saves **48.01 ms**; adding norm-RoPE saves another **11.38 ms**. The short-input norm-RoPE increment is within variation. Q8 versus FP16 fusion contributions were not timed separately end to end.

Each model removes 35 launches at 128 tokens and 70 at 3968. All 96 producer cases, ten norm-RoPE bitwise comparisons, 297 backend graph cases and 20 generation checks pass. The [SwiGLU follow-up](#swiglu-and-norm-rope-follow-up-2026-10-13) gives full settings and results. Evidence: `rdna35_swiglu_normrope_20261013/REPORT.md`.

## Dispatch and tradeoffs

| Direction | Shared implementation and eligibility | Why it is selective |
| --- | --- | --- |
| Qwen2.5-VL window attention | The existing vision graph groups adjacent windows of equal length and calls the common attention backend. The non-flash path retains the mask. | Window independence comes from the model graph. Gemma global attention cannot be partitioned into independent windows without changing the model. |
| D64/D80 vision attention | One RDNA3.5 rule selects the existing 64-query WMMA kernel for unmasked FP16 self-attention, GQA1, at least 64 queries, and no bias or softcap. | Short windows retain the generic tile. The same kernel supports both head widths; no model-name dispatch is needed. |
| D72 vision attention | Zero-pad Q/K/V to D80, reuse the existing WMMA kernel, then extract D72 output while preserving the original scale. Require RDNA3.5, unmasked equal-length GQA1 self-attention, FP32 Q, FP16 K/V, compatible layouts, no sinks/bias/softcap, at least 256 positions and 4096 query rows across heads/batches. | Padding adds four copy launches and temporary storage. Small workloads retain the generic tile because padding can cost more than the matrix-kernel saving. |
| D128 language attention | Verified causal-mask shortcuts cover 8x8, 16x4 and 32x4 tiles. Compatible D128/GQA4 prefill uses the 32-query tile when shape and occupancy guards pass. | The geometry and resource use differ from D256/D512. Mask-prefix shortcuts still validate the actual mask before skipping loads. |
| D512 language attention | Existing 4-query x 8-head WMMA tile, with query data in shared memory on RDNA3.5. Dispatch requires FP16 K/V, GQA8, a compatible padded mask, no softcap, and at least 32 queries. | Enabling WMMA without resource tuning caused register spills and was slower. Shared-query staging reduces spills. The setting is scoped in both host and device selection; other RDNA GPUs retain their configuration. |
| FP16/FP32 conversion | One packed conversion implementation handles two adjacent values per thread, with original casts and a scalar tail. | Requires RDNA3.5, contiguous conversion, alignment, and at least 256 elements. Wider vectors did not consistently help; cold-memory gains are smaller than hot-buffer gains. |
| RMSNorm | The existing 256-thread kernel is selected at width 1536 and at least 64 rows on RDNA3.5, including multiply/add variants. | Tiny batches favor the original 1024-thread block. Boundary measurements found the previous 32-row threshold too low for plain RMSNorm. |
| Activation quantization | The existing Q4_K activation packer interleaves feature chunks in its launch grid. Arithmetic and packed output layout stay unchanged. | Row strides divisible by 1024 floats showed large gains. An earlier unrestricted extension slowed Qwen7B quantization. Width 1536 passed byte comparisons but showed no useful end-to-end gain in the follow-up trial, so it retains ordinary traversal. |
| RoPE | Existing shape-guarded launch tuning remains shared. | The selected D128 geometry does not imply a benefit for Gemma's D256/D512 heads. More blocks or threads can trade scheduling overhead against occupancy. |
| Dense GEMM provider | `ROCBLAS_USE_HIPBLASLT=1` is a runtime choice, using the same source and weights. | It helps the measured E2B workload but is not uniformly faster across models and prompt shapes. It is not hardcoded globally. |

Correctness eligibility and performance eligibility are different. A shape can produce the right answer with both kernels and still be slower with the candidate. Dispatch should depend on hardware, types, layouts, shapes, and dependencies. Model-name branches are unnecessary for these backend changes.

## Why launch interleaving does not transfer uniformly

At 258 rows, the ordinary width 1536 quantizer took 21.08 us and the interleaved version took 20.84 us. Width 2048 took 40.92 us and 26.33 us respectively. Counting the same logical input/output bytes, ordinary width 1536 already achieves about 96 GB/s, close to the interleaved width 2048 result of 103 GB/s; ordinary width 2048 achieves about 66 GB/s. These are effective rates, not physical DRAM bandwidth.

The width 1536-only extension changed E2B median TTFT from 367.514 ms to 367.490 ms in the paired trial (20 samples per configuration). A 0.024 ms difference is below the observed variation. The extension is not enabled.

Earlier hardware counters on the slow aligned shape showed fewer memory-wait cycles after interleaving, with the same arithmetic and nearly unchanged request counts. This supports an access-order-sensitive memory bottleneck. It does not identify a specific DRAM bank, channel, or cache mechanism, and no such cause is claimed for width 1536. The existing ordering at that width simply has much less measured inefficiency to remove.

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

In this pre-D72-adapter comparison, Gemma-3 removes 297 launches through shared norm packing and GEGLU packing/conversion, but D72 vision attention still takes about 612 ms in summed profile kernel time. Both models have D72 vision heads, outside the D64/D80 WMMA dispatch available at that time. The 26B profile removes 560 launches: 380 from vision clamp/conversion and 180 from residual scaling. Its UD quantization has Q8_0 shared FFN projections and Q4_K routed gate/up projections, so the new dense norm/GEGLU packing fusions do not execute for this model. Its D72 vision attention takes about 225 ms in that pre-adapter profile. See `/proj/gdba/lichang/regression_results/rdna35_fusion_20261007/TRANSFER_REPORT.md` for the final profile attribution, exact weights, raw results, and limitations.

## D72 vision attention follow-up, 2026-10-13

Gemma-3's global vision attention uses width 72, which falls outside the existing matrix-tile assumptions. A small adapter now pads Q/K/V to width 80 with zeros, calls the shared D80 WMMA implementation, and extracts the first 72 output channels. The original 1/sqrt(72) scale is preserved. This retains global attention and does not partition the image into independent windows. Dispatch uses the hardware/type/shape guards above, without a model-name check. Logical temporary storage is 60 MiB for the tested image.

A fresh Halo41 ABBA comparison of Gemma-3-4B-IT Q4_K_M with an F16 projector reduces median HTTP TTFT from **1141.47 ms to 769.71 ms**, saving **371.76 ms / 32.57%**. Server prompt time falls from 1118.17 to 747.16 ms. Each configuration has 20 timed requests across two starts, with five warmups per start. The original 1024x800 image becomes 4096 vision patch positions; the language prompt has 781 total positions including 256 image embeddings. Checkpoints remain enabled with `--ctx-checkpoints 32`, prompt-cache reuse is disabled, and no GPU contention was detected.

The current build passes 909 D72 and 224 D64/D80 backend cases. All four 32-token generation checks match token IDs, strings, bytes and text. Floating-point arithmetic changes: selected-token log probabilities differ by 0.02354 on average and 0.16270 at most. This is targeted validation on gfx1151, not a broad quality or cross-GPU claim. The existing D80 configuration is unchanged. The tested HIP library matches the previously retained October 9 candidate exactly.

Raw results, exact commands, source/library hashes and analysis are in `/proj/gdba/lichang/regression_results/gemma3_d72_verify_20261013/REPORT.md`. Earlier controlled attention, small-shape threshold and FP64-reference measurements are in `/proj/gdba/lichang/regression_results/gemma3_d72_attention_20261009/REPORT.md`.

## SwiGLU and norm-RoPE follow-up, 2026-10-13

The existing GEGLU producer/packing and producer/conversion kernels now also support SwiGLU. Both operations retain the original scalar activation and FP32 rounding before writing DS4/D4 Q8 blocks or FP16 values. The Q8 path fills the padded tail and allocates for the padded row width, including Qwen2.5-3B's 11008-to-11264 case. Existing RDNA3.5, type, precision, layout, aliasing, consumer and matrix-dispatch guards remain in use. This extends the shared implementation without changing the downstream matrix strategy.

The fused Q/K RMSNorm + weight + RoPE path selects 128 threads for 128-wide heads with at least 128 token positions on RDNA3.5. Other shapes keep their prior selection. This reuses the existing kernel arithmetic and does not enable the numerically different 64-thread variant.

Fresh Halo41 HTTP timings with Q4_K_M weights and checkpoints enabled (`--ctx-checkpoints 32`):

| Model | Input tokens | Before | Combined | Saving |
| --- | ---: | ---: | ---: | ---: |
| Qwen2.5-3B | 128 | 45.95 ms | 44.78 ms | 1.18 ms / 2.56% |
| Qwen2.5-3B | 3968 | 1113.82 ms | 1060.78 ms | 53.04 ms / 4.76% |
| Qwen3-4B | 128 | 57.88 ms | 56.86 ms | 1.02 ms / 1.77% |
| Qwen3-4B | 3968 | 1543.96 ms | 1484.58 ms | 59.39 ms / 3.85% |

Each configuration/length has 20 samples across two server starts and five warmups per length/start. The runs use identical integer-token prompts, no prompt cache, one output token, context 4096, batch 4096, microbatch 2048, FP16 KV and `ROCBLAS_USE_HIPBLASLT=1`. Qwen2.5 uses ABBA order; Qwen3 adds a paired SwiGLU-only stage. Qwen3's 3968-token SwiGLU stage measures 1495.96 ms, saving 48.01 ms; the norm-RoPE addition saves another 11.38 ms. Its 128-token norm-RoPE increment is within run variation and does not establish an additional short-prompt benefit.

Validation passes 96 bitwise producer/packing/conversion cases, 10 bitwise norm-RoPE comparisons and 297 backend graph cases. All 20 model generation checks match the baseline's 32 token IDs, text and returned probability records exactly. The tests cover padded tails, output canaries, dispatch boundaries, extra consumers, precision fallbacks, RoPE modes, KV writes and batches. Runtime validation is limited to gfx1151 and the stated workloads.

Both fusion routes execute at long input: 36 Q8 producer fusions and 34 FP16 producer fusions per model request, removing 70 launches. At 128 tokens, 35 Q8 producer fusions remove 35 launches. The normal build contains these changes. Exact commands, immutable stage libraries, source hashes, generation checks and profiles are in `/proj/gdba/lichang/regression_results/rdna35_swiglu_normrope_20261013/REPORT.md`.

## Build and measurements

The local gfx1151 build is `build-rdna35-gfx1151`. It includes `llama-server`, `llama-bench`, and `test-backend-ops`; no experimental library preloads are required for normal use. The measured E2B setup uses `ROCBLAS_USE_HIPBLASLT=1`.

Consolidation measurements are in `/proj/gdba/lichang/regression_results/rdna35_consolidation_20261007`. The fusion report, exact requests, source/library hashes, per-stage variants, tests, profiles, and raw samples are in `/proj/gdba/lichang/regression_results/rdna35_fusion_20261007/REPORT.md` and its adjacent artifacts. The paired comparison uses the same built server and an immutable saved HIP library for each configuration. The normal build includes the final fusion changes and the subsequent D72, SwiGLU and norm-RoPE follow-ups; it needs no preload. GPU tests run serially on Halo41; samples overlapping another KFD process group are excluded. No overlap was detected in the final runs. Raw outliers and exploratory numerical differences are retained and described in the report.
