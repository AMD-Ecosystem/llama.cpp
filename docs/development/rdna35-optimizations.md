# RDNA3.5 optimization notes

The active implementation is `lichang.rdna35-prefill-quant-rope`. Qwen vision window batching, D64/D72/D80 vision attention, D128 language attention, D512 language attention, FP16/FP32 conversion, RMSNorm, activation quantization, and RoPE changes share this source tree. Earlier experimental worktrees are reference snapshots; changes should be developed and validated here.

## Gemma-3-12B vision kernel follow-up (2026-10-29)

The commit titled `hip: tune RDNA3.5 vision attention, copies and normalization` adds three small backend changes for the two vision targets from the serial profile: D72 attention and conversion/LayerNorm overhead. It builds on the shared Gemma vision fusions below. Language Q4_K/Q6_K GEMM and the actual 1/104/256/2/4 batch boundaries are unchanged. Increasing the configured batch limit does not merge those boundaries.

| Change added to the existing optimized baseline | Baseline HTTP TTFT | Candidate HTTP TTFT | Saving |
| --- | ---: | ---: | ---: |
| Full-tile D80 attention handling | 843.09 ms | 815.25 ms | 27.84 ms / 3.30% |
| Vectorized permuted K/V conversion | 854.76 ms | 836.25 ms | 18.51 ms / 2.17% |
| 128-thread affine LayerNorm | 837.46 ms | 829.00 ms | 8.46 ms / 1.01% |
| Combined | 857.01 ms | 818.46 ms | 38.55 ms / 4.50% |

These are separate Halo05 A-B-B-A comparisons, each with 20 samples per configuration across two starts and five warmups per start. The same server/supporting libraries are used on both sides. The combined baseline process medians are 883.10 and 846.35 ms; candidate medians are 819.17 and 814.85 ms. Heavy CPU/filesystem activity on the shared board makes the precise percentage uncertain; raw outliers are retained. No competing GPU process was detected. The separate-stage savings must not be added as an exact combined decomposition. A later repeat encountered a new external CPU-heavy job and a baseline step to about 1080 ms; it was stopped and is not used to enlarge the reported gain.

The saved 640x480 image becomes 896x896, giving 4096 vision patches and 256 image embeddings. The saved 100-text-token prompt has 367 total llama.cpp positions. Tests retain Q4_K_M language weights, F16 projector and KV, context 4096, batch 2048, microbatch 512, eight CPU threads, one output token, checkpoints 32, HIP graphs, normal vision/language overlap, and no prompt-cache reuse. There is no image-budget change.

- Attention reuses the existing D72-to-D80 adapter and shared WMMA kernel. On RDNA3.5, the dense D80 64-query/one-head-group path omits partial-tile bounds handling only for non-final KV tiles, which are known to be complete. Final tiles and sparse paths keep their checks. Global attention, the original D72 scale, precision, and tile geometry stay the same. Isolated call time falls from 5.56-5.58 to 4.73 ms. Compiler scratch falls from 224 to 192 bytes/thread and VGPR spill count from 92 to 83; VGPR allocation remains 256.
- The existing FP32-to-FP16 CPY dispatch gains an eight-value path for aligned, same-shape, noncontiguous-source/contiguous-destination copies. It amortizes indexing, uses vector memory operations, and retains the existing casts. Guards check disjoint buffers, alignment, strides, shape, minimum size, and the fast-division index range; byte offsets use size_t. The isolated K/V copy falls from 0.267 to 0.051 ms. Separate D72 padding kernels remain.
- The existing fused affine LayerNorm uses 128 threads for 1152-wide rows with at least 64 rows on RDNA3.5, instead of 1024. FP32 arithmetic/storage remain; reduction order changes. Isolated time falls from 0.275-0.278 to 0.168-0.169 ms. Other shapes and RMSNorm retain their previous choices.

The final rocprof trace confirms 27 attention calls at 160.93 -> 133.74 ms summed duration, 54 K/V copies at 19.31 -> 10.91 ms, and 55 affine norms at 23.54 -> 10.98 ms. Total launches remain 6468. Counting overlap once, first-kernel-start to last-kernel-finish falls 871.21 -> 834.26 ms, while the union of kernel intervals falls 822.94 -> 784.82 ms. These profiled values corroborate the normal TTFT gain but are not interchangeable with it; other raw kernel-duration sums vary with concurrent execution and host activity.

A 16-key attention tile cuts spills much further but is about 12% slower, so it is rejected. A compact-Q/shuffle variant reduces scratch and VGPR spills but has negligible speed benefit. Neither is retained. The 256-thread LayerNorm candidate is faster than the baseline but slower than 128 threads. Earlier Q-in-LDS and wider padding-copy failures remain documented; this result does not revive those prototypes.

The final build passes 1025 D72/D80 attention cases, 11 FP32-to-FP16 copy cases, and 30 affine LayerNorm cases. Attention probes at 4096, 4095, and 257 positions and copy probes including special FP values match baseline output bits. LayerNorm's FP64-reference relative L2 error is approximately 5.17e-8, comparable to the baseline. All three normal-server 32-token image checks produce the same token strings and text as the baseline. The final image embeddings differ: NMSE 5.21e-6 to 1.08e-5, relative RMS difference 0.23-0.33%, maximum absolute difference 0.1875; selected-token log probabilities differ by at most 0.10434. This is targeted numerical/output validation, not a broad task-accuracy claim. The LayerNorm-only build matches the combined build's vision embeddings bitwise on all three images, attributing the measured embedding change to reduction order in this workload. Rebuilt production matches the combined candidate's embeddings and returned token/probability records exactly on all three images.

The changed units compile for gfx1100 and gfx942; runtime validation is gfx1151 only. The final build preserves the baseline device code for Q4_K, Q6_K, MMVQ, and MMVDQ. NVIDIA was not compiled or tested. Other Gemma models with matching vision shapes are structurally eligible, but this round does not establish their end-to-end gains.

The saved gfx11 vLLM reference remains 541.770 ms, with 369 native-framing positions and W4A16/BF16 arithmetic. It was not rerun; using that historical reference gives approximately 1.582x -> 1.511x for the paired llama.cpp numbers, with the host-load and representation limitations above. Raw data, source snapshots, build commands, numerical comparisons, rejected candidates, and reproduction instructions are in `/proj/gdba/lichang/regression_results/gemma3_12b_vision_20261029/REPORT.md`.

## Gemma-3-12B profiling follow-up (2026-10-28)

The existing shared Gemma vision fusions also cover Gemma-3-12B Q4_K_M with an F16 projector. A Halo40 ABBA comparison (20 samples per variant, original response cap) reduces HTTP TTFT from 913.261 to 875.608 ms, saving 37.653 ms / 4.123%. Its 55 affine LayerNorms and 27 plain GELU vision FFNs remove 191 launches, 915 -> 724. Vision GPU duration sums fall 459.40 -> 406.33 ms; language graph signatures are unchanged. This is an increment over the optimized branch before the October 28 shared fusions.

Input is the saved 640x480 image and 100-token text prompt, with 367 total llama.cpp positions. Vision processes 896x896, 4096 patches and 256 image embeddings. Checkpoints remain 32, context 4096, batch 2048, microbatch 512, F16 KV, no prompt-cache reuse and unchanged text/image boundaries. Both servers use the same image resolution; there is no Gemma-4 image-budget mismatch in this test.

Fresh Halo05 production llama.cpp measures 824.042 ms with one output token and 830.717 ms with the original 128-token cap/EOS behavior. Fresh installed gfx11 vLLM measures 541.770 ms (three samples), giving approximately 1.52-1.53x. Its native framing produces 369 positions; it uses RedHatAI W4A16 weights, BF16 activations/KV, the gfx11 HybridW4A16 kernel and tuned Triton D72 vision attention. These are not identical-arithmetic runs. The same-board ABBA comparison above, not subtraction from older runs on other boards, establishes the fusion gain.

Diagnostic raw GPU sums show language matrix work at 545.98 versus 325.66 ms, vision attention/preparation at 182.00 versus 76.12 ms, and vision support at 97.77 versus 15.96 ms. llama.cpp vision/text overlap and different profiler overhead mean these differences are not additive TTFT savings. The existing Q4_K pipelined MMQ remains active. Tiny language batches contribute 102.27 ms of matrix-vector work; vLLM processes one 369-position batch and combines Q/K/V and gate/up projections.

Further probes do not justify another production change. Three supported attention tile/thread variants are slower or unchanged. Two-warps/64-query configurations violate the shared WMMA ownership mapping; vLLM's geometry cannot be copied as a configuration change. Wider D72 padding copies pass 909 backend cases and exact normal-server embedding/generation checks, but the final comparison with identical supporting libraries is 830.717 -> 833.855 ms, so the prototype is not retained. Earlier apparent copy gains used different supporting library builds and are superseded. A restricted Q6_K BLAS dispatch probe saves 5.416 ms / 0.625% on one workload, but substantially hurts the 104-position FFN if extended there; it remains outside production pending general shape and numerical validation. Serial vision costs 34.961 ms, earlier overlap costs 31.162 ms, and full semantic graph reuse saves only 5.244 ms. Current scheduling is retained.

Shared-fusion checks have bitwise-identical vision embeddings on three images, but one small-image generation differs and selected-token probabilities vary within fixed builds. The report preserves these results and does not claim broad accuracy equivalence. All new prototypes remain under `/proj/gdba/lichang/regression_results/gemma3_12b_optimization_20261028`; no backend source change, commit or push is added by this follow-up. See that directory's `REPORT.md` for raw timings, workload/profiler limitations, numerical checks, failed probes and commands.

## Gemma vision fusion follow-up (2026-10-28)

The commit titled `hip: extend shared vision fusions for Gemma`, based on `d79fcf20777641b2b5b172a11ec342e26241e4a5`, extends the existing backend fusion implementation for Gemma-3 and Gemma-4. It retains checkpoint support, text/image chunk boundaries, the model graph's public activation types, and the language matrix kernels.

| Model / board | Prompt positions | Baseline HTTP TTFT | With changes | Saving |
| --- | ---: | ---: | ---: | ---: |
| Gemma-3-4B, Q4_K_M/F16 / Halo40 | 779 | 727.32 ms | 689.11 ms | 38.21 ms / 5.25% |
| Gemma-4-E2B, Q4_0/BF16 / Halo05 | 880 | 402.91 ms | 378.16 ms | 24.75 ms / 6.14% |
| Gemma-4-26B-A4B, UD-Q4_K_M/F16 / Halo40 | 884 | 1002.42 ms | 946.84 ms | 55.58 ms / 5.54% |
| Gemma-4-31B, Q4_K_M/F16 / Halo05 | 884 | 3135.20 ms | 3155.44 ms | No demonstrated gain; 0.65% slower pooled median |
| Qwen2.5-VL-3B control / Halo40 | 1606 | 768.21 ms | 765.92 ms | 0.30%, within run variation |

Each variant has 20 measured requests over two server starts in A-B-B-A order, with five warmups per start. The original full response and EOS are retained, up to 128 output tokens. Tests keep checkpoints 32, no prompt reuse, F16 KV, context 4096, batch 2048, eight CPU threads and flash attention. Gemma uses microbatch 512; Qwen's control uses its recent microbatch 2048. No GPU contention was detected. All raw outliers are retained. The 31B baseline start medians vary from 3125.46 to 3164.78 ms, compared with 3153.74 and 3162.40 ms for the candidate; no end-to-end benefit is established for that workload.

These numbers measure an increment over the already optimized branch. The saved gfx11 vLLM references give Gemma-3 ratios of 1.560x -> 1.478x and Gemma-4-26B ratios of 1.601x -> 1.512x. vLLM was not rerun; the Gemma-4 comparison still has 779.3 vLLM positions versus 884 llama.cpp positions and different weight formats. E2B's default-image-budget Q4_0/BF16 test must not be compared directly with the earlier 775-position Q4_K_M/F16 test at an image-token limit of 280.

| Extension | Target structure and retained behavior | Confirmed coverage |
| --- | --- | --- |
| LayerNorm + vector scale + bias | Extend the existing LayerNorm kernel, retaining reduction geometry and intermediate FP32 rounding. RDNA3.5, F32, compatible contiguous layout and safe aliases only. | 55 Gemma-3 vision sites, removing 110 launches. Structurally relevant to Gemma-4 unified vision's affine norms, without a measured unified-model claim. |
| Plain GELU FFN | Extend the existing F16/BF16 FFN matcher for up GEMM, bias, GELU, down GEMM, optional output bias/residual. Keep the original narrow GEMM buffers; compute bias and GELU in FP32 and cast directly to the existing down input. | 27 Gemma-3 vision FFNs, removing 81 launches. The GEMMs themselves remain separate. |
| Clamped gated FFN | Reuse existing BLAS and unary/conversion kernels for clamped gate/up/down projections and QuickGELU. Share the input conversion only if both learned clamp limits match bitwise; preserve separate conversions otherwise. | All 16 E2B and 27 Gemma-4-26B FFNs, removing 64 and 108 launches. E2B's finite clamp limits are preserved. FP16 and BF16 are covered. |
| RMSNorm + scale + two rotary axes | Extend the existing norm/rotary kernel for compatible disjoint equal channel halves. Preserve the prior norm reduction geometry and FP32 store rounding. | 32 E2B and 54 Gemma-4-26B sites, removing one rotary launch per Q/K site. |

Gemma-3's SigLIP uses plain GELU, affine LayerNorm and global vision attention. Gemma-4's transformer vision tower uses gated QuickGELU, clamps, RMSNorm and separate x/y rotary passes. Those structures differ from the original biased gated Qwen pattern. They do not become independent attention windows, so Qwen's window partitioning is inapplicable.

The diagnostic vision kernel counts fall 915 -> 724 for Gemma-3, 584 -> 488 for E2B, and 1050 -> 888 for 26B. Vision GPU duration sums change 463.34 -> 412.58 ms, 149.95 -> 127.92 ms, and 377.69 -> 325.17 ms respectively. GEMM counts are unchanged. These are instrumented sums with vision/language overlap, not additive HTTP latency budgets. Remaining large work includes language matrix multiplication and global vision attention; fusion does not remove either computation.

A D72/D80 attention staging probe moved Q into LDS. It reduced spills (92 -> 18) and private storage (224 -> 96 bytes/thread), but VGPR use stayed at 256 and isolated attention worsened from about 5.75 to 6.37 ms. The probe was reverted. This rejects that implementation, not all possible D72 kernel improvements.

The ordinary production build passes 377 affected backend cases. A direct comparison against separate existing GPU operations passes 1176 cases / 72,904,608 values bitwise, including FP16/BF16, plain and gated activations, clamp bounds, misalignment, NaN payloads and canaries. Changed HIP units compile for gfx1100 and gfx942; runtime tests use gfx1151. NVIDIA was not compiled or tested.

The production Gemma-3 check and repeated final candidate match all three checked image embeddings, 32 generated tokens and chosen-token log probabilities exactly. Final E2B and 26B checks also match. One earlier Gemma-3 diagnostic process produced small embedding differences (NMSE up to 1.38e-6, maximum absolute difference 0.0625), with identical tokens and a maximum chosen-token log-probability difference of 0.26904. Repeating the same immutable binary reproduced the baseline; the cause of that process-level difference remains unresolved and is retained in the report. The checks are not a broad model-quality evaluation. One 31B baseline timing request also generated a different description; all candidate timing responses matched the other 19 baseline responses.

Gemma-4-31B's timestamped follow-up confirms that vision still accelerates: its span falls 750.62 -> 668.32 ms and its launches fall 1050 -> 888. Vision finishes at 1989.83 -> 1905.87 ms into the profiled request, before text-prefix GPU work finishes at 2067.37 -> 2030.33 ms. The shorter vision stage is therefore largely hidden behind language processing. Language GEMM/GEMV is approximately 2.75 seconds raw in the final profile and remains the main target. The instrumented run's modest improvement does not replace the normal ABBA result. All three 31B image embeddings and initial chosen-token probability records match across builds; repeated probability records vary within both builds.

The ordinary `build-rdna35-gfx1151` includes the changes. Standalone production checks measured Gemma-3 at 690.25 ms and E2B at 387.89 ms on Halo40; the latter is not the Halo05 paired comparison. Exact requests, profiles, immutable binaries, source snapshots, numerical checks, rejected experiments and reproduction commands are in `/proj/gdba/lichang/regression_results/gemma_shared_fusion_20261028/REPORT.md`.

## Vision tensor traffic and output fusion (2026-10-27)

The commit titled `hip: reduce RDNA3.5 vision tensor traffic` extends the existing implementation at `667af94f05375369079f549225067c315effe0b6`. It combines launch geometry, K/V conversion, normalized FFN input, residual conversion and window-output changes. The measured target is Qwen2.5-VL-7B; structural eligibility for other models is described below, without a new model-level performance claim for them.

| Qwen2.5-VL-7B / Halo40 | Baseline HTTP TTFT | With changes | Saving | Ratio to saved gfx11 vLLM |
| --- | ---: | ---: | ---: | ---: |
| Checkpoints off, alternating A-B-B-A, 20 samples per variant | 1310.36 ms | 1247.40 ms | 62.96 ms / 4.80% | 1.435x -> 1.366x |
| Checkpoints 32, 10 samples per variant | 1301.15 ms | 1246.46 ms | 54.69 ms / 4.20% | 1.425x -> 1.365x |
| Rebuilt production server, checkpoints 32, separate 10-sample run | - | 1236.12 ms | No paired control | 1.354x |

The saved gfx11 vLLM reference is 912.953 ms, from another date/board; vLLM was not rerun. A strict 1.35x result requires 1232.49 ms, so the alternating comparison is still 14.91 ms short. The faster standalone production run does not establish a repeatable 1.35x result. The baseline already includes the earlier FP16/BF16 and attention optimizations; their gains must not be added again.

Tests use the same 1024x800 uniform green image, resized to 1036x812, giving 4292 vision patches and 1073 image embeddings. Language positions are 527 prefix + 1073 image + 6 trailing text = 1606. Weights are Q4_K_M with an F16 projector, KV is F16, flash attention is enabled, context is 4096, CPU threads are 8, and batch/microbatch are both 2048 with `ROCBLAS_USE_HIPBLASLT=1`. This microbatch differs from the earlier ub512 fusion measurements. Normal timing retains the original 128-output-token cap and EOS behavior, measuring time to first content token after five warmups per process. Every measured request has zero prompt-cache reuse and 1606 prompt positions. GPU contention checks passed. The two baseline process medians were 1299.31 and 1318.54 ms, versus 1245.22 and 1250.13 ms for the two candidate processes; the pooled comparison retains that variation.

The changes target these structures:

- RDNA3.5 multimodal/vision RoPE with head widths 80 or 128 uses 64 threads instead of 256. The existing 256-thread RMSNorm path also covers widths 1280 and 3584 when there are at least 64 rows. RoPE arithmetic stays the same; RMSNorm changes the floating-point reduction order.
- The existing Qwen batched local-window graph converts each contiguous whole K/V tensor to F16 before taking window views. The shared attention builder accepts already-F16 K/V without redundant casts. This removes 784 copy launches for this input, with equivalent F16 rounding. Other attention structures do not acquire window batching from this change.
- The existing RDNA3.5 F16/BF16 BLAS fusion can combine output conversion, FP32 vector bias and FP32 residual addition. It preserves the intermediate FP32 bias result and rejects residuals that depend on an elided node, incompatible aliasing and extra intermediate consumers.
- A compatible biased gated FFN can consume its preceding RMSNorm and scale, performing normalization in FP32 and writing directly to its existing F16/BF16 scratch input. This removes an FP32 graph write and the following conversion. Existing BLAS, shape, precision, type, consumer and alias guards remain. Language MMQ outputs and the residual stream are not globally narrowed.
- Qwen window outputs use disjoint writes into one final tensor instead of a balanced CONCAT tree. A RDNA3.5 `FLASH_ATTN_EXT -> RESHAPE -> SET` fusion then lets attention write directly into that tensor, removing 420 assembly copies. It requires exclusive intermediate consumers, contiguous F32 storage, dense destination row stride, valid bounds, 16-byte alignment, no overlap with attention inputs, and no extra attention storage appended to the output. Unmatched cases use the ordinary SET path.

The changes retain text/image chunk boundaries and checkpoint support. Per-request launches fall from 4713 to 3441. The trace confirms 32 normalized-FFN fusions, 32 separate conversion/bias/residual fusions and 420 direct attention-output fusions. Language and vision GEMM counts and shapes are unchanged.

Independent rocprof traces contain three requests per variant, with exact per-stream kernel-sequence matching to an operation trace. Vision and language prefix execution overlap. The following times count each elapsed interval once, assigning GEMM priority, then attention, then support work; they are not predictions of removable latency.

| Profiled component | Baseline | With changes |
| --- | ---: | ---: |
| GEMM active | 1113.65 ms | 1103.95 ms |
| Attention active, no GEMM | 77.28 ms | 71.34 ms |
| Support only, no GEMM or attention | 203.75 ms | 146.23 ms |
| No traced GPU kernel | 45.55 ms | 44.27 ms |

The main change is 57.52 ms less exposed support work. Vision's mean span falls from 677.85 to 609.45 ms; image language prefill falls from 683.40 to 677.33 ms. The concurrent text prefix remains about 481 ms. Profiler overhead makes these timings higher than normal HTTP TTFT, and raw concurrent kernel sums must not be added to either table. Raw vision GEMM duration increases as more of its execution overlaps language, despite the shorter vision span; this is not an intrinsic GEMM regression measurement.

Two probes were not retained. Hoisting K/V casts after permuting the full tensor removed launches but left inefficient strided conversion: raw copy time barely changed, from 35.98 to 34.11 ms, and TTFT did not improve. Casting in the original contiguous layout reduced copy time to 15.69 ms. Dense flat RoPE indexing with 256-thread blocks added indexing/bounds work and measured 1237.31 ms versus 1236.57 ms for the preceding 64-thread geometry candidate, so it was reverted. These exploratory results are single-process measurements, not additive savings.

Remaining targets include sharing the three identical normalized-input conversions for vision Q/K/V (96 casts / 20.35 ms raw, only two of the three redundant), fusing K's RoPE with its required F16 store, and reducing FP32 gate/up output traffic before language GLU/packing (44.50 ms raw). Most language GLU output is already F16 or packed Q8. Narrowing MMQ outputs requires correct final rounding after stream-K accumulation; adding a standalone cast could erase the benefit. Language FFN gate/up GEMMs remain the largest workload at 578.75 ms raw with existing pipelining enabled. These raw times overlap and are not additive TTFT opportunities.

Validation passes ROPE, RMS_NORM, RMS_NORM_MUL_ADD, MUL_MAT_BIAS, F16_FFN_BIAS, BF16_FFN_BIAS and FLASH_ATTN_EXT_SET backend groups, including extra consumers, residual dependencies, forced-F32 fallbacks, boundary dimensions and in-place/out-of-place assembly. The rebuilt production library also passes the affected FFN, matrix/bias and attention/SET groups. The ordinary server/test build passes, and changed HIP units compile for gfx1100 and gfx942; runtime testing is on gfx1151. NVIDIA was not compiled or tested.

All 140 measured normal requests across the tested variants retain the same full generated text. Three image checks retain identical 32-token generations and finite embeddings, but the final embeddings are not bitwise equivalent:

| Image | Embedding NMSE | Maximum absolute difference | Maximum chosen-token log-probability difference |
| --- | ---: | ---: | ---: |
| Uniform 1024x800 benchmark | 4.48e-4 | 4.5625 | 0.06292 |
| Patterned 224x224 | 3.77e-6 | 0.0390625 | 0.02480 |
| Patterned 1008x756 | 9.44e-6 | 0.238281 | 0.06244 |

Repeating the baseline yields bit-identical embeddings. Geometry alone introduces differences consistent with the changed RMSNorm reduction order. Adding K/V hoisting to geometry is bit-identical on all three inputs, as is adding output assembly to the normalization/residual-fusion candidate. The additional difference is localized to the normalization/residual fusion bundle, without isolation of its individual operations. One repeat of that intermediate bundle changes log probabilities by at most 0.0115 while retaining identical tokens. The uniform image's approximately 2.1% relative RMS embedding difference is material: these checks do not establish zero accuracy loss. Broader image/task evaluation is required before treating the rounding changes as quality-neutral. There is no new Gemma or full VLM-suite speedup claim from this experiment.

Evidence: `/proj/gdba/lichang/regression_results/qwen25vl7b_tensor_traffic_20261027/REPORT.md`, `paired_timing.json`, `verification.json`, `verification_isolation.json`, the baseline/candidate timeline analyses, source snapshots, binary hashes and per-run requests/results. External profiler shutdown required a timeout kill after CSV output was flushed; all three requests in each trace match the complete operation sequence and interval accounting. Normal timing/correctness servers shut down normally. The ordinary `build-rdna35-gfx1151` includes these changes.

## BF16 fusion extension (2026-10-26)

This extends commit `71d2d68e3199810e094529a3bd04134f9afd7921` in the same implementation. The conversion/bias, SwiGLU/GEGLU conversion, biased gated FFN, and GEMM input/output clamp fusions now accept BF16 as well as FP16. Buffers use the original weight/compute type; BF16 GEMMs retain FP32 accumulation and BF16 output. Bias, activation and clamp arithmetic remain FP32. The change removes intermediate FP32 memory traffic and kernel launches; it does not change weights or globally lower activation precision.

| Model / board | Total prompt positions | Before HTTP TTFT | With BF16 fusion | Saving |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-Omni-30B-A3B / Halo40 | 1322 | 1083.64 ms | 1055.22 ms | 28.42 ms / 2.62% |
| Gemma-4-31B OpenNav / Halo05 | 473 | 1832.50 ms | 1815.22 ms | 17.28 ms / 0.94% |

Both model profile comparisons run on Halo40. Each timing variant has 20 measured one-token requests across two server starts in ABBA order, with five warmups per start. The baseline already includes the preceding FP16 fusion and all earlier branch optimizations. Tests retain checkpoints (`--ctx-checkpoints 32`), F16 KV, Q4_K_M language weights, batch 2048, microbatch 512, context 4096, and `ROCBLAS_USE_HIPBLASLT=1`. Both projectors are BF16. Qwen uses the saved 1024x800 image (3200 vision patches); OpenNav uses its saved 640x480 image (1170 patches). Prompt positions include image embeddings. Prompt reuse is disabled and loading time is excluded. No new vLLM run or full VLM-suite sweep is implied.

Qwen3-Omni's plain-GELU vision FFN benefits from projection conversion plus bias. It does not match the larger biased gated-FFN pattern. Gemma-4 OpenNav has no projection biases and benefits from BF16 clamp/conversion fusion. Other compatible BF16 SwiGLU/GEGLU and biased gated FFNs are covered by backend tests, without a separate model-level speedup claim.

Eligibility remains structural: RDNA3.5, the existing default BLAS route, contiguous 2D tensors, at least 32 rows, and the existing precision, environment, consumer and alias checks. A longer gated FFN must have one common weight type across gate/up/down; mixed FP16/BF16 projections fall back. Scratch remains 16 bits per value, and full-FFN scratch has the same lifetime/size tradeoff as the FP16 implementation.

Validation passes 169 focused backend cases across FP16 and BF16, including dispatch boundaries, strided inputs, extra consumers, forced FP32 precision and mixed weight types. A direct test compares against separate existing FP32 operations and conversion kernels: 756 cases / 46,867,248 values match bitwise (including the actual Gemma clamp bounds of +/-FLT_MAX), with no differing NaN payloads and intact output canaries. A BF16-only register compiler barrier before output clamping preserves signaling-NaN behavior across the removed memory boundary. All four 32-token generation checks per target model retain identical tokens/text. Qwen also retains identical chosen/top probability records. Gemma probability records vary within a server process for both baseline and candidate; all 24 checked Gemma responses retain the same 32 tokens. The baseline and production library match every returned probability record request-for-request in the ten-request repeat sequence, confirming pre-existing variation. See the repeated-generation analysis in the report; this is targeted validation, not a full model-quality evaluation. The changed HIP units compile for gfx1100 and gfx942; runtime validation is on gfx1151. NVIDIA compilation/runtime was not tested.

Qwen3-Omni's diagnostic vision GPU duration sum changes from 322.02 to 296.02 ms; vision launches change from 952 to 842. Profile instrumentation affects these timings, so they are separate from HTTP TTFT.

Gemma-4 OpenNav's diagnostic vision GPU duration sum changes from 329.11 to 274.24 ms; vision launches change from 1432 to 1052. Profile instrumentation affects these timings, so they are separate from HTTP TTFT.

Evidence: `rdna35_bf16_fusion_20261026/REPORT.md`, `summary.json`, `profiles.json`, immutable libraries/source snapshots, and per-run requests/results. The ordinary `build-rdna35-gfx1151` build includes the extension.

The subsequent full VLM refresh validates this build on 11 unique llama.cpp workloads covering 14 comparison rows: 220 timed requests, 22 server starts, and 22 passing image sanity checks, with checkpoints 32, zero prompt-cache reuse, and unchanged saved requests. The mean llama.cpp/gfx11-vLLM TTFT ratio is 1.373x, versus 1.443x in the October 16 report. This comparison includes all branch optimizations and retains the saved September 30 gfx11 vLLM baseline, board/date differences, and existing backend input/quantization differences; it does not isolate the BF16 gain above. Full results and validation are in `/proj/gdba/lichang/regression_results/rdna35_vlm_sweep_20261026/comparison.md`.

## Vision bias and gated FFN fusion (2026-10-26)

This change is based on `e01951930c93189bd377d80e5da5688dfcad931a`. It extends the existing FP16 BLAS and producer-fusion paths. It removes FP32 intermediate memory traffic while retaining FP32 bias/activation arithmetic, existing GEMM shapes, and the prior FP16 rounding boundaries.

A matrix-output conversion can now add a following FP32 vector bias before writing the final FP32 graph tensor. For a compatible biased gated FFN, the backend converts the common gate/up input once, retains both GEMM outputs in private FP16 buffers, performs both biases and SwiGLU/GEGLU in FP32 registers, writes the FP16 down-projection input directly, and optionally fuses the down-projection output conversion with bias. This does not change the language Q4_K MMQ kernels or globally switch graph activations to FP16.

| Model / board | Baseline HTTP TTFT | With fusion | Saving | Samples per variant |
| --- | ---: | ---: | ---: | ---: |
| Qwen2.5-VL-7B / Halo40, rebuilt production library | 1405.47 ms | 1325.64 ms | 79.84 ms / 5.68% | 20 |
| Qwen2.5-VL-7B / Halo05, isolated final candidate | 1346.87 ms | 1277.63 ms | 69.24 ms / 5.14% | 20 |
| Gemma-3-4B / Halo40, isolated final candidate | 744.81 ms | 720.92 ms | 23.90 ms / 3.21% | 10 |

The baseline is the already optimized branch before this change. Qwen uses two server processes per variant in ABBA/BAAB order; Gemma uses one paired run. Each process has five warmups and ten measured one-token requests. Tables use pooled medians, retaining timing outliers. Qwen's initial 92.89 ms gain was smaller in the repeat, so that initial sample set is not the headline result. No competing GPU context was detected during accepted requests; clocks and power were not locked. Results from different boards are not mixed.

These tests retain `--ctx-checkpoints 32`, FP16 K/V, Q4_K_M language weights, F16 projectors, batch 2048, microbatch 512, context 4096 and `ROCBLAS_USE_HIPBLASLT=1`. Prompt reuse is disabled and every measured `cache_n` is zero. The saved October 15 request gives Qwen 4292 vision patches and 1606 language prompt positions, and Gemma-3 4096 patches and 779 prompt positions. The latter differs from the earlier 781-position D72 test. No new vLLM comparison was run.

Matched Qwen profiles show vision GPU duration sums **447.21 -> 363.22 ms**, saving **84.00 ms / 18.78%**. The FFN group falls **197.08 -> 117.93 ms**, accounting for most of the improvement. All 32 vision layers match the full fusion. Another 130 standalone matrix-plus-bias groups match. Total launches fall **7368 -> 7046 per request**; the 228 vision GEMMs retain their shapes and approximately the same duration. Profile instrumentation affects timing; these sums are separate from unprofiled HTTP TTFT. Language kernel counts are unchanged.

Dispatch targets RDNA3.5's existing default FP16 BLAS route, contiguous 2D F16-weight/FP32-activation matrices with at least 32 rows, compatible FP32 vector biases, and split SwiGLU/GEGLU for the full FFN. It respects precision/hint/environment overrides, excludes existing MMV/MMF routes, and checks consumers and concurrency boundaries. Input/output graph-buffer reuse is allowed because input reads complete through private scratch before the final write; overlap between the final bias and output is rejected. Unmatched structures retain existing paths. Gemma-3's plain-GELU SigLIP FFN does not match the gated fusion, but compatible projection biases benefit from the shorter fusion. There is no new Gemma-4 performance measurement here.

The full Qwen FFN has about 105 MiB of simultaneous logical FP16 scratch for this image size, reused across layers. More scratch remains live to reduce memory traffic; the graph allocation is unchanged, and net process-memory savings are not claimed. The final candidate and ordinary rebuilt production library each pass 104 relevant backend cases and a direct 63-case check covering 3,905,604 values with bitwise agreement, including FP16 outputs and intact output canaries. All compared 32-token responses and returned chosen/top log probabilities match exactly for both tested models. This is targeted correctness validation, not a broad model-quality evaluation. The changed HIP units compile for gfx1100 and gfx942; runtime validation covers gfx1151.

Evidence: `rdna35_vision_bias_fusion_20261026/REPORT.md`, `final_summary.json`, `profile_comparison.json`, the per-run requests/results and `production/` binary/source hashes. Sources are in `ggml/src/ggml-cuda/{ggml-cuda.cu,convert.cu,unary.cu}` with declarations and cases in the existing backend test suite.

## Commit-by-commit record

This record covers all eleven source-changing commits on `lichang.rdna35-prefill-quant-rope` after the inherited gfx11 base [`0363d99d3`](https://github.com/AMD-Ecosystem/llama.cpp/commit/0363d99d3b276a5cfb8d2ab2940cd1a9a7e1d46b). It excludes inherited upstream changes. The documentation-only update titled `docs: record RDNA3.5 optimization targets and measured gains` adds this audit and has no runtime performance effect.

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
| [`71d2d68e3`](https://github.com/ROCm/llama.cpp/commit/71d2d68e3199810e094529a3bd04134f9afd7921) | Share gate/up input conversion and fuse FP16 GEMM output conversion, FP32 bias and gated activation | Qwen2.5-VL-7B saves 79.84 ms / 5.68% HTTP TTFT on Halo40 and 69.24 ms / 5.14% on Halo05; Gemma-3-4B saves 23.90 ms / 3.21%. |
| [`667af94f0`](https://github.com/ROCm/llama.cpp/commit/667af94f05375369079f549225067c315effe0b6) | Extend the same producer fusions to BF16 | Qwen3-Omni-30B-A3B saves 28.42 ms / 2.62%; Gemma-4-31B OpenNav saves 17.28 ms / 0.94% HTTP TTFT. |
| [Vision tensor traffic and output fusion (2026-10-27)](#vision-tensor-traffic-and-output-fusion-2026-10-27) | Hoist window K/V conversion, write attention directly to final output, fuse normalized FFN input and residual conversion, tune RoPE/RMSNorm blocks | Qwen2.5-VL-7B saves 62.96 ms / 4.80% HTTP TTFT; checkpoints-on control saves 54.69 ms / 4.20%. |

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
