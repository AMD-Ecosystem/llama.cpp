# F32 tiled accumulation and LDS reuse experiment

This branch is an accuracy/performance experiment for bead lemonade-ws-ve03, discovered during ROCm/ggml-staging-automation#161. It changes the shared tiled matmul core, so consumers beyond Q4 need validation before adoption. It is separate from the row64 rollout PR.

## Comparison revisions

- F16 baseline: `b47fa040fb88a0c85388f22404393dd541a0e359`.
- F32 accumulation with separate result LDS: `da8e9bbb0`.
- F32 accumulation with reused activation LDS: this branch's tip.
- Tested HRX/Loom revision: `82214dfb029975205d1c3a59164b678e2e21fc41`.

The candidate core is byte-identical to the gfx1100-tested source, SHA256 `39bc0acf26f0daf2de0615c783515dc0fa343e183f50427d34ec35a03ec9138c`.

The core is `kernel-corpus/kernels/loom-libs/motifs/matmul/tiled_input_f32_dequant_accumulate_f32.loom`, relative to this backend. Operands stay F16. Wave64 accumulator fragments change from `vector<8xf16>` to `vector<4xf32>`, and result staging stays F32 through publication. Reuse removes the separate result allocation and places two disjoint 1024-byte result regions into the 2560-byte activation allocation. The final K-group workgroup barrier completes activation reads before publication starts. No new barrier is needed.

## gfx1151 worker handoff

Start with kernel-level `loom-benchmark` comparisons of all three revisions, using identical nonzero inputs, quantized weights, launch geometry, epilogue and compiler revision. Build for the actual gfx1151 device; do not reuse gfx1100 binaries or hard-coded device IDs. Record the exact APU SKU, GPU agent, runtime/compiler revisions, memory configuration, power mode and clocks. Pin the GPU according to the workspace GPU-selection guide.

Use the existing `tools/benchmarks/run-model-benchmarks.py` runner to link benchmark manifests and obtain the direct `loom-benchmark` command, or use the same tools directly. The entry point is `ggml_mul_mat_tiled_input_f32_publish_f32`, with `_aligned` for token counts divisible by 32, target `ggml_matmul_tiled_input_f32_publish_f32_gfx11_wave64`. Verify the selected ISA and actual launch instead of assuming the entry point name guarantees gfx1151 selection.

Match these specializations: `ggml.mul_mat.input_size=K`, `ggml.mul_mat.output_size=N`, `ggml.mul_mat.weight_format=4` (Q4_K native) or `44` (Q4_K row64), `ggml.mul_mat.output_unary_op=15` (SiLU), and `ggml.workload.token_capacity=T`. Use the same remaining parameters for each variant. This is a shared-core stress benchmark, not a claim that the model scheduler selects this kernel for every shape.

First check the original accuracy cases K2560/N1024/T66 and T128, both layouts. Then reproduce the larger sweep:

| K | N | Tokens |
|---:|---:|---:|
| 2560 | 10240 | 66 |
| 2560 | 10240 | 129 |
| 2560 | 10240 | 512 |
| 2560 | 10240 | 513 |
| 5120 | 13824 | 512 |
| 5120 | 13824 | 1024 |
| 12288 | 4096 | 66 |
| 12288 | 4096 | 512 |

For each shape and both layouts, run F16/F32/reuse/reuse/F32/F16 twice (four runs per variant, 192 runs for the larger sweep). Use ten warmups, batch size eight, four input bindings, at least 50 batches and 300 ms per run, maximum 1000 batches, and final-batch GPU profiling. Keep uploads/fixture creation outside kernel timing. Report median and range of device-profile run means separately from host operation timing; do not equate host completion with GPU execution time. If a difference is near run-to-run noise, report it as unresolved rather than claiming no slowdown.

Inspect compiler reports for F32 WMMA selection, LDS allocation, VGPR count, private spills, occupancy and launch geometry. Compare reuse output bitwise against the F32 separate-LDS control on the same device; compare native against row64. Check finiteness and the independent dequantized-weight/F32-input matmul plus SiLU reference with the existing tolerance, without weakening it. The larger gfx1100 checks sampled all N channels at tokens 0, 21, 43 and T-1; the original small checks covered every output. Retain hashes and full outputs for layout equivalence.

Record results on lemonade-ws-ve03. A kernel-level pass does not establish model correctness or end-to-end speed. Before adopting the shared core, validate other consumers/formats (including Q6), fused epilogues, the HRX operation tests, model performance and perplexity. The parent tests contain an inherited Q4 accuracy exception; this branch does not yet remove it.

## Existing gfx1100 evidence

RX 7900 XTX, matched nonzero sweep, four runs per variant. K2560/N10240/T513 row64 device medians:

| Variant | Device microseconds | LDS bytes | Reported resident subgroups/SIMD |
|---|---:|---:|---:|
| F16 | 809.855 | 8704 | 7 |
| F32 separate LDS | 854.855 | 9728 | 6 |
| F32 reuse | 796.210 | 7680 | 8 |

All 16 larger reuse outputs were finite, bitwise identical to the F32 control, and passed sampled independent reference checks. Across row64 shapes, reuse versus F16 device changes ranged from -1.83% to +1.96%. Native K5120/N13824/T512 measured +4.50% on device but +0.52% on host. These results support addressing the unaligned LDS penalty, not a universal performance-neutral claim. All candidate shapes used 7680 bytes LDS with no private spills.
