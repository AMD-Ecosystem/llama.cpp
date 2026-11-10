# Qwen2.5-VL window batching on gfx11

This focused branch starts at ROCm llama.cpp `gfx11` revision `f70aba1cb60547b13a6d2f1ef70c158dd59ca51e`. It contains the window-attention graph transformation and its two input/output follow-ups. Measurements below were collected for this branch on November 10, 2026.

## Scope

Qwen2.5-VL local-attention layers become batches of independent windows instead of a single large masked attention operation. Consecutive windows of equal token count share a batch entry group; partial edge windows form separate groups. Existing patch order, position IDs, global-attention layers, image preprocessing, and the output projection are preserved.

The core change also skips allocation/upload of the unused dense window mask and grows graph/scheduler metadata capacity with the number of window groups. Scheduler evaluation callbacks are preserved when capacity grows. The first direct port exposed the fixed metadata-pool limit during model warmup; capacity growth is required support for batching, not an additional measured speedup.

Two companion changes convert complete K/V tensors before taking window views, and let eligible RDNA3.5 attention operations write directly into the joined output buffer. The latter retains checks for consumers, aliasing, alignment, output bounds, and temporary-buffer requirements.

No MMDQ changes, new attention kernels, WMMA dispatch changes, or language-model fusions are included.

## Controls

Each switch defaults to enabled. Set it to `0` in a new process to select the fallback.

| Switch | Function | Prerequisite |
| --- | --- | --- |
| `MTMD_QWEN25VL_WINDOW_BATCH` | Window batching, mask suppression, and graph-capacity support | Qwen2.5-VL window attention with Flash Attention enabled |
| `MTMD_QWEN25VL_KV_CAST_EARLY` | Convert K/V once before window views | Window batching |
| `GGML_CUDA_FUSE_ATTN_SET` | Write attention output directly into its joined buffer | Eligible attention -> reshape -> in-place SET graph on RDNA3.5 |

The three switches define one tested package. This experiment does not attribute an independent gain to either companion switch.

## One-model verification

- Board: `xconucstrhalo06`, Strix Halo / gfx1151.
- Model: Qwen2.5-VL-3B-Instruct, Q4_K_M language weights and FP16 projector.
- Input: the saved 1024 x 800 JPEG and original text request; **1,606 prompt positions** on both builds.
- Tool: `llama-server`, streamed HTTP TTFT to the first nonempty content token, measured identically for both builds.
- Settings: context 4096, batch and microbatch 2048, 8 CPU threads, one slot, full GPU offload, Flash Attention enabled, FP16 K/V cache, checkpoints 0, prompt caching disabled, temperature 0, output cap 32.
- Timing order: baseline, candidate, candidate, baseline. Each process has two untimed warmups and five retained requests; the table uses the median of ten retained samples per build. Loading and model staging are outside the timed interval.

| Configuration | Median TTFT ms | Relative to gfx11 |
| --- | --- | --- |
| Unchanged gfx11 | 2158.59 | 1.00x |
| Window package enabled | 974.06 | 2.216x faster; 54.88% less time |
| Candidate with all three switches disabled | 2163.24 | Fallback check; three retained samples |

One candidate request took 2,059.42 ms and remains included in the ten retained samples. Candidate process medians were 972.41 and 974.41 ms; baseline process medians were 2,156.07 and 2,161.65 ms.

The comparison uses matching model hashes, request hashes, prompt positions, and effective workload settings. No retained sample has a detected foreign-job overlap or cached prompt tokens. The initial Halo41 run was excluded after that board became occupied; both sides of the reported comparison were measured on Halo06.

## Correctness and build validation

- Release HIP build completed with the same ROCm SDK and compiler settings as the baseline. Target gfx1151, VMM disabled, unsafe math disabled.
- `test-backend-ops test -b ROCm0 -o FLASH_ATTN_EXT_SET`: 18/18 cases passed with fusion enabled, and 18/18 with it disabled, against the CPU reference. Cases include small/full/partial attention lengths, multiple batch entries, an additional consumer, and out-of-place SET fallbacks.
- Separate model verification generated identical text: **true**. Common generated-token prefix: 32 of 32 baseline tokens; maximum absolute log-probability difference over that prefix: 0.015620947.
- Final vision embedding shape: 1073 tokens x 2048 channels. Relative L2 difference: 0.0078391009; normalized mean square error: 6.1451503e-05; cosine similarity: 0.9999693038.
- These checks establish numerical agreement for the tested operators and request. They are not a model-wide accuracy benchmark or validation of other GPU backends.

## Reproduction and raw evidence

The [experiment directory](/proj/gdba/lichang/regression_results/qwen25vl_window_gfx11_20261110) contains the frozen request, source patch, binary hashes, exact commands, per-request timings, model output, and cleanup records. See [analysis.json](/proj/gdba/lichang/regression_results/qwen25vl_window_gfx11_20261110/analysis.json), [source_state.json](/proj/gdba/lichang/regression_results/qwen25vl_window_gfx11_20261110/source_state.json), and [build manifest](/proj/gdba/lichang/regression_results/qwen25vl_window_gfx11_20261110/builds/candidate/manifest.json).

Build:

```bash
python3 /proj/gdba/lichang/regression_results/qwen25vl_window_gfx11_20261110/build.py candidate /proj/gdba/lichang/llama.cpp-qwen25vl-window-batch-gfx11
```

The exact benchmark commands and runtime settings are saved in each run's `metadata.json`. The driver is [run_verification.py](/proj/gdba/lichang/regression_results/qwen25vl_window_gfx11_20261110/run_verification.py); use a new artifact directory or labels when collecting another run.
