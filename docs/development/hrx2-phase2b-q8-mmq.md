# HRX2 Phase 2b Q8_0 Prompt MMQ

Date: 2026-06-16

## Context

Llama 3.1 8B Q8_0 prefill was a clear Phase 2b blocker. The previous HRX2
prompt route used `hrx2_mul_mat_q8_0_f32_static_packed`, which assigns one
workgroup to one output element and streams F32 RHS values. Same-run baseline
before this pass was approximately:

- p64: HRX2 steady ~98 tok/s, Vulkan steady ~1517 tok/s, ratio ~0.064.
- p512: HRX2 steady ~102 tok/s, Vulkan steady ~2359 tok/s, ratio ~0.043.

## Accepted Route

Added:

- `mul_mat_q8_0_q8_1_x4_mmq64x32_k256_32768_r1_262144_c32_4096_wg256`
- Loom export: `hrx2_mul_mat_q8_0_q8_1_x4_mmq64x32_static`
- Rollback: `GGML_HRX2_DISABLE_Q8_0_Q8_1_PROMPT=1`

The route reuses the existing Q8_1 x4 RHS quantization substrate and mirrors the
existing HRX2 Q5/Q6 MMQ schedule:

- 64 rows x 32 columns per workgroup.
- 256 workitems.
- RHS Q8_1 x4 tile staged in LDS.
- One lane row and eight output columns per lane group.
- Explicit `vector.dot4i<s8s8>` over Q8_0 LHS and Q8_1 RHS payload words.

The route is guarded to `cols_multiple_of=32`; scalar packed routes remain the
fallback for narrow and irregular prompt shapes.

## Validation

Build:

```bash
cmake --build build/llama-hrx2 --target ggml-hrx2 -j"$(nproc)"
```

Focused model-derived op export:

```bash
build/llama-hrx2/bin/export-graph-ops \
  -m shared/models/llamacpp-hrx2-basket-v1/bartowski__Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf \
  -p "hello ..." -n 0 -ngl all -dev HRX20 -b 64 -ub 64 \
  -o cache/hrx2/phase2b/q8-mmq-opgate-20260616/llama31-q8-p64-ops.txt
```

Correctness:

```bash
build/llama-hrx2/bin/test-backend-ops test -b HRX20 -o MUL_MAT \
  --test-file cache/hrx2/phase2b/q8-mmq-opgate-20260616/llama31-q8-p64-ops.txt \
  --output csv
```

Result: passed for all exported Q8_0 `MUL_MAT` rows, including p64 prompt
shapes:

- Qcur: `q8_0[4096,4096] x f32[4096,64]`
- ffn_out: `q8_0[14336,4096] x f32[14336,64]`
- ffn_gate: `q8_0[4096,14336] x f32[4096,64]`
- result_output: `q8_0[4096,128256] x f32[4096,64]`

## Focused Op Perf

Same exported test file, `test-backend-ops perf`, compared with
`GGML_HRX2_DISABLE_Q8_0_Q8_1_PROMPT=1`.

| Shape | New route us | Scalar fallback us | Speedup |
| --- | ---: | ---: | ---: |
| Vcur `k4096 r1024 c64` | 114.8 | 243.4 | 2.1x |
| Qcur `k4096 r4096 c64` | 233.3 | 1155.2 | 5.0x |
| ffn_out `k14336 r4096 c64` | 813.6 | 3781.9 | 4.6x |
| ffn_gate `k4096 r14336 c64` | 691.0 | 4189.1 | 6.1x |
| result_output `k4096 r128256 c64` | 6489.3 | 59132.5 | 9.1x |

## Model Perf

Same-run `llama-bench`, Llama 3.1 8B Q8_0, FA on, `-b 512 -ub 512`, 3 reps,
no warmup:

| Case | HRX2 steady tok/s | Vulkan steady tok/s | Ratio |
| --- | ---: | ---: | ---: |
| p64 | ~399 | ~2014 | ~0.20 |
| p512 | ~536 | ~2778 | ~0.19 |

Route trace for both p64 and p512:

- CPU compute fallback: 0
- `mul_mat_q8_0_q8_1_x4_mmq64x32...`: 663 dispatches
- `quantize_q8_1_x4_f32_generic_wg128`: 378 dispatches
- `quantize_cache_hit`: 285 events

## Bottom Line

This is an accepted boulder fix, not a done-done Q8_0 prompt matmul. It removes
the scalar one-output-per-workgroup cliff and improves Q8_0 prefill about 4-5x
at model level, but HRX2 remains about 5x behind Vulkan on the same Q8 prefill
bucket.

Next work should compare against Vulkan/CUDA/HRX1 schedules with a stricter
tile search. The likely missing class is a stronger prompt matmul schedule that
raises arithmetic intensity beyond RHS-only staging: larger output tiles, A-side
reuse/staging, or target-specific dot/MMA forms if Loom can spell them cleanly.
