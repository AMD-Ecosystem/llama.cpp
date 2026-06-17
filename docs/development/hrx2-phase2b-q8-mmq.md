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
- Dot results are accumulated as per-column i32 `qsum` values across the eight
  Q8 payload words in a K block, then converted/scaled once per K block. This
  matches the HRX1 schedule more closely than the initial Loom spelling, which
  converted and applied A/B scales inside the `iqs` loop.

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

After the i32-qsum refinement on the same route, short HRX2-only checks showed:

| Case | Cold tok/s | Steady samples tok/s | Steady mean tok/s |
| --- | ---: | ---: | ---: |
| p64 | 193.8 | 457.5, 468.5 | 463.0 |
| p512 | 520.8 | 636.2, 642.1 | 639.1 |

The qsum route compiled cleanly for `k=4096, rows=4096, cols=64`: zero spills,
zero hazard gaps, peak live units 83, 64 dot ops, and instruction count 447
versus 585 for the earlier per-`iqs` scaling form.

Rejected probe: a compact vector-loop `mmq128x32` Loom spelling based on the
HRX1 tile shape did not promote. Rooted direct compile reached the intended
export but emitted hundreds of SGPR spill-storage warnings before termination.
The rejected patch is preserved in the workspace cache at
`cache/hrx2/phase2b/q8-mmq128x32-20260616/rejected/vector-loop-128x32-rejected.patch`.
The next 128x32 attempt should use explicit scalar lane ownership or a lower
level spelling rather than dynamic vector insert/extract loops.

Rejected follow-up probes after the compiler-report refresh:

- `mmq128x32_scalar`: explicit scalar lane ownership matching the HRX1
  `BM128/BN32/WG256/16-cols-per-lane` schedule compiled cleanly for
  `k=4096, rows=4096, cols=64`: zero spills, zero hazard gaps, 72 peak live
  units, 128 dot instructions, 27 global-memory instructions, 146 LDS
  instructions, 2 barriers, 1054 instructions, and 5936 code bytes. Focused
  backend-op CPU-reference testing failed every Q8 row with NaN at output index
  0 while route traces confirmed the candidate was JIT compiled and selected.
  The generated source/catalog patch is preserved at
  `cache/hrx2/phase2b/q8-mmq128x32-scalar-debug-20260616-182150/candidate-current.patch`.
  Disassembly showed the intended `v_dot4_i32_iu8` form and no `ds_*_addtid`
  LDS addressing, so this is not the old LDS addtid bug. The current hypothesis
  is a Loom/codegen issue or a high-level spelling hazard triggered by the
  16-output-column scalar state, not the Q8_1 x4 layout or route metadata.
- `mmq128x16_scalar`: diagnostic bracket using the same 128-row lane ownership
  but the accepted eight-output-column accumulator shape passed focused
  backend-op correctness for the Llama 3.1 8B Q8_0 p64 prompt rows and route
  traces confirmed selection. Same-runner `test-backend-ops perf` rejected it
  against the accepted `mmq64x32` baseline:

  | Shape | `128x16` us | `64x32` us | Result |
  | --- | ---: | ---: | --- |
  | Vcur `k4096 r1024 c64` | 111.91 | 101.84 | slower |
  | Qcur `k4096 r4096 c64` | 176.49 | 180.11 | marginally faster |
  | ffn_out `k14336 r4096 c64` | 657.72 | 642.25 | slower |
  | ffn_gate `k4096 r14336 c64` | 600.33 | 526.06 | slower |
  | result_output `k4096 r128256 c64` | 6590.06 | 5262.39 | slower |

  The rejected patch and traces are preserved at
  `cache/hrx2/phase2b/q8-mmq128x16-scalar-diag-20260616-182522/`.

These probes establish that 128-row ownership alone is not the missing Q8
boulder. The next useful Q8 prompt candidate should not be another RHS-only
row-count pivot. It should port the Vulkan integer-MMQ dataflow more directly:
stage both A and B tiles in LDS, use a `BM/BN/BK_STEP` family similar to
Vulkan's `matmul_q8_0_q8_1` pipeline, cache multiple A rows and B columns in
registers, and let route metadata/tuning select shape buckets.

Route trace for both p64 and p512:

- CPU compute fallback: 0
- `mul_mat_q8_0_q8_1_x4_mmq64x32...`: 663 dispatches
- `quantize_q8_1_x4_f32_generic_wg128`: 378 dispatches
- `quantize_cache_hit`: 285 events

## Bottom Line

This is an accepted boulder fix, not a done-done Q8_0 prompt matmul. It removes
the scalar one-output-per-workgroup cliff and improves Q8_0 prefill about 4-6x
at model level. The i32-qsum refinement gives another ~16-19% steady prefill
lift on the Llama 3.1 8B Q8_0 p64/p512 checks, but HRX2 remains well behind
Vulkan on the same Q8 prefill bucket.

Next work should implement the Vulkan-style integer-MMQ schedule rather than
continuing local RHS-only MMQ pivots. The relevant Vulkan prior stages both A
and B, uses `BK_STEP=4` for non-`MUL_MAT_ID` integer MMQ, has `BM/BN` route
families around `64/64` and `128/128`, and computes multiple rows and columns
per lane from register-cached `block_a_cache` and `block_b_cache` values.
