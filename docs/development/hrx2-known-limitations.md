# HRX2 Known Limitations

This file tracks concrete HRX2 implementation limitations that should survive
individual tuning sessions. General Loom author feedback belongs elsewhere;
this file is for llama.cpp backend behavior and follow-up work.

## Removed HIP Bridge Kernels

Temporary HIP C++ bridge kernels for Q4_K, Q5_K, Q6_K, and FA0 attention have
been removed from the production HRX2 build and catalog. Their schedules are
preserved as prior art in `docs/development/hrx2-hip-bridge-prior-art.md`.

Current status:

- Q4_K, Q5_K, and Q6_K route coverage uses Loom bytecode providers and JIT
  output only.
- Focused backend-op gates for the affected quantized matmul routes pass.
- A reduced Llama 3.2 3B Q4_K HRX2/Vulkan integration smoke runs with zero CPU
  compute fallback.
- FA0 direct attention fusion has no accepted Loom replacement yet; production
  uses the unfused Loom/HRX2 attention route chain.

## Remaining Performance Gaps

The no-HIP bridge smoke from 2026-06-16 shows HRX2 is functional but still below
Vulkan on the representative Q4_K slice:

| Case | HRX2/Vulkan |
| --- | ---: |
| `decode-p1n64` | 0.376 |
| `prefill-p64n0` | 0.150 |
| `prefill-p512n0` | 0.170 |

The largest p64/p512 blocker is quantized prompt matmul route quality, especially
`mul_mat_q4_k_q8_1_x4_mmq64x32`. Future work should start from prior schedules,
Loom compile reports, emitted assembly, and focused kernel sweeps before
promoting integration routes.

A narrow Loom port of the Q4_K VKM64x64 TM4/TN1 HIP prior is accepted only for
`k=3072, rows=8192, cols=64`, where it improves the Llama 3.2 3B ffn_gate prompt
row from about 415 us to 310-320 us. It is intentionally not a general Q4_K
prompt replacement: the same spelling regressed Kcur/Qcur/ffn_out c64 and p512
ffn_gate. The compile report has zero spills/hazard gaps but high peak live
units, so the next broad Q4 attempt should reduce live state or use a different
spelling rather than simply widening this provider.

A broader Loom Q4_K VKM64x64 TM2/TN2 route is accepted for p512 prompt rows
with `cols=512`. It roughly halves Llama 3.2 3B Q4_K_M p512 HRX2 integration
time versus the post-HIP-removal Loom baseline, improving steady prefill to
about 1445 tok/s, and focused backend-op timings improve all p512 Q4 rows.
This is still only about 0.24x same-run Vulkan steady throughput and remains
well behind the removed HIP bridge prior for equivalent rows. Treat it as a
real production improvement, not as proof that the Q4 prompt matmul family is
done. The next Q4 work should compare emitted ISA/load/staging facts against
the removed HIP bridge and Vulkan shader and close the remaining schedule or
codegen delta before spending time on smaller route knobs.

The current Q4 p512 delta has a concrete ISA lead. The accepted Loom route
matches the HIP wave32 `BM64/BN64/WG128/TM2/TN2` prior at the source-schedule
level, but emits a much larger fully-unrolled body: 2048 static dot ops,
1472 LDS instructions, 16 barriers, and 333 peak live units versus the HIP
prior's 256 static dot ops, 55 LDS instructions, 2 barriers, and 95 VGPRs. A
probe removing only the Loom `%group` loop unroll failed in `source-to-low`
while materializing branch arguments. Until that lowering path is fixed or the
kernel is respelled with equivalent loop-carried state, Q4 p512 is expected to
remain behind Vulkan/HIP despite using the right high-level tile.

Q8_0 prompt matmul remains an A+B-staging problem, not a simple row-ownership
problem. A scalar Loom port of the HRX1 `BM128/BN32/WG256` RHS-only schedule
compiled cleanly but failed focused backend-op correctness with NaNs. A
diagnostic `BM128/BN16` route proved the 128-row lane ownership can be correct,
but same-runner backend-op perf mostly regressed versus the accepted
`BM64/BN32` route. The next Q8 prompt attempt should port Vulkan's
`matmul_q8_0_q8_1` integer-MMQ dataflow with both A and B staged in LDS and
register-cached multi-row/multi-column dotting, rather than continuing
RHS-only tile-size pivots.

Generated catalog state is part of the benchmark environment. If a route is
reverted or removed in source, rebuild `ggml-hrx2` before taking a new
baseline; otherwise the embedded generated catalog in the build tree can still
contain and select a rejected provider. Confirm route stderr or
`GGML_HRX2_TRACE_JSONL` for every focused `test-backend-ops perf` run.

## Qwen 7B p512 Route Coverage

On 2026-06-16, Qwen2.5-Coder 7B Q5_K_M/Q6_K p512 still had CPU fallback for
hidden-size 3584 `RMS_NORM` and NEOX `ROPE` at `[128, 28, 512]` and
`[128, 4, 512]`. The catalog now has generic target-neutral routes for:

- `rms_norm_f32_n3584_r512_vector_vw4_wg512`
- `rope_neox_f32_n128_h28_t512_wg256`
- `rope_neox_f32_n128_h4_t512_wg256`

Focused model-derived `test-backend-ops` rows passed against CPU and real
`llama-bench` p512 traces now show zero provider misses and zero CPU fallback
for both Q5_K_M and Q6_K. Artifact:
`cache/hrx2/phase2b/qwen-p512-coverage-20260616/`.

This was a structural coverage fix, not a major throughput fix. Same-run p512
throughput after the route coverage change:

| Model | HRX2 tok/s | Vulkan tok/s | HRX2/Vulkan |
| --- | ---: | ---: | ---: |
| Qwen2.5-Coder 7B Q5_K_M | 456.6 | 2577.0 | 0.177 |
| Qwen2.5-Coder 7B Q6_K | 439.4 | 2416.8 | 0.182 |

The remaining Qwen p512 gap is therefore in the quantized prompt matmul and
nearby hero fusion schedule, not missing RMS_NORM/ROPE route coverage.
