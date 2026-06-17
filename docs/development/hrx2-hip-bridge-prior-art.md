# HRX2 HIP Bridge Prior Art

HRX2 no longer compiles or routes temporary HIP C++ bridge kernels in the
production backend. The removed bridge sources remain useful as schedule prior
art through git history and the HRX1 catalog, but new HRX2 production kernels
should be authored in Loom and validated with Loom compile reports, target
listings, backend op tests, and route traces.

Removed HRX2 bridge commits:

- `e909aef98` added the first Q4_K prompt bridge.
- `10a42bc10` and `37828eb12` added Q4_K Vulkan-medium and wave32 variants.
- `da3e2059` added Q5_K/Q6_K prompt bridges.
- `c08919e60` and `b884fff06` added Q6_K/Q5_K wave32 prompt variants.
- `b56c1085c` added the FA0 attention prefill HIP bridge.

Schedule facts worth preserving for Loom candidates:

- Q4_K prompt bridge: Vulkan medium K-quant integer MMQ family, `BM64`,
  `BN64`, workgroup 128 for wave32 and 128 or 64 for earlier wave64 pivots,
  packed Q8_1 x4 RHS, Q4 A-side pack/cache, `u8s8` dot contract, cols64 prompt
  shapes as the strongest production signal.
- Q5_K prompt bridge: packed Q8_1 x4 RHS, HRX1/Vulkan-style tiled MMQ,
  `BM64/BN64` wave32 for the narrow production row and `BM128/BN128` wave64 as
  an older HRX1 prior. The Loom catalog keeps Q5_K MMQ/direct routes for
  correctness and tuning.
- Q6_K prompt bridge: packed Q8_1 x4 RHS, Vulkan-medium `BM64/BN64` wave32
  cols64 rows for important basket shapes and HRX1 `BM64/BN128` wave64 as an
  older broader prompt prior. The Loom catalog keeps Q6_K MMQ/direct routes for
  correctness and tuning.
- FA0 attention bridge: direct FA0 graph fusion of `KQ -> SOFT_MAX -> KQV ->
  PERMUTE -> CONT` for D128, adapted from HRX1 direct flash-attention. It used
  BR16/BC64/BK16 style ownership with WMMA QK/PV and online softmax. There is
  no accepted Loom replacement in this cleanup; the unfused Loom/HRX2 routes
  are the production fallback until a Loom FA0 fusion is authored and proved.

Rejected Loom probe:

- Q5_K `x4_mmq64x64` high-level Loom port from the HRX1 WG256, RHS-LDS,
  one-row/16-cols-per-lane schedule was rejected. Direct Loom compile for
  `k=3584, rows=18944, cols=512` reported status OK, zero spills, zero hazard
  gaps, peak live units 133, LDS 2560 bytes, and 128 dot ops, but the route
  failed Qwen p512 backend-op correctness when admitted for `cols=512`
  (`ERR ~= 0.78`). Narrowing the route to `cols=64` passed the focused p64
  CPU-reference gate, but same-runner `test-backend-ops perf` regressed the
  Phi-4 Q5 `wqkv k3072 r5120 c64` row from 304.06 us on the existing
  `x4_mmq32x32` Loom route to 550.59 us. The rejected patch is preserved at
  `cache/hrx2/phase2b/q5-mmq64x64-cols64-rejected-20260616/rejected-candidate.patch`.
  Do not retry this 64x64 high-level spelling as a production default without
  a new schedule/codegen hypothesis. The old HRX1/Pyre ledger also rejected a
  Q5 64x64 default and instead points at the larger Q5 128x128/BK_STEP=1
  family as the stronger prompt prior.
- Q6_K `x4_mmql64x64` WG256 A+B-staged Loom probe was rejected as a p64
  production route. It passed the focused CPU-reference gate after being
  narrowed to `cols=64`, and route stderr confirmed
  `hrx2_mul_mat_q6_k_q8_1_x4_mmql64x64_static` was JIT compiled and selected
  for the prompt rows. Same-runner `test-backend-ops perf`, however, regressed
  `Vcur k3072 r1024 c64` from 178.83 us on the current Loom `mmq64x32` route
  to 394.42 us, and regressed `ffn_out k8192 r3072 c64` from 561.46 us to
  955.32 us. The removed HIP wave32 bridge was 161.92 us and 336.06 us on the
  same rows. The rejected patch is preserved at
  `cache/hrx2/phase2b/q6-mmql64x64-c64-probe-20260616-171538/rejected-q6-mmql64x64-c64-probe.patch`.
  Do not retry this WG256/wave64-like 64x64 spelling as a production default.
  The useful Q6 prior is the HIP bridge's actual BM64/BN64/BK_STEP=4,
  workgroup-128 wave32 schedule with four wave32 tiles per workgroup,
  two rows by two columns per lane, eight column sub-iterations, and both
  A and B staged in LDS.

When reusing any of these schedules, create a Loom candidate row before coding:

- prior source or commit;
- target model shape bucket;
- tile/workgroup/subgroup;
- lane ownership and per-lane outputs;
- vector and packed load width;
- dot/WMMA primitive and signedness;
- LDS staging, barriers, unroll, reduction, and writeback policy;
- compile-report checks for registers, spills, instruction mix, LDS/bank
  conflicts, hazards, and emitted target listing;
- focused backend-op correctness gate and promotion rule.

## Cleanup Evidence

The production cleanup removed every HRX2 CMake path that compiled `-x hip`
bridge sources and removed bridge route/source/artifact records from the split
catalog. A rebuilt generated catalog has no checked-in `amdgpu-hsaco` artifacts;
runtime HSACO still appears only as Loom JIT output.

Focused backend-op gates were run first for the affected Q4_K, Q5_K, and Q6_K
families using traced route evidence:

- Q4_K: 8 focused `MUL_MAT` rows passed, selecting Loom direct, cols4, and
  `x4_mmq64x32` routes plus Q8_1 quantization routes.
- Q5_K: 3 focused rows passed, selecting Loom dot16, direct-cols4, and
  `x4_mmq32x32` routes.
- Q6_K: 10 focused rows passed, selecting Loom rows2, direct-cols4, and
  `x4_mmq64x32` routes.

The sampled Loom compile reports for those routes had status `OK`, zero hazard
gaps, and zero allocation spills. Peak live units in the samples were:

| Route export | Peak live units | LDS bytes | Dot instructions |
| --- | ---: | ---: | ---: |
| `hrx2_mul_mat_q4_k_f32_static` | 30 | 32 | 0 |
| `hrx2_mul_mat_q4_k_q8_1_f32_cols4_static` | 51 | 128 | 4 |
| `hrx2_mul_mat_q4_k_q8_1_x4_mmq64x32_static` | 110 | 2688 | 64 |
| `hrx2_mul_mat_q5_k_f32_dot16_static` | 54 | 0 | 0 |
| `hrx2_mul_mat_q5_k_q8_1_x4_direct_cols4_static` | 53 | 128 | 4 |
| `hrx2_mul_mat_q5_k_q8_1_x4_mmq32x32_static` | 85 | 1280 | 64 |
| `hrx2_mul_mat_q6_k_f32_rows2_wg32_static` | 51 | 0 | 0 |
| `hrx2_mul_mat_q6_k_q8_1_x4_direct_cols4_static` | 30 | 128 | 4 |
| `hrx2_mul_mat_q6_k_q8_1_x4_mmq64x32_static` | 98 | 1088 | 64 |

Reduced integration smoke after removing the HIP bridges:

```sh
python3 tools/hrx2_phase2a_benchmark.py \
  --tag no-hip-loom-smoke-20260616 \
  --models llama32-3b-q4 \
  --cases decode-p1n64,prefill-p64n0,prefill-p512n0 \
  --backends hrx2,vulkan --repetitions 1 --timeout 900
```

| Case | HRX2 tok/s | Vulkan tok/s | HRX2/Vulkan | CPU compute |
| --- | ---: | ---: | ---: | ---: |
| `decode-p1n64` | 47.460 | 126.371 | 0.376 | 0 |
| `prefill-p64n0` | 226.852 | 1517.097 | 0.150 | 0 |
| `prefill-p512n0` | 811.506 | 4770.209 | 0.170 | 0 |

The top remaining p64/p512 blocker in that smoke is the Loom
`mul_mat_q4_k_q8_1_x4_mmq64x32` route, not missing bridge compilation.
