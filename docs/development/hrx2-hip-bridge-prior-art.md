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
- Vulkan Q8_0 integer-MMQ prompt path: `matmul_q8_0_q8_1` is the strongest
  current prior for closing the Q8 prompt gap. It is not a RHS-only staging
  schedule. The shader stages both A and B into shared memory (`buf_a` and
  `buf_b`), uses `block_a_cache` with eight packed Q8_0 payload words plus one
  scale and `block_b_cache` with eight Q8_1 payload words plus scales, then
  moves those shared tiles into register caches before dotting. The key control
  parameters are `BM`, `BN`, fixed `BK=32`, `BK_STEP=4` for normal prompt
  matmul, `BLOCK_SIZE`, `WM`, `WN`, `WMITER`, `TM`, `TN`, and `WARP`. The
  discrete Vulkan families include medium `BM64/BN64` and larger
  `BM128/BN128` tiles, with integer-dot variants using two-row/two-column or
  four-row/four-column ownership depending on device capability. Future Loom
  Q8/Q5/Q6 prompt work should port this A+B-staged register-cache dataflow
  before trying more one-row-per-lane RHS-only pivots.

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
- Q6_K `x4_vkm64x32_wg128_w32` Loom probe was rejected as another p64
  production route. It attempted to move toward the removed HIP wave32 bridge
  by using workgroup 128, A+B LDS staging, and `BK_STEP=4`, but it still used a
  64x32 workgroup tile and a four-row/four-column-per-lane ownership shape
  instead of the HIP prior's real 64x64 tile, two-row/two-column ownership, and
  eight column sub-iterations. It passed the focused CPU-reference gate and
  route stderr confirmed `hrx2_mul_mat_q6_k_q8_1_x4_vkm64x32_wg128_w32_static`
  was JIT compiled and selected, but same-runner `test-backend-ops perf`
  regressed `Vcur k3072 r1024 c64` from the clean current 178.79 us to
  337.28 us and regressed `ffn_out k8192 r3072 c64` from 540.78 us to
  903.33 us. The rejected patch and trace evidence are preserved at
  `cache/hrx2/phase2b/q6-vkm64x32-wg128-probe-20260616-172640/`.
  Do not retry this half-width WG128 spelling; the next Q6 attempt must match
  the HIP bridge's actual `BM64/BN64/BK_STEP=4/WG128/wave32` ownership or use a
  newly documented prior.
- Q6_K `x4_vkm64x64_tm2tn2` Loom probes matched the removed HIP bridge's
  visible p64 schedule more closely but were still rejected. The first spelling
  used `BM64/BN64/BK_STEP=4/WG128/wave32`, four wave32 tiles, two rows by two
  columns per lane, eight `WNITER` column sub-iterations, and A+B LDS staging,
  but scaled each dot inside the `iqs` loop. It passed the focused CPU-reference
  gate and selected correctly, but regressed `Vcur k3072 r1024 c64` to
  556.81 us and `ffn_out k8192 r3072 c64` to 1381.83 us. Its compile report
  showed zero spills, 256 static dot ops, 260 conversions, 175 peak live units,
  and 15388 code bytes. The corrected qsum spelling then accumulated integer
  `qsum0/qsum1` halves and applied scales once per half, matching the HIP
  arithmetic structure more closely. It improved the static report to zero
  spills, 256 static dot ops, 68 conversions, 140 peak live units, and
  10488 code bytes, but still regressed the same rows to 386.59 us and
  944.03 us versus the clean current `mmq64x32` baseline of 178.79 us and
  540.78 us. Rejected artifacts:
  `cache/hrx2/phase2b/q6-vkm64x64-tm2tn2-c64-opgate-20260616-190345/` and
  `cache/hrx2/phase2b/q6-vkm64x64-tm2tn2-qsum-c64-opgate-20260616-190717/`.
  Do not retry this high-level A+B-staged TM2/TN2 Loom spelling without a new
  low-level/codegen hypothesis. The next Q6 attempt should compare against the
  HIP target listing and focus on why the compact current RHS-staged
  `mmq64x32` route wins despite less column reuse.
- Q4_K `x4_mmq64x8_wg64` Loom probe was rejected as a bounded ownership test.
  It attempted to import the removed HIP bridge's row-reuse idea by using a
  WG64 schedule where each lane owns four rows across an eight-column tile, but
  it intentionally kept `BN=8` instead of implementing the full prior's
  `BN=32/64` column reuse. It passed the focused CPU-reference gate and route
  traces confirmed `hrx2_mul_mat_q4_k_q8_1_x4_mmq64x8_static` was selected, but
  same-runner `test-backend-ops perf` regressed Kcur `k3072 r1024 c64` from the
  clean current 183.18 us to 282.31 us, Qcur `k3072 r3072 c64` from 178.89 us
  to 299.32 us, ffn_out `k8192 r3072 c64` from 469.98 us to 735.31 us, and
  ffn_gate `k3072 r8192 c64` from 415.33 us to 501.63 us. The JIT HSACO was
  also much larger than the current route. The rejected patch and traces are
  preserved at
  `cache/hrx2/phase2b/q4-mmq64x8-wg64-probe-20260616-173427/`. Do not retry
  this narrow-tile spelling. The remaining Q4 prior is still the full Vulkan
  medium / HIP bridge MMQ family with packed Q8_1 x4 RHS, A-side pack/cache,
  `u8s8` dot contract, and production prompt shapes around `BM64` with
  `BN32/BN64` column reuse.
- Q4_K `x4_vkm64x64_tm4tn1` Loom port from the successful HIP wave32/TM4/TN1
  prior was only accepted for the narrow Llama 3.2 3B ffn_gate prompt row
  `k=3072, rows=8192, cols=64`. As a broad route it passed CPU-reference tests
  but regressed Kcur c64 from 183.18 us to 302.87 us, Qcur c64 from 178.89 us
  to 265.70 us, and ffn_out c64 from 469.98 us to 689.68 us. Narrowed to
  `k3072/r8192/c64`, it leaves those rows on the existing `mmq64x32` route and
  improves ffn_gate c64 from the clean current 415.33 us to roughly
  310-320 us in same-runner `test-backend-ops perf`. The same route was also
  rejected for the p512 ffn_gate row after measuring 1558.95 us, so production
  metadata caps it at `cols=64`. Compile report for the accepted shape showed
  status OK, zero hazard gaps, zero spills, 2048 static dot ops, and peak live
  units 342, which is a warning that the high-level spelling is register-heavy
  even when it wins a launch/column-reuse-sensitive shape. Artifact:
  `cache/hrx2/phase2b/q4-vkm64x64-tm4tn1-c64-final-20260616-174817/`.
- Q4_K `x4_vkm64x64_tm2tn2` Loom port is accepted for broad p512 prompt rows
  with `k=3072..8192`, `rows=1024..8192`, and `cols=512`. This is the Loom
  spelling closest to the removed HIP/Vulkan-medium `BM64/BN64/WG128/wave32`
  prior: four wave32-style tiles per workgroup, two rows by two columns per
  lane, eight column sub-iterations, packed Q8_1 x4 RHS, Q4 A-side staging,
  and `vector.dot4i<u8s8>`. Focused CPU-reference testing passed and route
  traces confirmed selection with no provider-unavailable events. Same-runner
  `test-backend-ops perf` on the Llama 3.2 3B p512-derived rows improved Kcur
  c512 from 422.33 us to 326.49 us, Qcur c512 from 1071.12 us to 737.68 us,
  ffn_out c512 from 2977.02 us to 1939.53 us, and ffn_gate c512 from
  2917.77 us to 1868.03 us. The compile report for
  `k=3072, rows=3072, cols=512` reported zero spills, zero hazard gaps, 2048
  static dot ops, 333 peak live units, and 68008 code bytes. Integration A/B on
  Llama 3.2 3B Q4_K_M p512 improved HRX2 steady prefill to about 1444.7 tok/s
  from the prior roughly 770-810 tok/s range, with zero CPU compute fallback.
  Vulkan remained about 5972.3 tok/s steady on the same run, so this is an
  accepted boulder improvement but not the final parity schedule. Artifact:
  `cache/hrx2/phase2b/q4-vkm64x64-tm2tn2-p512-probe-20260616-175737/`;
  integration artifact:
  `cache/hrx2/phase2a/phase2b-q4-tm2tn2-p512-20260616-180125/`.
  A direct c64 route probe using the same TM2/TN2 source passed CPU-reference
  testing and selected correctly, but was rejected for the p64 regime: Kcur c64
  measured 315.06 us, Qcur c64 280.66 us, ffn_out c64 731.68 us, and ffn_gate
  c64 341.84 us, all worse than the current production c64 routes. Preserve
  TM2/TN2 as a p512 schedule only. Rejected patch and traces:
  `cache/hrx2/phase2b/q4-vkm64x64-tm2tn2-c64-true-probe-20260616-180713/`.
  ISA comparison against the extracted HIP wave32 prior confirms the high-level
  schedule is not the main mismatch. The HIP
  `vkm64x64_pack2_wg128_w32_u32` function compiles to 256 static
  `v_dot4_i32_iu8` instructions, about 54 global-memory instructions, 55 LDS
  instructions, 2 barriers, 95 VGPRs, and no scratch. The accepted Loom p512
  route compiles to 2048 static dot instructions, 180 global-memory
  instructions, 1472 LDS instructions, 16 barriers, 333 peak live units, and
  68008 code bytes. A probe that removed the Loom `%group` loop unroll was the
  obvious attempt to move the spelling toward the HIP loop body, but it failed
  in `source-to-low` with `AMDGPU branch argument materializer selected for an
  unsupported type`. This is the current concrete Q4 p512 root-cause lead:
  close the gap by expressing the HIP-like loop without fully unrolling the
  group/K dimension, or get the Loom lowering fixed for that loop-carried state.
  Artifacts:
  `cache/hrx2/phase2b/q4-tm2tn2-isa-compare-20260616/` and
  `cache/hrx2/phase2b/q4-vkm64x64-tm2tn2-no-group-unroll-*`.
- Q8_0 RHS-only 128-row Loom probes were rejected as production routes. The
  scalar `BM128/BN32/WG256/16-cols-per-lane` HRX1-shaped spelling compiled
  cleanly but failed focused backend-op correctness with NaN at output index 0
  on all Llama 3.1 8B Q8_0 p64 prompt rows; disassembly showed the intended
  `v_dot4_i32_iu8` form and no `ds_*_addtid` addressing. A narrower diagnostic
  `BM128/BN16/WG256/8-cols-per-lane` spelling passed the same CPU-reference
  gate but regressed most focused rows against the accepted `BM64/BN32`
  RHS-only route, including result_output `k4096 r128256 c64` at 6590.06 us
  versus 5262.39 us. Artifacts:
  `cache/hrx2/phase2b/q8-mmq128x32-scalar-debug-20260616-182150/` and
  `cache/hrx2/phase2b/q8-mmq128x16-scalar-diag-20260616-182522/`. This is the
  concrete reason not to keep sweeping row ownership alone for Q8: the next
  candidate should match the Vulkan A+B-staged integer-MMQ family.
- Q8_0 `x4_vkm64x64_tm2tn2` high-level Loom port was rejected as the first
  direct A+B-staged c64 probe. It matched the successful Q4 TM2/TN2 route
  shape and the Vulkan medium prior at the visible schedule level: `BM64`,
  `BN64`, workgroup 128, four wave32-style subtiles, two rows by two columns
  per lane, eight column sub-iterations, both Q8_0 A and packed Q8_1 B staged
  in LDS, and `vector.dot4i<s8s8>`. It passed focused CPU-reference testing on
  all five Llama 3.1 8B Q8_0 p64 prompt rows and traces confirmed
  `hrx2_mul_mat_q8_0_q8_1_x4_vkm64x64_tm2tn2_static` was selected. Same-runner
  `test-backend-ops perf` rejected it against the current `mmq64x32` route:
  Vcur c64 346.05 us versus 101.68 us, Qcur c64 392.04 us versus 179.74 us,
  ffn_out c64 1364.98 us versus 641.67 us, ffn_gate c64 640.37 us versus
  525.17 us, and result_output c64 5512.27 us versus 5264.35 us. The candidate
  HSACO was also much larger, 17440 bytes versus 9232 bytes. The rejected patch
  and traces are preserved at
  `cache/hrx2/phase2b/q8-vkm64x64-tm2tn2-c64-probe-20260616-183621/`. Do not
  retry this generic high-level TM2/TN2 spelling as a Q8 c64 production route.
  The next Q8 A+B-staged attempt must match a proven Vulkan/HIP loop structure
  more exactly, including `BK_STEP`, register-cache lifetime, unroll depth, and
  lane ownership, or first prove the emitted instruction schedule against a
  known-good target listing.

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

After reverting or removing a route, rebuild `ggml-hrx2` before measuring. The
generated embedded catalog in the build tree can otherwise retain a rejected
provider and make a "current baseline" run select stale code. On 2026-06-16,
the stale artifact
`cache/hrx2/phase2b/q6-current-p64-baseline-20260616-172137/` still selected a
removed Q6 `mmql64x64` provider. The clean rebuilt baseline is
`cache/hrx2/phase2b/q6-current-cleanbuild-p64-baseline-20260616-172230/`.
Always inspect route stderr or `GGML_HRX2_TRACE_JSONL` before using perf CSVs.

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
