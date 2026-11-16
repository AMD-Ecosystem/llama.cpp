# Issue 161 performance reproduction

Compare PR196 (branch `users/astgeorg/qwen3-row64-161`) against upstream
`4a5e53048` after rebasing. The candidate includes the generic gfx11 wave64
Q6 x Q8 row64 projection. This experiment does not include F32 accumulation
or activation-LDS reuse from the separate accuracy investigation.

Build both sources as RelWithDebInfo with HRX
`8d78611874afe8b7aaaf0227e28e4c3bd632bc99` and the same ROCm SDK/compiler.
The reference SDK is `10.1.0a20260822-1100`. Use each machine's supported
SDK if needed and report the difference. Enable `GGML_HRX`, set
`HRX_SOURCE_DIR` and `IREE_ROCM_PATH`; build `llama-bench` and `hrx-ops-test`.
The CMake dependency build also produces loom-link and loom-benchmark.
Pin `ROCR_VISIBLE_DEVICES` to the actual gfx1100 or gfx1151 UUID; set the
SDK library directory in `LD_LIBRARY_PATH`. Record GPU, power mode, clocks,
RAM, compiler, exact source commits, model hashes and binary hashes.

Run from the candidate checkout, supplying absolute paths:

```sh
python3 ggml/src/ggml-hrx/tools/benchmarks/row64-161/run.py \
  --baseline-source "$BASELINE_SOURCE" --baseline-build "$BASELINE_BUILD" \
  --candidate-source "$CANDIDATE_SOURCE" --candidate-build "$CANDIDATE_BUILD" \
  --models "$MODELS" --output "$RESULTS"
```

The runner starts with kernel measurements, then runs the same six models
and pp66/pp128/pp512/tg32 scenarios as the original sweep. Model names are
listed in run.py; use the existing shared model store. ABBA ordering,
five samples per run, ten samples per build/scenario, batch/ubatch 512,
four CPU threads, flash attention, GPU offload and default warmup match
the original measurement. Summarize pooled median throughput changes and
individual run medians; retain all samples and failures. Small differences
are screening results, not statistical significance claims.

Kernel fixtures capture the original generic vocabulary projection at
K=1024, N=151936, one token. Measure native/row64 on both baseline and
candidate in forward/reverse order. This adds a native candidate control;
it excludes discarded prototype variants. Four binding sets total
515,383,296 bytes, batch eight, five warmups, minimum 30 iterations.
Report device replay timing separately from normalized host completion,
alongside VGPR/LDS and instruction counts. Fixtures contain zero data and
have no numerical oracle. Run relevant nonzero hrx-ops-test coverage
separately before interpreting timings as acceptance evidence.

Also capture model command programs with `GGML_HRX_DUMP_COMMAND_PROGRAM_DIR`
to verify selection. The generic F32 vocabulary benchmark above does not
exercise the newly generalized Q6 x Q8 wave64 projection. Compile/run its
`ggml_linear_q6k_q8_1_x4_row64_wave64_decode` and
`ggml_linear_q6k_q8_1_x4_row64_wave64_prefill_17` Loom benchmarks from
qwen_moe/ggml/linear_q6k_q8_1_x4.loom (K=2048,N=151936), comparing native
projection controls. Link the manifest's library_sources, set token_capacity
1/17 respectively and output_capacity 151936. Explicitly pass
`--measure=dispatch_complete`; the source benchmark default includes case
setup. Use batch8, warmup5, iterations30, input-ring-count4 and
profile-final-batch=true for separate device timing. Verify wave64 selection on
gfx1151 and run endpoint numerical tests. Any performance gap starts with
matched kernel-level investigation, not model-level tuning.

Original gfx1100 model deltas are in PR196's description: decode ranged
-0.58% to +0.21%; PromptBridge pp66 retained -2.06%. Generic vocabulary
kernel device times were native 143.498 us, row64 before reuse 164.550 us,
retained reuse 145.218 us. Those used HRX82214df and baseline010b2ee;
they are historical comparisons, not results for this rebase or gfx1151.

Use an idle GPU and a shared advisory lock (runner defaults to
/tmp/lemonade-ws-tek-gpu.lock). Report instability and power/clock drift;
do not discard slow samples. Publish findings to lemonade-ws-aj0j and
lemonade-ws-s9tq, with exact commands and source identities.
