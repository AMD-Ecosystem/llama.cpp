# gfx1100 measurements after the rebase

Baseline 4a5e53048 versus candidate code a27a4bef8; runner 6837ad78c.
Both built RelWithDebInfo using HRX8d78611874afe8b7aaaf0227e28e4c3bd632bc99
and ROCm10.1.0a20260822-1100. RX7900XTX, GPU-fc2143878bea4591,
automatic power policy. Benchmark-package commits do not alter kernel code.

All 19 selected nonzero/route tests passed, including native/row64/tied
endpoints, vocabulary embeddings and Q6 residual/short-prefill coverage.
All 24 model benchmark invocations completed. No new perplexity sweep was run.

## Model throughput

ABBA, five samples/run, ten samples/build/scenario; batch/ubatch512,
four CPU threads, flash attention, GPU offload, default warmup. Positive
percentages mean faster. All samples are retained, including warmup drift.

| Model | pp66 | pp128 | pp512 | tg32 |
|---|---:|---:|---:|---:|
| promptbridge06 | -1.38% | +21.07% | -0.72% | +2.23% |
| qwen3_17 | +0.45% | -0.83% | +1.11% | -0.78% |
| qwen3_4 | +5.00% | +13.40% | -0.61% | +0.57% |
| qwen35_08 | +2.99% | +5.17% | +0.98% | -0.55% |
| qwen35_2 | -0.21% | +0.14% | +0.46% | -1.17% |
| qwen35_9 | +0.19% | +0.18% | +0.20% | +0.34% |

These are screening measurements, not a zero-regression claim. PromptBridge
has substantial run drift: baseline pp128 run medians 18415.7 and15349.5tok/s,
candidate19808.5 and19741.2. Its decode run medians were baseline596.84/612.498
and candidate647.286/612.436. Do not treat the pooled gains as precise estimates.
Qwen3.5 2B decode measured -1.17%, Qwen3 1.7B -0.78%, and PromptBridge pp66
-1.38%; these losses remain explicit. This campaign did not tune them.

## Generic Q6 F32 vocabulary kernel

K1024,N151936,T1; four binding sets (515383296bytes), batch8,
five warmups, minimum30 iterations/100ms, two runs in forward/reverse order.
All eight runs stopped as stable. Device time is a separate eight-dispatch
profile replay; host time is normalized queue completion. Synthetic zero
buffers have no numerical oracle; use the separate test results above.

| Variant | Device us, median run mean | Host us, median run p50 |
|---|---:|---:|
| Baseline native |143.530|149.623|
| Baseline row64 |153.845|162.844|
| Candidate row64 |144.855|152.952|
| Candidate native |143.590|151.829|

Candidate row64 is 5.84% faster than baseline row64 on device, but 0.92%
slower than baseline native. The host comparison shows additional run drift.
This generic kernel is distinct from the Q6xQ8 projection generalized to gfx11.

Raw logs, commands, JSONL, model/binary hashes and summaries are retained in
the originating workspace under investigation/qwen3-row64-161/rebased.
The tracked runner and fixtures reproduce the procedure on another machine.
The first setup attempt could not resolve the renamed Loom executable and
performed no measurement; the fixed runner completed the campaign.

## Q6 x Q8 projection

Candidate source, K2048,N151936, native wave32 versus generic gfx11 row64
wave64, two runs each in forward/reverse order. Token capacity matches
T1/T17. Explicit dispatch_complete, batch8, warmup5, iterations30,
binding ring4; all eight runs reached stable. Medians of device replay
means and host p50s:

| Tokens | Native device us | Row64 device us | Native host us | Row64 host us |
|---|---:|---:|---:|---:|
| 1 |289.445|285.790|298.949|300.453|
| 17 |4900.678|5202.768|4886.586|5218.783|

Decode device time improves1.26%, while host time increases0.50%.
Prefill17 device time increases6.16%, host6.80%. The two prefill device
means were4899.010/4902.345us native and5198.285/5207.250us row64.
This is a repeatable kernel-level gap in this measurement, not proof of
which source change caused it. Preserve it for gfx1151 comparison and
further kernel-first investigation; no tuning was performed here.
Earlier setup probes used case_end_to_end timing (and one used excess
decode token capacity); their logs are retained but excluded from this table.

Fresh PromptBridge command captures each contain367 commands. Command366
selects the generic vocabulary kernel at K1024,N151936,T1, changing weight
format6(native) to46(row64), confirming the generic microbenchmark's route.
