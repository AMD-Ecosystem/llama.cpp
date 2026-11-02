# HRX Backend

## Weight residency diagnostics

Weight residency supports standard GGUF inference with immutable model weights. Alternative layouts require an HRX `buffer_from_host_ptr` import that does not alias a known HRX host allocation. Other weights remain native-only: host weights reuse a cached native copy, and directly bindable device weights reuse their existing allocation. Source availability for native caching does not grant permission to select a transformed layout.

Set `GGML_HRX_LOG_WEIGHT_STATS=1` and run `llama-bench -v` to emit one `ggml_hrx: weight_stats` line when each HRX backend is released. The `-v` flag enables the backend log callback in `llama-bench`. The line reports successful weight materializations, cache acquisition reuse, uploaded weight bytes, live allocation count and bytes, peak live bytes, and all host transfer uploads. `ggml_backend_hrx_get_cache_stats` exposes the same weight counters while a backend is running. Prepared program hits do not count as weight cache acquisitions, so unchanged replay can reuse a weight without increasing the `reused` counter.

Live weight bytes count physical allocations retained by the cache, a prepared program, or a recorded graph. Imported mappings and directly bound coherent host buffers are source storage, not additional weight allocations. Native and losslessly packed layouts can both be materialized from their retained bytes; numerical conversion layouts retain their existing conversion behavior.

Each immutable source region is materialized only when a command first requests its layout. Native residency copies the identified source byte range verbatim, so tensor type, shape, and strides do not create separate native copies. Transformed representations additionally distinguish layout, source type, transformation dimensions, and source/output byte lengths. Graph metadata still governs how those bytes are interpreted. The weight cache keeps every distinct representation until explicit cache clearing or backend teardown. Prefill/decode transitions, empty graphs, and other weights do not evict it. Prepared and recorded programs share allocation leases with the cache.

A graph instance remembers its validated prepared-cache entry. Its first execution resolves all physical bindings; repeated executions with the same graph UID and node array bind only mutable host staging. Structurally equivalent graphs can share command programs without sharing unchecked bindings: each new instance goes through the full binding lookup before using the fast path. Mutable source rebinding also returns to that lookup. Both execution paths use the same prepared-cache entry, lock, and replay implementation; warm executions do not rebuild per-weight demand.

Inserting a new physical representation logs a warning when another layout of the same source region and generation is already cached. The warning groups exact source byte ranges; it does not detect partial overlap or infer a whole tensor from a transformed subrange. It names the layouts, the new allocation's byte cost, and their total resident bytes. Aliases reuse the same representation, and warm executions do not repeat the warning. Keeping both layouts can increase steady-state device memory, while avoiding repeated conversion and upload.

Keep borrowed weight bytes unchanged and alive while HRX can use them. A foreign CPU host buffer and its storage must stay alive for the lifetime of the HRX backend that caches it. Its identity depends on its wrapper and base addresses; destroying and recreating a buffer at those addresses while that backend lives can reuse old residency. Training and in-place weight updates are outside this path. Ordinary mutable inputs and outputs retain their host staging, writeback and synchronization behavior.

## Compilation diagnostics

Set `GGML_HRX_TRACE_COMPILE=1` before starting an application using the ggml HRX backend to log compilation activity. Each record is a JSON object on stderr prefixed with `hrx compile: `. This output bypasses the application's log-level filter, so it also works with quiet benchmark runs. The setting is read once per process; unset or any value other than `1` disables it.

For example, use an existing benchmark to inspect two prefill/decode cycles in one context:

```sh
GGML_HRX_TRACE_COMPILE=1 ./build/bin/llama-bench \
  --model /path/to/model.gguf --device HRX0 --n-gpu-layers 99 \
  -p 0 -n 0 -pg 64,16 -b 64 -ub 64 -r 2 --no-warmup --progress \
  2>hrx-compile.log
```

Select a supported GPU with `ROCR_VISIBLE_DEVICES` as appropriate for the machine. The workload and repetition count are caller choices. Omitting `--no-warmup` captures the benchmark's warmup too.

Every record carries `pid`, a monotonic `time_ns`, and `cache_id`, `execution_id`, and `compilation_id`. IDs are process-local; zero means the corresponding scope does not apply. Use `(pid, cache_id)` to identify a kernel-cache lifetime and `compilation_id` to join a request to its worker events. Clearing the cache emits `cache_clear` and assigns a new cache ID for future work.

| Event | Information |
| --- | --- |
| `graph_begin` | A backend graph execution and its incoming `graph_uid`. Each invocation gets a new execution ID, even when the graph is reused. |
| `graph_end` | Compilation requests made by that execution, including zero, host duration, and success. |
| `compile_request` | A kernel-cache miss: full cache key, kernel name, exact/dynamic workload specialization, integer parameters, and merged compile parameters. |
| `compile_begin` | The worker or synchronous caller starts compiling that request. |
| `compile_end` | Compiler duration, success, and error text. `compiler_started=false` identifies failure before the compiler ran, such as JIT initialization failure. |
| `compile_wait` | A caller encountered an unresolved compilation: wait duration, success, and `waiting_execution_id`. `execution_id` still identifies the original requester. Already resolved compilations emit no wait event. |
| `cache_clear` | The end of a cache lifetime, after outstanding compiler workers finish. |

Graph executions are backend calls, not authoritative prefill/decode, token, or repetition labels. One application step may execute multiple backend graphs. Correlate the ordered executions and specialization parameters with the workload and the caller's progress output. Direct kernel-cache callers without a graph scope have `execution_id=0`.

To diagnose repeated compilation, group requests by kernel within a cache lifetime and compare their keys and parameters. Changing sequence-length parameters can reveal candidates for dynamic specialization. Repeating an identical key may indicate a retry after compilation or executable-loading failure. The records report observations; they do not decide whether a specialization is necessary. A graph-program change alone does not imply a kernel compilation.

Compiler durations can overlap across workers. Wait durations measure the host's unresolved-compilation wait, including scheduling and diagnostic overhead; they exclude executable loading and GPU execution. Synchronous compilation blocks the caller within the compile event and normally emits no `compile_wait`. Graph durations cover the host backend call, which may return before GPU work finishes. Summed compiler durations therefore do not measure application stall time.

Capture preserves asynchronous compilation and graph replay. When disabled, instrumentation uses cached enable state and cheap checks, with no added clock reads, event formatting, logging locks, or file I/O. When enabled, output and timing add diagnostic overhead; use ordinary runs for throughput measurements.

Extract just the JSON records for analysis:

```sh
sed -n 's/^hrx compile: //p' hrx-compile.log > hrx-compile.jsonl
jq -c 'select(.event == "compile_request") | {cache_id, execution_id, kernel, integer_parameters, compile_parameters}' hrx-compile.jsonl
```

## `hrx-system` backend compatibility

`hrx-minimum-commit.txt` records the minimum required commit in ROCm/hrx-system. It contains one full Git commit SHA. The backend's C++ APIs, Loom syntax, and numerical behavior can all depend on that revision. When `GGML_HRX=ON` and `HRX_SOURCE_DIR` is set, CMake requires the HRX checkout's HEAD to equal or descend from the recorded commit.

For builds using installed packages, check the source revision used to produce both HRX and Loom against this floor before validating a repair. From the staging checkout:

```sh
python3 llama.cpp/ggml/src/ggml-hrx/tools/check_hrx_revision.py hrx-system
```

### Updating the minimum

Raise the minimum in the same llama.cpp commit that introduces a newer HRX dependency. This includes new API calls, new Loom syntax, and removing a numerical workaround after an HRX compiler or runtime fix. Unrelated repairs do not require a bump.

Choose the earliest demonstrated required commit that retains the previous floor in its ancestry. Do not replace the floor with the latest tested revision merely because it is available. Record the dependency and its HRX commit or PR in the change's validation evidence. A minimum is not a promise that every future HRX revision is compatible.

## Backend test suites

The HRX suite targets, executables, and CTest names use `hrx-*-test`: `hrx-buffer-test`, `hrx-backend-test`, `hrx-loom-jit-test`, and `hrx-ops-test`. Build with `GGML_HRX=ON` and `LLAMA_BUILD_TESTS=ON`, then discover only these suites with:

```sh
ctest --test-dir build -L '^hrx$' --show-only=json-v1
```

Before building, the returned test names identify the CMake targets to build. After building, the returned commands include the executable paths and all suite arguments, so package automation can use this metadata without maintaining a separate argument map. CTest owns full case selection, worker counts (buffer 1, backend 4, Loom JIT 2, ops 4), full child output, a 900-second timeout per suite, and failure on missing-device skips. CTest controls suite concurrency; cases within each suite use the configured workers.

Select a supported GPU with `ROCR_VISIBLE_DEVICES` and run:

```sh
ctest --test-dir build -L '^hrx$' --no-tests=error --parallel 2 --verbose
```

Use verbose mode to retain output from passing cases as well as failures. Consumers executing the discovered commands outside CTest must also honor the returned test properties, including timeout and failure regex.
