# HRX Backend

## Weight residency diagnostics

Weight residency supports standard GGUF inference with immutable model weights. Alternative layouts require an HRX `buffer_from_host_ptr` import that does not alias a known HRX host allocation. Other weights remain native-only: host weights reuse a cached native copy, and directly bindable device weights reuse their existing allocation. Source availability for native caching does not grant permission to select a transformed layout.

Set `GGML_HRX_LOG_WEIGHT_STATS=1` and run `llama-bench -v` to emit one `ggml_hrx: weight_stats` line when each HRX backend is released. The `-v` flag enables the backend log callback in `llama-bench`. The line reports successful weight materializations, cache acquisition reuse, uploaded weight bytes, live allocation count and bytes, peak live bytes, and all host transfer uploads. `ggml_backend_hrx_get_cache_stats` exposes the same weight counters while a backend is running. Prepared program hits do not count as weight cache acquisitions, so unchanged replay can reuse a weight without increasing the `reused` counter.

Live weight bytes count physical allocations retained by the cache, a prepared program, or a recorded graph. Imported mappings and directly bound coherent host buffers are source storage, not additional weight allocations. Native and losslessly packed layouts can both be materialized from their retained bytes; numerical conversion layouts retain their existing conversion behavior.

Each successfully admitted HRX submission updates the retained layout set only for the weight regions it reads. A later fragment using other weights, or no weights, leaves those regions resident. When a region is used with a different set, completed obsolete layouts are released while unfinished submissions keep their bindings until completion. A coherent native read also replaces that region's layout set, even when it needs no device allocation. Repeating warm disjoint fragments reuses their allocations; explicit cache clearing and backend teardown require materialization again.

Keep borrowed weight bytes unchanged and alive while HRX can use them. A foreign CPU host buffer and its storage must stay alive for the lifetime of the HRX backend that caches it. Its identity depends on its wrapper and base addresses; destroying and recreating a buffer at those addresses while that backend lives can reuse old residency. Training and in-place weight updates are outside this path. Ordinary mutable inputs and outputs retain their host staging, writeback and synchronization behavior.

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
