# HRX Backend

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
