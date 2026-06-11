# HRX2 backend catalog

HRX2 kernel authoring uses checked-in Loom source and JSON route metadata as the
source of truth. The build generates embedded C++ data from these files so
production binaries are hermetic, while development can point the backend at an
exploded catalog with `GGML_HRX2_CATALOG_DIR`.

- `catalog.json` describes targets, route IDs, priorities, source IDs, bytecode
  artifact IDs, exported symbols, ABI expectations, shape domains, optional
  shape guards, optional JIT config bindings, and dispatch defaults.
- `kernels/*.loom` contains standalone Loom kernels with checks and benchmarks.
- `tools/link_hrx2_artifacts.py` uses `loom-link --mode=selective` to produce
  per-route Loom bytecode artifacts in the build tree.
- `tools/validate_hrx2_catalog.py` validates source metadata before linking and
  artifact availability after linking.
- `tools/generate_hrx2_embedded.py` emits embedded catalog, source, and bytecode
  data for the backend build.

During development, copy or edit this directory shape on disk and run with:

```sh
GGML_HRX2_CATALOG_DIR=/path/to/catalog ...
```

The disk catalog is intentionally isomorphic to the embedded catalog so the same
metadata can drive Loom validation, offline tuning, and llama.cpp execution.

The runtime prefers `loom-bytecode` artifacts when present and falls back to
Loom text otherwise. For an exploded dev catalog, copy this directory shape plus
the generated `artifacts/*.loombc` files from the build tree.

Phase0.2 adds exact-shape Q8_0/F32 `MUL_MAT` JIT routes. These routes use
`specialization.mode = "jit_config"` and bind Loom `config.decl` values from
the concrete ggml shape at provider creation time. The resolved config values
are part of the provider cache key, so a static-shape kernel is compiled once
per route/target/shape/config tuple and reused inside the process. Generic
routes remain fallback candidates.

Focused smoke:

```sh
cmake --build build/llama-hrx2 --target ggml-hrx2 test-backend-ops -j"$(nproc)"

LD_LIBRARY_PATH="$PWD/build/hrx-install/lib:$PWD/rocm/lib:${LD_LIBRARY_PATH:-}" \
  build/llama-hrx2/bin/test-backend-ops test -b HRX20 \
  -o MUL_MAT -p 'type_a=q8_0,type_b=f32,m=1,n=64,k=256' --output csv
```

`loom-link --strip-check` is intentionally not used yet. As of the phase0.1
bringup it can strip check dependencies that the linker still treats as
required symbols; linked bytecode without that flag compiles correctly through
the HRX Loom JIT.
