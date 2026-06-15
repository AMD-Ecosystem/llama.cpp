# HRX2 backend catalog

HRX2 kernel authoring uses checked-in Loom source and split JSON catalog
metadata as the source of truth. The build assembles those small files into the
single catalog JSON consumed by validation, artifact linking, embedding, and
runtime loading. Production binaries remain hermetic, while development can
point the backend at an exploded assembled catalog with `GGML_HRX2_CATALOG_DIR`.

- `catalog/` contains metadata, targets, sources, artifacts, families, route
  files by family, and fusion candidate files.
- `tools/assemble_hrx2_catalog.py` deterministically assembles `catalog/` into
  the monolithic `ggml-hrx2-catalog-v1` JSON shape used by existing tooling and
  the runtime.
- `kernels/*.loom` contains standalone Loom kernels with checks and benchmarks.
- `tools/link_hrx2_artifacts.py` uses `loom-link --mode=selective` to produce
  per-route Loom bytecode artifacts in the build tree.
- `tools/validate_hrx2_catalog.py` validates source metadata before linking and
  artifact availability after linking.
- `tools/generate_hrx2_embedded.py` emits embedded catalog, source, and bytecode
  data for the backend build.
- `loom-jit/` contains the llama.cpp-side Loom JIT shim used by this backend.
  HRX2 compiles this local shim directly against the Loom C API CMake targets
  exported by the HRX package.

During development, assemble a catalog directory containing `catalog.json` and
run with:

```sh
python3 ggml/src/ggml-hrx2/tools/assemble_hrx2_catalog.py \
  --catalog-dir ggml/src/ggml-hrx2/catalog \
  --out /path/to/catalog/catalog.json
GGML_HRX2_CATALOG_DIR=/path/to/catalog ...
```

The assembled disk catalog is intentionally isomorphic to the embedded catalog
so the same metadata can drive Loom validation, offline tuning, and llama.cpp
execution. Source editing should happen in split files under `catalog/`, not in
a generated monolithic JSON file.

The runtime prefers `loom-bytecode` artifacts when present and falls back to
Loom text otherwise. For an exploded dev catalog, copy this directory shape plus
the generated `artifacts/*.loombc` files from the build tree.

Keep reusable Loom sources target-neutral unless an algorithm genuinely depends
on target-specific provider code. Use catalog `target_key` values for measured
routes: tuned winners are target-qualified, while portable fallback routes leave
`target_key` empty and compile for the current HRX device architecture. Keep
scalar/vector baselines target-neutral. If a kernel source is truly
target-specific, such as a chip-specific WMMA layout, keep it as a separate
source/artifact entry and let route metadata select that source only for
matching targets. A new target such as `gfx1151` should get its own tuned route
rows, and only needs its own Loom source when the algorithm really depends on
target-specific primitives.

Phase0.2 adds exact-shape Q8_0/F32 `MUL_MAT` JIT routes. These routes use
`specialization.mode = "jit_config"` and bind Loom `config.decl` values from
the concrete ggml shape at provider creation time. The resolved config values
are part of the provider cache key, so a static-shape kernel is compiled once
per route/target/shape/config tuple and reused inside the process. Generic
routes remain fallback candidates.

Phase0.3 adds the first scaled-catalog backplane tools around the pilot
families. The runtime can emit JSONL route/provider evidence and dump per-JIT
compile artifacts:

```sh
GGML_HRX2_TRACE_JSONL=/path/to/events.jsonl
GGML_HRX2_TRACE_ROUTES=1
GGML_HRX2_EVIDENCE_DIR=/path/to/evidence
```

`GGML_HRX2_TRACE_JSONL` records provider cache misses/hits, compile success or
failure, selected route IDs, provider cache keys, concrete shapes, and dispatch
geometry. `GGML_HRX2_EVIDENCE_DIR` writes per-provider `provider.json`,
`compile_report.json`, and `manifest.json`.

The workspace-level phase0.3 flow is:

```sh
python3 tools/hrx2_collect_shapes.py --fixtures-only \
  --out cache/hrx2/shapes/phase0.3.jsonl
python3 tools/hrx2_generate_candidates.py \
  --shapes cache/hrx2/shapes/phase0.3.jsonl \
  --out cache/hrx2/candidates/phase0.3.jsonl
python3 tools/hrx2_run_loom_sweep.py \
  --candidates cache/hrx2/candidates/phase0.3.jsonl \
  --run-id phase0.3-smoke
python3 tools/hrx2_reduce_tuning.py \
  --run cache/hrx2/runs/phase0.3-smoke \
  --out cache/hrx2/reduced/gfx1100/phase0.3.json
python3 tools/hrx2_emit_catalog.py \
  --reduced cache/hrx2/reduced/gfx1100/phase0.3.json \
  --out cache/hrx2/catalog/phase0.3
```

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
