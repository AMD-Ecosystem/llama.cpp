# HRX2 backend catalog

HRX2 kernel authoring uses checked-in Loom source and JSON route metadata as the
source of truth. The build generates embedded C++ data from these files so
production binaries are hermetic, while development can point the backend at an
exploded catalog with `GGML_HRX2_CATALOG_DIR`.

- `catalog.json` describes route IDs, source IDs, exported symbols, ABI
  expectations, and dispatch defaults.
- `kernels/*.loom` contains standalone Loom kernels with checks and benchmarks.
- `tools/generate_hrx2_embedded.py` validates the catalog and emits embedded
  data for the backend build.

During development, copy or edit this directory shape on disk and run with:

```sh
GGML_HRX2_CATALOG_DIR=/path/to/ggml/src/ggml-hrx2 ...
```

The disk catalog is intentionally isomorphic to the embedded catalog so the same
metadata can drive Loom validation, offline tuning, and llama.cpp execution.
