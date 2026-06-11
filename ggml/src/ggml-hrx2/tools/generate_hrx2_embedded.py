#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate embedded HRX2 catalog sources.")
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--out-cpp", required=True, type=Path)
    parser.add_argument("--out-h", required=True, type=Path)
    return parser.parse_args()


def c_identifier(text):
    ident = re.sub(r"[^0-9A-Za-z_]", "_", text)
    if not ident or ident[0].isdigit():
        ident = "_" + ident
    return ident


def write_string_array(output, name, text):
    output.write(f"static const char {name}[] =\n")
    if text:
        for line in text.splitlines(keepends=True):
            output.write(f"    {json.dumps(line)}\n")
    else:
        output.write('    ""\n')
    output.write("    ;\n\n")


def main():
    args = parse_args()
    catalog_path = args.catalog.resolve()
    source_root = args.source_root.resolve()

    with catalog_path.open("r", encoding="utf-8") as f:
        catalog = json.load(f)

    if catalog.get("schema_version") != 1:
        raise SystemExit("unsupported HRX2 catalog schema_version")
    if not isinstance(catalog.get("sources"), dict):
        raise SystemExit("HRX2 catalog must contain a sources object")
    if not isinstance(catalog.get("kernels"), list):
        raise SystemExit("HRX2 catalog must contain a kernels array")

    source_texts = {}
    for source_id, source in catalog["sources"].items():
        if not isinstance(source, dict) or "path" not in source:
            raise SystemExit(f"source {source_id} must contain path")
        source_path = (source_root / source["path"]).resolve()
        try:
            source_path.relative_to(source_root)
        except ValueError as exc:
            raise SystemExit(f"source {source_id} escapes source root: {source_path}") from exc
        source_texts[source_id] = source_path.read_text(encoding="utf-8")

    known_sources = set(catalog["sources"].keys())
    for kernel in catalog["kernels"]:
        if not isinstance(kernel, dict):
            raise SystemExit("kernel entries must be objects")
        kernel_id = kernel.get("id")
        source_id = kernel.get("source_id")
        if not kernel_id:
            raise SystemExit("kernel entry missing id")
        if source_id not in known_sources:
            raise SystemExit(f"kernel {kernel_id} references unknown source {source_id}")

    args.out_cpp.parent.mkdir(parents=True, exist_ok=True)
    args.out_h.parent.mkdir(parents=True, exist_ok=True)

    catalog_json = json.dumps(catalog, indent=2, sort_keys=False) + "\n"

    with args.out_h.open("w", encoding="utf-8", newline="\n") as f:
        f.write("#pragma once\n\n")
        f.write("#include <cstddef>\n\n")
        f.write("struct ggml_hrx2_embedded_source {\n")
        f.write("    const char * id;\n")
        f.write("    const char * path;\n")
        f.write("    const char * text;\n")
        f.write("    size_t text_size;\n")
        f.write("};\n\n")
        f.write("const char * ggml_hrx2_embedded_catalog_json();\n")
        f.write("size_t ggml_hrx2_embedded_source_count();\n")
        f.write("const ggml_hrx2_embedded_source * ggml_hrx2_embedded_sources();\n")

    with args.out_cpp.open("w", encoding="utf-8", newline="\n") as f:
        f.write('#include "hrx2_embedded_catalog.h"\n\n')
        f.write("namespace {\n\n")
        write_string_array(f, "k_catalog_json", catalog_json)
        source_symbols = {}
        for source_id, text in source_texts.items():
            symbol = "k_source_" + c_identifier(source_id)
            source_symbols[source_id] = symbol
            write_string_array(f, symbol, text)

        f.write("const ggml_hrx2_embedded_source k_sources[] = {\n")
        for source_id, source in catalog["sources"].items():
            symbol = source_symbols[source_id]
            f.write(
                "    { "
                f"{json.dumps(source_id)}, "
                f"{json.dumps(source['path'])}, "
                f"{symbol}, "
                f"sizeof({symbol}) - 1"
                " },\n"
            )
        f.write("};\n\n")
        f.write("} // namespace\n\n")
        f.write("const char * ggml_hrx2_embedded_catalog_json() {\n")
        f.write("    return k_catalog_json;\n")
        f.write("}\n\n")
        f.write("size_t ggml_hrx2_embedded_source_count() {\n")
        f.write("    return sizeof(k_sources) / sizeof(k_sources[0]);\n")
        f.write("}\n\n")
        f.write("const ggml_hrx2_embedded_source * ggml_hrx2_embedded_sources() {\n")
        f.write("    return k_sources;\n")
        f.write("}\n")


if __name__ == "__main__":
    main()
