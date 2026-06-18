#!/usr/bin/env python3
import argparse
import fnmatch
import json
import pathlib
import sys

import compare_amdgcn_isa as isa


OPCODE_FIELDS = (
    ("v_dot", ("v_dot",)),
    ("v_wmma", ("v_wmma",)),
    ("lds_read", ("ds_load", "ds_read")),
    ("lds_write", ("ds_store", "ds_write")),
    ("vmem_load", ("global_load", "buffer_load", "flat_load")),
    ("vmem_store", ("global_store", "buffer_store", "flat_store")),
    ("s_waitcnt", ("s_waitcnt",)),
    ("s_barrier", ("s_barrier",)),
)


def load_hsaco_summary(path, llvm_objdump, llvm_readelf, symbol=None):
    readelf_lines = isa.run_text([str(llvm_readelf), "--notes", str(path)]).splitlines()
    resources = select_kernel_metadata(readelf_lines, symbol)
    inferred_symbol = symbol or resources.get("symbol") or resources.get("name")

    objdump_lines = isa.run_text([str(llvm_objdump), "--no-show-raw-insn", "-d", str(path)]).splitlines()
    try:
        symbol_lines = isa.extract_objdump_symbol(objdump_lines, inferred_symbol)
    except SystemExit:
        symbol_lines = objdump_lines

    opcodes = isa.extract_instructions(symbol_lines)
    summary = isa.summarize(path.name, symbol_lines, opcodes, resources, {})
    normalized = isa.normalize_resources(summary)
    interesting = summary.get("interesting_opcodes", {})
    event_summary = summary.get("event_summary", {})
    hot_score = event_summary.get("hot_op_score", {})

    row = {
        "file": str(path),
        "name": path.name,
        "kernel": resources.get("name"),
        "symbol": resources.get("symbol"),
        "resources": normalized,
        "instruction_count": summary.get("instruction_count"),
        "classes": summary.get("classes", {}),
        "interesting_opcodes": interesting,
        "hot_opcode": event_summary.get("hot_opcode"),
        "hot_op_score": hot_score,
    }

    for field, prefixes in OPCODE_FIELDS:
        row[field] = sum(
            count for opcode, count in interesting.items()
            if opcode.startswith(prefixes)
        )

    return row


def parse_hsaco_metadata_all(readelf_lines):
    kernels = []
    current = None
    for raw in readelf_lines:
        line = raw.strip()
        if line == "- .args:":
            if current:
                kernels.append(current)
            current = {}
            continue
        if current is None:
            continue
        if line.startswith(".name:"):
            current["name"] = line.split(":", 1)[1].strip()
        elif line.startswith(".symbol:"):
            current["symbol"] = line.split(":", 1)[1].strip()
        elif line.startswith(".group_segment_fixed_size:"):
            current["group_segment_fixed_size"] = isa.parse_int_tail(line)
        elif line.startswith(".private_segment_fixed_size:"):
            current["private_segment_fixed_size"] = isa.parse_int_tail(line)
        elif line.startswith(".sgpr_count:"):
            current["sgpr_count"] = isa.parse_int_tail(line)
        elif line.startswith(".sgpr_spill_count:"):
            current["sgpr_spill_count"] = isa.parse_int_tail(line)
        elif line.startswith(".vgpr_count:"):
            current["vgpr_count"] = isa.parse_int_tail(line)
        elif line.startswith(".vgpr_spill_count:"):
            current["vgpr_spill_count"] = isa.parse_int_tail(line)
        elif line.startswith(".wavefront_size:"):
            current["wavefront_size"] = isa.parse_int_tail(line)
    if current:
        kernels.append(current)
    return kernels


def select_kernel_metadata(readelf_lines, symbol=None):
    kernels = parse_hsaco_metadata_all(readelf_lines)
    if not kernels:
        return {}
    if symbol:
        for kernel in kernels:
            if symbol in kernel.get("name", "") or symbol in kernel.get("symbol", ""):
                return kernel
        raise SystemExit(f"symbol substring not found in HSACO metadata: {symbol}")

    public = [
        kernel for kernel in kernels
        if "_unused" not in kernel.get("name", "") and "_unused" not in kernel.get("symbol", "")
    ]
    if len(public) == 1:
        return public[0]
    if public:
        return max(public, key=lambda item: (
            item.get("vgpr_count", 0),
            item.get("sgpr_count", 0),
            item.get("group_segment_fixed_size", 0),
        ))
    return max(kernels, key=lambda item: (
        item.get("vgpr_count", 0),
        item.get("sgpr_count", 0),
        item.get("group_segment_fixed_size", 0),
    ))


def select_hsacos(hsaco_dir, patterns):
    paths = sorted(path for path in hsaco_dir.glob("*.hsaco") if path.is_file())
    if not patterns:
        return paths
    selected = []
    for path in paths:
        if any(fnmatch.fnmatch(path.name, pattern) for pattern in patterns):
            selected.append(path)
    return selected


def render_value(value):
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return "`" + json.dumps(value, sort_keys=True) + "`"
    return f"`{value}`"


def write_markdown(path, rows):
    lines = [
        "# HSACO Family Summary",
        "",
        "| HSACO | Wave | VGPR | SGPR | LDS | Scratch | VGPR Spills | v_dot | v_wmma | LDS Read | LDS Write | VMEM Load | VMEM Store | Wait | Barrier | Hot Op | Pre-Hot Loads | Final Wait | Hot Window |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        res = row["resources"]
        hot = row.get("hot_op_score", {})
        pre_hot_loads = hot.get("pre_hot_lds_load", 0) + hot.get("pre_hot_vmem_load", 0)
        lines.append(
            "| "
            + " | ".join([
                f"`{row['name']}`",
                render_value(res.get("wavefront_size")),
                render_value(res.get("vgpr")),
                render_value(res.get("sgpr")),
                render_value(res.get("lds_bytes")),
                render_value(res.get("scratch_bytes")),
                render_value(res.get("vgpr_spills")),
                render_value(row.get("v_dot")),
                render_value(row.get("v_wmma")),
                render_value(row.get("lds_read")),
                render_value(row.get("lds_write")),
                render_value(row.get("vmem_load")),
                render_value(row.get("vmem_store")),
                render_value(row.get("s_waitcnt")),
                render_value(row.get("s_barrier")),
                f"`{row.get('hot_opcode') or ''}`",
                render_value(pre_hot_loads),
                render_value(hot.get("final_pre_hot_lgkmcnt")),
                render_value(hot.get("hot_op_in_window")),
            ])
            + " |"
        )

    lines += [
        "",
        "## Notes",
        "",
        "- `Pre-Hot Loads` is the sum of LDS and VMEM loads in the parsed window before the first `v_dot`, `v_wmma`, or `v_mfma`.",
        "- `Final Wait` is the parsed `lgkmcnt` value of the last `s_waitcnt` before that hot op when present.",
        "- This is a static schedule triage aid. Promotion still requires focused CPU-reference gates, route traces, and same-runner timing.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Summarize a CMake-built HSACO family for static schedule/resource triage.")
    parser.add_argument("--hsaco-dir", required=True, type=pathlib.Path)
    parser.add_argument("--glob", action="append", default=[],
                        help="fnmatch pattern relative to --hsaco-dir, repeatable.")
    parser.add_argument("--symbol", help="Optional symbol substring to use for every HSACO.")
    parser.add_argument("--llvm-objdump", default="llvm-objdump", type=pathlib.Path)
    parser.add_argument("--llvm-readelf", default="llvm-readelf", type=pathlib.Path)
    parser.add_argument("--out-json", type=pathlib.Path)
    parser.add_argument("--out-md", type=pathlib.Path)
    args = parser.parse_args()

    hsacos = select_hsacos(args.hsaco_dir, args.glob)
    if not hsacos:
        raise SystemExit("no HSACO files matched")

    rows = [
        load_hsaco_summary(path, args.llvm_objdump, args.llvm_readelf, args.symbol)
        for path in hsacos
    ]
    payload = {
        "hsaco_dir": str(args.hsaco_dir),
        "patterns": args.glob,
        "count": len(rows),
        "rows": rows,
    }

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.out_md:
        write_markdown(args.out_md, rows)

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
