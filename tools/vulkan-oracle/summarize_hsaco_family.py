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
    wmma_score = event_summary.get("wmma_score", {})
    store_score = event_summary.get("store_cluster_score", {})

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
        "wmma_score": wmma_score,
        "store_cluster_score": store_score,
    }

    for field, prefixes in OPCODE_FIELDS:
        row[field] = sum(
            count for opcode, count in interesting.items()
            if opcode.startswith(prefixes)
        )

    return row


def load_reference_contract(path):
    if not path:
        return None

    data = json.loads(path.read_text(encoding="utf-8"))
    if "store_cluster_score" in data or "wmma_score" in data:
        return {
            "store_cluster_score": data.get("store_cluster_score", {}),
            "wmma_score": data.get("wmma_score", {}),
            "hot_op_score": data.get("hot_op_score", {}),
        }

    lhs_key = data.get("lhs_key", "radv")
    lhs = data.get(lhs_key, {})
    event_summary = lhs.get("event_summary", {})
    return {
        "name": lhs.get("name"),
        "store_cluster_score": event_summary.get("store_cluster_score", {}),
        "wmma_score": event_summary.get("wmma_score", {}),
        "hot_op_score": event_summary.get("hot_op_score", {}),
    }


def add_reference_deltas(rows, reference):
    if not reference:
        for row in rows:
            add_contract_summary(row)
        return rows

    ref_store = reference.get("store_cluster_score", {})
    ref_wmma = reference.get("wmma_score", {})
    ref_hot = reference.get("hot_op_score", {})
    store_fields = (
        "store_clusters",
        "store_ops",
        "vmem_store_ops",
        "lds_store_ops",
        "buffer_store_ops",
        "global_store_ops",
    )
    wmma_fields = (
        "pre_wmma_ds_load_b64",
        "load_b64_immediately_before_final_wait",
        "final_pre_wmma_lgkmcnt",
        "wmma_in_window",
    )
    hot_fields = (
        "pre_hot_lds_load",
        "pre_hot_vmem_load",
        "load_like_immediately_before_final_wait",
        "final_pre_hot_lgkmcnt",
        "hot_op_in_window",
        "load_like_after_first_hot",
    )

    for row in rows:
        store = row.get("store_cluster_score", {})
        wmma = row.get("wmma_score", {})
        store_delta = {}
        store_gap = 0
        for field in store_fields:
            lhs = ref_store.get(field)
            rhs = store.get(field)
            if isinstance(lhs, int) and isinstance(rhs, int):
                delta = rhs - lhs
                store_delta[field] = delta
                store_gap += abs(delta)
            else:
                store_delta[field] = None

        wmma_delta = {}
        wmma_gap = 0
        for field in wmma_fields:
            lhs = ref_wmma.get(field)
            rhs = wmma.get(field)
            if isinstance(lhs, int) and isinstance(rhs, int):
                delta = rhs - lhs
                wmma_delta[field] = delta
                wmma_gap += abs(delta)
            else:
                wmma_delta[field] = None

        hot_delta = {}
        hot_gap = 0
        for field in hot_fields:
            lhs = ref_hot.get(field)
            rhs = row.get("hot_op_score", {}).get(field)
            if isinstance(lhs, int) and isinstance(rhs, int):
                delta = rhs - lhs
                hot_delta[field] = delta
                hot_gap += abs(delta)
            else:
                hot_delta[field] = None

        row["reference_delta"] = {
            "store": store_delta,
            "wmma": wmma_delta,
            "hot": hot_delta,
            "store_gap": store_gap,
            "wmma_gap": wmma_gap,
            "hot_gap": hot_gap,
            "total_gap": store_gap + wmma_gap,
        }
        add_contract_summary(row)
    return rows


def add_contract_summary(row):
    failures = []
    resources = row.get("resources", {})
    scratch = resources.get("scratch_bytes") or 0
    sgpr_spills = resources.get("sgpr_spills") or 0
    vgpr_spills = resources.get("vgpr_spills") or 0
    if scratch:
        failures.append(f"scratch={scratch}")
    if sgpr_spills:
        failures.append(f"sgpr_spills={sgpr_spills}")
    if vgpr_spills:
        failures.append(f"vgpr_spills={vgpr_spills}")

    reference_delta = row.get("reference_delta", {})
    for group_name in ("store", "wmma"):
        for field, delta in sorted(reference_delta.get(group_name, {}).items()):
            if isinstance(delta, int) and delta != 0:
                failures.append(f"{group_name}.{field}={delta:+d}")

    row["contract_failures"] = failures
    row["static_contract_pass"] = not failures
    return row


def sort_rows(rows, sort_key):
    if sort_key == "name":
        return sorted(rows, key=lambda row: row["name"])
    if sort_key == "contract":
        return sorted(rows, key=lambda row: (
            0 if row.get("static_contract_pass") else 1,
            row.get("resources", {}).get("scratch_bytes") or 0,
            row.get("resources", {}).get("sgpr_spills") or 0,
            row.get("resources", {}).get("vgpr_spills") or 0,
            row.get("reference_delta", {}).get("total_gap", 10**9),
            row.get("reference_delta", {}).get("store_gap", 10**9),
            row.get("reference_delta", {}).get("wmma_gap", 10**9),
            row["name"],
        ))
    if sort_key == "store-gap":
        return sorted(rows, key=lambda row: (
            row.get("reference_delta", {}).get("store_gap", 10**9),
            row.get("resources", {}).get("scratch_bytes") or 0,
            row.get("resources", {}).get("vgpr_spills") or 0,
            row["name"],
        ))
    if sort_key == "total-gap":
        return sorted(rows, key=lambda row: (
            row.get("reference_delta", {}).get("total_gap", 10**9),
            row.get("reference_delta", {}).get("hot_gap", 10**9),
            row.get("resources", {}).get("scratch_bytes") or 0,
            row.get("resources", {}).get("vgpr_spills") or 0,
            row["name"],
        ))
    if sort_key == "resource":
        return sorted(rows, key=lambda row: (
            row.get("resources", {}).get("scratch_bytes") or 0,
            row.get("resources", {}).get("vgpr_spills") or 0,
            row.get("resources", {}).get("vgpr") or 0,
            row["name"],
        ))
    return rows


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


def write_markdown(path, rows, reference=None):
    lines = [
        "# HSACO Family Summary",
        "",
    ]
    if reference:
        lines += [
            "Reference contract:",
            "",
            f"- name: `{reference.get('name') or ''}`",
            f"- store score: `{json.dumps(reference.get('store_cluster_score', {}), sort_keys=True)}`",
            f"- WMMA score: `{json.dumps(reference.get('wmma_score', {}), sort_keys=True)}`",
            f"- hot-op score: `{json.dumps(reference.get('hot_op_score', {}), sort_keys=True)}`",
            "",
        ]
    lines += [
        "| HSACO | Static Contract | Wave | VGPR | SGPR | LDS | Scratch | VGPR Spills | v_dot | v_wmma | LDS Read | LDS Write | VMEM Load | VMEM Store | Store Clusters | Store VMEM | Store LDS | dClusters | dVMEM Store | dLDS Store | Store Gap | WMMA Gap | Hot Gap | Total Gap | Wait | Barrier | Hot Op | Pre-Hot Loads | Immediate Loads | Final Wait | Hot Window | Loads After Hot | Contract Failures |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        res = row["resources"]
        hot = row.get("hot_op_score", {})
        store = row.get("store_cluster_score", {})
        ref_delta = row.get("reference_delta", {})
        store_delta = ref_delta.get("store", {})
        pre_hot_loads = hot.get("pre_hot_lds_load", 0) + hot.get("pre_hot_vmem_load", 0)
        lines.append(
            "| "
            + " | ".join([
                f"`{row['name']}`",
                "`pass`" if row.get("static_contract_pass") else "`fail`",
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
                render_value(store.get("store_clusters")),
                render_value(store.get("vmem_store_ops")),
                render_value(store.get("lds_store_ops")),
                render_value(store_delta.get("store_clusters")),
                render_value(store_delta.get("vmem_store_ops")),
                render_value(store_delta.get("lds_store_ops")),
                render_value(ref_delta.get("store_gap")),
                render_value(ref_delta.get("wmma_gap")),
                render_value(ref_delta.get("hot_gap")),
                render_value(ref_delta.get("total_gap")),
                render_value(row.get("s_waitcnt")),
                render_value(row.get("s_barrier")),
                f"`{row.get('hot_opcode') or ''}`",
                render_value(pre_hot_loads),
                render_value(hot.get("load_like_immediately_before_final_wait")),
                render_value(hot.get("final_pre_hot_lgkmcnt")),
                render_value(hot.get("hot_op_in_window")),
                render_value(hot.get("load_like_after_first_hot")),
                "<br>".join(f"`{item}`" for item in row.get("contract_failures", [])),
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
    parser.add_argument("--reference-compare-json", type=pathlib.Path,
                        help="RADV-vs-HSACO compare JSON; uses the LHS event scores as the ranking contract.")
    parser.add_argument("--sort", choices=("input", "name", "contract", "store-gap", "total-gap", "resource"),
                        default="input")
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
    reference = load_reference_contract(args.reference_compare_json)
    add_reference_deltas(rows, reference)
    rows = sort_rows(rows, args.sort)
    payload = {
        "hsaco_dir": str(args.hsaco_dir),
        "patterns": args.glob,
        "reference": reference,
        "sort": args.sort,
        "count": len(rows),
        "rows": rows,
    }

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.out_md:
        write_markdown(args.out_md, rows, reference)

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
