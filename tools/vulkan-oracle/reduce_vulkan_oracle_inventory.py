#!/usr/bin/env python3
import argparse
import collections
import json
import pathlib


PIPELINE_KEY_FIELDS = [
    "pipeline",
    "entrypoint",
    "trace_source",
    "trace_reduction",
    "trace_workgroup_mode",
    "spv_hash",
    "spv_size",
    "push_constant_size",
    "parameter_count",
    "wg_denoms",
    "spec",
    "disable_robustness",
    "require_full_subgroups",
    "required_subgroup_size",
    "is_64b_indexing",
]


def pipeline_key(row):
    return tuple(json.dumps(row.get(field), sort_keys=True, separators=(",", ":")) for field in PIPELINE_KEY_FIELDS)


def pipeline_metadata(row):
    return {field: row.get(field) for field in PIPELINE_KEY_FIELDS}


def node_signature(node):
    if not node:
        return None
    return {
        "op": node.get("op"),
        "name": node.get("name"),
        "type": node.get("type"),
        "ne": node.get("ne"),
    }


def tensor_signature(tensor):
    if not tensor:
        return None
    return {
        "name": tensor.get("name"),
        "op": tensor.get("op"),
        "type": tensor.get("type"),
        "ne": tensor.get("ne"),
        "nb": tensor.get("nb"),
    }


def tensor_shape_signature(tensor):
    if not tensor:
        return None
    return {
        "op": tensor.get("op"),
        "type": tensor.get("type"),
        "ne": tensor.get("ne"),
        "nb": tensor.get("nb"),
    }


def dispatch_signature(row):
    return {
        "pipeline": row.get("pipeline"),
        "entrypoint": row.get("entrypoint"),
        "spv_hash": row.get("spv_hash"),
        "spec": row.get("spec"),
        "workgroups": row.get("workgroups"),
        "elements": row.get("elements"),
        "push_constant_words": row.get("push_constant_words"),
        "bindings": [
            {"binding": b.get("binding"), "offset": b.get("offset"), "range": b.get("range")}
            for b in row.get("bindings", [])
        ],
        "node": node_signature(row.get("node")),
        "src0": tensor_signature(row.get("src0")),
        "src1": tensor_signature(row.get("src1")),
        "src2": tensor_signature(row.get("src2")),
        "src3": tensor_signature(row.get("src3")),
    }


def normalized_shape_signature(row):
    node = row.get("node") or {}
    return {
        "pipeline": row.get("pipeline"),
        "entrypoint": row.get("entrypoint"),
        "spv_hash": row.get("spv_hash"),
        "spec": row.get("spec"),
        "workgroups": row.get("workgroups"),
        "elements": row.get("elements"),
        "node": {
            "op": node.get("op"),
            "type": node.get("type"),
            "ne": node.get("ne"),
        },
        "src0": tensor_shape_signature(row.get("src0")),
        "src1": tensor_shape_signature(row.get("src1")),
        "src2": tensor_shape_signature(row.get("src2")),
        "src3": tensor_shape_signature(row.get("src3")),
    }


def signature_key(sig):
    return json.dumps(sig, sort_keys=True, separators=(",", ":"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True, type=pathlib.Path)
    parser.add_argument("jsonl", type=pathlib.Path)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    compile_rows = []
    dispatch_rows = []
    with args.jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("event") == "pipeline_compile":
                compile_rows.append(row)
            elif row.get("event") == "dispatch":
                dispatch_rows.append(row)

    pipeline_counts = collections.Counter()
    pipeline_examples = {}
    for row in dispatch_rows:
        key = pipeline_key(row)
        pipeline_counts[key] += 1
        pipeline_examples.setdefault(key, row)

    pipeline_inventory = []
    for key, count in pipeline_counts.most_common():
        row = pipeline_examples[key]
        item = pipeline_metadata(row)
        item["dispatch_count"] = count
        item["example_workgroups"] = row.get("workgroups")
        item["example_node"] = node_signature(row.get("node"))
        item["example_src0"] = tensor_signature(row.get("src0"))
        item["example_src1"] = tensor_signature(row.get("src1"))
        pipeline_inventory.append(item)

    signature_counts = collections.Counter()
    signature_examples = {}
    for row in dispatch_rows:
        sig = dispatch_signature(row)
        key = signature_key(sig)
        signature_counts[key] += 1
        signature_examples.setdefault(key, sig)

    dispatch_signature_inventory = []
    for key, count in signature_counts.most_common():
        sig = signature_examples[key]
        sig["dispatch_count"] = count
        dispatch_signature_inventory.append(sig)

    normalized_counts = collections.Counter()
    normalized_examples = {}
    for row in dispatch_rows:
        sig = normalized_shape_signature(row)
        key = signature_key(sig)
        normalized_counts[key] += 1
        normalized_examples.setdefault(key, sig)

    normalized_shape_inventory = []
    for key, count in normalized_counts.most_common():
        sig = normalized_examples[key]
        sig["dispatch_count"] = count
        normalized_shape_inventory.append(sig)

    (args.out_dir / "pipeline_inventory.json").write_text(
        json.dumps(pipeline_inventory, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "dispatch_signature_inventory.json").write_text(
        json.dumps(dispatch_signature_inventory, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "normalized_shape_inventory.json").write_text(
        json.dumps(normalized_shape_inventory, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (args.out_dir / "dispatches_full.jsonl").open("w", encoding="utf-8") as f:
        for row in dispatch_rows:
            f.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")

    md = []
    md.append("# Vulkan Oracle Kernel Inventory\n\n")
    md.append(f"- pipeline compile rows: `{len(compile_rows)}`\n")
    md.append(f"- dispatch rows: `{len(dispatch_rows)}`\n")
    md.append(f"- unique pipeline identities: `{len(pipeline_inventory)}`\n")
    md.append(f"- unique dispatch signatures: `{len(dispatch_signature_inventory)}`\n\n")
    md.append(f"- normalized shape signatures: `{len(normalized_shape_inventory)}`\n\n")
    md.append("## Pipelines By Dispatch Count\n\n")
    md.append("| Count | Pipeline | Hash | Spec | Workgroups Example | Node Example |\n")
    md.append("| ---: | --- | --- | --- | --- | --- |\n")
    for item in pipeline_inventory:
        node = item.get("example_node") or {}
        src0 = item.get("example_src0") or {}
        node_text = f"{node.get('op','')} {node.get('name','')} {src0.get('type','')} {src0.get('ne','')}"
        md.append(
            f"| {item['dispatch_count']} | `{item.get('pipeline')}` | `{item.get('spv_hash')}` | "
            f"`{item.get('spec')}` | `{item.get('example_workgroups')}` | {node_text} |\n"
        )
    md.append("\n## Top Normalized Shape Signatures\n\n")
    md.append("| Count | Pipeline | Workgroups | Node | Src0 | Src1 |\n")
    md.append("| ---: | --- | --- | --- | --- | --- |\n")
    for sig in normalized_shape_inventory[:80]:
        node = sig.get("node") or {}
        src0 = sig.get("src0") or {}
        src1 = sig.get("src1") or {}
        md.append(
            f"| {sig['dispatch_count']} | `{sig.get('pipeline')}` | `{sig.get('workgroups')}` | "
            f"{node.get('op','')} `{node.get('ne','')}` | "
            f"{src0.get('type','')} `{src0.get('ne','')}` | {src1.get('type','')} `{src1.get('ne','')}` |\n"
        )
    md.append("\n## Top Dispatch Signatures\n\n")
    md.append("| Count | Pipeline | Workgroups | Node | Src0 | Src1 |\n")
    md.append("| ---: | --- | --- | --- | --- | --- |\n")
    for sig in dispatch_signature_inventory[:80]:
        node = sig.get("node") or {}
        src0 = sig.get("src0") or {}
        src1 = sig.get("src1") or {}
        md.append(
            f"| {sig['dispatch_count']} | `{sig.get('pipeline')}` | `{sig.get('workgroups')}` | "
            f"{node.get('op','')} `{node.get('name','')}` | "
            f"{src0.get('type','')} `{src0.get('ne','')}` | {src1.get('type','')} `{src1.get('ne','')}` |\n"
        )
    (args.out_dir / "kernel_inventory.md").write_text("".join(md), encoding="utf-8")

    print(f"pipeline_identities={len(pipeline_inventory)}")
    print(f"dispatch_signatures={len(dispatch_signature_inventory)}")
    print(f"normalized_shape_signatures={len(normalized_shape_inventory)}")


if __name__ == "__main__":
    raise SystemExit(main())
