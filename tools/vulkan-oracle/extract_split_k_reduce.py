#!/usr/bin/env python3
import argparse
import collections
import json
import pathlib
import re


def load_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def shape(tensor):
    if not tensor:
        return None
    return {
        "name": tensor.get("name"),
        "op": tensor.get("op"),
        "type": tensor.get("type"),
        "ne": tensor.get("ne"),
        "nb": tensor.get("nb"),
    }


def node(row):
    n = row.get("node") or {}
    return {
        "idx": n.get("idx"),
        "name": n.get("name"),
        "op": n.get("op"),
        "type": n.get("type"),
        "ne": n.get("ne"),
        "nb": n.get("nb"),
    }


def binding_key(binding):
    if not binding:
        return None
    return (binding.get("buffer"), binding.get("offset"), binding.get("range"))


def binding_summary(binding):
    if not binding:
        return None
    return {
        "binding": binding.get("binding"),
        "buffer": binding.get("buffer"),
        "offset": binding.get("offset"),
        "range": binding.get("range"),
    }


def normalize_node_name(name):
    if name is None:
        return None
    return re.sub(r"-\d+$", "-N", name)


def compact_row(row):
    return {
        "pipeline": row.get("pipeline"),
        "entrypoint": row.get("entrypoint"),
        "spv_hash": row.get("spv_hash"),
        "spec": row.get("spec"),
        "workgroups": row.get("workgroups"),
        "elements": row.get("elements"),
        "push_constant_words": row.get("push_constant_words"),
        "node": node(row),
        "src0": shape(row.get("src0")),
        "src1": shape(row.get("src1")),
        "bindings": [binding_summary(b) for b in row.get("bindings", [])],
    }


def find_producer(rows, reduce_index):
    reduce_row = rows[reduce_index]
    reduce_node = (reduce_row.get("node") or {}).get("idx")
    reduce_input = binding_key((reduce_row.get("bindings") or [None])[0])

    best = None
    for index in range(reduce_index - 1, max(-1, reduce_index - 12), -1):
        row = rows[index]
        if row.get("event") != "dispatch":
            continue
        if row.get("pipeline") == "split_k_reduce":
            continue
        if (row.get("node") or {}).get("idx") != reduce_node:
            continue
        bindings = row.get("bindings") or []
        for binding in reversed(bindings):
            if binding_key(binding) == reduce_input:
                best = (index, row, binding)
                break
        if best is not None:
            return best

    return None


def summarize_capture(label, jsonl):
    rows = load_jsonl(jsonl)
    reductions = []
    missing_producers = []

    for index, row in enumerate(rows):
        if row.get("event") != "dispatch" or row.get("pipeline") != "split_k_reduce":
            continue

        producer = find_producer(rows, index)
        producer_index = None
        producer_row = None
        producer_binding = None
        if producer is None:
            missing_producers.append(index)
        else:
            producer_index, producer_row, producer_binding = producer

        pc = row.get("push_constant_words") or []
        reduce_input = (row.get("bindings") or [None])[0]
        scratch_bytes = reduce_input.get("range") if reduce_input else None
        output_elements = pc[0] if len(pc) >= 1 else None
        reduce_factor = pc[1] if len(pc) >= 2 else None
        expected_scratch_bytes = None
        if isinstance(output_elements, int) and isinstance(reduce_factor, int):
            expected_scratch_bytes = output_elements * reduce_factor * 4

        reductions.append(
            {
                "dispatch_index": index,
                "producer_dispatch_index": producer_index,
                "producer_distance": index - producer_index if producer_index is not None else None,
                "scratch_matches_producer": (
                    binding_key(producer_binding) == binding_key(reduce_input)
                    if producer_binding is not None and reduce_input is not None
                    else False
                ),
                "scratch_bytes": scratch_bytes,
                "expected_scratch_bytes": expected_scratch_bytes,
                "output_elements": output_elements,
                "reduce_factor": reduce_factor,
                "producer_output_binding": binding_summary(producer_binding),
                "producer": compact_row(producer_row) if producer_row else None,
                "reduce": compact_row(row),
            }
        )

    family_counts = collections.Counter()
    family_examples = {}
    exact_counts = collections.Counter()
    exact_examples = {}

    for item in reductions:
        reduce_row = item["reduce"]
        producer_row = item["producer"] or {}
        reduce_node = reduce_row.get("node") or {}
        src0 = reduce_row.get("src0") or {}
        src1 = reduce_row.get("src1") or {}
        family_key = json.dumps(
            {
                "producer_pipeline": producer_row.get("pipeline"),
                "reduce_pipeline": reduce_row.get("pipeline"),
                "node_op": reduce_node.get("op"),
                "node_ne": reduce_node.get("ne"),
                "src0_type": src0.get("type"),
                "src0_ne": src0.get("ne"),
                "src1_type": src1.get("type"),
                "src1_ne": src1.get("ne"),
                "producer_workgroups": producer_row.get("workgroups"),
                "producer_elements": producer_row.get("elements"),
                "reduce_workgroups": reduce_row.get("workgroups"),
                "reduce_elements": reduce_row.get("elements"),
                "output_elements": item.get("output_elements"),
                "reduce_factor": item.get("reduce_factor"),
                "scratch_bytes": item.get("scratch_bytes"),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        exact_key = json.dumps(
            {
                "node_name": reduce_node.get("name"),
                "node_name_family": normalize_node_name(reduce_node.get("name")),
                "node_idx": reduce_node.get("idx"),
                "family_key": json.loads(family_key),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        family_counts[family_key] += 1
        family_examples.setdefault(family_key, item)
        exact_counts[exact_key] += 1
        exact_examples.setdefault(exact_key, item)

    families = []
    for key, count in family_counts.most_common():
        example = family_examples[key]
        decoded = json.loads(key)
        decoded["dispatch_count"] = count
        decoded["example_node_name"] = (example["reduce"].get("node") or {}).get("name")
        decoded["example_dispatch_index"] = example["dispatch_index"]
        decoded["scratch_matches_producer"] = example["scratch_matches_producer"]
        decoded["scratch_bytes_match_expected"] = example["scratch_bytes"] == example["expected_scratch_bytes"]
        families.append(decoded)

    exact_nodes = []
    for key, count in exact_counts.most_common():
        example = exact_examples[key]
        decoded = json.loads(key)
        decoded["dispatch_count"] = count
        decoded["example_dispatch_index"] = example["dispatch_index"]
        exact_nodes.append(decoded)

    return {
        "label": label,
        "jsonl": str(jsonl),
        "dispatch_rows": len(rows),
        "split_k_reduce_dispatches": len(reductions),
        "missing_producer_dispatch_indices": missing_producers,
        "families": families,
        "exact_nodes": exact_nodes,
        "reductions": reductions,
    }


def md_shape(value):
    return "`" + json.dumps(value, separators=(",", ":")) + "`"


def write_markdown(path, captures):
    lines = []
    lines.append("# Vulkan Split-K Reduce Oracle Summary\n\n")
    lines.append("This file summarizes `split_k_reduce` dispatches from Vulkan oracle `dispatches_full.jsonl` traces and pairs each reduce with the preceding producer dispatch that writes the scratch input.\n\n")

    lines.append("## Capture Totals\n\n")
    lines.append("| Capture | Dispatches | split_k_reduce | Missing producer pairs |\n")
    lines.append("| --- | ---: | ---: | ---: |\n")
    for capture in captures:
        lines.append(
            f"| `{capture['label']}` | {capture['dispatch_rows']} | {capture['split_k_reduce_dispatches']} | "
            f"{len(capture['missing_producer_dispatch_indices'])} |\n"
        )

    for capture in captures:
        lines.append(f"\n## {capture['label']}\n\n")
        lines.append(f"- source: `{capture['jsonl']}`\n")
        lines.append(f"- split-K reductions: `{capture['split_k_reduce_dispatches']}`\n")
        lines.append(f"- missing producer pairs: `{len(capture['missing_producer_dispatch_indices'])}`\n\n")

        lines.append("### Normalized Families\n\n")
        lines.append("| Count | Producer | Src0 | Src1 | Node | Producer WG | Reduce WG | Output elems | Factor | Scratch bytes | Scratch check |\n")
        lines.append("| ---: | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |\n")
        for family in capture["families"]:
            scratch_check = "ok" if family.get("scratch_matches_producer") and family.get("scratch_bytes_match_expected") else "check"
            lines.append(
                f"| {family['dispatch_count']} | `{family.get('producer_pipeline')}` | "
                f"{family.get('src0_type')} {md_shape(family.get('src0_ne'))} | "
                f"{family.get('src1_type')} {md_shape(family.get('src1_ne'))} | "
                f"{family.get('node_op')} {md_shape(family.get('node_ne'))} | "
                f"{md_shape(family.get('producer_workgroups'))} | {md_shape(family.get('reduce_workgroups'))} | "
                f"{family.get('output_elements')} | {family.get('reduce_factor')} | {family.get('scratch_bytes')} | {scratch_check} |\n"
            )

        lines.append("\n### First Producer/Reduce Examples\n\n")
        lines.append("| Reduce index | Producer index | Node | Producer pipeline | Scratch binding | Reduce bindings | Push constants |\n")
        lines.append("| ---: | ---: | --- | --- | --- | --- | --- |\n")
        for item in capture["reductions"][:12]:
            reduce_node = item["reduce"].get("node") or {}
            lines.append(
                f"| {item['dispatch_index']} | {item.get('producer_dispatch_index')} | "
                f"`{reduce_node.get('name')}` | `{(item.get('producer') or {}).get('pipeline')}` | "
                f"{md_shape(item.get('producer_output_binding'))} | "
                f"{md_shape(item['reduce'].get('bindings'))} | "
                f"{md_shape(item['reduce'].get('push_constant_words'))} |\n"
            )

    path.write_text("".join(lines), encoding="utf-8")


def parse_capture(value):
    if "=" in value:
        label, path = value.split("=", 1)
        if not label:
            raise argparse.ArgumentTypeError("capture label must not be empty")
        return label, pathlib.Path(path)
    path = pathlib.Path(value)
    return path.parent.parent.name, path


def main():
    parser = argparse.ArgumentParser(
        description="Extract Vulkan split_k_reduce producer/reduce pairs from oracle dispatch traces."
    )
    parser.add_argument(
        "--capture",
        action="append",
        required=True,
        type=parse_capture,
        metavar="LABEL=DISPATCHES_JSONL",
        help="Capture label and inventory/dispatches_full.jsonl path. May be repeated.",
    )
    parser.add_argument("--out-json", required=True, type=pathlib.Path)
    parser.add_argument("--out-md", required=True, type=pathlib.Path)
    args = parser.parse_args()

    captures = []
    for label, path in args.capture:
        if not path.exists():
            raise SystemExit(f"missing capture jsonl: {path}")
        captures.append(summarize_capture(label, path))

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(captures, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(args.out_md, captures)

    total = sum(capture["split_k_reduce_dispatches"] for capture in captures)
    print(f"captures={len(captures)}")
    print(f"split_k_reduce_dispatches={total}")
    for capture in captures:
        print(
            f"{capture['label']}: split_k_reduce={capture['split_k_reduce_dispatches']} "
            f"families={len(capture['families'])} missing_producers={len(capture['missing_producer_dispatch_indices'])}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
