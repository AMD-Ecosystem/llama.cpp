#!/usr/bin/env python3
"""Classify RADV AMDGCN store paths in a Vulkan oracle ISA dump.

RADV shader dumps include all control-flow alternatives for a shader. For
matmul coopmat stores this means the static opcode totals can combine the hot
full-aligned path with unaligned and partial-edge fallbacks. This helper keeps
those counts separate before using a static ISA signature as a HIP porting
target.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re


BB_RE = re.compile(r"^(BB[0-9]+):")
INSTRUCTION_RE = re.compile(r"^\s+([A-Za-z0-9_.]+)\b")


def strip_comment(line: str) -> str:
    for sep in ("//", ";"):
        if sep in line:
            line = line.split(sep, 1)[0]
    return line.rstrip()


def parse_blocks(path: pathlib.Path) -> list[dict]:
    blocks: list[dict] = []
    current: dict | None = None
    for line_no, raw in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        bb_match = BB_RE.match(raw.strip())
        if bb_match:
            current = {"bb": bb_match.group(1), "start_line": line_no, "events": []}
            blocks.append(current)
            continue
        if current is None:
            continue
        text = strip_comment(raw)
        insn_match = INSTRUCTION_RE.match(text)
        if not insn_match:
            continue
        current["events"].append(
            {
                "line": line_no,
                "opcode": insn_match.group(1),
                "text": raw.strip(),
            }
        )
    return blocks


def count_ops(events: list[dict]) -> collections.Counter:
    return collections.Counter(event["opcode"] for event in events)


def branch_targets(events: list[dict]) -> list[str]:
    targets: list[str] = []
    for event in events:
        if not event["opcode"].startswith(("s_branch", "s_cbranch")):
            continue
        parts = event["text"].split()
        if len(parts) >= 2 and parts[1].startswith("BB"):
            targets.append(parts[1])
    return targets


def classify_block(counts: collections.Counter) -> str:
    buffer_stores = sum(count for op, count in counts.items() if op.startswith("buffer_store"))
    lds_stores = sum(count for op, count in counts.items() if op.startswith(("ds_store", "ds_write")))
    lds_loads = sum(count for op, count in counts.items() if op.startswith(("ds_load", "ds_read")))
    branches = sum(count for op, count in counts.items() if op.startswith(("s_branch", "s_cbranch")))
    exec_branches = counts.get("s_cbranch_execz", 0)

    if buffer_stores and not lds_stores and not lds_loads:
        return "direct_global_store"
    if buffer_stores and lds_stores and lds_loads and not exec_branches:
        return "full_tile_staged_store"
    if lds_stores and exec_branches and not buffer_stores:
        return "partial_tile_stage_entry"
    if buffer_stores and lds_loads and exec_branches == 0 and branches == 0:
        return "partial_tile_scalar_store"
    if lds_stores and not buffer_stores:
        return "lds_stage_only"
    if buffer_stores:
        return "other_buffer_store"
    if lds_stores or lds_loads:
        return "other_lds_store_path"
    return "non_store"


def summarize(path: pathlib.Path) -> dict:
    blocks = parse_blocks(path)
    rows = []
    summary = collections.defaultdict(lambda: collections.Counter())
    total_opcodes = collections.Counter()

    for block in blocks:
        counts = count_ops(block["events"])
        total_opcodes.update(counts)
        has_store_path = any(
            op.startswith(("buffer_store", "global_store", "flat_store", "ds_store", "ds_write", "ds_load", "ds_read"))
            for op in counts
        )
        if not has_store_path:
            continue
        klass = classify_block(counts)
        buffer_stores = sum(count for op, count in counts.items() if op.startswith("buffer_store"))
        lds_stores = sum(count for op, count in counts.items() if op.startswith(("ds_store", "ds_write")))
        lds_loads = sum(count for op, count in counts.items() if op.startswith(("ds_load", "ds_read")))
        waitcnt = sum(count for op, count in counts.items() if op == "s_waitcnt")
        branches = sum(count for op, count in counts.items() if op.startswith(("s_branch", "s_cbranch")))
        row = {
            "bb": block["bb"],
            "start_line": block["start_line"],
            "class": klass,
            "buffer_store_ops": buffer_stores,
            "lds_store_ops": lds_stores,
            "lds_load_ops": lds_loads,
            "waitcnt_ops": waitcnt,
            "branch_ops": branches,
            "branch_targets": branch_targets(block["events"]),
            "opcodes": dict(sorted(counts.items())),
            "first_events": block["events"][:8],
        }
        rows.append(row)
        summary[klass]["blocks"] += 1
        summary[klass]["buffer_store_ops"] += buffer_stores
        summary[klass]["lds_store_ops"] += lds_stores
        summary[klass]["lds_load_ops"] += lds_loads
        summary[klass]["waitcnt_ops"] += waitcnt
        summary[klass]["branch_ops"] += branches

    return {
        "schema": "radv-store-path-classification-v1",
        "isa": str(path),
        "totals": {
            "buffer_store_ops": sum(count for op, count in total_opcodes.items() if op.startswith("buffer_store")),
            "lds_store_ops": sum(count for op, count in total_opcodes.items() if op.startswith(("ds_store", "ds_write"))),
            "lds_load_ops": sum(count for op, count in total_opcodes.items() if op.startswith(("ds_load", "ds_read"))),
            "v_wmma_ops": sum(count for op, count in total_opcodes.items() if op.startswith("v_wmma")),
            "barriers": total_opcodes.get("s_barrier", 0),
        },
        "classes": {key: dict(value) for key, value in sorted(summary.items())},
        "store_blocks": rows,
    }


def write_markdown(path: pathlib.Path, payload: dict) -> None:
    lines = [
        "# RADV Store Path Classification",
        "",
        f"- ISA: `{payload['isa']}`",
        "",
        "## Static Totals",
        "",
        "| Metric | Count |",
        "| --- | ---: |",
    ]
    for key, value in payload["totals"].items():
        lines.append(f"| `{key}` | {value} |")

    lines += [
        "",
        "## Classified Store Paths",
        "",
        "| Class | Blocks | Buffer Stores | LDS Stores | LDS Loads | Waits | Branches |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for klass, counts in payload["classes"].items():
        lines.append(
            f"| `{klass}` | {counts.get('blocks', 0)} | {counts.get('buffer_store_ops', 0)} | "
            f"{counts.get('lds_store_ops', 0)} | {counts.get('lds_load_ops', 0)} | "
            f"{counts.get('waitcnt_ops', 0)} | {counts.get('branch_ops', 0)} |"
        )

    lines += [
        "",
        "## Store Blocks",
        "",
        "| BB | Line | Class | Buffer Stores | LDS Stores | LDS Loads | Branch Targets |",
        "| --- | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for block in payload["store_blocks"]:
        targets = ", ".join(block["branch_targets"])
        lines.append(
            f"| `{block['bb']}` | {block['start_line']} | `{block['class']}` | "
            f"{block['buffer_store_ops']} | {block['lds_store_ops']} | {block['lds_load_ops']} | `{targets}` |"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isa", required=True, type=pathlib.Path)
    parser.add_argument("--json-out", required=True, type=pathlib.Path)
    parser.add_argument("--md-out", required=True, type=pathlib.Path)
    args = parser.parse_args()

    payload = summarize(args.isa)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(args.md_out, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
