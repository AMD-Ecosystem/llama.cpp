#!/usr/bin/env python3
"""Extract RADV store-path branch context from AMDGCN ISA.

RADV shader dumps include all writeback alternatives in one ISA file. The
store-path classifier separates the store blocks, but schedule porting also
needs the scalar branch context that decides which path is active for a shape.
This helper emits the predecessor branch tails for each classified store block
so p512 full-tile, p513 tail, and staged fallback targets can be reasoned about
without manually rereading the full shader dump.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re


BB_RE = re.compile(r"^(BB[0-9]+):")
INSTRUCTION_RE = re.compile(r"^\s+([A-Za-z0-9_.]+)\b")
BRANCH_TARGET_RE = re.compile(r"\b(BB[0-9]+)\b")


INTERESTING_BRANCH_PREFIXES = (
    "s_cmp",
    "s_cbranch",
    "s_branch",
    "s_and",
    "s_or",
    "s_xor",
    "s_mov",
    "s_cselect",
    "s_waitcnt",
)


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
            if current is not None:
                current["end_line"] = line_no - 1
            current = {"bb": bb_match.group(1), "start_line": line_no, "end_line": line_no, "events": [], "lines": []}
            blocks.append(current)
        if current is not None:
            current["lines"].append(raw)
            current["end_line"] = line_no
        if bb_match or current is None:
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


def event_target(event: dict) -> str | None:
    match = BRANCH_TARGET_RE.search(event["text"])
    return match.group(1) if match else None


def branch_edges(blocks: list[dict]) -> dict[str, list[dict]]:
    edges: dict[str, list[dict]] = collections.defaultdict(list)
    block_by_name = {block["bb"]: block for block in blocks}
    for index, block in enumerate(blocks):
        events = block["events"]
        branch_events = [event for event in events if event["opcode"].startswith(("s_branch", "s_cbranch"))]
        if branch_events:
            last = branch_events[-1]
            target = event_target(last)
            if target in block_by_name:
                edges[target].append(
                    {
                        "predecessor": block["bb"],
                        "edge_kind": "conditional_branch" if last["opcode"].startswith("s_cbranch") else "unconditional_branch",
                        "branch_event": last,
                    }
                )
            if last["opcode"].startswith("s_cbranch") and index + 1 < len(blocks):
                edges[blocks[index + 1]["bb"]].append(
                    {
                        "predecessor": block["bb"],
                        "edge_kind": "conditional_fallthrough",
                        "branch_event": last,
                    }
                )
        elif index + 1 < len(blocks):
            edges[blocks[index + 1]["bb"]].append(
                {
                    "predecessor": block["bb"],
                    "edge_kind": "implicit_fallthrough",
                    "branch_event": None,
                }
            )
    return edges


def tail_context(block: dict, tail_events: int) -> list[dict]:
    interesting = [
        event
        for event in block["events"]
        if event["opcode"].startswith(INTERESTING_BRANCH_PREFIXES)
    ]
    return interesting[-tail_events:]


def summarize(path: pathlib.Path, tail_events: int) -> dict:
    blocks = parse_blocks(path)
    preds = branch_edges(blocks)
    block_by_name = {block["bb"]: block for block in blocks}
    store_blocks: list[dict] = []
    class_counts: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)

    for block in blocks:
        counts = count_ops(block["events"])
        klass = classify_block(counts)
        if klass == "non_store":
            continue
        buffer_stores = sum(count for op, count in counts.items() if op.startswith("buffer_store"))
        lds_stores = sum(count for op, count in counts.items() if op.startswith(("ds_store", "ds_write")))
        lds_loads = sum(count for op, count in counts.items() if op.startswith(("ds_load", "ds_read")))
        class_counts[klass]["blocks"] += 1
        class_counts[klass]["buffer_store_ops"] += buffer_stores
        class_counts[klass]["lds_store_ops"] += lds_stores
        class_counts[klass]["lds_load_ops"] += lds_loads
        incoming = []
        for edge in preds.get(block["bb"], []):
            pred_block = block_by_name[edge["predecessor"]]
            incoming.append(
                {
                    "predecessor": edge["predecessor"],
                    "edge_kind": edge["edge_kind"],
                    "branch_event": edge["branch_event"],
                    "predecessor_tail": tail_context(pred_block, tail_events),
                }
            )
        store_blocks.append(
            {
                "bb": block["bb"],
                "class": klass,
                "start_line": block["start_line"],
                "end_line": block["end_line"],
                "buffer_store_ops": buffer_stores,
                "lds_store_ops": lds_stores,
                "lds_load_ops": lds_loads,
                "incoming_edges": incoming,
            }
        )

    examples = {}
    for block in store_blocks:
        examples.setdefault(block["class"], block)

    return {
        "schema": "radv-store-branch-paths-v1",
        "isa": str(path),
        "tail_events": tail_events,
        "classes": {key: dict(value) for key, value in sorted(class_counts.items())},
        "store_blocks": store_blocks,
        "class_examples": examples,
    }


def fmt_event(event: dict | None) -> str:
    if event is None:
        return ""
    return f"L{event['line']}: {event['text']}"


def write_markdown(path: pathlib.Path, payload: dict) -> None:
    lines = [
        "# RADV Store Branch Paths",
        "",
        f"- ISA: `{payload['isa']}`",
        f"- predecessor tail events: `{payload['tail_events']}`",
        "",
        "## Class Summary",
        "",
        "| Class | Blocks | Buffer Stores | LDS Stores | LDS Loads |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for klass, counts in payload["classes"].items():
        lines.append(
            f"| `{klass}` | {counts.get('blocks', 0)} | {counts.get('buffer_store_ops', 0)} | "
            f"{counts.get('lds_store_ops', 0)} | {counts.get('lds_load_ops', 0)} |"
        )

    lines += [
        "",
        "## Representative Branch Context",
        "",
    ]
    for klass, block in payload["class_examples"].items():
        lines += [
            f"### `{klass}`",
            "",
            f"- block: `{block['bb']}` lines `{block['start_line']}-{block['end_line']}`",
            f"- stores: buffer `{block['buffer_store_ops']}`, LDS stores `{block['lds_store_ops']}`, LDS loads `{block['lds_load_ops']}`",
            "",
        ]
        for edge in block["incoming_edges"]:
            lines += [
                f"Incoming from `{edge['predecessor']}` via `{edge['edge_kind']}`",
                "",
            ]
            if edge["branch_event"] is not None:
                lines.append(f"- branch: `{fmt_event(edge['branch_event'])}`")
            if edge["predecessor_tail"]:
                lines += ["", "```text"]
                lines += [fmt_event(event) for event in edge["predecessor_tail"]]
                lines += ["```", ""]

    lines += [
        "## Store Blocks",
        "",
        "| BB | Line | Class | Incoming Edges |",
        "| --- | ---: | --- | --- |",
    ]
    for block in payload["store_blocks"]:
        incoming = ", ".join(
            f"{edge['predecessor']}:{edge['edge_kind']}" for edge in block["incoming_edges"]
        )
        lines.append(f"| `{block['bb']}` | {block['start_line']} | `{block['class']}` | `{incoming}` |")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isa", required=True, type=pathlib.Path)
    parser.add_argument("--json-out", required=True, type=pathlib.Path)
    parser.add_argument("--md-out", required=True, type=pathlib.Path)
    parser.add_argument("--tail-events", type=int, default=12)
    args = parser.parse_args()

    payload = summarize(args.isa, tail_events=args.tail_events)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(args.md_out, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
