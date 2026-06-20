#!/usr/bin/env python3
"""Compare AMDGCN hot-op issue windows across RADV and HIP ISA text.

The Vulkan-oracle workflow often needs a tighter answer than whole-symbol
opcode counts: did a HIP candidate preserve the load/wait window leading into
the first `v_wmma`/`v_dot` region, or did the compiler split/sink the schedule?
This helper extracts that schedule shape from RADV dumps or symbol-scoped HIP
objdump text without relying on backend-specific basic block labels.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re


INSTRUCTION_RE = re.compile(r"^\s*(?:(?:[0-9a-fA-F]+|BB[0-9]+):\s*)?([A-Za-z0-9_.]+)\b")
VMCNT_RE = re.compile(r"\bvmcnt\((?P<value>[0-9]+)\)")
LGKMCNT_RE = re.compile(r"\blgkmcnt\((?P<value>[0-9]+)\)")
HOT_PREFIXES = ("v_wmma", "v_mfma", "v_dot")
LOAD_PREFIXES = ("buffer_load", "global_load", "flat_load", "ds_load", "ds_read")
STORE_PREFIXES = ("buffer_store", "global_store", "flat_store", "ds_store", "ds_write")
WAIT_PREFIXES = ("s_waitcnt", "s_waitcnt_depctr")
CONTROL_PREFIXES = ("s_barrier", "s_branch", "s_cbranch")
INTERESTING_PREFIXES = HOT_PREFIXES + LOAD_PREFIXES + STORE_PREFIXES + WAIT_PREFIXES + CONTROL_PREFIXES


def strip_comment(line: str) -> str:
    for sep in ("//", ";"):
        if sep in line:
            line = line.split(sep, 1)[0]
    return line.rstrip()


def parse_input(value: str) -> tuple[str, pathlib.Path]:
    if "=" not in value:
        path = pathlib.Path(value)
        return path.stem, path
    label, path = value.split("=", 1)
    if not label:
        raise argparse.ArgumentTypeError(f"empty label in input: {value}")
    return label, pathlib.Path(path)


def parse_lines(path: pathlib.Path) -> list[dict]:
    rows: list[dict] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        text = strip_comment(raw)
        match = INSTRUCTION_RE.match(text)
        opcode = match.group(1) if match else None
        if opcode is None or not opcode.startswith(INTERESTING_PREFIXES):
            continue
        rows.append({"line": line_no, "text": raw.rstrip(), "opcode": opcode})
    return rows


def starts_with(opcode: str | None, prefixes: tuple[str, ...]) -> bool:
    return opcode is not None and opcode.startswith(prefixes)


def is_hot(row: dict) -> bool:
    return starts_with(row["opcode"], HOT_PREFIXES)


def is_load(row: dict) -> bool:
    return starts_with(row["opcode"], LOAD_PREFIXES)


def is_wait(row: dict) -> bool:
    return starts_with(row["opcode"], WAIT_PREFIXES)


def is_boundary(row: dict) -> bool:
    return is_wait(row) or starts_with(row["opcode"], CONTROL_PREFIXES)


def wait_values(text: str) -> dict:
    vmcnt = VMCNT_RE.search(text)
    lgkmcnt = LGKMCNT_RE.search(text)
    return {
        "vmcnt": int(vmcnt.group("value")) if vmcnt else None,
        "lgkmcnt": int(lgkmcnt.group("value")) if lgkmcnt else None,
    }


def summarize_rows(rows: list[dict]) -> dict:
    counts: collections.Counter = collections.Counter()
    for row in rows:
        opcode = row["opcode"]
        counts[opcode] += 1
        if starts_with(opcode, HOT_PREFIXES):
            counts["hot_ops"] += 1
        if opcode.startswith("v_wmma"):
            counts["wmma_ops"] += 1
        if opcode.startswith("v_mfma"):
            counts["mfma_ops"] += 1
        if opcode.startswith("v_dot"):
            counts["dot_ops"] += 1
        if starts_with(opcode, LOAD_PREFIXES):
            counts["load_ops"] += 1
        if opcode.startswith(("ds_load", "ds_read")):
            counts["lds_load_ops"] += 1
        if opcode.startswith(("buffer_load", "global_load", "flat_load")):
            counts["vmem_load_ops"] += 1
        if starts_with(opcode, STORE_PREFIXES):
            counts["store_ops"] += 1
        if opcode.startswith(("ds_store", "ds_write")):
            counts["lds_store_ops"] += 1
        if opcode.startswith(("buffer_store", "global_store", "flat_store")):
            counts["vmem_store_ops"] += 1
        if starts_with(opcode, WAIT_PREFIXES):
            counts["wait_ops"] += 1
        if opcode == "s_waitcnt":
            counts["waitcnt_ops"] += 1
        if opcode.startswith("s_waitcnt_depctr"):
            counts["waitcnt_depctr_ops"] += 1
        if opcode == "s_barrier":
            counts["barrier_ops"] += 1
        if opcode.startswith(("s_branch", "s_cbranch")):
            counts["branch_ops"] += 1
    return dict(sorted(counts.items()))


def group_hot_regions(rows: list[dict], max_hot_gap: int) -> list[list[int]]:
    hot_indices = [idx for idx, row in enumerate(rows) if is_hot(row)]
    groups: list[list[int]] = []
    for idx in hot_indices:
        if not groups or idx - groups[-1][-1] > max_hot_gap:
            groups.append([idx])
        else:
            groups[-1].append(idx)
    return groups


def count_loads(rows: list[dict]) -> dict:
    counts = summarize_rows([row for row in rows if is_load(row)])
    return {
        "load_ops": counts.get("load_ops", 0),
        "lds_load_ops": counts.get("lds_load_ops", 0),
        "vmem_load_ops": counts.get("vmem_load_ops", 0),
    }


def last_wait_before(rows: list[dict], end_idx: int) -> int | None:
    for idx in range(end_idx - 1, -1, -1):
        if rows[idx]["opcode"] == "s_waitcnt":
            return idx
    return None


def previous_boundary_before(rows: list[dict], end_idx: int) -> int | None:
    for idx in range(end_idx - 1, -1, -1):
        if is_boundary(rows[idx]):
            return idx
    return None


def summarize_file(label: str, path: pathlib.Path, pre_events: int, post_events: int, max_hot_gap: int) -> dict:
    rows = parse_lines(path)
    groups = group_hot_regions(rows, max_hot_gap=max_hot_gap)
    regions = []
    for region_id, group in enumerate(groups, 1):
        first = group[0]
        last = group[-1]
        pre_start = max(0, first - pre_events)
        post_end = min(len(rows), last + post_events + 1)
        pre_rows = rows[pre_start:first]
        hot_rows = [rows[idx] for idx in group]
        window_rows = rows[pre_start:post_end]
        final_wait_idx = last_wait_before(rows, first)
        prev_boundary_idx = previous_boundary_before(rows, final_wait_idx) if final_wait_idx is not None else None
        immediate_start = (prev_boundary_idx + 1) if prev_boundary_idx is not None else pre_start
        immediate_rows = rows[immediate_start:final_wait_idx] if final_wait_idx is not None else []
        final_wait = rows[final_wait_idx] if final_wait_idx is not None else None

        regions.append(
            {
                "id": region_id,
                "start_line": rows[first]["line"],
                "end_line": rows[last]["line"],
                "hot_event_count": len(hot_rows),
                "hot_opcode_counts": dict(sorted(collections.Counter(row["opcode"] for row in hot_rows).items())),
                "first_hot": hot_rows[0]["text"],
                "last_hot": hot_rows[-1]["text"],
                "pre_event_count": len(pre_rows),
                "pre_loads": count_loads(pre_rows),
                "immediate_loads_before_final_wait": count_loads(immediate_rows),
                "final_wait": {
                    "line": final_wait["line"] if final_wait else None,
                    "text": final_wait["text"] if final_wait else None,
                    **(wait_values(final_wait["text"]) if final_wait else {"vmcnt": None, "lgkmcnt": None}),
                },
                "window_counts": summarize_rows(window_rows),
            }
        )

    return {
        "label": label,
        "isa": str(path),
        "totals": summarize_rows(rows),
        "region_count": len(regions),
        "regions": regions,
    }


def write_markdown(payload: dict, md_out: pathlib.Path) -> None:
    md_out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# AMDGCN Issue Window Compare",
        "",
        "| label | regions | hot ops | WMMA | dot | loads | LDS loads | stores | waits | barriers | first hot ops | pre loads | immediate loads | final lgkmcnt | final vmcnt |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in payload["files"]:
        totals = item["totals"]
        first = item["regions"][0] if item["regions"] else {}
        pre_loads = first.get("pre_loads", {})
        immediate = first.get("immediate_loads_before_final_wait", {})
        final_wait = first.get("final_wait", {})
        lines.append(
            "| "
            + " | ".join(
                str(value)
                for value in (
                    item["label"],
                    item["region_count"],
                    totals.get("hot_ops", 0),
                    totals.get("wmma_ops", 0),
                    totals.get("dot_ops", 0),
                    totals.get("load_ops", 0),
                    totals.get("lds_load_ops", 0),
                    totals.get("store_ops", 0),
                    totals.get("wait_ops", 0),
                    totals.get("barrier_ops", 0),
                    first.get("hot_event_count", 0),
                    pre_loads.get("load_ops", 0),
                    immediate.get("load_ops", 0),
                    final_wait.get("lgkmcnt"),
                    final_wait.get("vmcnt"),
                )
            )
            + " |"
        )

    lines += ["", "## First Region Detail", ""]
    for item in payload["files"]:
        lines += [f"### {item['label']}", "", f"- ISA: `{item['isa']}`"]
        if not item["regions"]:
            lines += ["- no hot region found", ""]
            continue
        first = item["regions"][0]
        lines += [
            f"- lines: `{first['start_line']}-{first['end_line']}`",
            f"- hot opcode counts: `{json.dumps(first['hot_opcode_counts'], sort_keys=True)}`",
            f"- pre loads: `{json.dumps(first['pre_loads'], sort_keys=True)}`",
            f"- immediate loads before final wait: `{json.dumps(first['immediate_loads_before_final_wait'], sort_keys=True)}`",
            f"- final wait: `{first['final_wait']['text']}`",
            f"- first hot: `{first['first_hot']}`",
            "",
        ]

    md_out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, type=parse_input, metavar="LABEL=ISA")
    parser.add_argument("--json-out", required=True, type=pathlib.Path)
    parser.add_argument("--md-out", required=True, type=pathlib.Path)
    parser.add_argument("--pre-events", type=int, default=96)
    parser.add_argument("--post-events", type=int, default=32)
    parser.add_argument("--max-hot-gap", type=int, default=12)
    args = parser.parse_args()

    payload = {
        "schema": "amdgcn-issue-window-compare-v1",
        "pre_events": args.pre_events,
        "post_events": args.post_events,
        "max_hot_gap": args.max_hot_gap,
        "files": [
            summarize_file(label, path, pre_events=args.pre_events, post_events=args.post_events, max_hot_gap=args.max_hot_gap)
            for label, path in args.input
        ],
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_markdown(payload, args.md_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
