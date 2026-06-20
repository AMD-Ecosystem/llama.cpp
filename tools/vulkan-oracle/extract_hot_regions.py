#!/usr/bin/env python3
"""Extract math-centered AMDGCN regions from RADV or HIP ISA text.

Store clusters answer where results are written. For Vulkan-oracle kernel
porting the next question is often whether the math loop around `v_wmma` or
`v_dot` has the same load/wait shape. This helper groups nearby hot ops and
captures the interesting instructions around each group without depending on
RADV basic-block labels or HIP symbol metadata.
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
INTERESTING_PREFIXES = HOT_PREFIXES + LOAD_PREFIXES + STORE_PREFIXES + (
    "s_waitcnt",
    "s_waitcnt_depctr",
    "s_barrier",
)


def strip_comment(line: str) -> str:
    for sep in ("//", ";"):
        if sep in line:
            line = line.split(sep, 1)[0]
    return line.rstrip()


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


def is_hot(opcode: str | None) -> bool:
    return opcode is not None and opcode.startswith(HOT_PREFIXES)


def is_load(opcode: str | None) -> bool:
    return opcode is not None and opcode.startswith(LOAD_PREFIXES)


def is_store(opcode: str | None) -> bool:
    return opcode is not None and opcode.startswith(STORE_PREFIXES)


def parse_wait_value(pattern: re.Pattern[str], text: str) -> int | None:
    match = pattern.search(text)
    return int(match.group("value")) if match else None


def summarize_rows(rows: list[dict]) -> collections.Counter:
    counts: collections.Counter = collections.Counter()
    for row in rows:
        opcode = row["opcode"]
        counts[opcode] += 1
        if opcode.startswith("v_wmma"):
            counts["wmma_ops"] += 1
        if opcode.startswith("v_mfma"):
            counts["mfma_ops"] += 1
        if opcode.startswith("v_dot"):
            counts["dot_ops"] += 1
        if is_load(opcode):
            counts["load_ops"] += 1
        if opcode.startswith(("ds_load", "ds_read")):
            counts["lds_load_ops"] += 1
        if opcode.startswith(("buffer_load", "global_load", "flat_load")):
            counts["vmem_load_ops"] += 1
        if is_store(opcode):
            counts["store_ops"] += 1
        if opcode.startswith(("ds_store", "ds_write")):
            counts["lds_store_ops"] += 1
        if opcode.startswith(("buffer_store", "global_store", "flat_store")):
            counts["vmem_store_ops"] += 1
        if opcode.startswith("s_waitcnt"):
            counts["waitcnt_ops"] += 1
        if opcode == "s_waitcnt":
            counts["waitcnt_strict_ops"] += 1
        if opcode.startswith("s_waitcnt_depctr"):
            counts["waitcnt_depctr_ops"] += 1
        if opcode.startswith("s_barrier"):
            counts["barrier_ops"] += 1
    return counts


def group_hot_regions(rows: list[dict], max_hot_gap: int) -> list[list[int]]:
    hot_indices = [index for index, row in enumerate(rows) if is_hot(row["opcode"])]
    groups: list[list[int]] = []
    for index in hot_indices:
        if not groups or index - groups[-1][-1] > max_hot_gap:
            groups.append([index])
        else:
            groups[-1].append(index)
    return groups


def wait_summary(rows: list[dict]) -> dict:
    waits = [row for row in rows if row["opcode"] == "s_waitcnt"]
    return {
        "waitcnt_ops": len(waits),
        "last_waitcnt_line": waits[-1]["line"] if waits else None,
        "last_vmcnt": parse_wait_value(VMCNT_RE, waits[-1]["text"]) if waits else None,
        "last_lgkmcnt": parse_wait_value(LGKMCNT_RE, waits[-1]["text"]) if waits else None,
    }


def summarize_region(rows: list[dict], group: list[int], pre_events: int, post_events: int, region_id: int) -> dict:
    first = group[0]
    last = group[-1]
    pre_start = max(0, first - pre_events)
    post_end = min(len(rows), last + post_events + 1)
    pre_rows = rows[pre_start:first]
    hot_rows = [rows[index] for index in group]
    hot_span_rows = rows[first : last + 1]
    post_rows = rows[last + 1 : post_end]
    window_rows = rows[pre_start:post_end]
    counts = summarize_rows(window_rows)
    hot_counts = summarize_rows(hot_rows)
    pre_counts = summarize_rows(pre_rows)
    post_counts = summarize_rows(post_rows)

    return {
        "id": region_id,
        "start_line": rows[first]["line"],
        "end_line": rows[last]["line"],
        "snippet_start_line": window_rows[0]["line"] if window_rows else rows[first]["line"],
        "snippet_end_line": window_rows[-1]["line"] if window_rows else rows[last]["line"],
        "event_count": len(window_rows),
        "hot_event_count": len(hot_rows),
        "hot_span_event_count": len(hot_span_rows),
        "hot_opcode_counts": dict(sorted(collections.Counter(row["opcode"] for row in hot_rows).items())),
        "first_hot": hot_rows[0]["text"],
        "last_hot": hot_rows[-1]["text"],
        "pre_counts": dict(sorted(pre_counts.items())),
        "hot_counts": dict(sorted(hot_counts.items())),
        "post_counts": dict(sorted(post_counts.items())),
        "window_counts": dict(sorted(counts.items())),
        "pre_wait": wait_summary(pre_rows),
        "post_wait": wait_summary(post_rows),
        "snippet_lines": window_rows,
    }


def summarize_file(label: str, path: pathlib.Path, pre_events: int, post_events: int, max_hot_gap: int) -> dict:
    rows = parse_lines(path)
    groups = group_hot_regions(rows, max_hot_gap=max_hot_gap)
    regions = [
        summarize_region(rows, group, pre_events=pre_events, post_events=post_events, region_id=index)
        for index, group in enumerate(groups, 1)
    ]
    totals = summarize_rows(rows)
    return {
        "label": label,
        "isa": str(path),
        "pre_events": pre_events,
        "post_events": post_events,
        "max_hot_gap": max_hot_gap,
        "totals": dict(sorted(totals.items())),
        "region_count": len(regions),
        "regions": regions,
    }


def summarize_motifs(regions: list[dict]) -> list[dict]:
    motifs: dict[str, dict] = {}
    for region in regions:
        counts = region["window_counts"]
        key = json.dumps(
            {
                "hot_opcode_counts": region["hot_opcode_counts"],
                "load_ops": counts.get("load_ops", 0),
                "lds_load_ops": counts.get("lds_load_ops", 0),
                "vmem_load_ops": counts.get("vmem_load_ops", 0),
                "store_ops": counts.get("store_ops", 0),
                "lds_store_ops": counts.get("lds_store_ops", 0),
                "vmem_store_ops": counts.get("vmem_store_ops", 0),
                "waitcnt_ops": counts.get("waitcnt_ops", 0),
                "barrier_ops": counts.get("barrier_ops", 0),
                "pre_last_lgkmcnt": region["pre_wait"].get("last_lgkmcnt"),
                "pre_last_vmcnt": region["pre_wait"].get("last_vmcnt"),
            },
            sort_keys=True,
        )
        motif = motifs.setdefault(
            key,
            {
                "count": 0,
                "example_region_ids": [],
                "hot_opcode_counts": region["hot_opcode_counts"],
                "window_counts": {
                    metric: counts.get(metric, 0)
                    for metric in (
                        "load_ops",
                        "lds_load_ops",
                        "vmem_load_ops",
                        "store_ops",
                        "lds_store_ops",
                        "vmem_store_ops",
                        "waitcnt_ops",
                        "barrier_ops",
                    )
                },
                "pre_wait": region["pre_wait"],
            },
        )
        motif["count"] += 1
        if len(motif["example_region_ids"]) < 8:
            motif["example_region_ids"].append(region["id"])
    return sorted(motifs.values(), key=lambda item: (-item["count"], item["example_region_ids"][0]))


def strip_snippets(payload: dict) -> dict:
    result = json.loads(json.dumps(payload))
    for file_row in result["files"]:
        for region in file_row["regions"]:
            region.pop("snippet_lines", None)
    return result


def write_markdown(payload: dict, path: pathlib.Path) -> None:
    lines = [
        "# AMDGCN Hot Regions",
        "",
        "| Label | Regions | WMMA | MFMA | Dot | Loads | LDS Loads | VMEM Loads | Stores | Waits | Barriers |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for file_row in payload["files"]:
        totals = file_row["totals"]
        lines.append(
            f"| `{file_row['label']}` | {file_row['region_count']} | "
            f"{totals.get('wmma_ops', 0)} | {totals.get('mfma_ops', 0)} | {totals.get('dot_ops', 0)} | "
            f"{totals.get('load_ops', 0)} | {totals.get('lds_load_ops', 0)} | {totals.get('vmem_load_ops', 0)} | "
            f"{totals.get('store_ops', 0)} | {totals.get('waitcnt_ops', 0)} | {totals.get('barrier_ops', 0)} |"
        )
    lines.append("")

    for file_row in payload["files"]:
        lines += [
            f"## {file_row['label']}",
            "",
            f"- ISA: `{file_row['isa']}`",
            "",
            "### Motifs",
            "",
            "| Count | Examples | Hot Ops | Loads | Stores | Waits | Barriers | Pre Last Wait |",
            "| ---: | --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
        for motif in file_row.get("motifs", [])[:16]:
            hot = ", ".join(f"{key}={value}" for key, value in motif["hot_opcode_counts"].items())
            counts = motif["window_counts"]
            pre_wait = motif["pre_wait"]
            wait_text = f"vmcnt={pre_wait.get('last_vmcnt')}, lgkmcnt={pre_wait.get('last_lgkmcnt')}"
            lines.append(
                f"| {motif['count']} | `{motif['example_region_ids']}` | `{hot}` | "
                f"{counts.get('load_ops', 0)} | {counts.get('store_ops', 0)} | "
                f"{counts.get('waitcnt_ops', 0)} | {counts.get('barrier_ops', 0)} | `{wait_text}` |"
            )

        lines += [
            "",
            "### Regions",
            "",
            "| ID | Lines | Hot Ops | Loads | LDS Loads | VMEM Loads | Stores | Waits | Barriers | First Hot |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
        for region in file_row["regions"][:64]:
            counts = region["window_counts"]
            first_hot = region["first_hot"].replace("|", "\\|")
            lines.append(
                f"| {region['id']} | {region['start_line']}-{region['end_line']} | "
                f"{region['hot_event_count']} | {counts.get('load_ops', 0)} | "
                f"{counts.get('lds_load_ops', 0)} | {counts.get('vmem_load_ops', 0)} | "
                f"{counts.get('store_ops', 0)} | {counts.get('waitcnt_ops', 0)} | "
                f"{counts.get('barrier_ops', 0)} | `{first_hot}` |"
            )
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_snippets(payload: dict, snippets_dir: pathlib.Path) -> None:
    snippets_dir.mkdir(parents=True, exist_ok=True)
    index_lines = [
        "# AMDGCN Hot Region Snippets",
        "",
        "| Label | Region | Snippet Lines | File |",
        "| --- | ---: | --- | --- |",
    ]
    for file_row in payload["files"]:
        safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", file_row["label"]).strip("_")
        for region in file_row["regions"]:
            name = f"{safe_label}-region-{region['id']:03d}-line{region['start_line']}.amdgcn.txt"
            path = snippets_dir / name
            path.write_text(
                "\n".join(row["text"] for row in region["snippet_lines"]) + "\n",
                encoding="utf-8",
            )
            index_lines.append(
                f"| `{file_row['label']}` | {region['id']} | "
                f"{region['snippet_start_line']}-{region['snippet_end_line']} | `{name}` |"
            )
    (snippets_dir / "index.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")


def parse_input(value: str) -> tuple[str, pathlib.Path]:
    if "=" in value:
        label, path = value.split("=", 1)
        return label, pathlib.Path(path)
    path = pathlib.Path(value)
    return path.stem, path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, help="LABEL=path or path; repeatable")
    parser.add_argument("--json-out", required=True, type=pathlib.Path)
    parser.add_argument("--md-out", required=True, type=pathlib.Path)
    parser.add_argument("--snippets-dir", type=pathlib.Path)
    parser.add_argument("--pre-events", type=int, default=48)
    parser.add_argument("--post-events", type=int, default=24)
    parser.add_argument("--max-hot-gap", type=int, default=12)
    args = parser.parse_args()

    files = [
        summarize_file(label, path, pre_events=args.pre_events, post_events=args.post_events, max_hot_gap=args.max_hot_gap)
        for label, path in (parse_input(item) for item in args.input)
    ]
    for file_row in files:
        file_row["motifs"] = summarize_motifs(file_row["regions"])

    payload = {
        "schema": "amdgcn-hot-regions-v1",
        "pre_events": args.pre_events,
        "post_events": args.post_events,
        "max_hot_gap": args.max_hot_gap,
        "files": files,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(strip_snippets(payload), indent=2) + "\n", encoding="utf-8")
    write_markdown(payload, args.md_out)
    if args.snippets_dir:
        write_snippets(payload, args.snippets_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
