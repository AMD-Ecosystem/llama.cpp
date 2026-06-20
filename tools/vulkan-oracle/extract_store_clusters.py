#!/usr/bin/env python3
"""Extract store-centered AMDGCN snippets from RADV or HIP ISA text.

This complements classify_radv_store_paths.py. RADV shader dumps have basic
block labels that make path classification possible; HIP objdump output often
does not. This helper uses store bursts and configurable context instead, so
RADV and HIP store/writeback shapes can be compared side by side.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re


INSTRUCTION_RE = re.compile(r"^\s*(?:(?:[0-9a-fA-F]+|BB[0-9]+):\s*)?([A-Za-z0-9_.]+)\b")
STORE_PREFIXES = ("buffer_store", "global_store", "flat_store", "ds_store", "ds_write")
LOAD_PREFIXES = ("buffer_load", "global_load", "flat_load", "ds_load", "ds_read")
HOT_PREFIXES = ("v_wmma", "v_dot")


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
        rows.append({"line": line_no, "text": raw.rstrip(), "opcode": opcode})
    return rows


def is_store(opcode: str | None) -> bool:
    return opcode is not None and opcode.startswith(STORE_PREFIXES)


def summarize_events(rows: list[dict]) -> collections.Counter:
    counts: collections.Counter = collections.Counter()
    for row in rows:
        opcode = row["opcode"]
        if opcode is None:
            continue
        counts[opcode] += 1
        if opcode.startswith(STORE_PREFIXES):
            counts["store_ops"] += 1
        if opcode.startswith("buffer_store"):
            counts["buffer_store_ops"] += 1
        if opcode.startswith(("ds_store", "ds_write")):
            counts["lds_store_ops"] += 1
        if opcode.startswith(LOAD_PREFIXES):
            counts["load_ops"] += 1
        if opcode.startswith(("ds_load", "ds_read")):
            counts["lds_load_ops"] += 1
        if opcode.startswith(("buffer_load", "global_load", "flat_load")):
            counts["vmem_load_ops"] += 1
        if opcode.startswith(HOT_PREFIXES):
            counts["hot_ops"] += 1
        if opcode.startswith("s_waitcnt"):
            counts["waitcnt_ops"] += 1
    return counts


def store_clusters(rows: list[dict], max_gap: int) -> list[dict]:
    store_indices = [index for index, row in enumerate(rows) if is_store(row["opcode"])]
    clusters: list[list[int]] = []
    for index in store_indices:
        if not clusters or index - clusters[-1][-1] > max_gap:
            clusters.append([index])
        else:
            clusters[-1].append(index)

    result = []
    for cluster_id, indices in enumerate(clusters, 1):
        start = indices[0]
        end = indices[-1]
        cluster_rows = rows[start : end + 1]
        counts = summarize_events(cluster_rows)
        result.append(
            {
                "id": cluster_id,
                "start_line": rows[start]["line"],
                "end_line": rows[end]["line"],
                "store_lines": [rows[index]["line"] for index in indices],
                "counts": dict(sorted(counts.items())),
                "first_store": rows[start]["text"],
            }
        )
    return result


def with_context(rows: list[dict], cluster: dict, context: int) -> list[dict]:
    start_index = max(0, cluster["store_lines"][0] - 1 - context)
    end_index = min(len(rows), cluster["store_lines"][-1] - 1 + context)
    return rows[start_index : end_index + 1]


def summarize(path: pathlib.Path, context: int, max_gap: int) -> dict:
    rows = parse_lines(path)
    clusters = store_clusters(rows, max_gap=max_gap)
    for cluster in clusters:
        snippet_rows = with_context(rows, cluster, context)
        cluster["snippet_start_line"] = snippet_rows[0]["line"] if snippet_rows else cluster["start_line"]
        cluster["snippet_end_line"] = snippet_rows[-1]["line"] if snippet_rows else cluster["end_line"]
        cluster["snippet_counts"] = dict(sorted(summarize_events(snippet_rows).items()))
        cluster["snippet_lines"] = snippet_rows
    return {
        "schema": "amdgcn-store-clusters-v1",
        "isa": str(path),
        "context": context,
        "max_gap": max_gap,
        "totals": dict(sorted(summarize_events(rows).items())),
        "clusters": clusters,
    }


def write_outputs(payload: dict, json_out: pathlib.Path, md_out: pathlib.Path, snippets_dir: pathlib.Path | None) -> None:
    json_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# AMDGCN Store Clusters",
        "",
        f"- ISA: `{payload['isa']}`",
        f"- context: `{payload['context']}`",
        f"- max gap: `{payload['max_gap']}`",
        "",
        "## Totals",
        "",
        "| Metric | Count |",
        "| --- | ---: |",
    ]
    for key in ("store_ops", "buffer_store_ops", "lds_store_ops", "load_ops", "lds_load_ops", "vmem_load_ops", "hot_ops", "waitcnt_ops"):
        lines.append(f"| `{key}` | {payload['totals'].get(key, 0)} |")

    lines += [
        "",
        "## Clusters",
        "",
        "| ID | Lines | Stores | Buffer Stores | LDS Stores | Loads In Window | Hot Ops In Window | First Store |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for cluster in payload["clusters"]:
        counts = cluster["counts"]
        snippet_counts = cluster["snippet_counts"]
        first_store = cluster["first_store"].replace("|", "\\|")
        lines.append(
            f"| {cluster['id']} | {cluster['start_line']}-{cluster['end_line']} | "
            f"{counts.get('store_ops', 0)} | {counts.get('buffer_store_ops', 0)} | "
            f"{counts.get('lds_store_ops', 0)} | {snippet_counts.get('load_ops', 0)} | "
            f"{snippet_counts.get('hot_ops', 0)} | `{first_store}` |"
        )

    md_out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if snippets_dir is None:
        return
    snippets_dir.mkdir(parents=True, exist_ok=True)
    index_lines = [
        "# AMDGCN Store Cluster Snippets",
        "",
        f"- ISA: `{payload['isa']}`",
        "",
        "| ID | Snippet Lines | Stores | File |",
        "| ---: | --- | ---: | --- |",
    ]
    for cluster in payload["clusters"]:
        name = f"cluster-{cluster['id']:03d}-line{cluster['start_line']}.amdgcn.txt"
        path = snippets_dir / name
        path.write_text(
            "\n".join(row["text"] for row in cluster["snippet_lines"]) + "\n",
            encoding="utf-8",
        )
        index_lines.append(
            f"| {cluster['id']} | {cluster['snippet_start_line']}-{cluster['snippet_end_line']} | "
            f"{cluster['counts'].get('store_ops', 0)} | `{name}` |"
        )
    (snippets_dir / "index.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isa", required=True, type=pathlib.Path)
    parser.add_argument("--json-out", required=True, type=pathlib.Path)
    parser.add_argument("--md-out", required=True, type=pathlib.Path)
    parser.add_argument("--snippets-dir", type=pathlib.Path)
    parser.add_argument("--context", type=int, default=8)
    parser.add_argument("--max-gap", type=int, default=4)
    args = parser.parse_args()

    payload = summarize(args.isa, context=args.context, max_gap=args.max_gap)
    write_outputs(payload, args.json_out, args.md_out, args.snippets_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
