#!/usr/bin/env python3
"""Rank HRX/Vulkan basket artifacts by measured Vulkan buckets and HRX routes."""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
from typing import Any


PERF_RE = re.compile(
    r"^(?P<label>.*?):\s+"
    r"(?P<count>\d+)\s+x\s+(?P<avg>[0-9.]+)\s+us\s+=\s+"
    r"(?P<total>[0-9.]+)\s+us"
)


def load_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def classify_label(label: str) -> str:
    parts = label.split()
    if not parts:
        return "unknown"
    op = parts[0]
    dtype = parts[1] if len(parts) > 1 else ""
    if op.startswith("MUL_MAT_ID") or op == "MUL_MAT_ID":
        return f"moe_id/{dtype}" if dtype else "moe_id"
    if op == "MUL_MAT":
        return f"dense/{dtype}" if dtype else "dense"
    if op == "MUL_MAT_VEC":
        return f"decode/{dtype}" if dtype else "decode"
    if op == "FLASH_ATTN_EXT":
        return "attention/flash"
    if op == "GET_ROWS":
        return "movement/get_rows"
    return op.lower()


def parse_perf_blocks(path: pathlib.Path) -> list[list[dict[str, Any]]]:
    blocks: list[list[dict[str, Any]]] = []
    if not path.exists():
        return blocks
    current: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("Vulkan Timings:"):
            if current:
                blocks.append(current)
                current = []
            continue
        match = PERF_RE.match(line)
        if not match:
            continue
        label = match.group("label")
        current.append(
            {
                "label": label,
                "family": classify_label(label),
                "count": int(match.group("count")),
                "avg_us": float(match.group("avg")),
                "total_us": float(match.group("total")),
            }
        )
    if current:
        blocks.append(current)
    return blocks


def flatten_blocks(blocks: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    return [row for block in blocks for row in block]


def summarize_perf(rows: list[dict[str, Any]], top: int) -> dict[str, Any]:
    by_label: dict[str, dict[str, Any]] = {}
    by_family: dict[str, dict[str, Any]] = {}
    for row in rows:
        label = row["label"]
        family = row["family"]
        label_acc = by_label.setdefault(
            label,
            {
                "label": label,
                "family": family,
                "samples": 0,
                "dispatch_count_sum": 0,
                "total_us_sum": 0.0,
                "avg_us_weighted": 0.0,
            },
        )
        label_acc["samples"] += 1
        label_acc["dispatch_count_sum"] += row["count"]
        label_acc["total_us_sum"] += row["total_us"]

        fam_acc = by_family.setdefault(
            family,
            {
                "family": family,
                "samples": 0,
                "dispatch_count_sum": 0,
                "total_us_sum": 0.0,
            },
        )
        fam_acc["samples"] += 1
        fam_acc["dispatch_count_sum"] += row["count"]
        fam_acc["total_us_sum"] += row["total_us"]

    for acc in by_label.values():
        count = acc["dispatch_count_sum"]
        acc["avg_us_weighted"] = acc["total_us_sum"] / count if count else 0.0

    return {
        "top_labels": sorted(by_label.values(), key=lambda row: row["total_us_sum"], reverse=True)[:top],
        "top_families": sorted(by_family.values(), key=lambda row: row["total_us_sum"], reverse=True)[:top],
    }


def summarize_perf_blocks(blocks: list[list[dict[str, Any]]], top: int) -> dict[str, Any]:
    all_rows = flatten_blocks(blocks)
    steady_blocks = blocks[1:] if len(blocks) > 1 else blocks
    steady_rows = flatten_blocks(steady_blocks)
    cold_rows = blocks[0] if blocks else []
    return {
        "blocks": len(blocks),
        "rows_all": len(all_rows),
        "rows_steady": len(steady_rows),
        "all": summarize_perf(all_rows, top),
        "steady": summarize_perf(steady_rows, top),
        "cold": summarize_perf(cold_rows, top),
    }


def load_records(root: pathlib.Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    records_path = root / "records.json"
    if not records_path.exists():
        return {}
    records = load_json(records_path)
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    for record in records:
        key = (record.get("backend"), record.get("model"), record.get("case"))
        out[key] = record
    return out


def load_summary_rows(root: pathlib.Path) -> dict[tuple[str, str], dict[str, Any]]:
    summary_path = root / "summary.json"
    if not summary_path.exists():
        return {}
    summary = load_json(summary_path)
    return {
        (row.get("model"), row.get("case")): row
        for row in summary.get("rows", [])
    }


def route_family(provider: str) -> str:
    if "_mul_mat_id_" in provider:
        return "moe_id"
    if "_flash_attn_" in provider:
        return "attention/flash"
    if "_get_rows_" in provider:
        return "movement/get_rows"
    if "_mul_mat_vec_f32" in provider or "_mul_mat_vec_f16" in provider:
        return "dense/f32_f16"
    if "_q4_k_" in provider:
        return "dense/q4_K"
    if "_q5_k_" in provider:
        return "dense/q5_K"
    if "_q6_k_" in provider:
        return "dense/q6_K"
    if "_q8_0_" in provider:
        return "dense/q8_0"
    return "other"


def summarize_routes(record: dict[str, Any] | None, top: int) -> dict[str, Any]:
    if not record:
        return {"top_routes": [], "route_families": []}
    top_routes = record.get("top_routes") or []
    families: collections.Counter[str] = collections.Counter()
    for row in top_routes:
        families[route_family(row.get("provider", ""))] += int(row.get("count") or 0)
    return {
        "top_routes": top_routes[:top],
        "route_families": [
            {"family": family, "count": count}
            for family, count in families.most_common(top)
        ],
    }


def analyze(root: pathlib.Path, top: int) -> dict[str, Any]:
    summary_rows = load_summary_rows(root)
    records = load_records(root)
    models = []

    for log in sorted((root / "vulkan").glob("*/p*/stderr.log")):
        model = log.parents[1].name
        case = log.parent.name
        perf_blocks = parse_perf_blocks(log)
        perf = summarize_perf_blocks(perf_blocks, top)
        summary = summary_rows.get((model, case), {})
        hrx_record = records.get(("hrx", model, case))
        models.append(
            {
                "model": model,
                "case": case,
                "steady_ratio": summary.get("steady_ratio"),
                "hrx_steady_ts": summary.get("hrx_steady_ts"),
                "vulkan_steady_ts": summary.get("vulkan_steady_ts"),
                "vulkan_perf_blocks": perf["blocks"],
                "vulkan_perf_rows": perf["rows_all"],
                "vulkan_perf_steady_rows": perf["rows_steady"],
                "vulkan": perf,
                "hrx": summarize_routes(hrx_record, top),
            }
        )

    models.sort(key=lambda row: (row.get("steady_ratio") is None, row.get("steady_ratio") or 0.0))
    return {
        "artifact": str(root),
        "models": models,
    }


def fmt_us(value: float | None) -> str:
    if value is None:
        return ""
    if value >= 1000:
        return f"{value / 1000:.2f} ms"
    return f"{value:.1f} us"


def write_markdown(path: pathlib.Path, report: dict[str, Any], top: int) -> None:
    lines = [
        "# HRX/Vulkan Basket Perf Rank",
        "",
        f"Artifact: `{report['artifact']}`",
        "",
    ]
    for model in report["models"]:
        ratio = model.get("steady_ratio")
        ratio_s = "" if ratio is None else f"{ratio:.3f}x"
        lines.extend(
            [
                f"## {model['model']} / {model['case']} / {ratio_s}",
                "",
                f"Vulkan perf blocks: `{model.get('vulkan_perf_blocks', 0)}`; steady skips the first block when multiple blocks are present.",
                "",
                "| Vulkan steady family | summed time | dispatches |",
                "| --- | ---: | ---: |",
            ]
        )
        for row in model["vulkan"]["steady"]["top_families"][:top]:
            lines.append(
                f"| `{row['family']}` | {fmt_us(row['total_us_sum'])} | {row['dispatch_count_sum']} |"
            )
        lines.extend(["", "| Vulkan steady label | summed time | weighted avg | dispatches |", "| --- | ---: | ---: | ---: |"])
        for row in model["vulkan"]["steady"]["top_labels"][:top]:
            lines.append(
                f"| `{row['label']}` | {fmt_us(row['total_us_sum'])} | "
                f"{fmt_us(row['avg_us_weighted'])} | {row['dispatch_count_sum']} |"
            )
        if model.get("vulkan_perf_blocks", 0) > 1:
            lines.extend(["", "| Vulkan cold label | summed time | weighted avg | dispatches |", "| --- | ---: | ---: | ---: |"])
            for row in model["vulkan"]["cold"]["top_labels"][: min(5, top)]:
                lines.append(
                    f"| `{row['label']}` | {fmt_us(row['total_us_sum'])} | "
                    f"{fmt_us(row['avg_us_weighted'])} | {row['dispatch_count_sum']} |"
                )
        lines.extend(["", "| HRX route | count |", "| --- | ---: |"])
        for row in model["hrx"]["top_routes"][:top]:
            lines.append(f"| `{row['provider']}` | {row['count']} |")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=pathlib.Path, help="Basket artifact directory.")
    parser.add_argument("--top", type=int, default=8)
    parser.add_argument("--out-dir", type=pathlib.Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.artifact.resolve()
    report = analyze(root, args.top)
    out_dir = args.out_dir.resolve() if args.out_dir else root
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "perf-rank.json"
    md_path = out_dir / "perf-rank.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_markdown(md_path, report, args.top)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
