#!/usr/bin/env python3
import argparse
import json
import pathlib
import sys


OPCODES = [
    "v_wmma_f16_16x16x16_f16",
    "v_dot4_i32_iu8",
    "ds_load_b64",
    "ds_load_u16_d16",
    "ds_store_b16",
    "buffer_store_b32",
    "global_store_b32",
    "s_barrier",
    "s_waitcnt",
    "s_waitcnt_depctr",
]


def normalized_resources(summary):
    resources = summary.get("resources", {})
    return {
        "sgpr": resources.get("SGPRs", resources.get("sgpr_count")),
        "vgpr": resources.get("VGPRs", resources.get("vgpr_count")),
        "lds": resources.get("LDS size", resources.get("group_segment_fixed_size")),
        "scratch": resources.get("Scratch size", resources.get("private_segment_fixed_size")),
        "wave": resources.get("wavefront_size"),
        "instructions": resources.get("Instructions", summary.get("instruction_count")),
    }


def interesting_opcodes(summary):
    return summary.get("interesting_opcodes", {})


def event_score(summary, name):
    return summary.get("event_summary", {}).get(name, {})


def fmt(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return str(value)


def load_row(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    lhs_key = data.get("lhs_key", "radv")
    rhs_key = data.get("rhs_key", "rhs")
    lhs = data[lhs_key]
    rhs = data[rhs_key]
    lhs_resources = normalized_resources(lhs)
    rhs_resources = normalized_resources(rhs)
    lhs_opcodes = interesting_opcodes(lhs)
    rhs_opcodes = interesting_opcodes(rhs)
    lhs_wmma = event_score(lhs, "wmma_score")
    rhs_wmma = event_score(rhs, "wmma_score")
    lhs_hot = event_score(lhs, "hot_op_score")
    rhs_hot = event_score(rhs, "hot_op_score")

    row = {
        "file": str(path),
        "label": path.stem.replace("-compare", ""),
        "radv_name": lhs.get("name"),
        "candidate_name": rhs.get("name"),
        "radv_resources": lhs_resources,
        "candidate_resources": rhs_resources,
        "radv_opcodes": {opcode: lhs_opcodes.get(opcode, 0) for opcode in OPCODES},
        "candidate_opcodes": {opcode: rhs_opcodes.get(opcode, 0) for opcode in OPCODES},
        "radv_wmma_score": lhs_wmma,
        "candidate_wmma_score": rhs_wmma,
        "radv_hot_score": lhs_hot,
        "candidate_hot_score": rhs_hot,
    }
    return row


def write_markdown(path, rows):
    lines = [
        "# ISA Compare Matrix",
        "",
        "| Candidate | SGPR | VGPR | LDS | Scratch | Wave | Hot op | WMMA/dot | LDS b64 | d16 loads | LDS stores | Stores | Barriers | Waits | First-window loads | Final lgkmcnt | Hot ops in window |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        res = row["candidate_resources"]
        op = row["candidate_opcodes"]
        hot = row["candidate_hot_score"]
        wmma = row["candidate_wmma_score"]
        hot_opcodes = hot.get("hot_opcodes_in_window") or {}
        hot_name = next(iter(hot_opcodes), "")
        math_count = op.get("v_wmma_f16_16x16x16_f16", 0) or op.get("v_dot4_i32_iu8", 0)
        store_count = op.get("buffer_store_b32", 0) or op.get("global_store_b32", 0)
        first_loads = (
            wmma.get("pre_wmma_ds_load_b64")
            if op.get("v_wmma_f16_16x16x16_f16", 0)
            else hot.get("pre_hot_lds_load")
        )
        final_lgkmcnt = (
            wmma.get("final_pre_wmma_lgkmcnt")
            if op.get("v_wmma_f16_16x16x16_f16", 0)
            else hot.get("final_pre_hot_lgkmcnt")
        )
        hot_window = (
            wmma.get("wmma_in_window")
            if op.get("v_wmma_f16_16x16x16_f16", 0)
            else hot.get("hot_op_in_window")
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['label']}`",
                    fmt(res.get("sgpr")),
                    fmt(res.get("vgpr")),
                    fmt(res.get("lds")),
                    fmt(res.get("scratch")),
                    fmt(res.get("wave")),
                    f"`{hot_name}`" if hot_name else "",
                    fmt(math_count),
                    fmt(op.get("ds_load_b64", 0)),
                    fmt(op.get("ds_load_u16_d16", 0)),
                    fmt(op.get("ds_store_b16", 0)),
                    fmt(store_count),
                    fmt(op.get("s_barrier", 0)),
                    fmt(op.get("s_waitcnt", 0)),
                    fmt(first_loads),
                    fmt(final_lgkmcnt),
                    fmt(hot_window),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## RADV Reference Rows",
            "",
            "| Candidate | RADV SGPR | RADV VGPR | RADV LDS | RADV math | RADV LDS b64 | RADV d16 loads | RADV stores | RADV first-window loads | RADV final lgkmcnt | RADV hot ops in window |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        res = row["radv_resources"]
        op = row["radv_opcodes"]
        hot = row["radv_hot_score"]
        wmma = row["radv_wmma_score"]
        math_count = op.get("v_wmma_f16_16x16x16_f16", 0) or op.get("v_dot4_i32_iu8", 0)
        store_count = op.get("buffer_store_b32", 0) or op.get("global_store_b32", 0)
        first_loads = wmma.get("pre_wmma_ds_load_b64") or hot.get("pre_hot_lds_load")
        final_lgkmcnt = wmma.get("final_pre_wmma_lgkmcnt") or hot.get("final_pre_hot_lgkmcnt")
        hot_window = wmma.get("wmma_in_window") or hot.get("hot_op_in_window")
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['label']}`",
                    fmt(res.get("sgpr")),
                    fmt(res.get("vgpr")),
                    fmt(res.get("lds")),
                    fmt(math_count),
                    fmt(op.get("ds_load_b64", 0)),
                    fmt(op.get("ds_load_u16_d16", 0)),
                    fmt(store_count),
                    fmt(first_loads),
                    fmt(final_lgkmcnt),
                    fmt(hot_window),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Summarize multiple RADV-vs-HRX ISA comparison JSON files.")
    parser.add_argument("compare_json", nargs="+", type=pathlib.Path)
    parser.add_argument("--out-json", type=pathlib.Path)
    parser.add_argument("--out-md", type=pathlib.Path)
    args = parser.parse_args()

    rows = [load_row(path) for path in args.compare_json]
    payload = {"rows": rows}

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(args.out_md, rows)
    if not args.out_json and not args.out_md:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
