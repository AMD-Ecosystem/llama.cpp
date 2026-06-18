#!/usr/bin/env python3
import argparse
import collections
import json
import pathlib
import re


INSTRUCTION_RE = re.compile(r"^\s+([A-Za-z0-9_.]+)\b")
STAT_RE = re.compile(r"^([^:]+):\s+(.+)$")
SPV_ID_RE = re.compile(r"^\s*(%[A-Za-z0-9_]+)\s+=\s+(Op\S+)\s+(.+)$")


INTERESTING_PREFIXES = (
    "v_wmma",
    "v_mfma",
    "v_dot",
    "v_fma",
    "v_fmac",
    "v_pk",
    "buffer_load",
    "buffer_store",
    "global_load",
    "global_store",
    "flat_load",
    "flat_store",
    "ds_load",
    "ds_read",
    "ds_store",
    "ds_write",
    "s_barrier",
    "s_waitcnt",
    "s_load",
)


def read_lines(path):
    return pathlib.Path(path).read_text(encoding="utf-8", errors="replace").splitlines()


def strip_comment(line):
    for sep in ("//", ";"):
        if sep in line:
            line = line.split(sep, 1)[0]
    return line.rstrip()


def extract_instructions(lines):
    opcodes = []
    for line in lines:
        match = INSTRUCTION_RE.match(strip_comment(line))
        if match:
            opcodes.append(match.group(1))
    return opcodes


def parse_stats(path):
    if not path:
        return {}
    stats = {}
    for line in read_lines(path):
        if line.startswith("# metadata:"):
            continue
        match = STAT_RE.match(line.strip())
        if not match:
            continue
        key, value = match.group(1).strip(), match.group(2).strip()
        try:
            stats[key] = int(value)
        except ValueError:
            stats[key] = value
    return stats


def parse_metadata(lines):
    for line in lines:
        if line.startswith("# metadata:"):
            return json.loads(line.split(":", 1)[1].strip())
    return {}


def parse_spvasm(path):
    lines = read_lines(path)
    names = {}
    constants = {}
    coop_types = []
    coop_ops = []
    execution_modes = []
    entry_points = []
    decorations = []

    for index, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped:
            continue
        if " OpName " in f" {stripped} ":
            parts = stripped.split(maxsplit=2)
            if len(parts) == 3 and parts[0] == "OpName":
                ident = parts[1]
                names[ident] = parts[2].strip('"')
            continue
        if stripped.startswith("OpEntryPoint"):
            entry_points.append({"line": index, "text": stripped})
            continue
        if stripped.startswith("OpExecutionMode"):
            execution_modes.append({"line": index, "text": stripped})
            continue
        if stripped.startswith("OpDecorate") and ("SpecId" in stripped or "BuiltIn" in stripped):
            decorations.append({"line": index, "text": stripped})
            continue

        if stripped.startswith("OpCooperativeMatrix"):
            opcode = stripped.split(maxsplit=1)[0]
            coop_ops.append({
                "line": index,
                "id": None,
                "opcode": opcode,
                "text": stripped,
                "context": lines[max(0, index - 3):min(len(lines), index + 2)],
            })
            continue

        match = SPV_ID_RE.match(stripped)
        if not match:
            continue
        ident, opcode, rest = match.groups()
        if opcode in ("OpConstant", "OpSpecConstant", "OpSpecConstantOp", "OpConstantComposite"):
            constants[ident] = {
                "line": index,
                "opcode": opcode,
                "text": stripped,
                "name": names.get(ident),
            }
        if opcode == "OpTypeCooperativeMatrixKHR":
            coop_types.append({
                "line": index,
                "id": ident,
                "name": names.get(ident),
                "text": stripped,
            })
        elif opcode.startswith("OpCooperativeMatrix"):
            coop_ops.append({
                "line": index,
                "id": ident,
                "opcode": opcode,
                "text": stripped,
                "context": lines[max(0, index - 3):min(len(lines), index + 2)],
            })

    return {
        "path": str(path),
        "entry_points": entry_points,
        "execution_modes": execution_modes,
        "decorations": decorations,
        "constants": constants,
        "coop_types": coop_types,
        "coop_ops": coop_ops,
    }


def summarize_isa(path, stats_path=None):
    lines = read_lines(path)
    opcodes = extract_instructions(lines)
    counts = collections.Counter(opcodes)
    interesting = {
        key: counts[key]
        for key in sorted(counts)
        if key.startswith(INTERESTING_PREFIXES)
    }
    return {
        "path": str(path),
        "metadata": parse_metadata(lines),
        "resources": parse_stats(stats_path),
        "instruction_count": len(opcodes),
        "interesting_opcodes": interesting,
        "top_opcodes": dict(counts.most_common(40)),
    }


def load_compare(path):
    if not path:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    result = {
        "path": str(path),
        "lhs_label": data.get("lhs_label", "lhs"),
        "rhs_label": data.get("rhs_label", "rhs"),
    }
    for key in (data.get("lhs_key", "radv"), data.get("rhs_key", "rhs")):
        if key in data:
            result[key] = {
                "name": data[key].get("name"),
                "metadata": data[key].get("metadata", {}),
                "resources": data[key].get("resources", {}),
                "interesting_opcodes": data[key].get("interesting_opcodes", {}),
                "top_opcodes": data[key].get("top_opcodes", {}),
            }
    return result


def extract_source_lines(path, start, end):
    if not path:
        return []
    lines = read_lines(path)
    return [
        {"line": i, "text": lines[i - 1]}
        for i in range(max(1, start), min(len(lines), end) + 1)
    ]


def write_markdown(path, payload):
    spv = payload["spvasm"]
    radv = payload["radv"]
    compare = payload.get("compare")
    lines = ["# Cooperative Matrix Schedule Extract", ""]

    meta = radv.get("metadata", {})
    if meta:
        lines += [
            "## Pipeline",
            "",
            f"- pipeline: `{meta.get('pipeline')}`",
            f"- hash: `{meta.get('spv_hash')}`",
            f"- spec: `{meta.get('spec')}`",
            f"- workgroup denominators: `{meta.get('wg_denoms')}`",
            f"- require full subgroups: `{meta.get('require_full_subgroups')}`",
            "",
        ]

    lines += ["## SPIR-V Cooperative Matrix", ""]
    for item in spv["coop_types"]:
        lines.append(f"- line {item['line']}: `{item['text']}`")
    lines.append("")
    op_counts = collections.Counter(item["opcode"] for item in spv["coop_ops"])
    lines += ["| Op | Count |", "| --- | ---: |"]
    for opcode, count in sorted(op_counts.items()):
        lines.append(f"| `{opcode}` | {count} |")
    lines.append("")
    for item in spv["coop_ops"]:
        if item["opcode"] in ("OpCooperativeMatrixLoadKHR", "OpCooperativeMatrixMulAddKHR", "OpCooperativeMatrixStoreKHR"):
            lines.append(f"- line {item['line']}: `{item['text']}`")
    lines.append("")

    if payload.get("shader_source"):
        lines += ["## Vulkan Source Window", ""]
        for item in payload["shader_source"]:
            lines.append(f"{item['line']}: `{item['text']}`")
        lines.append("")

    lines += ["## RADV ISA", "", "| Field | Value |", "| --- | ---: |"]
    for key in ("SGPRs", "VGPRs", "LDS size", "Spilled SGPRs", "Spilled VGPRs", "Scratch size", "Instructions", "VMEM", "SMEM"):
        if key in radv["resources"]:
            lines.append(f"| `{key}` | `{radv['resources'][key]}` |")
    lines.append("")
    lines += ["| Opcode | Count |", "| --- | ---: |"]
    for opcode, count in radv["interesting_opcodes"].items():
        lines.append(f"| `{opcode}` | {count} |")
    lines.append("")

    if compare:
        lines += ["## Current HIP Delta", ""]
        keys = [key for key in compare if isinstance(compare.get(key), dict) and "interesting_opcodes" in compare[key]]
        for key in keys:
            item = compare[key]
            lines.append(f"### {key}")
            lines.append("")
            resources = item.get("resources", {})
            for rkey in ("wavefront_size", "sgpr_count", "vgpr_count", "group_segment_fixed_size", "private_segment_fixed_size", "sgpr_spill_count", "vgpr_spill_count", "SGPRs", "VGPRs", "LDS size"):
                if rkey in resources:
                    lines.append(f"- `{rkey}`: `{resources[rkey]}`")
            lines.append("")
            lines += ["| Opcode | Count |", "| --- | ---: |"]
            for opcode, count in item.get("interesting_opcodes", {}).items():
                lines.append(f"| `{opcode}` | {count} |")
            lines.append("")

    lines += ["## Next Probe Contract", ""]
    lines += [
        "- Do not treat matching `BM/BN/BK/WG` as sufficient.",
        "- Preserve p33 as a medium/narrow route and test p513 as a large tail route.",
        "- A new direct-WMMA probe must explain the cooperative-matrix store/lane mapping it is trying to reproduce.",
        "- A packed-Q8_1 probe must state which RADV schedule fact is being imported without reintroducing the known 128x128 spill shape.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Extract SPIR-V/RADV cooperative-matrix schedule facts for an oracle pipeline.")
    parser.add_argument("--spvasm", required=True, type=pathlib.Path)
    parser.add_argument("--radv-isa", required=True, type=pathlib.Path)
    parser.add_argument("--radv-stats", type=pathlib.Path)
    parser.add_argument("--compare-json", type=pathlib.Path)
    parser.add_argument("--shader-source", type=pathlib.Path)
    parser.add_argument("--source-start", type=int, default=252)
    parser.add_argument("--source-end", type=int, default=425)
    parser.add_argument("--out-json", type=pathlib.Path)
    parser.add_argument("--out-md", type=pathlib.Path)
    args = parser.parse_args()

    payload = {
        "spvasm": parse_spvasm(args.spvasm),
        "radv": summarize_isa(args.radv_isa, args.radv_stats),
        "compare": load_compare(args.compare_json),
        "shader_source": extract_source_lines(args.shader_source, args.source_start, args.source_end),
    }

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(args.out_md, payload)
    if not args.out_json and not args.out_md:
        print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
