#!/usr/bin/env python3
import argparse
import collections
import json
import pathlib
import re


INSTRUCTION_RE = re.compile(r"^\s+([A-Za-z0-9_.]+)\b")
STAT_RE = re.compile(r"^([^:]+):\s+(.+)$")
SPV_ID_RE = re.compile(r"^\s*(%[A-Za-z0-9_]+)\s+=\s+(Op\S+)\s+(.+)$")
ISA_INSTRUCTION_RE = re.compile(r"^\s+([A-Za-z0-9_.]+)\s*(.*)$")
BB_RE = re.compile(r"^(BB[0-9]+):")
OFFSET_RE = re.compile(r"\boffset(?P<which>[0-9]*)[: ](?P<value>-?(?:0x[0-9a-fA-F]+|[0-9]+))")


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


def parse_int(value):
    return int(value, 0)


def extract_offset(rest):
    offsets = {}
    for match in OFFSET_RE.finditer(rest):
        key = "offset" if not match.group("which") else f"offset{match.group('which')}"
        offsets[key] = parse_int(match.group("value"))
    return offsets


def parse_isa_events(lines):
    events = []
    bb = None
    for index, line in enumerate(lines, 1):
        bb_match = BB_RE.match(line.strip())
        if bb_match:
            bb = bb_match.group(1)
            continue
        stripped = strip_comment(line)
        match = ISA_INSTRUCTION_RE.match(stripped)
        if not match:
            continue
        opcode, rest = match.groups()
        if not opcode.startswith(INTERESTING_PREFIXES):
            continue
        operands = [part.strip() for part in rest.split(",") if part.strip()]
        events.append({
            "line": index,
            "bb": bb,
            "opcode": opcode,
            "operands": operands,
            "offsets": extract_offset(rest),
            "text": line.strip(),
        })
    return events


def summarize_isa_events(events):
    store_blocks = []
    by_bb = collections.defaultdict(collections.Counter)
    for event in events:
        if event["bb"]:
            by_bb[event["bb"]][event["opcode"]] += 1
    for bb, counts in sorted(by_bb.items(), key=lambda item: int(item[0][2:])):
        if any(op.startswith(("buffer_store", "global_store", "flat_store", "ds_store", "ds_write")) for op in counts):
            store_blocks.append({
                "bb": bb,
                "opcodes": dict(sorted(counts.items())),
                "store_ops": sum(
                    count for op, count in counts.items()
                    if op.startswith(("buffer_store", "global_store", "flat_store", "ds_store", "ds_write"))
                ),
            })

    first_wmma_index = next(
        (index for index, event in enumerate(events) if event["opcode"].startswith(("v_wmma", "v_mfma"))),
        None)
    if first_wmma_index is None:
        pre_wmma = []
        wmma_window = []
    else:
        pre_wmma = events[max(0, first_wmma_index - 80):first_wmma_index]
        wmma_window = events[max(0, first_wmma_index - 16):min(len(events), first_wmma_index + 40)]

    lds_load_offsets = collections.defaultdict(list)
    for event in pre_wmma:
        if event["opcode"] != "ds_load_b64" or len(event["operands"]) < 2:
            continue
        base = event["operands"][1].split()[0]
        lds_load_offsets[base].append(event["offsets"].get("offset", 0))

    return {
        "store_blocks": store_blocks,
        "pre_wmma_ds_load_b64_offsets": {
            base: values for base, values in sorted(lds_load_offsets.items())
        },
        "wmma_window": wmma_window,
    }


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
    events = parse_isa_events(lines)
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
        "event_summary": summarize_isa_events(events),
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

    event_summary = radv.get("event_summary", {})
    if event_summary:
        lines += ["## RADV ISA Event Windows", ""]
        pre_offsets = event_summary.get("pre_wmma_ds_load_b64_offsets", {})
        if pre_offsets:
            lines += ["### Pre-WMMA `ds_load_b64` Offsets", ""]
            for base, offsets in pre_offsets.items():
                rendered = ", ".join(str(value) for value in offsets)
                lines.append(f"- `{base}`: `{rendered}`")
            lines.append("")

        store_blocks = event_summary.get("store_blocks", [])
        if store_blocks:
            lines += ["### Store Basic Blocks", "", "| BB | Store Ops | Interesting Ops |", "| --- | ---: | --- |"]
            for block in store_blocks:
                ops = ", ".join(f"{op}={count}" for op, count in block["opcodes"].items())
                lines.append(f"| `{block['bb']}` | {block['store_ops']} | `{ops}` |")
            lines.append("")

        window = event_summary.get("wmma_window", [])
        if window:
            lines += ["### First WMMA Window", ""]
            for event in window:
                lines.append(f"- line {event['line']}: `{event['text']}`")
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
