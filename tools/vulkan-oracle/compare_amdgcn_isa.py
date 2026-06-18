#!/usr/bin/env python3
import argparse
import collections
import json
import pathlib
import re
import subprocess


INSTRUCTION_RE = re.compile(r"^\s+([A-Za-z0-9_.]+)\b")
SYMBOL_RE = re.compile(r"^[0-9a-fA-F]+ <([^>]+)>:")
RADV_STAT_RE = re.compile(r"^([^:]+):\s+(.+)$")


def run_text(argv):
    result = subprocess.run(argv, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise SystemExit(f"command failed ({result.returncode}): {' '.join(argv)}\n{result.stderr}")
    return result.stdout


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
        line = strip_comment(line)
        match = INSTRUCTION_RE.match(line)
        if match:
            opcodes.append(match.group(1))
    return opcodes


def extract_objdump_symbol(lines, symbol):
    if not symbol:
        return lines

    result = []
    in_symbol = False
    for line in lines:
        match = SYMBOL_RE.match(line)
        if match:
            in_symbol = symbol in match.group(1)
            if in_symbol:
                result.append(line)
            continue
        if in_symbol:
            result.append(line)
    if not result:
        raise SystemExit(f"symbol substring not found in objdump: {symbol}")
    return result


def parse_radv_stats(path):
    if not path:
        return {}
    stats = {}
    for line in read_lines(path):
        if line.startswith("# metadata:"):
            continue
        match = RADV_STAT_RE.match(line.strip())
        if not match:
            continue
        key = match.group(1).strip()
        value = match.group(2).strip()
        try:
            stats[key] = int(value)
        except ValueError:
            stats[key] = value
    return stats


def parse_radv_metadata(lines):
    for line in lines:
        if line.startswith("# metadata:"):
            return json.loads(line.split(":", 1)[1].strip())
    return {}


def parse_hsaco_metadata(readelf_lines, symbol):
    kernels = []
    current = None
    for raw in readelf_lines:
        line = raw.strip()
        if line == "- .args:":
            if current:
                kernels.append(current)
            current = {}
            continue
        if current is None:
            continue
        if line.startswith(".name:"):
            current["name"] = line.split(":", 1)[1].strip()
        elif line.startswith(".symbol:"):
            current["symbol"] = line.split(":", 1)[1].strip()
        elif line.startswith(".group_segment_fixed_size:"):
            current["group_segment_fixed_size"] = parse_int_tail(line)
        elif line.startswith(".private_segment_fixed_size:"):
            current["private_segment_fixed_size"] = parse_int_tail(line)
        elif line.startswith(".sgpr_count:"):
            current["sgpr_count"] = parse_int_tail(line)
        elif line.startswith(".sgpr_spill_count:"):
            current["sgpr_spill_count"] = parse_int_tail(line)
        elif line.startswith(".vgpr_count:"):
            current["vgpr_count"] = parse_int_tail(line)
        elif line.startswith(".vgpr_spill_count:"):
            current["vgpr_spill_count"] = parse_int_tail(line)
        elif line.startswith(".wavefront_size:"):
            current["wavefront_size"] = parse_int_tail(line)
    if current:
        kernels.append(current)

    if symbol:
        for kernel in kernels:
            if symbol in kernel.get("name", "") or symbol in kernel.get("symbol", ""):
                return kernel
        raise SystemExit(f"symbol substring not found in HSACO metadata: {symbol}")
    return kernels[0] if kernels else {}


def parse_int_tail(line):
    return int(line.split(":", 1)[1].strip().split()[0])


def classify(opcode):
    if opcode.startswith(("global_", "buffer_", "flat_")):
        if "load" in opcode:
            return "vmem_load"
        if "store" in opcode:
            return "vmem_store"
        return "vmem_other"
    if opcode.startswith(("s_load", "s_buffer_load")):
        return "smem_load"
    if opcode.startswith("ds_read"):
        return "lds_read"
    if opcode.startswith("ds_write"):
        return "lds_write"
    if opcode.startswith("s_barrier"):
        return "barrier"
    if opcode.startswith("s_waitcnt"):
        return "waitcnt"
    if opcode.startswith(("s_branch", "s_cbranch", "s_setpc")):
        return "branch"
    if opcode.startswith(("v_wmma", "v_mfma", "v_dot", "v_fmac", "v_fma", "v_mad", "v_pk_", "v_add", "v_mul", "v_sub", "v_cmp", "v_lsh", "v_and", "v_or", "v_mov", "v_cvt", "v_read", "v_perm", "v_bfe", "v_min", "v_max", "v_rcp", "v_exp", "v_log", "v_med", "v_mac", "v_accvgpr")):
        return "valu"
    if opcode.startswith("v_"):
        return "valu_other"
    if opcode.startswith(("s_", "sop", "salu")):
        return "salu"
    return "other"


def summarize(name, opcodes, resources=None, metadata=None):
    opcode_counts = collections.Counter(opcodes)
    class_counts = collections.Counter(classify(opcode) for opcode in opcodes)
    return {
        "name": name,
        "metadata": metadata or {},
        "resources": resources or {},
        "instruction_count": len(opcodes),
        "classes": dict(sorted(class_counts.items())),
        "interesting_opcodes": {
            key: opcode_counts[key]
            for key in sorted(opcode_counts)
            if key.startswith(("v_wmma", "v_mfma", "v_dot", "global_", "buffer_", "flat_", "ds_", "s_barrier", "s_waitcnt"))
        },
        "top_opcodes": dict(opcode_counts.most_common(30)),
    }


def write_markdown(path, payload):
    lines = ["# AMDGCN ISA Comparison", ""]
    lhs_key = payload.get("lhs_key", "radv")
    rhs_key = payload.get("rhs_key", "hsaco")
    lhs_label = payload.get("lhs_label", lhs_key.upper())
    rhs_label = payload.get("rhs_label", rhs_key.upper())
    for key, label in ((lhs_key, lhs_label), (rhs_key, rhs_label)):
        summary = payload[key]
        lines += [
            f"## {label}",
            "",
            f"- name: `{summary['name']}`",
            f"- instruction count: `{summary['instruction_count']}`",
        ]
        for mkey, value in summary.get("metadata", {}).items():
            lines.append(f"- metadata {mkey}: `{value}`")
        for rkey, value in summary.get("resources", {}).items():
            lines.append(f"- {rkey}: `{value}`")
        lines += ["", "### Instruction Classes", "", "| Class | Count |", "| --- | ---: |"]
        for cls, count in summary["classes"].items():
            lines.append(f"| `{cls}` | {count} |")
        lines += ["", "### Interesting Opcodes", "", "| Opcode | Count |", "| --- | ---: |"]
        for opcode, count in summary["interesting_opcodes"].items():
            lines.append(f"| `{opcode}` | {count} |")
        lines += ["", "### Top Opcodes", "", "| Opcode | Count |", "| --- | ---: |"]
        for opcode, count in summary["top_opcodes"].items():
            lines.append(f"| `{opcode}` | {count} |")
        lines.append("")

    lines += ["## Delta", "", f"| Field | {lhs_label} | {rhs_label} |", "| --- | ---: | ---: |"]
    all_classes = sorted(set(payload[lhs_key]["classes"]) | set(payload[rhs_key]["classes"]))
    for cls in all_classes:
        lines.append(f"| class `{cls}` | {payload[lhs_key]['classes'].get(cls, 0)} | {payload[rhs_key]['classes'].get(cls, 0)} |")
    pathlib.Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Compare RADV AMDGCN text with either a CMake-built HSACO or another AMDGCN text file.")
    parser.add_argument("--radv-isa", required=True, type=pathlib.Path)
    parser.add_argument("--radv-stats", type=pathlib.Path)
    rhs = parser.add_mutually_exclusive_group(required=True)
    rhs.add_argument("--hsaco", type=pathlib.Path)
    rhs.add_argument("--other-isa", type=pathlib.Path, help="Compare against another AMDGCN text file instead of a HSACO.")
    parser.add_argument("--hsaco-symbol", help="Substring of the HSACO kernel symbol to compare.")
    parser.add_argument("--other-name", help="Display name for --other-isa in reports.")
    parser.add_argument("--other-stats", type=pathlib.Path, help="RADV-style stats file for --other-isa.")
    parser.add_argument("--llvm-objdump", default="llvm-objdump", type=pathlib.Path)
    parser.add_argument("--llvm-readelf", default="llvm-readelf", type=pathlib.Path)
    parser.add_argument("--out-json", type=pathlib.Path)
    parser.add_argument("--out-md", type=pathlib.Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.hsaco and not args.hsaco_symbol:
        raise SystemExit("--hsaco-symbol is required with --hsaco")
    if args.hsaco_symbol and not args.hsaco:
        raise SystemExit("--hsaco-symbol is only valid with --hsaco")

    radv_lines = read_lines(args.radv_isa)
    radv_opcodes = extract_instructions(radv_lines)
    radv_metadata = parse_radv_metadata(radv_lines)
    radv_resources = parse_radv_stats(args.radv_stats)

    if args.hsaco:
        objdump_lines = run_text([str(args.llvm_objdump), "--no-show-raw-insn", "-d", str(args.hsaco)]).splitlines()
        rhs_lines = extract_objdump_symbol(objdump_lines, args.hsaco_symbol)
        rhs_name = args.hsaco_symbol
        readelf_lines = run_text([str(args.llvm_readelf), "--notes", str(args.hsaco)]).splitlines()
        rhs_resources = parse_hsaco_metadata(readelf_lines, args.hsaco_symbol)
        rhs_label = "HSACO"
    else:
        rhs_lines = read_lines(args.other_isa)
        rhs_name = args.other_name or args.other_isa.name
        rhs_resources = parse_radv_stats(args.other_stats)
        rhs_label = "OTHER"
    rhs_opcodes = extract_instructions(rhs_lines)

    payload = {
        "lhs_key": "radv",
        "rhs_key": "rhs",
        "lhs_label": "RADV",
        "rhs_label": rhs_label,
        "radv": summarize(args.radv_isa.name, radv_opcodes, radv_resources, radv_metadata),
        "rhs": summarize(rhs_name, rhs_opcodes, rhs_resources),
    }

    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(args.out_md, payload)


if __name__ == "__main__":
    main()
