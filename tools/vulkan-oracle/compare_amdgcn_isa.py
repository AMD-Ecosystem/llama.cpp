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
ISA_INSTRUCTION_RE = re.compile(r"^\s+([A-Za-z0-9_.]+)\s*(.*)$")
BB_RE = re.compile(r"^\s*(BB[0-9]+|\.L[A-Za-z0-9_.$]+):")
OFFSET_RE = re.compile(r"\boffset(?P<which>[0-9]*)[: ](?P<value>-?(?:0x[0-9a-fA-F]+|[0-9]+))")
LGKMCNT_RE = re.compile(r"\blgkmcnt\((?P<value>[0-9]+)\)")
INTERESTING_PREFIXES = (
    "v_wmma",
    "v_mfma",
    "v_dot",
    "global_",
    "buffer_",
    "flat_",
    "ds_",
    "s_barrier",
    "s_waitcnt",
)
STORE_PREFIXES = ("buffer_store", "global_store", "flat_store", "ds_store", "ds_write")


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
    def bb_sort_key(name):
        if name.startswith("BB") and name[2:].isdigit():
            return (0, int(name[2:]), name)
        return (1, 0, name)

    for bb, counts in sorted(by_bb.items(), key=lambda item: bb_sort_key(item[0])):
        if any(op.startswith(STORE_PREFIXES) for op in counts):
            store_blocks.append({
                "bb": bb,
                "opcodes": dict(sorted(counts.items())),
                "store_ops": sum(count for op, count in counts.items() if op.startswith(STORE_PREFIXES)),
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

    wmma_score = score_wmma_window(pre_wmma, wmma_window)

    store_windows = []
    seen_store_lines = set()
    for index, event in enumerate(events):
        if not event["opcode"].startswith(STORE_PREFIXES) or event["line"] in seen_store_lines:
            continue
        window = events[max(0, index - 4):min(len(events), index + 8)]
        for item in window:
            if item["opcode"].startswith(STORE_PREFIXES):
                seen_store_lines.add(item["line"])
        store_windows.append({
            "first_store_line": event["line"],
            "opcodes": dict(collections.Counter(item["opcode"] for item in window)),
            "events": window,
        })
        if len(store_windows) >= 16:
            break

    return {
        "store_blocks": store_blocks,
        "store_windows": store_windows,
        "pre_wmma_ds_load_b64_offsets": {
            base: values for base, values in sorted(lds_load_offsets.items())
        },
        "wmma_score": wmma_score,
        "wmma_window": wmma_window,
    }


def parse_lgkmcnt(event):
    match = LGKMCNT_RE.search(event.get("text", ""))
    return int(match.group("value")) if match else None


def score_wmma_window(pre_wmma, wmma_window):
    pre_b64 = [event for event in pre_wmma if event["opcode"] == "ds_load_b64"]
    pre_waits = [event for event in pre_wmma if event["opcode"] == "s_waitcnt"]
    bases = sorted({
        event["operands"][1].split()[0]
        for event in pre_b64
        if len(event["operands"]) >= 2
    })

    last_wait_index = None
    for index in range(len(pre_wmma) - 1, -1, -1):
        if pre_wmma[index]["opcode"] == "s_waitcnt":
            last_wait_index = index
            break

    if last_wait_index is None:
        final_wait_lgkmcnt = None
        load_b64_before_final_wait = len(pre_b64)
    else:
        final_wait_lgkmcnt = parse_lgkmcnt(pre_wmma[last_wait_index])
        load_b64_before_final_wait = 0
        for event in reversed(pre_wmma[:last_wait_index]):
            if event["opcode"] == "ds_load_b64":
                load_b64_before_final_wait += 1
            elif event["opcode"] != "s_waitcnt_depctr":
                break

    wmma_seen = 0
    waits_between_wmma = []
    for event in wmma_window:
        if event["opcode"].startswith(("v_wmma", "v_mfma")):
            wmma_seen += 1
        elif wmma_seen and event["opcode"] == "s_waitcnt":
            waits_between_wmma.append(parse_lgkmcnt(event))

    return {
        "pre_wmma_ds_load_b64": len(pre_b64),
        "pre_wmma_ds_load_b64_bases": bases,
        "pre_wmma_waitcnt": len(pre_waits),
        "load_b64_immediately_before_final_wait": load_b64_before_final_wait,
        "final_pre_wmma_lgkmcnt": final_wait_lgkmcnt,
        "wmma_in_window": wmma_seen,
        "waitcnt_after_first_wmma": [value for value in waits_between_wmma if value is not None],
    }


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
    if opcode.startswith(("ds_read", "ds_load")):
        return "lds_read"
    if opcode.startswith(("ds_write", "ds_store")):
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


def summarize(name, lines, opcodes, resources=None, metadata=None):
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
            if key.startswith(INTERESTING_PREFIXES)
        },
        "top_opcodes": dict(opcode_counts.most_common(30)),
        "event_summary": summarize_isa_events(parse_isa_events(lines)),
    }


def normalize_resources(summary):
    resources = summary.get("resources", {})
    return {
        "sgpr": resources.get("SGPRs", resources.get("sgpr_count")),
        "vgpr": resources.get("VGPRs", resources.get("vgpr_count")),
        "lds_bytes": resources.get("LDS size", resources.get("group_segment_fixed_size")),
        "scratch_bytes": resources.get("Scratch size", resources.get("private_segment_fixed_size")),
        "sgpr_spills": resources.get("Spilled SGPRs", resources.get("sgpr_spill_count")),
        "vgpr_spills": resources.get("Spilled VGPRs", resources.get("vgpr_spill_count")),
        "wavefront_size": resources.get("wavefront_size"),
        "instructions": resources.get("Instructions", summary.get("instruction_count")),
    }


def compute_deltas(lhs_summary, rhs_summary):
    lhs_opcodes = lhs_summary.get("interesting_opcodes", {})
    rhs_opcodes = rhs_summary.get("interesting_opcodes", {})
    opcode_deltas = []
    for opcode in sorted(set(lhs_opcodes) | set(rhs_opcodes)):
        lhs_count = lhs_opcodes.get(opcode, 0)
        rhs_count = rhs_opcodes.get(opcode, 0)
        if lhs_count != rhs_count:
            opcode_deltas.append({
                "opcode": opcode,
                "lhs": lhs_count,
                "rhs": rhs_count,
                "rhs_minus_lhs": rhs_count - lhs_count,
            })

    lhs_resources = normalize_resources(lhs_summary)
    rhs_resources = normalize_resources(rhs_summary)
    resource_deltas = []
    for key in sorted(set(lhs_resources) | set(rhs_resources)):
        lhs_value = lhs_resources.get(key)
        rhs_value = rhs_resources.get(key)
        if lhs_value != rhs_value:
            delta = None
            if isinstance(lhs_value, int) and isinstance(rhs_value, int):
                delta = rhs_value - lhs_value
            resource_deltas.append({
                "resource": key,
                "lhs": lhs_value,
                "rhs": rhs_value,
                "rhs_minus_lhs": delta,
            })

    return {
        "resources": resource_deltas,
        "interesting_opcodes": opcode_deltas,
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
        event_summary = summary.get("event_summary", {})
        if event_summary:
            lines += ["### Event Summary", ""]
            pre_offsets = event_summary.get("pre_wmma_ds_load_b64_offsets", {})
            if pre_offsets:
                lines.append("Pre-WMMA `ds_load_b64` offsets:")
                lines.append("")
                for base, offsets in pre_offsets.items():
                    rendered = ", ".join(str(value) for value in offsets)
                    lines.append(f"- `{base}`: `{rendered}`")
                lines.append("")
            wmma_score = event_summary.get("wmma_score", {})
            if wmma_score:
                lines += ["First-WMMA schedule score:", "", "| Metric | Value |", "| --- | ---: |"]
                for score_key, score_value in wmma_score.items():
                    rendered = json.dumps(score_value) if isinstance(score_value, (list, dict)) else str(score_value)
                    lines.append(f"| `{score_key}` | `{rendered}` |")
                lines.append("")
            store_blocks = event_summary.get("store_blocks", [])
            if store_blocks:
                lines += ["Store basic blocks:", "", "| BB | Store Ops | Interesting Ops |", "| --- | ---: | --- |"]
                for block in store_blocks[:24]:
                    ops = ", ".join(f"{op}={count}" for op, count in block["opcodes"].items())
                    lines.append(f"| `{block['bb']}` | {block['store_ops']} | `{ops}` |")
                lines.append("")
            store_windows = event_summary.get("store_windows", [])
            if store_windows:
                lines += ["Store windows:", "", "| First Store Line | Interesting Ops |", "| ---: | --- |"]
                for window in store_windows[:12]:
                    ops = ", ".join(f"{op}={count}" for op, count in sorted(window["opcodes"].items()))
                    lines.append(f"| {window['first_store_line']} | `{ops}` |")
                lines.append("")
            wmma_window = event_summary.get("wmma_window", [])
            if wmma_window:
                lines.append("First WMMA window:")
                lines.append("")
                for event in wmma_window[:24]:
                    lines.append(f"- line {event['line']}: `{event['text']}`")
                lines.append("")

    lines += ["## Delta", "", f"| Field | {lhs_label} | {rhs_label} |", "| --- | ---: | ---: |"]
    all_classes = sorted(set(payload[lhs_key]["classes"]) | set(payload[rhs_key]["classes"]))
    for cls in all_classes:
        lines.append(f"| class `{cls}` | {payload[lhs_key]['classes'].get(cls, 0)} | {payload[rhs_key]['classes'].get(cls, 0)} |")
    lines += ["", "### Normalized Resources", "", f"| Resource | {lhs_label} | {rhs_label} | Delta |", "| --- | ---: | ---: | ---: |"]
    for item in payload.get("delta", {}).get("resources", []):
        lines.append(f"| `{item['resource']}` | `{item['lhs']}` | `{item['rhs']}` | `{item['rhs_minus_lhs']}` |")
    lines += ["", "### Interesting Opcodes", "", f"| Opcode | {lhs_label} | {rhs_label} | Delta |", "| --- | ---: | ---: | ---: |"]
    for item in payload.get("delta", {}).get("interesting_opcodes", []):
        lines.append(f"| `{item['opcode']}` | {item['lhs']} | {item['rhs']} | {item['rhs_minus_lhs']} |")
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

    lhs_summary = summarize(args.radv_isa.name, radv_lines, radv_opcodes, radv_resources, radv_metadata)
    rhs_summary = summarize(rhs_name, rhs_lines, rhs_opcodes, rhs_resources)

    payload = {
        "lhs_key": "radv",
        "rhs_key": "rhs",
        "lhs_label": "RADV",
        "rhs_label": rhs_label,
        "radv": lhs_summary,
        "rhs": rhs_summary,
        "delta": compute_deltas(lhs_summary, rhs_summary),
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
