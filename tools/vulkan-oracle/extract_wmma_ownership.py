#!/usr/bin/env python3
import argparse
import collections
import json
import pathlib
import re


ISA_INSTRUCTION_RE = re.compile(r"^\s+([A-Za-z0-9_.]+)\s*(.*)$")
SYMBOL_RE = re.compile(r"^[0-9a-fA-F]+ <([^>]+)>:")
BB_RE = re.compile(r"^\s*(BB[0-9]+|\.L[A-Za-z0-9_.$]+):")
OFFSET_RE = re.compile(r"\boffset(?P<which>[0-9]*)[: ](?P<value>-?(?:0x[0-9a-fA-F]+|[0-9]+))")
VREG_RE = re.compile(r"\bv(?:\[(?P<lo>[0-9]+):(?P<hi>[0-9]+)\]|(?P<single>[0-9]+)\b)")
LGKMCNT_RE = re.compile(r"\blgkmcnt\((?P<value>[0-9]+)\)")


def read_lines(path):
    return pathlib.Path(path).read_text(encoding="utf-8", errors="replace").splitlines()


def strip_comment(line):
    for sep in ("//", ";"):
        if sep in line:
            line = line.split(sep, 1)[0]
    return line.rstrip()


def parse_int(value):
    return int(value, 0)


def extract_offset(rest):
    offsets = {}
    for match in OFFSET_RE.finditer(rest):
        key = "offset" if not match.group("which") else f"offset{match.group('which')}"
        offsets[key] = parse_int(match.group("value"))
    return offsets


def extract_symbol(lines, symbol):
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
        raise SystemExit(f"symbol substring not found: {symbol}")
    return result


def parse_events(lines):
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
        if not opcode.startswith(("v_wmma", "v_mfma", "ds_load", "ds_store", "s_waitcnt", "s_barrier", "buffer_store", "global_store", "flat_store")):
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


def parse_vreg(token):
    match = VREG_RE.search(token)
    if not match:
        return None
    if match.group("single") is not None:
        value = int(match.group("single"))
        return [value, value]
    return [int(match.group("lo")), int(match.group("hi"))]


def render_range(rng):
    if not rng:
        return None
    if rng[0] == rng[1]:
        return f"v{rng[0]}"
    return f"v[{rng[0]}:{rng[1]}]"


def range_width(rng):
    return rng[1] - rng[0] + 1 if rng else None


def ranges_touch(left, right):
    return left and right and left[1] + 1 == right[0]


def ranges_overlap(left, right):
    return left and right and not (left[1] < right[0] or right[1] < left[0])


def range_contains(outer, inner):
    return outer and inner and outer[0] <= inner[0] and inner[1] <= outer[1]


def parse_lgkmcnt(text):
    match = LGKMCNT_RE.search(text)
    return int(match.group("value")) if match else None


def summarize_ds_loads(events):
    loads = []
    for event in events:
        if event["opcode"] != "ds_load_b64" or len(event["operands"]) < 2:
            continue
        dest = parse_vreg(event["operands"][0])
        if not dest:
            continue
        address = event["operands"][1].split()[0]
        loads.append({
            "line": event["line"],
            "dest": dest,
            "dest_text": render_range(dest),
            "address": address,
            "offset": event["offsets"].get("offset", 0),
            "text": event["text"],
        })

    groups = []
    current = []
    for load in loads:
        if (
            current
            and ranges_touch(current[-1]["dest"], load["dest"])
            and load["line"] - current[-1]["line"] <= 2
            and load["offset"] - current[-1]["offset"] in (0, 8)
        ):
            current.append(load)
        else:
            if current:
                groups.append(current)
            current = [load]
    if current:
        groups.append(current)

    summaries = []
    for group in groups:
        dest_range = [group[0]["dest"][0], group[-1]["dest"][1]]
        summaries.append({
            "line_range": [group[0]["line"], group[-1]["line"]],
            "dest": dest_range,
            "dest_text": render_range(dest_range),
            "dwords64": len(group),
            "addresses": sorted(set(item["address"] for item in group)),
            "offsets": [item["offset"] for item in group],
            "loads": group,
        })
    return loads, summaries


def find_defining_load_group(groups, rng, before_line):
    if not rng:
        return None
    candidates = [
        group for group in groups
        if group["line_range"][1] < before_line and range_contains(group["dest"], rng)
    ]
    if not candidates:
        candidates = [
            group for group in groups
            if group["line_range"][1] < before_line and ranges_overlap(group["dest"], rng)
        ]
    if not candidates:
        return None
    return max(candidates, key=lambda group: group["line_range"][1])


def find_preceding_wait(events, event_index):
    for index in range(event_index - 1, -1, -1):
        event = events[index]
        if event["opcode"] == "s_waitcnt":
            return {
                "line": event["line"],
                "lgkmcnt": parse_lgkmcnt(event["text"]),
                "text": event["text"],
            }
        if event["opcode"].startswith(("v_wmma", "v_mfma")):
            return None
    return None


def summarize_wmma(events, load_groups):
    wmma_events = []
    operand_uses = {
        "dst": collections.Counter(),
        "a": collections.Counter(),
        "b": collections.Counter(),
        "c": collections.Counter(),
    }
    load_group_uses = collections.Counter()
    wait_ladder = []

    for index, event in enumerate(events):
        if not event["opcode"].startswith(("v_wmma", "v_mfma")):
            continue
        operands = event["operands"]
        if len(operands) < 4:
            continue
        dst = parse_vreg(operands[0])
        a = parse_vreg(operands[1])
        b = parse_vreg(operands[2])
        c = parse_vreg(operands[3])
        wait = find_preceding_wait(events, index)
        if wait:
            wait_ladder.append(wait["lgkmcnt"])

        def loaded_by(rng):
            group = find_defining_load_group(load_groups, rng, event["line"])
            if not group:
                return None
            load_group_uses[group["dest_text"]] += 1
            return {
                "dest": group["dest_text"],
                "line_range": group["line_range"],
                "offsets": group["offsets"],
                "addresses": group["addresses"],
            }

        item = {
            "line": event["line"],
            "opcode": event["opcode"],
            "dst": render_range(dst),
            "a": render_range(a),
            "b": render_range(b),
            "c": render_range(c),
            "dst_width": range_width(dst),
            "a_width": range_width(a),
            "b_width": range_width(b),
            "c_width": range_width(c),
            "c_equals_dst": dst == c,
            "preceding_wait": wait,
            "a_loaded_by": loaded_by(a),
            "b_loaded_by": loaded_by(b),
            "text": event["text"],
        }
        for key, rng in (("dst", dst), ("a", a), ("b", b), ("c", c)):
            rendered = render_range(rng)
            if rendered:
                operand_uses[key][rendered] += 1
        wmma_events.append(item)

    return {
        "count": len(wmma_events),
        "events": wmma_events,
        "unique_operands": {
            key: dict(counter.most_common())
            for key, counter in operand_uses.items()
        },
        "load_group_uses": dict(load_group_uses.most_common()),
        "wait_ladder": wait_ladder,
    }


def summarize_store_surface(events):
    counts = collections.Counter(event["opcode"] for event in events)
    stores = []
    for event in events:
        if not event["opcode"].startswith(("ds_store", "buffer_store", "global_store", "flat_store")):
            continue
        stores.append({
            "line": event["line"],
            "opcode": event["opcode"],
            "operands": event["operands"],
            "offsets": event["offsets"],
            "text": event["text"],
        })
    return {
        "opcode_counts": {
            opcode: counts[opcode]
            for opcode in sorted(counts)
            if opcode.startswith(("ds_store", "buffer_store", "global_store", "flat_store"))
        },
        "first_stores": stores[:40],
    }


def summarize_isa(path, symbol=None, label=None):
    lines = extract_symbol(read_lines(path), symbol)
    events = parse_events(lines)
    loads, load_groups = summarize_ds_loads(events)
    wmma = summarize_wmma(events, load_groups)
    opcode_counts = collections.Counter(event["opcode"] for event in events)
    return {
        "label": label or pathlib.Path(path).name,
        "path": str(path),
        "symbol": symbol,
        "event_count": len(events),
        "opcode_counts": dict(sorted(opcode_counts.items())),
        "ds_load_b64_count": len(loads),
        "ds_load_b64_groups": [
            {
                "line_range": group["line_range"],
                "dest": group["dest_text"],
                "dwords64": group["dwords64"],
                "addresses": group["addresses"],
                "offsets": group["offsets"],
            }
            for group in load_groups
        ],
        "wmma": wmma,
        "store_surface": summarize_store_surface(events),
    }


def compare(lhs, rhs):
    def set_from(counter_dict):
        return set(counter_dict.keys())

    lhs_wmma = lhs["wmma"]
    rhs_wmma = rhs["wmma"]
    result = {
        "wmma_count": {
            "lhs": lhs_wmma["count"],
            "rhs": rhs_wmma["count"],
            "delta": rhs_wmma["count"] - lhs_wmma["count"],
        },
        "ds_load_b64_count": {
            "lhs": lhs["ds_load_b64_count"],
            "rhs": rhs["ds_load_b64_count"],
            "delta": rhs["ds_load_b64_count"] - lhs["ds_load_b64_count"],
        },
        "wait_ladder": {
            "lhs": lhs_wmma["wait_ladder"],
            "rhs": rhs_wmma["wait_ladder"],
        },
        "operand_bank_overlap": {},
    }
    for key in ("dst", "a", "b", "c"):
        lhs_set = set_from(lhs_wmma["unique_operands"].get(key, {}))
        rhs_set = set_from(rhs_wmma["unique_operands"].get(key, {}))
        result["operand_bank_overlap"][key] = {
            "lhs_unique": len(lhs_set),
            "rhs_unique": len(rhs_set),
            "common": sorted(lhs_set & rhs_set),
            "lhs_only": sorted(lhs_set - rhs_set),
            "rhs_only": sorted(rhs_set - lhs_set),
        }
    return result


def write_markdown(path, payload):
    lines = ["# WMMA Ownership Extract", ""]
    summaries = payload["summaries"]
    for summary in summaries:
        lines += [
            f"## {summary['label']}",
            "",
            f"- path: `{summary['path']}`",
            f"- symbol: `{summary.get('symbol')}`",
            f"- WMMA count: `{summary['wmma']['count']}`",
            f"- `ds_load_b64` count: `{summary['ds_load_b64_count']}`",
            "",
            "### Opcode Counts",
            "",
            "| Opcode | Count |",
            "| --- | ---: |",
        ]
        for opcode, count in summary["opcode_counts"].items():
            lines.append(f"| `{opcode}` | {count} |")
        lines += ["", "### Operand Banks", ""]
        for key in ("dst", "a", "b", "c"):
            lines += [f"#### {key.upper()}", "", "| Range | Uses |", "| --- | ---: |"]
            for operand, count in summary["wmma"]["unique_operands"].get(key, {}).items():
                lines.append(f"| `{operand}` | {count} |")
            lines.append("")
        lines += [
            "### WMMA Sequence",
            "",
            "| # | Line | Wait | Dst | A | B | C | A Load | B Load |",
            "| ---: | ---: | ---: | --- | --- | --- | --- | --- | --- |",
        ]
        for index, event in enumerate(summary["wmma"]["events"], 1):
            wait = event["preceding_wait"]["lgkmcnt"] if event.get("preceding_wait") else ""
            a_load = event["a_loaded_by"]["dest"] if event.get("a_loaded_by") else ""
            b_load = event["b_loaded_by"]["dest"] if event.get("b_loaded_by") else ""
            lines.append(
                f"| {index} | {event['line']} | `{wait}` | `{event['dst']}` "
                f"| `{event['a']}` | `{event['b']}` | `{event['c']}` "
                f"| `{a_load}` | `{b_load}` |"
            )
        lines += [
            "",
            "### LDS Load Groups",
            "",
            "| Lines | Dest | 64-bit Loads | Addresses | Offsets |",
            "| --- | --- | ---: | --- | --- |",
        ]
        for group in summary["ds_load_b64_groups"][:80]:
            offsets = ",".join(str(value) for value in group["offsets"])
            addresses = ",".join(group["addresses"])
            lines.append(
                f"| `{group['line_range'][0]}-{group['line_range'][1]}` "
                f"| `{group['dest']}` | {group['dwords64']} "
                f"| `{addresses}` | `{offsets}` |"
            )
        lines += [
            "",
            "### Store Surface",
            "",
            "| Opcode | Count |",
            "| --- | ---: |",
        ]
        for opcode, count in summary["store_surface"]["opcode_counts"].items():
            lines.append(f"| `{opcode}` | {count} |")
        lines.append("")

    if payload.get("comparison"):
        lines += ["## Comparison", ""]
        comparison = payload["comparison"]
        lines += [
            "| Metric | LHS | RHS | Delta |",
            "| --- | ---: | ---: | ---: |",
            f"| WMMA count | {comparison['wmma_count']['lhs']} | {comparison['wmma_count']['rhs']} | {comparison['wmma_count']['delta']} |",
            f"| `ds_load_b64` count | {comparison['ds_load_b64_count']['lhs']} | {comparison['ds_load_b64_count']['rhs']} | {comparison['ds_load_b64_count']['delta']} |",
            "",
            "### Operand Bank Overlap",
            "",
            "| Operand | LHS Unique | RHS Unique | Common | LHS Only | RHS Only |",
            "| --- | ---: | ---: | --- | --- | --- |",
        ]
        for key, item in comparison["operand_bank_overlap"].items():
            common = ", ".join(f"`{value}`" for value in item["common"])
            lhs_only = ", ".join(f"`{value}`" for value in item["lhs_only"])
            rhs_only = ", ".join(f"`{value}`" for value in item["rhs_only"])
            lines.append(
                f"| `{key}` | {item['lhs_unique']} | {item['rhs_unique']} "
                f"| {common} | {lhs_only} | {rhs_only} |"
            )
        lines += [
            "",
            "### Wait Ladders",
            "",
            f"- LHS: `{comparison['wait_ladder']['lhs']}`",
            f"- RHS: `{comparison['wait_ladder']['rhs']}`",
            "",
        ]

    pathlib.Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Extract WMMA operand ownership from AMDGCN ISA text.")
    parser.add_argument("--isa", required=True, type=pathlib.Path, help="Primary AMDGCN ISA or llvm-objdump text.")
    parser.add_argument("--label", help="Display label for the primary ISA.")
    parser.add_argument("--symbol", help="Optional llvm-objdump symbol substring for the primary ISA.")
    parser.add_argument("--compare-isa", type=pathlib.Path, help="Optional second ISA file to compare.")
    parser.add_argument("--compare-label", help="Display label for the comparison ISA.")
    parser.add_argument("--compare-symbol", help="Optional llvm-objdump symbol substring for the comparison ISA.")
    parser.add_argument("--out-json", type=pathlib.Path)
    parser.add_argument("--out-md", type=pathlib.Path)
    return parser.parse_args()


def main():
    args = parse_args()
    lhs = summarize_isa(args.isa, args.symbol, args.label)
    summaries = [lhs]
    comparison = None
    if args.compare_isa:
        rhs = summarize_isa(args.compare_isa, args.compare_symbol, args.compare_label)
        summaries.append(rhs)
        comparison = compare(lhs, rhs)

    payload = {
        "summaries": summaries,
        "comparison": comparison,
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
