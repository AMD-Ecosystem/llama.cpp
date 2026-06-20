#!/usr/bin/env python3
import argparse
import json
import pathlib
import sys


def parse_limit(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"expected NAME=VALUE, got {value!r}")
    name, raw = value.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError(f"empty NAME in {value!r}")
    try:
        limit = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"VALUE must be an integer in {value!r}") from exc
    return name, limit


def parse_score_limit(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"expected SCORE_FIELD=VALUE, got {value!r}")
    name, raw = value.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError(f"empty SCORE_FIELD in {value!r}")
    try:
        limit = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"VALUE must be an integer in {value!r}") from exc
    return name, limit


def interesting_opcodes(summary):
    return summary.get("interesting_opcodes", {})


def event_score(summary, name):
    return summary.get("event_summary", {}).get(name, {})


def score_field(summary, score_name, field):
    score = event_score(summary, score_name)
    if not isinstance(score, dict) or field not in score:
        return None, False
    return score[field], True


def normalized_resources(summary):
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


def add_check(checks, kind, name, lhs, rhs, passed, rule):
    checks.append({
        "kind": kind,
        "name": name,
        "lhs": lhs,
        "rhs": rhs,
        "passed": passed,
        "rule": rule,
    })


def add_match_score_check(checks, score_name, field, lhs, rhs):
    lhs_value, lhs_present = score_field(lhs, score_name, field)
    rhs_value, rhs_present = score_field(rhs, score_name, field)
    passed = lhs_present and rhs_present and lhs_value == rhs_value
    add_check(checks, score_name, field, lhs_value, rhs_value, passed, "rhs == lhs")


def add_score_max_check(checks, score_name, field, limit, rhs):
    rhs_value, rhs_present = score_field(rhs, score_name, field)
    passed = rhs_present and isinstance(rhs_value, int) and rhs_value <= limit
    add_check(checks, score_name, field, None, rhs_value, passed, f"rhs <= {limit}")


def load_compare(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    lhs_key = data.get("lhs_key", "radv")
    rhs_key = data.get("rhs_key", "rhs")
    if lhs_key not in data or rhs_key not in data:
        raise SystemExit(f"{path}: comparison JSON is missing {lhs_key!r} or {rhs_key!r}")
    return data, data[lhs_key], data[rhs_key]


def main():
    parser = argparse.ArgumentParser(
        description="Check a RADV-vs-HSACO ISA comparison against an exact schedule contract.")
    parser.add_argument("--compare-json", required=True, type=pathlib.Path)
    parser.add_argument("--match-opcode", action="append", default=[],
                        help="Require RHS opcode count to exactly match the RADV/LHS count.")
    parser.add_argument("--match-resource", action="append", default=[],
                        help="Require normalized RHS resource to exactly match the RADV/LHS value.")
    parser.add_argument("--rhs-opcode-max", type=parse_limit, action="append", default=[],
                        metavar="OPCODE=N", help="Require RHS opcode count <= N.")
    parser.add_argument("--rhs-resource-max", type=parse_limit, action="append", default=[],
                        metavar="RESOURCE=N", help="Require normalized RHS resource <= N.")
    parser.add_argument("--match-wmma-score", action="append", default=[],
                        metavar="FIELD",
                        help="Require RHS event_summary.wmma_score FIELD to exactly match RADV/LHS.")
    parser.add_argument("--match-hot-score", action="append", default=[],
                        metavar="FIELD",
                        help="Require RHS event_summary.hot_op_score FIELD to exactly match RADV/LHS.")
    parser.add_argument("--match-store-score", action="append", default=[],
                        metavar="FIELD",
                        help="Require RHS event_summary.store_cluster_score FIELD to exactly match RADV/LHS.")
    parser.add_argument("--rhs-wmma-score-max", type=parse_score_limit, action="append", default=[],
                        metavar="FIELD=N", help="Require RHS event_summary.wmma_score FIELD <= N.")
    parser.add_argument("--rhs-hot-score-max", type=parse_score_limit, action="append", default=[],
                        metavar="FIELD=N", help="Require RHS event_summary.hot_op_score FIELD <= N.")
    parser.add_argument("--rhs-store-score-max", type=parse_score_limit, action="append", default=[],
                        metavar="FIELD=N", help="Require RHS event_summary.store_cluster_score FIELD <= N.")
    parser.add_argument("--require-zero-spills", action="store_true",
                        help="Require normalized RHS sgpr_spills and vgpr_spills to be zero.")
    parser.add_argument("--out-json", type=pathlib.Path)
    args = parser.parse_args()

    data, lhs, rhs = load_compare(args.compare_json)
    lhs_opcodes = interesting_opcodes(lhs)
    rhs_opcodes = interesting_opcodes(rhs)
    lhs_resources = normalized_resources(lhs)
    rhs_resources = normalized_resources(rhs)

    checks = []
    for opcode in args.match_opcode:
        lhs_count = lhs_opcodes.get(opcode, 0)
        rhs_count = rhs_opcodes.get(opcode, 0)
        add_check(checks, "opcode", opcode, lhs_count, rhs_count, lhs_count == rhs_count, "rhs == lhs")

    for resource in args.match_resource:
        lhs_value = lhs_resources.get(resource)
        rhs_value = rhs_resources.get(resource)
        add_check(checks, "resource", resource, lhs_value, rhs_value, lhs_value == rhs_value, "rhs == lhs")

    for opcode, limit in args.rhs_opcode_max:
        rhs_count = rhs_opcodes.get(opcode, 0)
        add_check(checks, "opcode", opcode, None, rhs_count, rhs_count <= limit, f"rhs <= {limit}")

    for resource, limit in args.rhs_resource_max:
        rhs_value = rhs_resources.get(resource)
        passed = isinstance(rhs_value, int) and rhs_value <= limit
        add_check(checks, "resource", resource, None, rhs_value, passed, f"rhs <= {limit}")

    for field in args.match_wmma_score:
        add_match_score_check(checks, "wmma_score", field, lhs, rhs)

    for field in args.match_hot_score:
        add_match_score_check(checks, "hot_op_score", field, lhs, rhs)

    for field in args.match_store_score:
        add_match_score_check(checks, "store_cluster_score", field, lhs, rhs)

    for field, limit in args.rhs_wmma_score_max:
        add_score_max_check(checks, "wmma_score", field, limit, rhs)

    for field, limit in args.rhs_hot_score_max:
        add_score_max_check(checks, "hot_op_score", field, limit, rhs)

    for field, limit in args.rhs_store_score_max:
        add_score_max_check(checks, "store_cluster_score", field, limit, rhs)

    if args.require_zero_spills:
        for resource in ("sgpr_spills", "vgpr_spills"):
            rhs_value = rhs_resources.get(resource)
            add_check(checks, "resource", resource, None, rhs_value, rhs_value == 0, "rhs == 0")

    passed = all(check["passed"] for check in checks)
    payload = {
        "compare_json": str(args.compare_json),
        "lhs_name": lhs.get("name"),
        "rhs_name": rhs.get("name"),
        "passed": passed,
        "checks": checks,
    }

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
