#!/usr/bin/env python3
"""Create a Q6_K p33 medium-route static oracle screen.

This is a pre-timing gate for gfx1151 Q6_K dense prompt candidates that claim
to clone the Vulkan medium aligned route. It compares a CMake/Ninja-built HIP
HSACO against the RADV p33 oracle and records which exact schedule facts still
miss: WMMA count, LDS load shape, halfword LDS surface, store surface, waits,
resources, and optional compact f16 accumulator ownership.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys


CONTRACT_ARGS = [
    "--match-resource",
    "lds_bytes",
    "--match-opcode",
    "v_wmma_f16_16x16x16_f16",
    "--match-opcode",
    "ds_load_b64",
    "--match-opcode",
    "ds_load_u16_d16",
    "--match-opcode",
    "ds_store_b16",
    "--match-opcode",
    "buffer_store_b32",
    "--match-opcode",
    "s_barrier",
    "--match-wmma-score",
    "pre_wmma_ds_load_b64",
    "--match-wmma-score",
    "load_b64_immediately_before_final_wait",
    "--match-wmma-score",
    "final_pre_wmma_lgkmcnt",
    "--match-wmma-score",
    "wmma_in_window",
    "--match-store-score",
    "buffer_store_ops",
    "--match-store-score",
    "lds_store_ops",
    "--match-store-score",
    "vmem_store_ops",
    "--require-zero-spills",
]


def default_tool(name: str) -> str:
    candidates: list[pathlib.Path | str] = []
    for env_name in ("ROCM_PATH", "GGML_HRX_ROCM_PATH"):
        value = os.environ.get(env_name)
        if value:
            root = pathlib.Path(value)
            candidates.append(root / "llvm/bin" / name)
            candidates.append(root / "bin" / name)
    candidates.extend(
        [
            pathlib.Path("rocm/llvm/bin") / name,
            pathlib.Path("rocm/bin") / name,
            name,
        ]
    )
    for candidate in candidates:
        text = str(candidate)
        if pathlib.Path(text).exists() or shutil.which(text):
            return text
    return name


def run(cmd: list[str], *, stdout_path: pathlib.Path | None = None, stderr_path: pathlib.Path | None = None) -> subprocess.CompletedProcess[str]:
    stdout = subprocess.PIPE if stdout_path is None else stdout_path.open("w", encoding="utf-8")
    stderr = subprocess.PIPE if stderr_path is None else stderr_path.open("w", encoding="utf-8")
    try:
        return subprocess.run(cmd, text=True, stdout=stdout, stderr=stderr, check=False)
    finally:
        if stdout_path is not None:
            stdout.close()
        if stderr_path is not None:
            stderr.close()


def load_json(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def normalized_resources(summary: dict) -> dict:
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


def compact_check(script_dir: pathlib.Path, args: argparse.Namespace, out_dir: pathlib.Path) -> dict | None:
    if not args.check_compact_accumulators:
        return None
    compact_dir = out_dir / "compact-accumulator"
    cmd = [
        sys.executable,
        str(script_dir / "run_wmma_compact_screen.py"),
        "--radv-isa",
        str(args.radv_isa),
        "--radv-label",
        "RADV Q6 medium oracle",
        "--hip-hsaco",
        str(args.hip_hsaco),
        "--hip-label",
        args.label,
        "--hip-symbol",
        args.hip_symbol,
        "--out-dir",
        str(compact_dir),
        "--llvm-objdump",
        args.llvm_objdump,
    ]
    proc = run(cmd, stdout_path=out_dir / "compact-screen.stdout.json", stderr_path=out_dir / "compact-screen.stderr.log")
    summary_path = compact_dir / "summary.json"
    summary = load_json(summary_path) if summary_path.exists() else {}
    summary["screen_returncode"] = proc.returncode
    return summary


def reuse_histogram(summary: dict, operand: str) -> list[int]:
    uses = summary.get("wmma", {}).get("unique_operands", {}).get(operand, {})
    return sorted((int(count) for count in uses.values()), reverse=True)


def ownership_contract(compact: dict | None) -> dict | None:
    if compact is None:
        return None
    ownership_path = compact.get("artifacts", {}).get("ownership_json")
    if not ownership_path:
        return {
            "passed": False,
            "checks": [
                {
                    "kind": "ownership_artifact",
                    "name": "ownership_json",
                    "lhs": None,
                    "rhs": None,
                    "rule": "artifact must exist",
                    "passed": False,
                }
            ],
        }
    ownership_file = pathlib.Path(ownership_path)
    if not ownership_file.exists():
        return {
            "passed": False,
            "checks": [
                {
                    "kind": "ownership_artifact",
                    "name": "ownership_json",
                    "lhs": None,
                    "rhs": str(ownership_file),
                    "rule": "artifact must exist",
                    "passed": False,
                }
            ],
        }

    ownership = load_json(ownership_file)
    summaries = ownership.get("summaries", [])
    comparison = ownership.get("comparison", {})
    checks = []

    if len(summaries) >= 2:
        lhs_summary, rhs_summary = summaries[0], summaries[1]
        for operand in ("dst", "a", "b", "c"):
            lhs_hist = reuse_histogram(lhs_summary, operand)
            rhs_hist = reuse_histogram(rhs_summary, operand)
            checks.append(
                {
                    "kind": "wmma_operand_reuse",
                    "name": operand,
                    "lhs": lhs_hist,
                    "rhs": rhs_hist,
                    "rule": "rhs reuse histogram == lhs reuse histogram",
                    "passed": rhs_hist == lhs_hist,
                }
            )

    bank_overlap = comparison.get("operand_bank_overlap", {})
    for operand in ("dst", "a", "b", "c"):
        item = bank_overlap.get(operand, {})
        lhs_unique = item.get("lhs_unique")
        rhs_unique = item.get("rhs_unique")
        checks.append(
            {
                "kind": "wmma_operand_unique_banks",
                "name": operand,
                "lhs": lhs_unique,
                "rhs": rhs_unique,
                "rule": "rhs unique bank count == lhs unique bank count",
                "passed": rhs_unique == lhs_unique,
            }
        )

    operand_sequences = comparison.get("operand_sequences", {})
    for operand in ("dst", "a", "b", "c", "a_load", "b_load"):
        item = operand_sequences.get(operand, {})
        lhs_sequence = item.get("lhs", [])
        rhs_sequence = item.get("rhs", [])
        lhs_pattern = item.get("lhs_pattern", [])
        rhs_pattern = item.get("rhs_pattern", [])
        checks.append(
            {
                "kind": "wmma_operand_sequence_pattern",
                "name": operand,
                "lhs": lhs_pattern,
                "rhs": rhs_pattern,
                "lhs_exact": lhs_sequence,
                "rhs_exact": rhs_sequence,
                "rule": "rhs operand/load first-use pattern == lhs operand/load first-use pattern",
                "passed": rhs_pattern == lhs_pattern,
            }
        )

    wait_ladder = comparison.get("wait_ladder", {})
    lhs_wait = wait_ladder.get("lhs")
    rhs_wait = wait_ladder.get("rhs")
    checks.append(
        {
            "kind": "wmma_wait_ladder",
            "name": "lgkmcnt_sequence",
            "lhs": lhs_wait,
            "rhs": rhs_wait,
            "rule": "rhs wait ladder == lhs wait ladder",
            "passed": rhs_wait == lhs_wait,
        }
    )

    return {
        "passed": all(check["passed"] for check in checks),
        "ownership_json": str(ownership_file),
        "checks": checks,
    }


def write_markdown(path: pathlib.Path, payload: dict) -> None:
    compare = payload["compare"]
    contract = payload["contract"]
    radv = compare["radv"]
    rhs = compare["rhs"]
    radv_resources = normalized_resources(radv)
    rhs_resources = normalized_resources(rhs)
    radv_opcodes = radv.get("interesting_opcodes", {})
    rhs_opcodes = rhs.get("interesting_opcodes", {})

    lines = [
        "# Q6_K Medium Route Static Contract Screen",
        "",
        f"- label: `{payload['label']}`",
        f"- RADV ISA: `{payload['radv_isa']}`",
        f"- RADV stats: `{payload['radv_stats']}`",
        f"- HIP HSACO: `{payload['hip_hsaco']}`",
        f"- HIP symbol: `{payload['hip_symbol']}`",
        f"- exact contract passed: `{payload['passes_exact_contract']}`",
    ]
    if payload.get("compact_check") is not None:
        lines.append(
            f"- compact f16 accumulator screen passed: `{payload['compact_check'].get('passes_compact_f16_accumulator_screen')}`"
        )
    if payload.get("ownership_contract") is not None:
        lines.append(
            f"- WMMA operand-topology contract passed: `{payload['ownership_contract'].get('passed')}`"
        )
    lines += [
        "",
        "## Resource Facts",
        "",
        "| Resource | RADV | HIP |",
        "| --- | ---: | ---: |",
    ]
    for key in ("wavefront_size", "sgpr", "vgpr", "lds_bytes", "scratch_bytes", "sgpr_spills", "vgpr_spills", "instructions"):
        lines.append(f"| `{key}` | `{radv_resources.get(key)}` | `{rhs_resources.get(key)}` |")

    lines += [
        "",
        "## Key Opcode Facts",
        "",
        "| Opcode | RADV | HIP |",
        "| --- | ---: | ---: |",
    ]
    for opcode in (
        "v_wmma_f16_16x16x16_f16",
        "ds_load_b64",
        "ds_load_u16_d16",
        "ds_store_b16",
        "buffer_store_b32",
        "global_store_b32",
        "s_barrier",
        "s_waitcnt",
    ):
        lines.append(f"| `{opcode}` | {radv_opcodes.get(opcode, 0)} | {rhs_opcodes.get(opcode, 0)} |")

    lines += [
        "",
        "## Contract Checks",
        "",
        "| Kind | Name | RADV | HIP | Rule | Pass |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    for check in contract.get("checks", []):
        lines.append(
            f"| `{check['kind']}` | `{check['name']}` | `{check.get('lhs')}` | "
            f"`{check.get('rhs')}` | `{check['rule']}` | `{check['passed']}` |"
        )

    if payload.get("ownership_contract") is not None:
        lines += [
            "",
            "## WMMA Operand-Topology Checks",
            "",
            f"- ownership artifact: `{payload['ownership_contract'].get('ownership_json')}`",
            "",
            "| Kind | Name | RADV | HIP | Rule | Pass |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for check in payload["ownership_contract"].get("checks", []):
            lines.append(
                f"| `{check['kind']}` | `{check['name']}` | "
                f"`{json.dumps(check.get('lhs'), sort_keys=True)}` | "
                f"`{json.dumps(check.get('rhs'), sort_keys=True)}` | "
                f"`{check['rule']}` | `{check['passed']}` |"
            )

    lines += [
        "",
        "## Interpretation",
        "",
    ]
    if payload["passes_full_static_contract"]:
        lines.append(
            "The HIP candidate matches the source-controlled exact static contract, including requested ownership screens. "
            "This is still not promotion evidence; run focused CPU-reference, route-trace, and same-runner p33/p512/p513 timing gates next."
        )
    else:
        lines.append(
            "The HIP candidate misses at least one exact RADV medium-route contract fact. "
            "Treat this as a schedule-screen rejection for exact Vulkan-clone claims; only continue to timing if the candidate is explicitly framed as a measured deviation."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radv-isa", required=True, type=pathlib.Path)
    parser.add_argument("--radv-stats", required=True, type=pathlib.Path)
    parser.add_argument("--hip-hsaco", required=True, type=pathlib.Path)
    parser.add_argument("--hip-symbol", required=True)
    parser.add_argument("--label", default="HIP Q6 medium candidate")
    parser.add_argument("--out-dir", required=True, type=pathlib.Path)
    parser.add_argument("--llvm-objdump", default=default_tool("llvm-objdump"))
    parser.add_argument("--llvm-readelf", default=default_tool("llvm-readelf"))
    parser.add_argument(
        "--check-compact-accumulators",
        action="store_true",
        help="Also run the compact dst/C f16 WMMA ownership screen and include it in the artifact.",
    )
    parser.add_argument(
        "--fail-on-contract-miss",
        action="store_true",
        help="Return non-zero when the exact static contract or requested compact screen fails.",
    )
    args = parser.parse_args()

    script_dir = pathlib.Path(__file__).resolve().parent
    args.out_dir.mkdir(parents=True, exist_ok=True)

    compare_json = args.out_dir / "isa-compare.json"
    compare_md = args.out_dir / "isa-compare.md"
    contract_json = args.out_dir / "contract-check.json"

    compare_cmd = [
        sys.executable,
        str(script_dir / "compare_amdgcn_isa.py"),
        "--radv-isa",
        str(args.radv_isa),
        "--radv-stats",
        str(args.radv_stats),
        "--hsaco",
        str(args.hip_hsaco),
        "--hsaco-symbol",
        args.hip_symbol,
        "--llvm-objdump",
        args.llvm_objdump,
        "--llvm-readelf",
        args.llvm_readelf,
        "--out-json",
        str(compare_json),
        "--out-md",
        str(compare_md),
    ]
    compare_proc = run(compare_cmd, stdout_path=args.out_dir / "compare.stdout.log", stderr_path=args.out_dir / "compare.stderr.log")
    if compare_proc.returncode != 0:
        raise SystemExit(compare_proc.returncode)

    contract_cmd = [
        sys.executable,
        str(script_dir / "check_isa_contract.py"),
        "--compare-json",
        str(compare_json),
        "--out-json",
        str(contract_json),
        *CONTRACT_ARGS,
    ]
    contract_proc = run(contract_cmd, stdout_path=args.out_dir / "contract-check.stdout.json", stderr_path=args.out_dir / "contract-check.stderr.log")
    compact = compact_check(script_dir, args, args.out_dir)

    compare = load_json(compare_json)
    contract = load_json(contract_json)
    passes_contract = contract_proc.returncode == 0
    passes_compact = True
    if compact is not None:
        passes_compact = bool(compact.get("passes_compact_f16_accumulator_screen"))
    ownership = ownership_contract(compact)
    passes_ownership = True if ownership is None else bool(ownership.get("passed"))
    passes_full_static = passes_contract and passes_compact and passes_ownership

    summary = {
        "schema": "q6-medium-static-contract-screen-v1",
        "label": args.label,
        "radv_isa": str(args.radv_isa),
        "radv_stats": str(args.radv_stats),
        "hip_hsaco": str(args.hip_hsaco),
        "hip_symbol": args.hip_symbol,
        "compare_command": compare_cmd,
        "contract_command": contract_cmd,
        "compare_returncode": compare_proc.returncode,
        "contract_returncode": contract_proc.returncode,
        "passes_exact_contract": passes_contract,
        "passes_full_static_contract": passes_full_static,
        "compact_check": compact,
        "ownership_contract": ownership,
        "artifacts": {
            "isa_compare_json": str(compare_json),
            "isa_compare_md": str(compare_md),
            "contract_check_json": str(contract_json),
            "contract_check_stdout": str(args.out_dir / "contract-check.stdout.json"),
            "summary_md": str(args.out_dir / "summary.md"),
        },
        "compare": compare,
        "contract": contract,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(args.out_dir / "summary.md", summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("compare", "contract")}, indent=2, sort_keys=True))

    if args.fail_on_contract_miss and not passes_full_static:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
