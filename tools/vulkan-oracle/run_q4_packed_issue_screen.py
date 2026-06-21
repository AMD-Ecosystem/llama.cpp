#!/usr/bin/env python3
"""Create Q4_K packed-route static oracle screens for gfx1151.

This wraps the lower-level Vulkan-oracle ISA tools for the two Q4_K regimes
that must stay separate on gfx1151:

* large production-width p512/p513 routes, compared with RADV aligned_l
* narrow p33 routes, compared with RADV aligned_m

The script does not compile kernels. It records the CMake/Ninja-built HSACOs
that it inspected so route work can keep HIP compilation in the build system.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys


LARGE_ORACLE = pathlib.Path(
    "cache/hrxv1/gfx1151/vulkan-oracle-llama31-8b-q4km-p512-fa1-20260617-195212"
)
NARROW_ORACLE = pathlib.Path(
    "cache/hrxv1/gfx1151/vulkan-oracle-llama31-8b-q4km-p33-fa1-20260617-200738"
)
LARGE_ISA = "matmul_q4_k_f32_f16acc_aligned_l__main__5666175250529efb.amdgcn.txt"
NARROW_ISA = "matmul_q4_k_f32_f16acc_aligned_m__main__5666175250529efb.amdgcn.txt"
LARGE_STATS = "matmul_q4_k_f32_f16acc_aligned_l__main__5666175250529efb.stats.txt"
NARROW_STATS = "matmul_q4_k_f32_f16acc_aligned_m__main__5666175250529efb.stats.txt"


def find_workspace() -> pathlib.Path:
    explicit = os.environ.get("LLAMACPP_DEVWS")
    if explicit:
        return pathlib.Path(explicit).resolve()
    here = pathlib.Path.cwd().resolve()
    for candidate in (here, *here.parents):
        if (candidate / "sources/llama.cpp").exists() and (candidate / "cache").exists():
            return candidate
    raise SystemExit("could not infer workspace root; set LLAMACPP_DEVWS")


def default_tool(name: str) -> pathlib.Path:
    for env_name in ("ROCM_PATH", "GGML_HRX_ROCM_PATH"):
        root = os.environ.get(env_name)
        if root:
            for rel in (f"llvm/bin/{name}", f"bin/{name}"):
                path = pathlib.Path(root) / rel
                if path.exists():
                    return path
    for rel in (f"rocm/llvm/bin/{name}", f"rocm/bin/{name}"):
        path = pathlib.Path(rel)
        if path.exists():
            return path
    resolved = shutil.which(name)
    return pathlib.Path(resolved) if resolved else pathlib.Path(name)


def run(argv: list[str], *, stdout: pathlib.Path | None = None, stderr: pathlib.Path | None = None) -> subprocess.CompletedProcess:
    stdout_handle = stdout.open("w", encoding="utf-8") if stdout else subprocess.PIPE
    stderr_handle = stderr.open("w", encoding="utf-8") if stderr else subprocess.PIPE
    try:
        proc = subprocess.run(argv, text=True, stdout=stdout_handle, stderr=stderr_handle, check=False)
    finally:
        if stdout:
            stdout_handle.close()
        if stderr:
            stderr_handle.close()
    if proc.returncode != 0:
        rendered = " ".join(argv)
        detail = "" if stderr else (proc.stderr or "")
        raise SystemExit(f"command failed ({proc.returncode}): {rendered}\n{detail}")
    return proc


def require_file(path: pathlib.Path) -> pathlib.Path:
    if not path.exists():
        raise SystemExit(f"required file not found: {path}")
    return path


def screen(
    *,
    label: str,
    oracle_dir: pathlib.Path,
    isa_name: str,
    stats_name: str,
    hsaco_dir: pathlib.Path,
    patterns: list[str],
    reference_hsaco: str,
    reference_symbol: str,
    out_dir: pathlib.Path,
    tool_dir: pathlib.Path,
    llvm_objdump: pathlib.Path,
    llvm_readelf: pathlib.Path,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    isa_path = require_file(oracle_dir / "radv/isa" / isa_name)
    stats_path = require_file(oracle_dir / "radv/stats" / stats_name)
    reference_hsaco_path = require_file(hsaco_dir / reference_hsaco)

    compare_json = out_dir / "reference-compare.json"
    compare_md = out_dir / "reference-compare.md"
    compare_cmd = [
        sys.executable,
        str(tool_dir / "compare_amdgcn_isa.py"),
        "--radv-isa",
        str(isa_path),
        "--radv-stats",
        str(stats_path),
        "--hsaco",
        str(reference_hsaco_path),
        "--hsaco-symbol",
        reference_symbol,
        "--llvm-objdump",
        str(llvm_objdump),
        "--llvm-readelf",
        str(llvm_readelf),
        "--out-json",
        str(compare_json),
        "--out-md",
        str(compare_md),
    ]
    run(compare_cmd, stdout=out_dir / "reference-compare.stdout.log", stderr=out_dir / "reference-compare.stderr.log")

    family_json = out_dir / "family-summary.json"
    family_md = out_dir / "family-summary.md"
    family_cmd = [
        sys.executable,
        str(tool_dir / "summarize_hsaco_family.py"),
        "--hsaco-dir",
        str(hsaco_dir),
        "--llvm-objdump",
        str(llvm_objdump),
        "--llvm-readelf",
        str(llvm_readelf),
        "--reference-compare-json",
        str(compare_json),
        "--sort",
        "total-gap",
        "--out-json",
        str(family_json),
        "--out-md",
        str(family_md),
    ]
    for pattern in patterns:
        family_cmd += ["--glob", pattern]
    run(family_cmd, stdout=out_dir / "family-summary.stdout.json", stderr=out_dir / "family-summary.stderr.log")

    family = json.loads(family_json.read_text(encoding="utf-8"))
    rows = family.get("rows", [])
    top_rows = []
    for row in rows[:8]:
        res = row.get("resources", {})
        hot = row.get("hot_op_score", {})
        delta = row.get("reference_delta", {})
        top_rows.append(
            {
                "name": row.get("name"),
                "static_contract_pass": row.get("static_contract_pass"),
                "contract_failures": row.get("contract_failures", []),
                "wavefront_size": res.get("wavefront_size"),
                "vgpr": res.get("vgpr"),
                "sgpr": res.get("sgpr"),
                "lds_bytes": res.get("lds_bytes"),
                "scratch_bytes": res.get("scratch_bytes"),
                "vgpr_spills": res.get("vgpr_spills"),
                "v_dot": row.get("v_dot"),
                "v_wmma": row.get("v_wmma"),
                "pre_hot_loads": (hot.get("pre_hot_lds_load") or 0) + (hot.get("pre_hot_vmem_load") or 0),
                "immediate_loads": hot.get("load_like_immediately_before_final_wait"),
                "final_lgkmcnt": hot.get("final_pre_hot_lgkmcnt"),
                "hot_gap": delta.get("hot_gap"),
                "store_gap": delta.get("store_gap"),
                "total_gap": delta.get("total_gap"),
            }
        )

    summary = {
        "label": label,
        "oracle_dir": str(oracle_dir),
        "radv_isa": str(isa_path),
        "radv_stats": str(stats_path),
        "hsaco_dir": str(hsaco_dir),
        "patterns": patterns,
        "reference_hsaco": str(reference_hsaco_path),
        "reference_symbol": reference_symbol,
        "artifacts": {
            "reference_compare_json": str(compare_json),
            "reference_compare_md": str(compare_md),
            "family_summary_json": str(family_json),
            "family_summary_md": str(family_md),
        },
        "top_static_rows": top_rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def write_markdown(path: pathlib.Path, payload: dict) -> None:
    lines = ["# Q4_K Packed Issue Screen", ""]
    lines += [
        "This artifact compares current CMake/Ninja-built HIP packed Q4_K HSACOs",
        "against same-machine RADV Q4_K oracle contracts. It is a static schedule",
        "screen, not promotion evidence by itself.",
        "",
    ]
    for section in payload["screens"]:
        lines += [f"## {section['label']}", ""]
        lines += [
            f"- oracle: `{section['oracle_dir']}`",
            f"- RADV ISA: `{section['radv_isa']}`",
            f"- reference HSACO: `{section['reference_hsaco']}`",
            f"- family summary: `{section['artifacts']['family_summary_md']}`",
            "",
            "| HSACO | Pass | Wave | VGPR | LDS | Scratch | v_dot | v_wmma | Pre-Hot Loads | Immediate Loads | Final lgkmcnt | Hot Gap | Store Gap | Total Gap |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in section["top_static_rows"]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{row['name']}`",
                        "`yes`" if row["static_contract_pass"] else "`no`",
                        f"`{row['wavefront_size']}`",
                        f"`{row['vgpr']}`",
                        f"`{row['lds_bytes']}`",
                        f"`{row['scratch_bytes']}`",
                        f"`{row['v_dot']}`",
                        f"`{row['v_wmma']}`",
                        f"`{row['pre_hot_loads']}`",
                        f"`{row['immediate_loads']}`",
                        f"`{row['final_lgkmcnt']}`",
                        f"`{row['hot_gap']}`",
                        f"`{row['store_gap']}`",
                        f"`{row['total_gap']}`",
                    ]
                )
                + " |"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    workspace = find_workspace()
    source_root = workspace / "sources/llama.cpp"
    tool_dir = pathlib.Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run static Q4_K packed-route RADV issue screens.")
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument(
        "--hsaco-dir",
        type=pathlib.Path,
        default=workspace / "build/hrx-v1-catalog-gfx1151/ggml/src/ggml-hrx/generated/hsaco/gfx1151",
    )
    parser.add_argument("--large-oracle-dir", type=pathlib.Path, default=workspace / LARGE_ORACLE)
    parser.add_argument("--narrow-oracle-dir", type=pathlib.Path, default=workspace / NARROW_ORACLE)
    parser.add_argument("--llvm-objdump", type=pathlib.Path, default=default_tool("llvm-objdump"))
    parser.add_argument("--llvm-readelf", type=pathlib.Path, default=default_tool("llvm-readelf"))
    parser.add_argument("--skip-large", action="store_true")
    parser.add_argument("--skip-narrow", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    command = {
        "argv": sys.argv,
        "cwd": str(pathlib.Path.cwd()),
        "workspace": str(workspace),
        "source_root": str(source_root),
        "llvm_objdump": str(args.llvm_objdump),
        "llvm_readelf": str(args.llvm_readelf),
        "hsaco_dir": str(args.hsaco_dir),
    }
    (args.out_dir / "command.json").write_text(json.dumps(command, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    screens = []
    if not args.skip_large:
        screens.append(
            screen(
                label="large-p512-p513",
                oracle_dir=args.large_oracle_dir,
                isa_name=LARGE_ISA,
                stats_name=LARGE_STATS,
                hsaco_dir=args.hsaco_dir,
                patterns=[
                    "mul_mat_vec_q4_k_q8_1_x4_mmql128*.hsaco",
                ],
                reference_hsaco="mul_mat_vec_q4_k_q8_1_x4_mmql128_bquad.hsaco",
                reference_symbol="hrx_mul_mat_vec_q4_k_q8_1_x4_mmql128x128_bquad_wg256_f32",
                out_dir=args.out_dir / "large",
                tool_dir=tool_dir,
                llvm_objdump=args.llvm_objdump,
                llvm_readelf=args.llvm_readelf,
            )
        )
    if not args.skip_narrow:
        screens.append(
            screen(
                label="narrow-p33",
                oracle_dir=args.narrow_oracle_dir,
                isa_name=NARROW_ISA,
                stats_name=NARROW_STATS,
                hsaco_dir=args.hsaco_dir,
                patterns=[
                    "mul_mat_vec_q4_k_q8_1_x4_mmql64_bk2*.hsaco",
                ],
                reference_hsaco="mul_mat_vec_q4_k_q8_1_x4_mmql64_bk2_bquad.hsaco",
                reference_symbol="hrx_mul_mat_vec_q4_k_q8_1_x4_mmql64x64_bk2_bquad_wg256_f32",
                out_dir=args.out_dir / "narrow",
                tool_dir=tool_dir,
                llvm_objdump=args.llvm_objdump,
                llvm_readelf=args.llvm_readelf,
            )
        )

    payload = {"command": command, "screens": screens}
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(args.out_dir / "summary.md", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
