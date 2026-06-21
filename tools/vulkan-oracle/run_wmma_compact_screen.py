#!/usr/bin/env python3
import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys


def default_objdump():
    candidates = []
    for env_name in ("ROCM_PATH", "GGML_HRX_ROCM_PATH"):
        value = os.environ.get(env_name)
        if value:
            root = pathlib.Path(value)
            candidates.append(root / "llvm/bin/llvm-objdump")
            candidates.append(root / "bin/llvm-objdump")
    candidates += [
        pathlib.Path("rocm/llvm/bin/llvm-objdump"),
        pathlib.Path("rocm/bin/llvm-objdump"),
        pathlib.Path("llvm-objdump"),
    ]
    for candidate in candidates:
        if shutil.which(str(candidate)) or candidate.exists():
            return str(candidate)
    return "llvm-objdump"


def run(cmd, *, stdout_path=None, stderr_path=None, check=False):
    stdout = subprocess.PIPE if stdout_path is None else open(stdout_path, "w", encoding="utf-8")
    stderr = subprocess.PIPE if stderr_path is None else open(stderr_path, "w", encoding="utf-8")
    try:
        proc = subprocess.run(cmd, text=True, stdout=stdout, stderr=stderr, check=False)
    finally:
        if stdout_path is not None:
            stdout.close()
        if stderr_path is not None:
            stderr.close()
    if check and proc.returncode != 0:
        raise SystemExit(proc.returncode)
    return proc


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create a WMMA compact-accumulator comparison artifact for a RADV ISA "
            "oracle and a CMake-built HIP HSACO candidate."
        )
    )
    parser.add_argument("--radv-isa", required=True, type=pathlib.Path)
    parser.add_argument("--radv-label", default="RADV oracle")
    parser.add_argument("--hip-hsaco", required=True, type=pathlib.Path)
    parser.add_argument("--hip-label", default="HIP candidate")
    parser.add_argument("--hip-symbol", help="Optional symbol substring to isolate in HIP objdump text.")
    parser.add_argument("--out-dir", required=True, type=pathlib.Path)
    parser.add_argument("--llvm-objdump", default=default_objdump())
    parser.add_argument(
        "--objdump-mattr",
        default="+wmma-128b-insts,+wavefrontsize64",
        help=(
            "Optional llvm-objdump --mattr value. The gfx1151 compact f16 WMMA "
            "encoding disassembles as width8 without +wmma-128b-insts."
        ),
    )
    parser.add_argument(
        "--fail-on-compact-miss",
        action="store_true",
        help="Return the compact screen's non-zero exit code instead of only recording it in the artifact.",
    )
    args = parser.parse_args()

    script_dir = pathlib.Path(__file__).resolve().parent
    extractor = script_dir / "extract_wmma_ownership.py"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    radv_copy = args.out_dir / "radv.amdgcn.txt"
    hip_objdump = args.out_dir / "hip.objdump.txt"
    comparison_json = args.out_dir / "wmma-ownership.json"
    comparison_md = args.out_dir / "wmma-ownership.md"
    compact_stderr = args.out_dir / "compact-check.stderr.log"

    shutil.copyfile(args.radv_isa, radv_copy)

    objdump_cmd = [args.llvm_objdump, "-d"]
    if args.objdump_mattr:
        objdump_cmd.append(f"--mattr={args.objdump_mattr}")
    objdump_cmd.append(str(args.hip_hsaco))
    objdump_proc = run(objdump_cmd, stdout_path=hip_objdump, stderr_path=args.out_dir / "objdump.stderr.log")

    extract_cmd = [
        sys.executable,
        str(extractor),
        "--isa",
        str(radv_copy),
        "--label",
        args.radv_label,
        "--compare-isa",
        str(hip_objdump),
        "--compare-label",
        args.hip_label,
        "--out-json",
        str(comparison_json),
        "--out-md",
        str(comparison_md),
        "--require-compact-f16-accumulators",
    ]
    if args.hip_symbol:
        extract_cmd += ["--compare-symbol", args.hip_symbol]

    extract_proc = run(extract_cmd, stderr_path=compact_stderr)
    summary = {
        "radv_isa": str(args.radv_isa),
        "hip_hsaco": str(args.hip_hsaco),
        "hip_symbol": args.hip_symbol,
        "llvm_objdump": args.llvm_objdump,
        "objdump_mattr": args.objdump_mattr,
        "objdump_command": objdump_cmd,
        "objdump_returncode": objdump_proc.returncode,
        "compact_check_returncode": extract_proc.returncode,
        "passes_compact_f16_accumulator_screen": extract_proc.returncode == 0,
        "artifacts": {
            "radv_isa_copy": str(radv_copy),
            "hip_objdump": str(hip_objdump),
            "ownership_json": str(comparison_json),
            "ownership_md": str(comparison_md),
            "compact_check_stderr": str(compact_stderr),
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.fail_on_compact_miss and extract_proc.returncode != 0:
        return extract_proc.returncode
    return 0 if objdump_proc.returncode == 0 else objdump_proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
