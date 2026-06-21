#!/usr/bin/env python3
import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys


WIDTH8_ASM = "v_wmma_f16_16x16x16_f16 v[0:7], v[8:15], v[16:23], v[0:7]\n"
WIDTH4_ALL_ASM = "v_wmma_f16_16x16x16_f16 v[0:3], v[8:11], v[16:19], v[0:3]\n"
WIDTH4_MIXED_ASM = "v_wmma_f16_16x16x16_f16 v[0:3], v[8:15], v[16:23], v[0:3]\n"
RADV_MIXED_ASM = "v_wmma_f16_16x16x16_f16 v[72:75], v[96:103], v[104:111], v[72:75]\n"


def default_llvm_mc():
    candidates = []
    for env_name in ("ROCM_PATH", "GGML_HRX_ROCM_PATH"):
        value = os.environ.get(env_name)
        if value:
            root = pathlib.Path(value)
            candidates.append(root / "llvm/bin/llvm-mc")
            candidates.append(root / "bin/llvm-mc")
    candidates += [
        pathlib.Path("rocm/llvm/bin/llvm-mc"),
        pathlib.Path("rocm/bin/llvm-mc"),
        pathlib.Path("llvm-mc"),
    ]
    for candidate in candidates:
        resolved = shutil.which(str(candidate))
        if resolved:
            return resolved
        if candidate.exists():
            return str(candidate)
    return "llvm-mc"


def run_probe(llvm_mc, mcpu, mattr, source, out_dir, name):
    asm_path = out_dir / f"{name}.s"
    stdout_path = out_dir / f"{name}.stdout"
    stderr_path = out_dir / f"{name}.stderr"
    status_path = out_dir / f"{name}.status"
    asm_path.write_text(source, encoding="utf-8")

    cmd = [
        llvm_mc,
        "-triple=amdgcn-amd-amdhsa",
        f"-mcpu={mcpu}",
        "--show-encoding",
        str(asm_path),
    ]
    if mattr:
        cmd.insert(3, f"-mattr={mattr}")

    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")
    status_path.write_text(str(proc.returncode) + "\n", encoding="utf-8")
    return {
        "name": name,
        "asm": str(asm_path),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "status": str(status_path),
        "returncode": proc.returncode,
        "command": cmd,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Probe whether LLVM MC accepts RADV-style compact f16 WMMA operands for a target."
    )
    parser.add_argument("--out-dir", required=True, type=pathlib.Path)
    parser.add_argument("--mcpu", default="gfx1151")
    parser.add_argument("--mattr", default="")
    parser.add_argument("--llvm-mc", default=default_llvm_mc())
    parser.add_argument(
        "--require-width4",
        action="store_true",
        help="Return non-zero if the RADV mixed width4 dst/C + width8 A/B compact form is rejected.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    width8 = run_probe(args.llvm_mc, args.mcpu, args.mattr, WIDTH8_ASM, args.out_dir, "width8")
    width4_all = run_probe(args.llvm_mc, args.mcpu, args.mattr, WIDTH4_ALL_ASM, args.out_dir, "width4_all")
    width4_mixed = run_probe(args.llvm_mc, args.mcpu, args.mattr, WIDTH4_MIXED_ASM, args.out_dir, "width4_mixed")
    radv_mixed = run_probe(args.llvm_mc, args.mcpu, args.mattr, RADV_MIXED_ASM, args.out_dir, "radv_mixed")
    summary = {
        "llvm_mc": args.llvm_mc,
        "mcpu": args.mcpu,
        "mattr": args.mattr,
        "width8_accepted": width8["returncode"] == 0,
        "width4_all_accepted": width4_all["returncode"] == 0,
        "width4_mixed_accepted": width4_mixed["returncode"] == 0,
        "radv_mixed_accepted": radv_mixed["returncode"] == 0,
        "width8": width8,
        "width4_all": width4_all,
        "width4_mixed": width4_mixed,
        "radv_mixed": radv_mixed,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.out_dir / "summary.md").write_text(
        "\n".join([
            "# WMMA MC Compact Probe",
            "",
            f"- llvm-mc: `{args.llvm_mc}`",
            f"- mcpu: `{args.mcpu}`",
            f"- mattr: `{args.mattr}`",
            f"- width8 accepted: `{summary['width8_accepted']}`",
            f"- width4 all-operands accepted: `{summary['width4_all_accepted']}`",
            f"- width4 mixed dst/C accepted: `{summary['width4_mixed_accepted']}`",
            f"- RADV mixed dst/C accepted: `{summary['radv_mixed_accepted']}`",
            "",
            "## Interpretation",
            "",
            (
                "The RADV-style compact width4 dst/C plus width8 A/B f16 WMMA operand form is accepted by this assembler."
                if summary["width4_mixed_accepted"] and summary["radv_mixed_accepted"]
                else "The RADV-style compact width4 dst/C plus width8 A/B f16 WMMA operand form is rejected by this assembler."
            ),
            "",
        ]) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.require_width4 and not (summary["width4_mixed_accepted"] and summary["radv_mixed_accepted"]):
        return 1
    if summary["width8_accepted"] or (summary["width4_mixed_accepted"] and summary["radv_mixed_accepted"]):
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
