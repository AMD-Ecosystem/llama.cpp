#!/usr/bin/env python3
import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time


def repo_root():
    return pathlib.Path(__file__).resolve().parents[2]


def write_command(path, argv, env):
    interesting_env = {
        key: env[key]
        for key in [
            "MESA_SHADER_CACHE_DISABLE",
            "RADV_DEBUG",
            "GGML_VK_TRACE_JSONL",
            "GGML_VK_TRACE_SPV_DIR",
            "GGML_VK_TRACE_RADV_PIPELINE_LABELS",
            "GGML_VK_PERF_LOGGER",
        ]
        if key in env
    }
    payload = {
        "command": argv,
        "environment": interesting_env,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_checked(argv, cwd, stdout_path=None, stderr_path=None, env=None):
    stdout_f = stdout_path.open("w", encoding="utf-8") if stdout_path else None
    stderr_f = stderr_path.open("w", encoding="utf-8") if stderr_path else None
    try:
        return subprocess.run(
            argv,
            cwd=cwd,
            env=env,
            stdout=stdout_f if stdout_f else subprocess.PIPE,
            stderr=stderr_f if stderr_f else subprocess.PIPE,
            text=True,
            check=False,
        )
    finally:
        if stdout_f:
            stdout_f.close()
        if stderr_f:
            stderr_f.close()


def disassemble_spv(out_dir):
    spirv_dis = shutil.which("spirv-dis")
    if not spirv_dis:
        raise SystemExit("spirv-dis not found in PATH; install spirv-tools or add it to PATH")

    spv_dir = out_dir / "spv"
    spvasm_dir = out_dir / "spvasm"
    spvasm_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for spv_path in sorted(spv_dir.glob("*.spv")):
        asm_path = spvasm_dir / (spv_path.stem + ".spvasm")
        result = subprocess.run([spirv_dis, str(spv_path), "-o", str(asm_path)], check=False)
        if result.returncode != 0:
            raise SystemExit(f"spirv-dis failed for {spv_path}")
        count += 1
    return count


def parse_args():
    parser = argparse.ArgumentParser(description="Run a llama.cpp Vulkan oracle capture and post-process it.")
    parser.add_argument("--bench", required=True, type=pathlib.Path, help="Path to llama-bench.")
    parser.add_argument("--model", required=True, type=pathlib.Path, help="GGUF model path.")
    parser.add_argument("--out-dir", required=True, type=pathlib.Path, help="Fresh output directory.")
    parser.add_argument("--prompt", type=int, default=512)
    parser.add_argument("--gen", type=int, default=0)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--ubatch", type=int, default=1024)
    parser.add_argument("--flash-attn", type=int, choices=[0, 1], default=1)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--device", default="Vulkan0")
    parser.add_argument("--ngl", type=int, default=99)
    parser.add_argument("--force", action="store_true", help="Allow using an existing empty output directory.")
    return parser.parse_args()


def main():
    args = parse_args()
    root = repo_root()
    bench = args.bench.resolve()
    model = args.model.resolve()
    out_dir = args.out_dir.resolve()

    if out_dir.exists() and any(out_dir.iterdir()) and not args.force:
        raise SystemExit(f"output directory exists and is not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update({
        "MESA_SHADER_CACHE_DISABLE": "true",
        "RADV_DEBUG": "shaders,shaderstats",
        "GGML_VK_TRACE_JSONL": str(out_dir / "vulkan.jsonl"),
        "GGML_VK_TRACE_SPV_DIR": str(out_dir / "spv"),
        "GGML_VK_TRACE_RADV_PIPELINE_LABELS": "1",
        "GGML_VK_PERF_LOGGER": "1",
    })

    argv = [
        str(bench),
        "-m", str(model),
        "-p", str(args.prompt),
        "-n", str(args.gen),
        "-b", str(args.batch),
        "-ub", str(args.ubatch),
        "-fa", str(args.flash_attn),
        "-r", str(args.repetitions),
        "-o", "json",
        "--no-warmup",
        "-ngl", str(args.ngl),
        "-dev", args.device,
    ]

    write_command(out_dir / "command.txt", argv, env)
    started = time.time()
    result = run_checked(
        argv,
        cwd=root,
        stdout_path=out_dir / "stdout.json",
        stderr_path=out_dir / "radv-stderr.log",
        env=env,
    )
    finished = time.time()

    (out_dir / "run-metadata.json").write_text(
        json.dumps(
            {
                "returncode": result.returncode,
                "started_unix": started,
                "finished_unix": finished,
                "duration_seconds": finished - started,
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )

    if result.returncode != 0:
        raise SystemExit(result.returncode)

    spvasm_count = disassemble_spv(out_dir)

    split_script = root / "tools/vulkan-oracle/split_radv_shader_logs.py"
    reduce_script = root / "tools/vulkan-oracle/reduce_vulkan_oracle_inventory.py"

    split = subprocess.run(
        [sys.executable, str(split_script), "--out-dir", str(out_dir), str(out_dir / "radv-stderr.log")],
        cwd=root,
        check=False,
    )
    if split.returncode != 0:
        raise SystemExit(split.returncode)

    reduce = subprocess.run(
        [sys.executable, str(reduce_script), "--out-dir", str(out_dir / "inventory"), str(out_dir / "vulkan.jsonl")],
        cwd=root,
        check=False,
    )
    if reduce.returncode != 0:
        raise SystemExit(reduce.returncode)

    print(f"out_dir={out_dir}")
    print(f"spvasm_files={spvasm_count}")


if __name__ == "__main__":
    raise SystemExit(main())
