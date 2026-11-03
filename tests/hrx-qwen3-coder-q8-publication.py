#!/usr/bin/env python3
"""Check the Qwen3-Coder fused HRX kernel's published Q8 against its F32 row.

This model-gated integration check uses llama-debug dispatch traces. Invoke with
--model and --binary to capture two identical GPU runs, or --capture-dir to
check saved run1 and run2 traces. The caller owns GPU selection and locks.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys


KERNEL = "qwen3_moe:qwen3_moe_routed_down_weighted_reduce_next_rmsnorm_f32_publish_q8"
MODEL_SHA256 = "fadc3e5f8d42bf7e894a785b05082e47daee4df26680389817e2093056f088ad"
PROMPT = "The capital of France is"
BEFORE_BINDINGS = {0: "route_weights", 1: "routed_output", 2: "hidden_state", 3: "next_norm_weight"}
AFTER_BINDINGS = {2: "hidden_state", 4: "next_projection_input", 5: "q8_output"}
WIDTH = 2048
GROUP_WIDTH = 128
Q8_GROUP_BYTES = 144


def trace_rows(run):
    path = run / "trace.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def target_command(rows):
    targets = [row for row in rows if row["kernel"] == KERNEL]
    if not targets:
        raise ValueError("target fused Q8 kernel absent from dispatch trace")
    first = targets[0]
    index = first["command_index"]
    program_size = first["program_command_count"]
    selected = [row for row in rows if row["command_index"] == index
                and row["program_command_count"] == program_size]
    kernels = {row["kernel"] for row in selected}
    if kernels != {KERNEL}:
        raise ValueError(f"ambiguous command {index}: {kernels}")
    for stage, expected in (("before", BEFORE_BINDINGS), ("after", AFTER_BINDINGS)):
        stage_rows = [row for row in selected if row["stage"] == stage]
        actual = {row["binding_index"]: row["binding_name"] for row in stage_rows}
        if len(stage_rows) != len(actual):
            raise ValueError(f"duplicate {stage} bindings for command {index}")
        if actual != expected:
            raise ValueError(f"unexpected {stage} bindings for command {index}: {actual}")
    return index, program_size, selected


def blob_for(rows, run, name):
    matches = [row for row in rows if row["stage"] == "after" and row["binding_name"] == name]
    if len(matches) != 1:
        raise ValueError(f"expected one {name} output in {run}")
    blob = matches[0].get("blob")
    if not blob:
        raise ValueError(f"missing {name} blob in {run}")
    data = (run / blob).read_bytes()
    if len(data) != matches[0]["length"]:
        raise ValueError(f"truncated {name} blob in {run}")
    return data


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def expected_q8(f32_bytes):
    row_bytes = WIDTH * 4
    if len(f32_bytes) % row_bytes:
        raise ValueError("F32 publication has incomplete rows")
    values = struct.unpack(f"<{len(f32_bytes) // 4}f", f32_bytes)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("nonfinite F32 publication")
    encoded = bytearray()
    for base in range(0, len(values), GROUP_WIDTH):
        group = values[base : base + GROUP_WIDTH]
        metadata = bytearray()
        payload = bytearray()
        for block in range(4):
            block_values = group[block * 32 : (block + 1) * 32]
            maximum = max(abs(value) for value in block_values)
            scale = f32(maximum / 127.0)
            inverse = f32(1.0 / scale) if scale else 0.0
            quantized = [round(f32(value * inverse)) for value in block_values]
            sum_scaled = f32(f32(float(sum(quantized))) * scale)
            metadata.extend(struct.pack("<ee", scale, sum_scaled))
            payload.extend(struct.pack("<32b", *quantized))
        encoded.extend(metadata)
        encoded.extend(payload)
    return bytes(encoded)


def inspect_run(run):
    index, program_size, rows = target_command(trace_rows(run))
    f32_bytes = blob_for(rows, run, "next_projection_input")
    q8_bytes = blob_for(rows, run, "q8_output")
    expected = expected_q8(f32_bytes)
    if len(q8_bytes) != len(expected):
        raise ValueError(f"Q8 output length differs from F32 row in {run}")
    bad_groups = [group for group in range(len(expected) // Q8_GROUP_BYTES)
                  if q8_bytes[group * Q8_GROUP_BYTES : (group + 1) * Q8_GROUP_BYTES]
                  != expected[group * Q8_GROUP_BYTES : (group + 1) * Q8_GROUP_BYTES]]
    return {
        "command_index": index,
        "program_command_count": program_size,
        "tokens": len(f32_bytes) // (WIDTH * 4),
        "f32_sha256": hashlib.sha256(f32_bytes).hexdigest(),
        "q8_sha256": hashlib.sha256(q8_bytes).hexdigest(),
        "bad_groups": bad_groups,
    }


def run_debug(binary, model, output, name, command=None, program_size=None):
    run = output / name
    run.mkdir(parents=True, exist_ok=False)
    logits = output / f"{name}-logits"
    env = os.environ.copy()
    env["GGML_HRX_TRACE_DISPATCH_DIR"] = str(run)
    env["GGML_HRX_TRACE_BLOBS"] = "1" if command is not None else "0"
    if command is not None:
        env["GGML_HRX_TRACE_START_COMMAND"] = str(command)
        env["GGML_HRX_TRACE_END_COMMAND"] = str(command)
        env["GGML_HRX_TRACE_PROGRAM_COMMANDS"] = str(program_size)
    args = [str(binary), "-m", str(model), "-dev", "HRX0", "-ngl", "99",
            "-c", "2048", "-b", "64", "-ub", "64", "-fa", "on",
            "-ctk", "f16", "-ctv", "f16", "--no-warmup", "-s", "42",
            "-p", PROMPT, "-lv", "4", "--save-logits",
            "--logits-output-dir", str(logits)]
    with (output / f"{name}.log").open("w") as log:
        subprocess.run(args, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=900)
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--capture-dir", type=Path)
    source.add_argument("--model", type=Path)
    parser.add_argument("--binary", type=Path, help="llama-debug binary for --model")
    parser.add_argument("--output-dir", type=Path, help="new capture directory for --model")
    args = parser.parse_args()
    if args.model:
        has_binary = args.binary is not None
        has_output = args.output_dir is not None
        if not has_binary or not has_output:
            parser.error("--model requires --binary and --output-dir")
        digest = hashlib.sha256()
        with args.model.open("rb") as model_file:
            for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
                digest.update(chunk)
        model_hash = digest.hexdigest()
        if model_hash != MODEL_SHA256:
            parser.error(f"unexpected model SHA256: {model_hash}")
        args.output_dir.mkdir(parents=True, exist_ok=False)
        discovery = run_debug(args.binary, args.model, args.output_dir, "discovery")
        command, program_size, _ = target_command(trace_rows(discovery))
        for number in (1, 2):
            run_debug(args.binary, args.model, args.output_dir, f"run{number}", command, program_size)
        capture = args.output_dir
    else:
        capture = args.capture_dir
    runs = [inspect_run(capture / f"run{number}") for number in (1, 2)]
    consistent_f32 = runs[0]["f32_sha256"] == runs[1]["f32_sha256"]
    consistent_q8 = runs[0]["q8_sha256"] == runs[1]["q8_sha256"]
    canonical = all(not run["bad_groups"] for run in runs)
    print(json.dumps({"runs": runs, "f32_repeats": consistent_f32,
                      "q8_repeats": consistent_q8, "canonical": canonical}, indent=2))
    publication_valid = consistent_f32 and consistent_q8 and canonical
    return 0 if publication_valid else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        print(error, file=sys.stderr)
        sys.exit(2)
