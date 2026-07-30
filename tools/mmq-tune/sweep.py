#!/usr/bin/env python3
"""Sweep the MMQ compile-time tile hyperparameters.

mmq_x is swept inside mmq-tune at runtime, but mmq_y and the warp count are device-side
constexpr, so each combination needs its own build. This drives that outer loop: configure,
build only the mmq-tune target, run it, append the CSV.

    python tools/mmq-tune/sweep.py --mmq-y 32,64,128 --mmq-nwarps 2,4,8

Combinations that fail to build or run are recorded rather than aborting the sweep: the kernel
asserts nwarps*tile_C::I == mmq_y, so most pairs do not compile.
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def run(cmd, **kw):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run(cmd, **kw)


def build(build_dir, mmq_y, nwarps, jobs, cmake_args):
    cfg = [
        "cmake", "-S", str(REPO), "-B", str(build_dir),
        "-DGGML_HIP=ON", "-DGGML_MMQ_TUNE=ON",
        f"-DGGML_HIP_MMQ_Y={mmq_y}", f"-DGGML_HIP_MMQ_NWARPS={nwarps}",
        *cmake_args,
    ]
    if run(cfg, capture_output=True, text=True).returncode != 0:
        return "configure_failed"
    bld = ["cmake", "--build", str(build_dir), "--target", "mmq-tune", "-j", str(jobs)]
    if run(bld, capture_output=True, text=True).returncode != 0:
        return "build_failed"
    return None


def measure(build_dir, extra_args, gpu_lock, retries=6):
    cmd = ([gpu_lock] if gpu_lock else []) + [
        str(build_dir / "bin" / "mmq-tune"), "--output", "csv", *extra_args
    ]
    # amd-gpu-lock gives up after 10 minutes; retry rather than lose a build's measurements.
    for attempt in range(retries):
        p = run(cmd, capture_output=True, text=True)
        if p.returncode == 0:
            # The launcher and the ggml backend print to stdout before the CSV header.
            lines = p.stdout.splitlines()
            for i, line in enumerate(lines):
                if line.startswith("model,kind,"):
                    return "\n".join(lines[i:])
        sys.stderr.write(p.stderr[-2000:])
        print(f"  measure attempt {attempt + 1}/{retries} failed", flush=True)
    return None


def report(rows):
    """Best width per shape, then the (mmq_y, nwarps) ranking those widths imply."""
    rows = [r for r in rows if r.get("mmq_x")]
    if not rows:
        return

    def key(r):
        return (r["kind"], r["type"], r["n"], r["k"], r["m"])

    def usable(r):
        # mmq_x < 0 is the baseline measurement, not a candidate width.
        return int(r["mmq_x"]) > 0

    print("\n=== best config per shape ===")
    best = {}
    for r in rows:
        if not usable(r):
            continue
        cur = best.get(key(r))
        if cur is None or float(r["us_median"]) < float(cur["us_median"]):
            best[key(r)] = r
    for k in sorted(best, key=lambda t: (t[0], t[1], int(t[2]), int(t[3]), int(t[4]))):
        r = best[k]
        print(f"  {r['kind']:<5} {r['type']:>5} N={r['n']:>6} K={r['k']:>6} M={r['m']:>5}  ->  "
              f"mmq_y={r['mmq_y']:>3} nwarps={r['nwarps']} mmq_x={r['mmq_x']:>3}  "
              f"{float(r['us_median']):9.2f} us")

    # mmq_y and nwarps are fixed per build, so the ranking is over their totals, weighted by how
    # often each shape runs.
    print("\n=== global (mmq_y, nwarps), weighted by invocations per forward pass ===")
    per_build = defaultdict(dict)
    weight = {}
    for r in rows:
        if not usable(r):
            continue
        b = (int(r["mmq_y"]), int(r["nwarps"]))
        us = float(r["us_median"])
        if key(r) not in per_build[b] or us < per_build[b][key(r)]:
            per_build[b][key(r)] = us
        weight[key(r)] = int(r["weight"])

    print(f"  {'build':>16}{'dense ms':>12}{'MoE ms':>12}{'total ms':>12}")
    ranked = []
    for b, v in per_build.items():
        dense = sum(us*weight[k] for k, us in v.items() if k[0] == "dense")/1000
        moe = sum(us*weight[k] for k, us in v.items() if k[0] == "moe")/1000
        ranked.append((dense + moe, dense, moe, b))
    for total, dense, moe, b in sorted(ranked):
        print(f"  mmq_y={b[0]:>3} nwarps={b[1]}{dense:12.1f}{moe:12.1f}{total:12.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mmq-y", default="32,64,128")
    ap.add_argument("--mmq-nwarps", default="2,4,8")
    ap.add_argument("--build-dir", default=str(REPO / "build-tune"))
    ap.add_argument("--out", default="sweep.csv")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--gpu-lock", default="amd-gpu-lock", help="launcher prefix, empty to disable")
    ap.add_argument("--cmake-arg", action="append", default=[], help="extra -D flag forwarded to cmake configure")
    ap.add_argument("mmq_tune_args", nargs="*", help="extra args forwarded to mmq-tune")
    args = ap.parse_args()

    build_dir = Path(args.build_dir).resolve()
    rows = []

    with open(args.out, "w", newline="") as fout:
        writer = None
        for mmq_y in [int(v) for v in args.mmq_y.split(",")]:
            for nwarps in [int(v) for v in args.mmq_nwarps.split(",")]:
                print(f"\n### mmq_y={mmq_y} nwarps={nwarps}", flush=True)

                status = build(build_dir, mmq_y, nwarps, args.jobs, args.cmake_arg)
                csv_text = None if status else measure(build_dir, args.mmq_tune_args, args.gpu_lock)
                if csv_text is None:
                    status = status or "run_failed"
                    print(f"  {status}", flush=True)
                    if writer:
                        writer.writerow({**{k: "" for k in writer.fieldnames},
                                         "mmq_y": mmq_y, "nwarps": nwarps, "status": status})
                    continue

                for row in csv.DictReader(io.StringIO(csv_text)):
                    row["status"] = "ok"
                    if writer is None:
                        writer = csv.DictWriter(fout, fieldnames=list(row.keys()))
                        writer.writeheader()
                    writer.writerow(row)
                    rows.append(row)
                fout.flush()
                print(f"  ok ({len(rows)} rows total)", flush=True)

    report(rows)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
