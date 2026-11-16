#!/usr/bin/env python3
"""Repeat issue 161 performance comparisons with matched runtime builds.

Keep raw output and command provenance; synthetic kernel timings have no oracle.
ROCR_VISIBLE_DEVICES and the ROCm runtime library path must be set by the caller.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

MODELS = {
    'promptbridge06': 'PromptBridge-0.6b-Alpha-GGUF/PromptBridge-0.6b-Alpha.Q4_K_M.gguf',
    'qwen3_17': 'Qwen3-1.7B-GGUF/Qwen3-1.7B-Q4_0.gguf',
    'qwen3_4': 'Qwen3-4B-Instruct-2507-GGUF/Qwen3-4B-Instruct-2507-Q4_K_M.gguf',
    'qwen35_08': 'Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf',
    'qwen35_2': 'Qwen3.5-2B-GGUF/Qwen3.5-2B-UD-Q4_K_XL.gguf',
    'qwen35_9': 'Qwen3.5-9B-GGUF/Qwen3.5-9B-UD-Q4_K_XL.gguf',
}
parser = argparse.ArgumentParser(description=__doc__)
for option in ('baseline-source', 'baseline-build', 'candidate-source', 'candidate-build', 'models', 'output'):
    parser.add_argument('--'+option, type=Path, required=True)
parser.add_argument('--phase', choices=('kernel', 'model', 'all'), default='all')
parser.add_argument('--lock', default='/tmp/lemonade-ws-tek-gpu.lock')
a = parser.parse_args()
if not os.environ.get('ROCR_VISIBLE_DEVICES'):
    parser.error('set ROCR_VISIBLE_DEVICES to the supported GPU')
for key, value in vars(a).items():
    if isinstance(value, Path):
        setattr(a, key, value.resolve())
a.output.mkdir(parents=True, exist_ok=True)
sources = {'baseline': a.baseline_source, 'candidate': a.candidate_source}
builds = {'baseline': a.baseline_build, 'candidate': a.candidate_build}
provenance = {k: subprocess.check_output(['git', '-C', str(v), 'rev-parse', 'HEAD'], text=True).strip() for k, v in sources.items()}
(a.output/'sources.json').write_text(json.dumps(provenance, indent=2))
fixture = Path(__file__).resolve().parent

def run(folder, argv, env):
    folder.mkdir(parents=True, exist_ok=False)
    argv = list(map(str, argv))
    (folder/'command.json').write_text(json.dumps({'argv': argv, 'env': env}, indent=2))
    with open(a.lock, 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        print('START', folder, flush=True)
        start = time.monotonic()
        with (folder/'stdout.log').open('w') as stdout, (folder/'stderr.log').open('w') as stderr:
            try:
                code = subprocess.run(argv, env=os.environ | env, stdout=stdout, stderr=stderr, timeout=1800).returncode
            except subprocess.TimeoutExpired:
                code = 124
        (folder/'exit.json').write_text(json.dumps({'code': code, 'seconds': time.monotonic()-start}))
        print('DONE', folder.name, code, flush=True)
    if code:
        raise SystemExit(f'Failed: {folder}; retain logs before retrying in a new output directory')

if a.phase in ('kernel', 'all'):
    # Four controls identify layout cost separately from the retained reuse optimization.
    order = [('baseline','native'), ('baseline','row64'), ('candidate','row64'), ('candidate','native')]
    for i, (stage, layout) in enumerate(order + order[::-1]):
        hrx = sources[stage]/'ggml/src/ggml-hrx'
        d = json.loads((fixture/(layout+'.json')).read_text())
        d['loom_source'] = str(fixture/(layout+'.loom'))
        manifest = a.output/f'{stage}-{layout}.json'
        manifest.write_text(json.dumps(d, indent=2))
        argv = [sys.executable, hrx/'tools/benchmarks/run-model-benchmarks.py', '--model', d['model'], '--scenario', 'tg32', '--manifest', manifest, '--build-dir', builds[stage], '--output-dir', a.output/f'kernel-{i}-{stage}-{layout}', '--batch-size', '8', '--iterations', '30', '--warmup-iterations', '5', '--input-ring-count', '4', '--profile-final-batch', 'true']
        run(a.output/f'kernel-command-{i}', argv, {'ROCR_VISIBLE_DEVICES': os.environ['ROCR_VISIBLE_DEVICES']})

if a.phase in ('model', 'all'):
    for name, model in MODELS.items():
        for i, stage in enumerate(('baseline','candidate','candidate','baseline')):
            env = {'ROCR_VISIBLE_DEVICES': os.environ['ROCR_VISIBLE_DEVICES'], 'LD_LIBRARY_PATH': str(builds[stage]/'bin')+':'+os.environ.get('LD_LIBRARY_PATH','')}
            argv = [builds[stage]/'bin/llama-bench', '-m', a.models/model, '--device', 'HRX0', '-ngl', '99', '-t', '4', '-fa', 'on', '-p', '66,128,512', '-n', '32', '-b', '512', '-ub', '512', '-r', '5', '-o', 'jsonl']
            run(a.output/name/f'{i}-{stage}', argv, env)
