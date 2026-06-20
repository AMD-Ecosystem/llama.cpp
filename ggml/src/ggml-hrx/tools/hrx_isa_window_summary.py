#!/usr/bin/env python3
"""Summarize AMDGCN ISA windows for HRX/RADV schedule comparison.

This is intentionally text-based. The Vulkan oracle and HIP build artifacts
already keep objdump/RADV disassembly as plain text, and the first useful
question during schedule porting is usually whether the visible opcode/window
contract matches before promoting a route.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable


INTERESTING_PREFIXES = (
    "buffer_store",
    "buffer_load",
    "flat_store",
    "flat_load",
    "global_store",
    "global_load",
    "ds_store",
    "ds_load",
    "v_wmma",
    "v_dot",
    "s_waitcnt",
    "s_waitcnt_depctr",
    "s_barrier",
)


def parse_instruction(line: str) -> str | None:
    text = line.strip()
    if not text or text.startswith("#"):
        return None
    label_match = re.match(r"^(?:(?:[0-9a-fA-F]+)|(?:BB[0-9]+)):\s*(.*)$", text)
    if label_match:
        text = label_match.group(1).strip()
    if not text or text.startswith(";"):
        return None
    if ";" in text:
        text = text.split(";", 1)[0].strip()
    if not text:
        return None
    mnemonic = text.split(None, 1)[0]
    if mnemonic in {"BB0:", "BB1:"}:
        return None
    return mnemonic


def classify(mnemonic: str) -> Iterable[str]:
    yield mnemonic
    for prefix in INTERESTING_PREFIXES:
        if mnemonic.startswith(prefix):
            yield prefix + "*"


def load_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def summarize(path: Path, context: int, regexes: list[re.Pattern[str]]) -> dict:
    lines = load_lines(path)
    opcodes: Counter[str] = Counter()
    first_line: dict[str, int] = {}
    last_line: dict[str, int] = {}
    windows: list[dict] = []

    for index, line in enumerate(lines, start=1):
        mnemonic = parse_instruction(line)
        if mnemonic is not None:
            for key in classify(mnemonic):
                opcodes[key] += 1
                first_line.setdefault(key, index)
                last_line[key] = index

        if regexes and any(regex.search(line) for regex in regexes):
            start = max(1, index - context)
            end = min(len(lines), index + context)
            windows.append(
                {
                    "match_line": index,
                    "line": line.rstrip(),
                    "start": start,
                    "end": end,
                    "lines": [
                        {"line": n, "text": lines[n - 1].rstrip()}
                        for n in range(start, end + 1)
                    ],
                }
            )

    return {
        "path": str(path),
        "instruction_count": sum(
            count for mnemonic, count in opcodes.items() if not mnemonic.endswith("*")
        ),
        "counts": dict(sorted(opcodes.items())),
        "first_line": dict(sorted(first_line.items())),
        "last_line": dict(sorted(last_line.items())),
        "windows": windows,
    }


def write_markdown(summary: dict, out: Path) -> None:
    rows = summary["files"]
    keys = sorted(
        {
            key
            for row in rows
            for key in row["counts"]
            if key.endswith("*") or key in {"v_wmma_f16_16x16x16_f16", "s_barrier"}
        }
    )
    with out.open("w", encoding="utf-8") as f:
        f.write("# ISA Window Summary\n\n")
        f.write("| File | Instructions | " + " | ".join(keys) + " |\n")
        f.write("| --- | ---: | " + " | ".join("---:" for _ in keys) + " |\n")
        for row in rows:
            name = Path(row["path"]).name
            values = [str(row["counts"].get(key, 0)) for key in keys]
            f.write(f"| `{name}` | {row['instruction_count']} | " + " | ".join(values) + " |\n")
        f.write("\n")

        for row in rows:
            f.write(f"## {Path(row['path']).name}\n\n")
            if not row["windows"]:
                f.write("No requested windows matched.\n\n")
                continue
            for window in row["windows"]:
                f.write(f"### Match Line {window['match_line']}\n\n")
                f.write("```text\n")
                for item in window["lines"]:
                    f.write(f"{item['line']:6d}: {item['text']}\n")
                f.write("```\n\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("isa", nargs="+", type=Path, help="AMDGCN/RADV disassembly text files")
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path, required=True)
    parser.add_argument(
        "--window-regex",
        action="append",
        default=[],
        help="Regex to capture context windows around matching lines",
    )
    parser.add_argument("--context", type=int, default=8)
    args = parser.parse_args()

    regexes = [re.compile(pattern) for pattern in args.window_regex]
    summary = {
        "schema": "hrx-isa-window-summary-v1",
        "files": [summarize(path, args.context, regexes) for path in args.isa],
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_markdown(summary, args.md_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
