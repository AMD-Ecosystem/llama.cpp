#!/usr/bin/env python3
"""Extract one symbol body from llvm-objdump text.

The Vulkan-oracle workflow often needs symbol-scoped HIP ISA from a larger
fatbin objdump so that downstream region/cluster tools do not accidentally
count neighboring probe kernels. This helper preserves the original objdump
lines between the selected `<symbol>:` header and the next symbol header.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys


SYMBOL_RE = re.compile(r"^[0-9a-fA-F]+ <([^>]+)>:")


def extract_symbol(lines: list[str], symbol: str, exact: bool) -> list[str]:
    matches: list[tuple[str, list[str]]] = []
    current_name: str | None = None
    current_lines: list[str] = []

    def finish_current() -> None:
        if current_name is None:
            return
        matched = current_name == symbol if exact else symbol in current_name
        if matched:
            matches.append((current_name, list(current_lines)))

    for line in lines:
        match = SYMBOL_RE.match(line)
        if match:
            finish_current()
            current_name = match.group(1)
            current_lines = [line]
        elif current_name is not None:
            current_lines.append(line)
    finish_current()

    if not matches:
        mode = "exact symbol" if exact else "symbol substring"
        raise SystemExit(f"{mode} not found in objdump: {symbol}")
    if len(matches) > 1:
        names = ", ".join(name for name, _ in matches[:8])
        raise SystemExit(f"ambiguous symbol match for {symbol!r}: {names}")
    return matches[0][1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objdump", required=True, type=pathlib.Path)
    parser.add_argument("--symbol", required=True, help="Symbol name or substring to extract")
    parser.add_argument("--exact", action="store_true", help="Require an exact symbol-name match")
    parser.add_argument("--out", type=pathlib.Path, help="Output path; stdout if omitted")
    args = parser.parse_args()

    lines = args.objdump.read_text(encoding="utf-8", errors="replace").splitlines()
    body = extract_symbol(lines, args.symbol, exact=args.exact)
    text = "\n".join(body) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
