#!/usr/bin/env python3
import argparse
import json
import subprocess
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Link HRX2 Loom route artifacts.")
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--loom-link", required=True, type=Path)
    parser.add_argument("--strip-check", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    sources = catalog["sources"]
    artifacts = catalog["artifacts"]
    args.artifact_root.mkdir(parents=True, exist_ok=True)

    artifact_groups = {}
    for route in catalog["routes"]:
        source = sources[route["source_id"]]
        artifact = artifacts[route["artifact_id"]]
        if artifact.get("format") != "loom-bytecode":
            continue
        artifact_key = route["artifact_id"]
        group = artifact_groups.setdefault(artifact_key, {
            "source_id": route["source_id"],
            "source_path": args.source_root / source["path"],
            "artifact_path": args.artifact_root / artifact["path"],
            "roots": [],
        })
        if group["source_id"] != route["source_id"]:
            raise SystemExit(f"artifact {artifact_key} is referenced by multiple sources")
        if route["root_symbol"] not in group["roots"]:
            group["roots"].append(route["root_symbol"])

    for group in artifact_groups.values():
        source_path = group["source_path"]
        artifact_path = group["artifact_path"]
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(args.loom_link),
            str(source_path),
            "--mode=selective",
            "--to=bytecode",
            f"--output={artifact_path}",
        ]
        for root_symbol in group["roots"]:
            cmd.insert(-2, f"--root={root_symbol}")
        if args.strip_check:
            cmd.insert(-2, "--strip-check")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
