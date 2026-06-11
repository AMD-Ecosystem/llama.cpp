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

    linked_artifacts = {}
    for route in catalog["routes"]:
        source = sources[route["source_id"]]
        artifact = artifacts[route["artifact_id"]]
        if artifact.get("format") != "loom-bytecode":
            continue
        artifact_key = route["artifact_id"]
        link_key = (route["source_id"], route["root_symbol"])
        existing_link_key = linked_artifacts.get(artifact_key)
        if existing_link_key is not None:
            if existing_link_key != link_key:
                raise SystemExit(
                    f"artifact {artifact_key} is referenced with multiple source/root pairs: "
                    f"{existing_link_key} and {link_key}"
                )
            continue
        linked_artifacts[artifact_key] = link_key

        source_path = args.source_root / source["path"]
        artifact_path = args.artifact_root / artifact["path"]
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(args.loom_link),
            str(source_path),
            "--mode=selective",
            f"--root={route['root_symbol']}",
            "--to=bytecode",
            f"--output={artifact_path}",
        ]
        if args.strip_check:
            cmd.insert(-2, "--strip-check")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
