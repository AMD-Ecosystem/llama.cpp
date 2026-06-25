#!/usr/bin/env python3
"""Prepare a checked-in HRX split catalog for runtime use."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from assemble_hrx_catalog import assemble, require


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-dir", required=True, type=Path)
    parser.add_argument("--processed-dir", required=True, type=Path)
    parser.add_argument("--loom-link", required=True, type=Path)
    return parser.parse_args()


def copy_catalog_inputs(source_dir: Path, processed_dir: Path) -> None:
    if processed_dir.exists():
        shutil.rmtree(processed_dir)
    processed_dir.mkdir(parents=True)
    for rel in ("metadata.json", "kernels", "defs", "targets", "tests"):
        src = source_dir / rel
        if not src.exists():
            continue
        dst = processed_dir / rel
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def artifact_sources(catalog: dict[str, Any]) -> dict[str, str]:
    source_for_artifact: dict[str, str] = {}
    artifacts = catalog.get("artifacts") or {}
    for artifact_id, artifact in artifacts.items():
        if isinstance(artifact, dict) and artifact.get("source_id"):
            source_for_artifact[str(artifact_id)] = str(artifact["source_id"])
    for route in catalog.get("routes") or []:
        if not isinstance(route, dict):
            continue
        artifact_id = route.get("artifact_id")
        source_id = route.get("source_id")
        if artifact_id and source_id:
            source_for_artifact.setdefault(str(artifact_id), str(source_id))
    return source_for_artifact


def artifact_roots(catalog: dict[str, Any]) -> dict[str, list[str]]:
    roots: dict[str, list[str]] = {}
    artifacts = catalog.get("artifacts") or {}
    for artifact_id, artifact in artifacts.items():
        if isinstance(artifact, dict) and artifact.get("root_symbol"):
            roots.setdefault(str(artifact_id), []).append(str(artifact["root_symbol"]))
    for route in catalog.get("routes") or []:
        if not isinstance(route, dict):
            continue
        artifact_id = route.get("artifact_id")
        root_symbol = route.get("root_symbol")
        if artifact_id and root_symbol:
            root_list = roots.setdefault(str(artifact_id), [])
            if str(root_symbol) not in root_list:
                root_list.append(str(root_symbol))
    return roots


def link_artifacts(processed_dir: Path, catalog: dict[str, Any], loom_link: Path) -> None:
    sources = catalog.get("sources") or {}
    artifacts = catalog.get("artifacts") or {}
    source_for_artifact = artifact_sources(catalog)
    roots_for_artifact = artifact_roots(catalog)
    for artifact_id, artifact in artifacts.items():
        require(isinstance(artifact, dict), f"artifact {artifact_id} must be an object")
        require(artifact.get("format") == "loom-bytecode", f"artifact {artifact_id} must be loom-bytecode")
        artifact_path = processed_dir / str(artifact.get("path") or "")
        source_id = source_for_artifact.get(str(artifact_id))
        require(source_id is not None, f"artifact {artifact_id} has no source_id")
        source = sources.get(source_id)
        require(isinstance(source, dict), f"artifact {artifact_id} references unknown source {source_id}")
        roots = roots_for_artifact.get(str(artifact_id)) or []
        require(roots, f"artifact {artifact_id} has no root_symbol")
        source_path = processed_dir / str(source.get("path") or "")
        require(source_path.exists(), f"source {source_id} path does not exist: {source_path}")
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(loom_link),
            str(source_path),
            "--mode=link",
            "--strip-check",
            "--to=bc",
            f"--output={artifact_path}",
        ]
        cmd.extend(f"--root={root}" for root in roots)
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            sys.stderr.write(result.stdout)
            sys.stderr.write(result.stderr)
            raise SystemExit(f"failed to link HRX catalog artifact {artifact_id}")


def main() -> None:
    args = parse_args()
    source_dir = args.catalog_dir.resolve()
    processed_dir = args.processed_dir.resolve()
    loom_link = args.loom_link.resolve()
    require(source_dir.is_dir(), f"catalog directory does not exist: {source_dir}")
    require(loom_link.exists(), f"loom-link does not exist: {loom_link}")

    copy_catalog_inputs(source_dir, processed_dir)
    catalog = assemble(processed_dir)
    link_artifacts(processed_dir, catalog, loom_link)
    catalog_path = processed_dir / "catalog.json"
    catalog_path.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
