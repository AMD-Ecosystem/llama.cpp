#!/usr/bin/env python3
"""Validate the HRX v1 HIP catalog."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def contained_path(root: Path, relative: str) -> Path:
    full_path = (root / relative).resolve()
    require(full_path == root or root in full_path.parents, f"path escapes root: {relative}")
    return full_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--require-artifacts", action="store_true")
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    artifact_root = args.artifact_root.resolve() if args.artifact_root else source_root
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))

    require(catalog.get("schema") == "ggml-hrx-catalog-v1", "catalog schema must be ggml-hrx-catalog-v1")
    sources = catalog.get("sources")
    artifacts = catalog.get("artifacts")
    families = catalog.get("families")
    routes = catalog.get("routes")
    fusions = catalog.get("fusions", [])
    require(isinstance(sources, dict), "sources must be an object")
    require(isinstance(artifacts, dict), "artifacts must be an object")
    require(isinstance(families, list), "families must be an array")
    require(isinstance(routes, list), "routes must be an array")
    require(isinstance(fusions, list), "fusions must be an array")

    for source_id, source in sources.items():
        require(isinstance(source, dict), f"source {source_id} must be an object")
        path = source.get("path")
        require(isinstance(path, str) and path, f"source {source_id} missing path")
        require(contained_path(source_root, path).exists(), f"source {source_id} path does not exist: {path}")

    for artifact_id, artifact in artifacts.items():
        require(isinstance(artifact, dict), f"artifact {artifact_id} must be an object")
        require(artifact.get("format") == "amdgpu-hsaco", f"artifact {artifact_id} must be amdgpu-hsaco")
        path = artifact.get("path")
        require(isinstance(path, str) and path, f"artifact {artifact_id} missing path")
        artifact_path = contained_path(artifact_root, path)
        if args.require_artifacts:
            require(artifact_path.exists(), f"artifact {artifact_id} path does not exist: {path}")

    family_ids = set()
    for family in families:
        require(isinstance(family, dict), "family entries must be objects")
        family_id = family.get("family")
        require(isinstance(family_id, str) and family_id, "family entry missing family")
        require(family_id not in family_ids, f"duplicate family id: {family_id}")
        family_ids.add(family_id)

    source_ids = set(sources.keys())
    artifact_ids = set(artifacts.keys())
    route_ids = set()
    for route in routes:
        require(isinstance(route, dict), "route entries must be objects")
        route_id = route.get("id")
        require(isinstance(route_id, str) and route_id, "route id must be present")
        require(route_id not in route_ids, f"duplicate route id: {route_id}")
        route_ids.add(route_id)
        require(route.get("family") in family_ids, f"route {route_id} references unknown family")
        require(route.get("source_id") in source_ids, f"route {route_id} references unknown source")
        require(route.get("artifact_id") in artifact_ids, f"route {route_id} references unknown artifact")
        require(isinstance(route.get("export_name"), str) and route["export_name"], f"route {route_id} missing export_name")
        require(isinstance(route.get("priority", 0), int), f"route {route_id} priority must be an integer")
        abi = route.get("abi", {})
        require(isinstance(abi, dict), f"route {route_id} abi must be an object")
        require(isinstance(abi.get("binding_count"), int) and abi["binding_count"] > 0, f"route {route_id} missing binding_count")
        require(isinstance(abi.get("parameter_count"), int) and abi["parameter_count"] >= 0, f"route {route_id} missing parameter_count")
        require(
            isinstance(abi.get("constant_byte_length"), int) and abi["constant_byte_length"] >= 0,
            f"route {route_id} missing constant_byte_length",
        )
        dispatch = route.get("dispatch", {})
        workgroup = dispatch.get("workgroup_size")
        require(
            isinstance(workgroup, list) and len(workgroup) == 3 and all(isinstance(v, int) and v > 0 for v in workgroup),
            f"route {route_id} dispatch.workgroup_size must have three positive integers",
        )
        target_key = route.get("target_key")
        require(isinstance(target_key, str), f"route {route_id} target_key must be a string")
        prefixes = route.get("target_prefixes", [])
        require(isinstance(prefixes, list), f"route {route_id} target_prefixes must be an array")
        require(all(isinstance(prefix, str) and prefix for prefix in prefixes), f"route {route_id} target_prefixes must be strings")

    fusion_ids = set()
    for fusion in fusions:
        require(isinstance(fusion, dict), "fusion entries must be objects")
        fusion_id = fusion.get("id")
        require(isinstance(fusion_id, str) and fusion_id, "fusion id must be present")
        require(fusion_id not in fusion_ids, f"duplicate fusion id: {fusion_id}")
        fusion_ids.add(fusion_id)


if __name__ == "__main__":
    main()
