#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Validate the HRX2 phase0.1 catalog.")
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument(
        "--require-artifacts",
        action="store_true",
        help="Require artifact files to exist. Use after the link step.",
    )
    return parser.parse_args()


def require(condition, message):
    if not condition:
        raise SystemExit(message)


def main():
    args = parse_args()
    source_root = args.source_root.resolve()
    artifact_root = args.artifact_root.resolve() if args.artifact_root else source_root
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))

    require(catalog.get("schema") == "ggml-hrx2-catalog-v1", "catalog schema must be ggml-hrx2-catalog-v1")
    sources = catalog.get("sources")
    artifacts = catalog.get("artifacts")
    routes = catalog.get("routes")
    require(isinstance(sources, dict), "sources must be an object")
    require(isinstance(artifacts, dict), "artifacts must be an object")
    require(isinstance(routes, list), "routes must be an array")

    for source_id, source in sources.items():
        require(isinstance(source, dict), f"source {source_id} must be an object")
        path = source.get("path")
        require(path, f"source {source_id} missing path")
        full_path = (source_root / path).resolve()
        require(source_root in full_path.parents or full_path == source_root, f"source {source_id} escapes source root")
        require(full_path.exists(), f"source {source_id} path does not exist: {path}")

    for artifact_id, artifact in artifacts.items():
        require(isinstance(artifact, dict), f"artifact {artifact_id} must be an object")
        require(artifact.get("format") in ("loom-bytecode", "amdgpu-hsaco"), f"artifact {artifact_id} has unsupported format")
        path = artifact.get("path")
        require(path, f"artifact {artifact_id} missing path")
        full_path = (artifact_root / path).resolve()
        require(artifact_root in full_path.parents or full_path == artifact_root, f"artifact {artifact_id} escapes artifact root")
        if args.require_artifacts:
            require(full_path.exists(), f"artifact {artifact_id} path does not exist: {path}")

    source_ids = set(sources.keys())
    artifact_ids = set(artifacts.keys())
    route_ids = set()
    for route in routes:
        require(isinstance(route, dict), "route entries must be objects")
        route_id = route.get("id")
        require(route_id and route_id not in route_ids, "route id must be present and unique")
        route_ids.add(route_id)
        require(route.get("source_id") in source_ids, f"route {route_id} references unknown source")
        require(route.get("artifact_id") in artifact_ids, f"route {route_id} references unknown artifact")
        require(route.get("root_symbol", "").startswith("@"), f"route {route_id} root_symbol must be a symbol")
        require(route.get("export_name"), f"route {route_id} missing export_name")
        abi = route.get("abi", {})
        dispatch = route.get("dispatch", {})
        require(abi.get("binding_count", 0) > 0, f"route {route_id} missing binding_count")
        require(abi.get("parameter_count", 0) > 0, f"route {route_id} missing parameter_count")
        require(abi.get("constant_byte_length", -1) >= 0, f"route {route_id} missing constant_byte_length")
        require(len(dispatch.get("workgroup_size", [])) == 3, f"route {route_id} dispatch.workgroup_size must have 3 values")


if __name__ == "__main__":
    main()
