#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Validate the HRX catalog.")
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--require-artifacts", action="store_true")
    return parser.parse_args()


def require(condition, message):
    if not condition:
        raise SystemExit(message)


def main():
    args = parse_args()
    source_root = args.source_root.resolve()
    artifact_root = args.artifact_root.resolve() if args.artifact_root else source_root
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))

    require(catalog.get("schema") == "ggml-hrx-catalog-v1", "catalog schema must be ggml-hrx-catalog-v1")
    sources = catalog.get("sources")
    artifacts = catalog.get("artifacts")
    routes = catalog.get("routes")
    fusions = catalog.get("fusions", [])
    test_schedules = catalog.get("test_schedules", [])
    require(isinstance(sources, dict), "sources must be an object")
    require(isinstance(artifacts, dict), "artifacts must be an object")
    require(isinstance(routes, list), "routes must be an array")
    require(isinstance(fusions, list), "fusions must be an array")
    require(isinstance(test_schedules, list), "test_schedules must be an array")

    for source_id, source in sources.items():
        require(isinstance(source, dict), f"source {source_id} must be an object")
        path = source.get("path")
        require(path, f"source {source_id} missing path")
        full_path = (source_root / path).resolve()
        require(source_root in full_path.parents or full_path == source_root, f"source {source_id} escapes source root")
        require(full_path.exists(), f"source {source_id} path does not exist: {path}")

    for artifact_id, artifact in artifacts.items():
        require(isinstance(artifact, dict), f"artifact {artifact_id} must be an object")
        require(
            artifact.get("format") in ("loom-bytecode", "amdgpu-hsaco"),
            f"artifact {artifact_id} has unsupported format",
        )
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

    fusion_ids = set()
    for fusion in fusions:
        require(isinstance(fusion, dict), "fusion entries must be objects")
        fusion_id = fusion.get("id")
        require(fusion_id and fusion_id not in fusion_ids, "fusion id must be present and unique")
        fusion_ids.add(fusion_id)

    test_case_ids = set()
    for schedule in test_schedules:
        require(isinstance(schedule, dict), "test_schedules entries must be objects")
        require(schedule.get("schema") == "ggml-hrx-test-schedule-v1", "test schedule has unsupported schema")
        require(schedule.get("target_key"), "test schedule missing target_key")
        require(schedule.get("family"), "test schedule missing family")
        cases = schedule.get("cases")
        require(isinstance(cases, list), "test schedule cases must be an array")
        for case in cases:
            require(isinstance(case, dict), "test case entries must be objects")
            case_id = case.get("id")
            require(case_id and case_id not in test_case_ids, "test case id must be present and unique")
            test_case_ids.add(case_id)
            require(case.get("expected_route_id") in route_ids, f"test case {case_id} references unknown route")
            require(case.get("op"), f"test case {case_id} missing op")
            require(isinstance(case.get("shape"), dict), f"test case {case_id} missing shape")
            require(isinstance(case.get("supports"), dict), f"test case {case_id} missing supports")


if __name__ == "__main__":
    main()
