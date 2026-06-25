#!/usr/bin/env python3
"""Assemble the split HRX catalog into one build-time JSON artifact."""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any


TOP_LEVEL_METADATA_KEYS = ("schema", "catalog_id", "generated_at", "targets")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f, object_pairs_hook=OrderedDict)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def require_unique(kind: str, entries: list[dict[str, Any]]) -> None:
    seen: set[str] = set()
    for entry in entries:
        entry_id = entry.get("id")
        require(isinstance(entry_id, str) and entry_id, f"{kind} entry missing id")
        require(entry_id not in seen, f"duplicate {kind} id: {entry_id}")
        seen.add(entry_id)


def require_unique_key(kind: str, entries: list[dict[str, Any]], key: str) -> None:
    seen: set[str] = set()
    for entry in entries:
        entry_id = entry.get(key)
        require(isinstance(entry_id, str) and entry_id, f"{kind} entry missing {key}")
        require(entry_id not in seen, f"duplicate {kind} {key}: {entry_id}")
        seen.add(entry_id)


def load_object(path: Path, name: str) -> OrderedDict[str, Any]:
    value = load_json(path)
    require(isinstance(value, dict), f"{name} must be an object: {path}")
    return value


def load_array(path: Path, name: str) -> list[Any]:
    value = load_json(path)
    require(isinstance(value, list), f"{name} must be an array: {path}")
    return value


def load_routes(catalog_dir: Path) -> list[dict[str, Any]]:
    route_dir = catalog_dir / "routes"
    require(route_dir.is_dir(), f"missing routes directory: {route_dir}")
    routes: list[dict[str, Any]] = []
    index_path = route_dir / "index.json"
    route_paths = sorted(path for path in route_dir.rglob("*.json") if path.name != "index.json")
    for path in route_paths:
        value = load_json(path)
        if isinstance(value, dict) and "routes" in value:
            value = value["routes"]
        require(isinstance(value, list), f"route file must contain an array or routes object: {path}")
        for route in value:
            require(isinstance(route, dict), f"route entry must be an object: {path}")
            routes.append(route)
    require_unique("route", routes)
    if index_path.exists():
        route_order = load_array(index_path, "routes index")
        require(all(isinstance(item, str) for item in route_order), "routes index must contain route ids")
        route_by_id = {route["id"]: route for route in routes}
        require(set(route_order) == set(route_by_id), "routes index must contain exactly the route ids")
        routes = [route_by_id[route_id] for route_id in route_order]
    return routes


def load_fusions(catalog_dir: Path) -> list[dict[str, Any]]:
    fusion_dir = catalog_dir / "fusions"
    if not fusion_dir.exists():
        return []
    fusions: list[dict[str, Any]] = []
    for path in sorted(fusion_dir.rglob("*.json")):
        value = load_json(path)
        if isinstance(value, dict) and "fusions" in value:
            value = value["fusions"]
        elif isinstance(value, dict):
            value = [value]
        require(isinstance(value, list), f"fusion file must contain an object, array, or fusions object: {path}")
        for fusion in value:
            require(isinstance(fusion, dict), f"fusion entry must be an object: {path}")
            fusions.append(fusion)
    require_unique("fusion", fusions)
    return fusions


def assemble_legacy(catalog_dir: Path, metadata: OrderedDict[str, Any]) -> OrderedDict[str, Any]:
    catalog: OrderedDict[str, Any] = OrderedDict()
    for key in TOP_LEVEL_METADATA_KEYS:
        require(key in metadata, f"metadata missing {key}")
        catalog[key] = metadata[key]
    require(catalog["schema"] == "ggml-hrx-catalog-v1", "metadata schema must be ggml-hrx-catalog-v1")
    catalog["sources"] = load_object(catalog_dir / "sources.json", "sources")
    catalog["artifacts"] = load_object(catalog_dir / "artifacts.json", "artifacts")
    catalog["families"] = load_array(catalog_dir / "families.json", "families")
    catalog["routes"] = load_routes(catalog_dir)
    catalog["fusions"] = load_fusions(catalog_dir)
    catalog["test_schedules"] = []
    return catalog


def load_split_defs(catalog_dir: Path) -> tuple[OrderedDict[str, Any], OrderedDict[str, Any], list[dict[str, Any]]]:
    defs_dir = catalog_dir / "defs"
    require(defs_dir.is_dir(), f"missing defs directory: {defs_dir}")
    sources: OrderedDict[str, Any] = OrderedDict()
    artifacts: OrderedDict[str, Any] = OrderedDict()
    families: list[dict[str, Any]] = []
    for path in sorted(defs_dir.glob("*.json")):
        value = load_object(path, "kernel definition")
        require(value.get("schema") == "ggml-hrx-kernel-def-v1", f"unsupported kernel definition schema: {path}")
        family_id = value.get("family")
        require(isinstance(family_id, str) and family_id, f"definition missing family: {path}")
        families.append({
            "family": family_id,
            "op": value.get("op"),
            "definition_path": str(path.relative_to(catalog_dir)),
        })
        for source_id, source in value.get("sources", {}).items():
            require(isinstance(source, dict), f"source {source_id} in {path} must be an object")
            if source_id in sources:
                require(sources[source_id] == source, f"conflicting source definition for {source_id}")
            else:
                sources[source_id] = source
        for artifact_id, artifact in value.get("artifacts", {}).items():
            require(isinstance(artifact, dict), f"artifact {artifact_id} in {path} must be an object")
            if artifact_id in artifacts:
                require(artifacts[artifact_id] == artifact, f"conflicting artifact definition for {artifact_id}")
            else:
                artifacts[artifact_id] = artifact
    require_unique_key("family", families, "family")
    return sources, artifacts, families


def load_split_routes(catalog_dir: Path) -> list[dict[str, Any]]:
    target_dir = catalog_dir / "targets"
    if not target_dir.exists():
        return []
    routes: list[dict[str, Any]] = []
    for path in sorted(target_dir.rglob("*.json")):
        value = load_object(path, "target routes")
        require(value.get("schema") == "ggml-hrx-target-routes-v1", f"unsupported target route schema: {path}")
        file_routes = value.get("routes")
        require(isinstance(file_routes, list), f"target route file missing routes array: {path}")
        for index, route in enumerate(file_routes):
            require(isinstance(route, dict), f"route entry must be an object: {path}")
            row = OrderedDict(route)
            row.setdefault("route_file", str(path.relative_to(catalog_dir)))
            row.setdefault("route_ordinal", index)
            routes.append(row)
    require_unique("route", routes)
    return routes


def load_split_test_schedules(catalog_dir: Path) -> list[dict[str, Any]]:
    tests_dir = catalog_dir / "tests"
    if not tests_dir.exists():
        return []
    schedules: list[dict[str, Any]] = []
    case_ids: set[str] = set()
    for path in sorted(tests_dir.rglob("*.json")):
        value = load_object(path, "test schedule")
        require(value.get("schema") == "ggml-hrx-test-schedule-v1", f"unsupported test schedule schema: {path}")
        target_key = value.get("target_key")
        family = value.get("family")
        cases = value.get("cases")
        require(isinstance(target_key, str) and target_key, f"test schedule missing target_key: {path}")
        require(isinstance(family, str) and family, f"test schedule missing family: {path}")
        require(isinstance(cases, list), f"test schedule missing cases array: {path}")
        row = OrderedDict(value)
        row.setdefault("schedule_file", str(path.relative_to(catalog_dir)))
        for case in cases:
            require(isinstance(case, dict), f"test case must be an object: {path}")
            case_id = case.get("id")
            require(isinstance(case_id, str) and case_id, f"test case missing id: {path}")
            require(case_id not in case_ids, f"duplicate test case id: {case_id}")
            case_ids.add(case_id)
        schedules.append(row)
    return schedules


def assemble_split(catalog_dir: Path, metadata: OrderedDict[str, Any]) -> OrderedDict[str, Any]:
    catalog: OrderedDict[str, Any] = OrderedDict()
    for key in TOP_LEVEL_METADATA_KEYS:
        require(key in metadata, f"metadata missing {key}")
        catalog[key] = metadata[key]
    require(catalog["schema"] == "ggml-hrx-catalog-v1", "metadata schema must be ggml-hrx-catalog-v1")
    catalog["layout"] = metadata.get("layout", "ggml-hrx-split-catalog-v1")
    catalog["generator"] = metadata.get("generator")
    catalog["sources"], catalog["artifacts"], catalog["families"] = load_split_defs(catalog_dir)
    catalog["routes"] = load_split_routes(catalog_dir)
    catalog["fusions"] = []
    catalog["test_schedules"] = load_split_test_schedules(catalog_dir)
    return catalog


def assemble(catalog_dir: Path) -> OrderedDict[str, Any]:
    metadata = load_object(catalog_dir / "metadata.json", "metadata")
    if (catalog_dir / "defs").exists() or (catalog_dir / "targets").exists():
        catalog = assemble_split(catalog_dir, metadata)
    else:
        catalog = assemble_legacy(catalog_dir, metadata)

    source_ids = set(catalog["sources"].keys())
    artifact_ids = set(catalog["artifacts"].keys())
    family_ids = {
        family.get("family")
        for family in catalog["families"]
        if isinstance(family, dict) and isinstance(family.get("family"), str)
    }
    for route in catalog["routes"]:
        route_id = route["id"]
        require(route.get("source_id") in source_ids, f"route {route_id} references unknown source")
        require(route.get("artifact_id") in artifact_ids, f"route {route_id} references unknown artifact")
        require(route.get("family") in family_ids, f"route {route_id} references unknown family")
    route_ids = {route["id"] for route in catalog["routes"]}
    for schedule in catalog.get("test_schedules", []):
        family = schedule.get("family")
        require(family in family_ids, f"test schedule {family} references unknown family")
        for case in schedule.get("cases", []):
            expected_route_id = case.get("expected_route_id")
            require(
                expected_route_id in route_ids,
                f"test case {case.get('id')} references unknown expected_route_id {expected_route_id}",
            )
    return catalog


def main() -> None:
    args = parse_args()
    catalog = assemble(args.catalog_dir)
    text = json.dumps(catalog, indent=2) + "\n"
    if args.check:
        require(args.out.exists(), f"assembled catalog missing: {args.out}")
        require(args.out.read_text(encoding="utf-8") == text, f"assembled catalog is stale: {args.out}")
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
