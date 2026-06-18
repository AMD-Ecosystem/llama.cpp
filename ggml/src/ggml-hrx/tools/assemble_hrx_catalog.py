#!/usr/bin/env python3
"""Assemble the split HRX v1 HIP catalog into one JSON object."""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any


TOP_LEVEL_METADATA_KEYS = ("schema", "catalog_id", "generated_at", "targets")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f, object_pairs_hook=OrderedDict)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, indent=2)
        f.write("\n")


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


def load_routes(catalog_dir: Path) -> list[dict[str, Any]]:
    route_dir = catalog_dir / "routes"
    require(route_dir.is_dir(), f"missing routes directory: {route_dir}")
    routes: list[dict[str, Any]] = []
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

    index_path = route_dir / "index.json"
    if index_path.exists():
        route_order = load_json(index_path)
        require(isinstance(route_order, list), "routes index must be an array")
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


def assemble(catalog_dir: Path) -> OrderedDict[str, Any]:
    metadata = load_json(catalog_dir / "metadata.json")
    require(isinstance(metadata, dict), "metadata must be an object")
    catalog: OrderedDict[str, Any] = OrderedDict()
    for key in TOP_LEVEL_METADATA_KEYS:
        require(key in metadata, f"metadata missing {key}")
        catalog[key] = metadata[key]
    catalog["sources"] = load_json(catalog_dir / "sources.json")
    catalog["artifacts"] = load_json(catalog_dir / "artifacts.json")
    catalog["families"] = load_json(catalog_dir / "families.json")
    catalog["routes"] = load_routes(catalog_dir)
    catalog["fusions"] = load_fusions(catalog_dir)

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
    return catalog


def split_from(source_catalog: Path, catalog_dir: Path) -> None:
    catalog = load_json(source_catalog)
    metadata = OrderedDict((key, catalog[key]) for key in TOP_LEVEL_METADATA_KEYS if key in catalog)
    write_json(catalog_dir / "metadata.json", metadata)
    write_json(catalog_dir / "sources.json", catalog.get("sources", OrderedDict()))
    write_json(catalog_dir / "artifacts.json", catalog.get("artifacts", OrderedDict()))
    write_json(catalog_dir / "families.json", catalog.get("families", []))

    routes_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for route in catalog.get("routes", []):
        routes_by_family[route.get("family", "unknown")].append(route)
    for family, routes in routes_by_family.items():
        write_json(catalog_dir / "routes" / f"{family}.json", routes)
    write_json(catalog_dir / "routes" / "index.json", [route["id"] for route in catalog.get("routes", [])])
    write_json(catalog_dir / "fusions" / "candidates.json", catalog.get("fusions", []))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-dir", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--split-from", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if args.split_from:
        split_from(args.split_from, args.catalog_dir)
        return
    require(args.out is not None, "--out is required when assembling")
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
