#!/usr/bin/env python3
import argparse
import json
import pathlib
import re

BEGIN = "GGML_VK_RADV_PIPELINE_BEGIN "
END = "GGML_VK_RADV_PIPELINE_END "


def slug(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def metadata_key(metadata):
    return {
        "pipeline": metadata.get("pipeline", ""),
        "entrypoint": metadata.get("entrypoint", ""),
        "trace_source": metadata.get("trace_source", ""),
        "trace_reduction": metadata.get("trace_reduction", ""),
        "trace_workgroup_mode": metadata.get("trace_workgroup_mode", ""),
        "spv_hash": metadata.get("spv_hash", ""),
        "spv_size": metadata.get("spv_size", 0),
        "push_constant_size": metadata.get("push_constant_size", 0),
        "parameter_count": metadata.get("parameter_count", 0),
        "wg_denoms": metadata.get("wg_denoms", []),
        "spec": metadata.get("spec", []),
        "disable_robustness": metadata.get("disable_robustness", False),
        "require_full_subgroups": metadata.get("require_full_subgroups", False),
        "required_subgroup_size": metadata.get("required_subgroup_size", 0),
        "is_64b_indexing": metadata.get("is_64b_indexing", False),
    }


def extract_isa(lines):
    result = []
    in_disasm = False
    for line in lines:
        stripped = line.strip()
        if stripped == "disasm:":
            in_disasm = True
            result.append(line)
            continue
        if in_disasm and stripped == "Compute Shader:":
            break
        if in_disasm:
            result.append(line)
    return result


def extract_stats(lines):
    result = []
    in_stats = False
    for line in lines:
        stripped = line.strip()
        if stripped == "*** SHADER STATS ***":
            in_stats = True
        if in_stats:
            result.append(line)
            if stripped == "********************":
                break
    return result


def write_block(out_dir, metadata, lines, seen):
    key = json.dumps(metadata_key(metadata), sort_keys=True, separators=(",", ":"))
    if key in seen:
        return
    seen.add(key)

    stem = "__".join([
        slug(metadata.get("pipeline", "pipeline")),
        slug(metadata.get("entrypoint", "main")),
        str(metadata.get("spv_hash", "nohash")).replace("0x", ""),
    ])
    header = [
        "# metadata: " + json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n",
        "\n",
    ]

    for subdir in ["radv/full", "radv/isa", "radv/stats"]:
        (out_dir / subdir).mkdir(parents=True, exist_ok=True)

    (out_dir / "radv/full" / f"{stem}.radv.txt").write_text("".join(header + lines), encoding="utf-8")
    (out_dir / "radv/isa" / f"{stem}.amdgcn.txt").write_text("".join(header + extract_isa(lines)), encoding="utf-8")
    (out_dir / "radv/stats" / f"{stem}.stats.txt").write_text("".join(header + extract_stats(lines)), encoding="utf-8")


def split_log(path, out_dir, seen):
    metadata = None
    lines = []
    for line_no, line in enumerate(path.open("r", encoding="utf-8", errors="replace"), 1):
        if line.startswith(BEGIN):
            if metadata is not None:
                raise SystemExit(f"{path}:{line_no}: nested RADV block")
            metadata = json.loads(line[len(BEGIN):])
            lines = []
            continue
        if line.startswith(END):
            if metadata is None:
                raise SystemExit(f"{path}:{line_no}: END without BEGIN")
            end_metadata = json.loads(line[len(END):])
            if metadata_key(metadata) != metadata_key(end_metadata):
                raise SystemExit(f"{path}:{line_no}: END metadata mismatch")
            write_block(out_dir, metadata, lines, seen)
            metadata = None
            lines = []
            continue
        if metadata is not None:
            lines.append(line)
    if metadata is not None:
        raise SystemExit(f"{path}: unterminated RADV block")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True, type=pathlib.Path)
    parser.add_argument("logs", nargs="+", type=pathlib.Path)
    args = parser.parse_args()

    seen = set()
    for path in args.logs:
        split_log(path, args.out_dir, seen)
    print(f"radv_pipeline_blocks={len(seen)}")


if __name__ == "__main__":
    raise SystemExit(main())
