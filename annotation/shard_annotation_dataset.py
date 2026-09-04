from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

try:
    from annotation.replay_server import set_initial_constraints_must_have
except ModuleNotFoundError:
    from replay_server import set_initial_constraints_must_have


def load_instances(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"Expected a JSON list of trajectories in {path}")
    return value


def by_instance_id(instances: List[Dict[str, Any]], label: str) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for item in instances:
        instance_id = str(item.get("instance_id") or "").strip()
        if not instance_id:
            raise ValueError(f"Missing instance_id in {label}")
        if instance_id in result:
            raise ValueError(f"Duplicate instance_id {instance_id!r} in {label}")
        result[instance_id] = item
    return result


def build_shards(
    source_instances: List[Dict[str, Any]],
    annotation_instances: List[Dict[str, Any]],
    shard_size: int,
) -> List[Dict[str, Any]]:
    if shard_size < 1:
        raise ValueError("shard_size must be positive")
    source_by_id = by_instance_id(source_instances, "source dataset")
    by_instance_id(annotation_instances, "annotation dataset")
    missing = [
        item["instance_id"]
        for item in annotation_instances
        if item["instance_id"] not in source_by_id
    ]
    if missing:
        raise ValueError(f"Annotation trajectories missing from source dataset: {missing}")

    shards: List[Dict[str, Any]] = []
    for start in range(0, len(annotation_instances), shard_size):
        annotation_shard = annotation_instances[start : start + shard_size]
        instance_ids = [item["instance_id"] for item in annotation_shard]
        source_shard = [source_by_id[instance_id] for instance_id in instance_ids]
        shards.append(
            {
                "index": len(shards) + 1,
                "start_offset": start,
                "count": len(annotation_shard),
                "first_instance_id": instance_ids[0],
                "last_instance_id": instance_ids[-1],
                "instance_ids": instance_ids,
                "source": source_shard,
                "annotation": annotation_shard,
            }
        )
    return shards


def save_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split a replay source and its annotation working copy into aligned shards.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--annotation-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=60)
    parser.add_argument(
        "--preserve-existing-annotations",
        action="store_true",
        help="Keep existing annotation shard content and only enforce all-high Turn 0 priorities.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shards = build_shards(
        load_instances(args.source),
        load_instances(args.annotation),
        args.shard_size,
    )
    manifest_shards = []
    for shard in shards:
        name = f"shard_{shard['index']:03d}"
        source_path = args.source_dir / f"{name}.json"
        annotation_path = args.annotation_dir / f"{name}_human_annotated.json"
        save_json_atomic(source_path, shard["source"])
        if args.preserve_existing_annotations and annotation_path.is_file():
            existing_annotation = load_instances(annotation_path)
            expected_ids = [item["instance_id"] for item in shard["annotation"]]
            existing_ids = [item["instance_id"] for item in existing_annotation]
            if existing_ids != expected_ids:
                raise ValueError(
                    f"Cannot preserve {annotation_path}: instance IDs do not match its source shard"
                )
            set_initial_constraints_must_have(existing_annotation)
            save_json_atomic(annotation_path, existing_annotation)
        else:
            save_json_atomic(annotation_path, shard["annotation"])
        manifest_entry = {
            key: shard[key]
            for key in (
                "index",
                "start_offset",
                "count",
                "first_instance_id",
                "last_instance_id",
                "instance_ids",
            )
        }
        manifest_entry.update({"source": str(source_path), "annotation": str(annotation_path)})
        manifest_shards.append(manifest_entry)
    manifest = {
        "shard_size": args.shard_size,
        "total_count": sum(shard["count"] for shard in shards),
        "shard_count": len(shards),
        "shards": manifest_shards,
    }
    save_json_atomic(args.manifest, manifest)
    print(json.dumps({key: manifest[key] for key in ("shard_size", "total_count", "shard_count")}, indent=2))
    print("counts:", [shard["count"] for shard in shards])


if __name__ == "__main__":
    main()
