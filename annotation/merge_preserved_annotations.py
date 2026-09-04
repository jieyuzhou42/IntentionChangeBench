from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_instances(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"Expected a JSON list of trajectory objects in {path}")
    return value


def instance_map(instances: List[Dict[str, Any]], label: str) -> Dict[str, Dict[str, Any]]:
    mapped: Dict[str, Dict[str, Any]] = {}
    for item in instances:
        instance_id = str(item.get("instance_id") or "").strip()
        if not instance_id:
            raise ValueError(f"Missing instance_id in {label}")
        if instance_id in mapped:
            raise ValueError(f"Duplicate instance_id {instance_id!r} in {label}")
        mapped[instance_id] = item
    return mapped


def preserve_and_merge(
    old_source: List[Dict[str, Any]],
    old_annotation: List[Dict[str, Any]],
    new_dataset: List[Dict[str, Any]],
    first_count: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    if first_count < 1 or first_count > len(old_source):
        raise ValueError("first_count must select at least one existing old trajectory")

    original_ids = [str(item.get("instance_id") or "").strip() for item in old_source[:first_count]]
    annotated_by_id = instance_map(old_annotation, "old annotation")
    new_by_id = instance_map(new_dataset, "new dataset")
    missing_from_new = [instance_id for instance_id in original_ids if instance_id not in new_by_id]
    if missing_from_new:
        raise ValueError(f"Original first trajectories missing from new dataset: {missing_from_new}")

    preserved_ids = [instance_id for instance_id in original_ids if instance_id in annotated_by_id]
    deleted_ids = [instance_id for instance_id in original_ids if instance_id not in annotated_by_id]
    preserved = [annotated_by_id[instance_id] for instance_id in preserved_ids]
    preserved_by_id = {item["instance_id"]: item for item in preserved}

    merged: List[Dict[str, Any]] = []
    for new_item in new_dataset:
        instance_id = str(new_item.get("instance_id") or "").strip()
        if instance_id in deleted_ids:
            continue
        merged.append(preserved_by_id.get(instance_id, new_item))

    manifest = {
        "first_count_in_original_order": first_count,
        "original_first_instance_ids": original_ids,
        "preserved_annotated_instance_ids": preserved_ids,
        "deleted_instance_ids": deleted_ids,
        "preserved_count": len(preserved),
        "merged_count": len(merged),
    }
    return preserved, merged, manifest


def save_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preserve annotated trajectories from the beginning of an old dataset and merge them into a regenerated dataset."
    )
    parser.add_argument("--old-source", type=Path, required=True)
    parser.add_argument("--old-annotation", type=Path, required=True)
    parser.add_argument("--new-dataset", type=Path, required=True)
    parser.add_argument("--first-count", type=int, default=10)
    parser.add_argument("--preserved-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--merged-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preserved, merged, manifest = preserve_and_merge(
        load_instances(args.old_source),
        load_instances(args.old_annotation),
        load_instances(args.new_dataset),
        args.first_count,
    )
    save_json_atomic(args.preserved_output, preserved)
    save_json_atomic(args.manifest_output, manifest)
    save_json_atomic(args.merged_output, merged)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
