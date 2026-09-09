#!/usr/bin/env python3
"""Strictly validate and merge deterministic WebShop simulation shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def _load_list(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list")
    return payload


def _ids(items: list[dict[str, Any]]) -> list[str]:
    values = [str(item.get("instance_id") or "") for item in items]
    if any(not value for value in values):
        raise ValueError("Every item must have a non-empty instance_id")
    return values


def _write_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks-path", type=Path, required=True)
    parser.add_argument("--shards-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    tasks = _load_list(args.tasks_path)
    expected_ids = _ids(tasks)
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("Expected task instance_ids are not unique")

    merged_by_id: dict[str, dict[str, Any]] = {}
    shard_summaries = []
    for shard_index in range(args.num_shards):
        start = len(tasks) * shard_index // args.num_shards
        end = len(tasks) * (shard_index + 1) // args.num_shards
        expected_shard_ids = expected_ids[start:end]
        shard_path = args.shards_dir / f"shard_{shard_index:03d}.json"
        items = _load_list(shard_path)
        actual_ids = _ids(items)
        if actual_ids != expected_shard_ids:
            raise ValueError(
                f"Shard {shard_index} IDs differ from tasks[{start}:{end}]"
            )
        for item in items:
            instance_id = str(item["instance_id"])
            if instance_id in merged_by_id:
                raise ValueError(f"Duplicate instance_id: {instance_id}")
            merged_by_id[instance_id] = item
        shard_summaries.append(
            {
                "shard_index": shard_index,
                "start": start,
                "end": end,
                "count": len(items),
                "path": str(shard_path),
            }
        )

    missing = [instance_id for instance_id in expected_ids if instance_id not in merged_by_id]
    extras = sorted(set(merged_by_id) - set(expected_ids))
    if missing or extras or len(merged_by_id) != len(expected_ids):
        raise ValueError(
            f"Merge coverage failed: missing={missing}, extras={extras}, "
            f"actual={len(merged_by_id)}, expected={len(expected_ids)}"
        )

    merged = [merged_by_id[instance_id] for instance_id in expected_ids]
    _write_atomic(args.output, merged)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    manifest = {
        "output": str(args.output),
        "sha256": digest,
        "expected_instances": len(expected_ids),
        "merged_instances": len(merged),
        "unique_instances": len(set(_ids(merged))),
        "num_shards": args.num_shards,
        "shards": shard_summaries,
    }
    _write_atomic(args.manifest, manifest)
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
