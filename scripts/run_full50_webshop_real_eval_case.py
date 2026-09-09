#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT / "src", REPO_ROOT / "WebShop", REPO_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_full_webshop_real_eval_case as base
import run_single_agent_real_eval_case as baseline
from eval.human_annotated_pilot import load_json_list


def load_all_50_webshop_cases():
    instances = []
    for shard_index in range(1, 6):
        source_path = baseline.SHARD_DIR / f"shard_{shard_index:03d}.json"
        annotated_path = (
            baseline.SHARD_DIR / f"shard_{shard_index:03d}_human_annotated.json"
        )
        source_rows = load_json_list(source_path)
        annotated_rows = load_json_list(annotated_path)
        if len(source_rows) != 10:
            raise ValueError(f"Expected 10 source cases in {source_path}")
        annotated_by_id = {
            str(item.get("instance_id")): item
            for item in annotated_rows
        }
        for source in source_rows:
            instance_id = str(source.get("instance_id"))
            selected = dict(annotated_by_id.get(instance_id, source))
            selected["_eval_annotation_source"] = (
                "human" if instance_id in annotated_by_id else "source"
            )
            instances.append(selected)
    if len(instances) != 50:
        raise ValueError(f"Expected 50 WebShop cases, found {len(instances)}")
    return instances


base.load_all_webshop_annotations = load_all_50_webshop_cases


if __name__ == "__main__":
    raise SystemExit(base.main())
