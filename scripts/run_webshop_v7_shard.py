#!/usr/bin/env python3
"""Run one deterministic shard of the full WebShop v7 simulation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def _load_list(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list")
    return payload


def _instance_ids(items: list[dict[str, Any]]) -> list[str]:
    ids = [str(item.get("instance_id") or "") for item in items]
    if any(not instance_id for instance_id in ids):
        raise ValueError("Every task must have a non-empty instance_id")
    if len(ids) != len(set(ids)):
        raise ValueError("Task instance_ids must be unique")
    return ids


def _valid_existing_output(path: Path, expected_ids: list[str]) -> bool:
    if not path.exists():
        return False
    try:
        items = _load_list(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return _instance_ids(items) == expected_ids


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--tasks-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=50)
    parser.add_argument("--base-seed", type=int, default=201)
    args = parser.parse_args()

    if args.num_shards < 1:
        raise ValueError("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index is outside the shard range")

    tasks = _load_list(args.tasks_path)
    all_ids = _instance_ids(tasks)
    start = len(tasks) * args.shard_index // args.num_shards
    end = len(tasks) * (args.shard_index + 1) // args.num_shards
    shard_ids = all_ids[start:end]
    if not shard_ids:
        raise ValueError(f"Shard {args.shard_index} is empty")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"shard_{args.shard_index:03d}.json"
    if _valid_existing_output(output_path, shard_ids):
        print(f"Shard {args.shard_index} already complete: {output_path}", flush=True)
        return 0

    command = [
        sys.executable,
        str(args.repo_root / "src/simulation/simulation/run_simulation.py"),
        "--output",
        str(output_path),
        "--domain",
        "webshop",
        "--tasks_path",
        str(args.tasks_path),
        "--instance_ids",
        ",".join(shard_ids),
        "--num_instances",
        str(len(shard_ids)),
        "--max_turns",
        "6",
        "--webshop_num_products",
        "100000",
        "--parallelism",
        str(len(shard_ids)),
        "--executor_type",
        "gold",
        "--enable_reranking",
        "true",
        "--multi_candidate_samples",
        "4",
        "--max_multi_candidate_samples",
        "12",
        "--shift_distribution_baseline",
        str(args.tasks_path),
        "--distribution_control_mode",
        "prompt",
        "--distribution_balance_strength",
        "6",
        "--seed",
        str(args.base_seed + start),
    ]
    print(
        f"Running shard {args.shard_index}/{args.num_shards}: "
        f"tasks[{start}:{end}], seed={args.base_seed + start}",
        flush=True,
    )
    subprocess.run(command, cwd=args.repo_root, check=True)
    if not _valid_existing_output(output_path, shard_ids):
        raise RuntimeError(f"Shard output validation failed: {output_path}")
    print(f"Validated shard {args.shard_index}: {len(shard_ids)} instances", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
