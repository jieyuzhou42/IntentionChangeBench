#!/usr/bin/env python3
"""Coordinate 50 WebShop shards across five allocated Slurm nodes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from run_webshop_v2_formal import _summarize


REPO = Path(__file__).resolve().parents[1]
PYTHON = Path("/fsx/sihengx/miniforge3/envs/intention-change-bench/bin/python")
RUNNER = REPO / "scripts/run_webshop_v2_formal.py"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--node-groups", type=int, default=5)
    parser.add_argument("--shards-per-node", type=int, default=10)
    parser.add_argument("--shard-size", type=int, default=7)
    parser.add_argument("--max-worker-attempts", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    runtime = args.runtime_dir.resolve()
    instances = _load(source)
    expected = args.node_groups * args.shards_per_node * args.shard_size
    if len(instances) != expected:
        raise ValueError(f"Expected {expected} source cases, got {len(instances)}")
    expected_ids = [str(item.get("instance_id")) for item in instances]
    if len(set(expected_ids)) != expected:
        raise ValueError("Source instance IDs are not unique")

    runtime.mkdir(parents=True, exist_ok=True)
    status = {
        "state": "running",
        "started_at": _utc_now(),
        "source": str(source),
        "output": str(output),
        "node_groups": args.node_groups,
        "shards_per_node": args.shards_per_node,
        "total_shards": args.node_groups * args.shards_per_node,
        "groups": [],
    }
    _write(runtime / "status.json", status)

    group_size = args.shards_per_node * args.shard_size
    processes = []
    for group_index in range(args.node_groups):
        start = group_index * group_size
        group_instances = instances[start : start + group_size]
        group_dir = runtime / "groups" / f"group_{group_index:02d}"
        group_source = group_dir / "source.json"
        group_output = group_dir / "output.json"
        group_runtime = group_dir / "runtime"
        launcher_log = group_dir / "launcher.log"
        _write(group_source, group_instances)
        launcher_log.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "srun",
            "--exclusive",
            "--nodes=1",
            "--ntasks=1",
            str(PYTHON),
            str(RUNNER),
            "--source",
            str(group_source),
            "--distribution-baseline",
            str(source),
            "--output",
            str(group_output),
            "--runtime-dir",
            str(group_runtime),
            "--shard-size",
            str(args.shard_size),
            "--workers",
            str(args.shards_per_node),
            "--max-worker-attempts",
            str(args.max_worker_attempts),
            "--seed",
            str(args.seed + start),
        ]
        log = launcher_log.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=REPO,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append((group_index, process, log, group_output))

    failures = []
    for group_index, process, log, group_output in processes:
        returncode = process.wait()
        log.close()
        group_status = {
            "group": group_index,
            "returncode": returncode,
            "output": str(group_output),
        }
        status["groups"].append(group_status)
        _write(runtime / "status.json", status)
        if returncode != 0:
            failures.append(group_status)

    if failures:
        status.update(state="failed", finished_at=_utc_now(), failures=failures)
        _write(runtime / "FAILED.json", status)
        _write(runtime / "status.json", status)
        return 1

    merged = []
    for group_index in range(args.node_groups):
        group_output = runtime / "groups" / f"group_{group_index:02d}" / "output.json"
        merged.extend(_load(group_output))
    actual_ids = [str(item.get("instance_id")) for item in merged]
    if actual_ids != expected_ids:
        raise ValueError("Merged output IDs or order do not match the source")
    if any(len(item.get("turns", [])) != 7 for item in merged):
        raise ValueError("Every completed case must contain exactly seven turns")
    _write(output, merged)

    status.update(
        state="complete",
        completed_at=_utc_now(),
        summary=_summarize(merged),
    )
    _write(runtime / "status.json", status)
    _write(runtime / "COMPLETE.json", status)
    print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
