#!/usr/bin/env python3
"""Run the 350-case WebShop formal dataset in configurable resilient shards."""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import os
import subprocess
import sys
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO_ROOT / "data/simulation/webshop_diverse_350_formal.json"
DEFAULT_OUTPUT = REPO_ROOT / "data/simulation/webshop_v2_350_formal.json"
DEFAULT_RUNTIME_DIR = REPO_ROOT / "data/simulation/webshop_v2_runtime"
PYTHON = Path("/fsx/sihengx/miniforge3/envs/intention-change-bench/bin/python")
RUN_SIMULATION = REPO_ROOT / "src/simulation/simulation/run_simulation.py"
def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_instances(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise ValueError(f"{path} must contain a JSON array of objects")
    return payload


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _task_payload(instance: Dict[str, Any]) -> Dict[str, Any]:
    turns = instance.get("turns") or []
    first_turn = turns[0] if turns else {}
    initial_intention = first_turn.get("gold_current_intention")
    if not isinstance(initial_intention, dict):
        raise ValueError(f"{instance.get('instance_id')} has no initial intention")
    return {
        "instance_id": instance["instance_id"],
        "task_type": instance.get("task_type", "transaction"),
        "subtype": instance.get("subtype", "shopping"),
        "world_state": copy.deepcopy(instance.get("world_state") or {"domain": "webshop"}),
        "initial_intention": copy.deepcopy(initial_intention),
    }


def _validate_output(path: Path, expected_ids: List[str]) -> List[Dict[str, Any]]:
    instances = _load_instances(path)
    actual_ids = [str(item.get("instance_id")) for item in instances]
    if actual_ids != expected_ids:
        raise ValueError(
            f"{path} has unexpected IDs: expected {len(expected_ids)}, got {len(actual_ids)}"
        )
    if any(not isinstance(item.get("turns"), list) or not item["turns"] for item in instances):
        raise ValueError(f"{path} contains an instance without turns")
    return instances


def _run_shard(
    shard_index: int,
    task_path: Path,
    expected_ids: List[str],
    output_path: Path,
    runtime_dir: Path,
    distribution_baseline: Path,
    max_attempts: int,
    base_seed: int,
) -> Dict[str, Any]:
    try:
        _validate_output(output_path, expected_ids)
        return {"shard": shard_index, "state": "already_complete", "attempts": 0}
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        pass

    log_path = runtime_dir / "logs" / f"shard_{shard_index:02d}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start_index = shard_index * len(expected_ids)
    last_error = ""
    for attempt in range(1, max_attempts + 1):
        attempt_output = output_path.with_suffix(f".attempt{attempt}.json")
        attempt_output.unlink(missing_ok=True)
        command = [
            str(PYTHON),
            str(RUN_SIMULATION),
            "--output",
            str(attempt_output),
            "--domain",
            "webshop",
            "--tasks_path",
            str(task_path),
            "--num_instances",
            str(len(expected_ids)),
            "--max_turns",
            "6",
            "--webshop_num_products",
            "100000",
            "--parallelism",
            "1",
            "--executor_type",
            "gold",
            "--enable_reranking",
            "true",
            "--multi_change_rate",
            "0.30",
            "--multi_candidate_samples",
            "4",
            "--max_multi_candidate_samples",
            "12",
            "--shift_distribution_baseline",
            str(distribution_baseline),
            "--distribution_control_mode",
            "prompt",
            "--distribution_balance_strength",
            "6",
            "--seed",
            str(base_seed + start_index),
        ]
        with log_path.open("a", encoding="utf-8") as log:
            log.write(
                f"\n[{_utc_now()}] shard={shard_index} attempt={attempt} "
                f"cases={len(expected_ids)}\n"
            )
            log.flush()
            try:
                completed = subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=8 * 60 * 60,
                    check=False,
                )
                if completed.returncode != 0:
                    raise RuntimeError(f"run_simulation exited {completed.returncode}")
                _validate_output(attempt_output, expected_ids)
                os.replace(attempt_output, output_path)
                return {"shard": shard_index, "state": "complete", "attempts": attempt}
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                log.write(f"[{_utc_now()}] failed: {last_error}\n")
                log.flush()
                attempt_output.unlink(missing_ok=True)
        if attempt < max_attempts:
            time.sleep(min(30 * attempt, 90))
    return {
        "shard": shard_index,
        "state": "failed",
        "attempts": max_attempts,
        "error": last_error,
    }


def _summarize(instances: List[Dict[str, Any]]) -> Dict[str, Any]:
    categories: Counter[str] = Counter()
    conditions: Counter[str] = Counter()
    multi_turns = 0
    total_shift_turns = 0
    for instance in instances:
        for turn in instance.get("turns", []):
            shift = turn.get("shift_condition") or {}
            if not shift:
                continue
            total_shift_turns += 1
            conditions[str(shift.get("type") or "none")] += 1
            details = shift.get("details") or {}
            changes = details.get("changes") or []
            if len(changes) > 1:
                multi_turns += 1
            if changes:
                for change in changes:
                    if isinstance(change, dict):
                        categories[
                            str(change.get("change_category") or change.get("op") or "none")
                        ] += 1
            else:
                categories[
                    str(details.get("change_category") or details.get("op") or "none")
                ] += 1
    return {
        "instances": len(instances),
        "turns": sum(len(item.get("turns", [])) for item in instances),
        "shift_turns": total_shift_turns,
        "multi_change_turns": multi_turns,
        "category_counts": dict(sorted(categories.items())),
        "condition_counts": dict(sorted(conditions.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--distribution-baseline", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME_DIR)
    parser.add_argument("--shard-size", type=int, default=50)
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--max-worker-attempts", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source = args.source.resolve()
    distribution_baseline = (
        args.distribution_baseline.resolve()
        if args.distribution_baseline is not None
        else source
    )
    output = args.output.resolve()
    runtime_dir = args.runtime_dir.resolve()
    runtime_dir.mkdir(parents=True, exist_ok=True)
    (runtime_dir / "COMPLETE.json").unlink(missing_ok=True)
    (runtime_dir / "FAILED.json").unlink(missing_ok=True)

    source_instances = _load_instances(source)
    expected_cases = len(source_instances)
    if expected_cases < 1:
        raise ValueError("Source must contain at least one case")
    expected_ids = [str(item.get("instance_id")) for item in source_instances]
    if len(set(expected_ids)) != expected_cases:
        raise ValueError("Source instance IDs are not unique")

    if output.exists():
        try:
            final_instances = _validate_output(output, expected_ids)
            completion = {
                "state": "already_complete",
                "completed_at": _utc_now(),
                "output": str(output),
                "model": os.getenv("BEDROCK_MODEL"),
                "summary": _summarize(final_instances),
            }
            _atomic_json(runtime_dir / "COMPLETE.json", completion)
            print(json.dumps(completion, indent=2))
            return 0
        except Exception:
            backup = output.with_suffix(f".invalid-{int(time.time())}.json")
            os.replace(output, backup)

    shard_specs = []
    for shard_index, start in enumerate(range(0, expected_cases, args.shard_size)):
        shard_source = source_instances[start : start + args.shard_size]
        shard_tasks = [_task_payload(instance) for instance in shard_source]
        task_path = runtime_dir / "tasks" / f"tasks_{shard_index:02d}.json"
        output_path = runtime_dir / "parts" / f"part_{shard_index:02d}.json"
        _atomic_json(task_path, shard_tasks)
        shard_specs.append(
            (
                shard_index,
                task_path,
                [item["instance_id"] for item in shard_tasks],
                output_path,
            )
        )

    status_lock = threading.Lock()
    status: Dict[str, Any] = {
        "state": "running",
        "started_at": _utc_now(),
        "model": os.getenv("BEDROCK_MODEL"),
        "source": str(source),
        "output": str(output),
        "shards": [],
    }
    _atomic_json(runtime_dir / "status.json", status)

    results = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(args.workers, len(shard_specs))
    ) as executor:
        future_to_shard = {
            executor.submit(
                _run_shard,
                shard_index,
                task_path,
                shard_ids,
                output_path,
                runtime_dir,
                distribution_baseline,
                args.max_worker_attempts,
                args.seed,
            ): shard_index
            for shard_index, task_path, shard_ids, output_path in shard_specs
        }
        for future in concurrent.futures.as_completed(future_to_shard):
            result = future.result()
            results.append(result)
            with status_lock:
                status["shards"] = sorted(results, key=lambda item: item["shard"])
                _atomic_json(runtime_dir / "status.json", status)

    failures = [result for result in results if result["state"] == "failed"]
    if failures:
        failure = {
            **status,
            "state": "failed",
            "finished_at": _utc_now(),
            "failures": failures,
        }
        _atomic_json(runtime_dir / "FAILED.json", failure)
        print(json.dumps(failure, indent=2), file=sys.stderr)
        return 1

    merged: List[Dict[str, Any]] = []
    for shard_index, _task_path, shard_ids, output_path in shard_specs:
        merged.extend(_validate_output(output_path, shard_ids))
    if [str(item.get("instance_id")) for item in merged] != expected_ids:
        raise ValueError("Merged output order does not match the source")
    _atomic_json(output, merged)
    _validate_output(output, expected_ids)

    completion = {
        **status,
        "state": "complete",
        "completed_at": _utc_now(),
        "shards": sorted(results, key=lambda item: item["shard"]),
        "summary": _summarize(merged),
    }
    _atomic_json(runtime_dir / "status.json", completion)
    _atomic_json(runtime_dir / "COMPLETE.json", completion)
    print(json.dumps(completion, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
