#!/usr/bin/env python3
"""Rerun WebShop v2 cases containing reranker fallbacks and replace them atomically."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/fsx/sihengx/miniforge3/envs/intention-change-bench/bin/python")
RUN_SIMULATION = REPO_ROOT / "src/simulation/simulation/run_simulation.py"
SOURCE = REPO_ROOT / "data/simulation/webshop_diverse_350_formal.json"
OUTPUT = REPO_ROOT / "data/simulation/webshop_v2_350_formal.json"
RUNTIME_DIR = REPO_ROOT / "data/simulation/webshop_v2_runtime"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON array")
    return payload


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _rerank_failures(instance: Dict[str, Any]) -> List[int]:
    failed_turns = []
    for turn in instance.get("turns", []):
        info = (turn.get("env_feedback") or {}).get("rerank_info") or {}
        if info.get("fallback_used") or info.get("succeeded") is not True:
            failed_turns.append(int(turn.get("turn_id", -1)))
    return failed_turns


def _run_case(instance_id: str, global_index: int, max_attempts: int) -> Dict[str, Any]:
    repair_dir = RUNTIME_DIR / "repairs"
    log_dir = RUNTIME_DIR / "logs"
    repair_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    final_path = repair_dir / f"{instance_id}.json"
    log_path = log_dir / f"repair_{instance_id}.log"

    if final_path.exists():
        existing = _load(final_path)
        if (
            len(existing) == 1
            and existing[0].get("instance_id") == instance_id
            and not _rerank_failures(existing[0])
        ):
            return {"instance_id": instance_id, "attempts": 0, "state": "already_complete"}

    last_error = ""
    for attempt in range(1, max_attempts + 1):
        attempt_path = repair_dir / f"{instance_id}.attempt{attempt}.json"
        attempt_path.unlink(missing_ok=True)
        command = [
            str(PYTHON),
            str(RUN_SIMULATION),
            "--output",
            str(attempt_path),
            "--domain",
            "webshop",
            "--tasks_path",
            str(SOURCE),
            "--instance_ids",
            instance_id,
            "--num_instances",
            "1",
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
            str(SOURCE),
            "--distribution_control_mode",
            "prompt",
            "--distribution_balance_strength",
            "6",
            "--seed",
            str(42 + global_index),
        ]
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n[{_utc_now()}] attempt={attempt}\n")
            log.flush()
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=2 * 60 * 60,
                check=False,
            )
            try:
                instances = _load(attempt_path)
                if completed.returncode != 0:
                    raise RuntimeError(f"run_simulation exited {completed.returncode}")
                if len(instances) != 1 or instances[0].get("instance_id") != instance_id:
                    raise ValueError("repair output has an unexpected instance")
                failed_turns = _rerank_failures(instances[0])
                if failed_turns:
                    raise ValueError(f"reranker fallback remained on turns {failed_turns}")
                os.replace(attempt_path, final_path)
                return {"instance_id": instance_id, "attempts": attempt, "state": "complete"}
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                log.write(f"[{_utc_now()}] failed: {last_error}\n")
                attempt_path.unlink(missing_ok=True)
        if attempt < max_attempts:
            time.sleep(15 * attempt)
    return {
        "instance_id": instance_id,
        "attempts": max_attempts,
        "state": "failed",
        "error": last_error,
    }


def main() -> int:
    source = _load(SOURCE)
    output = _load(OUTPUT)
    source_ids = [str(item.get("instance_id")) for item in source]
    output_ids = [str(item.get("instance_id")) for item in output]
    if source_ids != output_ids or len(output) != 350:
        raise ValueError("Formal output does not match the 350-case source order")

    failed = [item for item in output if _rerank_failures(item)]
    if not failed:
        results: List[Dict[str, Any]] = []
    else:
        index_by_id = {instance_id: index for index, instance_id in enumerate(source_ids)}
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(failed)) as executor:
            results = list(
                executor.map(
                    lambda item: _run_case(
                        str(item["instance_id"]),
                        index_by_id[str(item["instance_id"])],
                        3,
                    ),
                    failed,
                )
            )
        failures = [item for item in results if item["state"] == "failed"]
        if failures:
            _atomic_json(
                RUNTIME_DIR / "REPAIR_FAILED.json",
                {"state": "failed", "finished_at": _utc_now(), "results": results},
            )
            raise RuntimeError(f"Reranker repair failed: {failures}")

        replacements = {
            result["instance_id"]: _load(
                RUNTIME_DIR / "repairs" / f"{result['instance_id']}.json"
            )[0]
            for result in results
        }
        output = [replacements.get(str(item["instance_id"]), item) for item in output]
        remaining = {
            str(item["instance_id"]): _rerank_failures(item)
            for item in output
            if _rerank_failures(item)
        }
        if remaining:
            raise ValueError(f"Reranker fallbacks remain: {remaining}")
        _atomic_json(OUTPUT, output)

    sha256 = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    completion = {
        "state": "complete",
        "completed_at": _utc_now(),
        "repaired_cases": len(failed),
        "results": results,
        "instances": len(output),
        "rerank_calls": sum(len(item.get("turns", [])) for item in output),
        "rerank_fallbacks": sum(len(_rerank_failures(item)) for item in output),
        "sha256": sha256,
    }
    _atomic_json(RUNTIME_DIR / "REPAIR_COMPLETE.json", completion)
    print(json.dumps(completion, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
