from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


BANNED_USER_TERMS = (
    "validator",
    "exact listed wording",
    "participant assignment",
)
ALLOWED_SHIFT_CONDITIONS = {
    "user_preference",
    "real_world_feasibility",
}
ALLOWED_CHANGE_OPS = {
    "add",
    "relax",
    "override",
    "reprioritize",
}


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def _load_tasks(path: Path) -> List[Dict[str, Any]]:
    payload = _load_json(path)
    tasks = payload.get("tasks") if isinstance(payload, dict) else payload
    if not isinstance(tasks, list):
        raise ValueError(f"{path} must contain a task list")
    if not all(isinstance(task, dict) and task.get("instance_id") for task in tasks):
        raise ValueError(f"Every task in {path} must have an instance_id")
    return tasks


def _shard_bounds(total: int, shard_index: int, num_shards: int) -> Tuple[int, int]:
    if not 0 <= shard_index < num_shards:
        raise ValueError(f"shard_index must be in [0, {num_shards})")
    return total * shard_index // num_shards, total * (shard_index + 1) // num_shards


def _valid_instances(
    payload: Any,
    expected_ids: Iterable[str],
    expected_turns: int,
) -> Dict[str, Dict[str, Any]]:
    expected = set(expected_ids)
    if not isinstance(payload, list):
        return {}
    valid: Dict[str, Dict[str, Any]] = {}
    for instance in payload:
        if not isinstance(instance, dict):
            continue
        instance_id = instance.get("instance_id")
        turns = instance.get("turns")
        if (
            instance_id not in expected
            or instance_id in valid
            or not isinstance(turns, list)
            or len(turns) != expected_turns
        ):
            continue
        if [turn.get("turn_id") for turn in turns if isinstance(turn, dict)] != list(
            range(expected_turns)
        ):
            continue
        previous_delta: Dict[str, Any] = {}
        target_turns: Dict[str, List[int]] = {}
        pure_add_streak = 0
        trajectory_is_valid = True
        for turn_index, turn in enumerate(turns):
            if not isinstance(turn, dict) or turn.get("error"):
                trajectory_is_valid = False
                break
            utterance = str(turn.get("user_utterance") or "")
            lowered_utterance = utterance.lower()
            if any(term in lowered_utterance for term in BANNED_USER_TERMS):
                trajectory_is_valid = False
                break
            if "entity_" in lowered_utterance:
                trajectory_is_valid = False
                break
            feedback = turn.get("env_feedback")
            if not isinstance(feedback, dict) or not feedback.get("submitted_plan"):
                trajectory_is_valid = False
                break
            delta = turn.get("gold_delta") or {}
            if turn_index == 0:
                if delta:
                    trajectory_is_valid = False
                    break
                previous_delta = {}
                continue
            if not isinstance(delta, dict) or not delta:
                trajectory_is_valid = False
                break
            shift_condition = turn.get("shift_condition")
            if (
                not isinstance(shift_condition, dict)
                or shift_condition.get("type") not in ALLOWED_SHIFT_CONDITIONS
            ):
                trajectory_is_valid = False
                break
            pure_add = True
            for field, change in delta.items():
                if not field or not isinstance(change, dict):
                    trajectory_is_valid = False
                    break
                op = change.get("op")
                if op not in ALLOWED_CHANGE_OPS:
                    trajectory_is_valid = False
                    break
                pure_add = pure_add and op == "add"
                if field in previous_delta and previous_delta[field].get("op") == op:
                    trajectory_is_valid = False
                    break
                target_turns.setdefault(field, []).append(turn_index)
                if len(target_turns[field]) > 2:
                    trajectory_is_valid = False
                    break
            if not trajectory_is_valid:
                break
            pure_add_streak = pure_add_streak + 1 if pure_add else 0
            if pure_add_streak > 3:
                trajectory_is_valid = False
                break
            current_intention = turn.get("gold_current_intention") or {}
            if current_intention.get("people_number") == 1 and any(
                field.startswith("entities.") for field in delta
            ):
                trajectory_is_valid = False
                break
            previous_delta = delta
        if not trajectory_is_valid:
            continue
        valid[instance_id] = instance
    return valid


def _load_valid_output(
    path: Path,
    expected_ids: Sequence[str],
    expected_turns: int,
) -> Dict[str, Dict[str, Any]]:
    if not path.is_file():
        return {}
    try:
        return _valid_instances(_load_json(path), expected_ids, expected_turns)
    except (OSError, json.JSONDecodeError):
        return {}


def _run_cases(
    *,
    python: Path,
    repo: Path,
    tasks_path: Path,
    baseline_path: Path,
    output_path: Path,
    instance_ids: Sequence[str],
    seed: int,
    max_turns: int,
    max_internal_steps: int,
    parallelism: int,
) -> int:
    command = [
        str(python),
        str(repo / "src/domains/travelplanner/run.py"),
        "--tasks_path",
        str(tasks_path),
        "--travelplanner_set_type",
        "test",
        "--instance_ids",
        ",".join(instance_ids),
        "--num_instances",
        str(len(instance_ids)),
        "--max_turns",
        str(max_turns),
        "--max_internal_steps",
        str(max_internal_steps),
        "--parallelism",
        str(parallelism),
        "--seed",
        str(seed),
        "--travelplanner_multi_change_rate",
        "0.30",
        "--shift_distribution_baseline",
        str(baseline_path),
        "--distribution_control_mode",
        "prompt",
        "--output",
        str(output_path),
    ]
    print(f"Running {len(instance_ids)} case(s), seed={seed}, output={output_path}")
    completed = subprocess.run(command, cwd=repo, check=False)
    print(f"Runner exit code: {completed.returncode}")
    return completed.returncode


def run_shard(args: argparse.Namespace) -> int:
    repo = args.repo.resolve()
    tasks_path = args.tasks_path.resolve()
    baseline_path = args.baseline_path.resolve()
    run_dir = args.run_dir.resolve()
    tasks = _load_tasks(tasks_path)
    start, end = _shard_bounds(len(tasks), args.shard_index, args.num_shards)
    selected = tasks[start:end]
    expected_ids = [str(task["instance_id"]) for task in selected]
    global_positions = {
        str(task["instance_id"]): index for index, task in enumerate(tasks)
    }
    expected_turns = args.max_turns + 1
    shard_path = run_dir / "shards" / f"shard_{args.shard_index:02d}.json"
    instances = _load_valid_output(shard_path, expected_ids, expected_turns)

    for attempt in range(1, args.max_attempts + 1):
        missing = [instance_id for instance_id in expected_ids if instance_id not in instances]
        if not missing:
            break

        attempt_path = (
            run_dir
            / "attempts"
            / f"shard_{args.shard_index:02d}_attempt_{attempt:02d}.json"
        )
        attempt_path.parent.mkdir(parents=True, exist_ok=True)
        if not instances and missing == expected_ids:
            run_ids = missing
            run_seed = args.seed + start
            _run_cases(
                python=args.python,
                repo=repo,
                tasks_path=tasks_path,
                baseline_path=baseline_path,
                output_path=attempt_path,
                instance_ids=run_ids,
                seed=run_seed,
                max_turns=args.max_turns,
                max_internal_steps=args.max_internal_steps,
                parallelism=args.parallelism,
            )
            recovered = _load_valid_output(attempt_path, run_ids, expected_turns)
            instances.update(recovered)
        else:
            for instance_id in missing:
                case_path = attempt_path.with_name(
                    f"{attempt_path.stem}_{instance_id}.json"
                )
                _run_cases(
                    python=args.python,
                    repo=repo,
                    tasks_path=tasks_path,
                    baseline_path=baseline_path,
                    output_path=case_path,
                    instance_ids=[instance_id],
                    seed=args.seed + global_positions[instance_id],
                    max_turns=args.max_turns,
                    max_internal_steps=args.max_internal_steps,
                    parallelism=1,
                )
                recovered = _load_valid_output(
                    case_path, [instance_id], expected_turns
                )
                instances.update(recovered)

        ordered = [
            instances[instance_id]
            for instance_id in expected_ids
            if instance_id in instances
        ]
        _write_json_atomic(shard_path, ordered)
        print(
            f"Shard {args.shard_index}: {len(ordered)}/{len(expected_ids)} valid "
            f"after attempt {attempt}"
        )

    missing = [instance_id for instance_id in expected_ids if instance_id not in instances]
    if missing:
        print(f"Shard {args.shard_index} remains incomplete: {','.join(missing)}")
        return 2
    print(f"Shard {args.shard_index} complete: {len(expected_ids)} cases")
    return 0


def finalize(args: argparse.Namespace) -> int:
    tasks = _load_tasks(args.tasks_path.resolve())
    expected_ids = [str(task["instance_id"]) for task in tasks]
    expected_turns = args.max_turns + 1
    run_dir = args.run_dir.resolve()
    merged: Dict[str, Dict[str, Any]] = {}
    missing_shards: List[int] = []

    for shard_index in range(args.num_shards):
        start, end = _shard_bounds(len(tasks), shard_index, args.num_shards)
        shard_ids = expected_ids[start:end]
        shard_path = run_dir / "shards" / f"shard_{shard_index:02d}.json"
        valid = _load_valid_output(shard_path, shard_ids, expected_turns)
        if len(valid) != len(shard_ids):
            missing_shards.append(shard_index)
        merged.update(valid)

    missing_ids = [instance_id for instance_id in expected_ids if instance_id not in merged]
    summary = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "requested_model": os.getenv("BEDROCK_MODEL"),
        "actual_model": os.getenv("BEDROCK_MODEL"),
        "provider": "bedrock",
        "parameters": {
            "num_instances": len(expected_ids),
            "max_turns": args.max_turns,
            "max_internal_steps": args.max_internal_steps,
            "seed": args.seed,
            "travelplanner_multi_change_rate": 0.30,
            "distribution_control_mode": "prompt",
            "num_shards": args.num_shards,
            "parallelism": args.parallelism,
            "baseline_path": str(args.baseline_path.resolve()),
            "baseline_was_empty": _load_json(args.baseline_path.resolve()) == [],
        },
        "valid_instances": len(merged),
        "missing_instance_ids": missing_ids,
        "missing_shards": missing_shards,
    }
    _write_json_atomic(run_dir / "status.json", summary)
    if missing_ids:
        print(",".join(str(index) for index in missing_shards))
        return 2

    ordered = [merged[instance_id] for instance_id in expected_ids]
    output_path = args.output.resolve()
    _write_json_atomic(output_path, ordered)
    summary["output"] = str(output_path)
    summary["completed_at"] = datetime.now(timezone.utc).isoformat()
    _write_json_atomic(run_dir / "status.json", summary)
    (run_dir / "COMPLETE").write_text(
        summary["completed_at"] + "\n", encoding="utf-8"
    )
    print(f"Finalized {len(ordered)} cases to {output_path}")
    return 0


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--tasks-path", type=Path, required=True)
    parser.add_argument("--baseline-path", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--max-internal-steps", type=int, default=50)
    parser.add_argument("--parallelism", type=int, default=50)


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    shard_parser = subparsers.add_parser("run-shard")
    _common_arguments(shard_parser)
    shard_parser.add_argument("--shard-index", type=int, required=True)
    shard_parser.add_argument("--max-attempts", type=int, default=3)
    shard_parser.set_defaults(handler=run_shard)

    finalize_parser = subparsers.add_parser("finalize")
    _common_arguments(finalize_parser)
    finalize_parser.add_argument("--output", type=Path, required=True)
    finalize_parser.set_defaults(handler=finalize)

    args = parser.parse_args()
    if args.num_shards < 1:
        parser.error("--num-shards must be positive")
    if args.parallelism < 1:
        parser.error("--parallelism must be positive")
    return int(args.handler(args))


if __name__ == "__main__":
    sys.exit(main())
