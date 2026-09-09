#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
WEBSHOP_DIR = REPO_ROOT / "WebShop"
for path in (str(SRC_DIR), str(WEBSHOP_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from common.llm_clients import create_llm_client_from_env
from eval.human_annotated_pilot import (
    aggregate_scored_rows,
    atomic_write_json,
    build_judge_prompt,
    call_json_with_retries,
    load_json_list,
    score_judged_turn,
    select_webshop_pilot,
)
from eval.real_environment_pilot import (
    replay_travelplanner_instance,
    webshop_action_evidence,
)


SHARD_DIR = (
    REPO_ROOT
    / "data"
    / "simulation"
    / "webshop_v2_350_formal_priority_classified_shards"
)
TRAVEL_PATH = REPO_ROOT / "data" / "simulation" / "travelplanner_tradeoff_annotation_v1.json"


def _score_instance_rows(
    *,
    domain: str,
    instance_id: str,
    rows: List[Dict[str, Any]],
    client: Any,
) -> List[Dict[str, Any]]:
    judge_payload = [
        {
            "turn_id": row["turn_id"],
            "user_utterance": row["user_utterance"],
            "gold_current_intention": row["gold_intention"],
            "agent_intention_prediction": row["agent_intention_prediction"],
            "action_evidence": row["action_evidence"],
        }
        for row in rows
    ]
    judgment = call_json_with_retries(
        client,
        build_judge_prompt(
            domain=domain,
            instance_id=instance_id,
            judged_turns=judge_payload,
        ),
    )
    judged_turns = judgment.get("turns")
    if not isinstance(judged_turns, list):
        raise ValueError(f"Judge output for {instance_id} has no turns list")
    by_turn = {
        int(item.get("turn_id")): item
        for item in judged_turns
        if isinstance(item, dict)
    }
    expected = {row["turn_id"] for row in rows}
    if set(by_turn) != expected:
        raise ValueError(f"Judge turn mismatch for {instance_id}")
    for row in rows:
        row["judge_output"] = by_turn[row["turn_id"]]
        row["scores"] = score_judged_turn(
            gold_intention=row["gold_intention"],
            judgment=row["judge_output"],
        )
    return rows


def _webshop_benchmark_task(instance: Dict[str, Any], index: int):
    from eval.run_benchmark import BenchmarkTask
    from models import BaseTask

    turns = instance.get("turns") or []
    first_utterance = str((turns[0] if turns else {}).get("user_utterance") or "")
    world_state = copy.deepcopy(instance.get("world_state") or {})
    world_state.setdefault("webshop_instruction_text", first_utterance)
    return BenchmarkTask(
        task=BaseTask(
            instance_id=str(instance.get("instance_id") or f"webshop_{index:03d}"),
            task_type=str(instance.get("task_type") or "transaction"),
            subtype=str(instance.get("subtype") or "shopping"),
            world_state=world_state,
            initial_intention={
                "request": first_utterance,
                "constraints": {},
                "priority": {},
            },
        ),
        source_instance=instance,
    )


def _run_webshop(
    *,
    instances: Sequence[Dict[str, Any]],
    client: Any,
    max_internal_steps: int,
) -> List[Dict[str, Any]]:
    from eval.run_benchmark import (
        _build_runtime_components,
        replay_dialogue_instance,
    )

    env, agent, _human, raw_env = _build_runtime_components(
        azure_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        webshop_num_products=100000,
        executor_type="fixed_user",
    )
    # Reuse the caller's configured client so agent and judge use the same explicit model.
    agent.llm_client = client
    rows: List[Dict[str, Any]] = []
    try:
        for index, instance in enumerate(instances, start=1):
            replayed = replay_dialogue_instance(
                benchmark_task=_webshop_benchmark_task(instance, index),
                env=env,
                execution_agent=agent,
                max_turns=None,
                max_internal_steps=max_internal_steps,
            ).to_dict()
            instance_rows = []
            for turn in replayed.get("turns") or []:
                instance_rows.append(
                    {
                        "domain": "webshop",
                        "instance_id": replayed["instance_id"],
                        "turn_id": int(turn["turn_id"]),
                        "user_utterance": turn.get("user_utterance"),
                        "gold_intention": copy.deepcopy(turn.get("gold_current_intention") or {}),
                        "agent_intention_prediction": copy.deepcopy(
                            turn.get("agent_intention_prediction") or {}
                        ),
                        "agent_action": copy.deepcopy(turn.get("agent_action")),
                        "env_feedback": copy.deepcopy(turn.get("env_feedback")),
                        "rollout_trace": copy.deepcopy(turn.get("rollout_trace") or []),
                        "num_internal_steps": turn.get("num_internal_steps"),
                        "stop_reason": turn.get("stop_reason"),
                        "action_evidence": webshop_action_evidence(turn),
                    }
                )
            rows.extend(
                _score_instance_rows(
                    domain="webshop",
                    instance_id=replayed["instance_id"],
                    rows=instance_rows,
                    client=client,
                )
            )
            print(f"completed webshop {replayed['instance_id']}", flush=True)
    finally:
        close = getattr(raw_env, "close", None)
        if callable(close):
            close()
    return rows


def _run_travel_instance(
    instance: Dict[str, Any],
    *,
    client: Any,
    max_internal_steps: int,
) -> List[Dict[str, Any]]:
    instance_id = str(instance.get("instance_id"))
    rows = replay_travelplanner_instance(
        instance=instance,
        client=client,
        max_internal_steps=max_internal_steps,
    )
    rows = _score_instance_rows(
        domain="travelplanner",
        instance_id=instance_id,
        rows=rows,
        client=client,
    )
    print(f"completed travelplanner {instance_id}", flush=True)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data" / "eval" / "human_annotated_real_env_pilot.json",
    )
    parser.add_argument("--domain", choices=["all", "webshop", "travelplanner"], default="all")
    parser.add_argument("--travel_parallelism", type=int, default=6)
    parser.add_argument("--webshop_max_internal_steps", type=int, default=12)
    parser.add_argument("--travel_max_internal_steps", type=int, default=30)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    shard_paths = sorted(SHARD_DIR.glob("*_human_annotated.json"))
    webshop = select_webshop_pilot(shard_paths, per_shard=2)
    travel = load_json_list(TRAVEL_PATH)
    if len(webshop) != 10 or len(travel) != 6:
        raise ValueError(f"Expected 10 WebShop and 6 TravelPlanner instances; got {len(webshop)} and {len(travel)}")

    client = create_llm_client_from_env(timeout=args.timeout)
    existing_payload: Dict[str, Any] = {}
    if args.output.exists():
        loaded = json.loads(args.output.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            existing_payload = loaded
    rows: List[Dict[str, Any]] = list(existing_payload.get("rows") or [])
    errors: List[Tuple[str, str]] = list(existing_payload.get("errors") or [])
    completed_instances = {
        (str(row.get("domain")), str(row.get("instance_id")))
        for row in rows
        if row.get("scores")
    }

    if args.domain in {"all", "travelplanner"}:
        pending_travel = [
            instance
            for instance in travel
            if ("travelplanner", str(instance.get("instance_id"))) not in completed_instances
        ]
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(args.travel_parallelism, len(pending_travel)) or 1
        ) as executor:
            futures = {
                executor.submit(
                    _run_travel_instance,
                    instance,
                    client=client,
                    max_internal_steps=args.travel_max_internal_steps,
                ): str(instance.get("instance_id"))
                for instance in pending_travel
            }
            for future in concurrent.futures.as_completed(futures):
                instance_id = futures[future]
                try:
                    rows.extend(future.result())
                except Exception as exc:
                    errors.append((instance_id, repr(exc)))
                    print(f"FAILED travelplanner {instance_id}: {exc!r}", flush=True)
                atomic_write_json(
                    args.output,
                    {
                        "metadata": {
                            "design": "blind real-environment two-layer pilot",
                            "model": getattr(client, "model", getattr(client, "deployment", "unknown")),
                            "webshop_retrieval": "real WebShop search action -> native Lucene BM25 results",
                        },
                        "rows": sorted(rows, key=lambda row: (row["domain"], row["instance_id"], row["turn_id"])),
                        "aggregate": aggregate_scored_rows(rows),
                        "errors": errors,
                    },
                )

    if args.domain in {"all", "webshop"}:
        pending_webshop = [
            instance
            for instance in webshop
            if ("webshop", str(instance.get("instance_id"))) not in completed_instances
        ]
        try:
            rows.extend(
                _run_webshop(
                    instances=pending_webshop,
                    client=client,
                    max_internal_steps=args.webshop_max_internal_steps,
                )
            )
        except Exception as exc:
            errors.append(("webshop", repr(exc)))
            print(f"FAILED webshop: {exc!r}", flush=True)

    rows.sort(key=lambda row: (row["domain"], row["instance_id"], row["turn_id"]))
    payload = {
        "metadata": {
            "design": "blind real-environment two-layer pilot",
            "model": getattr(client, "model", getattr(client, "deployment", "unknown")),
            "webshop_retrieval": "real WebShop search action -> native Lucene BM25 results",
            "travel_parallelism": args.travel_parallelism,
        },
        "rows": rows,
        "aggregate": aggregate_scored_rows(rows),
        "errors": errors,
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(payload["aggregate"], indent=2))
    if errors:
        raise RuntimeError(f"Pilot completed with errors: {errors}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
