#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List


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
from eval.single_agent_real_pilot import run_single_agent_travel_instance


SHARD_DIR = (
    REPO_ROOT
    / "data"
    / "simulation"
    / "webshop_v2_350_formal_priority_classified_shards"
)
TRAVEL_PATH = REPO_ROOT / "data" / "simulation" / "travelplanner_tradeoff_annotation_v1.json"


def _score_rows(
    *,
    domain: str,
    instance_id: str,
    rows: List[Dict[str, Any]],
    client: Any,
) -> List[Dict[str, Any]]:
    judged_turns = [
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
            judged_turns=judged_turns,
        ),
    )
    judge_rows = judgment.get("turns")
    if not isinstance(judge_rows, list):
        raise ValueError(f"Judge output for {instance_id} has no turns list")
    by_turn = {
        int(item.get("turn_id")): item
        for item in judge_rows
        if isinstance(item, dict)
    }
    if set(by_turn) != {row["turn_id"] for row in rows}:
        raise ValueError(f"Judge turn mismatch for {instance_id}")
    for row in rows:
        row["judge_output"] = by_turn[row["turn_id"]]
        row["scores"] = score_judged_turn(
            gold_intention=row["gold_intention"],
            judgment=row["judge_output"],
        )
    return rows


def _webshop_task(instance: Dict[str, Any], index: int):
    from eval.run_benchmark import BenchmarkTask
    from models import BaseTask

    turns = instance.get("turns") or []
    first_utterance = str((turns[0] if turns else {}).get("user_utterance") or "")
    source_world = instance.get("world_state") or {}
    world_state = {
        key: copy.deepcopy(source_world[key])
        for key in ("webshop_goal_index", "goal_index", "webshop_fixed_price_upper")
        if key in source_world
    }
    world_state["webshop_instruction_text"] = first_utterance
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


def _webshop_action_evidence(turn: Dict[str, Any]) -> Dict[str, Any]:
    feedback = turn.get("env_feedback") or {}
    return {
        "final_action": copy.deepcopy(turn.get("agent_action")),
        "selected_product": copy.deepcopy(
            feedback.get("selected_item") or feedback.get("selected_candidate")
        ),
        "selected_asin": feedback.get("selected_asin"),
        "gold_deterministic_constraint_eval": {
            "satisfied_constraints": feedback.get("gold_eval_satisfied_constraints") or [],
            "violated_constraints": feedback.get("gold_eval_violated_constraints") or [],
            "constraint_debug": feedback.get("gold_eval_constraint_debug") or {},
        },
        "rollout_trace": copy.deepcopy(turn.get("rollout_trace") or []),
    }


def _run_webshop_case(
    instance: Dict[str, Any],
    *,
    source_index: int,
    client: Any,
    max_internal_steps: int,
) -> List[Dict[str, Any]]:
    from eval.run_benchmark import _build_runtime_components, replay_dialogue_instance

    env, agent, _human, raw_env = _build_runtime_components(
        azure_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        webshop_num_products=100000,
        executor_type="fixed_user",
    )
    agent.llm_client = client
    try:
        replayed = replay_dialogue_instance(
            benchmark_task=_webshop_task(instance, source_index),
            env=env,
            execution_agent=agent,
            max_turns=None,
            max_internal_steps=max_internal_steps,
        ).to_dict()
    finally:
        close = getattr(raw_env, "close", None)
        if callable(close):
            close()

    rows = [
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
            "action_evidence": _webshop_action_evidence(turn),
        }
        for turn in replayed.get("turns") or []
    ]
    return _score_rows(
        domain="webshop",
        instance_id=replayed["instance_id"],
        rows=rows,
        client=client,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--webshop-max-internal-steps", type=int, default=12)
    parser.add_argument("--travel-max-internal-steps", type=int, default=30)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    if not 0 <= args.case_index < 16:
        raise ValueError("--case-index must be in [0, 15]")

    webshop = select_webshop_pilot(
        sorted(SHARD_DIR.glob("*_human_annotated.json")),
        per_shard=2,
    )
    travel = load_json_list(TRAVEL_PATH)
    if len(webshop) != 10 or len(travel) != 6:
        raise ValueError(f"Expected 10 WebShop and 6 TravelPlanner cases; got {len(webshop)} and {len(travel)}")

    client = create_llm_client_from_env(timeout=args.timeout)
    if args.case_index < 10:
        instance = webshop[args.case_index]
        rows = _run_webshop_case(
            instance,
            source_index=args.case_index + 1,
            client=client,
            max_internal_steps=args.webshop_max_internal_steps,
        )
        domain = "webshop"
    else:
        instance = travel[args.case_index - 10]
        instance_id = str(instance.get("instance_id"))
        rows = run_single_agent_travel_instance(
            instance=instance,
            client=client,
            max_internal_steps=args.travel_max_internal_steps,
        )
        rows = _score_rows(
            domain="travelplanner",
            instance_id=instance_id,
            rows=rows,
            client=client,
        )
        domain = "travelplanner"

    payload = {
        "metadata": {
            "design": "single evaluated agent; two offline metrics; strict public environment",
            "model": getattr(client, "model", getattr(client, "deployment", "unknown")),
            "case_index": args.case_index,
            "domain": domain,
            "webshop_retrieval": "native WebShop Lucene BM25 search only",
        },
        "rows": rows,
        "aggregate": aggregate_scored_rows(rows),
        "errors": [],
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(payload["aggregate"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
