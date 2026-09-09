#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
WEBSHOP_DIR = REPO_ROOT / "WebShop"
for path in (str(SRC_DIR), str(WEBSHOP_DIR), str(REPO_ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

import run_single_agent_real_eval_case as baseline
from common.llm_clients import BedrockConverseClient
from eval.human_annotated_pilot import (
    aggregate_scored_rows,
    atomic_write_json,
    load_json_list,
    select_webshop_pilot,
)
from eval.single_agent_real_pilot import run_single_agent_travel_instance


def _run_webshop(
    instance: Dict[str, Any],
    *,
    source_index: int,
    agent_client: Any,
    judge_client: Any,
    max_internal_steps: int,
) -> List[Dict[str, Any]]:
    from eval.run_benchmark import _build_runtime_components, replay_dialogue_instance

    env, agent, _human, raw_env = _build_runtime_components(
        azure_api_version="2024-10-21",
        webshop_num_products=100000,
        executor_type="fixed_user",
    )
    agent.llm_client = agent_client
    try:
        replayed = replay_dialogue_instance(
            benchmark_task=baseline._webshop_task(instance, source_index),
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
            "action_evidence": baseline._webshop_action_evidence(turn),
        }
        for turn in replayed.get("turns") or []
    ]
    return baseline._score_rows(
        domain="webshop",
        instance_id=replayed["instance_id"],
        rows=rows,
        client=judge_client,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--agent-model", required=True)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--webshop-max-internal-steps", type=int, default=12)
    parser.add_argument("--travel-max-internal-steps", type=int, default=30)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    if not 0 <= args.case_index < 16:
        raise ValueError("--case-index must be in [0, 15]")

    webshop = select_webshop_pilot(
        sorted(baseline.SHARD_DIR.glob("*_human_annotated.json")),
        per_shard=2,
    )
    travel = load_json_list(baseline.TRAVEL_PATH)
    if len(webshop) != 10 or len(travel) != 6:
        raise ValueError("Expected 10 WebShop and 6 TravelPlanner cases")

    agent_client = BedrockConverseClient(
        model=args.agent_model,
        timeout=args.timeout,
        max_tokens=8192,
    )
    judge_client = BedrockConverseClient(
        model=args.judge_model,
        timeout=args.timeout,
        max_tokens=8192,
    )

    if args.case_index < 10:
        rows = _run_webshop(
            webshop[args.case_index],
            source_index=args.case_index + 1,
            agent_client=agent_client,
            judge_client=judge_client,
            max_internal_steps=args.webshop_max_internal_steps,
        )
        domain = "webshop"
    else:
        instance = travel[args.case_index - 10]
        instance_id = str(instance.get("instance_id"))
        rows = run_single_agent_travel_instance(
            instance=instance,
            client=agent_client,
            max_internal_steps=args.travel_max_internal_steps,
        )
        rows = baseline._score_rows(
            domain="travelplanner",
            instance_id=instance_id,
            rows=rows,
            client=judge_client,
        )
        domain = "travelplanner"

    payload = {
        "metadata": {
            "design": "single evaluated agent; two offline metrics; strict public environment",
            "model": args.agent_model,
            "agent_model": args.agent_model,
            "judge_model": args.judge_model,
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
