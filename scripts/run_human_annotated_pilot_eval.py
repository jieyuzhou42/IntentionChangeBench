#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.llm_clients import create_llm_client_from_env
from eval.human_annotated_pilot import (
    aggregate_scored_rows,
    atomic_write_json,
    build_agent_prompt,
    build_judge_prompt,
    call_json_with_retries,
    constraint_field_vocabulary,
    load_json_list,
    normalize_agent_output,
    score_judged_turn,
    select_webshop_pilot,
)


DEFAULT_SHARD_DIR = (
    REPO_ROOT
    / "data"
    / "simulation"
    / "webshop_v2_350_formal_priority_classified_shards"
)
DEFAULT_TRAVEL_PATH = (
    REPO_ROOT / "data" / "simulation" / "travelplanner_tradeoff_annotation_v1.json"
)


def _action_evidence(
    domain: str,
    turn: Dict[str, Any],
    agent_output: Dict[str, Any],
) -> Dict[str, Any]:
    action = agent_output["action"]
    if domain == "webshop":
        selected_asin = str(action.get("selected_asin") or "").upper()
        candidates = (turn.get("env_feedback") or {}).get("candidate_items") or []
        selected = next(
            (
                copy.deepcopy(item)
                for item in candidates
                if isinstance(item, dict)
                and str(item.get("asin") or "").upper() == selected_asin
            ),
            None,
        )
        return {
            "action": action,
            "selected_product": selected,
            "human_gold_action": copy.deepcopy(turn.get("gold_action")),
        }
    return {"action": action}


def _run_instance(
    *,
    domain: str,
    instance: Dict[str, Any],
    client: Any,
    field_vocabulary: List[str],
    checkpoint: Dict[str, Any],
    output_path: Path,
) -> List[Dict[str, Any]]:
    instance_id = str(instance.get("instance_id"))
    existing = {
        (row["domain"], row["instance_id"], int(row["turn_id"])): row
        for row in checkpoint.get("rows") or []
    }
    agent_rows: List[Dict[str, Any]] = []
    for turn_index, turn in enumerate(instance.get("turns") or []):
        turn_id = int(turn.get("turn_id", turn_index))
        key = (domain, instance_id, turn_id)
        previous = existing.get(key)
        if previous and previous.get("agent_output"):
            agent_output = previous["agent_output"]
        else:
            prompt = build_agent_prompt(
                domain=domain,
                instance=instance,
                turn_index=turn_index,
                field_vocabulary=field_vocabulary,
            )
            raw = call_json_with_retries(client, prompt)
            valid_asins = [
                str(item.get("asin") or "")
                for item in ((turn.get("env_feedback") or {}).get("candidate_items") or [])
                if isinstance(item, dict)
            ]
            agent_output = normalize_agent_output(
                raw,
                domain=domain,
                valid_asins=valid_asins,
            )
            print(f"agent {domain} {instance_id} turn={turn_id}", flush=True)
        agent_rows.append(
            {
                "domain": domain,
                "instance_id": instance_id,
                "turn_id": turn_id,
                "user_utterance": turn.get("user_utterance"),
                "gold_intention": copy.deepcopy(turn.get("gold_current_intention") or {}),
                "agent_output": agent_output,
                "action_evidence": _action_evidence(domain, turn, agent_output),
            }
        )

    judge_payload = [
        {
            "turn_id": row["turn_id"],
            "user_utterance": row["user_utterance"],
            "gold_current_intention": row["gold_intention"],
            "agent_intention_prediction": row["agent_output"][
                "current_intention_understanding"
            ],
            "action_evidence": row["action_evidence"],
        }
        for row in agent_rows
    ]
    raw_judgment = call_json_with_retries(
        client,
        build_judge_prompt(
            domain=domain,
            instance_id=instance_id,
            judged_turns=judge_payload,
        ),
    )
    judged_turns = raw_judgment.get("turns")
    if not isinstance(judged_turns, list):
        raise ValueError(f"Judge output for {instance_id} has no turns list")
    judgment_by_turn = {
        int(item.get("turn_id")): item for item in judged_turns if isinstance(item, dict)
    }
    if set(judgment_by_turn) != {row["turn_id"] for row in agent_rows}:
        raise ValueError(f"Judge turn mismatch for {instance_id}")

    scored_rows = []
    for row in agent_rows:
        judgment = judgment_by_turn[row["turn_id"]]
        row["judge_output"] = judgment
        row["scores"] = score_judged_turn(
            gold_intention=row["gold_intention"],
            judgment=judgment,
        )
        scored_rows.append(row)
    print(f"judge {domain} {instance_id}", flush=True)
    return scored_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--webshop_shard_dir", type=Path, default=DEFAULT_SHARD_DIR)
    parser.add_argument("--travel_path", type=Path, default=DEFAULT_TRAVEL_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data" / "eval" / "human_annotated_pilot_eval.json",
    )
    parser.add_argument("--webshop_per_shard", type=int, default=2)
    parser.add_argument("--domain", choices=["all", "webshop", "travelplanner"], default="all")
    parser.add_argument(
        "--parallelism",
        type=int,
        default=1,
        help="Number of instances to evaluate concurrently; turns within an instance stay sequential.",
    )
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    if args.parallelism < 1:
        raise ValueError("--parallelism must be at least 1")

    shard_paths = sorted(args.webshop_shard_dir.glob("*_human_annotated.json"))
    if len(shard_paths) != 5:
        raise ValueError(f"Expected 5 human-annotated WebShop shards, found {len(shard_paths)}")
    webshop_instances = select_webshop_pilot(
        shard_paths,
        per_shard=args.webshop_per_shard,
    )
    travel_instances = load_json_list(args.travel_path)
    if len(travel_instances) != 6:
        raise ValueError(f"Expected 6 TravelPlanner annotations, found {len(travel_instances)}")

    selected_domains: List[tuple[str, List[Dict[str, Any]]]] = []
    if args.domain in {"all", "webshop"}:
        selected_domains.append(("webshop", webshop_instances))
    if args.domain in {"all", "travelplanner"}:
        selected_domains.append(("travelplanner", travel_instances))

    client = create_llm_client_from_env(timeout=args.timeout)
    checkpoint: Dict[str, Any] = {"rows": []}
    if args.output.exists():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        if isinstance(existing, dict):
            checkpoint = existing

    completed_by_key = {
        (row["domain"], row["instance_id"], int(row["turn_id"])): row
        for row in checkpoint.get("rows") or []
        if row.get("scores")
    }
    rows = list(completed_by_key.values())
    metadata = {
        "design": "blind fixed-evidence two-layer pilot",
        "agent_model": getattr(client, "model", getattr(client, "deployment", "unknown")),
        "webshop_selection": [
            {
                "instance_id": item.get("instance_id"),
                "source_shard": item.get("_pilot_source"),
                "turns": len(item.get("turns") or []),
            }
            for item in webshop_instances
        ],
        "travelplanner_selection": [
            {
                "instance_id": item.get("instance_id"),
                "turns": len(item.get("turns") or []),
            }
            for item in travel_instances
        ],
    }

    pending = []
    for domain, instances in selected_domains:
        vocabulary = constraint_field_vocabulary(instances)
        for instance in instances:
            instance_id = str(instance.get("instance_id"))
            expected_keys = {
                (domain, instance_id, int(turn.get("turn_id", index)))
                for index, turn in enumerate(instance.get("turns") or [])
            }
            if expected_keys and expected_keys.issubset(completed_by_key):
                print(f"skip completed {domain} {instance_id}", flush=True)
                continue
            pending.append((domain, instance, vocabulary))

    errors = []
    max_workers = min(args.parallelism, len(pending)) if pending else 1
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {
            executor.submit(
                _run_instance,
                domain=domain,
                instance=instance,
                client=client,
                field_vocabulary=vocabulary,
                checkpoint=checkpoint,
                output_path=args.output,
            ): (domain, str(instance.get("instance_id")))
            for domain, instance, vocabulary in pending
        }
        for future in concurrent.futures.as_completed(future_to_item):
            domain, instance_id = future_to_item[future]
            try:
                instance_rows = future.result()
            except Exception as exc:
                errors.append((domain, instance_id, repr(exc)))
                print(f"FAILED {domain} {instance_id}: {exc!r}", flush=True)
                continue
            rows = [
                row
                for row in rows
                if not (row["domain"] == domain and row["instance_id"] == instance_id)
            ]
            rows.extend(instance_rows)
            rows.sort(key=lambda row: (row["domain"], row["instance_id"], row["turn_id"]))
            checkpoint = {
                "metadata": {**metadata, "parallelism": max_workers},
                "rows": rows,
                "aggregate": aggregate_scored_rows(rows),
            }
            atomic_write_json(args.output, checkpoint)

    print(json.dumps(aggregate_scored_rows(rows), indent=2))
    if errors:
        raise RuntimeError(f"{len(errors)} instance(s) failed: {errors}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
