#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
for path in (str(SRC_DIR), str(REPO_ROOT / "WebShop"), str(REPO_ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from common.llm_clients import create_llm_client_from_env
from eval.human_annotated_pilot import aggregate_scored_rows, atomic_write_json, load_json_list
from eval.single_agent_real_pilot import run_single_agent_travel_instance
from run_single_agent_real_eval_case import _score_rows


def flatten_gold_for_entity_scoring(gold: dict) -> dict:
    flattened = copy.deepcopy(gold)
    constraints = copy.deepcopy(flattened.get("constraints") or {})
    for entity_id, entity in (flattened.get("entities") or {}).items():
        if not isinstance(entity, dict):
            continue
        reference = str(entity.get("reference") or entity_id)
        for field, value in (entity.get("constraints") or {}).items():
            constraints[f"entities.{entity_id}.constraints.{field}"] = {
                "reference": reference,
                "value": copy.deepcopy(value),
            }
    flattened["constraints"] = constraints
    return flattened


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-internal-steps", type=int, default=30)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    instances = load_json_list(args.input)
    matches = [row for row in instances if str(row.get("instance_id")) == args.instance_id]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {args.instance_id} in {args.input}; got {len(matches)}")

    client = create_llm_client_from_env(timeout=args.timeout)
    rows = run_single_agent_travel_instance(
        instance=matches[0],
        client=client,
        max_internal_steps=args.max_internal_steps,
    )
    for row in rows:
        full_gold = copy.deepcopy(row.get("gold_intention") or {})
        row["gold_intention_full"] = full_gold
        row["gold_intention"] = flatten_gold_for_entity_scoring(full_gold)
    rows = _score_rows(
        domain="travelplanner",
        instance_id=args.instance_id,
        rows=rows,
        client=client,
    )
    payload = {
        "metadata": {
            "design": "single evaluated agent; strict public TravelPlanner environment",
            "model": getattr(client, "model", getattr(client, "deployment", "unknown")),
            "domain": "travelplanner",
            "instance_id": args.instance_id,
            "source": str(args.input),
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
