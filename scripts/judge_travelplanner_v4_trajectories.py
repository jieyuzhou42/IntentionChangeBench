#!/usr/bin/env python3
"""LLM-judge saved TravelPlanner v4 agent trajectories.

The trajectory run directory stores, per (model, shard), only the agent's
per-turn ``agent_intention_prediction`` and ``action``. Gold lives in the v4
export (``annotation/data/exports/travelplanner_v4``); the pairing is verified by
dataset SHA-256.

Two judge calls are made per (model, shard, instance), both over all turns of
the instance:

* Pass A is main's shared judge, unmodified: ``build_judge_prompt`` with the
  fixed-search TravelPlanner action evidence (``{"action": action}``), scored by
  ``score_judged_turn``. Action-level metrics and main's aggregate come from it.
* Pass B aligns every predicted intent item to at most one gold constraint. It
  exists only because item-level precision, change precision and per-field tier
  accuracy need an item-to-gold mapping that pass A does not return.

With ``--evidence grounded`` pass A additionally receives, per turn, the
reference-database records of the entities the itinerary names (prices,
ratings, room types, house rules). Many agents write bare names and the
action-only judge then cannot verify cost or ratings.

With ``--evidence pool`` pass A instead receives, per turn, the full search
results the agent was shown (main's ``_compact_travel_search_results`` over the
gold turn's ``env_feedback``), so the judge sees the same candidate pool.

Gold entity constraints are flattened into the constraint list with main's
``flatten_gold_for_entity_scoring`` (as the single-agent TravelPlanner entry
does); the judge prompt asks for them and main's scorer rejects them otherwise.

Per-instance judge outputs are cached, so reruns resume where they stopped.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import json
import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
for path in (str(SRC_DIR), str(REPO_ROOT / "WebShop"), str(REPO_ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from common.llm_clients import OpenRouterChatClient
from eval.human_annotated_pilot import (
    _compact_travel_search_results,
    atomic_write_json,
    build_judge_prompt,
    score_judged_turn,
)
from run_single_agent_travel_eval_case import flatten_gold_for_entity_scoring

DOMAIN = "travelplanner"
DEFAULT_GOLD_DIR = REPO_ROOT / "annotation" / "data" / "exports" / "travelplanner_v4"
MAX_ATTEMPTS = 4
GROUND_MODE_CUES = {"self-driving": ("self-driv", "self driv", "self-drive"), "taxi": ("taxi",)}


def build_alignment_prompt(instance_id: str, turns: List[Dict[str, Any]]) -> str:
    return f"""
You align an evaluated agent's predicted intent items to gold constraints for a
multi-turn travel-planning benchmark. Do not judge the itinerary.

For every turn and every predicted item index, decide:
- gold_field: the exact gold constraint field this item expresses, or null if it
  expresses no gold constraint. Match semantically; a renamed but clearly
  equivalent field aligns. An item aligns to at most one gold field. Several
  items may align to the same gold field when each expresses part of it.
- value_match: true only if the item's value is semantically correct for that
  gold constraint's current value (false when gold_field is null). Respect
  context in fields/values: Day 2 does not satisfy Day 1, a budget ceiling is
  not an exact spending target, and an exclusion is not an inclusion. A correct
  part of a multi-part gold value counts as a match if nothing in it conflicts.
Ignore priority tiers; they are compared separately.

Return exactly:
{{
  "turns": [
    {{
      "turn_id": 0,
      "items": [{{"index": 0, "gold_field": "exact gold field or null", "value_match": true}}]
    }}
  ]
}}

Cover every supplied turn and every predicted item index exactly once.

INSTANCE: {instance_id}
PAYLOAD:
{json.dumps(turns, ensure_ascii=False, indent=2, default=str)}
""".strip()


def _gold_constraints(gold_intention: Dict[str, Any]) -> Dict[str, Any]:
    return {
        str(field): value
        for field, value in (gold_intention.get("constraints") or {}).items()
        if value is not None
    }


def _intent_items(prediction: Any) -> List[Dict[str, Any]]:
    if isinstance(prediction, dict) and isinstance(prediction.get("intent"), list):
        return [item for item in prediction["intent"] if isinstance(item, dict)]
    return []


def _call_json(client: Any, prompt: str) -> Dict[str, Any]:
    result = client.generate_json(prompt)
    if not isinstance(result, dict):
        raise ValueError("judge result is not a JSON object")
    return result


def selected_reference_records(action: Any, reference: Any) -> List[Dict[str, Any]]:
    """Reference-database records for the entities the itinerary names.

    Restaurants, accommodations and attractions match by name, flights by flight
    number, and ground transport by mode. Records are the tool environment's
    own data, i.e. what the agent's searches return.
    """
    text = json.dumps(action, ensure_ascii=False, default=str).lower()
    records: List[Dict[str, Any]] = []
    for source, value in (reference or {}).items() if isinstance(reference, dict) else []:
        if isinstance(value, list):
            for entry in value:
                if not isinstance(entry, dict):
                    continue
                name = str(entry.get("Name") or entry.get("NAME") or "").strip().lower()
                number = str(entry.get("Flight Number") or "").strip().lower()
                if (name and name in text) or (number and number in text):
                    records.append({"source": source, **entry})
        elif isinstance(value, str):
            mode = source.split(" from ")[0].strip().lower()
            if any(cue in text for cue in GROUND_MODE_CUES.get(mode, ())):
                records.append({"source": source, "record": value})
    return records


def action_evidence(row: Dict[str, Any], evidence: str) -> Dict[str, Any]:
    if evidence == "action":
        return {"action": row["action"]}
    if evidence == "pool":
        return {"action": row["action"], "agent_visible_search_results": row["visible_pool"]}
    return {
        "action": row["action"],
        "selected_item_reference_records": selected_reference_records(row["action"], row["reference"]),
    }


def run_pass_a(
    client: Any,
    instance_id: str,
    rows: List[Dict[str, Any]],
    evidence: str,
) -> Dict[str, Any]:
    judged_turns = [
        {
            "turn_id": row["turn_id"],
            "user_utterance": row["user_utterance"],
            "gold_current_intention": row["gold_intention"],
            "agent_intention_prediction": row["prediction"],
            "action_evidence": action_evidence(row, evidence),
        }
        for row in rows
    ]
    prompt = build_judge_prompt(domain=DOMAIN, instance_id=instance_id, judged_turns=judged_turns)
    last_error: Optional[Exception] = None
    for _ in range(MAX_ATTEMPTS):
        try:
            raw = _call_json(client, prompt)
            by_turn = {
                int(item.get("turn_id")): item
                for item in raw.get("turns") or []
                if isinstance(item, dict)
            }
            if set(by_turn) != {row["turn_id"] for row in rows}:
                raise ValueError(f"judge turn mismatch: {sorted(by_turn)}")
            # score_judged_turn validates the field coverage; a mismatch retries.
            for row in rows:
                score_judged_turn(gold_intention=row["gold_intention"], judgment=by_turn[row["turn_id"]])
            return {str(k): v for k, v in by_turn.items()}
        except Exception as exc:  # noqa: BLE001 - retry any malformed judgment
            last_error = exc
    raise RuntimeError(f"pass A failed for {instance_id}: {last_error!r}")


def run_pass_b(client: Any, instance_id: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    payload = [
        {
            "turn_id": row["turn_id"],
            "user_utterance": row["user_utterance"],
            "gold_constraints": _gold_constraints(row["gold_intention"]),
            "predicted_items": [
                {"index": i, "field": item.get("field"), "value": item.get("value")}
                for i, item in enumerate(_intent_items(row["prediction"]))
            ],
        }
        for row in rows
    ]
    prompt = build_alignment_prompt(instance_id, payload)
    last_error: Optional[Exception] = None
    for _ in range(MAX_ATTEMPTS):
        try:
            raw = _call_json(client, prompt)
            by_turn = {
                int(item.get("turn_id")): item
                for item in raw.get("turns") or []
                if isinstance(item, dict)
            }
            if set(by_turn) != {row["turn_id"] for row in rows}:
                raise ValueError(f"alignment turn mismatch: {sorted(by_turn)}")
            # A gold field absent from this turn (e.g. one active in another turn)
            # is kept as returned; the report counts it as unaligned.
            for row in rows:
                n_items = len(_intent_items(row["prediction"]))
                items = by_turn[row["turn_id"]].get("items") or []
                indices = sorted(int(item.get("index")) for item in items if isinstance(item, dict))
                if indices != list(range(n_items)):
                    raise ValueError(f"turn {row['turn_id']} alignment covers {indices}, expected {n_items} items")
            return {str(k): v for k, v in by_turn.items()}
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise RuntimeError(f"pass B failed for {instance_id}: {last_error!r}")


def load_units(run_dir: Path, gold_dir: Path, models: Optional[List[str]]) -> List[Dict[str, Any]]:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    gold_cache: Dict[str, List[Dict[str, Any]]] = {}
    units = []
    for entry in manifest["outputs"]:
        if models and entry["model_slug"] not in models:
            continue
        output = json.loads((run_dir / "output" / entry["output"]).read_text(encoding="utf-8"))
        gold_path = gold_dir / entry["shard_file"]
        digest = hashlib.sha256(gold_path.read_bytes()).hexdigest()
        if digest != output["metadata"]["dataset_sha256"]:
            raise ValueError(f"{entry['output']}: gold {gold_path.name} sha256 mismatch")
        if entry["shard_file"] not in gold_cache:
            gold_cache[entry["shard_file"]] = json.loads(gold_path.read_text(encoding="utf-8"))
        gold_by_id = {inst["instance_id"]: inst for inst in gold_cache[entry["shard_file"]]}
        for trajectory in output["trajectories"]:
            gold_instance = gold_by_id[trajectory["instance_id"]]
            gold_turns = {int(t["turn_id"]): t for t in gold_instance["turns"]}
            if set(gold_turns) != {int(t["turn_id"]) for t in trajectory["turns"]}:
                raise ValueError(f"{entry['output']} {trajectory['instance_id']}: turn ids differ from gold")
            rows = []
            for turn in sorted(trajectory["turns"], key=lambda t: int(t["turn_id"])):
                gold_turn = gold_turns[int(turn["turn_id"])]
                rows.append(
                    {
                        "turn_id": int(turn["turn_id"]),
                        "user_utterance": gold_turn.get("user_utterance"),
                        "gold_intention": flatten_gold_for_entity_scoring(
                            gold_turn.get("gold_current_intention") or {}
                        ),
                        "gold_delta": copy.deepcopy(gold_turn.get("gold_delta") or {}),
                        "linguistic_style": gold_turn.get("linguistic_style"),
                        "action_implication": gold_turn.get("action_implication"),
                        "prediction": turn.get("agent_intention_prediction"),
                        "action": turn.get("action"),
                        "reference": (gold_instance.get("world_state") or {}).get("reference_information"),
                        # Exactly what build_agent_prompt showed the agent this turn.
                        "visible_pool": _compact_travel_search_results(
                            (gold_turn.get("env_feedback") or {}).get("search_results")
                        ),
                    }
                )
            units.append(
                {
                    "model": entry["model_slug"],
                    "model_id": entry["model_id"],
                    "shard": entry["shard_slug"],
                    "instance_id": trajectory["instance_id"],
                    "rows": rows,
                }
            )
    return units


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--gold-dir", type=Path, default=DEFAULT_GOLD_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--judge-model", default="openai/gpt-6-luna")
    parser.add_argument(
        "--evidence",
        choices=["action", "grounded", "pool"],
        default="action",
        help="action: main's fixed-search TravelPlanner evidence ({'action': ...}). "
        "grounded: additionally attach the reference records of the itinerary's entities. "
        "pool: additionally attach the full search results the agent saw this turn.",
    )
    parser.add_argument("--models", nargs="*", help="Model slugs to judge (default: all).")
    parser.add_argument("--limit-instances", type=int, help="Judge only the first N units (pilot).")
    parser.add_argument("--parallelism", type=int, default=12)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--max-tokens", type=int, default=64000)
    args = parser.parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is required")
    client = OpenRouterChatClient(
        api_key=api_key,
        model=args.judge_model,
        timeout=args.timeout,
        max_tokens=args.max_tokens,
    )

    units = load_units(args.run_dir, args.gold_dir, args.models)
    if args.limit_instances:
        units = units[: args.limit_instances]
    cache_dir = args.output_dir / "judgments"
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(units)} (model, shard, instance) units, {sum(len(u['rows']) for u in units)} turns, judge={args.judge_model}", flush=True)

    lock = threading.Lock()
    done = {"n": 0}

    pass_a_key = "pass_a" if args.evidence == "action" else f"pass_a_{args.evidence}"

    def work(unit: Dict[str, Any]) -> Optional[str]:
        path = cache_dir / f"{unit['model']}__{unit['shard']}__{unit['instance_id']}.json"
        cached = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        if pass_a_key not in cached:
            cached[pass_a_key] = run_pass_a(client, unit["instance_id"], unit["rows"], args.evidence)
            atomic_write_json(path, cached)
        if "pass_b" not in cached:
            cached["pass_b"] = run_pass_b(client, unit["instance_id"], unit["rows"])
            atomic_write_json(path, cached)
        with lock:
            done["n"] += 1
            print(f"[{done['n']}/{len(units)}] {path.stem}", flush=True)
        return None

    errors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallelism) as pool:
        futures = {pool.submit(work, unit): unit for unit in units}
        for future in concurrent.futures.as_completed(futures):
            unit = futures[future]
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{unit['model']}__{unit['shard']}__{unit['instance_id']}: {exc!r}")
                print(f"FAILED {errors[-1]}", flush=True)

    # Assemble scored rows for every unit whose judgments are complete.
    scored = []
    for unit in units:
        path = cache_dir / f"{unit['model']}__{unit['shard']}__{unit['instance_id']}.json"
        if not path.exists():
            continue
        cached = json.loads(path.read_text(encoding="utf-8"))
        if pass_a_key not in cached or "pass_b" not in cached:
            continue
        for row in unit["rows"]:
            judgment = cached[pass_a_key][str(row["turn_id"])]
            scored.append(
                {
                    "domain": DOMAIN,
                    "model": unit["model"],
                    "model_id": unit["model_id"],
                    "shard": unit["shard"],
                    "instance_id": unit["instance_id"],
                    **{k: v for k, v in row.items() if k not in ("reference", "visible_pool")},
                    "action_evidence": action_evidence(row, args.evidence),
                    "judge_output": judgment,
                    "alignment": cached["pass_b"][str(row["turn_id"])],
                    "scores": score_judged_turn(gold_intention=row["gold_intention"], judgment=judgment),
                }
            )
    name = "scored_rows.json" if args.evidence == "action" else f"scored_rows_{args.evidence}.json"
    atomic_write_json(
        args.output_dir / name,
        {
            "metadata": {
                "run_dir": str(args.run_dir),
                "gold_dir": str(args.gold_dir),
                "judge_model": args.judge_model,
                "judge_provider": "openrouter",
                "evidence": args.evidence,
                "units": len(units),
                "errors": errors,
            },
            "rows": scored,
        },
    )
    print(f"wrote {len(scored)} scored turns; {len(errors)} failed units", flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
