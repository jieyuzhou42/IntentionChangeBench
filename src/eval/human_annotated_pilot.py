from __future__ import annotations

import copy
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple
from eval.priority_schema import PRIORITY_ALIASES
from eval.intent_schema import INTENT_RULES, INTENT_SCHEMA_JSON, normalize_intent_prediction
from eval.action_policy import BEST_AVAILABLE_RULES, TRAVEL_SELECTION_RULES


PRIORITY_LEVEL_WEIGHTS = {"high": 3.0, "medium": 2.0, "low": 1.0}


def load_json_list(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise ValueError(f"{path} must contain a JSON list of objects")
    return payload


def select_webshop_pilot(
    shard_paths: Sequence[Path],
    *,
    per_shard: int = 2,
) -> List[Dict[str, Any]]:
    if per_shard < 1:
        raise ValueError("per_shard must be at least 1")
    selected: List[Dict[str, Any]] = []
    for path in sorted(shard_paths):
        instances = load_json_list(path)
        if len(instances) < per_shard:
            raise ValueError(f"{path} has only {len(instances)} instances")
        for instance in instances[:per_shard]:
            item = copy.deepcopy(instance)
            item["_pilot_source"] = path.name
            selected.append(item)
    return selected


def constraint_field_vocabulary(instances: Iterable[Dict[str, Any]]) -> List[str]:
    fields = set()
    for instance in instances:
        for turn in instance.get("turns") or []:
            intention = turn.get("gold_current_intention") or {}
            constraints = intention.get("constraints") or {}
            if isinstance(constraints, dict):
                fields.update(str(field) for field in constraints if str(field).strip())
    return sorted(fields)


def _compact_text(value: Any, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _compact_webshop_candidate(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "asin": item.get("asin"),
        "title": _compact_text(item.get("title"), 240),
        "price": item.get("price"),
        "description": _compact_text(item.get("description"), 700),
        "bullet_points": [
            _compact_text(point, 240) for point in (item.get("bullet_points") or [])[:6]
        ],
        "attributes": list(item.get("attributes") or [])[:20],
        "options": copy.deepcopy(item.get("options") or {}),
        "brand": item.get("brand"),
        "color": item.get("color"),
        "product_category": item.get("product_category"),
    }


def _compact_travel_search_results(raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    compact: Dict[str, Any] = {}
    for category, pages in raw.items():
        if not isinstance(pages, list):
            continue
        compact_pages = []
        for page in pages:
            if not isinstance(page, dict):
                continue
            items = []
            for item in (page.get("items") or [])[:12]:
                if isinstance(item, dict):
                    items.append(copy.deepcopy(item))
                else:
                    items.append(_compact_text(item, 500))
            compact_pages.append(
                {
                    "query": page.get("query"),
                    "status": page.get("status"),
                    "message": page.get("message"),
                    "items": items,
                }
            )
        compact[category] = compact_pages
    return compact


def build_agent_prompt(
    *,
    domain: str,
    instance: Dict[str, Any],
    turn_index: int,
    field_vocabulary: Sequence[str],
) -> str:
    turns = instance.get("turns") or []
    turn = turns[turn_index]
    utterances = [
        str(item.get("user_utterance") or "").strip()
        for item in turns[: turn_index + 1]
        if str(item.get("user_utterance") or "").strip()
    ]
    common_context = {
        "instance_id": instance.get("instance_id"),
        "turn_id": turn.get("turn_id", turn_index),
        "user_utterances_in_order": utterances,
        "canonical_constraint_field_vocabulary": list(field_vocabulary),
    }

    if domain == "webshop":
        feedback = turn.get("env_feedback") or {}
        common_context["candidate_items"] = [
            _compact_webshop_candidate(item)
            for item in (feedback.get("candidate_items") or [])[:10]
            if isinstance(item, dict)
        ]
        action_schema = """
"action": {
  "action_type": "buy | no_match",
  "selected_asin": "ASIN from candidate_items, or empty for no_match",
  "selected_options": {"option_name": "chosen value"},
  "rationale": "brief evidence-based reason"
}
""".strip()
        domain_rules = """
- Select only from candidate_items. Never invent an ASIN.
- Choose buy for the best available candidate even if some requirements cannot
  be met. Use no_match only if candidate_items contains no real selectable item.
- Earlier requirements remain active unless a later utterance relaxes, replaces,
  or removes them.
""".strip()
        domain_rules += "\n" + BEST_AVAILABLE_RULES
    elif domain == "travelplanner":
        feedback = turn.get("env_feedback") or {}
        common_context["original_query"] = (
            (instance.get("world_state") or {})
            .get("travelplanner_query_data", {})
            .get("query")
        )
        common_context["fixed_search_results"] = _compact_travel_search_results(
            feedback.get("search_results")
        )
        action_schema = """
"action": {
  "action_type": "plan",
  "itinerary": [
    {
      "day": "date/day",
      "current_city": "...",
      "transportation": "...",
      "breakfast": "...",
      "lunch": "...",
      "dinner": "...",
      "attraction": "...",
      "accommodation": "..."
    }
  ],
  "rationale": "brief evidence-based reason"
}
""".strip()
        domain_rules = """
- Use only options grounded in fixed_search_results.
- Produce a complete itinerary for the currently requested trip.
- Earlier requirements remain active unless a later utterance relaxes, replaces,
  or removes them.
""".strip()
        domain_rules += "\n" + TRAVEL_SELECTION_RULES
    else:
        raise ValueError(f"Unsupported domain: {domain}")

    return f"""
You are the agent under evaluation. Infer the user's current cumulative intention
from the user utterances, then execute the final domain action.

The gold intention is hidden. The canonical field vocabulary is only a naming aid;
do not assume every listed field is active. Omit inactive constraints.

Priority rules:
{INTENT_RULES}

Domain rules:
{domain_rules}

Return exactly one JSON object with this schema:
{{
  "current_intention_understanding": {INTENT_SCHEMA_JSON},
  {action_schema}
}}

BLIND_EVAL_CONTEXT:
{json.dumps(common_context, ensure_ascii=False, indent=2, default=str)}
""".strip()


def normalize_agent_output(
    raw: Dict[str, Any],
    *,
    domain: str,
    valid_asins: Sequence[str] = (),
) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("Agent output must be a JSON object")
    understanding = raw.get("current_intention_understanding")
    if not isinstance(understanding, dict):
        raise ValueError("Missing current_intention_understanding")
    understanding = normalize_intent_prediction(understanding)

    action = raw.get("action")
    if not isinstance(action, dict):
        raise ValueError("Missing action")
    action_type = str(action.get("action_type") or "").strip().lower()
    if domain == "webshop":
        if action_type not in {"buy", "no_match"}:
            raise ValueError(f"Invalid WebShop action_type: {action_type}")
        selected_asin = str(action.get("selected_asin") or "").strip().upper()
        valid = {str(value).strip().upper() for value in valid_asins if str(value).strip()}
        if action_type == "no_match" and valid:
            raise ValueError("Select the best available candidate; no_match is not allowed when candidates exist")
        if action_type == "buy" and selected_asin not in valid:
            raise ValueError(f"selected_asin is not in candidate_items: {selected_asin}")
        action["selected_asin"] = selected_asin
    else:
        if action_type != "plan" or not isinstance(action.get("itinerary"), list):
            raise ValueError("TravelPlanner action must contain action_type=plan and an itinerary")

    return {
        "current_intention_understanding": understanding,
        "action": copy.deepcopy(action),
    }


def build_judge_prompt(
    *,
    domain: str,
    instance_id: str,
    judged_turns: Sequence[Dict[str, Any]],
) -> str:
    return f"""
You are a strict evaluator for a two-layer intention-change benchmark.
The evaluated agent never saw gold annotations.

Layer 1, intention understanding:
- Agent predictions use intent items with field, value, priority.
  Match these semantically to the original gold, including gold entity constraints.
  Respect context and limits encoded in fields/values: Day 2 does not satisfy
  Day 1, and a budget ceiling is not an exact spending target.
  A field may have multiple items; never collapse them by field name.
  Legacy gold high/medium/low correspond to must_have/preferred/optional.
- For every gold constraint, decide whether the agent recognized it and whether
  the predicted current value is semantically correct.
- A renamed but clearly equivalent field may count as recognized.
- Latest user utterances override earlier values.
- Score priority_order_score from 0 to 1 as gold-weighted tier classification
  accuracy after semantic field matching: must_have=high, preferred=medium,
  optional=low (weights 3/2/1). Missing fields or wrong tiers earn no credit.
  Ignore order within tiers; relative ordering alone does not establish correct tiers.

Layer 2, action compliance:
- Judge the actual selected product or itinerary, not the agent rationale.
- For every gold constraint, label action_status as satisfied, violated, or unknown.
- unknown means the supplied action evidence is genuinely insufficient.
- The agent must select its best available concrete compromise even if some
  requirements cannot be met. Judge those unmet requirements normally; an
  explanation of the trade-off does not earn compliance credit.
- Leaving a required choice unresolved is not satisfied. A no_match action is
  inappropriate when real selectable candidates exist, even if all have defects.
  Set no_match_appropriate=false for such cases; absence of all real selectable
  candidates is the only possible no_match exception.
- A human_gold_action with confirmed=false is an unconfirmed annotation hint, not
  an authoritative answer. Evaluate the candidate evidence yourself.
- Do not give credit merely because the action repeats a requirement in prose.

Return exactly:
{{
  "turns": [
    {{
      "turn_id": 0,
      "constraint_judgments": [
        {{
          "gold_field": "exact gold field",
          "recognized": true,
          "value_match": true,
          "action_status": "satisfied | violated | unknown",
          "note": "brief reason"
        }}
      ],
      "predicted_extra_constraints": ["unsupported predicted field/value"],
      "priority_order_score": 0.0,
      "no_match_appropriate": false,
      "summary": "brief"
    }}
  ]
}}

Evaluate every supplied turn and every non-null gold constraint exactly once.

DOMAIN: {domain}
INSTANCE: {instance_id}
EVAL_PAYLOAD:
{json.dumps(list(judged_turns), ensure_ascii=False, indent=2, default=str)}
""".strip()


def priority_weights(gold_intention: Dict[str, Any]) -> Dict[str, float]:
    constraints = {
        str(field): value
        for field, value in (gold_intention.get("constraints") or {}).items()
        if value is not None
    }
    priority = gold_intention.get("priority")
    weights: Dict[str, float] = {}
    if isinstance(priority, dict):
        priority = {PRIORITY_ALIASES.get(k, k): v for k, v in priority.items()}
        for level, level_weight in PRIORITY_LEVEL_WEIGHTS.items():
            for field in priority.get(level) or []:
                field_name = str(field)
                if field_name in constraints:
                    weights[field_name] = level_weight
    elif isinstance(priority, list):
        ranked = [str(field) for field in priority if str(field) in constraints]
        count = len(ranked)
        for index, field in enumerate(ranked):
            weights[field] = float(max(count - index, 1))
    for field in constraints:
        weights.setdefault(field, 1.0)
    return weights


def score_judged_turn(
    *,
    gold_intention: Dict[str, Any],
    judgment: Dict[str, Any],
) -> Dict[str, Any]:
    constraints = {
        str(field): value
        for field, value in (gold_intention.get("constraints") or {}).items()
        if value is not None
    }
    expected_fields = set(constraints)
    rows = judgment.get("constraint_judgments")
    if not isinstance(rows, list):
        raise ValueError("Judge output is missing constraint_judgments")
    by_field: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        field = str(row.get("gold_field") or "")
        if field in by_field:
            raise ValueError(f"Judge returned duplicate gold_field: {field}")
        by_field[field] = row
    if set(by_field) != expected_fields:
        missing = sorted(expected_fields - set(by_field))
        extra = sorted(set(by_field) - expected_fields)
        raise ValueError(f"Judge field mismatch; missing={missing}, extra={extra}")

    weights = priority_weights(gold_intention)
    total_weight = sum(weights.values()) or 1.0
    recognized_weight = 0.0
    value_match_weight = 0.0
    satisfied_weight = 0.0
    violated_weight = 0.0
    unknown_weight = 0.0
    per_constraint = []
    for field in constraints:
        row = by_field[field]
        weight = weights[field]
        recognized = bool(row.get("recognized"))
        value_match = bool(row.get("value_match"))
        status = str(row.get("action_status") or "").strip().lower()
        if status not in {"satisfied", "violated", "unknown"}:
            raise ValueError(f"Invalid action_status for {field}: {status}")
        if recognized:
            recognized_weight += weight
        if value_match:
            value_match_weight += weight
        if status == "satisfied":
            satisfied_weight += weight
        elif status == "violated":
            violated_weight += weight
        else:
            unknown_weight += weight
        per_constraint.append(
            {
                "field": field,
                "gold_value": constraints[field],
                "weight": weight,
                "recognized": recognized,
                "value_match": value_match,
                "action_status": status,
                "note": row.get("note"),
            }
        )

    priority_score = float(judgment.get("priority_order_score", 0.0))
    priority_score = min(max(priority_score, 0.0), 1.0)
    constraint_score = value_match_weight / total_weight
    no_match_appropriate = bool(judgment.get("no_match_appropriate"))
    action_score = 1.0 if no_match_appropriate else satisfied_weight / total_weight
    max_weight = max(weights.values(), default=1.0)
    hard_violations = [
        item["field"]
        for item in per_constraint
        if item["weight"] == max_weight and item["action_status"] == "violated"
    ]
    extras = judgment.get("predicted_extra_constraints") or []
    return {
        "intention_understanding": {
            "weighted_constraint_recognition": recognized_weight / total_weight,
            "weighted_constraint_value_accuracy": constraint_score,
            "priority_order_score": priority_score,
            "combined_score": (constraint_score + priority_score) / 2.0,
            "predicted_extra_constraints": list(extras) if isinstance(extras, list) else [],
        },
        "action_compliance": {
            "weighted_constraint_score": action_score,
            "satisfied_weight": satisfied_weight,
            "violated_weight": violated_weight,
            "unknown_weight": unknown_weight,
            "total_weight": total_weight,
            "hard_priority_violation": bool(hard_violations),
            "hard_priority_violations": hard_violations,
            "no_match_appropriate": no_match_appropriate,
        },
        "per_constraint": per_constraint,
        "judge_summary": judgment.get("summary"),
    }


def aggregate_scored_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    def mean(values: Sequence[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    by_domain: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_domain.setdefault(str(row["domain"]), []).append(row)

    def summarize(items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "instances": len({item["instance_id"] for item in items}),
            "turns": len(items),
            "intention_understanding_score": mean(
                [item["scores"]["intention_understanding"]["combined_score"] for item in items]
            ),
            "constraint_value_accuracy": mean(
                [
                    item["scores"]["intention_understanding"][
                        "weighted_constraint_value_accuracy"
                    ]
                    for item in items
                ]
            ),
            "priority_order_score": mean(
                [item["scores"]["intention_understanding"]["priority_order_score"] for item in items]
            ),
            "action_compliance_score": mean(
                [item["scores"]["action_compliance"]["weighted_constraint_score"] for item in items]
            ),
            "hard_priority_violation_rate": mean(
                [
                    float(item["scores"]["action_compliance"]["hard_priority_violation"])
                    for item in items
                ]
            ),
        }

    return {
        "overall": summarize(rows),
        "by_domain": {domain: summarize(items) for domain, items in sorted(by_domain.items())},
    }


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def call_json_with_retries(
    client: Any,
    prompt: str,
    *,
    attempts: int = 3,
) -> Dict[str, Any]:
    last_error: Exception | None = None
    for _attempt in range(attempts):
        try:
            result = client.generate_json(prompt)
            if not isinstance(result, dict):
                raise ValueError("LLM result is not a JSON object")
            return result
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"LLM failed after {attempts} attempts") from last_error


def shuffled_copy(items: Sequence[Dict[str, Any]], seed: int) -> List[Dict[str, Any]]:
    copied = list(items)
    random.Random(seed).shuffle(copied)
    return copied


def finite_or_zero(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0
