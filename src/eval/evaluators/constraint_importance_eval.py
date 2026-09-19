from __future__ import annotations

import copy
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

_SRC_DIR = Path(__file__).resolve().parents[2]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from models import DialogueInstance, TurnRecord
from eval.priority_schema import PRIORITY_ALIASES
from eval.intent_schema import normalize_intent_prediction


IMPORTANCE_WEIGHTS = {
    "high": 3,
    "medium": 2,
    "low": 1,
}


def _clean_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", str(value or "").strip().lower()).strip("_")


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _normalize_value(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = _normalize_text(value)
        if text in {"true", "yes", "y"}:
            return True
        if text in {"false", "no", "n"}:
            return False
        number_text = text.replace(",", "")
        try:
            return float(number_text)
        except ValueError:
            return text
    return _normalize_text(value)


def _values_match(gold_value: Any, predicted_value: Any) -> bool:
    gold = _normalize_value(gold_value)
    predicted = _normalize_value(predicted_value)
    if gold is None:
        return predicted is None
    if isinstance(gold, float) and isinstance(predicted, float):
        return math.isclose(gold, predicted, rel_tol=0.0, abs_tol=1e-6)
    return gold == predicted


def _non_null_constraints(raw_constraints: Any) -> Dict[str, Any]:
    if not isinstance(raw_constraints, dict):
        return {}
    return {
        _clean_key(field): value
        for field, value in raw_constraints.items()
        if _clean_key(field) and value is not None
    }


def _normalize_priority(priority: Any, constraint_fields: Iterable[str]) -> Dict[str, List[str]]:
    fields = [_clean_key(field) for field in constraint_fields if _clean_key(field)]
    known = set(fields)
    normalized = {"high": [], "medium": [], "low": []}

    if isinstance(priority, dict):
        priority = {PRIORITY_ALIASES.get(k, k): v for k, v in priority.items()}
        for level in normalized:
            values = priority.get(level) or []
            if not isinstance(values, list):
                continue
            for value in values:
                path = str(value)
                field = _clean_key(path[len("constraints."):] if path.startswith("constraints.") else path)
                if field and field in known and field not in normalized[level]:
                    normalized[level].append(field)
    elif isinstance(priority, list):
        for index, value in enumerate(priority):
            field = _clean_key(value)
            if not field or field not in known:
                continue
            if index == 0:
                level = "high"
            elif index <= 2:
                level = "medium"
            else:
                level = "low"
            if field not in normalized[level]:
                normalized[level].append(field)

    assigned = {field for values in normalized.values() for field in values}
    for field in fields:
        if field not in assigned:
            normalized["medium"].append(field)
    return normalized


def _priority_lookup(priority: Dict[str, List[str]]) -> Dict[str, Tuple[str, int]]:
    lookup: Dict[str, Tuple[str, int]] = {}
    for level, fields in priority.items():
        weight = IMPORTANCE_WEIGHTS[level]
        for field in fields:
            lookup[field] = (level, weight)
    return lookup


def evaluate_state_understanding(
    gold_intention: Dict[str, Any],
    predicted_intention: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if "intent" in gold_intention or (isinstance(predicted_intention, dict) and "intent" in predicted_intention):
        return _evaluate_atomic_understanding(gold_intention, predicted_intention or {})
    gold_constraints = _non_null_constraints(gold_intention.get("constraints"))
    gold_priority = _normalize_priority(gold_intention.get("priority"), gold_constraints.keys())
    gold_importance = _priority_lookup(gold_priority)

    predicted_intention = predicted_intention if isinstance(predicted_intention, dict) else {}
    predicted_constraints = _non_null_constraints(predicted_intention.get("constraints"))
    predicted_priority = _normalize_priority(predicted_intention.get("priority"), predicted_constraints.keys())
    predicted_importance = _priority_lookup(predicted_priority)

    total_weight = sum(gold_importance.get(field, ("medium", 2))[1] for field in gold_constraints)
    constraint_credit = 0
    priority_credit = 0
    per_constraint = []

    for field, gold_value in gold_constraints.items():
        gold_level, weight = gold_importance.get(field, ("medium", 2))
        predicted_value = predicted_constraints.get(field)
        predicted_level = predicted_importance.get(field, (None, 0))[0]
        value_match = field in predicted_constraints and _values_match(gold_value, predicted_value)
        priority_match = predicted_level == gold_level
        if value_match:
            constraint_credit += weight
        if priority_match:
            priority_credit += weight
        per_constraint.append(
            {
                "field": field,
                "gold_value": gold_value,
                "predicted_value": predicted_value,
                "importance": gold_level,
                "weight": weight,
                "value_match": value_match,
                "priority_match": priority_match,
                "constraint_credit": weight if value_match else 0,
                "priority_credit": weight if priority_match else 0,
            }
        )

    denominator = total_weight or 1
    constraint_score = constraint_credit / denominator
    priority_score = priority_credit / denominator
    return {
        "constraint_weighted_score": constraint_score,
        "priority_level_weighted_score": priority_score,
        "combined_weighted_score": (constraint_score + priority_score) / 2,
        "constraint_credit": constraint_credit,
        "priority_credit": priority_credit,
        "total_possible_credit": total_weight,
        "per_constraint": per_constraint,
        "predicted_explanation": predicted_intention.get("explanation"),
    }


def _atomic_entries(intention: Dict[str, Any]) -> List[Dict[str, Any]]:
    if "intent" in intention:
        return normalize_intent_prediction(intention)["intent"]
    # Legacy gold remains untouched on disk. Prose/compound gold still requires
    # the semantic judge; only top-level fields are bridged here.
    constraints = _non_null_constraints(intention.get("constraints"))
    priority = _priority_lookup(_normalize_priority(intention.get("priority"), constraints))
    inverse = {v:k for k,v in PRIORITY_ALIASES.items()}
    return [{"field": field,
             "value": value,
             "priority": inverse[priority[field][0]]}
            for field, value in constraints.items()]


def _evaluate_atomic_understanding(gold: Dict[str, Any], prediction: Dict[str, Any]) -> Dict[str, Any]:
    expected = _atomic_entries(gold)
    actual = _atomic_entries(prediction)
    weights = {"must_have": 3, "preferred": 2, "optional": 1}

    def key(item):
        field = _clean_key(item["field"])
        field = "budget" if field == "budget_max" else field
        return field

    used = set()
    rows = []
    constraint_credit = priority_credit = total_weight = 0
    for item in expected:
        candidates = [i for i,x in enumerate(actual) if i not in used and key(x) == key(item)]
        matching = [i for i in candidates if _values_match(item["value"], actual[i]["value"])]
        index = (matching or candidates or [None])[0]
        candidate = actual[index] if index is not None else None
        if index is not None:
            used.add(index)
        value_match = candidate is not None and _values_match(item["value"], candidate["value"])
        priority_match = candidate is not None and candidate["priority"] == item["priority"]
        weight = weights[item["priority"]]
        total_weight += weight
        constraint_credit += weight if value_match else 0
        priority_credit += weight if priority_match else 0
        rows.append({"field": item["field"],
                     "gold_value": item["value"], "predicted_value": candidate["value"] if candidate else None,
                     "importance": PRIORITY_ALIASES[item["priority"]], "weight": weight,
                     "value_match": value_match, "priority_match": priority_match,
                     "constraint_credit": weight if value_match else 0, "priority_credit": weight if priority_match else 0})
    denominator = total_weight or 1
    return {"constraint_weighted_score": constraint_credit / denominator,
            "priority_level_weighted_score": priority_credit / denominator,
            "combined_weighted_score": (constraint_credit + priority_credit) / (2 * denominator),
            "constraint_credit": constraint_credit, "priority_credit": priority_credit,
            "total_possible_credit": total_weight, "per_constraint": rows,
            "predicted_extra_intent": [x for i,x in enumerate(actual) if i not in used],
            "predicted_explanation": prediction.get("explanation"),
            "matching_mode": "field/value matching; contextual or paraphrased gold requires semantic judging"}


def evaluate_action_selection(turn: Dict[str, Any]) -> Dict[str, Any]:
    gold_intention = turn.get("gold_current_intention") or {}
    gold_constraints = _non_null_constraints(gold_intention.get("constraints"))
    gold_priority = _normalize_priority(gold_intention.get("priority"), gold_constraints.keys())
    gold_importance = _priority_lookup(gold_priority)
    env_feedback = turn.get("env_feedback") or {}

    satisfied = {
        _clean_key(field)
        for field in (
            env_feedback.get("gold_eval_satisfied_constraints")
            or env_feedback.get("satisfied_constraints")
            or []
        )
    }
    violated = {
        _clean_key(field)
        for field in (
            env_feedback.get("gold_eval_violated_constraints")
            or env_feedback.get("violated_constraints")
            or []
        )
    }
    selected_asin = env_feedback.get("selected_asin")

    total_weight = sum(gold_importance.get(field, ("medium", 2))[1] for field in gold_constraints)
    satisfied_credit = 0
    unknown_credit = 0
    per_constraint = []

    for field in gold_constraints:
        level, weight = gold_importance.get(field, ("medium", 2))
        if field in satisfied:
            status = "satisfied"
            credit = weight
            satisfied_credit += credit
        elif field in violated:
            status = "violated"
            credit = 0
        else:
            status = "unknown"
            credit = 0
            unknown_credit += weight
        per_constraint.append(
            {
                "field": field,
                "importance": level,
                "weight": weight,
                "status": status,
                "credit": credit,
            }
        )

    denominator = total_weight or 1
    return {
        "selected_asin": selected_asin,
        "weighted_score": satisfied_credit / denominator,
        "satisfied_credit": satisfied_credit,
        "unknown_credit": unknown_credit,
        "total_possible_credit": total_weight,
        "per_constraint": per_constraint,
        "constraint_debug": copy.deepcopy(
            env_feedback.get("gold_eval_constraint_debug")
            or env_feedback.get("constraint_debug")
            or {}
        ),
    }


def evaluate_turn_dict(turn: Dict[str, Any]) -> Dict[str, Any]:
    gold_intention = turn.get("gold_current_intention") or {}
    predicted_intention = turn.get("agent_intention_prediction")
    state_eval = evaluate_state_understanding(gold_intention, predicted_intention)
    action_eval = evaluate_action_selection(turn)
    return {
        "state_understanding_eval": state_eval,
        "action_selection_eval": action_eval,
    }


def attach_turn_evaluation(turn: TurnRecord) -> None:
    turn_dict = {
        "gold_current_intention": copy.deepcopy(turn.gold_current_intention),
        "agent_intention_prediction": copy.deepcopy(turn.agent_intention_prediction),
        "env_feedback": copy.deepcopy(turn.env_feedback),
    }
    turn.evaluation = evaluate_turn_dict(turn_dict)


def attach_instance_evaluations(instance: DialogueInstance) -> DialogueInstance:
    for turn in instance.turns:
        attach_turn_evaluation(turn)
    return instance


def evaluate_dataset_payload(payload: Any) -> List[Dict[str, Any]]:
    instances = payload.get("instances") if isinstance(payload, dict) else payload
    if not isinstance(instances, list):
        raise ValueError("Evaluation input must be a list of instances or an object with an instances list")

    evaluated = copy.deepcopy(instances)
    for instance in evaluated:
        if not isinstance(instance, dict):
            continue
        for turn in instance.get("turns") or []:
            if isinstance(turn, dict):
                turn["evaluation"] = evaluate_turn_dict(turn)
    return evaluated


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Benchmark/simulation JSON to evaluate.")
    parser.add_argument("--output", required=True, help="Path for evaluated JSON.")
    args = parser.parse_args()

    input_path = Path(args.input)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    evaluated = evaluate_dataset_payload(payload)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(evaluated, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
