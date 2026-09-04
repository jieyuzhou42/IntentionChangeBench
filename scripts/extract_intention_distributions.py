"""Extract intention-focused trajectories and summarize shift distributions."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


TURN_FIELDS = (
    "turn_id",
    "user_utterance",
    "trigger_evidence",
    "shift_condition",
    "gold_delta",
    "gold_current_intention",
    "linguistic_style",
    "action_implication",
)


def _atomic_changes(turn: Dict[str, Any]) -> List[Dict[str, Any]]:
    condition = turn.get("shift_condition") or {}
    details = condition.get("details") or {}
    changes = details.get("changes") or []
    valid = [change for change in changes if isinstance(change, dict)]
    if valid:
        return valid

    category = str(details.get("change_category") or details.get("op") or "none")
    if category == "none" or not details.get("intention_changed", bool(condition)):
        return []
    return [details]


def _distribution(counter: Counter[str]) -> Dict[str, Dict[str, float]]:
    total = sum(counter.values())
    return {
        key: {
            "count": count,
            "percent": round(100.0 * count / total, 2) if total else 0.0,
        }
        for key, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    }


def extract_and_summarize(
    instances: Iterable[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    extracted: List[Dict[str, Any]] = []
    change_count_distribution: Counter[str] = Counter()
    category_distribution: Counter[str] = Counter()
    condition_distribution: Counter[str] = Counter()
    linguistic_distribution: Counter[str] = Counter()
    multi_category_combinations: Counter[str] = Counter()
    selection_modes: Counter[str] = Counter()
    preferred_multi = Counter()
    initial_turns = 0
    shift_turns = 0
    changed_turns = 0
    multi_turns = 0
    atomic_changes_total = 0

    for instance in instances:
        output_instance = {
            key: instance.get(key)
            for key in ("instance_id", "task_type", "subtype", "world_state")
            if key in instance
        }
        output_turns: List[Dict[str, Any]] = []
        for turn in instance.get("turns", []):
            output_turns.append({key: turn.get(key) for key in TURN_FIELDS if key in turn})
            if int(turn.get("turn_id") or 0) == 0:
                initial_turns += 1
                continue

            shift_turns += 1
            changes = _atomic_changes(turn)
            change_count = len(changes)
            change_count_distribution[str(change_count)] += 1
            if change_count:
                changed_turns += 1
            if change_count >= 2:
                multi_turns += 1
            atomic_changes_total += change_count

            categories: List[str] = []
            for change in changes:
                category = str(
                    change.get("change_category") or change.get("category") or change.get("op") or "none"
                )
                category_distribution[category] += 1
                categories.append(category)
            if change_count >= 2:
                multi_category_combinations[" + ".join(sorted(categories))] += 1

            shift_condition = turn.get("shift_condition") or {}
            condition_distribution[str(shift_condition.get("type") or "none")] += 1
            linguistic_distribution[str(turn.get("linguistic_style") or "unknown")] += 1

            sampling = (shift_condition.get("details") or {}).get("candidate_sampling") or {}
            selection_modes[str(sampling.get("selection_mode") or "unknown")] += 1
            preferred_multi["preferred" if sampling.get("prefer_multi") else "not_preferred"] += 1

        output_instance["turns"] = output_turns
        extracted.append(output_instance)

    summary = {
        "instances": len(extracted),
        "initial_turns": initial_turns,
        "shift_turns": shift_turns,
        "changed_turns": changed_turns,
        "atomic_changes": atomic_changes_total,
        "multi_change_turns": multi_turns,
        "multi_change_rate_percent": round(100.0 * multi_turns / changed_turns, 2)
        if changed_turns
        else 0.0,
        "change_count_per_shift_turn": _distribution(change_count_distribution),
        "change_category_per_atomic_change": _distribution(category_distribution),
        "condition_per_shift_turn": _distribution(condition_distribution),
        "linguistic_realization_per_shift_turn": _distribution(linguistic_distribution),
        "candidate_selection_mode": _distribution(selection_modes),
        "multi_preference_schedule": _distribution(preferred_multi),
        "multi_category_combinations": _distribution(multi_category_combinations),
    }
    return extracted, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.input.open("r", encoding="utf-8") as handle:
        instances = json.load(handle)
    if not isinstance(instances, list):
        raise ValueError("Input trajectory must be a JSON list")

    extracted, summary = extract_and_summarize(instances)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(extracted, handle, ensure_ascii=False, indent=2)
    with args.report.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
