"""Deterministically classify per-turn constraint priorities in a trajectory dataset."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, MutableMapping, Optional, Sequence, Tuple


PRIORITY_LEVELS = ("high", "medium", "low")
REMOVAL_OPS = {"remove", "delete", "drop"}


def _append_once(values: List[str], field: str) -> None:
    if field and field not in values:
        values.append(field)


def _non_null_fields(constraints: Any) -> List[str]:
    if not isinstance(constraints, dict):
        return []
    return [str(field) for field, value in constraints.items() if value is not None]


def _changed_fields(delta: Any) -> List[str]:
    if not isinstance(delta, dict):
        return []
    return [str(field) for field in delta if field != "priority"]


def _is_removed(field: str, delta: Any, constraints: Any) -> bool:
    if not isinstance(constraints, dict) or constraints.get(field) is None:
        return True
    change = delta.get(field) if isinstance(delta, dict) else None
    return isinstance(change, dict) and str(change.get("op", "")).lower() in REMOVAL_OPS


def _copy_priority(priority: Dict[str, List[str]]) -> Dict[str, List[str]]:
    return {level: list(priority[level]) for level in PRIORITY_LEVELS}


def classify_instance(instance: MutableMapping[str, Any]) -> Counter:
    turns = instance.get("turns")
    stats: Counter = Counter()
    if not isinstance(turns, list) or not turns:
        return stats

    first_gold = turns[0].get("gold_current_intention") or {}
    initial_constraints = first_gold.get("constraints") or {}
    initial_order = _non_null_fields(initial_constraints)

    previously_focused: List[str] = []
    removed_order: List[str] = []
    previous_priority: Optional[Dict[str, List[str]]] = None

    for turn_index, turn in enumerate(turns):
        gold = turn.get("gold_current_intention")
        if not isinstance(gold, dict):
            stats["missing_gold_intention"] += 1
            continue

        constraints = gold.get("constraints") or {}
        active_order = _non_null_fields(constraints)
        active = set(active_order)
        delta = turn.get("gold_delta") or {}
        changed = [] if turn_index == 0 else _changed_fields(delta)
        removed_now = [field for field in changed if _is_removed(field, delta, constraints)]
        active_focus = [field for field in changed if field in active and field not in removed_now]

        # Reintroduced fields are active again and must no longer be represented as abandoned.
        removed_order = [field for field in removed_order if field not in active_focus]
        for field in removed_now:
            _append_once(removed_order, field)

        high: List[str] = []
        if "category" in active:
            high.append("category")
        for field in active_focus:
            if field != "category":
                _append_once(high, field)

        medium: List[str] = []
        for field in previously_focused:
            if field in active and field not in high:
                _append_once(medium, field)

        low: List[str] = []
        for field in initial_order:
            if field != "category" and field not in high and field not in medium:
                _append_once(low, field)
        for field in removed_order:
            if field != "category" and field not in high and field not in medium:
                _append_once(low, field)

        # Keep the output exhaustive if a malformed trajectory introduces a field
        # without recording the corresponding delta.
        for field in active_order:
            if field not in high and field not in medium and field not in low:
                _append_once(low, field)
                stats["untracked_active_fallback_low"] += 1

        priority = {"high": high, "medium": medium, "low": low}
        gold["priority"] = priority
        stats["turns_classified"] += 1
        stats["removed_field_turns"] += len(removed_now)

        # Some source trajectories contain an LLM-authored priority delta. Keep
        # its rationale, but make its old/new payload agree with the classified state.
        if isinstance(delta, dict) and isinstance(delta.get("priority"), dict):
            priority_delta = delta["priority"]
            priority_delta["old"] = (
                _copy_priority(previous_priority)
                if previous_priority is not None
                else {"high": [], "medium": [], "low": []}
            )
            priority_delta["new"] = _copy_priority(priority)
            stats["priority_deltas_updated"] += 1

        for field in active_focus:
            _append_once(previously_focused, field)
        previous_priority = priority

    return stats


def load_instances(path: Path) -> Tuple[Any, Sequence[MutableMapping[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload, payload
    if isinstance(payload, dict):
        for key in ("instances", "data"):
            instances = payload.get(key)
            if isinstance(instances, list):
                return payload, instances
    raise ValueError("Expected a JSON list or an object containing an instances/data list")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    payload, instances = load_instances(args.input)
    totals: Counter = Counter()
    for instance in instances:
        if isinstance(instance, dict):
            totals.update(classify_instance(instance))
        else:
            totals["invalid_instances"] += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)

    print(f"instances={len(instances)}")
    for key in sorted(totals):
        print(f"{key}={totals[key]}")
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
