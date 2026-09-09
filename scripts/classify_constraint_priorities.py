"""Deterministically classify per-turn constraint priorities by mention recency.

high means must-have in the current turn, medium means preferred from the
previous turn, and low means optional because it was last mentioned earlier.
"""

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


def _active_fields(gold: Any) -> List[str]:
    if not isinstance(gold, dict):
        return []
    fields = _non_null_fields(gold.get("constraints"))
    entities = gold.get("entities")
    if isinstance(entities, dict):
        for entity_id, entity in entities.items():
            if not isinstance(entity, dict):
                continue
            for field in _non_null_fields(entity.get("constraints")):
                _append_once(fields, f"entities.{entity_id}.constraints.{field}")
    return fields


def _explicit_reprioritized_fields(turn: Any, active: set[str]) -> List[str]:
    if not isinstance(turn, dict):
        return []
    details = ((turn.get("shift_condition") or {}).get("details") or {})
    candidates: List[Any] = []
    if details.get("change_category") == "reprioritize":
        candidates.extend(details.get("priority_update") or [])
    for change in details.get("changes") or []:
        if isinstance(change, dict) and change.get("op") == "reprioritize":
            candidates.extend(change.get("priority_update") or [])
    entity_priority = (turn.get("gold_current_intention") or {}).get("entity_priority")
    if isinstance(entity_priority, list):
        candidates.extend(entity_priority[:1])
    prioritized: List[str] = []
    for field in candidates:
        field = str(field)
        if field in active:
            _append_once(prioritized, field)
            break
    return prioritized


def _changed_fields(delta: Any) -> List[str]:
    if not isinstance(delta, dict):
        return []
    return [str(field) for field in delta if field != "priority"]


def _is_removed(field: str, delta: Any, active: set[str]) -> bool:
    if field not in active:
        return True
    change = delta.get(field) if isinstance(delta, dict) else None
    return isinstance(change, dict) and (
        str(change.get("op", "")).lower() in REMOVAL_OPS
        or change.get("new") is None
    )


def _copy_priority(priority: Dict[str, List[str]]) -> Dict[str, List[str]]:
    return {level: list(priority[level]) for level in PRIORITY_LEVELS}


def classify_instance(instance: MutableMapping[str, Any]) -> Counter:
    turns = instance.get("turns")
    stats: Counter = Counter()
    if not isinstance(turns, list) or not turns:
        return stats

    first_gold = turns[0].get("gold_current_intention") or {}
    initial_order = _active_fields(first_gold)
    last_mentioned_turn: Dict[str, int] = {}
    previous_priority: Optional[Dict[str, List[str]]] = None

    for turn_index, turn in enumerate(turns):
        gold = turn.get("gold_current_intention")
        if not isinstance(gold, dict):
            stats["missing_gold_intention"] += 1
            continue

        active_order = _active_fields(gold)
        active = set(active_order)
        delta = turn.get("gold_delta") or {}
        changed = active_order if turn_index == 0 else _changed_fields(delta)
        removed_now = [field for field in changed if _is_removed(field, delta, active)]
        active_focus = [
            field for field in changed if field in active and field not in removed_now
        ]
        reprioritized = _explicit_reprioritized_fields(turn, active)

        for field in list(last_mentioned_turn):
            if field not in active:
                del last_mentioned_turn[field]

        mentioned_now: List[str] = []
        for field in reprioritized + active_focus:
            _append_once(mentioned_now, field)
            last_mentioned_turn[field] = turn_index

        priority = {level: [] for level in PRIORITY_LEVELS}
        ordered_fields: List[str] = []
        for field in mentioned_now + initial_order + active_order:
            if field in active:
                _append_once(ordered_fields, field)
        for field in ordered_fields:
            last_turn = last_mentioned_turn.get(field)
            if last_turn == turn_index:
                level = "high"
            elif last_turn == turn_index - 1:
                level = "medium"
            else:
                level = "low"
                if last_turn is None:
                    stats["untracked_active_fallback_low"] += 1
            priority[level].append(field)

        gold["priority"] = priority
        stats["turns_classified"] += 1
        stats["removed_field_turns"] += len(removed_now)

        # Keep any existing priority delta consistent with the rule-derived state.
        if isinstance(delta, dict) and isinstance(delta.get("priority"), dict):
            priority_delta = delta["priority"]
            priority_delta["old"] = (
                _copy_priority(previous_priority)
                if previous_priority is not None
                else {"high": [], "medium": [], "low": []}
            )
            priority_delta["new"] = _copy_priority(priority)
            stats["priority_deltas_updated"] += 1

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
