"""Deterministically classify priorities from each turn's latest constraint focus."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple


PRIORITY_LEVELS = ("high", "medium", "low")
REMOVAL_OPS = {"remove", "delete", "drop"}
TRAVELPLANNER_CONTEXT_FIELDS = {
    "org",
    "dest",
    "start_date",
    "end_date",
    "visiting_city_number",
}


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


def _extracted_mentioned_fields(turn: Any, active: set[str]) -> Tuple[bool, List[str]]:
    """Return whether extraction is authoritative and its grounded fields."""
    if not isinstance(turn, dict):
        return False, []
    extraction = turn.get("constraint_extraction") or {}
    if not isinstance(extraction, dict) or not isinstance(extraction.get("mentioned_fields"), list):
        return False, []
    mentioned: List[str] = []
    for raw_field in extraction.get("mentioned_fields") or []:
        field = str(raw_field)
        if field in active:
            _append_once(mentioned, field)
    return True, mentioned


def _is_removed(field: str, delta: Any, active_fields: set[str]) -> bool:
    change = delta.get(field) if isinstance(delta, dict) else None
    if isinstance(change, dict):
        if str(change.get("op", "")).lower() in REMOVAL_OPS:
            return True
        # A relaxed/overridden constraint whose new value is null is also an
        # explicit removal, even if a malformed trajectory retained its old
        # value in gold_current_intention.constraints.
        if "new" in change and change.get("new") is None:
            return True
    return field not in active_fields


def _remove_constraint(gold: MutableMapping[str, Any], field: str) -> None:
    """Repair stale constraint state when gold_delta explicitly removes a field."""
    prefix = "entities."
    marker = ".constraints."
    if field.startswith(prefix) and marker in field:
        entity_id, constraint_field = field[len(prefix) :].split(marker, 1)
        entities = gold.get("entities")
        entity = entities.get(entity_id) if isinstance(entities, dict) else None
        constraints = entity.get("constraints") if isinstance(entity, dict) else None
        if isinstance(constraints, dict):
            constraints.pop(constraint_field, None)
        return

    constraints = gold.get("constraints")
    if isinstance(constraints, dict):
        constraints.pop(field, None)


def _copy_priority(priority: Dict[str, List[str]]) -> Dict[str, List[str]]:
    return {level: list(priority[level]) for level in PRIORITY_LEVELS}


def classify_instance(
    instance: MutableMapping[str, Any],
    mention_overrides: Optional[Mapping[str, Sequence[str]]] = None,
    *,
    repair_removed_constraints: bool = True,
    drop_travelplanner_context_fields: bool = False,
) -> Counter:
    turns = instance.get("turns")
    stats: Counter = Counter()
    if not isinstance(turns, list) or not turns:
        return stats

    previous_turn_focus: List[str] = []
    previous_priority: Optional[Dict[str, List[str]]] = None

    for turn_index, turn in enumerate(turns):
        gold = turn.get("gold_current_intention")
        if not isinstance(gold, dict):
            stats["missing_gold_intention"] += 1
            continue

        delta = turn.get("gold_delta") or {}
        if drop_travelplanner_context_fields:
            constraints = gold.get("constraints")
            if isinstance(constraints, dict):
                for field in TRAVELPLANNER_CONTEXT_FIELDS:
                    if field in constraints:
                        constraints.pop(field)
                        stats["travelplanner_context_constraints_removed"] += 1
            if isinstance(delta, dict):
                for field in TRAVELPLANNER_CONTEXT_FIELDS:
                    delta.pop(field, None)
        changed = [] if turn_index == 0 else _changed_fields(delta)
        raw_active = set(_active_fields(gold))
        removed_now = [field for field in changed if _is_removed(field, delta, raw_active)]
        if repair_removed_constraints:
            for field in removed_now:
                _remove_constraint(gold, field)

        active_order = _active_fields(gold)
        active = set(active_order)
        active_focus = [field for field in changed if field in active and field not in removed_now]

        reprioritized = _explicit_reprioritized_fields(turn, active)
        has_extracted_mentions, extracted_mentions = _extracted_mentioned_fields(turn, active)
        turn_key = str(turn.get("turn_id", turn_index))
        if mention_overrides is not None and turn_key in mention_overrides:
            has_extracted_mentions = True
            extracted_mentions = []
            for raw_field in mention_overrides[turn_key]:
                field = str(raw_field)
                if field in active:
                    _append_once(extracted_mentions, field)
        # Turn 0 is the user's current request, so every stated constraint is
        # a must-have. On later turns, only constraints actually changed or
        # explicitly reprioritized in this turn remain must-have.
        high: List[str] = list(active_order) if turn_index == 0 else []
        if turn_index > 0:
            focus = (
                extracted_mentions
                if has_extracted_mentions
                else reprioritized + active_focus
            )
            for field in focus:
                _append_once(high, field)

        medium: List[str] = []
        # Preferred is deliberately one-turn memory, not an accumulating set
        # of everything that was important at any point in the trajectory.
        for field in previous_turn_focus:
            if field in active and field not in high and field not in removed_now:
                _append_once(medium, field)

        low: List[str] = []
        # Every other still-active constraint is optional. This also keeps the
        # output exhaustive when a trajectory introduces a field without a
        # matching delta.
        for field in active_order:
            if field not in high and field not in medium and field not in low:
                _append_once(low, field)

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

        previous_turn_focus = list(high)
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
    parser.add_argument(
        "--mention-overrides",
        type=Path,
        help="Optional JSON object mapping instance_id -> turn_id -> explicitly mentioned fields.",
    )
    parser.add_argument(
        "--priority-only",
        action="store_true",
        help="Update priority without repairing or deleting constraint state.",
    )
    parser.add_argument(
        "--drop-travelplanner-context-fields",
        action="store_true",
        help=(
            "Remove org, dest, start_date, end_date, and visiting_city_number "
            "from constraints before classifying priority."
        ),
    )
    args = parser.parse_args()

    payload, instances = load_instances(args.input)
    overrides: Dict[str, Mapping[str, Sequence[str]]] = {}
    if args.mention_overrides:
        loaded_overrides = json.loads(args.mention_overrides.read_text(encoding="utf-8"))
        if not isinstance(loaded_overrides, dict):
            raise ValueError("--mention-overrides must contain a JSON object")
        overrides = loaded_overrides
    totals: Counter = Counter()
    for instance in instances:
        if isinstance(instance, dict):
            instance_id = str(instance.get("instance_id") or "")
            totals.update(
                classify_instance(
                    instance,
                    overrides.get(instance_id),
                    repair_removed_constraints=not args.priority_only,
                    drop_travelplanner_context_fields=args.drop_travelplanner_context_fields,
                )
            )
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
