#!/usr/bin/env python3
"""Apply annotate_shard002_spec.py to shard_002_annotated.json.

Rewrites turns 1..N of the nine instances after 0003, following the conventions
0003 was reviewed under. 0003 itself is left alone except for one arithmetic
fix: its self-driving legs wrote "cost: $68" with no "=", which the cost parser
in check_gold_action.py does not pick up, so its trips were being checked
$136 light. Rendering them as "... = $68" makes the budget check honest; the
plan is unchanged and still fits.

What gets written per turn:
    user_utterance, gold_delta, gold_current_intention.{constraints,priority},
    linguistic_style, action_implication, gold_action, and a trigger_evidence /
    shift_condition pair regenerated from the new utterance so the replay UI
    does not show a rationale left over from a previous version of the turn.

Everything else on the turn -- agent_action, env_feedback, rollout_trace, the
search results the pools come from -- is left untouched.

Usage:
    python annotation/tools/build_shard002_annotation.py            # in place, with backup
    python annotation/tools/build_shard002_annotation.py --output OUT.json
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

SPEC_PATH = Path(__file__).with_name("annotate_shard002_spec.py")
_spec = importlib.util.spec_from_file_location("annotate_shard002_spec", SPEC_PATH)
plan = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(plan)

DEFAULT_PATH = Path("annotation/data/travelplanner_v3_shards/shard_002_annotated.json")
MEALS = ("breakfast", "lunch", "dinner")
LEVELS = ("high", "medium", "low")


# --------------------------------------------------------------------------
# pools
# --------------------------------------------------------------------------
def collect_pool(instance: Dict[str, Any], key: str, name_field: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for turn in instance.get("turns") or []:
        results = (turn.get("env_feedback") or {}).get("search_results") or {}
        for page in results.get(key) or []:
            for item in page.get("items") or []:
                if isinstance(item, dict) and item.get(name_field):
                    out.setdefault(str(item[name_field]), item)
    return out


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------
def render_leg(leg: Dict[str, Any], people: int) -> str:
    kind = leg["kind"]
    if kind == "text":
        return leg["text"]
    if kind == "flight":
        label = "Outbound" if leg["leg"] == "outbound" else "Return"
        total = leg["price"] * people
        return (
            f"{label} Flight {leg['number']}: {leg['org']} to {leg['dest']} on {leg['date']}; "
            f"{leg['dep']}–{leg['arr']} local times; "
            f"${leg['price']} per person × {people} = ${total}; "
            "airport transfers and check-in timing pending verification"
        )
    label = "Own-car outbound" if leg["leg"] == "outbound" else "Return by car"
    # The "= $N" is what makes the leg visible to the itinerary cost parser;
    # a car is priced per vehicle, so it does not multiply by people.
    return (
        f"{label}: self-driving, from {leg['org']} to {leg['dest']}, "
        f"duration: {leg['duration']}, distance: {leg['distance']}, "
        f"cost: ${leg['cost']} per vehicle = ${leg['cost']}"
    )


def render_accommodation(item: Dict[str, Any], nights: int, checkin: str, checkout: str) -> str:
    price = float(item.get("price") or 0)
    name = re.sub(r"\s+", " ", str(item.get("NAME") or "")).strip()
    return (
        f"{name}; {item.get('room type')}; rating {item.get('review rate number')}; "
        f"${price:.0f}/night × {nights} nights = ${price * nights:.0f} for one unit; "
        f"minimum stay {item.get('minimum nights')} nights; "
        f"maximum occupancy {item.get('maximum occupancy')}; "
        f"check in {checkin}, check out {checkout}; "
        f"house rules: {item.get('house_rules')}"
    )


def render_meal(item: Dict[str, Any], people: int) -> str:
    name = re.sub(r"\s+", " ", str(item.get("Name") or "")).strip()
    return (
        f"{name}; rating {item.get('Aggregate Rating')}; "
        f"listed Average Cost ${item.get('Average Cost')} per person × {people}; "
        f"Cuisines: {item.get('Cuisines')}"
    )


def build_itinerary(
    spec: Dict[str, Any],
    turn_spec: Dict[str, Any],
    people: int,
    accommodations: Dict[str, Dict[str, Any]],
    restaurants: Dict[str, Dict[str, Any]],
    transport: Dict[str, Any],
) -> List[Dict[str, Any]]:
    gold = turn_spec["gold"]
    nights = gold.get("nights_override") or spec["nights"]
    checkout = gold.get("checkout_override") or spec["checkout"]

    if gold.get("acc"):
        item = accommodations.get(gold["acc"])
        if item is None:
            raise KeyError(f"{spec['instance_id']}: accommodation {gold['acc']!r} not in pool")
        acc_text = render_accommodation(item, nights, spec["checkin"], checkout)
    else:
        acc_text = gold["infeasible"]

    meal_names = gold.get("meals")
    if meal_names is None:
        meal_texts = [gold.get("meal_infeasible") or plan.TRAVEL_MEAL] * 3
    else:
        meal_texts = []
        for name in meal_names:
            item = restaurants.get(name)
            if item is None:
                raise KeyError(f"{spec['instance_id']}: restaurant {name!r} not in pool")
            meal_texts.append(render_meal(item, people))

    itinerary: List[Dict[str, Any]] = []
    for day in spec["days"]:
        role = day["role"]
        entry = {
            "day": day["date"],
            "current_city": day["city"],
            "transportation": (
                plan.LOCAL_TRANSFER if role == "stay"
                else render_leg(transport[role], people)
            ),
            "breakfast": plan.TRAVEL_MEAL,
            "lunch": plan.TRAVEL_MEAL,
            "dinner": plan.TRAVEL_MEAL,
            "attraction": "-",
            "accommodation": acc_text if role in ("depart", "stay") else "-",
        }
        if role == "stay":
            entry["breakfast"], entry["lunch"], entry["dinner"] = meal_texts
            entry["attraction"] = gold.get("attraction") or "-"
        itinerary.append(entry)
    return itinerary


# --------------------------------------------------------------------------
# constraints / priority / delta
# --------------------------------------------------------------------------
def apply_delta(constraints: Dict[str, Any], delta: Dict[str, Any]) -> Dict[str, Any]:
    """Return the gold_delta record for this turn and mutate constraints in place."""
    record: Dict[str, Any] = {}
    for field, (op, value) in delta.items():
        old = constraints.get(field)
        if op == "remove":
            constraints.pop(field, None)
            record[field] = {"op": "remove", "old": old, "new": None}
            continue
        constraints[field] = value
        record[field] = {"op": op, "old": old if op == "override" else None, "new": value}
    return record


def build_priority(
    constraints: Dict[str, Any],
    overrides: Dict[str, str],
    touched: set,
) -> Dict[str, List[str]]:
    """budget and every actively requested constraint are must-have; query-derived
    context the user has never touched drops to optional so it cannot dominate the
    3/2/1 weighting; `overrides` demotes the fields a turn deliberately softened.

    A context field the user does change -- 0031 moves `days` to 4 and adds a
    second traveller -- stops being background and goes back to must-have."""
    priority = {level: [] for level in LEVELS}
    for field in constraints:
        level = overrides.get(field)
        if level is None:
            level = "low" if field in plan.CONTEXT_FIELDS and field not in touched else "high"
        priority[level].append(field)
    return priority


def build_evidence(utterance: str, delta_record: Dict[str, Any],
                   old_priority: Dict[str, List[str]],
                   new_priority: Dict[str, List[str]]) -> Dict[str, Any]:
    changes = []
    for field, change in delta_record.items():
        changes.append({
            "field": field,
            "op": change["op"],
            "old_value": change["old"],
            "value": change["new"],
            "rationale": f"Current user request: {utterance}",
        })
    if old_priority != new_priority:
        changes.append({
            "field": "priority",
            "op": "reprioritize",
            "old_value": old_priority,
            "value": new_priority,
            "rationale": "Update relative importance only as expressed by the "
                         "cumulative user requests; numeric bounds remain active.",
        })
    rationale = f"Apply the current utterance to the preceding corrected intention: {utterance}"
    op = "multiple" if len(changes) > 1 else (changes[0]["op"] if changes else "none")
    return {
        "trigger_evidence": {
            "trigger_type": "user_request",
            "source": "annotation_revision",
            "details": {"rationale": rationale, "changes": changes},
        },
        "shift_condition": {
            "type": "user_preference",
            "source": "annotation_revision",
            "reason": rationale,
            "details": {"intention_changed": bool(changes), "op": op, "changes": changes},
        },
    }


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def build_instance(instance: Dict[str, Any], spec: Dict[str, Any]) -> None:
    turns = instance["turns"]
    wanted = len(spec["turns"]) + 1
    if len(turns) < wanted:
        raise ValueError(
            f"{spec['instance_id']}: spec has {len(spec['turns'])} turns after t0, "
            f"instance only has {len(turns) - 1}"
        )
    # A spec may deliberately use fewer turns than the source trajectory -- 0023
    # packs its constraints into three dense turns instead of six thin ones, the
    # same shape as the V1_single_turn control. Every turn carries the full
    # search_results, so dropping the tail loses no candidate pool.
    if len(turns) > wanted:
        del turns[wanted:]

    accommodations = collect_pool(instance, "accommodations", "NAME")
    restaurants = collect_pool(instance, "restaurants", "Name")

    base = (turns[0].get("gold_current_intention") or {}).get("constraints") or {}
    constraints = copy.deepcopy(base)
    priority = {"high": list(constraints), "medium": [], "low": []}
    turns[0]["gold_current_intention"]["priority"] = copy.deepcopy(priority)

    overrides: Dict[str, str] = {}
    touched: set = set()
    # A transport swap persists for the rest of the trajectory, the same way a
    # constraint does -- 0007 changes from flying to driving at t3 and stays there.
    transport = copy.deepcopy(spec["transport"])
    for index, turn_spec in enumerate(spec["turns"], start=1):
        turn = turns[index]
        old_priority = copy.deepcopy(priority)
        transport.update(turn_spec.get("transport") or {})

        delta_record = apply_delta(constraints, turn_spec.get("delta") or {})
        touched.update(delta_record)
        levels_before = {
            field: level
            for level in LEVELS
            for field in old_priority.get(level) or []
        }
        # Priority overrides are sticky: a field softened at turn k stays soft
        # unless a later turn moves it again.
        for level, fields in (turn_spec.get("priority") or {}).items():
            for field in fields:
                overrides[field] = level
        for field in list(overrides):
            if field not in constraints:
                overrides.pop(field)
        priority = build_priority(constraints, overrides, touched)

        # A turn can move a constraint between tiers without touching its value
        # -- 0006 t5 promotes a preference to a requirement, 0040 t4 demotes one.
        # Record that in gold_delta too, or the UI shows the turn as a no-op.
        for level in LEVELS:
            for field in priority[level]:
                before = levels_before.get(field)
                if before is not None and before != level and field not in delta_record:
                    delta_record[field] = {"op": "reprioritize", "old": before, "new": level}

        people = int(constraints.get("people_number") or spec["people"])
        itinerary = build_itinerary(
            spec, turn_spec, people, accommodations, restaurants, transport
        )

        turn["user_utterance"] = turn_spec["utterance"]
        turn["gold_delta"] = delta_record
        turn["gold_current_intention"] = {
            **(turn.get("gold_current_intention") or {}),
            "constraints": copy.deepcopy(constraints),
            "priority": copy.deepcopy(priority),
            "domain": "travelplanner",
        }
        turn["linguistic_style"] = turn_spec["style"]
        turn["action_implication"] = "continue"
        turn["gold_action"] = {
            "action_type": "Planner",
            "confirmed": False,
            "status": "pending_adjudication",
            "action_payload": {"plan": {"itinerary": itinerary}},
        }
        turn.update(build_evidence(turn_spec["utterance"], delta_record, old_priority, priority))


def fix_0003_drive_cost(instance: Dict[str, Any]) -> int:
    """Make 0003's self-driving legs visible to the itinerary cost parser."""
    fixed = 0
    for turn in instance.get("turns") or []:
        payload = ((turn.get("gold_action") or {}).get("action_payload") or {}).get("plan") or {}
        for day in payload.get("itinerary") or []:
            text = str(day.get("transportation") or "")
            match = re.search(r"cost: \$(\d+)$", text)
            if match:
                day["transportation"] = f"{text[:match.start()]}cost: ${match.group(1)} per vehicle = ${match.group(1)}"
                fixed += 1
    return fixed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()

    instances = json.loads(args.path.read_text(encoding="utf-8"))
    by_id = {str(item["instance_id"]): item for item in instances}

    for spec in plan.SPECS:
        instance = by_id.get(spec["instance_id"])
        if instance is None:
            raise KeyError(f"{spec['instance_id']} not in {args.path}")
        build_instance(instance, spec)
        print(f"  {spec['instance_id']}  {spec['category']}  "
              f"{len(spec['turns'])} turns rewritten")

    if "travelplanner_test_0003" in by_id:
        fixed = fix_0003_drive_cost(by_id["travelplanner_test_0003"])
        print(f"  travelplanner_test_0003  drive legs made cost-parseable: {fixed}")

    out = args.output or args.path
    if out == args.path and not args.no_backup:
        stamp = datetime.now().strftime("%m%d_%H%M")
        backup = args.path.with_name(f".{args.path.stem}.bak_{stamp}.json")
        shutil.copy2(args.path, backup)
        print(f"  backup -> {backup}")
    out.write_text(json.dumps(instances, ensure_ascii=False, indent=2, default=str),
                   encoding="utf-8")
    print(f"written -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
