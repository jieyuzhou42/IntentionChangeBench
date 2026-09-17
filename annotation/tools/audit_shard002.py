#!/usr/bin/env python3
"""Per-turn audit: budget usage, and whether a turn claiming infeasibility really is.

check_gold_action.py verifies that the written itinerary matches the written
constraints. It cannot tell whether a turn that reports "no feasible option" is
telling the truth, so this brute-forces the turn's own candidate pool: every
stay that clears the room type, rating, minimum-stay and occupancy rules, paired
with the cheapest compliant restaurant, against the turn's budget.

Usage:
    python annotation/tools/audit_shard002.py FILE.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

_spec = importlib.util.spec_from_file_location(
    "check_gold_action", Path(__file__).with_name("check_gold_action.py")
)
checker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(checker)

MEALS = ("breakfast", "lunch", "dinner")


def pool(turn: Dict[str, Any], key: str) -> List[Dict[str, Any]]:
    out = []
    for page in ((turn.get("env_feedback") or {}).get("search_results") or {}).get(key) or []:
        out.extend(x for x in (page.get("items") or []) if isinstance(x, dict))
    return out


def num(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)", str(value))
    return float(m.group(1)) if m else None


def best_plan(turn, constraints, nights, transport_cost, people, soft=frozenset()):
    """Cheapest compliant (stay, meal) pair; None when the pool has none.

    Fields the turn moved to medium/low are preferences the user agreed to give
    ground on, so they are not treated as filters -- otherwise every tradeoff
    turn reads as infeasible."""
    def hard(field):
        return None if field in soft else constraints.get(field)

    want_room = hard("room_type")
    want_rating = num(hard("accommodation_rating"))
    want_meal_rating = num(hard("restaurant_rating"))
    meal_cap = num(hard("meal_cost"))
    # A date-scoped ban ("2022-03-17 dinner must not ...") only rules out one
    # meal, so it is not a filter on the whole pool.
    style = str(hard("dining_style") or "")
    no_ff = "fast food" in style.lower() and not re.search(r"\d{4}-\d{2}-\d{2}", style)
    variety = bool(re.search(r"three different", str(hard("dining_variety") or ""), re.I))

    stays = []
    for a in pool(turn, "accommodations"):
        if want_room and str(a.get("room type", "")).lower() != str(want_room).lower():
            continue
        if want_rating is not None and float(a.get("review rate number") or 0) < want_rating:
            continue
        if float(a.get("minimum nights") or 0) > nights:
            continue
        if float(a.get("maximum occupancy") or 0) < people:
            continue
        stays.append(a)

    meals = []
    for r in pool(turn, "restaurants"):
        if want_meal_rating is not None and float(r.get("Aggregate Rating") or 0) < want_meal_rating:
            continue
        if meal_cap is not None and float(r.get("Average Cost") or 0) > meal_cap:
            continue
        if no_ff and "fast food" in str(r.get("Cuisines") or "").lower():
            continue
        meals.append(r)
    meals.sort(key=lambda r: float(r.get("Average Cost") or 0))
    meals = {r["Name"]: r for r in meals}
    meals = sorted(meals.values(), key=lambda r: float(r.get("Average Cost") or 0))

    if not stays or not meals or (variety and len(meals) < 3):
        return None, ("no compliant stay" if not stays else
                      "no compliant restaurant" if not meals else
                      "fewer than three compliant restaurants")
    picks = meals[:3] if variety else [meals[0]] * 3
    meal_cost = sum(float(r.get("Average Cost") or 0) for r in picks) * people
    stays.sort(key=lambda a: float(a.get("price") or 0))
    total = transport_cost + float(stays[0]["price"]) * nights + meal_cost
    return total, f"{stays[0]['NAME'][:26]} + {picks[0]['Name'][:18]}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()

    for instance in json.loads(args.path.read_text(encoding="utf-8")):
        print("=" * 104)
        print(instance["instance_id"])
        for index, turn in enumerate(instance.get("turns") or []):
            itinerary = checker._itinerary(turn)
            if not itinerary:
                continue
            constraints = (turn.get("gold_current_intention") or {}).get("constraints") or {}
            cost = checker.itinerary_cost(itinerary)
            budget = num(constraints.get("budget")) or 0
            people = int(num(constraints.get("people_number")) or 1)
            nights = int(num(re.search(r"(\d+)\s*nights", str(constraints.get("accommodation_stay") or "2 nights")).group(1)) if re.search(r"(\d+)\s*nights", str(constraints.get("accommodation_stay") or "")) else 2)
            declared = any("No feasible option" in str(d.get(f) or "")
                           for d in itinerary for f in ("accommodation",) + MEALS)
            soft = checker.soft_fields((turn.get("gold_current_intention") or {}).get("priority"))
            best, note = best_plan(turn, constraints, nights, cost["transportation"], people, soft)
            if best is None:
                truth = f"INFEASIBLE ({note})"
            elif best > budget:
                truth = f"INFEASIBLE (cheapest compliant ${best:.0f} > ${budget:.0f})"
            else:
                truth = f"feasible (cheapest ${best:.0f}; {note})"
            flag = "  " if declared == truth.startswith("INFEASIBLE") else "!!"
            pct = (cost["total"] / budget * 100) if budget else 0
            print(f" {flag} t{index}  cap ${budget:<6.0f} plan ${cost['total']:<6.0f} ({pct:3.0f}%) "
                  f"[stay ${cost['accommodation']:.0f} trans ${cost['transportation']:.0f} "
                  f"meals ${cost['meals']:.0f}]  gold_says={'infeasible' if declared else 'booked'}  {truth}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
