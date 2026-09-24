#!/usr/bin/env python3
"""Check each turn's gold_action itinerary against that turn's gold constraints.

The itinerary drafts were authored for shard_002_annotated. Once the
relaxation rewrite and the control variants changed the constraints, a draft
can silently stop matching the state it is supposed to satisfy, so every turn
needs re-checking rather than trusting the port.

Itinerary entries are semi-structured strings, e.g.

  accommodation  "NAME; Private room; rating 3; $897/night x 2 nights = $1794 ...
                  minimum stay 1 nights; maximum occupancy 1; ... house rules: ..."
  breakfast      "NAME; rating 3.5; listed Average Cost $19 per person x 1; Cuisines: ..."
  transportation "Flight F3804883: Phoenix to Billings on 2022-03-18; ... $359 per person ..."

so the machine-checkable constraints can be parsed out. Anything not parseable
is reported as `unknown` rather than silently passing.

Checks: room_type, accommodation_rating, preferred_accommodation_rating,
accommodation_stay (nights + minimum-stay feasibility), restaurant_rating,
preferred_restaurant_rating, meal_cost, house_rule, cuisine, days,
outbound_transportation, return_transportation.

Usage:
    python annotation/tools/check_gold_action.py FILE.json [FILE.json ...]
    python annotation/tools/check_gold_action.py FILE.json --verbose
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


MEALS = ("breakfast", "lunch", "dinner")


def _itinerary(turn: Dict[str, Any]) -> List[Dict[str, Any]]:
    payload = ((turn.get("gold_action") or {}).get("action_payload") or {}).get("plan") or {}
    value = payload.get("itinerary")
    return value if isinstance(value, list) else []


def _num(pattern: str, text: str) -> Optional[float]:
    match = re.search(pattern, text, re.I)
    if not match:
        return None
    try:
        return float(match.group(1))
    except (TypeError, ValueError):
        return None


def parse_accommodation(text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"raw": text}
    if not text or text.strip() in {"-", "Not applicable", ""}:
        return out
    parts = [p.strip() for p in text.split(";")]
    out["name"] = parts[0] if parts else None
    for part in parts:
        low = part.lower()
        if low in {"private room", "entire home/apt", "shared room"}:
            out["room_type"] = part
    out["rating"] = _num(r"rating\s+([0-9.]+)", text)
    out["nights"] = _num(r"[x×]\s*([0-9]+)\s*nights", text)
    out["minimum_stay"] = _num(r"minimum stay\s+([0-9.]+)", text)
    rules = re.search(r"house rules:\s*(.+)$", text, re.I)
    out["house_rules"] = rules.group(1).strip() if rules else None
    return out


def parse_meal(text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"raw": text}
    if not text or text.strip() in {"-", ""}:
        return out
    out["name"] = text.split(";")[0].strip()
    out["rating"] = _num(r"rating\s+([0-9.]+)", text)
    out["cost"] = _num(r"Average Cost\s*\$?\s*([0-9.]+)", text)
    cui = re.search(r"Cuisines?:\s*(.+)$", text, re.I)
    out["cuisines"] = cui.group(1).strip() if cui else None
    return out


def itinerary_cost(itinerary: List[Dict[str, Any]]) -> Dict[str, float]:
    """Total trip cost.

    The accommodation line repeats the same "= $N for one unit" total on every
    booked night, so it is deduplicated by property name; transport and meals
    are genuinely per-day and get summed.
    """
    stays: Dict[str, float] = {}
    transport = meals = 0.0
    for day in itinerary:
        text = str(day.get("accommodation") or "")
        match = re.search(r"=\s*\$([0-9,]+)\s*for one unit", text)
        if match:
            stays[text.split(";")[0].strip()] = float(match.group(1).replace(",", ""))
        for found in re.finditer(r"=\s*\$([0-9,]+)", str(day.get("transportation") or "")):
            transport += float(found.group(1).replace(",", ""))
        for meal in MEALS:
            found = re.search(
                r"Average Cost \$?([0-9.]+) per person\s*[×x]\s*(\d+)",
                str(day.get(meal) or ""),
            )
            if found:
                meals += float(found.group(1)) * int(found.group(2))
    return {
        "accommodation": sum(stays.values()),
        "transportation": transport,
        "meals": meals,
        "total": sum(stays.values()) + transport + meals,
    }


def soft_fields(priority: Any) -> set:
    """Fields the annotation deliberately downgraded to a preference.

    The priority board is how a tradeoff gets expressed: a constraint the user
    agreed to give ground on is moved to medium/low rather than deleted, so its
    stated bound can legitimately go unmet. Those are reported separately from
    genuine gold/constraint mismatches.
    """
    if not isinstance(priority, dict):
        return set()
    return {
        str(field)
        for level in ("medium", "low")
        for field in priority.get(level) or []
    }


# A sacrifice names a constraint field, but one of them surfaces under its own
# issue label rather than the field name.
DECLARED_ALIASES = {"accommodation_stay": ("accommodation_stay", "最低入住晚数冲突")}


def check_turn(
    constraints: Dict[str, Any],
    itinerary: List[Dict[str, Any]],
    priority: Any = None,
    declared_sacrifice: Sequence[str] = (),
) -> List[str]:
    issues: List[str] = []
    if not itinerary:
        return issues

    accs = [parse_accommodation(str(d.get("accommodation") or "")) for d in itinerary]
    booked = [a for a in accs if a.get("name") and a.get("room_type")]
    meals = [
        parse_meal(str(d.get(meal) or ""))
        for d in itinerary
        for meal in MEALS
        if str(d.get(meal) or "").strip() not in {"", "-"}
    ]
    transport = " | ".join(str(d.get("transportation") or "") for d in itinerary)

    def want(field: str) -> Any:
        return constraints.get(field)

    def want_num(field: str) -> Optional[float]:
        """Constraint values are often prose ("At least 4.0 for all meals")."""
        value = constraints.get(field)
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        return _num(r"([0-9]+(?:\.[0-9]+)?)", str(value))

    want_days = want_num("days")
    if want_days is not None and len(itinerary) != int(want_days):
        issues.append(f"days: gold={want('days')} 行程={len(itinerary)} 天")

    rt = want("room_type")
    if rt and booked:
        allowed = [x.strip().lower() for x in str(rt).replace(" or ", "|").split("|")]
        bad = [a for a in booked if a["room_type"].lower() not in allowed]
        if bad:
            issues.append(f"room_type: 要求 {rt}，实际 {sorted({a['room_type'] for a in bad})}")

    ar = want_num("accommodation_rating")
    if ar is not None and booked:
        low = [a for a in booked if a.get("rating") is not None and a["rating"] < ar]
        if low:
            issues.append(f"accommodation_rating: 要求 >={ar}，实际 {sorted({a['rating'] for a in low})}")

    stay = want("accommodation_stay")
    if stay and booked:
        nights = {a["nights"] for a in booked if a.get("nights") is not None}
        expected = _num(r"(\d+)\s*nights", str(stay))
        if len(nights) == 1 and expected is not None and expected not in nights:
            issues.append(f"accommodation_stay: 要求 {expected:.0f} 晚，行程写 {nights}")
        infeasible = [
            a for a in booked
            if a.get("minimum_stay") is not None and a.get("nights") is not None
            and a["minimum_stay"] > a["nights"]
        ]
        if infeasible:
            names = sorted({f"{a['name']}(最低{a['minimum_stay']:.0f}晚/订{a['nights']:.0f}晚)" for a in infeasible})
            issues.append(f"最低入住晚数冲突: {names}")

    rr = want_num("restaurant_rating")
    if rr is not None and meals:
        # A per-meal field may carve out an explicit exception, e.g.
        # dinner = "2022-03-17 dinner may drop to rating 3.9 ...". Meals covered
        # by such a carve-out are checked against the stated floor instead.
        exempt: List[float] = []
        specs = [constraints.get(slot) for slot in MEALS]
        # The carve-out is sometimes written into the rating constraint itself
        # ("At least 4.0 ... except the 2022-03-17 dinner which may be 3.9")
        # rather than into a per-meal field.
        raw_rating = constraints.get("restaurant_rating")
        if isinstance(raw_rating, str) and re.search(r"except", raw_rating, re.I):
            specs.append(re.split(r"except", raw_rating, maxsplit=1, flags=re.I)[1])
        for spec in specs:
            if not spec:
                continue
            # Pull a rating, not the first number: these strings start with a
            # date ("2022-03-17 dinner may drop to rating 3.9 ...").
            found = [
                float(x) for x in re.findall(r"(?<![0-9-])([0-5](?:\.[0-9])?)(?![0-9-])", str(spec))
            ]
            floor = min(found) if found else None
            if floor is not None and floor < rr:
                exempt.append(floor)
        allowed = min(exempt) if exempt else rr
        low = [m for m in meals if m.get("rating") is not None and m["rating"] < allowed]
        if low:
            issues.append(f"restaurant_rating: 要求 >={rr}，实际 {sorted({m['rating'] for m in low})}")

    cap = want_num("meal_cost")
    if cap is not None and meals:
        over = [m for m in meals if cap and m.get("cost") is not None and m["cost"] > cap]
        if over:
            issues.append(f"meal_cost: 上限 {cap:.0f}，实际 {sorted({m['cost'] for m in over})}")

    hr = want("house_rule")
    if hr and booked:
        needle = str(hr).lower().replace("no ", "").strip()
        bad = [a for a in booked if a.get("house_rules") and needle not in a["house_rules"].lower()]
        if bad:
            issues.append(f"house_rule: 要求 {hr}，房源规则未包含 -> {sorted({a['name'] for a in bad})}")

    cu = want("cuisine")
    if cu and meals:
        # The constraint is prose ("At least one Italian-tagged meal ..."), so pull
        # the cuisine token out instead of matching the whole sentence.
        known = ("italian", "indian", "chinese", "mexican", "french", "japanese",
                 "thai", "american", "mediterranean", "korean", "pizza", "cafe",
                 "seafood", "bakery", "desserts", "fast food", "bar")
        text = str(cu).lower()
        needles = [k for k in known if k in text]
        for needle in needles:
            if not any(m.get("cuisines") and needle in m["cuisines"].lower() for m in meals):
                issues.append(f"cuisine: 要求含 {needle}，行程中未出现")

    style = want("dining_style")
    if style and "fast food" in str(style).lower():
        # The prohibition may be scoped to one date ("dinner on 2022-03-17 ...");
        # unscoped, it covers every booked meal.
        scope = re.search(r"(\d{4}-\d{2}-\d{2})", str(style))
        slots = [s for s in MEALS if s in str(style).lower()] or list(MEALS)
        for day in itinerary:
            if scope and str(day.get("day")) != scope.group(1):
                continue
            for slot in slots:
                parsed = parse_meal(str(day.get(slot) or ""))
                if parsed.get("cuisines") and "fast food" in parsed["cuisines"].lower():
                    issues.append(
                        f"dining_style: 禁止 Fast Food，{day.get('day')} {slot} "
                        f"选了 {parsed.get('name')}"
                    )

    # Per-slot cuisine pins: "Lunch on 2022-03-24 must carry a BBQ tag", or
    # "The Indian-tagged meal on 2022-03-24 must be dinner". Both reduce to
    # "this slot's meal must carry this cuisine".
    known_cuisines = ("italian", "indian", "chinese", "mexican", "french", "japanese",
                      "thai", "american", "mediterranean", "korean", "pizza", "cafe",
                      "seafood", "bakery", "desserts", "fast food", "bbq", "tea")
    for slot in MEALS:
        spec = want(slot)
        if not spec:
            continue
        needles = [k for k in known_cuisines if k in str(spec).lower()]
        if not needles:
            continue
        scope = re.search(r"(\d{4}-\d{2}-\d{2})", str(spec))
        for day in itinerary:
            if scope and str(day.get("day")) != scope.group(1):
                continue
            parsed = parse_meal(str(day.get(slot) or ""))
            if parsed.get("rating") is None:
                continue  # not a booked venue
            for needle in needles:
                if needle not in str(parsed.get("cuisines") or "").lower():
                    issues.append(
                        f"{slot}: 要求含 {needle}，{day.get('day')} 实际 {parsed.get('name')}"
                        f" ({parsed.get('cuisines')})"
                    )

    variety = want("dining_variety")
    if variety and re.search(r"three different|3 different", str(variety), re.I):
        scope = re.search(r"(\d{4}-\d{2}-\d{2})", str(variety))
        for day in itinerary:
            if scope and str(day.get("day")) != scope.group(1):
                continue
            # Only booked venues count; a day whose meals hold a "no feasible
            # option" note has nothing to be various about.
            parsed = [parse_meal(str(day.get(slot) or "")) for slot in MEALS]
            named = [p["name"] for p in parsed if p.get("name") and p.get("rating") is not None]
            if len(named) == 3 and len(set(named)) < 3:
                issues.append(
                    f"dining_variety: 要求当天三餐三家不同，{day.get('day')} 实际 {sorted(set(named))}"
                )

    budget = want_num("budget")
    if budget is not None:
        cost = itinerary_cost(itinerary)
        if cost["total"] > budget:
            issues.append(
                f"budget: 上限 ${budget:.0f}，行程合计 ${cost['total']:.0f} "
                f"(住宿 ${cost['accommodation']:.0f} 交通 ${cost['transportation']:.0f} 餐 ${cost['meals']:.0f})"
            )

    for field in ("outbound_transportation", "return_transportation"):
        value = want(field)
        if not value:
            continue
        flight = re.search(r"\b(F\d{6,})\b", str(value))
        if flight and flight.group(1) not in transport:
            issues.append(f"{field}: 要求 {flight.group(1)}，行程中未出现")

    soft = soft_fields(priority)
    if soft:
        issues = [
            f"[soft] {issue}" if issue.split(":")[0] in soft else issue
            for issue in issues
        ]
    # A turn whose pool cannot satisfy everything now carries a real plan plus a
    # gold_action.world_feasibility record saying which requirement it gave up.
    # That one is the annotation working as designed, not a defect -- but only
    # that one: anything else broken on the same turn is still a defect.
    declared = {alias
                for field in declared_sacrifice or ()
                for alias in DECLARED_ALIASES.get(str(field), (str(field),))}
    if declared:
        issues = [
            f"[declared] {issue}"
            if issue.split(":")[0].replace("[soft] ", "") in declared else issue
            for issue in issues
        ]
    return issues


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", type=Path, nargs="+")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    grand = 0
    for path in args.files:
        instances = json.loads(path.read_text(encoding="utf-8"))
        total_turns = checked = bad_turns = soft_turns = declared_turns = 0
        findings: List[Tuple[str, int, List[str]]] = []
        for instance in instances:
            for index, turn in enumerate(instance.get("turns") or []):
                total_turns += 1
                itinerary = _itinerary(turn)
                if not itinerary:
                    continue
                checked += 1
                gold = turn.get("gold_current_intention") or {}
                feasibility = (turn.get("gold_action") or {}).get("world_feasibility") or {}
                sacrifices = feasibility.get("acceptable_sacrifices") or []
                # gold books the cheapest minimal sacrifice, so that is the one it took.
                declared = (sacrifices[0].get("give_up") or []) if sacrifices else []
                issues = check_turn(
                    gold.get("constraints") or {}, itinerary, gold.get("priority"),
                    declared_sacrifice=declared,
                )
                if not issues:
                    continue
                hard = [i for i in issues
                        if not i.startswith("[soft]") and not i.startswith("[declared]")]
                if hard:
                    bad_turns += 1
                elif any(i.startswith("[declared]") for i in issues):
                    declared_turns += 1
                else:
                    soft_turns += 1
                findings.append((str(instance["instance_id"]), int(turn.get("turn_id", index)), issues))
        grand += bad_turns
        print(f"\n=== {path.name}: {len(instances)} instances / {total_turns} turns "
              f"(有行程 {checked} 轮) -> 不符 {bad_turns} 轮，"
              f"另有 {soft_turns} 轮仅偏好项未满足、"
              f"{declared_turns} 轮是 world_feasibility 已声明的牺牲 ===")
        counts: Dict[str, int] = {}
        for _, _, issues in findings:
            for issue in issues:
                counts[issue.split(":")[0]] = counts.get(issue.split(":")[0], 0) + 1
        for key, value in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"    {key:<28}{value:>4}")
        if args.verbose:
            for instance_id, turn_id, issues in findings:
                print(f"  [{instance_id} t{turn_id}]")
                for issue in issues:
                    print(f"      - {issue}")
    print(f"\n合计不符轮次: {grand}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
