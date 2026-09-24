#!/usr/bin/env python3
"""Mine TravelPlanner judge verdicts that are wrong or error-prone.

Reads scored rows from ``judge_travelplanner_v4_trajectories.py`` and writes a
candidate list for few-shot curation. Signals:

* ``db_mismatch``   the judge disagrees with a verdict computed from the
                    reference database (room type, ratings, house rule)
* ``unstable``      two judge runs with different evidence disagree
* ``unknown``       the judge answered unknown
* ``layer_mixing``  a violated verdict whose note blames the intent prediction
* ``hotel``         a booked stay breaks minimum nights or capacity
* ``must_violated`` a Must constraint is violated (for disclosure labelling)

Budget is excluded: it is priced by code, not judged.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from eval.travelplanner_checks import (
    MEALS,
    _is_empty_slot,
    _match_entry,
    _text,
    hotel_validity,
)
from report_travelplanner_v4_eval import gold_tiers, load_world

LAYER_MIXING = re.compile(r"\b(prediction|predicted|omits?)\b", re.IGNORECASE)


def _entries(reference: Any, kind: str) -> List[Dict[str, Any]]:
    out = []
    for value in (reference or {}).values() if isinstance(reference, dict) else []:
        for entry in value if isinstance(value, list) else []:
            if not isinstance(entry, dict):
                continue
            if kind == "restaurant" and "Average Cost" in entry:
                out.append(entry)
            if kind == "stay" and entry.get("NAME") and "price" in entry:
                out.append(entry)
    return out


def _per_city(value: Any, city: Optional[str]) -> Any:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in str(city or "").lower():
                return item
        return None
    return value


def db_verdict(field: str, value: Any, itinerary: Any, reference: Any) -> Optional[str]:
    """satisfied / violated from the database, or None when not checkable."""
    days = [d for d in itinerary if isinstance(d, dict)] if isinstance(itinerary, list) else []
    stays = _entries(reference, "stay")
    restaurants = _entries(reference, "restaurant")
    nights = [
        _match_entry(_text(d.get("accommodation")), stays, "NAME")
        for d in days
        if not _is_empty_slot(_text(d.get("accommodation")))
    ]
    meals = [
        _match_entry(_text(d.get(m)), restaurants, "Name")
        for d in days
        for m in MEALS
        if not _is_empty_slot(_text(d.get(m)))
    ]
    name = field.lower()
    if name in {"accommodation_rating", "room_type", "house_rule"}:
        if not nights or any(n is None for n in nights):
            return None
        bad = False
        for record in nights:
            want = _per_city(value, record.get("city"))
            if want is None:
                continue
            if name == "accommodation_rating":
                try:
                    bad |= float(record.get("review rate number") or 0) < float(want)
                except (TypeError, ValueError):
                    return None
            elif name == "room_type":
                w, got = str(want).lower(), str(record.get("room type")).lower()
                if "not shared" in w:
                    bad |= "shared" in got
                elif "entire" in w or "apartment" in w:
                    bad |= "entire" not in got
                elif "private" in w:
                    bad |= "private" not in got
                elif "shared" in w:
                    bad |= "shared" not in got
                else:
                    return None
            else:
                rules = str(record.get("house_rules")).lower()
                w = str(want).strip().lower()
                if w.startswith("no "):
                    bad |= w not in rules
                elif w in {"parties", "pets", "smoking", "visitors", "children under 10"}:
                    bad |= f"no {w}" in rules
                else:
                    return None
        return "violated" if bad else "satisfied"
    if name == "restaurant_rating":
        priced = [m for m in meals if m is not None]
        try:
            threshold = float(value)
        except (TypeError, ValueError):
            return None
        if not priced:
            return None
        return "violated" if any(float(m.get("Aggregate Rating") or 0) < threshold for m in priced) else "satisfied"
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report_dir", type=Path)
    parser.add_argument("--primary", default="scored_rows_pool.json")
    parser.add_argument("--other", default="scored_rows_grounded.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    primary = json.loads((args.report_dir / args.primary).read_text(encoding="utf-8"))
    other_rows = json.loads((args.report_dir / args.other).read_text(encoding="utf-8"))["rows"]
    world = load_world(primary["metadata"])
    key = lambda r: (r["model"], r["shard"], r["instance_id"], r["turn_id"])
    other = {key(r): {c["field"]: c for c in r["scores"]["per_constraint"]} for r in other_rows}

    cases: List[Dict[str, Any]] = []
    for row in primary["rows"]:
        state = world[(row["shard"], row["instance_id"])]
        reference = state.get("reference_information")
        itinerary = (row.get("action") or {}).get("itinerary")
        tiers = gold_tiers(row["gold_intention"])
        people = (row["gold_intention"].get("constraints") or {}).get("people_number") or (
            state.get("travelplanner_query_data") or {}
        ).get("people_number")
        base = {
            "model": row["model"],
            "shard": row["shard"],
            "instance_id": row["instance_id"],
            "turn_id": row["turn_id"],
        }
        hotel = hotel_validity(itinerary, reference, people)
        if not hotel["valid"]:
            cases.append({**base, "signal": "hotel", "detail": hotel})
        for item in row["scores"]["per_constraint"]:
            field = item["field"]
            if field == "budget":
                continue
            status = item["action_status"]
            signals = []
            truth = db_verdict(field, item["gold_value"], itinerary, reference)
            if truth and truth != status:
                signals.append("db_mismatch")
            second = other[key(row)].get(field)
            if second and second["action_status"] != status:
                signals.append("unstable")
            if status == "unknown":
                signals.append("unknown")
            if status == "violated" and LAYER_MIXING.search(item.get("note") or ""):
                signals.append("layer_mixing")
            if status == "violated" and tiers.get(field) == "must_have":
                signals.append("must_violated")
            for signal in signals:
                cases.append(
                    {
                        **base,
                        "signal": signal,
                        "field": field,
                        "tier": tiers.get(field, "entity"),
                        "gold_value": item["gold_value"],
                        "judge_status": status,
                        "judge_note": item.get("note"),
                        "other_status": second["action_status"] if second else None,
                        "other_note": second.get("note") if second else None,
                        "db_verdict": truth,
                    }
                )

    counts = Counter((c["signal"], c.get("field", "-")) for c in cases)
    output = args.output or args.report_dir / "judge_error_candidates.json"
    output.write_text(
        json.dumps({"counts": {f"{s}|{f}": n for (s, f), n in counts.most_common()}, "cases": cases}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    for (signal, field), n in counts.most_common(40):
        print(f"{n:5d}  {signal:14s} {field}")
    print(f"wrote {len(cases)} candidates to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
