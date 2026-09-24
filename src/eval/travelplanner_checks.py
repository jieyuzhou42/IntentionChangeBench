"""Deterministic TravelPlanner plan checks against the instance reference database.

Used by the v4 trajectory evaluation: budget pricing, hotel validity (minimum
nights, capacity) and meal-slot coverage against the gold reference plan. These
replace or gate LLM-judge verdicts where the judge proved unreliable.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, List, Optional


FLIGHT_NUMBER = re.compile(r"\bF\d{5,}\b", re.IGNORECASE)
GROUND_RECORD = re.compile(r"(self-driving|taxi), from (.+?) to (.+?), duration.*?cost: (\d+(?:\.\d+)?)", re.IGNORECASE)
MEALS = ("breakfast", "lunch", "dinner")


def _text(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False).lower()
    return str(value or "").strip().lower()


def _compact(text: str) -> str:
    return re.sub(r"[\W_]", "", text.lower())


def _is_empty_slot(text: str) -> bool:
    return not text or text.startswith("-")


def _match_entry(text: str, entries: List[Dict[str, Any]], key: str) -> Optional[Dict[str, Any]]:
    """Database entry named in ``text``.

    Names are compared as alphanumeric-only strings because database names carry
    stray spaces and newlines. A full name inside the text wins; otherwise a
    leading fragment of the text (>= 10 chars, before any parenthetical or
    separator) may identify a single entry whose name it prefixes, since agents
    often shorten long listing names.
    """
    compact_text = _compact(text)
    named = [(e, _compact(str(e.get(key) or ""))) for e in entries]
    full = [(e, n) for e, n in named if n and n in compact_text]
    if full:
        return max(full, key=lambda pair: len(pair[1]))[0]
    core = _compact(re.split(r"\(|;|—|–|,|\$|\|", text)[0])
    if len(core) >= 10:
        prefixed = [e for e, n in named if n.startswith(core)]
        if len({str(e.get(key)) for e in prefixed}) == 1:
            return prefixed[0]
    return None


def _match_all(text: str, entries: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    """Every entry named in ``text`` (a night may split a group across listings).

    Falls back to ``_match_entry`` when no full name appears. Names contained in
    a longer matched name are dropped, so "Cozy Room" does not double-count
    "Cozy Room in Harlem".
    """
    compact_text = _compact(text)
    full = [e for e in entries if _compact(str(e.get(key) or "")) and _compact(str(e.get(key))) in compact_text]
    if not full:
        single = _match_entry(text, entries, key)
        return [single] if single else []
    names = [_compact(str(e[key])) for e in full]
    kept, seen = [], set()
    for entry, name in zip(full, names):
        if name in seen or any(name != other and name in other for other in names):
            continue
        seen.add(name)
        kept.append(entry)
    return kept


def trip_cost(itinerary: Any, reference: Any, people: int) -> Dict[str, Any]:
    """TravelPlanner total cost of an itinerary, priced from the reference database.

    Follows the official TravelPlanner cost rule: flights and meals per person,
    self-driving per 5 people, taxi per 4 people, accommodation per night per
    ceil(people / maximum occupancy) rooms; attractions are free. Slots that do
    not resolve to a database record contribute 0 and are listed as unpriced.
    """
    flights: Dict[str, Dict[str, Any]] = {}
    restaurants: List[Dict[str, Any]] = []
    stays: List[Dict[str, Any]] = []
    ground: List[tuple] = []
    for value in (reference or {}).values() if isinstance(reference, dict) else []:
        if isinstance(value, list):
            for entry in value:
                if not isinstance(entry, dict):
                    continue
                if entry.get("Flight Number"):
                    flights[str(entry["Flight Number"]).upper()] = entry
                elif "Average Cost" in entry:
                    restaurants.append(entry)
                elif entry.get("NAME") and "price" in entry:
                    stays.append(entry)
        elif isinstance(value, str):
            match = GROUND_RECORD.search(value)
            if match:
                mode, origin, dest, cost = match.groups()
                ground.append((mode.lower(), origin.lower(), dest.lower(), float(cost)))

    people = max(int(people or 1), 1)
    total = 0.0
    unpriced: List[str] = []
    charged_legs = set()
    for day in itinerary if isinstance(itinerary, list) else []:
        if not isinstance(day, dict):
            continue
        transport = _text(day.get("transportation"))
        if not _is_empty_slot(transport):
            numbers = FLIGHT_NUMBER.findall(transport)
            if numbers:
                for number in numbers:
                    flight = flights.get(number.upper())
                    if flight:
                        total += float(flight.get("Price") or 0) * people
                    else:
                        unpriced.append(f"flight {number}")
            else:
                mode = "self-driving" if re.search(r"self[- ]?driv|drive", transport) else "taxi" if "taxi" in transport else None
                if mode:
                    route_text = transport + " " + _text(day.get("current_city"))
                    legs = [g for g in ground if g[0] == mode and g[1] in route_text and g[2] in route_text]
                    # Prefer the leg whose cities appear in the stated order.
                    legs.sort(key=lambda g: route_text.find(g[1]) > route_text.find(g[2]))
                    if legs:
                        # A long drive split over several days is one leg, charged once.
                        if legs[0] not in charged_legs:
                            charged_legs.add(legs[0])
                            per_vehicle = 5 if mode == "self-driving" else 4
                            total += legs[0][3] * math.ceil(people / per_vehicle)
                    else:
                        unpriced.append(f"{mode}: {transport[:60]}")
        for meal in MEALS:
            text = _text(day.get(meal))
            if _is_empty_slot(text):
                continue
            restaurant = _match_entry(text, restaurants, "Name")
            if restaurant:
                total += float(restaurant.get("Average Cost") or 0) * people
            else:
                unpriced.append(f"{meal}: {text[:60]}")
        text = _text(day.get("accommodation"))
        if not _is_empty_slot(text):
            booked = _match_all(text, stays, "NAME")
            if len(booked) > 1:
                # The plan splits the group across listings: one unit of each.
                total += sum(float(stay.get("price") or 0) for stay in booked)
            elif booked:
                occupancy = max(int(float(booked[0].get("maximum occupancy") or 1)), 1)
                total += float(booked[0].get("price") or 0) * math.ceil(people / occupancy)
            else:
                unpriced.append(f"accommodation: {text[:60]}")
    return {"total": total, "unpriced": unpriced}


def _reference_entries(reference: Any) -> Dict[str, List[Dict[str, Any]]]:
    restaurants: List[Dict[str, Any]] = []
    stays: List[Dict[str, Any]] = []
    for value in (reference or {}).values() if isinstance(reference, dict) else []:
        for entry in value if isinstance(value, list) else []:
            if isinstance(entry, dict) and "Average Cost" in entry:
                restaurants.append(entry)
            elif isinstance(entry, dict) and entry.get("NAME") and "price" in entry:
                stays.append(entry)
    return {"restaurants": restaurants, "stays": stays}


def stay_nights(itinerary: Any, reference: Any) -> List[Dict[str, Any]]:
    """One entry per itinerary day with accommodation: {"day", "text", "records"}.

    Every day whose accommodation is not empty counts as one night; ``records``
    lists every database listing named for that night (possibly several).
    """
    stays = _reference_entries(reference)["stays"]
    nights = []
    for index, day in enumerate(itinerary if isinstance(itinerary, list) else []):
        if not isinstance(day, dict):
            continue
        text = _text(day.get("accommodation"))
        if not _is_empty_slot(text):
            nights.append({"day": index, "text": text[:80], "records": _match_all(text, stays, "NAME")})
    return nights


def hotel_validity(itinerary: Any, reference: Any, people: Any) -> Dict[str, Any]:
    """Minimum-night and capacity check for every booked listing.

    Each listing's run of consecutive nights must reach its ``minimum nights``.
    Each night's listings together must host every traveler (sum of
    ``maximum occupancy`` >= people); a single listing is one unit.
    """
    people = max(int(people or 1), 1)
    nights = stay_nights(itinerary, reference)
    minimum: List[str] = []
    capacity: List[str] = []
    runs: Dict[str, List[int]] = {}
    for night in nights:
        for record in night["records"]:
            runs.setdefault(record["NAME"], []).append(night["day"])
        rooms = sum(float(r.get("maximum occupancy") or 0) for r in night["records"])
        if night["records"] and rooms < people:
            names = " + ".join(r["NAME"] for r in night["records"])
            capacity.append(f"day {night['day'] + 1} {names}: {people} travelers, maximum occupancy {int(rooms)}")
    for name, days in runs.items():
        record = next(r for n in nights for r in n["records"] if r["NAME"] == name)
        need = int(float(record.get("minimum nights") or 1))
        streaks: List[List[int]] = []
        for day in days:
            if streaks and day == streaks[-1][-1] + 1:
                streaks[-1].append(day)
            else:
                streaks.append([day])
        for streak in streaks:
            if len(streak) < need:
                minimum.append(f"{name}: {len(streak)} consecutive night(s) booked, minimum {need}")
    return {"valid": not minimum and not capacity, "minimum_nights": minimum, "capacity": sorted(set(capacity))}


def meal_coverage_gaps(itinerary: Any, gold_itinerary: Any, reference: Any) -> List[str]:
    """Meal slots the gold plan fills with a priced restaurant but the agent leaves out.

    Days are aligned by position. A slot the gold plan leaves empty may be
    omitted; a slot the gold plan prices must be a priced restaurant in the
    agent plan too, otherwise the agent's total no longer covers that meal.
    """
    restaurants = _reference_entries(reference)["restaurants"]
    agent_days = [d for d in itinerary if isinstance(d, dict)] if isinstance(itinerary, list) else []
    gold_days = [d for d in gold_itinerary if isinstance(d, dict)] if isinstance(gold_itinerary, list) else []
    gaps: List[str] = []
    for index, gold_day in enumerate(gold_days):
        agent_day = agent_days[index] if index < len(agent_days) else {}
        for meal in MEALS:
            gold_text = _text(gold_day.get(meal))
            if _is_empty_slot(gold_text) or not _match_entry(gold_text, restaurants, "Name"):
                continue
            agent_text = _text(agent_day.get(meal))
            if _is_empty_slot(agent_text) or not _match_entry(agent_text, restaurants, "Name"):
                gaps.append(f"day {index + 1} {meal}: gold '{gold_text[:40]}', agent '{agent_text[:40] or '-'}'")
    return gaps
