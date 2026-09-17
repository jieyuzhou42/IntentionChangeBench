#!/usr/bin/env python3
"""Summarise each instance's candidate pool so an arc can be designed against real options.

The annotation only means something if the constraint it adds is actually
satisfiable (or deliberately not) by the options the agent can see, so this
prints the pools the turns expose in env_feedback.search_results.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any, Dict, List


def pool(instance: Dict[str, Any], key: str) -> List[Dict[str, Any]]:
    """Union the key's pool across every turn; later turns repeat the same search."""
    seen: Dict[str, Dict[str, Any]] = {}
    for turn in instance.get("turns") or []:
        results = (turn.get("env_feedback") or {}).get("search_results") or {}
        for page in results.get(key) or []:
            for item in page.get("items") or []:
                if isinstance(item, dict):
                    name = str(item.get("NAME") or item.get("Name") or item.get("Flight Number") or item)
                    seen.setdefault(name, item)
    return list(seen.values())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()

    for instance in json.loads(args.path.read_text(encoding="utf-8")):
        iid = str(instance["instance_id"])
        if args.only and not any(o in iid for o in args.only):
            continue
        qd = (instance.get("world_state") or {}).get("travelplanner_query_data") or {}
        dates = qd.get("date")
        if isinstance(dates, str):
            try:
                dates = ast.literal_eval(dates)
            except (SyntaxError, ValueError):
                dates = None
        print("=" * 110)
        print(f"{iid}  {qd.get('org')} -> {qd.get('dest')}  {dates}  "
              f"days={qd.get('days')} people={qd.get('people_number')} budget=${qd.get('budget')} "
              f"level={qd.get('level')}")
        print(f"  t0 utterance: {str((instance['turns'][0] or {}).get('user_utterance'))[:220]}")

        accs = pool(instance, "accommodations")
        accs.sort(key=lambda x: float(x.get("price") or 0))
        print(f"  -- accommodations ({len(accs)}) --")
        for a in accs:
            print(f"     ${float(a.get('price') or 0):>7.0f}/n  {str(a.get('room type')):<16} "
                  f"r{a.get('review rate number')}  min{a.get('minimum nights')}n  "
                  f"occ{a.get('maximum occupancy')}  {str(a.get('NAME'))[:34]:<34} "
                  f"| {str(a.get('house_rules'))[:46]}")

        res = pool(instance, "restaurants")
        res.sort(key=lambda x: -float(x.get("Aggregate Rating") or 0))
        print(f"  -- restaurants ({len(res)}) top by rating --")
        for r in res[:14]:
            print(f"     r{r.get('Aggregate Rating')}  ${r.get('Average Cost'):>4}  "
                  f"{str(r.get('Name'))[:30]:<30} | {str(r.get('Cuisines'))[:52]}")

        fl = pool(instance, "transportation")
        print(f"  -- transportation ({len(fl)}) --")
        for f in sorted(fl, key=lambda x: float(x.get("Price") or x.get("cost") or 0)):
            if f.get("Flight Number"):
                print(f"     {f.get('Flight Number')}  {f.get('OriginCityName')}->{f.get('DestCityName')} "
                      f"{f.get('FlightDate')}  ${f.get('Price')}  dep {f.get('DepTime')} arr {f.get('ArrTime')}")
            else:
                print("     " + json.dumps(f, ensure_ascii=False)[:180])

        att = pool(instance, "attractions")
        print(f"  -- attractions ({len(att)}) --")
        print("     " + "; ".join(str(a.get("Name")) for a in att)[:400])

        selfdrive = pool(instance, "googleDistanceMatrix") or pool(instance, "distance_matrix")
        if selfdrive:
            for s in selfdrive[:4]:
                print(f"  -- ground: {json.dumps(s, ensure_ascii=False)[:260]}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
