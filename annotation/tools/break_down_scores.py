#!/usr/bin/env python3
"""Break a travel_pilot_eval.py score file down by the levers the shard was built around.

The aggregate number says how hard the shard is; it does not say which design
choice made it hard. This groups the same per-turn rows three ways:

  * by difficulty category (C1 load / C2 infeasible / C3 squeeze / C4 conflict)
  * by turn shape -- the two dense instances (3 packed turns) against the eight
    that add one constraint per turn, which is the V1/V2 comparison run on real
    annotation rather than on generated variants
  * by turn position, to see whether scores decay as constraints accumulate

Usage:
    python annotation/tools/break_down_scores.py SCORES.json [--dataset DATA.json]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

CATEGORY = {
    "0023": "C1 load (dense)", "0031": "C1 load (dense)",
    "0006": "C2 infeasible", "0037": "C2 infeasible",
    "0003": "C3 squeeze", "0007": "C3 squeeze",
    "0017": "C3 squeeze", "0041": "C3 squeeze",
    "0025": "C4 conflict", "0040": "C4 conflict",
}
DENSE = {"0023", "0031"}

METRICS = [
    ("意图理解", lambda s: s["intention_understanding"]["combined_score"]),
    ("Constraint准确率", lambda s: s["intention_understanding"]["weighted_constraint_value_accuracy"]),
    ("Priority顺序", lambda s: s["intention_understanding"]["priority_order_score"]),
    ("动作合规", lambda s: s["action_compliance"]["weighted_constraint_score"]),
    ("Hard violation", lambda s: float(s["action_compliance"]["hard_priority_violation"])),
]


def short(instance_id: str) -> str:
    return str(instance_id)[-4:]


def table(title: str, groups: Dict[str, List[Dict[str, Any]]], extra: Dict[str, str] | None = None) -> None:
    print(f"\n=== {title} ===")
    head = f"{'':<20}{'轮数':>6}" + "".join(f"{name:>17}" for name, _ in METRICS)
    print(head)
    for key in sorted(groups):
        rows = groups[key]
        cells = "".join(f"{mean(fn(r['scores']) for r in rows):>17.3f}" for _, fn in METRICS)
        label = key if not extra else f"{key} {extra.get(key, '')}"
        print(f"{label:<20}{len(rows):>6}{cells}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("scores", type=Path)
    parser.add_argument("--dataset", type=Path, default=None)
    args = parser.parse_args()

    payload = json.loads(args.scores.read_text(encoding="utf-8"))
    rows = [r for r in payload["rows"] if r.get("scores")]
    meta = payload.get("metadata") or {}
    print(f"agent: {meta.get('agent_model')}   judge: {meta.get('judge_model')}")
    print(f"评分轮数: {len(rows)}")

    by_instance: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_instance[short(row["instance_id"])].append(row)

    table("按实例", by_instance, {k: f"[{CATEGORY.get(k, '?')}]" for k in by_instance})

    by_cat: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for iid, items in by_instance.items():
        by_cat[CATEGORY.get(iid, "?")].extend(items)
    table("按难度类型", by_cat)

    by_shape: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for iid, items in by_instance.items():
        by_shape["密集 (3轮打包)" if iid in DENSE else "滴灌 (每轮1个)"].extend(items)
    table("按轮次形态", by_shape)

    by_turn: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_turn[f"t{row['turn_id']}"].append(row)
    table("按轮次位置", by_turn)

    if args.dataset and args.dataset.exists():
        instances = {short(i["instance_id"]): i for i in json.loads(args.dataset.read_text(encoding="utf-8"))}
        print("\n=== 约束数量 vs 得分（每轮一个点）===")
        buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in rows:
            inst = instances.get(short(row["instance_id"]))
            if not inst:
                continue
            turn = next((t for t in inst["turns"] if t.get("turn_id") == row["turn_id"]), None)
            if not turn:
                continue
            n = len((turn.get("gold_current_intention") or {}).get("constraints") or {})
            buckets[f"{(n // 3) * 3}-{(n // 3) * 3 + 2} 个约束"].append(row)
        table("按约束数量分桶", buckets)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
