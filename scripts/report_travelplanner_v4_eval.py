#!/usr/bin/env python3
"""Aggregate judged TravelPlanner v4 trajectories into intention/action tables.

Reads ``scored_rows.json`` from ``judge_travelplanner_v4_trajectories.py`` and
writes ``metrics.json`` and ``tables.md`` next to it. Metric definitions are in
the report README; the short version:

Action level (main judge, pass A)
  success      every must_have gold constraint is ``satisfied`` in the action
  lenient      no must_have gold constraint is ``violated`` (unknown allowed)
  tier rates   satisfied / violated / unknown share of gold constraints per tier

Intention level (alignment judge, pass B)
  P / R / F1   item precision (value-correct aligned items / predicted items) and
               gold recall (gold fields with a value-correct item / gold fields)
  change       the same restricted to this turn's add/override/relax gold_delta
               fields and to items new since the agent's previous turn
  priority     tier agreement on recognised gold fields
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from eval.human_annotated_pilot import aggregate_scored_rows, score_judged_turn

DISPLAY = {
    "claude-sonnet-4.6": "Claude Sonnet 4.6",
    "claude-sonnet-5": "Claude Sonnet 5",
    "deepseek-r1": "DeepSeek R1",
    "gpt-5.6-luna": "GPT-5.6 Luna",
    "gpt-5.6-sol": "GPT-5.6 Sol",
    "grok-4.6": "Grok 4.6",
    "kimi-k2.5": "Kimi K2.5",
    "minimax-m2.5": "MiniMax M2.5",
    "nova-pro": "Nova Pro",
    "qwen3-235b": "Qwen3 235B",
    "qwen3-32b": "Qwen3 32B",
}
SHARDS = ("shard1", "shard2", "shard3")
GOLD_TIER = {"high": "must_have", "medium": "preferred", "low": "optional"}
TIERS = ("must_have", "preferred", "optional")
VALUE_CHANGE_OPS = ("add", "override", "relax")
# Trip parameters from the original query. From turn 1 on, shard 2/3 gold demotes
# them to optional wholesale, which no agent can infer from the utterances.
BACKGROUND_FIELDS = {"days", "people_number", "org", "dest", "visiting_city_number", "start_date", "end_date"}


def mean(values: Sequence[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def ratio(num: float, den: float) -> Optional[float]:
    return num / den if den else None


def f1(p: Optional[float], r: Optional[float]) -> float:
    p, r = p or 0.0, r or 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def gold_tiers(gold_intention: Dict[str, Any]) -> Dict[str, str]:
    priority = gold_intention.get("priority") or {}
    return {
        str(field): GOLD_TIER[level]
        for level in GOLD_TIER
        for field in priority.get(level) or []
    }


def intent_items(prediction: Any) -> List[Dict[str, Any]]:
    if isinstance(prediction, dict) and isinstance(prediction.get("intent"), list):
        return [item for item in prediction["intent"] if isinstance(item, dict)]
    return []


def item_key(item: Dict[str, Any]) -> str:
    value = json.dumps(item.get("value"), sort_keys=True, ensure_ascii=False).lower()
    return str(item.get("field") or "").strip().lower() + "=" + "".join(value.split())


from eval.travelplanner_checks import (  # noqa: E402  (re-exported for callers of this script)
    FLIGHT_NUMBER,
    MEALS,
    _compact,
    _is_empty_slot,
    _match_entry,
    _text,
    hotel_validity,
    meal_coverage_gaps,
    trip_cost,
)


def with_deterministic_budget(row: Dict[str, Any], reference: Any, query_people: Any) -> Dict[str, Any]:
    """Replace the judge's budget action_status with the database-priced verdict."""
    gold = row["gold_intention"].get("constraints") or {}
    budget = gold.get("budget")
    if not isinstance(budget, (int, float)):
        return row
    people = gold.get("people_number") if isinstance(gold.get("people_number"), int) else query_people
    cost = trip_cost((row.get("action") or {}).get("itinerary"), reference, people or 1)
    row = copy.deepcopy(row)
    for item in row["judge_output"].get("constraint_judgments") or []:
        if item.get("gold_field") == "budget":
            item["judge_action_status"] = item.get("action_status")
            item["action_status"] = "satisfied" if cost["total"] <= budget else "violated"
    row["deterministic_cost"] = {**cost, "budget": budget, "people": people}
    judged = [i for i in row["judge_output"].get("constraint_judgments") or [] if i.get("gold_field") == "budget"]
    if judged:
        row["budget_judge_vs_deterministic"] = [judged[0]["judge_action_status"], judged[0]["action_status"]]
    row["scores"] = score_judged_turn(gold_intention=row["gold_intention"], judgment=row["judge_output"])
    return row


def turn_metrics(row: Dict[str, Any], previous: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    gold = {
        str(k): v for k, v in (row["gold_intention"].get("constraints") or {}).items() if v is not None
    }
    tiers = gold_tiers(row["gold_intention"])
    items = intent_items(row["prediction"])
    alignment = {int(a["index"]): dict(a) for a in row["alignment"].get("items") or []}
    # An alignment to a field that is not in this turn's gold expresses no current
    # gold constraint, so it counts as unaligned.
    stale = 0
    for a in alignment.values():
        if a.get("gold_field") is not None and str(a["gold_field"]) not in gold:
            a["gold_field"], a["value_match"] = None, False
            stale += 1

    # --- pass B: item <-> gold alignment -----------------------------------
    aligned: Dict[str, List[int]] = defaultdict(list)
    for index, a in alignment.items():
        if a.get("gold_field") is not None:
            aligned[str(a["gold_field"])].append(index)
    correct_items = [i for i, a in alignment.items() if a.get("gold_field") is not None and a.get("value_match")]
    covered = {f for f in gold if any(alignment[i].get("value_match") for i in aligned.get(f, []))}
    recognized = {f for f in gold if aligned.get(f)}

    precision = ratio(len(correct_items), len(items)) or 0.0
    recall = ratio(len(covered), len(gold)) or 0.0

    def predicted_tier(field: str) -> Optional[str]:
        indices = aligned.get(field) or []
        matching = [i for i in indices if alignment[i].get("value_match")]
        chosen = (matching or indices)[:1]
        return items[chosen[0]].get("priority") if chosen else None

    tier_pairs = [(tiers[f], predicted_tier(f)) for f in gold if f in tiers]
    tier_recognized = [(g, p) for g, p in tier_pairs if p is not None]

    delta = row.get("gold_delta") or {}
    reprioritized = [f for f, d in delta.items() if f in tiers and isinstance(d, dict) and d.get("op") == "reprioritize"]
    changed = {
        f: d.get("op")
        for f, d in delta.items()
        if f in gold and isinstance(d, dict) and d.get("op") in VALUE_CHANGE_OPS
    }

    change = None
    if previous is not None and changed:
        previous_keys = {item_key(item) for item in intent_items(previous["prediction"])}
        claimed = []
        for index, item in enumerate(items):
            if item_key(item) in previous_keys:
                continue
            a = alignment.get(index) or {}
            field = a.get("gold_field")
            # A value-correct restatement of an unchanged constraint is not a claimed change.
            if field is not None and field not in changed and a.get("value_match"):
                continue
            claimed.append(bool(field in changed and a.get("value_match")))
        change_tp_fields = [f for f in changed if f in covered]
        change = {
            "gold": len(changed),
            "caught": len(change_tp_fields),
            "claimed": len(claimed),
            "claimed_correct": sum(claimed),
            "by_op": {op: [f in covered for f, o in changed.items() if o == op] for op in VALUE_CHANGE_OPS},
            "precision": ratio(sum(claimed), len(claimed)) or 0.0,
            "recall": len(change_tp_fields) / len(changed),
        }
        change["f1"] = f1(change["precision"], change["recall"])

    # --- pass A: main judge ---------------------------------------------------
    per_constraint = {c["field"]: c for c in row["scores"]["per_constraint"]}
    tier_status: Dict[str, Counter] = defaultdict(Counter)
    for field, c in per_constraint.items():
        tier_status[tiers.get(field, "entity")][c["action_status"]] += 1
    must = [per_constraint[f]["action_status"] for f in gold if tiers.get(f) == "must_have"]

    return {
        "intent": {
            "n_gold": len(gold),
            "n_pred": len(items),
            "correct_items": len(correct_items),
            "covered": len(covered),
            "precision": precision,
            "recall": recall,
            "f1": f1(precision, recall),
            "turn_exact": len(covered) == len(gold) and len(correct_items) == len(items),
            "tier_pairs": tier_pairs,
            "tier_pairs_nonbg": [
                (tiers[f], predicted_tier(f)) for f in gold if f in tiers and f not in BACKGROUND_FIELDS
            ],
            "tier_correct": sum(g == p for g, p in tier_recognized),
            "tier_recognized": len(tier_recognized),
            "priority_turn_exact": all(g == p for g, p in tier_pairs),
            "reprioritize": [tiers[f] == predicted_tier(f) for f in reprioritized],
            "judge_a_value_match": {f: bool(per_constraint[f]["value_match"]) for f in gold},
            "judge_b_covered": {f: f in covered for f in gold},
            "unaligned_items": sum(1 for a in alignment.values() if a.get("gold_field") is None),
            "stale_alignments": stale,
        },
        "change": change,
        "action": {
            "success": all(s == "satisfied" for s in must),
            "lenient_success": not any(s == "violated" for s in must),
            "tier_status": {k: dict(v) for k, v in tier_status.items()},
        },
    }


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    m = [r["_m"] for r in rows]
    intent = [x["intent"] for x in m]
    change = [x["change"] for x in m if x["change"]]

    tier_status: Dict[str, Counter] = defaultdict(Counter)
    for x in m:
        for tier, counts in x["action"]["tier_status"].items():
            tier_status[tier].update(counts)

    def tier_rates(tier: str) -> Dict[str, Any]:
        counts = tier_status.get(tier, Counter())
        total = sum(counts.values())
        return {
            "n": total,
            "satisfied": ratio(counts["satisfied"], total),
            "violated": ratio(counts["violated"], total),
            "unknown": ratio(counts["unknown"], total),
        }

    confusion: Dict[str, Counter] = defaultdict(Counter)
    for x in intent:
        for g, p in x["tier_pairs"]:
            confusion[g][p or "missed"] += 1
    reprio = [ok for x in intent for ok in x["reprioritize"]]
    by_op = {op: [ok for c in change for ok in c["by_op"][op]] for op in VALUE_CHANGE_OPS}
    agree = [
        x["judge_a_value_match"][f] == x["judge_b_covered"][f]
        for x in intent
        for f in x["judge_a_value_match"]
    ]

    micro_p = ratio(sum(x["correct_items"] for x in intent), sum(x["n_pred"] for x in intent))
    micro_r = ratio(sum(x["covered"] for x in intent), sum(x["n_gold"] for x in intent))
    change_micro_p = ratio(sum(c["claimed_correct"] for c in change), sum(c["claimed"] for c in change))
    change_micro_r = ratio(sum(c["caught"] for c in change), sum(c["gold"] for c in change))

    by_shard = {}
    for shard in SHARDS:
        shard_rows = [r for r in rows if r["shard"] == shard]
        by_shard[shard] = {
            "turns": len(shard_rows),
            "success": sum(r["_m"]["action"]["success"] for r in shard_rows),
            "lenient_success": sum(r["_m"]["action"]["lenient_success"] for r in shard_rows),
        }

    by_style = {}
    for style in sorted({str(r.get("linguistic_style")) for r in rows}):
        style_rows = [r for r in rows if str(r.get("linguistic_style")) == style]
        style_change = [r["_m"]["change"] for r in style_rows if r["_m"]["change"]]
        by_style[style] = {
            "turns": len(style_rows),
            "success": mean([float(r["_m"]["action"]["success"]) for r in style_rows]),
            "intent_f1": mean([r["_m"]["intent"]["f1"] for r in style_rows]),
            "change_turns": len(style_change),
            "change_recall": mean([c["recall"] for c in style_change]),
        }

    priced = [r for r in rows if "deterministic_cost" in r]
    transitions = Counter("->".join(r["budget_judge_vs_deterministic"]) for r in priced if r.get("budget_judge_vs_deterministic"))
    budget_rows = [
        c["action_status"] for r in rows for c in r["scores"]["per_constraint"] if c["field"] == "budget"
    ]
    budget_counts = Counter(budget_rows)
    budget = {
        "turns": len(priced),
        "unpriced_slots_per_turn": mean([len(r["deterministic_cost"]["unpriced"]) for r in priced]),
        "deterministic_within_budget": mean([float(r["deterministic_cost"]["total"] <= r["deterministic_cost"]["budget"]) for r in priced]),
        "judge_to_deterministic": dict(transitions),
    }

    return {
        "turns": len(rows),
        "instances": len({(r["shard"], r["instance_id"]) for r in rows}),
        "budget": budget,
        "budget_status": {k: ratio(v, len(budget_rows)) for k, v in budget_counts.items()},
        "action": {
            "success": mean([float(x["action"]["success"]) for x in m]),
            "success_count": sum(x["action"]["success"] for x in m),
            "lenient_success": mean([float(x["action"]["lenient_success"]) for x in m]),
            "tiers": {tier: tier_rates(tier) for tier in (*TIERS, "entity")},
            "by_shard": by_shard,
        },
        "intent": {
            "precision": mean([x["precision"] for x in intent]),
            "recall": mean([x["recall"] for x in intent]),
            "f1": mean([x["f1"] for x in intent]),
            "micro_precision": micro_p,
            "micro_recall": micro_r,
            "micro_f1": f1(micro_p, micro_r),
            "turn_exact": mean([float(x["turn_exact"]) for x in intent]),
            "avg_pred_items": mean([x["n_pred"] for x in intent]),
            "avg_gold_fields": mean([x["n_gold"] for x in intent]),
            "unaligned_items_per_turn": mean([x["unaligned_items"] for x in intent]),
            "stale_alignments": sum(x["stale_alignments"] for x in intent),
            "priority_accuracy": ratio(
                sum(x["tier_correct"] for x in intent), sum(x["tier_recognized"] for x in intent)
            ),
            "priority_accuracy_strict": ratio(
                sum(x["tier_correct"] for x in intent), sum(len(x["tier_pairs"]) for x in intent)
            ),
            "priority_turn_exact": mean([float(x["priority_turn_exact"]) for x in intent]),
            "priority_accuracy_nonbg": ratio(
                sum(g == p for x in intent for g, p in x["tier_pairs_nonbg"] if p is not None),
                sum(1 for x in intent for _, p in x["tier_pairs_nonbg"] if p is not None),
            ),
            "tier_confusion": {g: dict(confusion[g]) for g in TIERS},
            "reprioritize_accuracy": mean([float(v) for v in reprio]),
            "reprioritize_n": len(reprio),
            "judge_agreement_value": mean([float(v) for v in agree]),
        },
        "change": {
            "turns": len(change),
            "precision": mean([c["precision"] for c in change]),
            "recall": mean([c["recall"] for c in change]),
            "f1": mean([c["f1"] for c in change]),
            "micro_precision": change_micro_p,
            "micro_recall": change_micro_r,
            "micro_f1": f1(change_micro_p, change_micro_r),
            "recall_by_op": {op: {"n": len(v), "recall": mean([float(x) for x in v])} for op, v in by_op.items()},
        },
        "by_linguistic_style": by_style,
        "main_scorer": aggregate_scored_rows(rows)["overall"],
    }


def pct(value: Optional[float]) -> str:
    return "–" if value is None else f"{100 * value:.1f}"


def tables(variants: Dict[str, Dict[str, Any]], judge_model: str, comparison: Optional[Dict[str, Any]] = None) -> str:
    """Headline tables use the grounded action evidence when it is available.

    Intention-level numbers come from the alignment pass, which is shared by
    both evidence variants.
    """
    primary_name = next(name for name in ("pool_det", "grounded_det", "grounded", "action") if name in variants)
    primary = variants[primary_name]
    models = [m for m in DISPLAY if m in primary]
    out: List[str] = []

    def action_table(results: Dict[str, Any], title: str) -> None:
        nonlocal out
        out += [
            title,
            "",
            "| 模型 | shard1 成功/32 | shard2 成功/64 | shard3 成功/64 | 合并成功率 (/160) | 宽松成功率 | Must 满足 | Preferred 满足 | Optional 满足 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for m in models:
            r = results[m]
            a = r["action"]
            shard = a["by_shard"]
            out.append(
                f"| {DISPLAY[m]} | {shard['shard1']['success']} | {shard['shard2']['success']} | "
                f"{shard['shard3']['success']} | **{a['success_count']}/{r['turns']} · {pct(a['success'])}%** | "
                f"{pct(a['lenient_success'])}% | {pct(a['tiers']['must_have']['satisfied'])}% | "
                f"{pct(a['tiers']['preferred']['satisfied'])}% | {pct(a['tiers']['optional']['satisfied'])}% |"
            )
        out.append("")

    def tier_detail(results: Dict[str, Any], title: str) -> None:
        nonlocal out
        out += [
            title,
            "",
            "| 模型 | Must (n) S / V / U | Preferred (n) S / V / U | Optional (n) S / V / U | Entity (n) S / V / U |",
            "|---|---|---|---|---|",
        ]
        for m in models:
            cells = []
            for tier in (*TIERS, "entity"):
                t = results[m]["action"]["tiers"][tier]
                cells.append(f"({t['n']}) {pct(t['satisfied'])} / {pct(t['violated'])} / {pct(t['unknown'])}")
            out.append(f"| {DISPLAY[m]} | " + " | ".join(cells) + " |")
        out.append("")

    def main_scorer(results: Dict[str, Any], title: str) -> None:
        nonlocal out
        out += [
            title,
            "",
            "| 模型 | Intention understanding | Constraint value acc. | Priority order score | Action compliance | Hard-priority violation rate |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for m in models:
            sc = results[m]["main_scorer"]
            out.append(
                f"| {DISPLAY[m]} | {pct(sc['intention_understanding_score'])} | {pct(sc['constraint_value_accuracy'])} | "
                f"{pct(sc['priority_order_score'])} | {pct(sc['action_compliance_score'])} | "
                f"{pct(sc['hard_priority_violation_rate'])} |"
            )
        out.append("")

    labels = {
        "grounded_det": "grounded evidence + 确定性 budget",
        "grounded": "grounded evidence，budget 由 judge 判",
        "pool_det": "候选池 evidence + 确定性 budget",
        "pool": "候选池 evidence，budget 由 judge 判",
        "action": "action-only evidence，严格 main fixed-search",
    }
    action_table(primary, f"### Action level（{labels[primary_name]}，主结果）")
    out += [
        "### Intention level",
        "",
        "| 模型 | Overall P / R / F1 | Change P / R / F1 | Turn Exact | Priority Accuracy | Priority Turn Exact |",
        "|---|---|---|---:|---:|---:|",
    ]
    for m in models:
        i, c = primary[m]["intent"], primary[m]["change"]
        out.append(
            f"| {DISPLAY[m]} | {pct(i['precision'])} / {pct(i['recall'])} / {pct(i['f1'])} | "
            f"{pct(c['precision'])} / {pct(c['recall'])} / {pct(c['f1'])} | {pct(i['turn_exact'])} | "
            f"{pct(i['priority_accuracy'])} | {pct(i['priority_turn_exact'])} |"
        )
    out.append("")

    present = [name for name in ("action", "grounded", "pool", "grounded_det", "pool_det") if name in variants]
    out += [
        "### Action level：三种判法对照（严格成功率 / 宽松成功率 / Must 满足率 / budget satisfied 率）",
        "",
        "| 模型 | " + " | ".join(labels[name] for name in present) + " |",
        "|---|" + "---|" * len(present),
    ]
    for m in models:
        cells = []
        for name in present:
            a = variants[name][m]["action"]
            bs = variants[name][m].get("budget_status") or {}
            cells.append(
                f"{pct(a['success'])} / {pct(a['lenient_success'])} / {pct(a['tiers']['must_have']['satisfied'])} / "
                f"{pct(bs.get('satisfied'))}"
            )
        out.append(f"| {DISPLAY[m]} | " + " | ".join(cells) + " |")
    out.append("")

    if comparison:
        out += [
            "### 对照实验：grounded（只附选中实体）vs 候选池（附 agent 可见的全部搜索结果），budget 均为确定性",
            "",
            "| 模型 | 严格成功 grounded → 候选池 | 非 budget 约束判定一致率 | 非 budget S / V / U（grounded） | 非 budget S / V / U（候选池） | 主要变化 | LLM 自判 budget unknown 率 grounded → 候选池 | LLM 自判 budget 与计价一致率（已判定部分）grounded → 候选池 |",
            "|---|---|---:|---|---|---|---|---|",
        ]
        for m in models:
            c = comparison[m]
            br, orr = c["base_rates"], c["other_rates"]
            flips = sorted(((k, v) for k, v in c["moves"].items() if k[0] != k[-1]), key=lambda kv: -kv[1])[:3]
            out.append(
                f"| {DISPLAY[m]} | {c['success_base']} → {c['success_other']} | {pct(c['agreement'])} | "
                f"{pct(br.get('satisfied'))} / {pct(br.get('violated'))} / {pct(br.get('unknown'))} | "
                f"{pct(orr.get('satisfied'))} / {pct(orr.get('violated'))} / {pct(orr.get('unknown'))} | "
                + ", ".join(f"{k} {v}" for k, v in flips)
                + f" | {pct(c['llm_budget_base']['_unknown'])} → {pct(c['llm_budget_other']['_unknown'])}"
                + f" | {pct(c['llm_budget_base']['_agree_when_decided'])} → {pct(c['llm_budget_other']['_agree_when_decided'])} |"
            )
        out.append("")

    for name in ("pool_det", "grounded_det", "action"):
        if name in variants:
            main_scorer(variants[name], f"### Main shared scorer（`aggregate_scored_rows`，{labels[name]}）")
    tier_detail(primary, f"### Action detail：各 gold tier 的 satisfied / violated / unknown（{labels[primary_name]}，约束级合并）")

    if primary_name.endswith("_det"):
        out += [
            "### 确定性 budget 诊断（judge 判定 → 数据库计价判定，按轮计数）",
            "",
            "| 模型 | 有 budget 的轮数 | 计价 ≤ budget | 未定价槽位/轮 | judge→确定性 转移 |",
            "|---|---:|---:|---:|---|",
        ]
        for m in models:
            bd = primary[m]["budget"]
            moves = ", ".join(f"{k} {v}" for k, v in sorted(bd["judge_to_deterministic"].items()))
            out.append(
                f"| {DISPLAY[m]} | {bd['turns']} | {pct(bd['deterministic_within_budget'])} | "
                f"{bd['unpriced_slots_per_turn']:.2f} | {moves} |"
            )
        out.append("")

    out += [
        "### Intention detail",
        "",
        "| 模型 | micro P / R / F1 | 平均预测条数 | 平均 gold 字段 | 未对齐条数/轮 | Change 轮数 | Change micro P / R / F1 | Change recall add / override / relax | Priority acc.（未识别计错） | Priority acc.（排除背景字段） | Reprioritize acc. |",
        "|---|---|---:|---:|---:|---:|---|---|---:|---:|---:|",
    ]
    for m in models:
        i, c = primary[m]["intent"], primary[m]["change"]
        ops = " / ".join(pct(c["recall_by_op"][op]["recall"]) for op in VALUE_CHANGE_OPS)
        out.append(
            f"| {DISPLAY[m]} | {pct(i['micro_precision'])} / {pct(i['micro_recall'])} / {pct(i['micro_f1'])} | "
            f"{i['avg_pred_items']:.1f} | {i['avg_gold_fields']:.1f} | {i['unaligned_items_per_turn']:.2f} | "
            f"{c['turns']} | {pct(c['micro_precision'])} / {pct(c['micro_recall'])} / {pct(c['micro_f1'])} | {ops} | "
            f"{pct(i['priority_accuracy_strict'])} | {pct(i['priority_accuracy_nonbg'])} | "
            f"{pct(i['reprioritize_accuracy'])} |"
        )
    out.append("")

    out += [
        "### Priority tier confusion（行 = gold tier，单元格 = 预测 tier 占比）",
        "",
        "| 模型 | gold Must → must / pref / opt / missed | gold Preferred → must / pref / opt / missed | gold Optional → must / pref / opt / missed |",
        "|---|---|---|---|",
    ]
    for m in models:
        cells = []
        for g in TIERS:
            row = primary[m]["intent"]["tier_confusion"].get(g, {})
            total = sum(row.values())
            cells.append(
                f"({total}) " + " / ".join(pct(ratio(row.get(p, 0), total)) for p in (*TIERS, "missed"))
            )
        out.append(f"| {DISPLAY[m]} | " + " | ".join(cells) + " |")
    out.append("")

    styles = [s for s in sorted({s for m in models for s in primary[m]["by_linguistic_style"]}) if s != "None"]
    out += [
        "### 按 linguistic style 拆分（success / intent F1 / change recall）",
        "",
        "| 模型 | " + " | ".join(f"{s}" for s in styles) + " |",
        "|---|" + "---|" * len(styles),
    ]
    for m in models:
        cells = []
        for style in styles:
            b = primary[m]["by_linguistic_style"].get(style)
            cells.append(
                "–" if not b else
                f"({b['turns']}) {pct(b['success'])} / {pct(b['intent_f1'])} / {pct(b['change_recall'])}"
            )
        out.append(f"| {DISPLAY[m]} | " + " | ".join(cells) + " |")

    judge_a_agreement = " · ".join(
        f"{DISPLAY[m]} {pct(primary[m]['intent']['judge_agreement_value'])}" for m in models
    )
    out += [
        "",
        f"Pass A 与 pass B 对 gold 字段 value 判定的一致率：{judge_a_agreement}",
        "",
        f"Judge: `{judge_model}` via OpenRouter.",
    ]
    return "\n".join(out) + "\n"


def load_world(metadata: Dict[str, Any]) -> Dict[tuple, Dict[str, Any]]:
    """(shard slug, instance id) -> gold world_state, via the run manifest."""
    run_dir = Path(metadata["run_dir"])
    run_dir = run_dir if run_dir.is_absolute() else REPO_ROOT / run_dir
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    files = {entry["shard_slug"]: entry["shard_file"] for entry in manifest["outputs"]}
    world = {}
    for shard, filename in files.items():
        for instance in json.loads((Path(metadata["gold_dir"]) / filename).read_text(encoding="utf-8")):
            world[(shard, instance["instance_id"])] = instance.get("world_state") or {}
    return world


def load_results(path: Path, deterministic_budget: bool = False) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["rows"]
    if deterministic_budget:
        world = load_world(payload["metadata"])
        rows = [
            with_deterministic_budget(
                row,
                world[(row["shard"], row["instance_id"])].get("reference_information"),
                (world[(row["shard"], row["instance_id"])].get("travelplanner_query_data") or {}).get("people_number"),
            )
            for row in rows
        ]
    grouped: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["shard"], row["instance_id"])].append(row)
    for group in grouped.values():
        group.sort(key=lambda r: r["turn_id"])
        previous = None
        for row in group:
            row["_m"] = turn_metrics(row, previous)
            previous = row
    by_model: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_model[row["model"]].append(row)
    return {
        "metadata": payload["metadata"],
        "models": {model: summarize(model_rows) for model, model_rows in sorted(by_model.items())},
        "_rows": rows,
    }


def llm_budget_vs_deterministic(raw_rows: List[Dict[str, Any]], det_rows: List[Dict[str, Any]]) -> Dict[str, Counter]:
    """model -> Counter of 'LLM budget status|deterministic status'."""
    key = lambda r: (r["model"], r["shard"], r["instance_id"], r["turn_id"])
    det = {key(r): r for r in det_rows}
    out: Dict[str, Counter] = defaultdict(Counter)
    for r in raw_rows:
        judged = {c["field"]: c["action_status"] for c in r["scores"]["per_constraint"]}
        d = det[key(r)]
        if "budget" in judged and "deterministic_cost" in d:
            verdict = "satisfied" if d["deterministic_cost"]["total"] <= d["deterministic_cost"]["budget"] else "violated"
            out[r["model"]][f"{judged['budget']}|{verdict}"] += 1
    return out


def _budget_summary(counts: Counter) -> Dict[str, Optional[float]]:
    total = sum(counts.values())
    decided = {k: v for k, v in counts.items() if not k.startswith("unknown")}
    return {
        "unknown": ratio(total - sum(decided.values()), total),
        "agree_when_decided": ratio(sum(v for k, v in decided.items() if k.split("|")[0] == k.split("|")[1]), sum(decided.values())),
    }


def compare_evidence(
    base_rows: List[Dict[str, Any]],
    other_rows: List[Dict[str, Any]],
    raw_base: List[Dict[str, Any]],
    raw_other: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Per-constraint comparison of two evidence variants on identical turns.

    ``base_rows`` / ``other_rows`` carry the deterministic budget; ``raw_*`` are
    the same judgments before the budget override, used to see how the LLM
    itself judged budget with each evidence.
    """
    key = lambda r: (r["model"], r["shard"], r["instance_id"], r["turn_id"])
    other = {key(r): r for r in other_rows}
    llm_base = llm_budget_vs_deterministic(raw_base, base_rows)
    llm_other = llm_budget_vs_deterministic(raw_other, other_rows)
    out: Dict[str, Dict[str, Any]] = {}
    for model in sorted({r["model"] for r in base_rows}):
        moves: Counter = Counter()
        base_status: Counter = Counter()
        other_status: Counter = Counter()
        success = Counter()
        for r in (r for r in base_rows if r["model"] == model):
            o = other[key(r)]
            a = {c["field"]: c["action_status"] for c in r["scores"]["per_constraint"]}
            b = {c["field"]: c["action_status"] for c in o["scores"]["per_constraint"]}
            for field in a:
                if field == "budget":
                    continue
                moves[f"{a[field][0].upper()}→{b[field][0].upper()}"] += 1
                base_status[a[field]] += 1
                other_status[b[field]] += 1
            success["base"] += r["_m"]["action"]["success"]
            success["other"] += o["_m"]["action"]["success"]
        n = sum(base_status.values())
        out[model] = {
            "non_budget_constraints": n,
            "agreement": ratio(sum(v for k, v in moves.items() if k[0] == k[-1]), n),
            "moves": dict(moves),
            "base_rates": {k: ratio(v, n) for k, v in base_status.items()},
            "other_rates": {k: ratio(v, n) for k, v in other_status.items()},
            "success_base": success["base"],
            "success_other": success["other"],
            "llm_budget_base": {**dict(llm_base[model]), **{"_" + k: v for k, v in _budget_summary(llm_base[model]).items()}},
            "llm_budget_other": {**dict(llm_other[model]), **{"_" + k: v for k, v in _budget_summary(llm_other[model]).items()}},
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report_dir", type=Path)
    args = parser.parse_args()
    sources = (
        ("grounded_det", "scored_rows_grounded.json", True),
        ("grounded", "scored_rows_grounded.json", False),
        ("pool_det", "scored_rows_pool.json", True),
        ("pool", "scored_rows_pool.json", False),
        ("action", "scored_rows.json", False),
    )
    variants = {
        name: load_results(args.report_dir / filename, deterministic_budget=det)
        for name, filename, det in sources
        if (args.report_dir / filename).exists()
    }
    comparison = None
    if {"grounded_det", "grounded", "pool_det", "pool"} <= set(variants):
        comparison = compare_evidence(
            variants["grounded_det"]["_rows"],
            variants["pool_det"]["_rows"],
            variants["grounded"]["_rows"],
            variants["pool"]["_rows"],
        )
    (args.report_dir / "metrics.json").write_text(
        json.dumps(
            {
                **{name: {k: v for k, v in data.items() if k != "_rows"} for name, data in variants.items()},
                "grounded_vs_pool": comparison,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    judge_model = next(iter(variants.values()))["metadata"]["judge_model"]
    text = tables({name: v["models"] for name, v in variants.items()}, judge_model, comparison)
    (args.report_dir / "tables.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
