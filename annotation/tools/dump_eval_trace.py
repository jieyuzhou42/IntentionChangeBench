#!/usr/bin/env python3
"""Render an eval score file as a readable per-turn trace.

The result file holds everything needed to audit a run -- what the agent thought
the user wanted, what it planned, and how the judge ruled on each constraint --
but it is one 600KB-1.4MB blob. This writes the same content as text, one section
per turn, so a run can be read start to finish.

Handles both scorers. They store the same evidence under different keys, and the
v2 rows carry three things v1 has no slot for: which candidate-pool records each
verdict was anchored to, how the three judge samples voted, and the contract-free
findings (plan self-contradictions, utterance-implied violations) with their
derivation chains.

Usage:
    python annotation/tools/dump_eval_trace.py SCORES.json --out trace.txt
    python annotation/tools/dump_eval_trace.py SCORES.json --only 0006 --out one.txt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

MARK = {"satisfied": "✓", "violated": "✗", "unknown": "?"}


def is_v2(payload: Dict[str, Any]) -> bool:
    rows = payload.get("rows") or []
    return bool(rows) and "intention" in ((rows[0].get("scores") or {}))


def fmt_itinerary(action: Dict[str, Any]) -> List[str]:
    lines = []
    for day in action.get("itinerary") or []:
        lines.append(f"    {day.get('day')}  [{day.get('current_city')}]")
        for field in ("transportation", "breakfast", "lunch", "dinner",
                      "attraction", "accommodation"):
            value = str(day.get(field) or "-").strip()
            if value in {"", "-"}:
                continue
            lines.append(f"       {field:<15}{value[:150]}")
    return lines


def fmt_prediction(row: Dict[str, Any], out: List[str]) -> None:
    gold = (row.get("gold_intention") or {}).get("constraints") or {}
    pred = row.get("agent_intention_prediction") or {}
    predicted = pred.get("constraints") or {}
    out.append(f"【agent 理解的意图】 {len(predicted)} 条 (gold {len(gold)} 条)")
    for key, value in predicted.items():
        same = key in gold and str(gold[key]).strip().lower() == str(value).strip().lower()
        flag = " " if same else ("~" if key in gold else "+")
        out.append(f"    {flag} {key:<26}{str(value)[:80]}")
    for key in gold:
        if key not in predicted:
            out.append(f"    - {key:<26}{str(gold[key])[:80]}   ← 漏了")
    ranked = ((pred.get("priority") or {}).get("ranked_fields")) or []
    out.append(f"    优先级排序: {', '.join(str(r) for r in ranked[:10])}")
    if pred.get("explanation"):
        out.append(f"    理由: {str(pred['explanation'])[:220]}")


def dump_v1(row: Dict[str, Any], out: List[str]) -> None:
    scores = row.get("scores")
    if not scores:
        out.append("【judge】 该 instance 打分失败")
        return
    iu, ac = scores["intention_understanding"], scores["action_compliance"]
    out.append(f"【judge 判决】 意图 {iu['combined_score']:.3f} "
               f"(constraint {iu['weighted_constraint_value_accuracy']:.3f} / "
               f"priority {iu['priority_order_score']:.3f})   "
               f"动作 {ac['weighted_constraint_score']:.3f}   "
               f"硬违规 {'是' if ac['hard_priority_violation'] else '否'}")
    for item in scores["per_constraint"]:
        out.append(
            f"  {MARK.get(item['action_status'], '?')} {item['field']:<26}"
            f"权重{item['weight']:.0f}  "
            f"识别{'√' if item['recognized'] else '×'} 值对{'√' if item['value_match'] else '×'}  "
            f"gold={str(item['gold_value'])[:40]}"
        )
        if item.get("note"):
            out.append(f"      {str(item['note'])[:170]}")
    if scores.get("judge_summary"):
        out.append(f"  总评: {scores['judge_summary']}")


def dump_v2(row: Dict[str, Any], out: List[str]) -> None:
    scores = row.get("scores") or {}
    intent, action = scores["intention"], scores["action"]
    capture = scores["change_capture"]
    priority = scores.get("priority")

    changed = row.get("changed_this_turn") or {}
    if changed:
        out.append("【本轮 gold 变化】")
        for field, change in changed.items():
            if not isinstance(change, dict):
                continue
            op = change.get("op")
            arrow = f"{change.get('old')} → {change.get('new')}" if op != "add" else str(change.get("new"))
            hit = "✓抓到" if field in (capture.get("caught_fields") or []) else (
                "✗漏了" if field in (capture.get("missed_fields") or []) else "  ")
            out.append(f"    {hit}  {field:<24}[{op}] {str(arrow)[:70]}")
        out.append("")

    fmt_prediction(row, out)
    out.append("")
    out.append("【agent 产出的行程】")
    out.extend(fmt_itinerary(row.get("action") or {}) or ["    (空)"])
    out.append("")

    if not intent.get("scored"):
        out.append("【计分】 本轮全部约束都是用户从未动过的，整轮不进主指标")
        out.append("")

    # t0 这类整轮被排除的轮次，主指标是 None，只报对照口径。
    num = lambda v, d="—": f"{v:.3f}" if isinstance(v, (int, float)) else d
    head = (f"【judge 判决】 识别 {num(intent['constraint_recall'])} "
            f"值准确 {num(intent['constraint_value_accuracy'])} "
            if intent.get("scored") else "【judge 判决】(不计分) ")
    tail = (f"动作 {num(action['score'])}   硬违规 "
            + ("是" if action["hard_violation"] else "否" if action["hard_violation"] is not None else "—")
            + f"   [含未变约束 动作 {num(action['score_contract_only'])}]")
    out.append(head + tail
               + (f"   优先级 {priority['score']:.3f}({priority['pairs']}对)" if priority else ""))

    for item in scores["per_constraint"]:
        if item["source"] != "gold":
            continue
        tag = " [未变动·不计分]" if item["is_background"] else ""
        agree = item.get("vote_agreement")
        vote = f"  票{agree:.2f}" if isinstance(agree, (int, float)) and agree < 1 else ""
        out.append(
            f"  {MARK.get(item['action_status'], '?')} {item['field']:<24}"
            f"{item['tier']:<7}识别{'√' if item['recognized'] else '×'} "
            f"值对{'√' if item['value_match'] else '×'}{vote}{tag}"
        )
        out.append(f"      gold={str(item['gold_value'])[:96]}")
        if item.get("matched_pool_items"):
            out.append(f"      锚定→ {', '.join(str(x)[:40] for x in item['matched_pool_items'][:3])}")
        if item.get("evidence"):
            out.append(f"      {str(item['evidence'])[:170]}")

    findings = [i for i in scores["per_constraint"] if i["source"] != "gold"]
    for item in findings:
        detail = item.get("detail") or {}
        label = "方案自相矛盾" if item["source"] == "inconsistency" else "utterance 隐含要求被违反"
        out.append(f"  ⚠ {label}  [{detail.get('votes', '')}]  权重{item['weight']:.0f} 计为违规")
        for key, name in (("cited_fact", "依据"), ("quote", "原话"),
                          ("requirement", "要求"), ("agent_did", "agent"),
                          ("why_incompatible", "冲突")):
            if detail.get(key):
                out.append(f"      {name:<5}: {str(detail[key])[:150]}")
    if intent.get("extra_constraints"):
        out.append(f"  + agent 多编的约束: {', '.join(intent['extra_constraints'])}")
    if scores.get("judge_summary"):
        out.append(f"  总评: {scores['judge_summary']}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("scores", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--only", nargs="*", default=None, help="只导这些 instance（后四位即可）")
    args = parser.parse_args()

    payload = json.loads(args.scores.read_text(encoding="utf-8"))
    meta = payload.get("metadata") or {}
    v2 = is_v2(payload)
    out: List[str] = [
        f"agent: {meta.get('agent_model')}    judge: {meta.get('judge_model')}"
        + (f" × {meta.get('votes')} 票" if v2 else ""),
        f"数据集: {meta.get('dataset')}",
        f"计分器: {meta.get('scorer')}",
        "",
    ]
    if v2:
        out += [
            "符号: ✓满足 ✗违反 ?证据不足 | 识别=agent 意图复述里提到了 | 值对=值语义正确",
            "      [未变动·不计分]=t0 就有且用户从未动过 | 票x.xx=三票未达成一致",
            "",
        ]

    for row in payload["rows"]:
        iid = str(row["instance_id"])
        if args.only and not any(o in iid for o in args.only):
            continue
        out.append("=" * 118)
        out.append(f"{iid}  t{row['turn_id']}")
        out.append("=" * 118)
        out.append(f"【用户说】 {row.get('user_utterance')}")
        out.append("")
        if v2:
            dump_v2(row, out)
        else:
            fmt_prediction(row, out)
            out.append("")
            out.append("【agent 产出的行程】")
            out.extend(fmt_itinerary((row.get("action_evidence") or {}).get("action") or {})
                       or ["    (空)"])
            out.append("")
            dump_v1(row, out)
        out.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out), encoding="utf-8")
    print(f"{len(out)} 行 -> {args.out}  ({'v2' if v2 else 'v1'} 格式)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
