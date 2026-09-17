#!/usr/bin/env python3
"""把 v2 judge 报出来的契约外违规，整理成标注补漏清单。

为什么需要这一步:

打分器只遍历 gold constraints，`set(by_field) != expected_fields` 会直接抛错，
所以没写进 constraints 的要求在计分里是不存在的 —— agent 违反了不扣分，遵守了也
不加分。我们自己的数据里就有: 0003 t1 的 utterance 说"16 号到了之后和 17 号才是
有效游玩时间"，但 schedule 没进 constraints，agent 把 The Gateway Arch 排在了要开
12 小时 43 分车的那天，judge 判的 11 个字段没有一条涉及这件事。

v2 让 judge 可以报这类问题（implicit_violations / inconsistencies）并计入分数，
但那条轨道靠 LLM 判断，不如契约内可复现。所以这里做反馈闭环: 反复被报出来的要求，
补进 gold constraints —— 一旦补进去就从"隐含"变成"契约内"，回到完全可复现的轨道。
契约会随着跑越来越完整，而不是停在标注那天的水平。

用法:
    python annotation/tools/audit_annotation.py V2_SCORES.json
    python annotation/tools/audit_annotation.py V2_SCORES.json --dataset DATA.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


def _norm(value: Any) -> str:
    return " ".join(str(value or "").lower().split())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("scores", type=Path, help="travel_eval_v2.py 的输出")
    parser.add_argument("--dataset", type=Path, default=None,
                        help="给出的话会顺带列出该轮已有的 constraint 字段，方便判断补哪个")
    parser.add_argument("--min-votes", type=int, default=2,
                        help="至少多少票报过才列出来")
    args = parser.parse_args()

    payload = json.loads(args.scores.read_text(encoding="utf-8"))
    rows = payload.get("rows") or []
    instances = {}
    if args.dataset and args.dataset.exists():
        instances = {str(i["instance_id"]): i
                     for i in json.loads(args.dataset.read_text(encoding="utf-8"))}

    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        judgment = row.get("judgment") or {}
        for kind in ("implicit_violations", "inconsistencies"):
            for item in judgment.get(kind) or []:
                votes = item.get("votes") or ""
                got = int(re.match(r"(\d+)", str(votes)).group(1)) if re.match(r"(\d+)", str(votes)) else 0
                if got < args.min_votes:
                    continue
                buckets[kind].append({
                    "instance_id": row["instance_id"],
                    "turn_id": row["turn_id"],
                    "votes": votes,
                    **item,
                })

    total = sum(len(v) for v in buckets.values())
    print(f"{len(rows)} 轮里共 {total} 条契约外发现（≥{args.min_votes} 票）\n")

    for kind, label in (("implicit_violations", "utterance 里说了但 constraints 没有"),
                        ("inconsistencies", "方案自相矛盾 / 违背候选池事实")):
        items = buckets.get(kind) or []
        print("=" * 96)
        print(f"{label}    {len(items)} 条")
        print("=" * 96)
        if not items:
            print("  （无）\n")
            continue
        by_turn: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
        for item in items:
            by_turn[(item["instance_id"], item["turn_id"])].append(item)
        for (iid, tid), group in sorted(by_turn.items()):
            print(f"\n  {iid} t{tid}")
            instance = instances.get(iid)
            if instance:
                turn = next((t for t in instance["turns"]
                             if int(t.get("turn_id", -1)) == int(tid)), None)
                if turn:
                    print(f"     utterance: {str(turn.get('user_utterance'))[:110]}")
                    fields = sorted((turn.get("gold_current_intention") or {}).get("constraints") or {})
                    print(f"     已有字段 : {', '.join(fields)}")
            for item in group:
                print(f"     ── [{item['votes']}]")
                if item.get("quote"):
                    print(f"        原话  : \"{str(item['quote'])[:100]}\"")
                if item.get("cited_fact"):
                    print(f"        依据  : {str(item['cited_fact'])[:100]}")
                if item.get("requirement"):
                    print(f"        要求  : {str(item['requirement'])[:100]}")
                print(f"        agent : {str(item.get('agent_did'))[:100]}")
                print(f"        冲突  : {str(item.get('why_incompatible'))[:110]}")
        print()

    # 同一个要求在多轮反复出现，最值得补进 constraints。
    repeated: Dict[str, List[str]] = defaultdict(list)
    for item in buckets.get("implicit_violations") or []:
        repeated[_norm(item.get("requirement"))].append(f"{item['instance_id']} t{item['turn_id']}")
    recurring = {k: v for k, v in repeated.items() if len(v) > 1}
    if recurring:
        print("=" * 96)
        print("跨轮重复出现的要求 —— 优先补进 gold constraints")
        print("=" * 96)
        for requirement, where in sorted(recurring.items(), key=lambda kv: -len(kv[1])):
            print(f"  {len(where)} 次  {requirement[:88]}")
            print(f"         {', '.join(where)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
