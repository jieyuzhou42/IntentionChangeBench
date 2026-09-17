"""v2 计分: 从 judge 的布尔算分，judge 自己不给任何分数。

相对 human_annotated_pilot.score_judged_turn 的改动，每条都对应实测:

- 背景字段（org/dest/日期/天数/人数）移出主分母。它们从 t0 到结束一个字没变过，
  却占总权重的 37.7%，agent 正确率 79.7% —— 相当于白送三成。单独报 background_echo。
- predicted_extra_constraints 进 precision。旧版收集了但零权重，实测 13 条多编的
  约束一分不扣，过度约束完全免费。
- priority 改成纯代码算，而且只算取舍那几对。实测 high-vs-low 84.3% 看着健康，但
  662 对里 622 对是在问"预算是不是比『出发地是 Charleston』更重要"；真正有信息的
  high-vs-medium 只有 62 对、53.2%，被彻底淹没。
- 新增 change_capture: 本轮 gold_delta 的字段是否全部答对。这是 benchmark 的核心
  问题，旧版五个指标没有一个在回答它。
- inconsistencies / implicit_violations 作为动态约束注入 per_constraint，权重按
  must-have 计。语义是: utterance 里说了就等同于一条硬约束，标注有没有写出来不
  影响它的分量。同时输出不含这部分的 contract_only 口径，便于判断噪声。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

PRIORITY_LEVEL_WEIGHTS = {"high": 3.0, "medium": 2.0, "low": 1.0}
MUST_HAVE_WEIGHT = PRIORITY_LEVEL_WEIGHTS["high"]

# 查询自带、且整条 trajectory 里用户从未改动的字段。
BACKGROUND_FIELDS = (
    "days", "people_number", "org", "dest",
    "visiting_city_number", "start_date", "end_date",
)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def field_tiers(gold_intention: Dict[str, Any]) -> Dict[str, str]:
    priority = (gold_intention or {}).get("priority")
    if not isinstance(priority, dict):
        return {}
    return {
        str(field): level
        for level in ("high", "medium", "low")
        for field in priority.get(level) or []
    }


def background_fields(
    gold_intention: Dict[str, Any],
    touched: Sequence[str],
    baseline_fields: Sequence[str] = BACKGROUND_FIELDS,
) -> set:
    """本轮不计分的约束: t0 就存在、且用户至今一次都没动过的。

    跟队友的 score_changed_only.py 同一条规则（"没变过的约束不计入准确率"），
    但判定依据换成 gold_delta 有没有碰过，而不是"当前值等于 t0 的值" —— A-B-A
    回摆时后者会把用户明明改过两次的字段误排。

    不写死字段名单: 任何 t0 带进来又没人碰的字段都算，用户改过的立刻回到计分范围
    （0031 就把 days 改成 4、人数改成 2）。

    t0 的所有约束按定义都"没变过"，所以 t0 整轮为空，不进主指标 —— 那一轮量的是
    "能不能读懂初始 query"，跟意图变化不是一回事。
    """
    return {str(f) for f in baseline_fields if str(f) not in set(touched)}


TIER_RANK = {"high": 3, "medium": 2, "low": 1}


def priority_concordance(
    gold_intention: Dict[str, Any],
    ranked_fields: Sequence[str],
) -> Optional[Dict[str, Any]]:
    """优先级一致性: 标注里档位更高的字段，agent 有没有排在更低的之前。

    只排除背景字段。那些配对是废题 —— 实测 high↔background 87.3%（n=622）、
    medium↔background 100%（n=37），加起来占全部配对的 82%，问的是"预算是不是比
    『出发地是 Charleston』更重要"，纯稀释。

    非背景的配对全部保留，包括 activity 这类 low 档真可选项: 实测
    high↔optional 只有 37.5%（n=40），是所有组合里最低的一类，把它切掉会丢掉
    最有信息的那部分。按配对类型分别计数，便于分开报告。

    None 表示这一轮的非背景字段全在同一档，没有顺序可判，不计入平均。
    """
    tiers = field_tiers(gold_intention)
    constraints = (gold_intention or {}).get("constraints") or {}
    fields = [f for f in tiers if f in constraints and f not in BACKGROUND_FIELDS]
    position = {str(f): i for i, f in enumerate(ranked_fields or [])}

    pairs = correct = 0
    by_kind: Dict[str, List[int]] = {}
    unranked = sorted({f for f in fields if f not in position})
    for upper in fields:
        for lower in fields:
            if TIER_RANK[tiers[upper]] <= TIER_RANK[tiers[lower]]:
                continue
            if upper not in position or lower not in position:
                continue
            ok = position[upper] < position[lower]
            pairs += 1
            correct += int(ok)
            by_kind.setdefault(f"{tiers[upper]}>{tiers[lower]}", []).append(int(ok))
    if not pairs:
        return None
    return {
        "score": correct / pairs,
        "pairs": pairs,
        "correct": correct,
        "by_kind": {k: sum(v) / len(v) for k, v in by_kind.items()},
        "pairs_by_kind": {k: len(v) for k, v in by_kind.items()},
        "unranked_fields": unranked,
    }


# 旧名字保留，避免调用方改动。
tradeoff_concordance = priority_concordance


def _dynamic_rows(judgment: Dict[str, Any]) -> List[Dict[str, Any]]:
    """把契约外的违规变成 per_constraint 里的条目，好让它走同一套公式。"""
    rows = []
    for item in judgment.get("inconsistencies") or []:
        rows.append({
            "field": "[inconsistency]",
            "weight": MUST_HAVE_WEIGHT,
            "recognized": True,
            "value_match": True,
            "action_status": "violated",
            "source": "inconsistency",
            "evidence": item.get("cited_fact"),
            "detail": item,
        })
    for item in judgment.get("implicit_violations") or []:
        rows.append({
            "field": "[implicit]",
            "weight": MUST_HAVE_WEIGHT,
            "recognized": True,
            "value_match": True,
            "action_status": "violated",
            "source": "implicit",
            "evidence": item.get("quote"),
            "detail": item,
        })
    return rows


def score_turn_v2(
    *,
    gold_intention: Dict[str, Any],
    changed_this_turn: Dict[str, Any],
    agent_intention_prediction: Dict[str, Any],
    judgment: Dict[str, Any],
    touched_fields: Sequence[str] = (),
    baseline_fields: Sequence[str] = BACKGROUND_FIELDS,
) -> Dict[str, Any]:
    constraints = {
        str(f): v for f, v in ((gold_intention or {}).get("constraints") or {}).items()
        if v is not None
    }
    tiers = field_tiers(gold_intention)
    background = background_fields(gold_intention, touched_fields, baseline_fields)

    by_field = {str(r.get("gold_field")): r for r in judgment.get("constraint_judgments") or []
                if isinstance(r, dict)}
    missing = sorted(set(constraints) - set(by_field))
    if missing:
        raise ValueError(f"judge 漏判字段: {missing}")

    per_constraint = []
    for field in constraints:
        row = by_field[field]
        status = str(row.get("action_status") or "unknown").strip().lower()
        per_constraint.append({
            "field": field,
            "gold_value": constraints[field],
            "tier": tiers.get(field, "high"),
            "weight": PRIORITY_LEVEL_WEIGHTS.get(tiers.get(field, "high"), 1.0),
            "is_background": field in background,
            "recognized": bool(row.get("recognized")),
            "value_match": bool(row.get("value_match")),
            "action_status": status,
            "matched_pool_items": row.get("matched_pool_items") or [],
            "evidence": row.get("evidence"),
            "vote_agreement": row.get("vote_agreement"),
            "source": "gold",
        })

    dynamic = _dynamic_rows(judgment)
    scored = per_constraint + dynamic

    # ---- 意图层: 只看非背景字段 ----
    foreground = [r for r in per_constraint if not r["is_background"]]
    background_rows = [r for r in per_constraint if r["is_background"]]
    predicted = set((agent_intention_prediction or {}).get("constraints") or {})
    extras = [str(x) for x in (judgment.get("predicted_extra_constraints") or [])]

    intention = {
        "constraint_recall": _mean([float(r["recognized"]) for r in foreground]) if foreground else None,
        "constraint_value_accuracy": _mean([float(r["value_match"]) for r in foreground]) if foreground else None,
        "constraint_precision": (
            (len(predicted) - len(extras)) / len(predicted) if predicted else 0.0
        ),
        "background_echo": _mean([float(r["value_match"]) for r in background_rows]) if background_rows else None,
        "scored": bool(foreground),
        "foreground_fields": len(foreground),
        "background_fields": len(background_rows),
        "extra_constraints": extras,
    }

    # ---- 变更捕捉: 本轮变了的字段是不是全抓到了 ----
    changed = [f for f, change in (changed_this_turn or {}).items()
               if isinstance(change, dict) and change.get("op") != "reprioritize"]
    caught_map = judgment.get("change_caught") or {}
    caught = [f for f in changed
              if bool(caught_map.get(f)) and by_field.get(f, {}).get("value_match")]
    change_capture = {
        "changed_fields": changed,
        "caught_fields": caught,
        "missed_fields": [f for f in changed if f not in caught],
        "all_caught": (len(caught) == len(changed)) if changed else None,
        "field_rate": (len(caught) / len(changed)) if changed else None,
    }

    # ---- 动作层 ----
    def action_score(rows: Sequence[Dict[str, Any]]) -> float:
        total = sum(r["weight"] for r in rows)
        if not total:
            return 0.0
        return sum(r["weight"] for r in rows if r["action_status"] == "satisfied") / total

    # 硬违规也要两个口径。注入的矛盾条目按 must-have 计权，只要有一条就必然触发，
    # 而 76.6% 的轮次有矛盾 —— 只报合并后的数字，会让人拿它去跟 v1 比，得出
    # "硬违规率暴涨"的错误结论。contract_only 才是跟 v1 同口径的那个。
    max_weight = max([r["weight"] for r in scored], default=1.0)
    hard = [r["field"] for r in scored
            if r["weight"] == max_weight and r["action_status"] == "violated"]
    gold_max = max([r["weight"] for r in per_constraint], default=1.0)
    hard_gold = [r["field"] for r in per_constraint
                 if r["weight"] == gold_max and r["action_status"] == "violated"]

    fg_rows = foreground + dynamic
    fg_max = max([r["weight"] for r in fg_rows], default=1.0)
    hard_fg = [r["field"] for r in fg_rows
               if r["weight"] == fg_max and r["action_status"] == "violated"]

    action = {
        # 主口径: 只算用户动过的约束 + 注入的契约外违规
        "score": action_score(fg_rows) if fg_rows else None,
        "hard_violation": bool(hard_fg) if fg_rows else None,
        "hard_violations": hard_fg,
        # 对照口径
        "score_with_implicit": action_score(scored),
        "score_contract_only": action_score(per_constraint),
        "score_foreground_contract_only": action_score(foreground) if foreground else None,
        "hard_violation_all": bool(hard),
        "hard_violations_all": hard,
        "hard_violation_contract_only": bool(hard_gold),
        "hard_violations_contract_only": hard_gold,
        "inconsistencies": len(judgment.get("inconsistencies") or []),
        "implicit_violations": len(judgment.get("implicit_violations") or []),
        # 锚定不上 = 这条约束退回了"只能读 agent 散文"的状态，等于 v2 的修复没生效。
        "unanchored_fields": [
            r["field"] for r in per_constraint
            if not r["matched_pool_items"] and not r["is_background"]
        ],
    }

    return {
        "intention": intention,
        "change_capture": change_capture,
        "priority": priority_concordance(
            gold_intention,
            ((agent_intention_prediction or {}).get("priority") or {}).get("ranked_fields") or [],
        ),
        "action": action,
        "per_constraint": scored,
        "judge_summary": judgment.get("summary"),
        "judge_votes": judgment.get("votes"),
    }


def aggregate_v2(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    scored = [r for r in rows if r.get("scores")]
    if not scored:
        return {}
    # 没有任何"用户动过"的约束的轮次（t0，以及整轮都在复述旧约束的轮次）退出主指标。
    s = [r["scores"] for r in scored if r["scores"]["intention"]["scored"]]
    skipped = len(scored) - len(s)
    if not s:
        return {}
    turn_capture = [x["change_capture"]["all_caught"] for x in s
                    if x["change_capture"]["all_caught"] is not None]
    field_capture = [x["change_capture"]["field_rate"] for x in s
                     if x["change_capture"]["field_rate"] is not None]
    priorities = [x["priority"]["score"] for x in s if x.get("priority")]
    by_kind: Dict[str, List[float]] = {}
    for x in s:
        for kind, value in ((x.get("priority") or {}).get("by_kind") or {}).items():
            by_kind.setdefault(kind, []).append(value)
    return {
        "instances": len({r["instance_id"] for r in scored}),
        "turns": len(s),
        "turns_seen": len(scored),
        "turns_skipped_unchanged_only": skipped,
        "excluded_unchanged_judgments": sum(x["intention"]["background_fields"] for x in
                                            [r["scores"] for r in scored]),
        # 主指标
        "change_capture_turn": _mean([float(v) for v in turn_capture]),
        "change_capture_field": _mean(field_capture),
        "change_turns": len(turn_capture),
        # 意图
        "constraint_recall": _mean([x["intention"]["constraint_recall"] for x in s]),
        "constraint_value_accuracy": _mean([x["intention"]["constraint_value_accuracy"] for x in s]),
        "constraint_precision": _mean([x["intention"]["constraint_precision"] for x in s]),
        "background_echo": _mean([x["intention"]["background_echo"] for x in s
                                  if x["intention"]["background_echo"] is not None]),
        # 优先级
        "priority_concordance": _mean(priorities),
        "priority_turns": len(priorities),
        "priority_by_kind": {k: _mean(v) for k, v in sorted(by_kind.items())},
        # 动作
        "action_score": _mean([x["action"]["score"] for x in s if x["action"]["score"] is not None]),
        "hard_violation_rate": _mean([float(x["action"]["hard_violation"]) for x in s
                                      if x["action"]["hard_violation"] is not None]),
        "action_with_implicit_all": _mean([x["action"]["score_with_implicit"] for x in s]),
        "action_contract_only_all": _mean([x["action"]["score_contract_only"] for x in s]),
        "hard_violation_rate_all": _mean([float(x["action"]["hard_violation_all"]) for x in s]),
        "inconsistency_turns": _mean([float(bool(x["action"]["inconsistencies"])) for x in s]),
        "implicit_violation_turns": _mean([float(bool(x["action"]["implicit_violations"])) for x in s]),
    }
