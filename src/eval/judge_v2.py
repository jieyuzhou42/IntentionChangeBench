"""v2 judge: 只输出可核对的事实，不输出任何分数。

跟 human_annotated_pilot.build_judge_prompt 的区别，逐条对应实测出来的问题:

1. 附候选池。旧版 action_evidence 只有 {action}，judge 拿不到 minimum nights /
   maximum occupancy 这类 agent 不会主动写的字段，16/53 轮漏判。
2. 按轮判，不是一次判完整条 instance。旧版一次调用覆盖 7 轮，分数从 t0 的 0.836
   掉到 t6 的 0.680，无法区分"后面的轮次真的更难"和"长输出后半段注意力衰减"。
3. 删掉 priority_order_score。那是个没有 rubric 的 0-1 浮点，64 轮里出现 26 个
   不同取值，却占「意图理解」的一半；而它想测的东西可以直接用 ranked_fields 和
   gold 分层算出来，根本不需要模型。
4. 删掉 no_match_appropriate。实测 64 轮零触发。
5. 新增 matched_pool_item: 把 agent 的自由文本锚定到池子里的结构化记录，这是
   TravelPlanner 版的 WebShop selected_product。
6. 新增 inconsistencies / implicit_violations，允许 judge 报告 gold constraints
   之外的违规。旧版被 `set(by_field) != expected_fields` 锁死，读到了也没地方写 ——
   agent 被要求按 utterance 理解，judge 却只能按 constraints 打分，两边标准不一致。

所有新增项都强制写推导链（依据 + agent 做了什么 + 为什么不相容）。不用自评
severity: 让模型给自己的判断打信心分，它一定会往低了报，指标就软掉了。推导链
写不出来就不许报，而且人花三秒读一遍就能复核。

两层的取证范围必须写清楚，这是**程序性**说明、不是裁定。早期版本只给 Layer 2 写了
"判行程不判散文"，Layer 1 没写，于是 judge 把行程也当成了意图证据: 实测 336 条前景
判定里有 24 条（7.1%）judge 判 recognized=True 而 agent 的预测里根本没有那个字段，
理由引的全是行程（"The itinerary selects F3500974..."）。后果是反向激励 —— 0007 的
transportation 字段，agent 明确写出来反而判错，什么都不写反而判对。

prompt 里不写任何裁定性的规则 —— 不定义"订不到的房源算不算满足房型"这类问题。
写进去等于我们先定好答案再让模型盖章，分数就变成在量我们的规则而不是 agent；
而且改一次就要重跑（改 prompt = 缓存全失效），队友那套也没有这些规则，跨组不可比。

这类歧义靠**分数层**处理，不靠 prompt。judge 每条判定都带 vote_agreement，
分歧本身是可测的事实: 实测 room_type 36 条里有 10 条三票不一致，两次独立采样
之间 31% 的判定翻转。要不要把低一致度的判定降级、怎么降级，是计分决策，属于
score_v2.py —— 那里改一次是免费的，而且原始票数留在结果文件里，任何口径都能
事后重算。
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Sequence

SCHEMA = """{
  "constraint_judgments": [
    {
      "gold_field": "exact gold field name",
      "recognized": true,
      "value_match": true,
      "matched_pool_items": ["every candidate pool item this constraint covers"],
      "action_status": "satisfied | violated | unknown",
      "evidence": "what in the plan or the pool record supports this verdict"
    }
  ],
  "change_caught": {"field_that_changed_this_turn": true},
  "inconsistencies": [
    {
      "cited_fact": "a fact the plan itself states, or a candidate pool field",
      "agent_did": "what the plan does",
      "why_incompatible": "why those cannot both hold"
    }
  ],
  "implicit_violations": [
    {
      "quote": "verbatim span copied from the user utterances",
      "requirement": "what that span requires",
      "agent_did": "what the plan does",
      "why_incompatible": "why that breaks the requirement"
    }
  ],
  "predicted_extra_constraints": ["predicted field the utterances do not support"],
  "summary": "one sentence"
}"""


def build_judge_prompt_v2(
    *,
    domain: str,
    instance_id: str,
    turn_id: int,
    all_utterances: Sequence[str],
    gold_intention: Dict[str, Any],
    changed_this_turn: Dict[str, Any],
    agent_intention_prediction: Dict[str, Any],
    action: Dict[str, Any],
    candidate_pool: Dict[str, List[Dict[str, Any]]],
    sample_index: int = 0,
) -> str:
    payload = {
        "instance_id": instance_id,
        "turn_id": turn_id,
        "user_utterances_in_order": list(all_utterances),
        "gold_constraints": (gold_intention or {}).get("constraints") or {},
        "changed_this_turn": changed_this_turn or {},
        "agent_intention_prediction": agent_intention_prediction or {},
        "agent_action": action or {},
        "candidate_pool": candidate_pool or {},
    }
    prompt = f"""
You are a strict evaluator for a multi-turn intention-change benchmark.
The evaluated agent never saw gold annotations. It was told to infer the user's
cumulative intention from the user utterances alone, so hold it to the
utterances, not only to the gold constraint list.

Report facts, never scores. Every number in the final report is computed from
your booleans by code, so do not grade, rank, or rate anything.

The two layers below take their evidence from different places. Keep them apart:
Layer 1 reads agent_intention_prediction only, Layer 2 reads agent_action and the
candidate pool only. An agent that plans well without saying what it understood,
or says the right thing and then plans something else, must show up as a split
between the two layers rather than being averaged inside one of them.

Layer 1, intention understanding. Read agent_intention_prediction only. For every
gold constraint, decide whether that prediction recognizes the constraint and
whether the value it states is semantically correct. A renamed but clearly
equivalent field inside the prediction counts as recognized; a constraint the
itinerary happens to honour but the prediction never states is not recognized,
and its value cannot match. Later utterances override earlier values.

Layer 2, action compliance. Judge the selected itinerary, not the agent's prose.
- First anchor: for each constraint about booked items, list every candidate_pool
  entry the constraint covers in matched_pool_items, using the pool's exact names.
  The agent writes names loosely; match on meaning. A lodging constraint usually
  covers one stay, but a meal constraint covers every restaurant in the plan, so
  list them all -- an empty list means you could not find any, not that you did
  not look.
- Then judge action_status against those pool records, not against the numbers
  the plan quotes. The records carry fields the agent often omits, such as
  minimum nights, maximum occupancy, price, rating and cuisines.
- unknown means the evidence is genuinely insufficient, not that you are unsure.
- Do not give credit merely because the plan repeats a requirement in prose.

Layer 3, things the gold constraint list does not cover. The gold list is
incomplete by construction; a requirement stated in an utterance binds the agent
whether or not anyone wrote it down.
- inconsistencies: the plan contradicts a fact it states itself, or a field of
  the pool record it selected. The contradiction must be derivable from those two
  sources alone, without appeal to anything outside the payload.
- implicit_violations: an utterance requires something the plan breaks, and no
  gold constraint covers it. quote must be copied verbatim from
  user_utterances_in_order.
- Report an item only if you can state the derivation: the fact or quote it rests
  on, what the plan does, and why the two cannot both hold. If you cannot write
  that chain, leave it out. Do not report restatements of gold constraints you
  already judged in Layer 1, and do not report stylistic preferences.

Return exactly one JSON object:
{SCHEMA}

Judge exactly one turn. constraint_judgments must contain every gold constraint
exactly once and nothing else.

change_caught must contain every key of changed_this_turn exactly once. It asks
whether agent_intention_prediction states that field's new value -- same evidence
scope as Layer 1, so an itinerary that quietly reflects the change while the
prediction stays silent is not caught.

DOMAIN: {domain}
EVAL_PAYLOAD:
{json.dumps(payload, ensure_ascii=False, indent=2, default=str)}
""".strip()
    if sample_index:
        # 投票要的是同一个 prompt 的多次独立采样，但 Client 的缓存键是
        # sha256(model + prompt)，不加区分的话第二三票会直接命中第一票的缓存。
        # 这一行语义上是惰性的，只用来让每一票有自己的缓存位置，重跑时同样免费。
        prompt = f"{prompt}\n\nSAMPLE_INDEX: {sample_index}"
    return prompt


ALLOWED_STATUS = {"satisfied", "violated", "unknown"}


def _norm(value: Any) -> str:
    return " ".join(str(value or "").lower().split())


# 判定文本里反复出现、不带区分力的词。留着会把不同发现的相似度整体拉高。
_STOPWORDS = frozenset("""
a an the and or but for of to in on at by with from that this those these is are was were be been
it its as not no than then so such only also more most can cannot could would should may might will
plan itinerary agent record pool candidate selected stated listed which while however because
""".split())


def _tokens(item: Dict[str, Any]) -> set:
    text = " ".join(str(item.get(k) or "") for k in
                    ("cited_fact", "quote", "requirement", "agent_did", "why_incompatible"))
    words = re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower())
    return {w for w in words if len(w) > 2 and w not in _STOPWORDS}


def _same_finding(left: set, right: set, threshold: float = 0.45) -> bool:
    """同一个发现在不同票里措辞不同 —— 三票可能写成 "The pool record ... has minimum"、
    "... has a minimum"、"The candidate-pool record ..."，逐字比对会判成三个不同发现，
    于是全部达不到票数门槛被丢掉。改用内容词重合度聚类。"""
    if not left or not right:
        return False
    return len(left & right) / len(left | right) >= threshold


def merge_votes(votes: List[Dict[str, Any]], gold_fields: Sequence[str]) -> Dict[str, Any]:
    """多票合并成一份判定。

    布尔和枚举取多数；平票时倒向保守的一侧（action_status 归 unknown，
    recognized/value_match 归 False）。inconsistencies / implicit_violations
    要求至少半数票里出现过同一个发现才保留 —— 只在一票里冒出来的，按噪声处理。
    """
    votes = [v for v in votes if isinstance(v, dict)]
    if not votes:
        return {}
    need = len(votes) / 2.0

    merged_fields = []
    for field in gold_fields:
        rows = []
        for vote in votes:
            for row in vote.get("constraint_judgments") or []:
                if isinstance(row, dict) and str(row.get("gold_field") or "") == field:
                    rows.append(row)
                    break
        if not rows:
            continue
        statuses = [str(r.get("action_status") or "").strip().lower() for r in rows]
        statuses = [s if s in ALLOWED_STATUS else "unknown" for s in statuses]
        winner = max(set(statuses), key=lambda s: (statuses.count(s), s == "unknown"))
        merged_fields.append({
            "gold_field": field,
            "recognized": sum(bool(r.get("recognized")) for r in rows) > need,
            "value_match": sum(bool(r.get("value_match")) for r in rows) > need,
            # 多家餐厅的约束会覆盖多个池子条目，合并各票的并集。
            "matched_pool_items": sorted({
                str(name) for r in rows for name in (r.get("matched_pool_items") or [])
                if str(name).strip()
            }),
            "action_status": winner,
            "evidence": next((r.get("evidence") for r in rows if r.get("evidence")), None),
            "vote_agreement": max(statuses.count(s) for s in set(statuses)) / len(statuses),
        })

    caught: Dict[str, bool] = {}
    for key in {k for v in votes for k in (v.get("change_caught") or {})}:
        hits = sum(bool((v.get("change_caught") or {}).get(key)) for v in votes)
        caught[key] = hits > need

    def merge_findings(name: str) -> List[Dict[str, Any]]:
        # 聚类而不是精确匹配；每一票在同一簇里最多贡献一票，避免一票拆成两句灌票。
        clusters: List[Dict[str, Any]] = []
        for index, vote in enumerate(votes):
            for item in vote.get(name) or []:
                if not isinstance(item, dict):
                    continue
                if not (item.get("agent_did") and item.get("why_incompatible")):
                    continue  # 推导链不完整的不算
                tokens = _tokens(item)
                if not tokens:
                    continue
                for cluster in clusters:
                    if _same_finding(tokens, cluster["tokens"]):
                        cluster["voters"].add(index)
                        cluster["tokens"] |= tokens
                        break
                else:
                    clusters.append({"item": item, "tokens": tokens, "voters": {index}})
        out = []
        for cluster in clusters:
            if len(cluster["voters"]) > need:
                item = dict(cluster["item"])
                item["votes"] = f"{len(cluster['voters'])}/{len(votes)}"
                out.append(item)
        return out

    extras = sorted({str(x) for v in votes for x in (v.get("predicted_extra_constraints") or [])
                     if str(x).strip()})
    return {
        "constraint_judgments": merged_fields,
        "change_caught": caught,
        "inconsistencies": merge_findings("inconsistencies"),
        "implicit_violations": merge_findings("implicit_violations"),
        "predicted_extra_constraints": extras,
        "summary": next((v.get("summary") for v in votes if v.get("summary")), None),
        "votes": len(votes),
    }
