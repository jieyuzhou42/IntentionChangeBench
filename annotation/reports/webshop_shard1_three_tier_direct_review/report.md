# WebShop shard 001：Must → Preferred → Optional

由当前会话 Assistant 补评全部 Optional，未调用外部 Judge。沿用原来的 Must/Preferred 判定和商品有效性、未承认已知 Must 违反门槛；本次仅重算 WebShop。

| Case | 总轮数 | 成功 | 失败 | 暂无法判定 | 可判定轮次成功率 |
|---:|---:|---:|---:|---:|---:|
| 8 | 40 | 18 | 5 | 17 | 18/23 = 78.26% |

Success 当且仅当有效性/披露门槛通过，且 `(m_a,p_a,o_a) >=lex (m_g,p_g,o_g)`。三层全部持平也成功。低层不能弥补高层不足。

未知事实保留区间，比较所有可能取值；若胜负可能变化则暂无法判定。仅当明确是同一商品的同一未知固有属性时，双方共享该未知值并抵消。缺 Gold 不以零满足数代替。

## 与两层指标的变化

- `webshop_goal_00318` t0：success → unscorable。
- `webshop_goal_01181` t0：success → unscorable。

00318 t0：双方 Must=1、Preferred=0。Agent Optional=3–4，Gold=4–5；Agent 超过20美元且粉色未确认，Gold在预算内但魔术贴闭合未确认。可能是平手，也可能 Agent 更少，不能确定。
01181 t0：双方 Must=1、Preferred=0。Agent Optional=7–9，Gold=5–9。Gold 未选尺码及双方部分材质/性能/适用人群未知，不能确认 Agent 不劣于 Gold。

原有5个失败仍失败。原有15个暂无法判定仍保留：10轮缺 Gold，5轮高层属性证据不足；新增加2轮因 Optional 无法确定，合计17轮。

沿用原始人工字段，不擅自修正 stale/冲突标注。例如00000 t1 的 Optional 尺码仍是28x28；05121 t2 的提前出现的鸡肉/牛肉 Must 仍按原标注评定。

## 逐轮结果

t 为0-based turn_id；各向量顺序均为 **(Must, Preferred, Optional)**。区间表示证据不足；不是把未知算作0。

| instance_id | t | Agent (M,P,O) | Gold (M,P,O) | 结果 | 原因 |
|---|---:|---|---|---|---|
| webshop_goal_00000 | 0 | (1, 1, 4) | (1, 1, 2–3) | success | lexicographic_at_least_gold |
| webshop_goal_00000 | 1 | (1, 2, 2) | (1, 2, 2) | success | lexicographic_at_least_gold |
| webshop_goal_00000 | 2 | (2, 0, 3) | (1, 0, 4) | failure | undisclosed_must_violation |
| webshop_goal_00000 | 3 | (1, 1, 4) | (3, 1, 4) | failure | undisclosed_must_violation |
| webshop_goal_00318 | 0 | (1, 0, 3–4) | (1, 0, 4–5) | unscorable | insufficient_evidence |
| webshop_goal_00318 | 1 | (3, 0, 4) | (3, 0, 4) | success | lexicographic_at_least_gold |
| webshop_goal_00318 | 2 | (1, 1–2, 4–5) | (1, 0–1, 4) | success | lexicographic_at_least_gold |
| webshop_goal_01181 | 0 | (1, 0, 7–9) | (1, 0, 5–9) | unscorable | insufficient_evidence |
| webshop_goal_01181 | 1 | (3–4, 5–8, 0) | (3–4, 4–8, 0) | unscorable | insufficient_evidence |
| webshop_goal_01181 | 2 | (2, 7–11, 0) | (2, 7–10, 0) | unscorable | insufficient_evidence |
| webshop_goal_01181 | 3 | (1–3, 6–11, 0) | (2–3, 5–11, 0) | unscorable | insufficient_evidence |
| webshop_goal_02449 | 0 | (1, 0, 4) | (1, 0, 4) | success | lexicographic_at_least_gold |
| webshop_goal_02449 | 1 | (1–2, 1–2, 4) | 缺 Gold | unscorable | missing_gold_action |
| webshop_goal_02449 | 2 | (2, 2–3, 4) | 缺 Gold | unscorable | missing_gold_action |
| webshop_goal_02449 | 3 | (1, 3–4, 4) | (1, 4–5, 3) | unscorable | insufficient_evidence |
| webshop_goal_02449 | 4 | (2, 5–6, 1–3) | (2, 5–6, 2–3) | unscorable | insufficient_evidence |
| webshop_goal_02449 | 5 | (2, 7, 1–3) | (2, 7, 1–3) | success | lexicographic_at_least_gold |
| webshop_goal_02702 | 0 | (0, 0, 5–9) | (1, 0, 4–8) | failure | lexicographic_below_gold |
| webshop_goal_02702 | 1 | (1, 0, 5–9) | (0, 0, 4–9) | success | lexicographic_at_least_gold |
| webshop_goal_02702 | 2 | (2, 0, 8–9) | (1, 0, 6–8) | success | lexicographic_at_least_gold |
| webshop_goal_02702 | 3 | (2, 1, 7–8) | 缺 Gold | unscorable | missing_gold_action |
| webshop_goal_02702 | 4 | (2, 1, 7–8) | (0, 1, 3–8) | success | lexicographic_at_least_gold |
| webshop_goal_02702 | 5 | (1–2, 2, 7–8) | (0–1, 1, 3–8) | success | lexicographic_at_least_gold |
| webshop_goal_05038 | 0 | (1, 0, 3) | 缺 Gold | unscorable | missing_gold_action |
| webshop_goal_05038 | 1 | (2, 0, 3) | 缺 Gold | unscorable | missing_gold_action |
| webshop_goal_05038 | 2 | (2, 1, 2) | 缺 Gold | unscorable | missing_gold_action |
| webshop_goal_05038 | 3 | (2, 2, 2) | 缺 Gold | unscorable | missing_gold_action |
| webshop_goal_05038 | 4 | (1–4, 1, 1–2) | 缺 Gold | unscorable | missing_gold_action |
| webshop_goal_05038 | 5 | (2–3, 3, 1–2) | 缺 Gold | unscorable | missing_gold_action |
| webshop_goal_05038 | 6 | (2, 2–4, 1–2) | 缺 Gold | unscorable | missing_gold_action |
| webshop_goal_05121 | 0 | (6, 1, 0) | (6, 1, 0) | success | lexicographic_at_least_gold |
| webshop_goal_05121 | 1 | (6, 0, 1) | (6, 0, 1) | success | lexicographic_at_least_gold |
| webshop_goal_05121 | 2 | (2, 6, 0) | (2, 6, 0) | failure | undisclosed_must_violation |
| webshop_goal_05121 | 3 | (2, 5, 3) | (2, 5, 3) | success | lexicographic_at_least_gold |
| webshop_goal_05121 | 4 | (2, 5, 4) | (2, 5, 3) | success | lexicographic_at_least_gold |
| webshop_goal_06824 | 0 | (1, 0, 3) | (1, 0, 3) | success | lexicographic_at_least_gold |
| webshop_goal_06824 | 1 | (1, 0, 2) | (1, 0, 2) | success | lexicographic_at_least_gold |
| webshop_goal_06824 | 2 | (3, 0, 2) | (3, 0, 2) | success | lexicographic_at_least_gold |
| webshop_goal_06824 | 3 | (4, 1, 2) | (4, 1, 2) | success | lexicographic_at_least_gold |
| webshop_goal_06824 | 4 | (2, 1, 2) | (2, 3, 2) | failure | undisclosed_must_violation |

逐约束状态、证据、差值上下界及旧结果见 `results.json`；可复现逻辑及本会话 Optional 判定见 `scripts/eval_webshop_three_tier_direct_review.py`。
02449 t5 的钢材属性虽未知，但双方是同一47.5英寸商品，材质真值相同，故该未知项可抵消；没有将“metal”擅自判成“steel”。
