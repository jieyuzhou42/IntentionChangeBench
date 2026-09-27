# WebShop shard 001：缺 Gold 时假设全部约束满足

范围：8 cases，40 turns。沿用直接人工语义评定，未调用外部 Judge。

有 Gold：按 Must → Preferred → Optional 比较，全部持平算成功。缺 Gold：按用户要求假设参考方案满足全部 active constraints，Agent 必须全部确认满足；明确违反或证据不足均判 Failure。证据不足不代表事实已证实违反。仍保留有效性和未承认的已知 Must 违反检查。

成功 22，失败 11，暂无法判定 7；可判定轮次成功率 22/33 = 66.67%。这不是 Must 满足率。

## 缺 Gold 的 10 轮

| Case | turn_id（从 0 开始） | 判定 | 未确认满足的约束 |
|---|---:|---|---|
| webshop_goal_02449 | 1 | failure | finish (unknown); table_type (unknown) |
| webshop_goal_02449 | 2 | failure | excluded_styles (violated); finish (unknown) |
| webshop_goal_02702 | 3 | failure | artificial_ingredients (unknown) |
| webshop_goal_05038 | 0 | success | 全部满足 |
| webshop_goal_05038 | 1 | success | 全部满足 |
| webshop_goal_05038 | 2 | success | 全部满足 |
| webshop_goal_05038 | 3 | success | 全部满足 |
| webshop_goal_05038 | 4 | failure | budget_max (violated); coverage_min_sqft (violated); heavy_duty (unknown); outdoor_antenna (unknown); indoor_antenna (unknown); cables_included (unknown) |
| webshop_goal_05038 | 5 | failure | heavy_duty (unknown); indoor_antenna (unknown) |
| webshop_goal_05038 | 6 | failure | budget_max (violated); coverage_min_sqft (unknown); heavy_duty (unknown); indoor_antenna (unknown) |

02702 t3 沿用原评定：`artificial_ingredients=true` 与“无人工成分”的字段语义有歧义，因此该 Optional 未确认满足；按本次严格规则判失败，并非认定商品含人工成分。

其余有 Gold 的 7 轮仍因证据不足暂无法判定；本次只调整缺 Gold 的规则。原报告和逐约束证据保留。

## 全部轮次

| Case | turn_id | 判定 | Gold 来源 | 原因 |
|---|---:|---|---|---|
| webshop_goal_00000 | 0 | success | 实际 Gold | lexicographic_at_least_gold |
| webshop_goal_00000 | 1 | success | 实际 Gold | lexicographic_at_least_gold |
| webshop_goal_00000 | 2 | failure | 实际 Gold | undisclosed_must_violation |
| webshop_goal_00000 | 3 | failure | 实际 Gold | undisclosed_must_violation |
| webshop_goal_00318 | 0 | unscorable | 实际 Gold | insufficient_evidence |
| webshop_goal_00318 | 1 | success | 实际 Gold | lexicographic_at_least_gold |
| webshop_goal_00318 | 2 | success | 实际 Gold | lexicographic_at_least_gold |
| webshop_goal_01181 | 0 | unscorable | 实际 Gold | insufficient_evidence |
| webshop_goal_01181 | 1 | unscorable | 实际 Gold | insufficient_evidence |
| webshop_goal_01181 | 2 | unscorable | 实际 Gold | insufficient_evidence |
| webshop_goal_01181 | 3 | unscorable | 实际 Gold | insufficient_evidence |
| webshop_goal_02449 | 0 | success | 实际 Gold | lexicographic_at_least_gold |
| webshop_goal_02449 | 1 | failure | 假设全部满足 | perfect_gold_constraint_not_verified |
| webshop_goal_02449 | 2 | failure | 假设全部满足 | perfect_gold_constraint_violated |
| webshop_goal_02449 | 3 | unscorable | 实际 Gold | insufficient_evidence |
| webshop_goal_02449 | 4 | unscorable | 实际 Gold | insufficient_evidence |
| webshop_goal_02449 | 5 | success | 实际 Gold | lexicographic_at_least_gold |
| webshop_goal_02702 | 0 | failure | 实际 Gold | lexicographic_below_gold |
| webshop_goal_02702 | 1 | success | 实际 Gold | lexicographic_at_least_gold |
| webshop_goal_02702 | 2 | success | 实际 Gold | lexicographic_at_least_gold |
| webshop_goal_02702 | 3 | failure | 假设全部满足 | perfect_gold_constraint_not_verified |
| webshop_goal_02702 | 4 | success | 实际 Gold | lexicographic_at_least_gold |
| webshop_goal_02702 | 5 | success | 实际 Gold | lexicographic_at_least_gold |
| webshop_goal_05038 | 0 | success | 假设全部满足 | perfect_gold_all_constraints_satisfied |
| webshop_goal_05038 | 1 | success | 假设全部满足 | perfect_gold_all_constraints_satisfied |
| webshop_goal_05038 | 2 | success | 假设全部满足 | perfect_gold_all_constraints_satisfied |
| webshop_goal_05038 | 3 | success | 假设全部满足 | perfect_gold_all_constraints_satisfied |
| webshop_goal_05038 | 4 | failure | 假设全部满足 | perfect_gold_constraint_violated |
| webshop_goal_05038 | 5 | failure | 假设全部满足 | perfect_gold_constraint_not_verified |
| webshop_goal_05038 | 6 | failure | 假设全部满足 | perfect_gold_constraint_violated |
| webshop_goal_05121 | 0 | success | 实际 Gold | lexicographic_at_least_gold |
| webshop_goal_05121 | 1 | success | 实际 Gold | lexicographic_at_least_gold |
| webshop_goal_05121 | 2 | failure | 实际 Gold | undisclosed_must_violation |
| webshop_goal_05121 | 3 | success | 实际 Gold | lexicographic_at_least_gold |
| webshop_goal_05121 | 4 | success | 实际 Gold | lexicographic_at_least_gold |
| webshop_goal_06824 | 0 | success | 实际 Gold | lexicographic_at_least_gold |
| webshop_goal_06824 | 1 | success | 实际 Gold | lexicographic_at_least_gold |
| webshop_goal_06824 | 2 | success | 实际 Gold | lexicographic_at_least_gold |
| webshop_goal_06824 | 3 | success | 实际 Gold | lexicographic_at_least_gold |
| webshop_goal_06824 | 4 | failure | 实际 Gold | undisclosed_must_violation |
