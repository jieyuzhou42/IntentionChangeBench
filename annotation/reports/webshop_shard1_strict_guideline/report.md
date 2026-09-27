# WebShop shard 001 严格约束评定

范围：8 cases / 40 turns。直接沿用已逐项审核的商品证据，本次按用户新 guideline 重算，无外部模型调用。

1. 商品缺少某项约束的支持证据，该约束记为不满足（0），不再暂无法判定。实际 Gold 和 Agent 使用同一口径。
2. 缺 Gold 时，假设 Gold 满足全部 active constraints；Agent 必须全部满足才 Success。
3. 有 Gold 时继续按 Must → Preferred → Optional 比较满足数量，全部持平成功。某个约束不满足，不自动等于整轮失败；整轮由词典序和有效性/已知 Must 未披露门槛判定。
4. 保留原始 unknown 证据状态供审计，但评分不给满足分；不把未知事实伪称为已知违反。

**Success 25 / Failure 15 / 暂无法判定 0；Action Success = 25/40 = 62.5%。**

计数向量为 (Must, Preferred, Optional)，turn_id 从 0 开始。

| Case | turn_id | Agent | Gold | Gold 来源 | 判定 | Agent 未满足项 |
|---|---:|---|---|---|---|---|
| webshop_goal_00000 | 0 | (1, 1, 4) | (1, 1, 2) | 实际 Gold | success | 无 |
| webshop_goal_00000 | 1 | (1, 2, 2) | (1, 2, 2) | 实际 Gold | success | size |
| webshop_goal_00000 | 2 | (2, 0, 3) | (1, 0, 4) | 实际 Gold | failure | size, insert, double_sided |
| webshop_goal_00000 | 3 | (1, 1, 4) | (3, 1, 4) | 实际 Gold | failure | excluded_themes, style |
| webshop_goal_00318 | 0 | (1, 0, 3) | (1, 0, 4) | 实际 Gold | failure | budget_max, color [证据不足] |
| webshop_goal_00318 | 1 | (3, 0, 4) | (3, 0, 4) | 实际 Gold | success | 无 |
| webshop_goal_00318 | 2 | (1, 1, 4) | (1, 0, 4) | 实际 Gold | success | color [证据不足], water_resistant [证据不足] |
| webshop_goal_01181 | 0 | (1, 0, 7) | (1, 0, 5) | 实际 Gold | success | knee_high, rubber_sole [证据不足], teen_girls [证据不足] |
| webshop_goal_01181 | 1 | (3, 5, 0) | (3, 4, 0) | 实际 Gold | success | knee_high, non_slip [证据不足], rubber_sole [证据不足], teen_girls [证据不足], covered [证据不足] |
| webshop_goal_01181 | 2 | (2, 7, 0) | (2, 7, 0) | 实际 Gold | success | knee_high, non_slip [证据不足], rubber_sole [证据不足], teen_girls [证据不足], covered [证据不足] |
| webshop_goal_01181 | 3 | (1, 6, 0) | (2, 5, 0) | 实际 Gold | failure | size [证据不足], color [证据不足], width, brand_quality [证据不足], knee_high, high_heel [证据不足], teen_girls [证据不足], covered [证据不足], chunky_heel [证据不足] |
| webshop_goal_02449 | 0 | (1, 0, 4) | (1, 0, 4) | 实际 Gold | success | 无 |
| webshop_goal_02449 | 1 | (1, 1, 4) | (2, 2, 4) | 假设全部满足 | failure | finish [证据不足], table_type [证据不足] |
| webshop_goal_02449 | 2 | (2, 2, 4) | (3, 3, 4) | 假设全部满足 | failure | excluded_styles, finish [证据不足] |
| webshop_goal_02449 | 3 | (1, 3, 4) | (1, 4, 3) | 实际 Gold | failure | excluded_styles, finish [证据不足] |
| webshop_goal_02449 | 4 | (2, 5, 1) | (2, 5, 2) | 实际 Gold | failure | coated_steel [证据不足], tempered_glass, steel_frame [证据不足], finish [证据不足] |
| webshop_goal_02449 | 5 | (2, 7, 1) | (2, 7, 1) | 实际 Gold | success | coated_steel [证据不足], tempered_glass, steel_frame [证据不足] |
| webshop_goal_02702 | 0 | (0, 0, 5) | (1, 0, 4) | 实际 Gold | failure | category, grain_free [证据不足], artificial_ingredients [证据不足], low_calorie [证据不足], non_gmo [证据不足] |
| webshop_goal_02702 | 1 | (1, 0, 5) | (0, 0, 4) | 实际 Gold | success | grain_free [证据不足], artificial_ingredients [证据不足], low_calorie [证据不足], non_gmo [证据不足] |
| webshop_goal_02702 | 2 | (2, 0, 8) | (1, 0, 6) | 实际 Gold | success | artificial_ingredients [证据不足] |
| webshop_goal_02702 | 3 | (2, 1, 7) | (2, 1, 8) | 假设全部满足 | failure | artificial_ingredients [证据不足] |
| webshop_goal_02702 | 4 | (2, 1, 7) | (0, 1, 3) | 实际 Gold | success | artificial_ingredients [证据不足] |
| webshop_goal_02702 | 5 | (1, 2, 7) | (0, 1, 3) | 实际 Gold | success | ingredient_exclusions [证据不足], artificial_ingredients [证据不足] |
| webshop_goal_05038 | 0 | (1, 0, 3) | (1, 0, 3) | 假设全部满足 | success | 无 |
| webshop_goal_05038 | 1 | (2, 0, 3) | (2, 0, 3) | 假设全部满足 | success | 无 |
| webshop_goal_05038 | 2 | (2, 1, 2) | (2, 1, 2) | 假设全部满足 | success | 无 |
| webshop_goal_05038 | 3 | (2, 2, 2) | (2, 2, 2) | 假设全部满足 | success | 无 |
| webshop_goal_05038 | 4 | (1, 1, 1) | (4, 3, 2) | 假设全部满足 | failure | budget_max, coverage_min_sqft, heavy_duty [证据不足], outdoor_antenna [证据不足], indoor_antenna [证据不足], cables_included [证据不足] |
| webshop_goal_05038 | 5 | (2, 3, 1) | (3, 3, 2) | 假设全部满足 | failure | heavy_duty [证据不足], indoor_antenna [证据不足] |
| webshop_goal_05038 | 6 | (2, 2, 1) | (2, 5, 2) | 假设全部满足 | failure | budget_max, coverage_min_sqft [证据不足], heavy_duty [证据不足], indoor_antenna [证据不足] |
| webshop_goal_05121 | 0 | (6, 1, 0) | (6, 1, 0) | 实际 Gold | success | 无 |
| webshop_goal_05121 | 1 | (6, 0, 1) | (6, 0, 1) | 实际 Gold | success | 无 |
| webshop_goal_05121 | 2 | (2, 6, 0) | (2, 6, 0) | 实际 Gold | failure | entree_type |
| webshop_goal_05121 | 3 | (2, 5, 3) | (2, 5, 3) | 实际 Gold | success | 无 |
| webshop_goal_05121 | 4 | (2, 5, 4) | (2, 5, 3) | 实际 Gold | success | 无 |
| webshop_goal_06824 | 0 | (1, 0, 3) | (1, 0, 3) | 实际 Gold | success | 无 |
| webshop_goal_06824 | 1 | (1, 0, 2) | (1, 0, 2) | 实际 Gold | success | color |
| webshop_goal_06824 | 2 | (3, 0, 2) | (3, 0, 2) | 实际 Gold | success | budget_max |
| webshop_goal_06824 | 3 | (4, 1, 2) | (4, 1, 2) | 实际 Gold | success | 无 |
| webshop_goal_06824 | 4 | (2, 1, 2) | (2, 3, 2) | 实际 Gold | failure | budget_max, shade_style, pattern |

沿用原约束标注及语义评定，包括 02702 t3 的 artificial_ingredients=true 字段极性歧义：该项仍无满足分，未擅自修改为 no_artificial_ingredients。逐约束原始证据见 results.json。
