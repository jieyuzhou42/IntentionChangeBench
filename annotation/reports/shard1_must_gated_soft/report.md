# 全部 Must 门槛 × 相对 Gold 的 Soft 分数

Score = 1[全部 active Must 确认满足] × Q；Q=min(1,Sa/Sg)，S=(nO+1)P+O。任一 Must 违反或无支持证据均得 0，明确承认违反也不豁免。

缺 Gold 按 Gold 满足全部 soft 约束；Sg=0 时 Q=1。先逐轮封顶和应用 Must 门槛，再对全部轮次平均，包括所有 0 分轮。沿用逐约束直接评定；没有调用外部 Judge。

TravelPlanner 暂沿用明确标注优先级的口径，14 个缺失/冲突优先级实例不纳入，结果不能视为包含这些歧义项的最终全量分数。该公式未另加整轮有效性乘数；原有效性检查结果仍保留在 JSON evidence 中。

| 数据集 | cases | turns | Must 全通过 | Must 通过率 | 平均最终分 | 满分轮 | 部分分轮 | 零分轮 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| webshop | 8 | 40 | 28 | 70.00% | 67.2300% | 23 | 5 | 12 |
| travelplanner | 6 | 32 | 17 | 53.12% | 43.5601% | 10 | 6 | 16 |

| case | turn_id | Ma/nM | Sa | Sg | Q | 最终分 | 未通过的 Must |
|---|---:|---|---:|---:|---:|---:|---|
| travelplanner_test_0081 | 0 | 3/3 | 0 | 0 | 1.0000 | 1.0000 | 无 |
| travelplanner_test_0081 | 1 | 2/3 | 2 | 3 | 0.6667 | 0.0000 | day_1_lunch |
| travelplanner_test_0081 | 2 | 1/1 | 16 | 23 | 0.6957 | 0.6957 | 无 |
| travelplanner_test_0081 | 3 | 0/0 | 2 | 17 | 0.1176 | 0.1176 | 无 |
| travelplanner_test_0081 | 4 | 1/1 | 9 | 12 | 0.7500 | 0.7500 | 无 |
| travelplanner_test_0081 | 5 | 0/1 | 25 | 14 | 1.0000 | 0.0000 | dining_style |
| travelplanner_test_0126 | 0 | 1/1 | 0 | 0 | 1.0000 | 1.0000 | 无 |
| travelplanner_test_0126 | 1 | 3/3 | 4 | 4 | 1.0000 | 1.0000 | 无 |
| travelplanner_test_0126 | 2 | 1/1 | 1 | 1 | 1.0000 | 1.0000 | 无 |
| travelplanner_test_0126 | 3 | 2/2 | 0 | 1 | 0.0000 | 0.0000 | 无 |
| travelplanner_test_0126 | 4 | 2/3 | 0 | 0 | 1.0000 | 0.0000 | budget |
| travelplanner_test_0129 | 0 | 0/0 | 0 | 0 | 1.0000 | 1.0000 | 无 |
| travelplanner_test_0129 | 1 | 4/4 | 6 | 4 | 1.0000 | 1.0000 | 无 |
| travelplanner_test_0129 | 2 | 2/3 | 7 | 5 | 1.0000 | 0.0000 | room_type |
| travelplanner_test_0129 | 3 | 1/2 | 10 | 10 | 1.0000 | 0.0000 | schedule |
| travelplanner_test_0129 | 4 | 3/4 | 10 | 10 | 1.0000 | 0.0000 | schedule |
| travelplanner_test_0129 | 5 | 2/2 | 18 | 18 | 1.0000 | 1.0000 | 无 |
| travelplanner_test_0144 | 0 | 1/2 | 0 | 0 | 1.0000 | 0.0000 | budget |
| travelplanner_test_0144 | 1 | 4/6 | 0 | 0 | 1.0000 | 0.0000 | budget, visiting_city_number |
| travelplanner_test_0144 | 2 | 3/4 | 1 | 0 | 1.0000 | 0.0000 | budget |
| travelplanner_test_0357 | 0 | 3/4 | 0 | 0 | 1.0000 | 0.0000 | budget |
| travelplanner_test_0357 | 1 | 1/2 | 3 | 2 | 1.0000 | 0.0000 | budget |
| travelplanner_test_0357 | 2 | 3/3 | 2 | 4 | 0.5000 | 0.5000 | 无 |
| travelplanner_test_0357 | 3 | 1/1 | 18 | 18 | 1.0000 | 1.0000 | 无 |
| travelplanner_test_0357 | 4 | 1/1 | 18 | 19 | 0.9474 | 0.9474 | 无 |
| travelplanner_test_0873 | 0 | 4/5 | 0 | 0 | 1.0000 | 0.0000 | budget |
| travelplanner_test_0873 | 1 | 3/3 | 1 | 1 | 1.0000 | 1.0000 | 无 |
| travelplanner_test_0873 | 2 | 1/4 | 3 | 3 | 1.0000 | 0.0000 | budget, return_transportation, schedule |
| travelplanner_test_0873 | 3 | 1/2 | 5 | 5 | 1.0000 | 0.0000 | budget |
| travelplanner_test_0873 | 4 | 0/1 | 8 | 8 | 1.0000 | 0.0000 | dining_style |
| travelplanner_test_0873 | 5 | 1/1 | 8 | 8 | 1.0000 | 1.0000 | 无 |
| travelplanner_test_0873 | 6 | 1/1 | 13 | 14 | 0.9286 | 0.9286 | 无 |
| webshop_goal_00000 | 0 | 1/1 | 9 | 7 | 1.0000 | 1.0000 | 无 |
| webshop_goal_00000 | 1 | 1/1 | 10 | 10 | 1.0000 | 1.0000 | 无 |
| webshop_goal_00000 | 2 | 2/3 | 3 | 4 | 0.7500 | 0.0000 | insert |
| webshop_goal_00000 | 3 | 1/3 | 9 | 9 | 1.0000 | 0.0000 | excluded_themes, style |
| webshop_goal_00318 | 0 | 1/1 | 3 | 4 | 0.7500 | 0.7500 | 无 |
| webshop_goal_00318 | 1 | 3/3 | 4 | 4 | 1.0000 | 1.0000 | 无 |
| webshop_goal_00318 | 2 | 1/1 | 10 | 4 | 1.0000 | 1.0000 | 无 |
| webshop_goal_01181 | 0 | 1/1 | 7 | 5 | 1.0000 | 1.0000 | 无 |
| webshop_goal_01181 | 1 | 3/4 | 10 | 8 | 1.0000 | 0.0000 | covered |
| webshop_goal_01181 | 2 | 2/2 | 14 | 14 | 1.0000 | 1.0000 | 无 |
| webshop_goal_01181 | 3 | 1/3 | 12 | 10 | 1.0000 | 0.0000 | brand_quality, high_heel |
| webshop_goal_02449 | 0 | 1/1 | 4 | 4 | 1.0000 | 1.0000 | 无 |
| webshop_goal_02449 | 1 | 1/2 | 9 | 14 | 0.6429 | 0.0000 | finish |
| webshop_goal_02449 | 2 | 2/3 | 14 | 19 | 0.7368 | 0.0000 | excluded_styles |
| webshop_goal_02449 | 3 | 1/1 | 19 | 23 | 0.8261 | 0.8261 | 无 |
| webshop_goal_02449 | 4 | 2/2 | 26 | 27 | 0.9630 | 0.9630 | 无 |
| webshop_goal_02449 | 5 | 2/2 | 36 | 36 | 1.0000 | 1.0000 | 无 |
| webshop_goal_02702 | 0 | 0/1 | 5 | 4 | 1.0000 | 0.0000 | category |
| webshop_goal_02702 | 1 | 1/1 | 5 | 4 | 1.0000 | 1.0000 | 无 |
| webshop_goal_02702 | 2 | 2/2 | 8 | 6 | 1.0000 | 1.0000 | 无 |
| webshop_goal_02702 | 3 | 2/2 | 16 | 17 | 0.9412 | 0.9412 | 无 |
| webshop_goal_02702 | 4 | 2/2 | 16 | 12 | 1.0000 | 1.0000 | 无 |
| webshop_goal_02702 | 5 | 1/2 | 25 | 12 | 1.0000 | 0.0000 | ingredient_exclusions |
| webshop_goal_05038 | 0 | 1/1 | 3 | 3 | 1.0000 | 1.0000 | 无 |
| webshop_goal_05038 | 1 | 2/2 | 3 | 3 | 1.0000 | 1.0000 | 无 |
| webshop_goal_05038 | 2 | 2/2 | 5 | 5 | 1.0000 | 1.0000 | 无 |
| webshop_goal_05038 | 3 | 2/2 | 8 | 8 | 1.0000 | 1.0000 | 无 |
| webshop_goal_05038 | 4 | 1/4 | 4 | 11 | 0.3636 | 0.0000 | cables_included, indoor_antenna, outdoor_antenna |
| webshop_goal_05038 | 5 | 2/3 | 10 | 11 | 0.9091 | 0.0000 | indoor_antenna |
| webshop_goal_05038 | 6 | 2/2 | 7 | 17 | 0.4118 | 0.4118 | 无 |
| webshop_goal_05121 | 0 | 6/6 | 1 | 1 | 1.0000 | 1.0000 | 无 |
| webshop_goal_05121 | 1 | 6/6 | 1 | 1 | 1.0000 | 1.0000 | 无 |
| webshop_goal_05121 | 2 | 2/3 | 6 | 6 | 1.0000 | 0.0000 | entree_type |
| webshop_goal_05121 | 3 | 2/2 | 23 | 23 | 1.0000 | 1.0000 | 无 |
| webshop_goal_05121 | 4 | 2/2 | 29 | 28 | 1.0000 | 1.0000 | 无 |
| webshop_goal_06824 | 0 | 1/1 | 3 | 3 | 1.0000 | 1.0000 | 无 |
| webshop_goal_06824 | 1 | 1/1 | 2 | 2 | 1.0000 | 1.0000 | 无 |
| webshop_goal_06824 | 2 | 3/3 | 2 | 2 | 1.0000 | 1.0000 | 无 |
| webshop_goal_06824 | 3 | 4/4 | 5 | 5 | 1.0000 | 1.0000 | 无 |
| webshop_goal_06824 | 4 | 2/3 | 5 | 11 | 0.4545 | 0.0000 | budget_max |
