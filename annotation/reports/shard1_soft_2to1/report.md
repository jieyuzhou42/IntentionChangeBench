# Shard 1：Must 门槛 + Preferred:Optional = 2:1

全部明确标注的 active Must 满足后，总价值为 2Pa+Oa，满分为 2nP+nO；否则得 0。归一化后对全部轮次等权平均。无 soft 约束时，Must 全通过得 1，否则为 0。实际 Gold 不参与。

沿用前次逐约束证据与判定，未知项不计满足；TravelPlanner 的 14 个优先级缺失/冲突约束实例仍未纳入，结果为该口径下的暂定值。没有额外有效性清零乘数。

| 数据集 | cases | turns | Must 通过 | 满分轮 | 部分分轮 | 零分轮 | 平均分 |
|---|---:|---:|---:|---:|---:|---:|---:|
| webshop | 8 | 40 | 28 | 13 | 15 | 12 | 59.1552% |
| travelplanner | 6 | 32 | 17 | 3 | 12 | 17 | 32.7197% |

| case | turn_id | M满足/总数 | P满足/总数 | O满足/总数 | soft价值 | 满分 | 门槛后得分 | 未通过Must |
|---|---:|---|---|---|---:|---:|---:|---|
| travelplanner_test_0081 | 0 | 3/3 | 0/0 | 0/0 | 0 | 0 | 100.0000% | 无 |
| travelplanner_test_0081 | 1 | 2/3 | 2/3 | 0/0 | 4 | 6 | 0.0000% | day_1_lunch |
| travelplanner_test_0081 | 2 | 1/1 | 2/3 | 4/5 | 8 | 11 | 72.7273% | 无 |
| travelplanner_test_0081 | 3 | 0/0 | 0/1 | 2/8 | 2 | 10 | 20.0000% | 无 |
| travelplanner_test_0081 | 4 | 1/1 | 0/0 | 9/14 | 9 | 14 | 64.2857% | 无 |
| travelplanner_test_0081 | 5 | 0/1 | 1/1 | 9/15 | 11 | 17 | 0.0000% | dining_style |
| travelplanner_test_0126 | 0 | 1/1 | 0/0 | 0/0 | 0 | 0 | 100.0000% | 无 |
| travelplanner_test_0126 | 1 | 3/3 | 4/4 | 0/0 | 8 | 8 | 100.0000% | 无 |
| travelplanner_test_0126 | 2 | 1/1 | 1/3 | 0/0 | 2 | 6 | 33.3333% | 无 |
| travelplanner_test_0126 | 3 | 2/2 | 0/2 | 0/0 | 0 | 4 | 0.0000% | 无 |
| travelplanner_test_0126 | 4 | 2/3 | 0/0 | 0/1 | 0 | 1 | 0.0000% | budget |
| travelplanner_test_0129 | 0 | 0/0 | 0/1 | 0/0 | 0 | 2 | 0.0000% | 无 |
| travelplanner_test_0129 | 1 | 4/4 | 3/3 | 0/1 | 6 | 7 | 85.7143% | 无 |
| travelplanner_test_0129 | 2 | 2/3 | 3/3 | 1/1 | 7 | 7 | 0.0000% | room_type |
| travelplanner_test_0129 | 3 | 1/2 | 5/5 | 0/1 | 10 | 11 | 0.0000% | schedule |
| travelplanner_test_0129 | 4 | 3/4 | 5/5 | 0/1 | 10 | 11 | 0.0000% | schedule |
| travelplanner_test_0129 | 5 | 2/2 | 9/10 | 0/1 | 18 | 21 | 85.7143% | 无 |
| travelplanner_test_0144 | 0 | 1/2 | 0/0 | 0/0 | 0 | 0 | 0.0000% | budget |
| travelplanner_test_0144 | 1 | 4/6 | 0/0 | 0/0 | 0 | 0 | 0.0000% | budget, visiting_city_number |
| travelplanner_test_0144 | 2 | 3/4 | 1/2 | 0/0 | 2 | 4 | 0.0000% | budget |
| travelplanner_test_0357 | 0 | 3/4 | 0/0 | 0/0 | 0 | 0 | 0.0000% | budget |
| travelplanner_test_0357 | 1 | 1/2 | 3/3 | 0/0 | 6 | 6 | 0.0000% | budget |
| travelplanner_test_0357 | 2 | 3/3 | 0/2 | 2/2 | 2 | 6 | 33.3333% | 无 |
| travelplanner_test_0357 | 3 | 1/1 | 4/5 | 2/3 | 10 | 13 | 76.9231% | 无 |
| travelplanner_test_0357 | 4 | 1/1 | 3/4 | 3/4 | 9 | 12 | 75.0000% | 无 |
| travelplanner_test_0873 | 0 | 4/5 | 0/0 | 0/0 | 0 | 0 | 0.0000% | budget |
| travelplanner_test_0873 | 1 | 3/3 | 1/2 | 0/0 | 2 | 4 | 50.0000% | 无 |
| travelplanner_test_0873 | 2 | 1/4 | 3/3 | 0/0 | 6 | 6 | 0.0000% | budget, return_transportation, schedule |
| travelplanner_test_0873 | 3 | 1/2 | 1/1 | 2/2 | 4 | 4 | 0.0000% | budget |
| travelplanner_test_0873 | 4 | 0/1 | 2/3 | 2/2 | 6 | 8 | 0.0000% | dining_style |
| travelplanner_test_0873 | 5 | 1/1 | 1/2 | 3/4 | 5 | 8 | 62.5000% | 无 |
| travelplanner_test_0873 | 6 | 1/1 | 2/2 | 3/4 | 7 | 8 | 87.5000% | 无 |
| webshop_goal_00000 | 0 | 1/1 | 1/1 | 4/4 | 6 | 6 | 100.0000% | 无 |
| webshop_goal_00000 | 1 | 1/1 | 2/2 | 2/3 | 6 | 7 | 85.7143% | 无 |
| webshop_goal_00000 | 2 | 2/3 | 0/0 | 3/5 | 3 | 5 | 0.0000% | insert |
| webshop_goal_00000 | 3 | 1/3 | 1/1 | 4/4 | 6 | 6 | 0.0000% | excluded_themes, style |
| webshop_goal_00318 | 0 | 1/1 | 0/0 | 3/5 | 3 | 5 | 60.0000% | 无 |
| webshop_goal_00318 | 1 | 3/3 | 0/0 | 4/4 | 4 | 4 | 100.0000% | 无 |
| webshop_goal_00318 | 2 | 1/1 | 1/2 | 4/5 | 6 | 9 | 66.6667% | 无 |
| webshop_goal_01181 | 0 | 1/1 | 0/0 | 7/10 | 7 | 10 | 70.0000% | 无 |
| webshop_goal_01181 | 1 | 3/4 | 5/8 | 0/1 | 10 | 17 | 0.0000% | covered |
| webshop_goal_01181 | 2 | 2/2 | 7/11 | 0/1 | 14 | 23 | 60.8696% | 无 |
| webshop_goal_01181 | 3 | 1/3 | 6/12 | 0/1 | 12 | 25 | 0.0000% | brand_quality, high_heel |
| webshop_goal_02449 | 0 | 1/1 | 0/0 | 4/4 | 4 | 4 | 100.0000% | 无 |
| webshop_goal_02449 | 1 | 1/2 | 1/2 | 4/4 | 6 | 8 | 0.0000% | finish |
| webshop_goal_02449 | 2 | 2/3 | 2/3 | 4/4 | 8 | 10 | 0.0000% | excluded_styles |
| webshop_goal_02449 | 3 | 1/1 | 3/5 | 4/4 | 10 | 14 | 71.4286% | 无 |
| webshop_goal_02449 | 4 | 2/2 | 5/6 | 1/4 | 11 | 16 | 68.7500% | 无 |
| webshop_goal_02449 | 5 | 2/2 | 7/7 | 1/4 | 15 | 18 | 83.3333% | 无 |
| webshop_goal_02702 | 0 | 0/1 | 0/0 | 5/9 | 5 | 9 | 0.0000% | category |
| webshop_goal_02702 | 1 | 1/1 | 0/0 | 5/9 | 5 | 9 | 55.5556% | 无 |
| webshop_goal_02702 | 2 | 2/2 | 0/0 | 8/9 | 8 | 9 | 88.8889% | 无 |
| webshop_goal_02702 | 3 | 2/2 | 1/1 | 7/8 | 9 | 10 | 90.0000% | 无 |
| webshop_goal_02702 | 4 | 2/2 | 1/1 | 7/8 | 9 | 10 | 90.0000% | 无 |
| webshop_goal_02702 | 5 | 1/2 | 2/2 | 7/8 | 11 | 12 | 0.0000% | ingredient_exclusions |
| webshop_goal_05038 | 0 | 1/1 | 0/0 | 3/3 | 3 | 3 | 100.0000% | 无 |
| webshop_goal_05038 | 1 | 2/2 | 0/0 | 3/3 | 3 | 3 | 100.0000% | 无 |
| webshop_goal_05038 | 2 | 2/2 | 1/1 | 2/2 | 4 | 4 | 100.0000% | 无 |
| webshop_goal_05038 | 3 | 2/2 | 2/2 | 2/2 | 6 | 6 | 100.0000% | 无 |
| webshop_goal_05038 | 4 | 1/4 | 1/3 | 1/2 | 3 | 8 | 0.0000% | cables_included, indoor_antenna, outdoor_antenna |
| webshop_goal_05038 | 5 | 2/3 | 3/3 | 1/2 | 7 | 8 | 0.0000% | indoor_antenna |
| webshop_goal_05038 | 6 | 2/2 | 2/5 | 1/2 | 5 | 12 | 41.6667% | 无 |
| webshop_goal_05121 | 0 | 6/6 | 1/1 | 0/0 | 2 | 2 | 100.0000% | 无 |
| webshop_goal_05121 | 1 | 6/6 | 0/0 | 1/1 | 1 | 1 | 100.0000% | 无 |
| webshop_goal_05121 | 2 | 2/3 | 6/6 | 0/0 | 12 | 12 | 0.0000% | entree_type |
| webshop_goal_05121 | 3 | 2/2 | 5/5 | 3/3 | 13 | 13 | 100.0000% | 无 |
| webshop_goal_05121 | 4 | 2/2 | 5/5 | 4/4 | 14 | 14 | 100.0000% | 无 |
| webshop_goal_06824 | 0 | 1/1 | 0/0 | 3/3 | 3 | 3 | 100.0000% | 无 |
| webshop_goal_06824 | 1 | 1/1 | 0/0 | 2/3 | 2 | 3 | 66.6667% | 无 |
| webshop_goal_06824 | 2 | 3/3 | 0/0 | 2/3 | 2 | 3 | 66.6667% | 无 |
| webshop_goal_06824 | 3 | 4/4 | 1/1 | 2/2 | 4 | 4 | 100.0000% | 无 |
| webshop_goal_06824 | 4 | 2/3 | 1/3 | 2/2 | 4 | 8 | 0.0000% | budget_max |
