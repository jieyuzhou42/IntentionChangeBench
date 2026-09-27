# Shard 1 动态权重约束满足比例

每轮 R = (wM·Ma + wP·Pa + Oa)/(wM·nM + wP·nP + nO)，wP=nO+1，wM=(nP+1)(nO+1)。主汇总为逐轮 R 的算术平均。

缺少支持证据的约束不计满足分。TravelPlanner 的 69 个 Optional 约束实例本次已补评，未直接将之前未审核项计为失败。Gold 不参与本指标；也不额外以整轮有效性或披露门槛把比例清零。

| 数据集 / 优先级口径 | cases | turns | 逐轮平均 | 总得分 / 总满分 | 汇总比例 |
|---|---:|---:|---:|---:|---:|
| webshop / labeled_only | 8 | 40 | 82.6860% | 1310 / 1658 | 79.0109% |
| travelplanner / labeled_only | 6 | 32 | 70.8498% | 560 / 747 | 74.9665% |
| travelplanner / missing_as_must_conflict_highest | 6 | 32 | 72.8495% | 719 / 931 | 77.2288% |

TravelPlanner 有 14 个约束实例优先级缺失或冲突。labeled_only 仅计算明确分层的约束；另一列是未标注按 Must、冲突取最高优先级的假设结果，不是原始标注事实。所有歧义项都纳入并允许各自取可选层级时，逐轮平均的保守范围为 70.1286%–73.5892%。

原始标注不做语义修订（如 <2 景点与要求两个景点的冲突）；补评的预算沿用既有 benchmark 价格范围，明确未定价的餐食/必需交通仍不计满足。

| 数据集 | case | turn_id | 已标层级总数 M/P/O | 满足 M/P/O | 权重 M/P/O | 得分 / 满分 | 比例（仅明确层级） |
|---|---|---:|---|---|---|---|---:|
| travelplanner | travelplanner_test_0081 | 0 | [3, 0, 0] | [3, 0, 0] | [1, 1, 1] | 3 / 3 | 100.0000% |
| travelplanner | travelplanner_test_0081 | 1 | [3, 3, 0] | [2, 2, 0] | [4, 1, 1] | 10 / 15 | 66.6667% |
| travelplanner | travelplanner_test_0081 | 2 | [1, 3, 5] | [1, 2, 4] | [24, 6, 1] | 40 / 47 | 85.1064% |
| travelplanner | travelplanner_test_0081 | 3 | [0, 1, 8] | [0, 0, 2] | [18, 9, 1] | 2 / 17 | 11.7647% |
| travelplanner | travelplanner_test_0081 | 4 | [1, 0, 14] | [1, 0, 9] | [15, 15, 1] | 24 / 29 | 82.7586% |
| travelplanner | travelplanner_test_0081 | 5 | [1, 1, 15] | [0, 1, 9] | [32, 16, 1] | 25 / 63 | 39.6825% |
| travelplanner | travelplanner_test_0126 | 0 | [1, 0, 0] | [1, 0, 0] | [1, 1, 1] | 1 / 1 | 100.0000% |
| travelplanner | travelplanner_test_0126 | 1 | [3, 4, 0] | [3, 4, 0] | [5, 1, 1] | 19 / 19 | 100.0000% |
| travelplanner | travelplanner_test_0126 | 2 | [1, 3, 0] | [1, 1, 0] | [4, 1, 1] | 5 / 7 | 71.4286% |
| travelplanner | travelplanner_test_0126 | 3 | [2, 2, 0] | [2, 0, 0] | [3, 1, 1] | 6 / 8 | 75.0000% |
| travelplanner | travelplanner_test_0126 | 4 | [3, 0, 1] | [2, 0, 0] | [2, 2, 1] | 4 / 7 | 57.1429% |
| travelplanner | travelplanner_test_0129 | 0 | [0, 1, 0] | [0, 0, 0] | [2, 1, 1] | 0 / 1 | 0.0000% |
| travelplanner | travelplanner_test_0129 | 1 | [4, 3, 1] | [4, 3, 0] | [8, 2, 1] | 38 / 39 | 97.4359% |
| travelplanner | travelplanner_test_0129 | 2 | [3, 3, 1] | [2, 3, 1] | [8, 2, 1] | 23 / 31 | 74.1935% |
| travelplanner | travelplanner_test_0129 | 3 | [2, 5, 1] | [1, 5, 0] | [12, 2, 1] | 22 / 35 | 62.8571% |
| travelplanner | travelplanner_test_0129 | 4 | [4, 5, 1] | [3, 5, 0] | [12, 2, 1] | 46 / 59 | 77.9661% |
| travelplanner | travelplanner_test_0129 | 5 | [2, 10, 1] | [2, 9, 0] | [22, 2, 1] | 62 / 65 | 95.3846% |
| travelplanner | travelplanner_test_0144 | 0 | [2, 0, 0] | [1, 0, 0] | [1, 1, 1] | 1 / 2 | 50.0000% |
| travelplanner | travelplanner_test_0144 | 1 | [6, 0, 0] | [4, 0, 0] | [1, 1, 1] | 4 / 6 | 66.6667% |
| travelplanner | travelplanner_test_0144 | 2 | [4, 2, 0] | [3, 1, 0] | [3, 1, 1] | 10 / 14 | 71.4286% |
| travelplanner | travelplanner_test_0357 | 0 | [4, 0, 0] | [3, 0, 0] | [1, 1, 1] | 3 / 4 | 75.0000% |
| travelplanner | travelplanner_test_0357 | 1 | [2, 3, 0] | [1, 3, 0] | [4, 1, 1] | 7 / 11 | 63.6364% |
| travelplanner | travelplanner_test_0357 | 2 | [3, 2, 2] | [3, 0, 2] | [9, 3, 1] | 29 / 35 | 82.8571% |
| travelplanner | travelplanner_test_0357 | 3 | [1, 5, 3] | [1, 4, 2] | [24, 4, 1] | 42 / 47 | 89.3617% |
| travelplanner | travelplanner_test_0357 | 4 | [1, 4, 4] | [1, 3, 3] | [25, 5, 1] | 43 / 49 | 87.7551% |
| travelplanner | travelplanner_test_0873 | 0 | [5, 0, 0] | [4, 0, 0] | [1, 1, 1] | 4 / 5 | 80.0000% |
| travelplanner | travelplanner_test_0873 | 1 | [3, 2, 0] | [3, 1, 0] | [3, 1, 1] | 10 / 11 | 90.9091% |
| travelplanner | travelplanner_test_0873 | 2 | [4, 3, 0] | [1, 3, 0] | [4, 1, 1] | 7 / 19 | 36.8421% |
| travelplanner | travelplanner_test_0873 | 3 | [2, 1, 2] | [1, 1, 2] | [6, 3, 1] | 11 / 17 | 64.7059% |
| travelplanner | travelplanner_test_0873 | 4 | [1, 3, 2] | [0, 2, 2] | [12, 3, 1] | 8 / 23 | 34.7826% |
| travelplanner | travelplanner_test_0873 | 5 | [1, 2, 4] | [1, 1, 3] | [15, 5, 1] | 23 / 29 | 79.3103% |
| travelplanner | travelplanner_test_0873 | 6 | [1, 2, 4] | [1, 2, 3] | [15, 5, 1] | 28 / 29 | 96.5517% |
| webshop | webshop_goal_00000 | 0 | [1, 1, 4] | [1, 1, 4] | [10, 5, 1] | 19 / 19 | 100.0000% |
| webshop | webshop_goal_00000 | 1 | [1, 2, 3] | [1, 2, 2] | [12, 4, 1] | 22 / 23 | 95.6522% |
| webshop | webshop_goal_00000 | 2 | [3, 0, 5] | [2, 0, 3] | [6, 6, 1] | 15 / 23 | 65.2174% |
| webshop | webshop_goal_00000 | 3 | [3, 1, 4] | [1, 1, 4] | [10, 5, 1] | 19 / 39 | 48.7179% |
| webshop | webshop_goal_00318 | 0 | [1, 0, 5] | [1, 0, 3] | [6, 6, 1] | 9 / 11 | 81.8182% |
| webshop | webshop_goal_00318 | 1 | [3, 0, 4] | [3, 0, 4] | [5, 5, 1] | 19 / 19 | 100.0000% |
| webshop | webshop_goal_00318 | 2 | [1, 2, 5] | [1, 1, 4] | [18, 6, 1] | 28 / 35 | 80.0000% |
| webshop | webshop_goal_01181 | 0 | [1, 0, 10] | [1, 0, 7] | [11, 11, 1] | 18 / 21 | 85.7143% |
| webshop | webshop_goal_01181 | 1 | [4, 8, 1] | [3, 5, 0] | [18, 2, 1] | 64 / 89 | 71.9101% |
| webshop | webshop_goal_01181 | 2 | [2, 11, 1] | [2, 7, 0] | [24, 2, 1] | 62 / 71 | 87.3239% |
| webshop | webshop_goal_01181 | 3 | [3, 12, 1] | [1, 6, 0] | [26, 2, 1] | 38 / 103 | 36.8932% |
| webshop | webshop_goal_02449 | 0 | [1, 0, 4] | [1, 0, 4] | [5, 5, 1] | 9 / 9 | 100.0000% |
| webshop | webshop_goal_02449 | 1 | [2, 2, 4] | [1, 1, 4] | [15, 5, 1] | 24 / 44 | 54.5455% |
| webshop | webshop_goal_02449 | 2 | [3, 3, 4] | [2, 2, 4] | [20, 5, 1] | 54 / 79 | 68.3544% |
| webshop | webshop_goal_02449 | 3 | [1, 5, 4] | [1, 3, 4] | [30, 5, 1] | 49 / 59 | 83.0508% |
| webshop | webshop_goal_02449 | 4 | [2, 6, 4] | [2, 5, 1] | [35, 5, 1] | 96 / 104 | 92.3077% |
| webshop | webshop_goal_02449 | 5 | [2, 7, 4] | [2, 7, 1] | [40, 5, 1] | 116 / 119 | 97.4790% |
| webshop | webshop_goal_02702 | 0 | [1, 0, 9] | [0, 0, 5] | [10, 10, 1] | 5 / 19 | 26.3158% |
| webshop | webshop_goal_02702 | 1 | [1, 0, 9] | [1, 0, 5] | [10, 10, 1] | 15 / 19 | 78.9474% |
| webshop | webshop_goal_02702 | 2 | [2, 0, 9] | [2, 0, 8] | [10, 10, 1] | 28 / 29 | 96.5517% |
| webshop | webshop_goal_02702 | 3 | [2, 1, 8] | [2, 1, 7] | [18, 9, 1] | 52 / 53 | 98.1132% |
| webshop | webshop_goal_02702 | 4 | [2, 1, 8] | [2, 1, 7] | [18, 9, 1] | 52 / 53 | 98.1132% |
| webshop | webshop_goal_02702 | 5 | [2, 2, 8] | [1, 2, 7] | [27, 9, 1] | 52 / 80 | 65.0000% |
| webshop | webshop_goal_05038 | 0 | [1, 0, 3] | [1, 0, 3] | [4, 4, 1] | 7 / 7 | 100.0000% |
| webshop | webshop_goal_05038 | 1 | [2, 0, 3] | [2, 0, 3] | [4, 4, 1] | 11 / 11 | 100.0000% |
| webshop | webshop_goal_05038 | 2 | [2, 1, 2] | [2, 1, 2] | [6, 3, 1] | 17 / 17 | 100.0000% |
| webshop | webshop_goal_05038 | 3 | [2, 2, 2] | [2, 2, 2] | [9, 3, 1] | 26 / 26 | 100.0000% |
| webshop | webshop_goal_05038 | 4 | [4, 3, 2] | [1, 1, 1] | [12, 3, 1] | 16 / 59 | 27.1186% |
| webshop | webshop_goal_05038 | 5 | [3, 3, 2] | [2, 3, 1] | [12, 3, 1] | 34 / 47 | 72.3404% |
| webshop | webshop_goal_05038 | 6 | [2, 5, 2] | [2, 2, 1] | [18, 3, 1] | 43 / 53 | 81.1321% |
| webshop | webshop_goal_05121 | 0 | [6, 1, 0] | [6, 1, 0] | [2, 1, 1] | 13 / 13 | 100.0000% |
| webshop | webshop_goal_05121 | 1 | [6, 0, 1] | [6, 0, 1] | [2, 2, 1] | 13 / 13 | 100.0000% |
| webshop | webshop_goal_05121 | 2 | [3, 6, 0] | [2, 6, 0] | [7, 1, 1] | 20 / 27 | 74.0741% |
| webshop | webshop_goal_05121 | 3 | [2, 5, 3] | [2, 5, 3] | [24, 4, 1] | 71 / 71 | 100.0000% |
| webshop | webshop_goal_05121 | 4 | [2, 5, 4] | [2, 5, 4] | [30, 5, 1] | 89 / 89 | 100.0000% |
| webshop | webshop_goal_06824 | 0 | [1, 0, 3] | [1, 0, 3] | [4, 4, 1] | 7 / 7 | 100.0000% |
| webshop | webshop_goal_06824 | 1 | [1, 0, 3] | [1, 0, 2] | [4, 4, 1] | 6 / 7 | 85.7143% |
| webshop | webshop_goal_06824 | 2 | [3, 0, 3] | [3, 0, 2] | [4, 4, 1] | 14 / 15 | 93.3333% |
| webshop | webshop_goal_06824 | 3 | [4, 1, 2] | [4, 1, 2] | [6, 3, 1] | 29 / 29 | 100.0000% |
| webshop | webshop_goal_06824 | 4 | [3, 3, 2] | [2, 1, 2] | [12, 3, 1] | 29 / 47 | 61.7021% |
