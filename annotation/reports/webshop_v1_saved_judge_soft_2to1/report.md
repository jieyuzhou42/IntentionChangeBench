# WebShop v1：沿用文件 Judge，仅替换为 Must + soft 2:1 公式

输入：annotation/data/webshop_shard-1-5_v1.json；Agent 与原 Judge 均为 us.openai.gpt-5.6-sol。

按用户明确选择，完全保留文件内逐约束 action_status，不重新判断商品，也不调用外部 Judge。以 gold_intention 的 active constraints 和优先级计数。全部 Must satisfied 才得分，否则0；通过后 (2Pa+Oa)/(2nP+nO)。unknown/violated 均不计满足。无 soft 约束时，Must 全满足为1。全体轮次等权平均，包含0分轮。

该文件包含50cases/288turns，其中44个人工标注cases/246turns与之前范围对应，另6个source_gold cases/42turns。这里的结果与之前直接复核商品证据的结果判定来源不同，不能把分数差异全部归因于Agent能力。

| 范围 | cases | turns | Must全通过 | 满分轮 | 部分分轮 | 零分轮 | 平均分 |
|---|---:|---:|---:|---:|---:|---:|---:|
| all_50_cases | 50 | 288 | 128 | 22 | 103 | 163 | 29.2677% |
| human_annotated_44_cases | 44 | 246 | 108 | 20 | 86 | 140 | 29.1812% |
| source_gold_6_cases | 6 | 42 | 20 | 2 | 17 | 23 | 29.7747% |
| human shard 1 | 8 | 40 | 16 | 0 | 15 | 25 | 20.6663% |
| human shard 2 | 9 | 45 | 21 | 3 | 18 | 24 | 33.0291% |
| human shard 3 | 9 | 47 | 16 | 4 | 12 | 31 | 23.4051% |
| human shard 4 | 8 | 44 | 19 | 6 | 12 | 26 | 30.9497% |
| human shard 5 | 10 | 70 | 36 | 7 | 29 | 34 | 34.3398% |

额外source_gold cases：webshop_goal_00219, webshop_goal_00333, webshop_goal_01038, webshop_goal_01581, webshop_goal_03784, webshop_goal_04266

| case | turn | shard | M满足/总数 | P满足/总数 | O满足/总数 | soft值/满分 | 最终分 | 未通过Must |
|---|---:|---|---|---|---|---|---:|---|
| webshop_goal_00000 | 0 | 1 | 1/1 | 0/1 | 2/4 | 2/6 | 33.3333% | 无 |
| webshop_goal_00000 | 1 | 1 | 1/1 | 0/2 | 1/3 | 1/7 | 14.2857% | 无 |
| webshop_goal_00000 | 2 | 1 | 0/3 | 0/0 | 1/5 | 1/5 | 0.0000% | category, color, insert |
| webshop_goal_00000 | 3 | 1 | 1/2 | 1/1 | 2/4 | 4/6 | 0.0000% | style |
| webshop_goal_00042 | 0 | 5 | 2/8 | 0/0 | 0/0 | 0/0 | 0.0000% | color, size, machine_wash, wash_cold, dry_clean, tumble_dry |
| webshop_goal_00042 | 1 | 5 | 1/1 | 0/0 | 6/7 | 6/7 | 85.7143% | 无 |
| webshop_goal_00042 | 2 | 5 | 1/2 | 0/0 | 5/6 | 5/6 | 0.0000% | color |
| webshop_goal_00042 | 3 | 5 | 2/2 | 0/1 | 4/6 | 4/8 | 50.0000% | 无 |
| webshop_goal_00042 | 4 | 5 | 2/2 | 1/2 | 4/6 | 6/10 | 60.0000% | 无 |
| webshop_goal_00042 | 5 | 5 | 2/2 | 1/3 | 2/5 | 4/11 | 36.3636% | 无 |
| webshop_goal_00042 | 6 | 5 | 2/2 | 1/4 | 2/5 | 4/13 | 30.7692% | 无 |
| webshop_goal_00074 | 0 | 5 | 1/6 | 0/0 | 0/0 | 0/0 | 0.0000% | category, color, size, machine_washable, living_room |
| webshop_goal_00074 | 1 | 5 | 0/2 | 0/0 | 0/5 | 0/5 | 0.0000% | category, light_control |
| webshop_goal_00074 | 2 | 5 | 0/3 | 0/1 | 0/3 | 0/5 | 0.0000% | category, budget_max, color |
| webshop_goal_00074 | 3 | 5 | 0/2 | 1/3 | 0/3 | 2/9 | 0.0000% | category, panel_count |
| webshop_goal_00074 | 4 | 5 | 4/4 | 1/2 | 2/3 | 4/7 | 57.1429% | 无 |
| webshop_goal_00074 | 5 | 5 | 0/2 | 1/5 | 0/3 | 2/13 | 0.0000% | category, hanging_style |
| webshop_goal_00074 | 6 | 5 | 2/2 | 4/5 | 2/3 | 10/13 | 76.9231% | 无 |
| webshop_goal_00100 | 0 | 5 | 4/4 | 0/0 | 0/0 | 0/0 | 100.0000% | 无 |
| webshop_goal_00100 | 1 | 5 | 1/1 | 0/0 | 3/3 | 3/3 | 100.0000% | 无 |
| webshop_goal_00100 | 2 | 5 | 2/2 | 0/0 | 3/3 | 3/3 | 100.0000% | 无 |
| webshop_goal_00100 | 3 | 5 | 2/2 | 0/1 | 3/3 | 3/5 | 60.0000% | 无 |
| webshop_goal_00100 | 4 | 5 | 2/2 | 1/2 | 3/3 | 5/7 | 71.4286% | 无 |
| webshop_goal_00100 | 5 | 5 | 2/2 | 0/3 | 2/2 | 2/8 | 25.0000% | 无 |
| webshop_goal_00100 | 6 | 5 | 2/2 | 3/3 | 2/2 | 8/8 | 100.0000% | 无 |
| webshop_goal_00209 | 0 | 3 | 4/10 | 0/0 | 0/0 | 0/0 | 0.0000% | category, hand_wash, stretch_fabric, polyester_spandex, teen_girls, daily_wear |
| webshop_goal_00209 | 1 | 3 | 3/3 | 1/5 | 1/2 | 3/12 | 25.0000% | 无 |
| webshop_goal_00209 | 2 | 3 | 3/3 | 5/8 | 0/0 | 10/16 | 62.5000% | 无 |
| webshop_goal_00209 | 3 | 3 | 0/1 | 0/8 | 0/1 | 0/17 | 0.0000% | category |
| webshop_goal_00209 | 4 | 3 | 2/3 | 3/7 | 1/2 | 7/16 | 0.0000% | style |
| webshop_goal_00219 | 0 | source_gold | 5/7 | 0/0 | 0/0 | 0/0 | 0.0000% | category, easy_clean |
| webshop_goal_00219 | 1 | source_gold | 5/5 | 0/0 | 1/3 | 1/3 | 33.3333% | 无 |
| webshop_goal_00219 | 2 | source_gold | 1/1 | 3/4 | 1/3 | 7/11 | 63.6364% | 无 |
| webshop_goal_00219 | 3 | source_gold | 2/2 | 4/4 | 1/3 | 9/11 | 81.8182% | 无 |
| webshop_goal_00219 | 4 | source_gold | 3/3 | 3/4 | 1/3 | 7/11 | 63.6364% | 无 |
| webshop_goal_00219 | 5 | source_gold | 2/2 | 4/5 | 2/3 | 10/13 | 76.9231% | 无 |
| webshop_goal_00219 | 6 | source_gold | 2/2 | 5/6 | 2/3 | 12/15 | 80.0000% | 无 |
| webshop_goal_00273 | 0 | 2 | 0/7 | 0/0 | 0/0 | 0/0 | 0.0000% | category, budget_max, color, size, slim_fit, button_closure, classic_fit |
| webshop_goal_00273 | 1 | 2 | 0/3 | 0/0 | 0/5 | 0/5 | 0.0000% | category, color, style |
| webshop_goal_00273 | 2 | 2 | 0/4 | 0/1 | 0/4 | 0/6 | 0.0000% | category, budget_max, color, size |
| webshop_goal_00273 | 3 | 2 | 1/1 | 1/4 | 2/4 | 4/12 | 33.3333% | 无 |
| webshop_goal_00277 | 0 | 3 | 3/3 | 0/0 | 0/0 | 0/0 | 100.0000% | 无 |
| webshop_goal_00277 | 1 | 3 | 3/3 | 1/1 | 0/1 | 2/3 | 66.6667% | 无 |
| webshop_goal_00277 | 2 | 3 | 1/3 | 3/3 | 0/0 | 6/6 | 0.0000% | u shape, spoon shape |
| webshop_goal_00277 | 3 | 3 | 4/6 | 2/2 | 0/0 | 4/4 | 0.0000% | storage_case, spoon shape |
| webshop_goal_00277 | 4 | 3 | 1/2 | 3/6 | 0/1 | 6/13 | 0.0000% | color |
| webshop_goal_00318 | 0 | 1 | 0/1 | 0/0 | 0/5 | 0/5 | 0.0000% | category |
| webshop_goal_00318 | 1 | 1 | 3/3 | 0/0 | 3/4 | 3/4 | 75.0000% | 无 |
| webshop_goal_00318 | 2 | 1 | 1/1 | 1/2 | 4/5 | 6/9 | 66.6667% | 无 |
| webshop_goal_00331 | 0 | 5 | 3/4 | 0/0 | 0/0 | 0/0 | 0.0000% | quick_release |
| webshop_goal_00331 | 1 | 5 | 1/1 | 0/0 | 2/3 | 2/3 | 66.6667% | 无 |
| webshop_goal_00331 | 2 | 5 | 0/2 | 0/0 | 0/3 | 0/3 | 0.0000% | category, load_capacity_min |
| webshop_goal_00331 | 3 | 5 | 2/2 | 1/1 | 2/2 | 4/4 | 100.0000% | 无 |
| webshop_goal_00331 | 4 | 5 | 2/2 | 1/1 | 1/2 | 3/4 | 75.0000% | 无 |
| webshop_goal_00331 | 5 | 5 | 1/2 | 2/2 | 1/2 | 5/6 | 0.0000% | camera_quick_release_plate |
| webshop_goal_00331 | 6 | 5 | 2/3 | 2/3 | 1/2 | 5/8 | 0.0000% | weight_max_lb |
| webshop_goal_00333 | 0 | source_gold | 3/4 | 0/0 | 0/0 | 0/0 | 0.0000% | aluminum_alloy |
| webshop_goal_00333 | 1 | source_gold | 1/1 | 0/0 | 2/3 | 2/3 | 66.6667% | 无 |
| webshop_goal_00333 | 2 | source_gold | 1/1 | 0/0 | 2/3 | 2/3 | 66.6667% | 无 |
| webshop_goal_00333 | 3 | source_gold | 2/3 | 0/0 | 2/3 | 2/3 | 0.0000% | construction_material |
| webshop_goal_00333 | 4 | source_gold | 2/2 | 0/2 | 1/2 | 1/6 | 16.6667% | 无 |
| webshop_goal_00333 | 5 | source_gold | 0/2 | 1/3 | 0/2 | 2/8 | 0.0000% | category, optical_quality |
| webshop_goal_00333 | 6 | source_gold | 1/2 | 1/4 | 0/2 | 2/10 | 0.0000% | category |
| webshop_goal_00611 | 0 | 4 | 3/4 | 0/0 | 0/0 | 0/0 | 0.0000% | easy_install |
| webshop_goal_00611 | 1 | 4 | 1/1 | 1/1 | 0/0 | 2/2 | 100.0000% | 无 |
| webshop_goal_00611 | 2 | 4 | 0/2 | 1/2 | 0/0 | 2/4 | 0.0000% | category, style |
| webshop_goal_00611 | 3 | 4 | 0/5 | 1/2 | 0/0 | 2/4 | 0.0000% | category, compatibility, setup, waterproof_rating, style |
| webshop_goal_00611 | 4 | 4 | 2/2 | 0/5 | 0/0 | 0/10 | 0.0000% | 无 |
| webshop_goal_00679 | 0 | 4 | 4/7 | 0/0 | 0/0 | 0/0 | 0.0000% | light_weight, high_resolution, easy_carry |
| webshop_goal_00679 | 1 | 4 | 2/2 | 0/0 | 2/5 | 2/5 | 40.0000% | 无 |
| webshop_goal_00679 | 2 | 4 | 0/2 | 0/1 | 0/5 | 0/7 | 0.0000% | category, material |
| webshop_goal_00679 | 3 | 4 | 0/2 | 0/2 | 0/5 | 0/9 | 0.0000% | category, visual_style |
| webshop_goal_00679 | 4 | 4 | 2/2 | 2/4 | 1/4 | 5/12 | 41.6667% | 无 |
| webshop_goal_00679 | 5 | 4 | 1/3 | 3/4 | 1/3 | 7/11 | 0.0000% | visual_style, light_weight |
| webshop_goal_00704 | 0 | 5 | 4/5 | 0/0 | 0/0 | 0/0 | 0.0000% | gold_plated |
| webshop_goal_00704 | 1 | 5 | 2/2 | 0/0 | 2/3 | 2/3 | 66.6667% | 无 |
| webshop_goal_00704 | 2 | 5 | 1/2 | 1/1 | 2/3 | 4/5 | 0.0000% | installation_features |
| webshop_goal_00704 | 3 | 5 | 2/2 | 1/2 | 2/3 | 4/7 | 57.1429% | 无 |
| webshop_goal_00704 | 4 | 5 | 2/2 | 2/3 | 2/3 | 6/9 | 66.6667% | 无 |
| webshop_goal_00704 | 5 | 5 | 0/3 | 0/3 | 0/2 | 0/8 | 0.0000% | category, size, installation_features |
| webshop_goal_00704 | 6 | 5 | 0/2 | 0/4 | 0/2 | 0/10 | 0.0000% | category, variant_verification |
| webshop_goal_00916 | 0 | 4 | 3/4 | 0/0 | 0/0 | 0/0 | 0.0000% | storage_space |
| webshop_goal_00916 | 1 | 4 | 0/2 | 0/0 | 0/3 | 0/3 | 0.0000% | category, storage_type |
| webshop_goal_00916 | 2 | 4 | 3/3 | 0/0 | 2/2 | 2/2 | 100.0000% | 无 |
| webshop_goal_00916 | 3 | 4 | 2/2 | 1/1 | 2/2 | 4/4 | 100.0000% | 无 |
| webshop_goal_00916 | 4 | 4 | 1/2 | 2/2 | 2/3 | 6/7 | 0.0000% | weight_capacity_min |
| webshop_goal_00956 | 0 | 5 | 5/6 | 0/0 | 0/0 | 0/0 | 0.0000% | hand_crafted |
| webshop_goal_00956 | 1 | 5 | 0/2 | 0/0 | 0/5 | 0/5 | 0.0000% | category, chocolate_type |
| webshop_goal_00956 | 2 | 5 | 2/2 | 0/0 | 4/5 | 4/5 | 80.0000% | 无 |
| webshop_goal_00956 | 3 | 5 | 2/2 | 0/1 | 4/5 | 4/7 | 57.1429% | 无 |
| webshop_goal_00956 | 4 | 5 | 1/1 | 1/2 | 4/5 | 6/9 | 66.6667% | 无 |
| webshop_goal_00956 | 5 | 5 | 1/2 | 0/1 | 4/5 | 4/7 | 0.0000% | gift_presentation |
| webshop_goal_00956 | 6 | 5 | 2/3 | 0/1 | 4/5 | 4/7 | 0.0000% | chocolate_type |
| webshop_goal_01038 | 0 | source_gold | 3/5 | 0/0 | 0/0 | 0/0 | 0.0000% | ultra_hd, stereo_sound |
| webshop_goal_01038 | 1 | source_gold | 2/2 | 0/0 | 2/4 | 2/4 | 50.0000% | 无 |
| webshop_goal_01038 | 2 | source_gold | 2/2 | 1/1 | 2/4 | 4/6 | 66.6667% | 无 |
| webshop_goal_01038 | 3 | source_gold | 2/2 | 1/1 | 2/4 | 4/6 | 66.6667% | 无 |
| webshop_goal_01038 | 4 | source_gold | 2/2 | 0/2 | 0/3 | 0/7 | 0.0000% | 无 |
| webshop_goal_01038 | 5 | source_gold | 2/2 | 2/2 | 1/3 | 5/7 | 71.4286% | 无 |
| webshop_goal_01038 | 6 | source_gold | 2/2 | 2/2 | 1/3 | 5/7 | 71.4286% | 无 |
| webshop_goal_01181 | 0 | 1 | 1/1 | 0/0 | 5/10 | 5/10 | 50.0000% | 无 |
| webshop_goal_01181 | 1 | 1 | 0/4 | 0/8 | 0/1 | 0/17 | 0.0000% | category, ankle_strap, covered, chunky_heel |
| webshop_goal_01181 | 2 | 1 | 2/2 | 5/11 | 0/1 | 10/23 | 43.4783% | 无 |
| webshop_goal_01181 | 3 | 1 | 0/3 | 0/12 | 0/1 | 0/25 | 0.0000% | category, brand_quality, high_heel |
| webshop_goal_01237 | 0 | 3 | 3/4 | 1/1 | 0/0 | 2/2 | 0.0000% | clear_glass |
| webshop_goal_01237 | 1 | 3 | 1/2 | 0/0 | 4/4 | 4/4 | 0.0000% | category |
| webshop_goal_01237 | 2 | 3 | 0/2 | 0/3 | 0/1 | 0/7 | 0.0000% | category, seeded glass |
| webshop_goal_01237 | 3 | 3 | 0/2 | 0/2 | 0/2 | 0/6 | 0.0000% | category, color |
| webshop_goal_01237 | 4 | 3 | 0/1 | 0/3 | 0/4 | 0/10 | 0.0000% | budget_max |
| webshop_goal_01259 | 0 | 2 | 3/5 | 0/0 | 0/0 | 0/0 | 0.0000% | category, easy_clean |
| webshop_goal_01259 | 1 | 2 | 3/3 | 0/0 | 1/2 | 1/2 | 50.0000% | 无 |
| webshop_goal_01259 | 2 | 2 | 2/2 | 1/1 | 1/2 | 3/4 | 75.0000% | 无 |
| webshop_goal_01259 | 3 | 2 | 2/2 | 1/2 | 1/2 | 3/6 | 50.0000% | 无 |
| webshop_goal_01259 | 4 | 2 | 2/2 | 1/2 | 1/2 | 3/6 | 50.0000% | 无 |
| webshop_goal_01259 | 5 | 2 | 0/1 | 0/2 | 0/3 | 0/7 | 0.0000% | category |
| webshop_goal_01268 | 0 | 5 | 3/5 | 0/0 | 0/0 | 0/0 | 0.0000% | color, steel_frame |
| webshop_goal_01268 | 1 | 5 | 2/2 | 0/0 | 1/4 | 1/4 | 25.0000% | 无 |
| webshop_goal_01268 | 2 | 5 | 1/2 | 1/1 | 2/4 | 4/6 | 0.0000% | color |
| webshop_goal_01268 | 3 | 5 | 1/2 | 0/1 | 3/4 | 3/6 | 0.0000% | style |
| webshop_goal_01268 | 4 | 5 | 2/2 | 0/2 | 2/4 | 2/8 | 25.0000% | 无 |
| webshop_goal_01268 | 5 | 5 | 1/2 | 0/2 | 3/4 | 3/8 | 0.0000% | quantity |
| webshop_goal_01268 | 6 | 5 | 2/2 | 0/3 | 2/3 | 2/9 | 22.2222% | 无 |
| webshop_goal_01456 | 0 | 2 | 3/5 | 0/0 | 0/0 | 0/0 | 0.0000% | quick_release, wireless_charging |
| webshop_goal_01456 | 1 | 2 | 1/2 | 1/1 | 4/4 | 6/6 | 0.0000% | category |
| webshop_goal_01456 | 2 | 2 | 2/2 | 2/2 | 2/3 | 6/7 | 85.7143% | 无 |
| webshop_goal_01456 | 3 | 2 | 3/3 | 2/2 | 2/3 | 6/7 | 85.7143% | 无 |
| webshop_goal_01456 | 4 | 2 | 0/2 | 3/3 | 3/3 | 9/9 | 0.0000% | category, budget_max |
| webshop_goal_01456 | 5 | 2 | 1/3 | 1/1 | 2/3 | 4/5 | 0.0000% | category, budget_max |
| webshop_goal_01581 | 0 | source_gold | 3/4 | 0/0 | 0/0 | 0/0 | 0.0000% | high_quality |
| webshop_goal_01581 | 1 | source_gold | 1/3 | 0/0 | 2/3 | 2/3 | 0.0000% | sterilization, material |
| webshop_goal_01581 | 2 | source_gold | 0/2 | 0/1 | 0/3 | 0/5 | 0.0000% | category, material |
| webshop_goal_01581 | 3 | source_gold | 1/2 | 0/1 | 2/3 | 2/5 | 0.0000% | material |
| webshop_goal_01581 | 4 | source_gold | 1/2 | 0/2 | 2/3 | 2/7 | 0.0000% | surface_treatment |
| webshop_goal_01581 | 5 | source_gold | 0/2 | 0/3 | 0/3 | 0/9 | 0.0000% | category, handle_material |
| webshop_goal_01581 | 6 | source_gold | 1/2 | 0/3 | 2/3 | 2/9 | 0.0000% | handle_material |
| webshop_goal_01614 | 0 | 5 | 4/5 | 0/0 | 0/0 | 0/0 | 0.0000% | machine_washable |
| webshop_goal_01614 | 1 | 5 | 2/2 | 0/0 | 3/4 | 3/4 | 75.0000% | 无 |
| webshop_goal_01614 | 2 | 5 | 2/3 | 1/1 | 2/4 | 4/6 | 0.0000% | size |
| webshop_goal_01614 | 3 | 5 | 2/2 | 2/3 | 2/4 | 6/10 | 60.0000% | 无 |
| webshop_goal_01614 | 4 | 5 | 1/3 | 2/3 | 3/4 | 7/10 | 0.0000% | color, backing_color |
| webshop_goal_01614 | 5 | 5 | 2/2 | 2/4 | 3/4 | 7/12 | 58.3333% | 无 |
| webshop_goal_01614 | 6 | 5 | 2/2 | 3/5 | 3/4 | 9/14 | 64.2857% | 无 |
| webshop_goal_01620 | 0 | 4 | 3/4 | 0/0 | 0/0 | 0/0 | 0.0000% | easy_assemble |
| webshop_goal_01620 | 1 | 4 | 3/3 | 0/0 | 2/3 | 2/3 | 66.6667% | 无 |
| webshop_goal_01620 | 2 | 4 | 2/2 | 2/2 | 1/2 | 5/6 | 83.3333% | 无 |
| webshop_goal_01620 | 3 | 4 | 0/2 | 0/2 | 0/3 | 0/7 | 0.0000% | category, desktop_construction |
| webshop_goal_01620 | 4 | 4 | 2/2 | 2/3 | 2/3 | 6/9 | 66.6667% | 无 |
| webshop_goal_01620 | 5 | 4 | 2/2 | 4/4 | 1/2 | 9/10 | 90.0000% | 无 |
| webshop_goal_01620 | 6 | 4 | 2/3 | 2/4 | 2/3 | 6/11 | 0.0000% | weight_capacity |
| webshop_goal_01892 | 0 | 3 | 4/6 | 0/0 | 0/0 | 0/0 | 0.0000% | heavy_duty, easy_clean |
| webshop_goal_01892 | 1 | 3 | 2/2 | 1/1 | 3/5 | 5/7 | 71.4286% | 无 |
| webshop_goal_01892 | 2 | 3 | 0/0 | 3/4 | 2/4 | 8/12 | 66.6667% | 无 |
| webshop_goal_01892 | 3 | 3 | 0/3 | 0/4 | 1/4 | 1/12 | 0.0000% | mobility, security, drawer_count |
| webshop_goal_01892 | 4 | 3 | 1/1 | 2/3 | 0/4 | 4/10 | 40.0000% | 无 |
| webshop_goal_02449 | 0 | 1 | 1/1 | 0/0 | 3/4 | 3/4 | 75.0000% | 无 |
| webshop_goal_02449 | 1 | 1 | 0/2 | 0/1 | 0/4 | 0/6 | 0.0000% | category, finish |
| webshop_goal_02449 | 2 | 1 | 1/2 | 0/2 | 3/4 | 3/8 | 0.0000% | design_style |
| webshop_goal_02449 | 3 | 1 | 2/2 | 0/3 | 3/4 | 3/10 | 30.0000% | 无 |
| webshop_goal_02449 | 4 | 1 | 3/3 | 3/4 | 1/4 | 7/12 | 58.3333% | 无 |
| webshop_goal_02449 | 5 | 1 | 3/3 | 3/5 | 1/4 | 7/14 | 50.0000% | 无 |
| webshop_goal_02592 | 0 | 4 | 5/5 | 0/0 | 0/0 | 0/0 | 100.0000% | 无 |
| webshop_goal_02592 | 1 | 4 | 2/2 | 0/0 | 4/4 | 4/4 | 100.0000% | 无 |
| webshop_goal_02592 | 2 | 4 | 2/2 | 1/2 | 4/4 | 6/8 | 75.0000% | 无 |
| webshop_goal_02592 | 3 | 4 | 1/2 | 2/3 | 4/4 | 8/10 | 0.0000% | printing |
| webshop_goal_02592 | 4 | 4 | 3/3 | 2/2 | 3/3 | 7/7 | 100.0000% | 无 |
| webshop_goal_02656 | 0 | 2 | 4/5 | 0/0 | 0/0 | 0/0 | 0.0000% | long_lasting |
| webshop_goal_02656 | 1 | 2 | 4/4 | 0/0 | 2/4 | 2/4 | 50.0000% | 无 |
| webshop_goal_02656 | 2 | 2 | 2/3 | 3/3 | 1/4 | 7/10 | 0.0000% | color |
| webshop_goal_02656 | 3 | 2 | 2/3 | 3/3 | 1/4 | 7/10 | 0.0000% | burn_time_min |
| webshop_goal_02702 | 0 | 1 | 0/1 | 0/0 | 5/9 | 5/9 | 0.0000% | category |
| webshop_goal_02702 | 1 | 1 | 1/1 | 0/0 | 5/9 | 5/9 | 55.5556% | 无 |
| webshop_goal_02702 | 2 | 1 | 0/2 | 0/0 | 7/9 | 7/9 | 0.0000% | category, dietary_claims |
| webshop_goal_02702 | 3 | 1 | 1/2 | 0/1 | 6/8 | 6/10 | 0.0000% | category |
| webshop_goal_02702 | 4 | 1 | 1/2 | 1/1 | 6/8 | 8/10 | 0.0000% | category |
| webshop_goal_02702 | 5 | 1 | 0/2 | 2/2 | 6/8 | 10/12 | 0.0000% | category, ingredient_exclusions |
| webshop_goal_02798 | 0 | 5 | 0/5 | 0/0 | 0/0 | 0/0 | 0.0000% | category, budget_max, color, heavy_duty, wireless_charging |
| webshop_goal_02798 | 1 | 5 | 1/1 | 0/0 | 3/4 | 3/4 | 75.0000% | 无 |
| webshop_goal_02798 | 2 | 5 | 1/2 | 0/0 | 2/4 | 2/4 | 0.0000% | category |
| webshop_goal_02798 | 3 | 5 | 0/2 | 0/0 | 0/4 | 0/4 | 0.0000% | category, surge_protection_rating_min_joules |
| webshop_goal_02798 | 4 | 5 | 0/1 | 1/1 | 1/3 | 3/5 | 0.0000% | category |
| webshop_goal_02798 | 5 | 5 | 2/3 | 1/1 | 1/3 | 3/5 | 0.0000% | category |
| webshop_goal_02798 | 6 | 5 | 0/2 | 0/3 | 0/3 | 0/9 | 0.0000% | category, usb_c_ports_min |
| webshop_goal_03085 | 0 | 3 | 3/4 | 0/0 | 0/0 | 0/0 | 0.0000% | engineered_wood |
| webshop_goal_03085 | 1 | 3 | 2/3 | 1/2 | 0/0 | 2/4 | 0.0000% | storage_type |
| webshop_goal_03085 | 2 | 3 | 1/2 | 2/4 | 0/0 | 4/8 | 0.0000% | width_max |
| webshop_goal_03085 | 3 | 3 | 2/2 | 2/5 | 0/0 | 4/10 | 40.0000% | 无 |
| webshop_goal_03085 | 4 | 3 | 2/3 | 3/5 | 0/0 | 6/10 | 0.0000% | width_max |
| webshop_goal_03085 | 5 | 3 | 1/1 | 4/8 | 0/0 | 8/16 | 50.0000% | 无 |
| webshop_goal_03259 | 0 | 4 | 3/4 | 0/0 | 0/0 | 0/0 | 0.0000% | easy_use |
| webshop_goal_03259 | 1 | 4 | 2/2 | 0/0 | 2/3 | 2/3 | 66.6667% | 无 |
| webshop_goal_03259 | 2 | 4 | 2/2 | 1/1 | 2/3 | 4/5 | 80.0000% | 无 |
| webshop_goal_03259 | 3 | 4 | 1/2 | 2/2 | 2/3 | 6/7 | 0.0000% | drying_feature |
| webshop_goal_03259 | 4 | 4 | 2/2 | 0/2 | 1/3 | 1/7 | 14.2857% | 无 |
| webshop_goal_03259 | 5 | 4 | 3/3 | 2/3 | 1/2 | 5/8 | 62.5000% | 无 |
| webshop_goal_03383 | 0 | 3 | 4/4 | 0/0 | 0/0 | 0/0 | 100.0000% | 无 |
| webshop_goal_03383 | 1 | 3 | 0/3 | 0/1 | 0/0 | 0/2 | 0.0000% | category, budget_max, shade_type |
| webshop_goal_03383 | 2 | 3 | 2/2 | 1/2 | 0/0 | 2/4 | 50.0000% | 无 |
| webshop_goal_03383 | 3 | 3 | 2/2 | 2/3 | 0/0 | 4/6 | 66.6667% | 无 |
| webshop_goal_03383 | 4 | 3 | 1/2 | 5/6 | 0/0 | 10/12 | 0.0000% | dimming |
| webshop_goal_03784 | 0 | source_gold | 4/4 | 0/0 | 0/0 | 0/0 | 100.0000% | 无 |
| webshop_goal_03784 | 1 | source_gold | 2/2 | 0/0 | 3/3 | 3/3 | 100.0000% | 无 |
| webshop_goal_03784 | 2 | source_gold | 0/1 | 1/1 | 3/3 | 5/5 | 0.0000% | category |
| webshop_goal_03784 | 3 | source_gold | 0/4 | 0/0 | 0/3 | 0/3 | 0.0000% | category, bluetooth_version, digital_audio_input, remote_control |
| webshop_goal_03784 | 4 | source_gold | 0/1 | 0/3 | 0/3 | 0/9 | 0.0000% | category |
| webshop_goal_03784 | 5 | source_gold | 3/4 | 0/1 | 2/3 | 2/5 | 0.0000% | bluetooth_version |
| webshop_goal_03784 | 6 | source_gold | 2/2 | 2/4 | 1/2 | 5/10 | 50.0000% | 无 |
| webshop_goal_03884 | 0 | 3 | 5/6 | 0/0 | 0/0 | 0/0 | 0.0000% | easy_install |
| webshop_goal_03884 | 1 | 3 | 2/2 | 0/0 | 5/5 | 5/5 | 100.0000% | 无 |
| webshop_goal_03884 | 2 | 3 | 1/2 | 1/1 | 5/5 | 7/7 | 0.0000% | screen_type |
| webshop_goal_03884 | 3 | 3 | 2/2 | 1/2 | 5/5 | 7/9 | 77.7778% | 无 |
| webshop_goal_03884 | 4 | 3 | 1/2 | 2/4 | 5/5 | 9/13 | 0.0000% | night_photography_compatibility |
| webshop_goal_03987 | 0 | 4 | 2/3 | 0/0 | 0/0 | 0/0 | 0.0000% | color |
| webshop_goal_03987 | 1 | 4 | 1/2 | 1/1 | 0/0 | 2/2 | 0.0000% | color |
| webshop_goal_03987 | 2 | 4 | 2/3 | 2/2 | 0/0 | 4/4 | 0.0000% | height_max |
| webshop_goal_03987 | 3 | 4 | 2/2 | 3/4 | 0/0 | 6/8 | 75.0000% | 无 |
| webshop_goal_04166 | 0 | 2 | 0/5 | 0/0 | 0/0 | 0/0 | 0.0000% | category, budget_max, style, high_density, memory_foam |
| webshop_goal_04166 | 1 | 2 | 2/2 | 0/0 | 2/3 | 2/3 | 66.6667% | 无 |
| webshop_goal_04166 | 2 | 2 | 2/2 | 1/1 | 2/3 | 4/5 | 80.0000% | 无 |
| webshop_goal_04166 | 3 | 2 | 5/5 | 0/0 | 2/3 | 2/3 | 66.6667% | 无 |
| webshop_goal_04166 | 4 | 2 | 3/3 | 4/4 | 1/2 | 9/10 | 90.0000% | 无 |
| webshop_goal_04166 | 5 | 2 | 2/2 | 6/6 | 1/2 | 13/14 | 92.8571% | 无 |
| webshop_goal_04266 | 0 | source_gold | 2/4 | 0/0 | 0/0 | 0/0 | 0.0000% | high_speed, easy_use |
| webshop_goal_04266 | 1 | source_gold | 3/4 | 0/0 | 0/2 | 0/2 | 0.0000% | color_printing |
| webshop_goal_04266 | 2 | source_gold | 2/2 | 3/3 | 0/2 | 6/8 | 75.0000% | 无 |
| webshop_goal_04266 | 3 | source_gold | 2/3 | 3/3 | 0/2 | 6/8 | 0.0000% | high_speed_printing |
| webshop_goal_04266 | 4 | source_gold | 1/2 | 4/5 | 0/2 | 8/12 | 0.0000% | automatic_duplex_printing |
| webshop_goal_04266 | 5 | source_gold | 2/2 | 3/5 | 0/2 | 6/12 | 50.0000% | 无 |
| webshop_goal_04266 | 6 | source_gold | 1/2 | 4/6 | 0/2 | 8/14 | 0.0000% | automatic_document_feeder |
| webshop_goal_04307 | 0 | 5 | 0/5 | 0/0 | 0/0 | 0/0 | 0.0000% | category, budget_max, color, easy_clean, dead_skin |
| webshop_goal_04307 | 1 | 5 | 0/2 | 0/0 | 0/4 | 0/4 | 0.0000% | category, cleaning_method |
| webshop_goal_04307 | 2 | 5 | 2/2 | 0/1 | 3/3 | 3/5 | 60.0000% | 无 |
| webshop_goal_04307 | 3 | 5 | 2/2 | 2/2 | 3/3 | 7/7 | 100.0000% | 无 |
| webshop_goal_04307 | 4 | 5 | 2/2 | 3/3 | 3/3 | 9/9 | 100.0000% | 无 |
| webshop_goal_04307 | 5 | 5 | 2/2 | 3/4 | 2/3 | 8/11 | 72.7273% | 无 |
| webshop_goal_04307 | 6 | 5 | 2/2 | 4/5 | 2/3 | 10/13 | 76.9231% | 无 |
| webshop_goal_04335 | 0 | 2 | 3/4 | 0/0 | 0/0 | 0/0 | 0.0000% | resealable_bag |
| webshop_goal_04335 | 1 | 2 | 0/2 | 0/0 | 0/2 | 0/2 | 0.0000% | category, budget_max |
| webshop_goal_04335 | 2 | 2 | 2/2 | 2/2 | 0/1 | 4/5 | 80.0000% | 无 |
| webshop_goal_04335 | 3 | 2 | 1/1 | 2/3 | 1/2 | 5/8 | 62.5000% | 无 |
| webshop_goal_04512 | 0 | 2 | 5/5 | 0/0 | 0/0 | 0/0 | 100.0000% | 无 |
| webshop_goal_04512 | 1 | 2 | 0/3 | 0/3 | 0/1 | 0/7 | 0.0000% | category, shape, style |
| webshop_goal_04512 | 2 | 2 | 2/2 | 1/3 | 2/2 | 4/8 | 50.0000% | 无 |
| webshop_goal_04512 | 3 | 2 | 4/4 | 1/3 | 1/1 | 3/7 | 42.8571% | 无 |
| webshop_goal_04512 | 4 | 2 | 3/4 | 1/2 | 0/1 | 2/5 | 0.0000% | frameless |
| webshop_goal_04512 | 5 | 2 | 2/4 | 1/2 | 0/1 | 2/5 | 0.0000% | category, shape |
| webshop_goal_04592 | 0 | 3 | 2/5 | 0/0 | 0/0 | 0/0 | 0.0000% | quick_drying, machine_wash, high_waist |
| webshop_goal_04592 | 1 | 3 | 0/2 | 0/3 | 0/0 | 0/6 | 0.0000% | category, coverage |
| webshop_goal_04592 | 2 | 3 | 0/3 | 0/3 | 0/0 | 0/6 | 0.0000% | category, sleeve_length, coverage |
| webshop_goal_04592 | 3 | 3 | 0/3 | 1/4 | 0/0 | 2/8 | 0.0000% | category, closure_style, coverage |
| webshop_goal_04592 | 4 | 3 | 0/2 | 1/4 | 0/0 | 2/8 | 0.0000% | category, coverage |
| webshop_goal_04592 | 5 | 3 | 0/1 | 0/5 | 0/0 | 0/10 | 0.0000% | quick_drying |
| webshop_goal_05038 | 0 | 1 | 1/1 | 0/0 | 2/3 | 2/3 | 66.6667% | 无 |
| webshop_goal_05038 | 1 | 1 | 2/2 | 0/0 | 2/3 | 2/3 | 66.6667% | 无 |
| webshop_goal_05038 | 2 | 1 | 2/2 | 1/1 | 1/2 | 3/4 | 75.0000% | 无 |
| webshop_goal_05038 | 3 | 1 | 1/2 | 2/2 | 1/2 | 5/6 | 0.0000% | category |
| webshop_goal_05038 | 4 | 1 | 0/2 | 0/3 | 0/2 | 0/8 | 0.0000% | category, kit_components |
| webshop_goal_05038 | 5 | 1 | 0/2 | 3/3 | 1/2 | 7/8 | 0.0000% | category, kit_components |
| webshop_goal_05038 | 6 | 1 | 0/2 | 0/4 | 0/2 | 0/10 | 0.0000% | category, brand_preference |
| webshop_goal_05044 | 0 | 2 | 6/8 | 0/0 | 0/0 | 0/0 | 0.0000% | clinically_proven, dead_skin |
| webshop_goal_05044 | 1 | 2 | 0/1 | 5/7 | 0/0 | 10/14 | 0.0000% | category |
| webshop_goal_05044 | 2 | 2 | 0/3 | 0/6 | 0/1 | 0/13 | 0.0000% | retinol sa, set of treatment, salicylic acid |
| webshop_goal_05044 | 3 | 2 | 0/2 | 0/7 | 0/2 | 0/16 | 0.0000% | formulation_exclusions, oil free |
| webshop_goal_05044 | 4 | 2 | 0/3 | 0/5 | 0/3 | 0/13 | 0.0000% | formulation_exclusions, retinol sa, oil free |
| webshop_goal_05121 | 0 | 1 | 0/7 | 0/0 | 0/0 | 0/0 | 0.0000% | category, budget_max, size, fully_cooked, freeze_dried, shelf_stable, ready_eat |
| webshop_goal_05121 | 1 | 1 | 5/6 | 0/0 | 1/1 | 1/1 | 0.0000% | category |
| webshop_goal_05121 | 2 | 1 | 1/2 | 6/6 | 0/0 | 12/12 | 0.0000% | category |
| webshop_goal_05121 | 3 | 1 | 1/2 | 5/5 | 1/2 | 11/12 | 0.0000% | category |
| webshop_goal_05121 | 4 | 1 | 1/2 | 5/5 | 2/3 | 12/13 | 0.0000% | category |
| webshop_goal_05351 | 0 | 3 | 0/4 | 0/0 | 0/0 | 0/0 | 0.0000% | category, size, non_gmo, dietary_fiber |
| webshop_goal_05351 | 1 | 3 | 0/3 | 0/0 | 0/2 | 0/2 | 0.0000% | category, size, budget |
| webshop_goal_05351 | 2 | 3 | 1/1 | 2/2 | 1/2 | 5/6 | 83.3333% | 无 |
| webshop_goal_05351 | 3 | 3 | 2/2 | 1/1 | 2/2 | 4/4 | 100.0000% | 无 |
| webshop_goal_05351 | 4 | 3 | 0/3 | 1/1 | 1/2 | 3/4 | 0.0000% | dried apricots, dried cranberries, multiple packs |
| webshop_goal_05380 | 0 | 4 | 4/5 | 0/0 | 0/0 | 0/0 | 0.0000% | rubber_sole |
| webshop_goal_05380 | 1 | 4 | 1/4 | 0/0 | 1/1 | 1/1 | 0.0000% | category, relaxed_fit, rubber_sole |
| webshop_goal_05380 | 2 | 4 | 1/5 | 1/1 | 0/0 | 2/2 | 0.0000% | category, fit, relaxed_fit, rubber_sole |
| webshop_goal_05380 | 3 | 4 | 0/5 | 0/2 | 0/0 | 0/4 | 0.0000% | category, style, relaxed_fit, arch_support, rubber_sole |
| webshop_goal_05380 | 4 | 4 | 2/5 | 2/2 | 0/0 | 4/4 | 0.0000% | relaxed_fit, arch_support, rubber_sole |
| webshop_goal_05380 | 5 | 4 | 0/5 | 0/3 | 0/0 | 0/6 | 0.0000% | category, toe_shape, relaxed_fit, arch_support, rubber_sole |
| webshop_goal_05775 | 0 | 2 | 4/4 | 0/0 | 0/0 | 0/0 | 100.0000% | 无 |
| webshop_goal_05775 | 1 | 2 | 3/3 | 3/3 | 0/0 | 6/6 | 100.0000% | 无 |
| webshop_goal_05775 | 2 | 2 | 3/4 | 4/4 | 0/0 | 8/8 | 0.0000% | connector_type |
| webshop_goal_05775 | 3 | 2 | 2/2 | 6/8 | 0/0 | 12/16 | 75.0000% | 无 |
| webshop_goal_06824 | 0 | 1 | 1/1 | 0/0 | 2/3 | 2/3 | 66.6667% | 无 |
| webshop_goal_06824 | 1 | 1 | 0/2 | 0/0 | 0/3 | 0/3 | 0.0000% | category, style |
| webshop_goal_06824 | 2 | 1 | 0/2 | 0/1 | 0/3 | 0/5 | 0.0000% | category, shade_style |
| webshop_goal_06824 | 3 | 1 | 3/3 | 0/2 | 0/2 | 0/6 | 0.0000% | 无 |
| webshop_goal_06824 | 4 | 1 | 0/3 | 0/3 | 0/2 | 0/8 | 0.0000% | category, budget_max, height_min |
