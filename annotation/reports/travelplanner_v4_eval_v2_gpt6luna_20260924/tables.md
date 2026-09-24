### Action level (v2)

| 模型 | shard1 Hard Success | shard2 Hard Success | shard3 Hard Success | Hard Success (turn macro) | Strict (all Must) | Must 满足 | Preferred（仅 Hard Success 轮） | Optional（仅 Hard Success 轮） |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Claude Sonnet 4.6 | 19/32 | 24/64 | 21/64 | **40.0 (64/160)** | 41.9 (67/160) | 83.9 (133/158) | 48.4 (12/24) | 84.0 (35/42) |
| Claude Sonnet 5 | 17/32 | 35/64 | 22/64 | **46.2 (74/160)** | 42.5 (68/160) | 82.7 (131/158) | 36.3 (9/25) | 91.8 (50/54) |
| DeepSeek R1 | 12/32 | 15/64 | 14/64 | **25.6 (41/160)** | 23.1 (37/160) | 73.2 (116/158) | 41.8 (5/12) | 84.8 (21/25) |
| GPT-5.6 Luna | 16/32 | 27/64 | 26/64 | **43.1 (69/160)** | 41.9 (67/160) | 83.0 (131/158) | 41.4 (9/21) | 94.7 (45/48) |
| GPT-5.6 Sol | 21/32 | 35/64 | 24/64 | **50.0 (80/160)** | 50.0 (80/160) | 85.5 (135/158) | 42.6 (13/30) | 94.0 (54/57) |
| Grok 4.6 | 19/32 | 40/64 | 28/64 | **54.4 (87/160)** | 51.2 (82/160) | 84.5 (134/158) | 33.5 (10/30) | 91.2 (58/64) |
| Kimi K2.5 | 12/32 | 10/64 | 7/64 | **18.1 (29/160)** | 21.2 (34/160) | 74.2 (117/158) | 57.5 (7/12) | 71.3 (10/14) |
| MiniMax M2.5 | 8/32 | 10/64 | 11/64 | **18.1 (29/160)** | 25.0 (40/160) | 72.4 (114/158) | 45.8 (4/8) | 91.7 (16/17) |
| Nova Pro | 8/32 | 5/64 | 5/64 | **11.2 (18/160)** | 16.9 (27/160) | 69.5 (110/158) | 41.8 (3/8) | 60.2 (5/9) |
| Qwen3 235B | 12/32 | 8/64 | 8/64 | **17.5 (28/160)** | 19.4 (31/160) | 70.1 (111/158) | 48.7 (5/11) | 73.6 (13/18) |
| Qwen3 32B | 5/32 | 4/64 | 3/64 | **7.5 (12/160)** | 15.6 (25/160) | 62.6 (99/158) | 32.2 (2/6) | 78.9 (6/7) |

### Action gates and diagnostics (v2)

| 模型 | 酒店门槛失败 | 其中最低入住 | 其中容量 | budget 满足 | 餐位覆盖缺口 | Must 违反说明率（按约束，含 budget） | 其中 budget 违反说明率 | 存在未说明 Must 违反的轮 | not-feasible 轮 | 无 gold 行程 | judge 失败 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Claude Sonnet 4.6 | 11.9 (19/160) | 11.9 (19/160) | 0.6 (1/160) | 48.1 (76/158) | 12.7 (20/158) | 16.3 (22/135) | 3.1 (2/64) | 51.9 (83/160) | 55 | 21 | 0 |
| Claude Sonnet 5 | 7.5 (12/160) | 7.5 (12/160) | 0.0 (0/160) | 55.7 (88/158) | 13.3 (21/158) | 28.2 (40/142) | 5.6 (3/54) | 45.6 (73/160) | 55 | 21 | 0 |
| DeepSeek R1 | 6.2 (10/160) | 6.2 (10/160) | 0.6 (1/160) | 30.4 (48/158) | 36.1 (57/158) | 15.0 (33/220) | 1.1 (1/92) | 69.4 (111/160) | 55 | 21 | 0 |
| GPT-5.6 Luna | 10.6 (17/160) | 10.6 (17/160) | 0.0 (0/160) | 49.4 (78/158) | 8.9 (14/158) | 20.8 (30/144) | 0.0 (0/68) | 51.9 (83/160) | 55 | 21 | 0 |
| GPT-5.6 Sol | 15.0 (24/160) | 15.0 (24/160) | 0.0 (0/160) | 64.6 (102/158) | 27.2 (43/158) | 21.5 (26/121) | 9.3 (4/43) | 38.1 (61/160) | 55 | 21 | 0 |
| Grok 4.6 | 13.1 (21/160) | 13.1 (21/160) | 0.0 (0/160) | 68.4 (108/158) | 13.3 (21/158) | 32.2 (38/118) | 9.1 (3/33) | 31.2 (50/160) | 55 | 21 | 0 |
| Kimi K2.5 | 18.8 (30/160) | 18.8 (30/160) | 0.0 (0/160) | 20.9 (33/158) | 56.3 (89/158) | 22.7 (52/229) | 1.9 (2/106) | 73.1 (117/160) | 55 | 21 | 0 |
| MiniMax M2.5 | 33.1 (53/160) | 30.6 (49/160) | 3.1 (5/160) | 33.5 (53/158) | 41.8 (66/158) | 13.1 (33/251) | 0.0 (0/86) | 73.1 (117/160) | 55 | 21 | 0 |
| Nova Pro | 45.0 (72/160) | 43.1 (69/160) | 1.9 (3/160) | 29.7 (47/158) | 53.8 (85/158) | 2.2 (6/268) | 0.0 (0/93) | 82.5 (132/160) | 55 | 21 | 0 |
| Qwen3 235B | 26.9 (43/160) | 26.2 (42/160) | 1.2 (2/160) | 26.6 (42/158) | 45.6 (72/158) | 23.4 (61/261) | 2.1 (2/97) | 73.8 (118/160) | 55 | 21 | 0 |
| Qwen3 32B | 47.5 (76/160) | 47.5 (76/160) | 1.9 (3/160) | 32.9 (52/158) | 36.7 (58/158) | 6.0 (20/334) | 1.1 (1/88) | 84.4 (135/160) | 55 | 21 | 0 |

### Intention level (v2, atoms, turn macro)

| 模型 | Overall P / R / F1 | micro P / R | Change P / R / F1 | Turn Exact | Priority Accuracy | Priority Turn Exact |
|---|---|---|---|---:|---:|---:|
| Claude Sonnet 4.6 | 80.4 / 73.7 / 74.6 | 78.3 / 70.0 | 70.5 / 76.8 / 69.5 (n=126) | 0.0 (0/160) | 52.6 (879/1670) | 3.1 (5/160) |
| Claude Sonnet 5 | 81.3 / 76.8 / 76.9 | 78.9 / 73.3 | 68.5 / 76.4 / 68.4 (n=126) | 5.6 (9/160) | 53.5 (930/1737) | 7.5 (12/160) |
| DeepSeek R1 | 77.9 / 48.6 / 57.5 | 76.6 / 44.7 | 61.2 / 66.4 / 59.5 (n=126) | 0.6 (1/160) | 61.4 (694/1131) | 2.5 (4/160) |
| GPT-5.6 Luna | 72.7 / 74.4 / 71.2 | 69.5 / 72.9 | 61.7 / 79.3 / 65.6 (n=126) | 0.0 (0/160) | 55.7 (969/1739) | 3.1 (5/160) |
| GPT-5.6 Sol | 74.1 / 74.2 / 72.0 | 71.6 / 72.2 | 61.1 / 75.1 / 63.7 (n=126) | 0.0 (0/160) | 53.3 (944/1770) | 2.5 (4/160) |
| Grok 4.6 | 78.9 / 73.9 / 74.1 | 76.1 / 70.8 | 68.7 / 74.2 / 68.1 (n=126) | 1.2 (2/160) | 51.2 (868/1694) | 3.8 (6/160) |
| Kimi K2.5 | 76.8 / 68.7 / 70.2 | 73.8 / 65.8 | 61.4 / 73.6 / 63.0 (n=126) | 0.0 (0/160) | 50.0 (794/1587) | 2.5 (4/160) |
| MiniMax M2.5 | 75.8 / 63.6 / 67.2 | 73.7 / 60.4 | 60.8 / 68.6 / 60.8 (n=126) | 0.0 (0/160) | 48.6 (732/1505) | 1.9 (3/160) |
| Nova Pro | 72.8 / 46.9 / 55.0 | 71.5 / 43.0 | 54.9 / 64.5 / 54.2 (n=126) | 0.0 (0/160) | 61.0 (670/1099) | 0.6 (1/160) |
| Qwen3 235B | 78.4 / 66.4 / 69.5 | 75.5 / 62.3 | 61.9 / 70.1 / 61.5 (n=126) | 0.0 (0/160) | 48.0 (737/1537) | 2.5 (4/160) |
| Qwen3 32B | 67.7 / 58.0 / 60.1 | 61.2 / 54.4 | 35.5 / 57.6 / 38.5 (n=126) | 0.6 (1/160) | 52.2 (736/1409) | 2.5 (4/160) |

### Calibration (draft labels): 16/16

| case | category | expected | v2 |
|---|---|---|---|
| cal-01 | people_number judged from capacity | {"action_status": "satisfied"} | satisfied |
| cal-02 | people_number / capacity violated | {"action_status": "violated", "unmet_disclosed": false} | violated |
| cal-03 | group split across listings | {"action_status": "satisfied"} | satisfied |
| cal-04 | restaurant rating with self-catered meals | {"action_status": "satisfied"} | satisfied |
| cal-05 | meal cost cap with self-catered meals | {"action_status": "satisfied"} | satisfied |
| cal-06 | constraint scope read from dialogue | {"action_status": "satisfied"} | satisfied |
| cal-07 | house rule convention (allowed unless prohibited) | {"action_status": "satisfied"} | satisfied |
| cal-08 | house rule 'No X' + agent's false claim | {"action_status": "violated", "unmet_disclosed": false} | violated |
| cal-09 | cross-turn contamination | {"action_status": "satisfied"} | satisfied |
| cal-10 | cross-turn contamination | {"action_status": "satisfied"} | satisfied |
| cal-11 | stay dates + minimum nights | {"action_status": "violated", "unmet_disclosed": false} | violated |
| cal-12 | Must violated with explicit disclosure | {"action_status": "violated", "unmet_disclosed": true, "disclosure_quote": "the apartment rating requirement is sacrificed"} | violated +disclosed |
| cal-13 | scope | "out_of_scope" | out_of_scope |
| cal-14 | scope | "out_of_scope" | out_of_scope |
| cal-15 | activity order without clock times | {"action_status": "satisfied"} | satisfied |
| cal-16 | annotation issue: empty candidate pool | {"action_status": "violated", "annotation_issue": "gold turn missing search results; report separately from model failure"} | violated |

### Judge quality: unknown rate and agreement with the database verdict (all models pooled)

| field | v1 n / unknown / DB agree (checked) | v2 n / unknown / DB agree (checked) |
|---|---|---|
| people_number | 1617 / 21.8 / – (0) | 1617 / 0.0 / – (0) |
| room_type | 935 / 0.2 / 99.1 (886) | 935 / 0.0 / 99.8 (886) |
| accommodation_rating | 671 / 0.1 / 100.0 (528) | 671 / 0.0 / 99.8 (528) |
| restaurant_rating | 638 / 0.8 / 96.7 (615) | 638 / 1.3 / 97.9 (615) |
| house_rule | 121 / 5.0 / 90.4 (115) | 121 / 0.0 / 99.1 (115) |
| accommodation_stay | 1199 / 0.0 / – (0) | 1199 / 0.0 / – (0) |
| meal_cost | 110 / 18.2 / – (0) | 110 / 0.0 / – (0) |
| cuisine | 132 / 0.0 / – (0) | 132 / 0.8 / – (0) |
| activity | 198 / 7.1 / – (0) | 198 / 0.0 / – (0) |
| schedule | 550 / 4.9 / – (0) | 550 / 1.6 / – (0) |
| (all other fields) | 12749 / 0.7 / – (0) | 12584 / 0.2 / – (0) |
| (all fields) | 18920 / 2.7 / 98.2 (2144) | 18755 / 0.2 / 99.2 (2144) |
