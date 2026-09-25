# TravelPlanner v4 agent trajectory 评测（judge：GPT-6 Luna）

> **2026-09-24 更新**：已有规则化的 v2 评测，见 [`../travelplanner_v4_eval_v2_gpt6luna_20260924/README.md`](../travelplanner_v4_eval_v2_gpt6luna_20260924/README.md)。v2 做了这些改动：每一轮单独调用 judge；使用版本化的规则文件；生成冻结的公共审核基准；由代码计价、做酒店门槛和餐位覆盖检查；按 Feasible / Not feasible 分支计算 Hard Success；Intention 改成语义原子一对一匹配。**v2 与本报告的口径不同，数字不能直接比较**。本报告保留，作为 main 原版 judge（v1）的结果和问题分析。


- 日期：2026-09-23
- 轨迹：`travelplanner_v4_agent_trajectories_20260919_094132 2/`（11 个模型 × 3 个 shard）
- Gold：`annotation/data/exports/travelplanner_v4/`
- Judge：`openai/gpt-6-luna`（OpenRouter）
- 覆盖：1760 轮（每个模型 160 轮 = shard1 32 + shard2 64 + shard3 64），286 个 instance，**0 失败**

## 0. 结论速览

- **Action 层**（主口径：judge 看到 agent 当轮可见的整个候选池，budget 用代码按数据库计价）：Grok 4.6 55.6%、GPT-5.6 Sol 53.8%（两者差距在 judge 噪声以内，视为并列）、Claude Sonnet 5 43.8% 位居前三；Qwen3 32B 16.2%、Nova Pro 21.2%、Qwen3 235B 21.9% 垫底。
- **Intention 层**：Claude Sonnet 5 和 GPT-5.6 Sol 的 F1 约为 86。DeepSeek R1 和 Nova Pro 的 precision 与其他模型相当，但 recall 只有约 57–59，因为它们每轮只预测 7.4 / 7.7 条，而 gold 平均有 11.7 个字段。
- **意图变化**：Change recall 在 76–92 之间，但 `partial` 风格的轮次会掉到 53–76。
- **Priority**：只有 48–62。主要原因是 gold 在 shard2/3 从 t1 起把背景字段（org/dest/日期/人数）整体降为 optional，而 agent 几乎都标成 must。排除背景字段后是 75–86（§6.3）。
- **⚠️ 口径问题**：如果严格按 main 的 fixed-search 判法（只给 judge action），GPT-5.6 Luna、GPT-5.6 Sol 和 Grok 4.6 在 shard2/3 上都是 0/64。这是 judge 无法核算 budget 造成的假象，不代表模型差。五种口径的完整对照见 §5.1。
- **主口径的选择（§5.5）**：我们对比了两种给 judge 的证据：只附选中实体的记录（grounded），以及附上 agent 可见的整个候选池。两者对非 budget 约束的判定 95–99% 一致，没有任何模型的成功数差异是显著的。最终选用候选池作为主口径，因为 judge 看到的信息和 agent 完全相同，而且不依赖名称匹配规则。无论用哪种证据，judge 都算不好 budget。

## 1. 本地 eval 与远端 main 的同步核对

- `origin/main` 当前是 `2c6a0d6`（2026-09-22，"Add annotated travel planner shards 007 and 008"），这个 commit 只新增了两个标注数据文件，没有代码改动。
- 本分支 `travelplanner-zijian-update` 与 main 的 merge-base 是 `47346d5`：本地领先 3 个 commit，落后 1 个（就是上面那个数据 commit）。
- 逐文件对比了 `src/` 和 `scripts/` 下 main 跟踪的所有文件：本地工作区与 `origin/main` **完全一致**。唯一的例外是 `src/eval/README.md`，它是本地未提交的文档改动。
- 本地独有的文件：`src/eval/judge_v2.py`、`score_v2.py`、`pool_evidence.py`（已提交），以及 `src/eval/agent_v2.py`、`scripts/rejudge_saved_eval.py`（未跟踪）。这些都不在 main 的调用链上，本次评测**没有使用**。
- 结论：本次用到的 main 代码，包括 `build_judge_prompt`、`score_judged_turn`、`aggregate_scored_rows`、`flatten_gold_for_entity_scoring` 和 `intent_schema`，都与 main 一致。本次没有做 merge。

## 2. 评测设置

**输入。** 每个 `output/<model>__<shard>.json` 每一轮只保存了 `agent_intention_prediction`（atomic intent：field / value / priority）和 `action`（`action_type=plan` 加 itinerary），没有 rollout trace。这是 fixed-search 的格式。Gold 通过 manifest 的 `shard_file` 配对，**SHA-256 与轨迹 metadata 中的 `dataset_sha256` 逐个校验一致**，instance id 和 turn id 也全部对齐。

**模型：** `us.anthropic.claude-sonnet-4.6`、`claude-sonnet-5`、`deepseek-r1`、`openai.gpt-5.6-luna`、`gpt-5.6-sol`、`xai.grok-4.6`、`kimi-k2.5`、`minimax-m2.5`、`nova-pro`、`qwen3-235b`、`qwen3-32b`（均通过 Bedrock 生成）。

**Judge 调用。** 每个 instance 调用一次，一次覆盖该 instance 的全部轮次，和 main 的做法相同。设置为 temperature 0.1、JSON 模式。如果 judge 输出的字段或轮次对不上，最多重试 4 次。

| 调用 | 内容 | 用于 |
|---|---|---|
| Pass A（action-only） | main 的 `build_judge_prompt` **原文**，证据为 `{"action": action}`，即 main fixed-search TravelPlanner 的证据形式；用 `score_judged_turn` 计分 | 严格 main 口径的对照 |
| Pass A（grounded） | 同一个 prompt。证据额外附上 `selected_item_reference_records`：**只附行程中选中实体**的完整数据库记录，不附全部候选项，细节见 §2.1 | 对照（与候选池口径比较，见 §5.5） |
| Pass B（alignment） | 新写的对齐 prompt：把 agent 的每条 intent 对到至多一个 gold 字段，并判断 value 是否正确 | Intention 层中 main 输出不了的指标（item 级 precision、Change、按字段比较 priority） |
| **Pass A（候选池）** | 同一个 prompt。证据为 action 加上 agent 当轮可见的全部搜索结果，与 agent prompt 中的 `fixed_search_results` 逐字相同，见 §2.1、§5.5 | **Action 层主结果（除 budget 外的所有约束）** |
| 确定性 budget | 不调用 LLM。按 TravelPlanner 官方计价规则，用 reference 数据库算出行程总价，**替换** judge 对 `budget` 的判定，规则见 §3.4 | Action 层主结果（仅 budget） |

Gold 的实体约束（例如"妈妈行动不便"）用 main 的 `flatten_gold_for_entity_scoring` 展平进约束列表，和单 agent TravelPlanner 入口的做法一致。这样做是因为 judge prompt 会要求评估实体约束，而不展平的话 main 的 scorer 会报字段不匹配。这类约束在全部数据里只有 15 个 constraint-turn，它们没有 tier，所以单列为 "Entity"。

### 2.1 judge 看到了什么证据，和确定性 budget 是什么关系

**action-only 口径下 judge 看到的内容。** 每一轮只有 agent 的 `action`，也就是 itinerary 加上 rationale，外加 gold 意图和 agent 的意图预测。judge 看不到任何数据库信息。

**候选池口径（主口径）额外附上的内容。** 每一轮附上 `agent_visible_search_results`，也就是 agent 当轮在 prompt 里看到的全部搜索结果：景点、住宿、餐厅、交通（航班、"该日无航班"提示、自驾路线），每类每页最多 12 条，平均每轮 37 条，约 2.2k token。它由 main 生成 agent prompt 的同一个函数 `_compact_travel_search_results` 产生，和 agent 看到的内容逐字相同。每条记录都是完整字段，字段内容与下表相同。为什么说这就是 agent 的候选池，见 §5.5。

**grounded 口径（对照）额外附上的内容。** 每一轮附一个 `selected_item_reference_records` 列表，里面**只有这一轮 agent 实际选中的那些实体**在该 instance 的 `reference_information`（TravelPlanner 数据库）中对应的记录，**没有给全部候选项**。平均每轮附 10.3 条记录，而该 instance 数据库平均有 81.7 条候选。每条记录给的是**完整字段**，不只是价格：

| 实体 | 匹配方式 | 附上的字段 |
|---|---|---|
| 餐厅 | 数据库中的名称出现在 action 文本里 | Name、Average Cost（人均价）、Cuisines（菜系）、Aggregate Rating（评分）、City |
| 住宿 | 同上 | NAME、price（每晚价格）、room type、house_rules、minimum nights、maximum occupancy、review rate number（评分）、city |
| 景点 | 同上 | Name、Address、Phone、Website、Latitude、Longitude、City（没有价格；TravelPlanner 中景点不计费） |
| 航班 | 航班号出现在 action 文本里 | Flight Number、Price、DepTime、ArrTime、ActualElapsedTime、FlightDate、OriginCityName、DestCityName、Distance |
| 自驾 / 出租车 | action 中出现 self-driving/self-drive 或 taxi | 该 instance 所有同方式路线的描述串：起止城市、时长、距离、cost |

匹配时用的是整个 action 的 JSON 文本，包括 rationale，采用小写子串匹配。所以只在 rationale 里提到的实体也可能被附上；反过来，名称写法和数据库有出入的实体（多余空格、换行、截断）会漏掉。拿确定性计价使用的更宽松匹配来对照，有 350 / 13,373 个（2.6%）已定价的餐厅/住宿槽位，judge 那边没有拿到对应记录。这些槽位上的**非 budget** 约束（评分、房型、菜系等），judge 看到的证据会少一些。

**与确定性 budget 的关系：两件独立的事。** 无论附上选中实体的记录，还是附上整个候选池，judge 都不擅长算 budget（§5.4、§5.5）：
- 它会以"门票、油费、自带食物没有标价"为由继续判 `unknown`；
- 它会相信 agent 在行程或 rationale 里自报的错误总价（例子见 §5）。

因此 **budget 这一项** 我另外用代码计算（规则见 §3.4），并替换 judge 的判定。**其余所有约束**，例如房型、评分、菜系、航班时间、日期、人数、实体约束，仍然采用 judge 的判定：主结果用候选池口径的 judge，对照用 grounded 口径的 judge。确定性计价只看 itinerary 的 transportation、breakfast、lunch、dinner、accommodation 五个字段，不看 rationale。

### 2.2 judge prompt 是怎么拼起来的

两个 judge 调用都是**单条 user 消息**，没有 system prompt。请求发往 OpenRouter 的 `/chat/completions`，参数为：模型 `openai/gpt-6-luna`，`temperature=0.1`，`response_format={"type": "json_object"}`（如果路由到的模型不支持 JSON 模式，会自动去掉这个参数重试），`max_tokens=64000`，超时 900 秒。HTTP 层遇到 408/429/5xx 最多重试 4 次；返回的 JSON 如果缺轮次或字段对不上，整个 instance 重新调用，最多 4 次。

#### Pass A（行为 + 意图判定，main 的 `build_judge_prompt`）

**代码位置**：
- prompt 模板：[src/eval/human_annotated_pilot.py:250-319](../../../src/eval/human_annotated_pilot.py)，本地与 main 一致；
- payload 组装：[scripts/judge_travelplanner_v4_trajectories.py](../../../scripts/judge_travelplanner_v4_trajectories.py) 中的 `run_pass_a` 和 `action_evidence`。

**拼接方式**：每个（模型, shard, instance）调用一次，一次包含这个 instance 的**所有轮次**。

```
prompt = 固定指令（下方原文）
       + "DOMAIN: travelplanner"
       + "INSTANCE: <instance_id>"
       + "EVAL_PAYLOAD:"
       + json.dumps(judged_turns, ensure_ascii=False, indent=2)
```

`judged_turns` 是一个列表，每一轮对应一个元素：

| 字段 | 来源 | 说明 |
|---|---|---|
| `turn_id` | gold | 轮次编号 |
| `user_utterance` | gold 该轮的 `user_utterance` | **只有这一轮的话**。历史轮次的话出现在同一个列表的前面几个元素里 |
| `gold_current_intention` | gold 该轮的 `gold_current_intention`，经 `flatten_gold_for_entity_scoring` 展平 | 包含 `constraints`（当前全部约束）、`priority`（high/medium/low 分档）、`entities` 等。实体约束被复制为 `entities.<id>.constraints.<field>` 形式的约束 |
| `agent_intention_prediction` | 轨迹该轮的 `agent_intention_prediction` | agent 的 intent 列表：field / value / priority |
| `action_evidence` | 按口径不同而不同，见下表 | 用来判行程 |

`action_evidence` 在三种口径下的内容：

| 口径 | `action_evidence` 内容 |
|---|---|
| action-only（main fixed-search 原版） | `{"action": <agent 该轮的 action，含 itinerary 和 rationale>}` |
| grounded（对照） | 上面的内容，加上 `selected_item_reference_records`：行程中选中实体的数据库记录（§2.1） |
| **候选池（主口径）** | 上面 action 的内容，加上 `agent_visible_search_results`：agent 当轮看到的全部搜索结果，与 agent prompt 中的 `fixed_search_results` 逐字相同 |

**prompt 里没有的东西**：gold 的人工参考行程（`gold_action`）、gold 的 `gold_delta` 和 `shift_condition`、原始 TravelPlanner query、代码算出的 budget 总价（budget 是事后替换的，见 §3.4）。

**固定指令原文**：

```text
You are a strict evaluator for a two-layer intention-change benchmark.
The evaluated agent never saw gold annotations.

Layer 1, intention understanding:
- Agent predictions use intent items with field, value, priority.
  Match these semantically to the original gold, including gold entity constraints.
  Respect context and limits encoded in fields/values: Day 2 does not satisfy
  Day 1, and a budget ceiling is not an exact spending target.
  A field may have multiple items; never collapse them by field name.
  Legacy gold high/medium/low correspond to must_have/preferred/optional.
- For every gold constraint, decide whether the agent recognized it and whether
  the predicted current value is semantically correct.
- A renamed but clearly equivalent field may count as recognized.
- Latest user utterances override earlier values.
- Score priority_order_score from 0 to 1 as gold-weighted tier classification
  accuracy after semantic field matching: must_have=high, preferred=medium,
  optional=low (weights 3/2/1). Missing fields or wrong tiers earn no credit.
  Ignore order within tiers; relative ordering alone does not establish correct tiers.

Layer 2, action compliance:
- Judge the actual selected product or itinerary, not the agent rationale.
- For every gold constraint, label action_status as satisfied, violated, or unknown.
- unknown means the supplied action evidence is genuinely insufficient.
- The agent must select its best available concrete compromise even if some
  requirements cannot be met. Judge those unmet requirements normally; an
  explanation of the trade-off does not earn compliance credit.
- Leaving a required choice unresolved is not satisfied. A no_match action is
  inappropriate when real selectable candidates exist, even if all have defects.
  Set no_match_appropriate=false for such cases; absence of all real selectable
  candidates is the only possible no_match exception.
- A human_gold_action with confirmed=false is an unconfirmed annotation hint, not
  an authoritative answer. Evaluate the candidate evidence yourself.
- Do not give credit merely because the action repeats a requirement in prose.

Return exactly:
{
  "turns": [
    {
      "turn_id": 0,
      "constraint_judgments": [
        {
          "gold_field": "exact gold field",
          "recognized": true,
          "value_match": true,
          "action_status": "satisfied | violated | unknown",
          "note": "brief reason"
        }
      ],
      "predicted_extra_constraints": ["unsupported predicted field/value"],
      "priority_order_score": 0.0,
      "no_match_appropriate": false,
      "summary": "brief"
    }
  ]
}

Evaluate every supplied turn and every non-null gold constraint exactly once.
```

固定指令里提到的 `human_gold_action`，在 TravelPlanner 的三种口径里都**没有**提供，这是 main 为 WebShop 保留的说明。`no_match` 也只适用于 WebShop。

**payload 实例**：Claude Sonnet 5 在 shard1 `travelplanner_test_0144` 的 turn 1，候选池口径。为了便于阅读，行程只保留第 1 天，候选池每页只保留前 2 条，其余内容都与实际发送的一致。这一轮 agent 在 rationale 里**明确说明**自己放弃了"评分 4+"这个 Must，选了一间评分 2 的公寓；按现在的 prompt，这个说明不会给它任何加分。

<details>
<summary>展开 turn 1 的 payload（截断版）</summary>

```json
{
  "turn_id": 1,
  "user_utterance": "I'd rather just do Denver for now: an apartment rated 4+, just March 19-22. I'll take my own car, passing through Gunnison on the way home on the 23rd. launch there.",
  "gold_current_intention": {
    "constraints": {
      "budget": 3100,
      "visiting_city_number": 1,
      "room_type": {
        "Denver": "Entire home/apt"
      },
      "accommodation_rating": {
        "Denver": 4
      },
      "transportation": "Own car ",
      "launch at Gunnison": true
    },
    "priority": {
      "high": [
        "budget",
        "room_type",
        "accommodation_rating",
        "transportation",
        "launch at Gunnison",
        "visiting_city_number"
      ],
      "medium": [],
      "low": []
    },
    "entities": {
      "entity_1": {
        "reference": "the user",
        "constraints": {}
      }
    },
    "entity_priority": [],
    "domain": "travelplanner"
  },
  "agent_intention_prediction": {
    "intent": [
      {
        "field": "people_number",
        "value": 1,
        "priority": "must_have"
      },
      {
        "field": "days",
        "value": "2022-03-19 to 2022-03-23",
        "priority": "must_have"
      },
      {
        "field": "budget",
        "value": 3100,
        "priority": "must_have"
      },
      {
        "field": "city_sequence",
        "value": [
          "Moab",
          "Denver",
          "Gunnison",
          "Moab"
        ],
        "priority": "must_have"
      },
      {
        "field": "required_cities",
        "value": [
          "Denver"
        ],
        "priority": "must_have"
      },
      {
        "field": "launch at Gunnison",
        "value": "2022-03-23, en route home, pass through/depart from Gunnison",
        "priority": "must_have"
      },
      {
        "field": "accommodation_stay",
        "value": "Denver, 2022-03-19 to 2022-03-22",
        "priority": "must_have"
      },
      {
        "field": "room_type",
        "value": "Entire home/apt",
        "priority": "must_have"
      },
      {
        "field": "accommodation_rating",
        "value": ">=4",
        "priority": "must_have"
      },
      {
        "field": "intercity_transportation",
        "value": "self-driving (own car) for all legs",
        "priority": "must_have"
      },
      {
        "field": "local_transportation",
        "value": "own car",
        "priority": "must_have"
      }
    ]
  },
  "action_evidence": {
    "action": {
      "action_type": "plan",
      "itinerary": [
        {
          "day": "2022-03-19",
          "current_city": "Moab to Denver",
          "transportation": "Self-driving, Moab to Denver, ~5h23m, cost $28",
          "breakfast": "-",
          "lunch": "-",
          "dinner": "The Urban Socialite (Denver, Cafe/Indian/Desserts/Italian, $34)",
          "attraction": "-",
          "accommodation": "*NO GUEST SERVICE FEE* Luxury Studio Suite w/ Free Continental Breakfast, Denver (Entire home/apt, min 3 nights, rating 2, $390/night)"
        },
        "…（另有 4 天，已省略）"
      ],
      "rationale": "The plan honors the solo traveler's route (Moab→Denver→Gunnison→Moab, all self-driving) and the exact Denver stay window (3/19–3/22). No Denver 'Entire home/apt' option both meets the 4+ rating and the required 3-night stay: the only rating-4 apartment (Harlem cozy nights) requires a 4-night minimum, which would force extending the Denver stay beyond 3/22 and disrupt the requested Gunnison departure on 3/23. To preserve the explicit date/schedule requirement and 'own car' routing (both stated firmly and tied to the trip structure), the apartment rating requirement is sacrificed: the Denver lodging chosen (Luxury Studio Suite, rating 2) satisfies room type and the 3-night stay instead. Gunnison lodging (Full-size loft bed, rating 4) incidentally meets a 4+ rating. All transportation is self-driving per the user's stated preference, and Gunnison is used as the through-point on 3/23 before returning to Moab, consistent with 'launch there.' Total estimated cost (~$1674) is well within the $3100 budget."
    },
    "agent_visible_search_results": {
      "attractions": [
        {
          "query": "Attractions in Denver",
          "status": "results",
          "message": null,
          "items": [
            {
              "Name": "Denver Zoo",
              "Latitude": 39.74959609999999,
              "Longitude": -104.9508519,
              "Address": "2300 Steele St, Denver, CO 80205, USA",
              "Phone": "(720) 337-1400",
              "Website": "https://denverzoo.org/",
              "City": "Denver"
            },
            {
              "Name": "Denver Botanic Gardens",
              "Latitude": 39.7320964,
              "Longitude": -104.9612839,
              "Address": "1007 York St, Denver, CO 80206, USA",
              "Phone": "(720) 865-3500",
              "Website": "https://www.botanicgardens.org/",
              "City": "Denver"
            },
            "…（此页另有 10 条，已省略）"
          ]
        },
        {
          "query": "Attractions in Gunnison",
          "status": "results",
          "message": null,
          "items": [
            {
              "Name": "I.O.O.F. Park",
              "Latitude": 38.5456696,
              "Longitude": -106.9265976,
              "Address": "124 E Virginia Ave, Gunnison, CO 81230, USA",
              "Phone": "(970) 641-8060",
              "Website": "Unknown",
              "City": "Gunnison"
            },
            {
              "Name": "Jorgensen Park",
              "Latitude": 38.5424494,
              "Longitude": -106.9203665,
              "Address": "Gunnison, CO 81230, USA",
              "Phone": "(970) 641-8061",
              "Website": "https://www.gunnisonco.gov/jorgensen_park/index.php",
              "City": "Gunnison"
            },
            "…（此页另有 6 条，已省略）"
          ]
        }
      ],
      "accommodations": [
        {
          "query": "Accommodations in Denver",
          "status": "results",
          "message": null,
          "items": [
            {
              "NAME": "Harlem cozy nights",
              "price": 457,
              "room type": "Entire home/apt",
              "house_rules": "No visitors",
              "minimum nights": 4,
              "maximum occupancy": 3,
              "review rate number": 4,
              "city": "Denver"
            },
            {
              "NAME": "Studio Style Basement in Shared Apartment",
              "price": 498,
              "room type": "Private room",
              "house_rules": "No parties & No visitors",
              "minimum nights": 20,
              "maximum occupancy": 1,
              "review rate number": 5,
              "city": "Denver"
            },
            "…（此页另有 7 条，已省略）"
          ]
        },
        {
          "query": "Accommodations in Gunnison",
          "status": "results",
          "message": null,
          "items": [
            {
              "NAME": "Full-size loft bed in East Village",
              "price": 104,
              "room type": "Private room",
              "house_rules": "No visitors & No pets & No parties",
              "minimum nights": 1,
              "maximum occupancy": 1,
              "review rate number": 4,
              "city": "Gunnison"
            },
            {
              "NAME": "Clean and convenient 2BR apartment",
              "price": 405,
              "room type": "Private room",
              "house_rules": "No visitors & No children under 10",
              "minimum nights": 7,
              "maximum occupancy": 2,
              "review rate number": 3,
              "city": "Gunnison"
            },
            "…（此页另有 2 条，已省略）"
          ]
        }
      ],
      "restaurants": [
        {
          "query": "Restaurants in Denver",
          "status": "results",
          "message": null,
          "items": [
            {
              "Name": "The Fatty Bao - Asian Gastro Bar",
              "Average Cost": 93,
              "Cuisines": "Bakery, BBQ, Cafe, Indian, Seafood",
              "Aggregate Rating": 4.7,
              "City": "Denver"
            },
            {
              "Name": "The Urban Socialite",
              "Average Cost": 34,
              "Cuisines": "Cafe, Indian, Desserts, Italian",
              "Aggregate Rating": 3.8,
              "City": "Denver"
            },
            "…（此页另有 10 条，已省略）"
          ]
        },
        {
          "query": "Restaurants in Gunnison",
          "status": "results",
          "message": null,
          "items": [
            {
              "Name": "Happy Joe's Pizza & Ice Cream",
              "Average Cost": 74,
              "Cuisines": "Desserts, Italian, Bakery, Indian, Seafood",
              "Aggregate Rating": 3.5,
              "City": "Gunnison"
            },
            {
              "Name": "SALT",
              "Average Cost": 44,
              "Cuisines": "Tea, Chinese, BBQ",
              "Aggregate Rating": 4.3,
              "City": "Gunnison"
            },
            "…（此页另有 10 条，已省略）"
          ]
        }
      ],
      "transportation": [
        {
          "query": "Flight from Moab to Denver on 2022-03-19",
          "status": "results",
          "message": null,
          "items": [
            {
              "Flight Number": "F3811327",
              "Price": 100,
              "DepTime": "18:16",
              "ArrTime": "19:28",
              "ActualElapsedTime": "1 hours 12 minutes",
              "FlightDate": "2022-03-19",
              "OriginCityName": "Moab",
              "DestCityName": "Denver",
              "Distance": 283
            }
          ]
        },
        {
          "query": "Self-driving from Moab to Denver",
          "status": "reference_text",
          "message": "self-driving, from Moab to Denver, duration: 5 hours 23 mins, distance: 570 km, cost: 28",
          "items": []
        },
        {
          "query": "Taxi from Moab to Denver",
          "status": "reference_text",
          "message": "taxi, from Moab to Denver, duration: 5 hours 23 mins, distance: 570 km, cost: 570",
          "items": []
        },
        {
          "query": "Flight from Denver to Gunnison on 2022-03-21",
          "status": "results",
          "message": null,
          "items": [
            {
              "Flight Number": "F3827189",
              "Price": 61,
              "DepTime": "13:26",
              "ArrTime": "14:17",
              "ActualElapsedTime": "0 hours 51 minutes",
              "FlightDate": "2022-03-21",
              "OriginCityName": "Denver",
              "DestCityName": "Gunnison",
              "Distance": 152
            },
            {
              "Flight Number": "F3827701",
              "Price": 49,
              "DepTime": "19:01",
              "ArrTime": "20:19",
              "ActualElapsedTime": "1 hours 18 minutes",
              "FlightDate": "2022-03-21",
              "OriginCityName": "Denver",
              "DestCityName": "Gunnison",
              "Distance": 152
            }
          ]
        },
        {
          "query": "Self-driving from Denver to Gunnison",
          "status": "reference_text",
          "message": "self-driving, from Denver to Gunnison, duration: 3 hours 39 mins, distance: 323 km, cost: 16",
          "items": []
        },
        {
          "query": "Taxi from Denver to Gunnison",
          "status": "reference_text",
          "message": "taxi, from Denver to Gunnison, duration: 3 hours 39 mins, distance: 323 km, cost: 323",
          "items": []
        },
        {
          "query": "Flight from Gunnison to Moab on 2022-03-23",
          "status": "reference_text",
          "message": "There is no flight from Gunnison to Moab on 2022-03-23.",
          "items": []
        },
        {
          "query": "Self-driving from Gunnison to Moab",
          "status": "reference_text",
          "message": "self-driving, from Gunnison to Moab, duration: 4 hours 1 min, distance: 380 km, cost: 19",
          "items": []
        },
        {
          "query": "Taxi from Gunnison to Moab",
          "status": "reference_text",
          "message": "taxi, from Gunnison to Moab, duration: 4 hours 1 min, distance: 380 km, cost: 380",
          "items": []
        }
      ]
    }
  }
}
```

</details>

这个 instance 的完整拼接结果在 `prompt_examples/` 目录下，三种口径各一份：

| 文件 | 内容 |
|---|---|
| `pass_a_action__claude-sonnet-5__travelplanner_test_0144.txt` | action-only |
| `pass_a_grounded__claude-sonnet-5__travelplanner_test_0144.txt` | grounded |
| `pass_a_pool__claude-sonnet-5__travelplanner_test_0144.txt` | 候选池（主口径） |
| `pass_b__claude-sonnet-5__travelplanner_test_0144.txt` | Pass B |

#### Pass B（意图对齐，本次新写的 `build_alignment_prompt`）

**代码位置**：[scripts/judge_travelplanner_v4_trajectories.py](../../../scripts/judge_travelplanner_v4_trajectories.py) 中的 `build_alignment_prompt` 和 `run_pass_b`。

**拼接方式**：同样是每个 instance 调用一次，包含全部轮次：

```
prompt = 固定指令（下方原文）
       + "INSTANCE: <instance_id>"
       + "PAYLOAD:"
       + json.dumps(turns, ensure_ascii=False, indent=2)
```

每一轮只有四个字段：
- `turn_id`
- `user_utterance`
- `gold_constraints`：gold 当前约束，已展平，不含 priority
- `predicted_items`：agent 的 intent 条目，加了 `index` 编号，**去掉了 priority**

**Pass B 看不到**行程、候选池和 gold priority，只负责判断"每条预测对应哪个 gold 字段、值对不对"。

**固定指令原文**：

```text
You align an evaluated agent's predicted intent items to gold constraints for a
multi-turn travel-planning benchmark. Do not judge the itinerary.

For every turn and every predicted item index, decide:
- gold_field: the exact gold constraint field this item expresses, or null if it
  expresses no gold constraint. Match semantically; a renamed but clearly
  equivalent field aligns. An item aligns to at most one gold field. Several
  items may align to the same gold field when each expresses part of it.
- value_match: true only if the item's value is semantically correct for that
  gold constraint's current value (false when gold_field is null). Respect
  context in fields/values: Day 2 does not satisfy Day 1, a budget ceiling is
  not an exact spending target, and an exclusion is not an inclusion. A correct
  part of a multi-part gold value counts as a match if nothing in it conflicts.
Ignore priority tiers; they are compared separately.

Return exactly:
{
  "turns": [
    {
      "turn_id": 0,
      "items": [{"index": 0, "gold_field": "exact gold field or null", "value_match": true}]
    }
  ]
}

Cover every supplied turn and every predicted item index exactly once.
```

<details>
<summary>展开同一轮（turn 1）的 Pass B payload</summary>

```json
{
  "turn_id": 1,
  "user_utterance": "I'd rather just do Denver for now: an apartment rated 4+, just March 19-22. I'll take my own car, passing through Gunnison on the way home on the 23rd. launch there.",
  "gold_constraints": {
    "budget": 3100,
    "visiting_city_number": 1,
    "room_type": {
      "Denver": "Entire home/apt"
    },
    "accommodation_rating": {
      "Denver": 4
    },
    "transportation": "Own car ",
    "launch at Gunnison": true
  },
  "predicted_items": [
    {
      "index": 0,
      "field": "people_number",
      "value": 1
    },
    {
      "index": 1,
      "field": "days",
      "value": "2022-03-19 to 2022-03-23"
    },
    {
      "index": 2,
      "field": "budget",
      "value": 3100
    },
    {
      "index": 3,
      "field": "city_sequence",
      "value": [
        "Moab",
        "Denver",
        "Gunnison",
        "Moab"
      ]
    },
    {
      "index": 4,
      "field": "required_cities",
      "value": [
        "Denver"
      ]
    },
    {
      "index": 5,
      "field": "launch at Gunnison",
      "value": "2022-03-23, en route home, pass through/depart from Gunnison"
    },
    {
      "index": 6,
      "field": "accommodation_stay",
      "value": "Denver, 2022-03-19 to 2022-03-22"
    },
    {
      "index": 7,
      "field": "room_type",
      "value": "Entire home/apt"
    },
    {
      "index": 8,
      "field": "accommodation_rating",
      "value": ">=4"
    },
    {
      "index": 9,
      "field": "intercity_transportation",
      "value": "self-driving (own car) for all legs"
    },
    {
      "index": 10,
      "field": "local_transportation",
      "value": "own car"
    }
  ]
}
```

</details>

#### prompt 长度（按 字符数 / 4 估算 token，286 次调用）

| 调用 | 平均 | 中位数 | 最大 |
|---|---:|---:|---:|
| Pass A action-only | 8.8k | 9.0k | 15.0k |
| Pass A grounded | 14.0k | 14.0k | 26.9k |
| Pass A 候选池（主口径） | 32.2k | 32.8k | 57.2k |
| Pass B | 3.3k | 3.4k | 5.7k |

候选池比 agent 实际看到的每轮 2.2k token 大得多，原因有两个：一是一个 instance 有多轮，每轮都各自带一份候选池；二是 `indent=2` 的 JSON 缩进本身占了不少字符。

## 3. 指标说明

### 3.0 共同概念

- **Gold 约束（gold 字段）。** 每一轮 `gold_current_intention.constraints` 中所有非空的字段，加上展平后的实体约束，例如 `entities.entity_2.constraints.mobility`。每轮平均 11.7 个，所有模型面对的 gold 相同。
- **Tier。** gold 的 `priority` 把字段分为 high / medium / low，本文分别称为 **Must / Preferred / Optional**，与 agent 输出的 `must_have / preferred / optional` 一一对应。实体约束不在 `priority` 里，没有 tier，单独归为 **Entity**。
- **背景字段。** `days, people_number, org, dest, visiting_city_number, start_date, end_date`，即原始查询自带的行程参数。
- **Action 判定（来自 Pass A）。** judge 对每个 gold 约束给出 `action_status ∈ {satisfied, violated, unknown}`，意思是 agent 选出的实际行程满足、违反，或者证据不足以判断。它还会给出 `recognized`（agent 的意图预测里是否识别出这个约束）和 `value_match`（预测的值是否正确）。
- **对齐（来自 Pass B）。** agent 每条 intent 条目会被对到**至多一个** gold 字段（也可以对不上任何字段），同时判断 `value_match`。基于此定义三个概念：
  - 一个 gold 字段被 **"识别"**：至少有一条条目对到它，值对不对都算；
  - 一个 gold 字段被 **"覆盖"**：至少有一条对到它的条目值是正确的；
  - 一条条目是 **"正确条目"**：它对到了某个 gold 字段，并且值正确。
  - 如果条目对到的字段在本轮 gold 里并不存在（见 §8 中 test_0873 的例子），这条条目按"未对齐"处理。
- **宏平均与微平均。** 宏平均（macro）是先逐轮计算指标、再对轮次取平均，每一轮权重相同。微平均（micro）是先把所有轮次的分子分母分别加总、再相除，所以约束多的轮次权重更大。
- **分母。** 除非特别说明，每个模型都是 160 轮（shard1 32 + shard2 64 + shard3 64）。

### 3.1 Action 层（§4.1、§5.1、§6.2）

| 指标 | 怎么算 | 代表什么 |
|---|---|---|
| **shard1/2/3 成功数、合并成功率** | 一轮算"成功"的条件是：该轮**所有** Must 约束的 `action_status` 都是 `satisfied`。`violated` 和 `unknown` 都算失败，Preferred 和 Optional 不影响成功与否。表中给出每个 shard 的成功轮数，以及 160 轮的合计和比例。有 2 轮 gold 没有 Must 约束，它们自动记为成功（空真，见 §8） | 最严格的任务完成度：这一轮给出的行程是否守住了用户的全部硬性要求。**这是 Action 层的主指标** |
| **宽松成功率** | 条件是没有任何 Must 约束被判为 `violated`，`unknown` 放行 | 与严格成功率的差值就是"因证据不足而无法确认"的那部分。两者差得越多，说明结果越依赖 judge 能不能判定（例如 §5 中 action-only 口径下 GPT/Grok 严格约 10%、宽松约 60%） |
| **Must / Preferred / Optional 满足率** | 把该 tier 在所有轮次里的 gold 约束合在一起（micro），计算 `satisfied` 所占比例。n 分别是 928 / 112 / 823 | 不同强度的要求被满足得怎么样。Preferred 衡量模型在权衡时是否顾及用户的偏好；Optional 在 shard2/3 基本等于背景字段（见 §4.1 表下说明），所以它主要衡量"日期、人数、起止城市有没有保住" |
| **S / V / U（§6.2）** | 在同一 tier 的 gold 约束中，`satisfied / violated / unknown` 各占多少比例，三者之和为 100% | 把"不满足"进一步拆成"明确违反"和"无法判断"。U 高说明行程写得太简略，无法核验 |
| **budget satisfied 率（§5.1 第 4 个数）** | 在所有含 `budget` 的轮次（每个模型 158 轮）中，budget 约束被判为 `satisfied` 的比例 | 行程总价是否在预算内。`–` 或 0 表示在该口径下一次都没有被判为 satisfied |

### 3.2 Intention 层（§4.2、§6.3、§6.4）

| 指标 | 怎么算 | 代表什么 |
|---|---|---|
| **Overall P（precision）** | 每一轮：正确条目数 / agent 预测的条目总数。agent 没有输出条目时 P = 0。表中是 160 轮的宏平均 | agent 说出来的需求里有多少是对的。它会因为编造需求、值写错，或多出 gold 里没有的字段而下降 |
| **Overall R（recall）** | 每一轮：被覆盖的 gold 字段数 / gold 字段总数，宏平均 | 用户当前的需求有多少被 agent 正确识别出来。它会因为漏掉约束，或保留了过时的值而下降 |
| **Overall F1** | 每一轮先由该轮的 P、R 算 F1 = 2PR/(P+R)，再取宏平均。所以 F1 不一定落在 P 和 R 之间 | 意图理解的综合分 |
| **micro P / R / F1（§6.3）** | 把所有轮次的"正确条目数 / 预测条目数"和"被覆盖字段数 / gold 字段数"分别加总后再相除，F1 用 micro P、R 计算 | 与宏平均相比，约束多的轮次（shard2/3，每轮 12–13 个字段）权重更大 |
| **Change 轮次** | 满足两个条件的轮次：t ≥ 1，并且 `gold_delta` 中至少有一个字段的 op 是 add、override 或 relax，同时该字段仍在本轮 gold 中。每个模型 126 轮，其中涉及 add 字段 208 个、override 61 个、relax 9 个。`remove` 全数据集只有 1 例，不计入；只改变 tier 的 `reprioritize` 不属于值变化，单独在下方统计 | 用户在这一轮真正改变了需求内容的时刻 |
| **Change R** | 每个 Change 轮次：本轮发生变化的字段中，在本轮被覆盖的比例（值要等于**新值**）。取宏平均 | agent 有没有跟上用户的这次改变。沿用旧值或漏掉新加的要求，都会拉低它 |
| **Change P** | 每个 Change 轮次：<br>① 找出 agent 本轮的"新条目"，即 field 与 value 规范化后（小写、去空白）在 **agent 自己上一轮**的预测里不存在的条目；<br>② 从中剔除"值正确地对到了本轮**未变化**字段"的条目，因为那只是换了个说法重复、或补上了以前漏的内容，不算提出了变化；<br>③ 剩下的就是 agent "声称的变化"，P = 其中正确对到**本轮变化字段**的条目数 / 声称变化的条目数。<br>agent 本轮没有声称任何变化时 P = 0。取宏平均 | agent 改动的内容是不是用户真正改的。编造变化、改错字段或改错值，都会拉低它 |
| **Change F1** | 每轮由 Change P、R 计算，再取宏平均 | 变化追踪的综合分 |
| **Change micro P / R / F1（§6.3）** | 把各 Change 轮的"正确声称数 / 声称数"和"覆盖的变化字段数 / 变化字段数"分别加总后相除 | 同上，按变化数加权 |
| **Change recall add / override / relax（§6.3）** | 把 (轮次, 变化字段) 按 op 分组，每组里被覆盖的比例（micro） | 不同类型的变化难度不同：add 是新增要求；override 是把旧值改成新值，最容易出现"沿用旧值"的错误；relax 是放宽要求，只有 9 例，样本太小 |
| **Turn Exact** | 该轮 P = 1 且 R = 1 的轮次比例，也就是 agent 的意图列表与 gold 完全一致，不多也不少 | 最严格的意图匹配。每轮有约 12 个字段，几乎不可能完全一致，所以在当前 gold 粒度下区分度很低 |
| **Priority Accuracy** | 分母是：有 tier、并且至少被一条条目**识别**的 gold 字段（值不必正确）。对每个这样的字段，取对到它的第一条"值正确"条目的 priority；如果都不正确，就取第一条对到它的条目。这个 priority 等于 gold tier 就记为正确。结果是所有轮次合计的比例（micro） | agent 在识别出约束的前提下，能否判断它有多重要。没识别出的字段不进分母，所以漏报多的模型分数反而可能偏高（见 §4.2 表下说明） |
| **Priority acc.（未识别计错，§6.3）** | 分子同上，分母换成所有有 tier 的 gold 字段 | 把"没识别"也算作没判断对，更接近 main 的 priority_order_score 的计分方式 |
| **Priority acc.（排除背景字段，§6.3）** | 与 Priority Accuracy 相同，但不计 7 个背景字段 | 去掉"gold 把背景字段降为 optional"这个约定的影响之后，agent 对真正意义上的用户需求判断优先级的能力 |
| **Reprioritize acc.（§6.3）** | 对本轮 `gold_delta` 中 op 为 `reprioritize` 的字段（也就是 gold 调整了它的 tier），检查 agent 本轮给出的 tier 是否等于**新** tier。未识别算错，取 micro | agent 能否跟上"重要性的变化"。这类变化绝大多数是 t1 时背景字段整体从 high 降到 low，所以分数接近 0 |
| **Priority Turn Exact** | 该轮所有有 tier 的 gold 字段都被识别，并且 tier 全部正确的轮次比例 | 整轮优先级完全一致。受背景字段降级的影响，几乎只有 t0 可能达到 |
| **Priority tier 混淆（§6.4）** | 对每个 gold tier，把该 tier 的 gold 字段按 agent 给出的 tier（must / pref / opt）或"missed"（未识别）分类，列出各类所占比例，括号中为 n | 看错误具体往哪个方向偏。例如 gold Preferred 有 52–68% 被 agent 标成了 must，说明 agent 倾向于把偏好当成硬性要求 |
| **平均预测条数 / 平均 gold 字段 / 未对齐条数每轮（§6.3）** | 分别是：每轮 agent 输出的 intent 条目数、gold 字段数、没对到任何 gold 字段的条目数，各自取平均 | 用来解释 P 和 R：条数少会导致 R 低（例如 DeepSeek R1、Nova Pro）；未对齐条数多则说明编造或多报了需求 |

### 3.3 Main shared scorer（§6.1，main 的 `score_judged_turn` + `aggregate_scored_rows`，没有做任何改动）

每一轮先给每个 gold 约束一个权重：Must = 3，Preferred = 2，Optional = 1，没有 tier 的 Entity = 1。下面每个指标都是先逐轮计算，再对 160 轮取平均。

| 指标 | 怎么算 | 代表什么 |
|---|---|---|
| **Constraint value acc.** | 该轮 `value_match = true` 的约束的权重之和 / 该轮总权重。`value_match` 由 Pass A 的 judge 判断 | 按重要性加权的意图识别准确率，相当于加权版本的 recall |
| **Priority order score** | judge **直接给出**的 0–1 分数，没有经过代码计算。prompt 要求它这样打分：在语义对齐之后，按 gold 加权（3/2/1）计算 tier 分类的准确率，缺失的字段和 tier 错误的字段都不得分 | main 口径下的优先级判断能力。和本文的 Priority Accuracy 有三点不同：它是 LLM 估算出来的、有加权、未识别的字段算零分 |
| **Intention understanding** | 每轮取 (Constraint value acc. + Priority order score) / 2 | main 口径的意图理解总分 |
| **Action compliance** | 该轮 `action_status = satisfied` 的约束的权重之和 / 总权重。只有 judge 认为 `no_match` 合理时才记 1，但 TravelPlanner 不存在这种情况 | 按重要性加权的行程满足率。和"成功率"不同，它给部分满足的行程部分分数 |
| **Hard-priority violation rate** | 如果该轮有任何一个"权重等于本轮最大权重"的约束被判为 `violated`，这一轮就记 1，最后求比例。本轮有 Must 时，最大权重就是 Must 的 3；没有 Must 时就退而取当前最高的一档 | 行程违反了最重要要求的轮次比例，越低越好 |

### 3.4 确定性 budget 与相关诊断（§5）

**计价规则**，与 TravelPlanner 官方 `get_total_cost` 一致，数据取自该 instance 的 `reference_information`：

| 项目 | 计价方式 |
|---|---|
| 航班 | 从 transportation 文本里提取航班号，按该航班的 Price × 人数计 |
| 自驾 | 在 transportation 与 current_city 文本中找到起止城市都出现的那条路线，按 cost × ⌈人数/5⌉ 计。同一段路线只计一次，因为一段长途驾驶可能被拆成好几天写 |
| 出租车 | 同上，按 cost × ⌈人数/4⌉ 计 |
| 早餐、午餐、晚餐 | 找到对应餐厅，按 Average Cost × 人数计 |
| 住宿 | 每一天的 accommodation 视为住一晚，按 price × ⌈人数 / maximum occupancy⌉ 计 |
| 景点 | 不计费 |

- **人数**取本轮 gold 的 `people_number`；gold 里没有时，取原始查询中的人数。
- **名称匹配**：去掉空格和标点后，数据库名称包含在行程文本中即算匹配；也允许行程里写的名称（≥10 个字符）是某个唯一数据库名称的前缀，用来处理截断的情况。
- **未定价**：匹配不到数据库的槽位计 $0，并记为"未定价"，例如"自己做饭"、"自带午餐"、编造的酒店名。写成 `-` 的槽位视为空。
- **判定**：总价 ≤ gold `budget` 时，budget 约束判为 `satisfied`，否则判为 `violated`。
- **怎么接到 judge 的结果上**：这一步是**事后替换**，计算结果**不会发给 judge**。judge 照常对所有约束（包括 budget）给出判定；之后代码在 judge 输出里找到 `gold_field = budget` 的那一条，把它的 `action_status` 改成计价结果（原判定保留在 `judge_action_status` 字段），再用 main 的 `score_judged_turn` 重新计分。其他约束的判定保持不变。

**诊断列（§5.2）**：

| 列 | 含义 |
|---|---|
| 有 budget 的轮数 | gold 中有数值型 budget 的轮次数，每个模型都是 158；另有 2 轮没有 budget |
| 计价 ≤ budget | 按上述规则计算的总价不超预算的轮次比例 |
| 未定价槽位/轮 | 每轮平均有几个槽位没能定价（按 $0 计入）。这个值越大，总价越可能被低估，GPT-5.6 Sol 的情况见 §5.3 |
| judge → 确定性 | 把"judge 原判定 → 计价判定"的组合按轮计数。例如 `U→S 132` 表示有 132 轮 judge 判了 unknown，计价后判为 satisfied；`S→V` 表示 judge 放过、实际却超支 |

**敏感性分析（§5.3）**：把未定价的餐食改按该 instance 中最便宜的数据库餐厅计价，重新计算 budget 判定和成功轮数，用来检验"自带餐食按 $0 计"会不会抬高排名。

### 3.5 其他拆分（§6.5、§6.6）

| 指标 | 怎么算 | 代表什么 |
|---|---|---|
| **按 linguistic style 拆分** | 按 gold 的 `linguistic_style` 把轮次分为 elliptical（省略）、explicit（明确）和 partial（部分表达），括号中是轮数。每组分别给出严格成功率、Overall F1（宏平均）和 Change R（只算该组内的 Change 轮次） | 用户表达越含糊，agent 的理解和执行会差多少 |
| **Judge 一致性** | 对每一轮的每一个 gold 字段，比较主口径 Pass A（候选池）的 `value_match` 与 Pass B 的"已覆盖"是否一致，计算一致的比例 | 两个独立的 judge 调用在"agent 有没有说对这个值"上的一致程度，可以用来粗略衡量 judge 噪声 |

## 4. 主结果

### 4.1 Action level（候选池 evidence + 确定性 budget）

| 模型 | shard1 成功/32 | shard2 成功/64 | shard3 成功/64 | 合并成功率 (/160) | 宽松成功率 | Must 满足 | Preferred 满足 | Optional 满足 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Claude Sonnet 4.6 | 20 | 27 | 20 | **67/160 · 41.9%** | 41.9% | 85.3% | 73.2% | 96.1% |
| Claude Sonnet 5 | 21 | 30 | 19 | **70/160 · 43.8%** | 48.8% | 85.0% | 65.2% | 93.4% |
| DeepSeek R1 | 14 | 17 | 16 | **47/160 · 29.4%** | 34.4% | 79.0% | 67.0% | 93.2% |
| GPT-5.6 Luna | 20 | 23 | 22 | **65/160 · 40.6%** | 42.5% | 83.8% | 63.4% | 97.0% |
| GPT-5.6 Sol | 21 | 31 | 34 | **86/160 · 53.8%** | 58.8% | 88.1% | 68.8% | 97.2% |
| Grok 4.6 | 22 | 33 | 34 | **89/160 · 55.6%** | 60.0% | 89.0% | 65.2% | 93.8% |
| Kimi K2.5 | 19 | 20 | 12 | **51/160 · 31.9%** | 32.5% | 78.8% | 72.3% | 94.7% |
| MiniMax M2.5 | 18 | 12 | 15 | **45/160 · 28.1%** | 31.2% | 74.8% | 69.6% | 93.6% |
| Nova Pro | 14 | 11 | 9 | **34/160 · 21.2%** | 27.5% | 73.1% | 67.9% | 87.7% |
| Qwen3 235B | 15 | 9 | 11 | **35/160 · 21.9%** | 25.0% | 75.3% | 62.5% | 91.4% |
| Qwen3 32B | 12 | 6 | 8 | **26/160 · 16.2%** | 18.8% | 65.6% | 64.3% | 91.9% |

gold 约束总数（每个模型相同）：Must 928，Preferred 112，Optional 823，Entity 15。shard2/3 的 Optional 几乎全是被降级的背景字段（754 个中有 747 个），因此 Optional 满足率在很大程度上反映的是"行程有没有保住出发地、目的地、日期和人数"。

### 4.2 Intention level

| 模型 | Overall P / R / F1 | Change P / R / F1 | Turn Exact | Priority Accuracy | Priority Turn Exact |
|---|---|---|---:|---:|---:|
| Claude Sonnet 4.6 | 87.9 / 83.7 / 84.1 | 79.9 / 90.0 / 81.7 | 0.0 | 52.3 | 3.1 |
| Claude Sonnet 5 | 90.0 / 86.0 / 86.4 | 81.2 / 91.0 / 82.9 | 5.6 | 51.3 | 7.5 |
| DeepSeek R1 | 89.9 / 59.2 / 69.3 | 80.2 / 82.8 / 78.2 | 1.2 | 61.8 | 3.1 |
| GPT-5.6 Luna | 88.9 / 82.7 / 84.2 | 79.3 / 92.3 / 82.4 | 0.0 | 54.0 | 3.1 |
| GPT-5.6 Sol | 89.1 / 85.9 / 86.2 | 78.8 / 88.5 / 80.9 | 0.0 | 51.9 | 3.1 |
| Grok 4.6 | 87.8 / 86.0 / 85.2 | 77.7 / 88.9 / 80.0 | 1.2 | 50.8 | 3.1 |
| Kimi K2.5 | 88.2 / 81.6 / 83.2 | 75.0 / 88.4 / 78.4 | 0.0 | 49.9 | 2.5 |
| MiniMax M2.5 | 89.0 / 75.5 / 80.3 | 75.6 / 83.4 / 76.5 | 0.0 | 49.0 | 2.5 |
| Nova Pro | 85.0 / 57.2 / 66.8 | 70.0 / 77.6 / 69.9 | 0.0 | 61.5 | 1.2 |
| Qwen3 235B | 87.4 / 78.1 / 81.1 | 75.4 / 84.7 / 76.8 | 0.0 | 48.1 | 3.1 |
| Qwen3 32B | 81.5 / 69.6 / 73.8 | 55.4 / 76.3 / 60.4 | 0.6 | 52.8 | 2.5 |

DeepSeek R1 和 Nova Pro 的 Priority Accuracy 看起来偏高，是因为这个指标只统计**已识别**的字段，而这两个模型恰好漏掉了大部分背景字段，也就避开了"背景字段被 gold 降为 optional"这一类错误。把未识别的字段也算作错误后，它们是最低的两名（36.7 / 36.6，见 §6.3）。

## 5. 为什么 Action 层主结果不直接用 main 的 action-only 判法

main 的 fixed-search TravelPlanner 入口只把 `{"action": action}` 交给 judge，同时 prompt 要求 judge "Judge the actual selected product or itinerary, not the agent rationale"。实际跑下来有两个系统性问题，都集中在 budget 这一项上：

1. **有些模型的行程只写名字，不写价格。** 以 GPT-5.6 Luna、GPT-5.6 Sol、Grok 4.6 和 Nova Pro 为例，judge 无从核算，只能判 `unknown`。在 action-only 口径下，GPT-5.6 Luna、GPT-5.6 Sol 和 Grok 4.6 的 budget 一次都没有被判为 `satisfied`，因此这三个模型在 shard2/3 上的严格成功数都是 0/64，而宽松成功率却有约 60%。
2. **即便给了 grounded 记录，GPT-6 Luna 依然不会算账。** 它会以"景点门票、油费、自带食物没有标价"为由继续判 `unknown`，但按 TravelPlanner 的规则景点本来就不计费。另一方面，它又会**相信 agent 自报的错误总价**。例如 DeepSeek R1 在 test_0081 t0 把一间每晚 $677 的公寓写成 `$677/2 nights`，judge 据此判为 satisfied，而按数据库计价实际是 $2,054，超过了 $1,800 的预算。

因此主口径的 budget 采用**确定性计价**，计价规则与 TravelPlanner 官方 `get_total_cost` 一致：

- 航班、餐食按人数计费；
- 自驾每 5 人一辆车，出租车每 4 人一辆；
- 住宿每晚按 ⌈人数 / 最大入住人数⌉ 间计费；
- 景点不计费；
- 名称匹配不到数据库的槽位（例如"自己做饭""自带午餐"）计 $0，并记为"未定价"；
- 跨多天的同一段长途驾驶只计一次费用。

其他约束仍由 judge 判定，主口径下是看到候选池的 judge。

### 5.1 五种口径对照（严格成功率 / 宽松成功率 / Must 满足率 / budget satisfied 率）

| 模型 | action-only evidence，严格 main fixed-search | grounded evidence，budget 由 judge 判 | 候选池 evidence，budget 由 judge 判 | grounded evidence + 确定性 budget | **候选池 evidence + 确定性 budget（主）** |
|---|---|---|---|---|---|
| Claude Sonnet 4.6 | 36.2 / 45.0 / 82.9 / 43.0 | 36.2 / 41.2 / 82.5 / 45.6 | 38.1 / 41.9 / 83.4 / 43.0 | 38.8 / 41.9 / 84.2 / 53.8 | 41.9 / 41.9 / 85.3 / 53.8 |
| Claude Sonnet 5 | 38.8 / 52.5 / 83.4 / 49.4 | 45.0 / 48.8 / 84.6 / 57.0 | 38.8 / 48.1 / 82.9 / 50.0 | 46.2 / 48.8 / 85.6 / 63.9 | 43.8 / 48.8 / 85.0 / 63.9 |
| DeepSeek R1 | 26.9 / 45.0 / 73.1 / 60.1 | 31.9 / 42.5 / 79.3 / 60.1 | 35.0 / 40.6 / 80.4 / 59.5 | 26.2 / 35.6 / 78.0 / 51.9 | 29.4 / 34.4 / 79.0 / 51.9 |
| GPT-5.6 Luna | 10.6 / 61.9 / 74.5 / – | 18.1 / 60.0 / 77.5 / 10.1 | 13.8 / 59.4 / 76.4 / 4.4 | 40.0 / 43.1 / 83.9 / 54.4 | 40.6 / 42.5 / 83.8 / 54.4 |
| GPT-5.6 Sol | 10.0 / 60.6 / 75.6 / – | 10.6 / 58.8 / 75.4 / – | 10.6 / 58.1 / 75.3 / – | 56.2 / 60.0 / 88.3 / 85.4 | 53.8 / 58.8 / 88.1 / 85.4 |
| Grok 4.6 | 9.4 / 60.0 / 75.2 / – | 10.6 / 57.5 / 75.9 / – | 11.2 / 61.3 / 76.6 / – | 53.1 / 57.5 / 88.3 / 77.8 | 55.6 / 60.0 / 89.0 / 77.8 |
| Kimi K2.5 | 31.9 / 39.4 / 79.7 / 56.3 | 33.8 / 38.8 / 79.7 / 57.0 | 35.0 / 39.4 / 79.2 / 54.4 | 29.4 / 32.5 / 79.0 / 52.5 | 31.9 / 32.5 / 78.8 / 52.5 |
| MiniMax M2.5 | 36.2 / 48.1 / 76.0 / 65.8 | 31.9 / 41.2 / 75.1 / 66.5 | 36.2 / 40.0 / 77.0 / 75.9 | 26.9 / 33.8 / 74.1 / 61.4 | 28.1 / 31.2 / 74.8 / 61.4 |
| Nova Pro | 4.4 / 53.1 / 47.6 / 3.8 | 24.4 / 38.1 / 71.2 / 53.8 | 24.4 / 35.6 / 71.8 / 51.3 | 21.2 / 28.7 / 71.7 / 62.7 | 21.2 / 27.5 / 73.1 / 62.7 |
| Qwen3 235B | 26.9 / 34.4 / 74.0 / 48.7 | 28.7 / 32.5 / 75.3 / 57.6 | 25.6 / 30.0 / 75.5 / 61.4 | 23.1 / 26.2 / 75.6 / 55.1 | 21.9 / 25.0 / 75.3 / 55.1 |
| Qwen3 32B | 6.2 / 34.4 / 51.3 / 11.4 | 15.0 / 22.5 / 65.1 / 48.7 | 15.0 / 23.1 / 64.1 / 41.1 | 14.4 / 16.9 / 65.4 / 55.7 | 16.2 / 18.8 / 65.6 / 55.7 |

表中 `–` 表示该口径下 budget 一次都没有被判为 satisfied。

### 5.2 确定性 budget 诊断（候选池 judge 的 budget 判定 → 数据库计价判定，按轮计数）

| 模型 | 有 budget 的轮数 | 计价 ≤ budget | 未定价槽位/轮 | judge → 确定性 |
|---|---:|---:|---:|---|
| Claude Sonnet 4.6 | 158 | 53.8 | 2.51 | S→S 62, S→V 6, U→S 17, U→V 1, V→S 6, V→V 66 |
| Claude Sonnet 5 | 158 | 63.9 | 0.73 | S→S 76, S→V 3, U→S 19, U→V 4, V→S 6, V→V 50 |
| DeepSeek R1 | 158 | 51.9 | 2.29 | S→S 68, S→V 26, U→S 7, U→V 1, V→S 7, V→V 49 |
| GPT-5.6 Luna | 158 | 54.4 | 1.27 | S→S 6, S→V 1, U→S 77, U→V 46, V→S 3, V→V 25 |
| GPT-5.6 Sol | 158 | 85.4 | 3.95 | U→S 132, U→V 2, V→S 3, V→V 21 |
| Grok 4.6 | 158 | 77.8 | 0.78 | U→S 120, U→V 11, V→S 3, V→V 24 |
| Kimi K2.5 | 158 | 52.5 | 2.64 | S→S 55, S→V 31, U→S 15, U→V 3, V→S 13, V→V 41 |
| MiniMax M2.5 | 158 | 61.4 | 1.23 | S→S 84, S→V 36, U→S 8, V→S 5, V→V 25 |
| Nova Pro | 158 | 62.7 | 0.61 | S→S 66, S→V 15, U→S 30, U→V 17, V→S 3, V→V 27 |
| Qwen3 235B | 158 | 55.1 | 0.91 | S→S 71, S→V 26, U→S 11, U→V 1, V→S 5, V→V 44 |
| Qwen3 32B | 158 | 55.7 | 1.11 | S→S 53, S→V 12, U→S 29, U→V 9, V→S 6, V→V 49 |

（S = satisfied，V = violated，U = unknown。）"计价 ≤ budget"和"未定价槽位/轮"只取决于行程本身，与给 judge 的证据无关。judge 判 `satisfied` 但计价后超支（S→V）的情况，集中出现在 DeepSeek R1、Kimi K2.5、MiniMax M2.5 和 Qwen3 235B 上，每个模型 26–36 轮，基本都是 agent 在行程里自报了偏低的价格或总价。GPT-5.6 Sol 和 Grok 4.6 则几乎全部是 U→S，即 judge 不敢下结论，而实际并没有超支。

### 5.3 敏感性："自己做饭 / 自带餐食"按 $0 计是否抬高了排名

GPT-5.6 Sol 平均每轮有 3.95 个未定价槽位，绝大多数是 self-catering 或 packed 餐食。这里把所有未定价的餐食改按该 instance 中最便宜的数据库餐厅计价，再用主口径重新计算：

| 模型 | budget 满足（规则 $0 → 最便宜餐厅） | 成功轮数 |
|---|---|---|
| Grok 4.6 | 123 → 113 / 158 | 89 → 84（52.5%） |
| GPT-5.6 Sol | 135 → 114 / 158 | 86 → 78（48.8%） |
| Claude Sonnet 4.6 | 85 → 76 / 158 | 67 → 65 |
| Kimi K2.5 | 83 → 77 / 158 | 51 → 47 |
| 其余模型 | 变化 ≤ 5 轮 | 变化 ≤ 1 轮 |

在这种计法下，Grok 4.6 和 GPT-5.6 Sol 仍然是前两名，其余模型的排序不变，**排名基本稳定**。

### 5.4 judge 为什么算不好 budget：全量统计

结论来自 grounded 口径下 judge 给 budget 写的判定理由（note），用关键词对全部 1760 轮做了统计。候选池口径下的情况基本相同，见 §5.5 最后两列：

- **judge 判 `unknown` 的 budget 有 530 条。**
  - 其中 391 条的理由明确是"有费用没标价"。具体提到门票/入场费的 124 条，自带或自做餐食的 50 条，油费、停车费、自驾费的 23 条。按 TravelPlanner 的规则，景点本来就不计费。
  - 按数据库计价后，这 530 条里有 449 条其实在预算内。
- **judge 判 `satisfied`、但计价后超支的有 162 条。** 其中 99 条的理由直接引用了 agent 自报的总价或估算。例如 GPT-5.6 Luna 在 test_0037 t4 的 note 写的是 "listed priced components total about $2,023"，而计价结果是 $2,638，预算为 $2,400。

关键词统计是近似的。逐条手工核对过计价的只有 DeepSeek R1 test_0081 的 3 轮。

### 5.5 对照实验：只附选中实体（grounded）vs 附 agent 可见的整个候选池

**agent 看到的候选池是什么。** 这批轨迹由 fixed-search 流程生成。判断依据有三点：
- 轨迹 metadata 的统计字段（`no_json / empty_constraints / bad_action`）与 main 上的 `annotation/tools/travel_pilot_eval.py` 相同；
- action 的格式与 main 的 `build_agent_prompt` 一致；
- 在这个流程里，agent 每一轮在 prompt 中看到的 `fixed_search_results`，就是 gold 该轮的 `env_feedback.search_results`，经 `_compact_travel_search_results` 压缩后的结果：每类每页最多保留 12 条，内容包括景点、住宿、餐厅，以及交通（航班、"该日无航班"提示、自驾路线）。

用数据核对过：agent 选中的餐厅 9,627/9,627、住宿 3,675/3,748、航班 2,478/2,480 都出现在**当轮**的这份结果里。这份候选池平均每轮 37 条，约 2.2k token；整个 instance 的数据库平均是 82 条。有 74 页原始结果超过 12 条，agent 只看到了前 12 条。有 1 轮（test_0081 t3）gold 本身没有搜索结果，候选池为空。

**实验设置。** 新增一个"候选池"口径：judge 的证据是 `action` 加上 `agent_visible_search_results`。后者直接调用 main 生成 agent prompt 时用的同一个函数，所以 **judge 看到的候选池和 agent 看到的逐字相同**。judge prompt 和计分都不变，budget 仍然用 §3.4 的代码计价替换。这个口径和主口径（grounded：只附选中实体的记录）在完全相同的 1760 轮上逐约束对比。这次运行花费约 $2.18。

| 模型 | 严格成功 grounded → 候选池 | 成功翻转（成功→失败 / 失败→成功，McNemar p） | 非 budget 约束判定一致率 | 非 budget S / V / U（grounded） | 非 budget S / V / U（候选池） | 变化最多的三类 | LLM 自判 budget 的 unknown 率 grounded → 候选池 | LLM 自判 budget 与计价一致率（已判定部分）grounded → 候选池 |
|---|---|---|---:|---|---|---|---|---|
| Claude Sonnet 4.6 | 62 → 67 | 1 / 6，p=0.12 | 98.9 | 91.7 / 7.0 / 1.3 | 92.6 / 6.7 / 0.6 | U→S 12, V→S 5, V→U 1 | 7.6 → 11.4 | 91.1 → 91.4 |
| Claude Sonnet 5 | 74 → 70 | 5 / 1，p=0.22 | 97.9 | 89.7 / 8.1 / 2.2 | 89.4 / 8.1 / 2.5 | S→U 14, V→S 8, U→S 7 | 8.2 → 14.6 | 95.9 → 93.3 |
| DeepSeek R1 | 42 → 47 | 3 / 8，p=0.23 | 96.0 | 84.4 / 9.7 / 5.9 | 87.2 / 9.8 / 3.0 | U→S 56, S→U 7, U→V 3 | 7.0 → 5.1 | 78.9 → 78.0 |
| GPT-5.6 Luna | 64 → 65 | 1 / 2，p=1.00 | 99.2 | 91.9 / 6.9 / 1.2 | 91.5 / 7.3 / 1.2 | S→V 8, S→U 2, U→S 2 | 71.5 → 77.8 | 88.9 → 88.6 |
| GPT-5.6 Sol | 90 → 86 | 7 / 3，p=0.34 | 97.8 | 91.7 / 6.8 / 1.5 | 91.5 / 7.0 / 1.5 | S→U 12, U→S 11, S→V 8 | 84.2 → 84.8 | 88.0 → 87.5 |
| Grok 4.6 | 85 → 89 | 5 / 9，p=0.42 | 96.9 | 90.5 / 7.6 / 2.0 | 90.6 / 7.2 / 2.2 | S→U 20, U→S 17, V→S 10 | 80.4 → 82.9 | 90.3 → 88.9 |
| Kimi K2.5 | 47 → 51 | 3 / 7，p=0.34 | 96.0 | 88.7 / 9.9 / 1.4 | 88.5 / 9.7 / 1.9 | S→U 26, U→S 19, V→S 11 | 7.0 → 11.4 | 70.1 → 68.6 |
| MiniMax M2.5 | 43 → 45 | 3 / 5，p=0.73 | 95.9 | 84.3 / 12.0 / 3.7 | 84.4 / 12.7 / 3.0 | U→S 24, S→U 15, S→V 14 | 12.0 → 5.1 | 74.8 → 72.7 |
| Nova Pro | 34 → 34 | 5 / 5，p=1.00 | 94.9 | 79.0 / 13.1 / 7.9 | 80.3 / 13.3 / 6.5 | U→S 38, S→U 17, V→S 12 | 29.7 → 29.7 | 79.3 → 83.8 |
| Qwen3 235B | 37 → 35 | 4 / 2，p=0.69 | 95.7 | 84.5 / 12.0 / 3.5 | 83.8 / 12.8 / 3.4 | U→S 23, S→U 22, S→V 20 | 10.8 → 7.6 | 76.6 → 78.8 |
| Qwen3 32B | 23 → 26 | 2 / 5，p=0.45 | 94.6 | 79.2 / 16.3 / 4.5 | 78.8 / 16.8 / 4.4 | S→U 26, U→S 24, S→V 17 | 17.1 → 24.1 | 81.7 → 85.0 |

表中各列的含义：
- **成功翻转**：同一轮在两种口径下严格成功与否的变化次数。p 值来自 McNemar 精确检验，零假设是"两种口径没有系统差别"。
- **非 budget 约束**：除 budget 以外所有 gold 约束的判定。每个模型约 1,720 个，因为 budget 两种口径都用代码计价，所以不纳入比较。
- **变化最多的三类**：例如 `U→S 56` 表示有 56 个约束从 grounded 的 unknown 变成了候选池口径下的 satisfied。
- **LLM 自判 budget**：替换成代码计价之前，judge 自己给 budget 的判定。unknown 率越低越好；一致率是 judge 给出 S/V 的那部分里，和代码计价结果相同的比例。

**结论：**

1. **给候选池不会让 judge 学会算 budget。** GPT/Grok 的 budget 在候选池口径下仍有 78–85% 判为 unknown；已判定部分与计价的一致率也几乎没变：grounded 为 70–96%，候选池为 69–93%。budget 的问题出在算账能力，而不是信息不够，所以**不论给什么证据，budget 都应继续用代码计价**。
2. **非 budget 约束的判定基本不变。** 一致率为 94.9–99.2%，S/V/U 比例几乎相同，翻转主要是 S↔U 双向互换，看起来是 judge 的随机性。**没有任何模型的成功数变化是显著的**：每个模型变化在 −4 到 +5 之间，翻转 3–14 轮，p 全部 ≥ 0.12。
3. **候选池能补上 grounded 的名称匹配漏洞。** 最大的系统性变化是 DeepSeek R1 的 56 个 U→S。grounded 口径因为名称写法不一致，没有附上它所选的部分住宿（例如 Fayetteville 那一家），judge 只能写"Fayetteville 的评分无法确认"；候选池里有这条记录，judge 就能判为满足。Nova Pro 的 38 个 U→S 性质相同。
4. **排名。** 前两名在两种口径下互换：grounded 是 Sol 90、Grok 85，候选池是 Grok 89、Sol 86。但两者的差距都在 judge 噪声范围之内，应视为**并列**。其余模型的相对位置基本不变，只有相邻、分差 ≤ 3 的模型之间有交换，例如 Claude Sonnet 4.6 和 GPT-5.6 Luna、DeepSeek R1 和 MiniMax M2.5。
5. **主口径的选择。** 候选池口径与 agent 的信息完全对等，不依赖我写的名称匹配规则，还能让 judge 判断"所选方案是不是候选里最好的折中"（main prompt 要求的 best-available）。代价是每轮多约 2.2k token 的输入，本次候选池口径的 Pass A 全量运行实测花费约 $2.18。**因此本报告的主结果（§0、§4、§6）采用"候选池 + 确定性 budget"。** grounded 口径的完整指标保留在 `metrics.json` 的 `grounded_det` 中。

## 6. 详细 metrics

### 6.1 Main shared scorer（`aggregate_scored_rows`）

候选池 evidence + 确定性 budget（主口径）：

| 模型 | Intention understanding | Constraint value acc. | Priority order score | Action compliance | Hard-priority violation rate |
|---|---:|---:|---:|---:|---:|
| Claude Sonnet 4.6 | 74.7 | 84.7 | 64.6 | 85.7 | 58.8 |
| Claude Sonnet 5 | 76.7 | 87.8 | 65.7 | 85.0 | 51.9 |
| DeepSeek R1 | 59.5 | 64.2 | 54.9 | 80.8 | 65.6 |
| GPT-5.6 Luna | 77.1 | 87.3 | 66.9 | 84.6 | 58.8 |
| GPT-5.6 Sol | 76.4 | 87.2 | 65.6 | 88.9 | 41.9 |
| Grok 4.6 | 75.4 | 86.2 | 64.5 | 87.5 | 40.6 |
| Kimi K2.5 | 71.5 | 83.0 | 59.9 | 82.2 | 68.1 |
| MiniMax M2.5 | 66.8 | 77.1 | 56.5 | 79.4 | 69.4 |
| Nova Pro | 58.1 | 60.8 | 55.4 | 75.6 | 72.5 |
| Qwen3 235B | 67.2 | 76.7 | 57.6 | 78.0 | 75.0 |
| Qwen3 32B | 64.9 | 70.7 | 59.1 | 72.1 | 81.2 |

grounded evidence + 确定性 budget（对照）：

| 模型 | Intention understanding | Constraint value acc. | Priority order score | Action compliance | Hard-priority violation rate |
|---|---:|---:|---:|---:|---:|
| Claude Sonnet 4.6 | 75.3 | 84.2 | 66.5 | 84.7 | 58.8 |
| Claude Sonnet 5 | 75.4 | 86.2 | 64.6 | 85.0 | 51.9 |
| DeepSeek R1 | 59.0 | 63.2 | 54.8 | 78.4 | 64.4 |
| GPT-5.6 Luna | 76.9 | 86.8 | 67.0 | 85.5 | 58.1 |
| GPT-5.6 Sol | 76.4 | 87.3 | 65.4 | 89.1 | 40.6 |
| Grok 4.6 | 75.3 | 86.1 | 64.4 | 87.9 | 43.1 |
| Kimi K2.5 | 72.5 | 83.7 | 61.3 | 81.9 | 68.1 |
| MiniMax M2.5 | 66.4 | 76.6 | 56.2 | 79.1 | 66.2 |
| Nova Pro | 58.5 | 62.2 | 54.9 | 73.7 | 71.2 |
| Qwen3 235B | 67.0 | 76.6 | 57.4 | 78.4 | 73.8 |
| Qwen3 32B | 63.9 | 70.4 | 57.5 | 72.0 | 83.1 |

action-only（严格 main fixed-search）：

| 模型 | Intention understanding | Constraint value acc. | Priority order score | Action compliance | Hard-priority violation rate |
|---|---:|---:|---:|---:|---:|
| Claude Sonnet 4.6 | 75.1 | 84.6 | 65.7 | 84.6 | 55.6 |
| Claude Sonnet 5 | 76.5 | 87.3 | 65.6 | 83.1 | 48.1 |
| DeepSeek R1 | 59.5 | 64.7 | 54.2 | 75.1 | 55.0 |
| GPT-5.6 Luna | 77.2 | 87.2 | 67.3 | 77.4 | 38.1 |
| GPT-5.6 Sol | 76.9 | 87.6 | 66.3 | 78.5 | 40.0 |
| Grok 4.6 | 75.1 | 85.5 | 64.6 | 77.4 | 40.0 |
| Kimi K2.5 | 72.0 | 83.3 | 60.6 | 80.2 | 61.3 |
| MiniMax M2.5 | 67.2 | 77.7 | 56.7 | 78.7 | 51.9 |
| Nova Pro | 57.6 | 60.1 | 55.0 | 54.2 | 46.9 |
| Qwen3 235B | 67.2 | 76.8 | 57.6 | 77.6 | 66.2 |
| Qwen3 32B | 65.3 | 71.2 | 59.4 | 58.2 | 65.6 |

main scorer 的 Intention 部分在三种口径下几乎相同（差异 ≤ 2.1 个点），因为三次 Pass A 调用看到的 intent 输入完全一样，差异只来自 judge 的随机性。

### 6.2 各 gold tier 的 satisfied / violated / unknown（主口径，约束级合并）

| 模型 | Must (n) S / V / U | Preferred (n) S / V / U | Optional (n) S / V / U | Entity (n) S / V / U |
|---|---|---|---|---|
| Claude Sonnet 4.6 | (928) 85.3 / 14.3 / 0.3 | (112) 73.2 / 26.8 / 0.0 | (823) 96.1 / 3.2 / 0.7 | (15) 86.7 / 0.0 / 13.3 |
| Claude Sonnet 5 | (928) 85.0 / 13.8 / 1.2 | (112) 65.2 / 34.8 / 0.0 | (823) 93.4 / 3.0 / 3.5 | (15) 53.3 / 26.7 / 20.0 |
| DeepSeek R1 | (928) 79.0 / 19.6 / 1.4 | (112) 67.0 / 33.0 / 0.0 | (823) 93.2 / 3.2 / 3.6 | (15) 46.7 / 0.0 / 53.3 |
| GPT-5.6 Luna | (928) 83.8 / 15.4 / 0.8 | (112) 63.4 / 33.0 / 3.6 | (823) 97.0 / 2.2 / 0.9 | (15) 86.7 / 0.0 / 13.3 |
| GPT-5.6 Sol | (928) 88.1 / 10.6 / 1.3 | (112) 68.8 / 25.0 / 6.2 | (823) 97.2 / 1.9 / 0.9 | (15) 93.3 / 6.7 / 0.0 |
| Grok 4.6 | (928) 89.0 / 9.9 / 1.1 | (112) 65.2 / 32.1 / 2.7 | (823) 93.8 / 3.8 / 2.4 | (15) 66.7 / 0.0 / 33.3 |
| Kimi K2.5 | (928) 78.8 / 19.9 / 1.3 | (112) 72.3 / 27.7 / 0.0 | (823) 94.7 / 3.0 / 2.3 | (15) 93.3 / 0.0 / 6.7 |
| MiniMax M2.5 | (928) 74.8 / 23.9 / 1.3 | (112) 69.6 / 30.4 / 0.0 | (823) 93.6 / 2.3 / 4.1 | (15) 40.0 / 26.7 / 33.3 |
| Nova Pro | (928) 73.1 / 24.4 / 2.6 | (112) 67.9 / 30.4 / 1.8 | (823) 87.7 / 2.4 / 9.8 | (15) 26.7 / 46.7 / 26.7 |
| Qwen3 235B | (928) 75.3 / 23.7 / 1.0 | (112) 62.5 / 36.6 / 0.9 | (823) 91.4 / 3.8 / 4.9 | (15) 46.7 / 0.0 / 53.3 |
| Qwen3 32B | (928) 65.6 / 32.0 / 2.4 | (112) 64.3 / 31.2 / 4.5 | (823) 91.9 / 2.7 / 5.5 | (15) 40.0 / 33.3 / 26.7 |

### 6.3 Intention 细节

| 模型 | micro P / R / F1 | 平均预测条数 | 平均 gold 字段 | 未对齐条数/轮 | Change micro P / R / F1 | Change recall add / override / relax | Priority acc.（未识别计错） | Priority acc.（排除背景字段） | Reprioritize acc. |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| Claude Sonnet 4.6 | 87.8 / 82.6 / 85.1 | 11.3 | 11.7 | 1.04 | 70.5 / 85.3 / 77.2 | 88.0 / 77.0 / 77.8 | 44.8 | 85.5 | 4.1 |
| Claude Sonnet 5 | 90.1 / 84.5 / 87.2 | 11.2 | 11.7 | 0.87 | 70.8 / 86.7 / 78.0 | 90.4 / 73.8 / 88.9 | 44.5 | 83.0 | 3.4 |
| DeepSeek R1 | 90.4 / 56.2 / 69.3 | 7.4 | 11.7 | 0.38 | 80.0 / 77.3 / 78.6 | 79.8 / 68.9 / 77.8 | 36.7 | 82.9 | 1.4 |
| GPT-5.6 Luna | 89.4 / 82.0 / 85.5 | 11.1 | 11.7 | 0.87 | 71.6 / 90.3 / 79.9 | 94.2 / 80.3 / 66.7 | 45.5 | 84.4 | 2.7 |
| GPT-5.6 Sol | 89.0 / 85.4 / 87.2 | 11.8 | 11.7 | 1.02 | 72.5 / 86.3 / 78.8 | 91.3 / 72.1 / 66.7 | 45.6 | 84.4 | 4.7 |
| Grok 4.6 | 87.1 / 85.2 / 86.1 | 11.8 | 11.7 | 1.15 | 71.6 / 86.7 / 78.4 | 92.3 / 68.9 / 77.8 | 44.8 | 82.1 | 3.4 |
| Kimi K2.5 | 88.0 / 80.4 / 84.0 | 11.2 | 11.7 | 0.99 | 64.4 / 83.8 / 72.8 | 87.0 / 73.8 / 77.8 | 41.5 | 77.6 | 4.1 |
| MiniMax M2.5 | 89.1 / 74.1 / 80.9 | 10.2 | 11.7 | 0.59 | 69.3 / 79.1 / 73.9 | 82.2 / 68.9 / 77.8 | 38.5 | 74.6 | 3.4 |
| Nova Pro | 85.8 / 55.0 / 67.0 | 7.7 | 11.7 | 0.59 | 66.9 / 69.1 / 68.0 | 70.7 / 62.3 / 77.8 | 36.6 | 84.9 | 1.4 |
| Qwen3 235B | 87.3 / 76.8 / 81.7 | 10.6 | 11.7 | 0.82 | 68.3 / 80.6 / 73.9 | 83.2 / 70.5 / 88.9 | 39.1 | 76.1 | 3.4 |
| Qwen3 32B | 80.9 / 68.5 / 74.2 | 10.2 | 11.7 | 1.12 | 50.5 / 73.0 / 59.7 | 77.9 / 57.4 / 66.7 | 39.8 | 82.4 | 1.4 |

每个模型的 Change 轮数都是 126。三类变化的 gold 数量分别为：add 约 208，override 61，relax 9（relax 只出现在 shard1，样本很小）。Reprioritize acc. 衡量的是：本轮被 gold 重新定级的字段，agent 的 tier 是否等于新的 tier。这类重新定级绝大多数是背景字段 high→low，所以接近 0。

### 6.4 Priority tier 混淆（行 = gold tier，单元格 = 预测 tier 占比，最后一列为未识别）

| 模型 | gold Must → must / pref / opt / missed | gold Preferred → must / pref / opt / missed | gold Optional → must / pref / opt / missed |
|---|---|---|---|
| Claude Sonnet 4.6 | (928) 88.0 / 1.7 / 0.0 / 10.2 | (112) 67.9 / 16.1 / 0.9 / 15.2 | (823) 79.8 / 1.6 / 0.0 / 18.6 |
| Claude Sonnet 5 | (928) 86.9 / 4.1 / 0.1 / 8.9 | (112) 65.2 / 20.5 / 3.6 / 10.7 | (823) 79.8 / 1.6 / 0.0 / 18.6 |
| DeepSeek R1 | (928) 72.3 / 3.7 / 0.1 / 23.9 | (112) 52.7 / 10.7 / 0.9 / 35.7 | (823) 38.3 / 1.5 / 0.0 / 60.3 |
| GPT-5.6 Luna | (928) 89.0 / 3.2 / 0.0 / 7.8 | (112) 65.2 / 19.6 / 0.0 / 15.2 | (823) 73.9 / 1.3 / 0.0 / 24.8 |
| GPT-5.6 Sol | (928) 88.1 / 3.7 / 0.1 / 8.1 | (112) 62.5 / 26.8 / 0.9 / 9.8 | (823) 81.2 / 1.5 / 0.1 / 17.3 |
| Grok 4.6 | (928) 87.4 / 5.3 / 0.1 / 7.2 | (112) 64.3 / 21.4 / 2.7 / 11.6 | (823) 81.3 / 1.7 / 0.0 / 17.0 |
| Kimi K2.5 | (928) 80.5 / 9.6 / 0.2 / 9.7 | (112) 61.6 / 23.2 / 2.7 / 12.5 | (823) 72.3 / 2.3 / 0.1 / 25.3 |
| MiniMax M2.5 | (928) 75.0 / 11.5 / 0.1 / 13.4 | (112) 57.1 / 19.6 / 0.9 / 22.3 | (823) 68.2 / 1.6 / 0.0 / 30.3 |
| Nova Pro | (928) 72.5 / 1.6 / 0.1 / 25.8 | (112) 52.7 / 8.0 / 0.0 / 39.3 | (823) 41.6 / 1.2 / 0.0 / 57.2 |
| Qwen3 235B | (928) 75.9 / 9.5 / 0.9 / 13.8 | (112) 54.5 / 19.6 / 3.6 / 22.3 | (823) 74.5 / 1.6 / 0.2 / 23.7 |
| Qwen3 32B | (928) 79.1 / 3.9 / 0.2 / 16.8 | (112) 57.1 / 7.1 / 0.9 / 34.8 | (823) 67.0 / 1.0 / 0.0 / 32.1 |

### 6.5 按 linguistic style 拆分（主口径；success / intent F1 / change recall）

| 模型 | elliptical (24) | explicit (104) | partial (31) |
|---|---|---|---|
| Claude Sonnet 4.6 | 33.3 / 83.7 / 95.1 | 48.1 / 84.7 / 94.0 | 25.8 / 83.3 / 73.0 |
| Claude Sonnet 5 | 37.5 / 85.5 / 93.1 | 49.0 / 86.8 / 95.2 | 29.0 / 86.5 / 76.0 |
| DeepSeek R1 | 29.2 / 65.6 / 88.9 | 32.7 / 72.0 / 87.8 | 16.1 / 63.9 / 61.3 |
| GPT-5.6 Luna | 25.0 / 83.4 / 95.1 | 48.1 / 84.8 / 97.3 | 25.8 / 83.3 / 74.3 |
| GPT-5.6 Sol | 54.2 / 87.1 / 91.0 | 58.7 / 86.2 / 96.1 | 35.5 / 86.0 / 62.7 |
| Grok 4.6 | 50.0 / 83.6 / 95.1 | 62.5 / 85.7 / 94.6 | 35.5 / 86.2 / 65.3 |
| Kimi K2.5 | 20.8 / 83.8 / 95.1 | 38.5 / 83.7 / 92.5 | 16.1 / 81.8 / 69.0 |
| MiniMax M2.5 | 20.8 / 77.4 / 84.7 | 33.7 / 82.5 / 91.3 | 12.9 / 76.1 / 58.0 |
| Nova Pro | 25.0 / 62.5 / 88.9 | 25.0 / 68.7 / 82.1 | 3.2 / 64.1 / 52.7 |
| Qwen3 235B | 16.7 / 79.6 / 88.9 | 26.9 / 82.3 / 92.7 | 6.5 / 78.4 / 56.0 |
| Qwen3 32B | 8.3 / 67.4 / 66.0 | 21.2 / 76.9 / 86.4 | 3.2 / 68.7 / 55.3 |

另有 1 轮没有 linguistic_style 标注，未列入此表。

### 6.6 Judge 一致性

对同一个 gold 字段，主口径 Pass A 的 `value_match` 与 Pass B 的"已覆盖"判定一致率为 92.3–97.1%。按模型分别为：Claude Sonnet 4.6 97.1 · Claude Sonnet 5 93.8 · DeepSeek R1 95.6 · GPT-5.6 Luna 94.3 · GPT-5.6 Sol 96.3 · Grok 4.6 96.3 · Kimi K2.5 94.7 · MiniMax M2.5 95.7 · Nova Pro 92.3 · Qwen3 235B 95.2 · Qwen3 32B 93.1。

### 6.7 与截图口径的粗略对照（只看 shard1 + shard2）

| 模型 | 本次（主口径） | 截图 |
|---|---|---|
| Claude Sonnet 4.6 | 47/96 | 48/95 |
| Claude Sonnet 5 | 51/96 | 63/95 |
| DeepSeek R1 | 31/96 | 31/95 |
| GPT-5.6 Luna | 43/96 | 45/95 |
| GPT-5.6 Sol | 52/96 | 61/95 |
| Grok 4.6 | 55/96 | 68/95 |
| Kimi K2.5 | 39/96 | 22/95 |
| MiniMax M2.5 | 30/96 | 14/95 |
| Nova Pro | 25/96 | 4/95 |
| Qwen3 235B | 24/96 | 20/95 |
| Qwen3 32B | 18/96 | 6/95 |

截图用的是哪个 judge、什么证据、怎样定义"成功"都不清楚，shard1 的分母也是 31 而不是 32，所以两者不能直接比较。Claude Sonnet 4.6、DeepSeek R1 和 GPT-5.6 Luna 与截图的差距在 2 轮以内；Claude Sonnet 5、GPT-5.6 Sol 和 Grok 4.6 本次低 9–13 轮；Kimi K2.5、MiniMax M2.5、Nova Pro 和 Qwen3 32B 本次高 12–21 轮。其中 Nova Pro 和 Qwen3 32B 的行程基本不写价格，它们在 action-only 口径下的 budget 满足率只有 3.8% 和 11.4%（§5.1），因此判法不同时受影响最大。其余差异的来源无法从截图判断。

## 7. 观察

1. **Budget 是 Action 层的主要瓶颈。** 按数据库计价，各模型的行程只有 52–85% 落在预算之内，而除 budget 以外的 must 约束满足率在 68–90% 之间，每个模型都高于它自己的 budget 满足率。模型之间的差距主要来自是否会算账：GPT-5.6 Sol 和 Grok 4.6 的 budget 满足率分别为 85.4% 和 77.8%。DeepSeek R1、Kimi K2.5 和 MiniMax M2.5 则常常自报偏低的价格。
2. **Intention 和 Action 是脱节的。** GPT-5.6 Luna 与 Sol 的 intention F1 几乎相同（84.2 对 86.2），但 action 成功率分别是 40.6% 和 53.8%。DeepSeek R1 的 precision 最高（89.9），但因为漏报约束，recall 只有 59.2。
3. **变化类型之间的难度差别很大。** 新增约束（add）的 recall 在 71–94 之间，覆盖改写（override）只有 57–80。`partial` 风格的轮次在所有模型上都明显更难：change recall 比 explicit 低 21–37 个点，严格成功率也更低。
4. **Priority 很难区分模型。** agent 几乎从不使用 optional（在所有预测条目中只占 0.1–1.4%），gold 标为 Preferred 的字段有 52–68% 被 agent 标成 must。一旦排除背景字段的降级约定，各模型都在 75–86 之间。
5. **Turn Exact 与 Priority Turn Exact 几乎全为 0。** 每轮平均有 11.7 个 gold 字段，逐轮完全一致基本不可能达到，所以这两个指标在当前 gold 粒度下没有区分度，建议不要作为主指标。

## 8. 注意事项与数据问题

- **Gold 静默删除字段。** `travelplanner_test_0873` 的 `required_cities: ["New Orleans"]` 在 t1 被加入，从 t3 起就从 gold 中消失了，而 `gold_delta` 里没有任何对应记录。agent 延续这个约束是合理的，但对齐时它会指向一个"本轮不存在"的字段。处理方式是把这类对齐视为未对齐，计作 FP，并在 `stale_alignments` 中计数。**建议标注侧核查。**
- **Gold 缺少查询参数字段。** 部分 shard1 instance 的 gold 没有 `people_number`、`org` 等查询字段（例如 test_0144），agent 正确复述这些字段时会被算作多余条目。这对所有模型的 precision 影响相同。
- **背景字段的降级约定。** 见 §0 和 §6.3。这个约定会大幅拉低 Priority 指标，也让 shard2/3 的 Optional 满足率基本等同于"背景字段是否满足"。
- **空真成功。** 有 2 轮（shard1 test_0081 t3、test_0129 t0）没有 must_have 约束，它们的严格成功是空真的。
- **判定口径。** Pass B（对齐 prompt）和确定性 budget 都是本次新增的，不属于 main。Pass A 的 prompt 和计分完全使用 main，只有 grounded 和候选池两种口径改变了证据内容。
- **确定性计价的局限。**
  - 名称匹配不到的槽位计 $0，每个模型平均每轮 0.6–4.0 个；GPT-5.6 Sol 的影响已在 §5.3 量化。
  - 只写了一个城市的自驾或出租车路线无法定价。
  - 没有检查"所选实体是否真实存在于数据库"这类 commonsense 约束。
- **Judge 噪声。** 每种口径只跑了一次，没有做重复采样。三次 Pass A 输入的 intent 完全相同，main-scorer 的 intention 数字相差 ≤ 2.1 个点。grounded 与候选池两种口径之间，每个模型有 3–14 轮成功与否发生翻转，且方向基本对称（§5.5）。所以严格成功数相差几轮以内的模型，应视为同一水平。

## 9. 复现与文件

```bash
set -a; source .env.llm; set +a   # 提供 OPENROUTER_API_KEY
RUN="travelplanner_v4_agent_trajectories_20260919_094132 2"
OUT=annotation/reports/travelplanner_v4_agent_eval_gpt6luna_20260923
.venv/bin/python scripts/judge_travelplanner_v4_trajectories.py --run-dir "$RUN" --output-dir $OUT                      # Pass A(action-only) + Pass B
.venv/bin/python scripts/judge_travelplanner_v4_trajectories.py --run-dir "$RUN" --output-dir $OUT --evidence grounded  # Pass A(grounded)
.venv/bin/python scripts/judge_travelplanner_v4_trajectories.py --run-dir "$RUN" --output-dir $OUT --evidence pool      # Pass A(候选池，对照实验)
.venv/bin/python scripts/report_travelplanner_v4_eval.py $OUT
```

judge 的输出按 instance 缓存在 `judgments/` 下，重跑时只会补上缺失的部分。本次的缓存**不在仓库里**（2026-09-24 清理 git 历史时移除，只保存在作者本地）。

| 文件 | 内容 |
|---|---|
| `scripts/judge_travelplanner_v4_trajectories.py` | 配对 gold、校验 SHA、调用 judge（Pass A 两种证据、Pass B），输出 scored rows |
| `scripts/report_travelplanner_v4_eval.py` | 确定性计价、全部指标计算，生成 `metrics.json` 和 `tables.md` |
| `judgments/*.json` | 286 个 instance 的原始 judge 输出（`pass_a`、`pass_a_grounded`、`pass_a_pool`、`pass_b`），**不在仓库里**（2026-09-24 清理 git 历史时移除，只保存在作者本地） |
| `scored_rows.json` / `scored_rows_grounded.json` / `scored_rows_pool.json` | 逐轮行：gold、预测、action、证据、judge 输出、main scorer 分数。不在仓库里，需要有 judge 缓存才能用上面的命令重建 |
| `metrics.json` | 五种口径（action、grounded、grounded_det、pool、pool_det）× 11 个模型的全部汇总指标，以及 `grounded_vs_pool` 对照结果 |
| `tables.md` | 由脚本生成的全部表格 |
| `prompt_examples/` | `travelplanner_test_0144`（Claude Sonnet 5）在三种 Pass A 口径和 Pass B 下完整拼好的 judge prompt（§2.2） |

**花费（OpenRouter 实际扣费）**：第一天的 action-only、Pass B 和 grounded 三次全量运行合计约 $3.64；候选池对照实验约 $2.18。
