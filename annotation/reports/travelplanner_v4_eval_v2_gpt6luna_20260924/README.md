# TravelPlanner v4 评测 v2：规则化 judge（judge：GPT-6 Luna）

- 日期：2026-09-24
- 轨迹：`travelplanner_v4_agent_trajectories_20260919_094132 2/`（11 个模型 × 3 个 shard，每个模型 160 轮）
- Gold：`annotation/data/exports/travelplanner_v4/`
- Judge：`openai/gpt-6-luna`（OpenRouter）；**本报告的判定结果来自规则 v2.1**。代码和规则已经更新到 v2.2，但还没有重跑，见 §12
- 覆盖：公共基准 160/160，action 判定 1760/1760，intention 判定 1760/1760，**judge 失败 0**
- 上一版（v1，main 原版 judge）的报告在 [`../travelplanner_v4_agent_eval_gpt6luna_20260923/README.md`](../travelplanner_v4_agent_eval_gpt6luna_20260923/README.md)。**v2 口径不同，不能与 v1 的数字直接比较**，差异见 §8。

## 0. 结论速览

- **Action 层（Hard Success）**：Grok 4.6 54.4%、GPT-5.6 Sol 50.0%、Claude Sonnet 5 46.2% 位居前三，之后是 GPT-5.6 Luna 43.1% 和 Claude Sonnet 4.6 40.0%；DeepSeek R1 25.6%；Kimi K2.5、MiniMax M2.5、Qwen3 235B 在 17–18% 左右；Nova Pro 11.2%，Qwen3 32B 7.5%。
- **强弱模型的差距比 v1 明显拉大。** 两道新门槛主要打在较弱的模型上：
  - 酒店门槛（订的晚数少于最低入住晚数，或容量不够）：Qwen3 32B 有 47.5% 的轮次不过，Nova Pro 45.0%，MiniMax M2.5 33.1%；
  - 餐位覆盖（gold 行程安排了带价格的餐厅，agent 却没安排）：Kimi K2.5 有 56.3% 的轮次出现缺口，Nova Pro 53.8%。
- **Must 违反说明率（按约束，含 budget）**：模型违反的 Must 里，有多少被它明确告诉了用户。Grok 4.6 32%、Claude Sonnet 5 28%，其余多数在 13–23% 之间，Qwen3 32B 6%，Nova Pro 2%。其中 **budget 违反几乎没有被说明**：各模型的 budget 违反说明率都在 0–9% 之间（§7.2）。
- **Intention 层（语义原子，一对一匹配）**：Claude Sonnet 5 的 F1 最高，为 76.9；DeepSeek R1 和 Nova Pro 最低，约为 55–58，原因是漏报约束，recall 只有 47–49。
- **judge 质量**：unknown 比例从 v1 的 2.7% 降到 0.2%，people_number 的 unknown 从 21.8% 降到 0；可用数据库核对的判定，与数据库结论的一致率从 98.2% 升到 99.2%，其中 house rule 从 90.4% 升到 99.1%。16 个校准案例全部判对（标签为草稿，见 §6）。
- **gold 标注**：公共基准自动标出 98 条疑似标注问题，分布在 23/26 个 instance 上。此外，139 个有 gold 参考行程的轮次里，有 55 个行程自己就违反了某条 Must。

## 1. 为什么要做 v2

v1 使用 main 原版的 judge：一个 instance 只调用一次，让 LLM 直接判断所有约束。深挖 v1 的判定后，发现了以下系统性问题：

| 问题 | v1 中的表现 | v2 的处理 |
|---|---|---|
| people_number 不知道怎么判 | 行程里不写人数，21.8% 被判 unknown，两次运行之间有 16.9% 的判定互相矛盾 | 规则 6：按当晚所订住宿的容量来判 |
| 串轮 | judge 一次看到同一 instance 的所有轮次，会拿第 1 轮的住宿去判第 2 轮（test_0066、test_0017） | 每一轮单独调用 judge |
| 算不好 budget | 以"门票没标价"为由判 unknown，或者相信 agent 自报的错误总价 | 由代码计价 |
| 没人检查最低入住晚数 | 各模型有 9–75/160 轮违规，却完全没有被扣分 | 由代码做酒店门槛 |
| 自带或自做餐食 | 餐厅评分、每餐价格这类约束被判成 unknown 或违反 | 规则 9；餐位是否覆盖交给代码对照 gold 行程 |
| 误读 house rule | listing 没写 "No parties" 却被判成不允许派对，与数据库结论只有 90.4% 一致 | 规则 8：house_rules 只列禁止事项 |
| 行程没有具体时刻 | 景点"上午先去 A"这类要求被判 unknown | 规则 5：按安排在哪一天和列出的先后顺序来判 |
| 无障碍、城市内交通 | 所有模型都被判 unknown，而严格成功率把 unknown 算作失败 | 公共基准把这两类排除在审核范围外 |
| 意图层和行为层混在一起 | 行程判定的理由里写着 "the prediction omits..." | action judge 看不到意图预测，intention judge 看不到行程 |
| gold 与对话冲突 | 由 judge 每次自行决定听谁的 | 冻结的公共基准只记录冲突，另设带版本号的 gold 审核层 |

## 2. 架构

```
每个 (case, turn)，所有模型共用、冻结：
  对话 + gold 约束 + gold 行程 + 候选池 ──> 公共基准 judge ──> gold 原子、活动要求、排除范围、
                                                              gold 行程的逐项判定（可行性）、标注问题
                                          Python ──> gold 行程的计价与酒店检查

每个 (模型, turn)：
  基准 + agent 的 action + 候选池 + 代码预匹配 ──> action judge ──> 最终方案（槽位 → 候选记录）、
                                                              明确的修改、逐约束判定与证据、未满足的说明、标注问题
  gold 原子 + agent 本轮和上一轮的意图 ──────────> intention judge ──> 预测原子、一对一匹配、值是否正确、
                                                              相对上一轮是新增 / 修改 / 不变

Python：计价（含餐位覆盖）、酒店门槛、排除范围、Hard Success 两个分支、全部分数
```

- **每个 judge 看得到什么、看不到什么**：
  - action judge 看不到 agent 的意图预测；
  - intention judge 看不到任何行程；
  - 三个 judge 都看不到模型名，prompt 里只有轨迹内容。
- **每次调用只包含一轮**。v2.1 的每个 judge 都附上了截至这一轮的全部用户对话；v2.2 起 action judge 不再看对话，改为按 baseline 冻结的判定标准来判（§12）。
- **候选池**：action judge 看到的候选记录，就是 agent 当轮看到的全部搜索结果，与 agent prompt 逐字相同（v1 报告 §5.5）。代码计价用的是完整的数据库记录。
- **缓存和失败处理**：每次调用单独缓存（本次的缓存不在仓库里）；输出不合格（缺字段、字段名不对、匹配不是一对一等）最多重试 4 次；仍失败的记入 `errors.jsonl`，不计入模型的成败。本次共有 1 个公共基准、75 个 action、114 个 intention 调用是在重试后才通过校验的，最终全部成功。运行中还遇到两次失败，都已补齐：一次是 OpenRouter 账户额度用完，2,440 次调用返回 HTTP 402，充值后续跑完成；另一次是 test_0144 t2 的公共基准 judge 自造了字段名，之后在 prompt 里要求照抄 gold 字段名，重跑通过（其余 159 个公共基准生成时还没有这句要求）。

### 2.1 prompt 组成：和 v1 相比改了什么

> 本节描述的是实际运行时的 v2.1 版本。v2.2 又改了 baseline 和 action 两个 prompt，见 §12。

v1 只有一个 prompt，即 main 的 `build_judge_prompt`：一段约 685 token 的固定指令，加上这个 instance **所有轮次**的数据（每轮包括 gold、agent 的意图预测、action 和候选池）。一次调用同时判意图和行为。以 test_0144 为例，3 轮的 payload 约 2.1 万 token。

v2 拆成了三个 prompt，每个的结构都是"规则 + few-shot + 输出格式 + PAYLOAD"：

| | 公共基准 judge | action judge | intention judge |
|---|---|---|---|
| 调用粒度 | 每个 (case, turn) 一次，所有模型共用 | 每个 (模型, turn) 一次 | 每个 (模型, turn) 一次 |
| 规则 | Baseline rules 5 条，另附 Action rules 13 条（判 gold 行程时用），约 1,535 token | Action rules 13 条，约 1,011 token | Intention rules 4 条，约 310 token |
| few-shot | 2 个"排除范围"案例，约 292 token | 13–14 个约束级案例，约 1,253 token（来自同一 instance 的样本排除） | 无 |
| 输出格式 | gold 原子、活动要求、排除范围、gold 行程判定、标注问题 | 最终方案（含 `revised`）、修改、逐约束判定（含证据）、说明、标注问题 | 预测原子（来源条目、对应 gold 原子、值是否正确、变化状态） |
| PAYLOAD | 截至本轮的对话、gold 约束和档位、本轮 gold_delta、gold 行程、agent 可见的候选池 | 截至本轮的对话、需要判的约束（在审核范围内，不含 budget）和档位、公共基准的活动要求、agent 本轮的 itinerary 和 rationale、代码预匹配结果、agent 可见的候选池 | 截至本轮的对话、gold 原子、agent 本轮和上一轮的意图条目（不含 priority） |
| 以 test_0144 t1 为例 | 约 7.8k token | 约 7.3k token（payload 约 4.8k） | 约 0.9k token |

**和 v1 相比新加的内容**：
1. 你给的六条规则（§3.1），以及从 v1 错误里挖出来的规则（§3.2）：人数按容量判、事实以候选记录为准、以 gold 为准、没有具体时刻不算 unknown、house rule 的约定、自带餐食不算依据、unknown 只作最后手段。
2. few-shot 校准案例（§6）。
3. 新的输出字段：`final_plan`、`applied_revisions`、`evidence`、`unmet_constraint_disclosures`、`annotation_issues`、`gold_atoms`、`activity_requirements`、`out_of_scope`、`pred_atoms`、`change_vs_previous`。
4. 新的输入：截至本轮的全部对话、冻结的活动要求、代码预匹配的候选记录、上一轮的意图（用来判断变化）。

**从 prompt 中去掉的内容**：
- 其他轮次的数据，这是串轮的根源；
- action judge 不再看到意图预测，intention judge 不再看到行程；
- judge 不再打分：v1 里的 `priority_order_score` 是 judge 直接给的，现在由代码比较档位；
- budget 不再交给 judge 判；
- v1 里针对 WebShop 的 `no_match` 和 `human_gold_action` 说明。

三个 prompt 的完整样例在 `prompt_examples/` 目录下（test_0144 的三轮），规则原文见 [`src/eval/rules/travelplanner_v2.md`](../../../src/eval/rules/travelplanner_v2.md)。

## 3. 规则

规则文件是 [`src/eval/rules/travelplanner_v2.md`](../../../src/eval/rules/travelplanner_v2.md)，所有模型共用。文件中 `## Baseline rules`、`## Action rules`、`## Intention rules` 三节原文发给对应的 judge；`## Enforced in code` 一节只作说明，不发给 judge。

### 3.1 你给的六条规则分别怎么实现

| 规则 | 实现方式 |
|---|---|
| Gold 没安排的餐位允许省略；Gold 有报价的餐位而 Agent 缺项，保留预算覆盖失败 | **代码** `meal_coverage_gaps`：按天对齐 gold 行程和 agent 行程，gold 在某餐位安排了能在数据库里找到价格的餐厅，agent 这一餐却为空或找不到对应记录，就记为覆盖缺口，budget 判违反。gold 为空的餐位允许省略 |
| 酒店最低入住、人数容量、要求的入住日期统一校验；不满足最低入住直接失败 | 最低入住晚数和容量由**代码**检查（`hotel_validity`），不通过则该轮直接失败；入住日期由 **action judge** 按实际订了哪几晚来判（规则 7） |
| 明确的最终修改覆盖旧选择，并重算整份方案 | **action judge** 规则 1：只采用 action 里明确写出的修改，记入 `applied_revisions` 并附原文，同时把被改动的槽位标为 `revised`；代码按修改后的最终方案重新计价，并重新检查酒店 |
| 活动依据用户历史对话审核 | **公共基准** 规则 2：从截至本轮的对话中提取用户明确说过的活动要求，冻结后所有模型共用；**action judge** 规则 5 依据这份要求来判 |
| 无法判断的无障碍、城市内交通可行性排除 | **公共基准** 规则 3：只有这两类可以放进 `out_of_scope`。本次被排除的是实体约束 mobility 10 轮、accessibility 2 轮，以及 local_transportation 3 轮 |
| 违反 Must 却未明确说明未满足，直接失败；说明冲突本身不代表通过 | **action judge** 规则 12 返回 `unmet_constraint_disclosures`，并附原文；**代码**按 §5 的 Hard Success 分支使用这些说明。说明本身不会改变约束的判定 |

### 3.2 从 v1 错误中挖出来补充的规则（action judge）

- 规则 2：每个槽位都要对应到候选记录的确切名称，一晚可以对应多处住宿（团队分住的情况）；
- 规则 3：价格、评分、房型等事实一律以候选记录为准，不信 agent 自报；数据库里不存在的实体，按违反处理，不判 unknown；
- 规则 4：以 gold 为准，对话只用来理解约束的适用范围；按城市给的值只适用于那个城市；
- 规则 6：people_number 按容量判；
- 规则 7：最低入住晚数和容量只作为门槛，不影响房型、评分这些约束的判定。v2.0 曾经让它们连带影响，导致房型与数据库结论不一致，v2.1 改正了这一点；
- 规则 8：house rule 的约定；
- 规则 9：自带或自做的餐食既不算违反、也不能当作满足餐厅类约束的依据；
- 规则 11：unknown 只在候选记录确实缺少判断所需的属性时才能用。

### 3.3 待确认的默认设置

以下几点我按默认值实现了，已写在规则文件开头，请你和队友确认：

1. **Not feasible 分支。** gold 行程也违反的 Must，agent 违反了但明确说明了，算通过；gold 行程能满足的 Must，agent 违反了，说明了也算失败。
2. **house rule。** 只有 listing 写了 "No X" 才算不允许 X；写了 "No visitors" 不影响"允许派对"的要求。
3. **gold 行程全部是 `confirmed=false`**，但仍被用作可行性判断和餐位覆盖的参照。
4. **intention 的"部分正确"。** 例如 agent 写 "Drive from A to B"，gold 原子是 "Self-driving A to B; own car"，目前判为值不匹配。这一条尚未校准，见 §9。

## 4. 返回结构

队友提出的每个字段，都能在这里找到对应：

| 字段 | 由谁产生 | 内容 |
|---|---|---|
| `final_plan` | action judge | 每个交通、餐食、住宿槽位的最终文本、对应的候选记录名称，以及 `revised` 标记 |
| `applied_revisions` | action judge | 明确修改的原文和作用 |
| `constraint_judgments` | action judge | 本轮在审核范围内、除 budget 外的每个 gold 约束：`satisfied / violated / unknown`，加证据和简短理由 |
| `unmet_constraint_disclosures` | action judge | agent 明确承认未满足的约束，加原文 |
| `annotation_issues` | 公共基准 judge、action judge | 数据问题，只记录，不修正 |
| `hotel_checks` | Python | 按住宿逐晚、逐连续段计算最低入住和容量，存在 `scored_rows.json` 的 `action.hotel` |
| `meal_coverage` | Python | 餐位覆盖缺口，存在 `action.budget.coverage_gaps` |
| budget | Python | 按最终方案计价，存在 `action.budget`（总价、未定价槽位、覆盖缺口） |
| gold 原子、活动要求、排除范围、gold 行程判定 | 公共基准 judge | 在 `baseline/<instance>__t<turn>.json` |
| `pred_atoms` | intention judge | 预测原子：对应的预测条目、对应的 gold 原子、值是否正确、相对上一轮的变化 |

## 5. 计分

所有分数都由 [`src/eval/travelplanner_eval_v2.py`](../../../src/eval/travelplanner_eval_v2.py) 计算，judge 不给任何分数。

**Action 层（每轮）**

- **审核范围**：gold 约束去掉公共基准排除的字段；排除的数量单独统计。
- **可行性**：
  - `feasible`：gold 行程满足所有在审核范围内的 Must，budget 由代码判；
  - `not_feasible`：gold 行程违反了至少一个 Must；
  - `gold_plan_missing`：该轮没有 gold 行程（21 轮），按 feasible 分支处理，并单独计数。
- **Hard Success**：
  - feasible 或无 gold 行程：所有 Must 都满足，并且通过酒店门槛；
  - not_feasible：gold 行程能满足的 Must 必须满足；gold 行程也违反的 Must，只有在 agent 违反且明确说明的情况下才放行；同时必须通过酒店门槛。
  - 在两个分支里，没有说明的 Must 违反都直接失败；unknown 一律按失败计。
- **Strict**：所有 Must 都满足，不考虑酒店门槛，也不考虑 not_feasible 分支。作为参照。
- **Must 满足率**：每轮先算在审核范围内的 Must 中满足的比例，再对轮次取宏平均。
- **Preferred / Optional 满足率**：**只在 Hard Success 的轮次上**计算，再对轮次取宏平均；没有该档约束的轮次计入 excluded。
- **诊断项**：酒店门槛（最低入住 / 容量）、budget 满足、餐位覆盖缺口、"有 Must 违反的轮里全部都说明了"、"存在没说明的 Must 违反"。

**Intention 层（每轮，基于语义原子）**

- **P / R / F1**：P = 值正确且对上了 gold 的预测原子数 / 预测原子总数；R = 同一个数 / gold 原子总数。因为是一对一匹配，这两个分子相同。先逐轮计算 F1，再取宏平均；同时给出 micro 版本。
- **Change**：只统计 t≥1、且本轮 `gold_delta` 中有 add / override / relax 的轮次，每个模型 126 轮。
  - gold 的变化原子：来源字段在本轮发生了变化的 gold 原子。
  - agent 声称的变化：intention judge 标为 new 或 changed 的预测原子，但排除"值正确地对上了未变化的 gold 原子"的那些，因为那是补上以前漏的内容，不算提出了变化。
  - R = 被正确覆盖的变化原子数 / gold 的变化原子数；P = 声称的变化中正确对上变化原子的数量 / 声称的变化数；没有声称任何变化时 P 记为 0。
- **Priority Accuracy**：在对上了 gold 的原子里（值对不对都算），预测原子来源条目的 priority 与 gold 档位一致的比例。**由代码比较**，judge 看不到档位。
- **Turn Exact / Priority Turn Exact**：定义同 v1，单位换成原子。

**所有指标都同时给出分子、分母和 excluded 数量**，见 `metrics.json`；下面的表里写成"百分比（分子/分母）"。合并各 shard 时，是合并逐轮记录后再计算，不是把各 shard 的百分比取平均。

## 6. few-shot 与校准集

- **校准集**：[`annotation/data/travelplanner_judge_calibration_v1.json`](../../../annotation/data/travelplanner_judge_calibration_v1.json)，共 16 个边界案例，都来自 v1 中判错或容易判错的地方，每个都对照数据库和对话核实过。**标签是我写的草稿，需要人工复核。**
- **它同时也是 few-shot 的来源**：
  - action 类 14 个案例放进 action judge 的 prompt；
  - 2 个"排除范围"案例放进公共基准 judge 的 prompt；
  - 每个案例只给"要点 + 约束 + 证据 + 正确判定"，不给整轮上下文；budget 不做 few-shot。
- **不给 judge 看它自己的题**：判某个 instance 时，来自同一个 instance 的案例会被排除，所以校准集上的结果并不是看过答案之后做出来的。
- **校准结果**：16/16。但这些案例的"要点"和规则写得很像，而且样本量小，只能说明 judge 学会了这 16 类情况，不能证明它整体无误。

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

## 7. 结果

### 7.1 Action level

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

### 7.2 Action 门槛与诊断

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

- **"not-feasible 轮"与"无 gold 行程"** 由公共基准决定，所有模型相同：not-feasible 55 轮，无 gold 行程 21 轮。
- **Must 违反说明率（按约束，含 budget）**：分母是这个模型在所有轮次里违反的 Must 条数，分子是其中 agent 明确告诉了用户的条数。budget 也计算在内：即使违反的原因是餐位覆盖缺口，agent 也知道自己哪一餐没安排餐厅，可以说明"这一餐未计入预算"。
- **其中 budget 违反说明率**：只看 budget。各模型几乎都不承认超预算，包括 agent 自己就能算出来的超支。
- **存在未说明 Must 违反的轮**：该轮至少有一条 Must 违反没有被说明。这类轮次在两个分支里都直接判失败。
- **judge 能查出没被说明的违反。** 有没有违反，是 judge 拿行程对照约束和候选记录独立判断的，budget、最低入住晚数和容量由代码计算；agent 有没有说明，是另外单独标注的，不影响判定（规则 12）。说明只在 Hard Success 的 Not feasible 分支里起作用。

### 7.3 两道新门槛各自的影响（成功轮数 /160，用已保存的判定重新计算，不重新调用 judge）

| 模型 | v1 主口径 | v2 不加门槛 | v2 + 餐位覆盖 | v2 + 酒店门槛 | v2 完整 |
|---|---:|---:|---:|---:|---:|
| Claude Sonnet 4.6 | 67 | 73 | 70 | 66 | 64 |
| Claude Sonnet 5 | 70 | 79 | 76 | 77 | 74 |
| DeepSeek R1 | 47 | 49 | 42 | 48 | 41 |
| GPT-5.6 Luna | 65 | 75 | 73 | 71 | 69 |
| GPT-5.6 Sol | 86 | 104 | 89 | 94 | 80 |
| Grok 4.6 | 89 | 99 | 95 | 91 | 87 |
| Kimi K2.5 | 51 | 58 | 36 | 48 | 29 |
| MiniMax M2.5 | 45 | 58 | 43 | 38 | 29 |
| Nova Pro | 34 | 43 | 27 | 28 | 18 |
| Qwen3 235B | 35 | 45 | 34 | 33 | 28 |
| Qwen3 32B | 26 | 27 | 25 | 12 | 12 |

- **"v2 不加门槛"比 v1 高**：新 judge 几乎不再判 unknown，而且 not_feasible 分支会放行"gold 也做不到、agent 也说明了"的 Must。
- **餐位覆盖**对 GPT-5.6 Sol 的影响最大（−15），它常用"自带、自己做饭"来省预算；对 Kimi、Nova、MiniMax 的影响也很大。
- **酒店门槛**对 Qwen3 32B 的影响最大（−15），它有 76 轮订的晚数少于最低入住晚数。

### 7.4 Intention level

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

## 8. v2 与 v1 的数字为什么不能直接比较

- **Action**：审核范围不同（v2 排除了无障碍和城市内交通）；成功的定义不同（多了酒店门槛、餐位覆盖和 not_feasible 分支）；判定方式不同（每轮单独调用、有规则、有 few-shot）。
- **Intention**：单位不同。v1 是"预测条目对 gold 字段"，允许部分匹配、允许多对一；v2 是"语义原子一对一"，值必须完整正确。gold 原子平均每轮 13.8 个，比 v1 的 11.7 个 gold 字段多，所以 v2 的 P 比 v1 低 7.5–16.2 个点，R 低 8.3–12.9 个点。模型之间的 F1 排序大体没变，只是 Claude Sonnet 4.6 上升到第 2，GPT-5.6 Sol 和 Luna 相对下降。

## 9. judge 质量：v1 与 v2 对比（11 个模型合并，同一批约束）

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

"DB agree"的分母，是能用数据库客观核对的判定（房型、评分、house rule），分子是 judge 与数据库结论一致的数量。其余字段没有客观答案，只能看 unknown 比例。

**尚未校准的部分：**
- intention judge 目前没有校准案例。已知一处尺度问题：agent 少写了一个细节（例如 "own car"）时，整条原子会被判为值不匹配。
- 带时间窗的 schedule 约束（例如"14:00–16:00 不安排活动"），行程格式无法表达，目前仍有 1.6% 判 unknown。

## 10. 标注问题

- **自动发现**：公共基准共标出 98 条疑似标注问题，分布在 23 个 instance 上，存在 [`annotation_issues.json`](annotation_issues.json)。其中较典型的有：
  - test_0873 t4：gold 预算是 $1,500，用户已经提到 $2,000；
  - test_0081 t4：gold 指定的回程航班 10:57 到达，和用户要求的"10 点前到"冲突；
  - test_0017 t6：gold 的评分下限是 3.4，用户说的是 3.7；
  - test_0144 t1：用户原话是 "launch there"（多半是 "lunch" 的笔误），gold 照原样写成了 "launch at Gunnison"。
- **gold 审核层**：`annotation/data/exports/travelplanner_v4/gold_audit_v1.json`，**不在仓库里**（2026-09-24 清理 git 历史时移除，只保存在作者本地）。代码支持这个文件：它不存在时，gold 保持原样；需要修正 gold 时，按 `apply_gold_audit` 的格式重新建立即可（条目包含 `instance_id`、`turns`、`status`、`apply.remove_fields` / `apply.set_fields`，只有 `status` 为 `applied` 的条目才会生效）。当时记录的 3 个问题是：0357 的 cuisine、0873 的 required_cities，以及 0081 t3 的候选池为空，另见 `annotation_issues.json`。
- **gold 行程本身的问题**：139 个有 gold 行程的轮次里，有 55 个在审核范围内违反了某条 Must。这可能是标注时就做不到，也可能是标注本身有误，建议标注侧复核。

## 11. 复现、文件与花费

```bash
set -a; source .env.llm; set +a
RUN="travelplanner_v4_agent_trajectories_20260919_094132 2"
OUT=annotation/reports/travelplanner_v4_eval_v2_gpt6luna_20260924
.venv/bin/python scripts/run_travelplanner_eval_v2.py prepare   --run-dir "$RUN" --out $OUT
.venv/bin/python scripts/run_travelplanner_eval_v2.py run       --run-dir "$RUN" --out $OUT --parallelism 32 --timeout 300
.venv/bin/python scripts/run_travelplanner_eval_v2.py summarize --run-dir "$RUN" --out $OUT \
    --compare-v1 annotation/reports/travelplanner_v4_agent_eval_gpt6luna_20260923/scored_rows_pool.json
```

- **加 `--dump-prompts DIR`**：只写出拼好的 prompt，不调用 API。本目录下的 `prompt_examples/`，就是 Claude Sonnet 5 在 test_0144 三轮中三个 judge 各自的完整 prompt。
- **重复运行**：加 `--tag _r2`，缓存会写到另一个目录，可以用来测 judge 的自身稳定性。本次还没有做重复运行。

| 文件 | 内容 |
|---|---|
| `src/eval/rules/travelplanner_v2.md` | 规则（版本化） |
| `src/eval/travelplanner_eval_v2.py` | prompt、输出校验、few-shot、计分、汇总 |
| `src/eval/travelplanner_checks.py` | 计价、酒店门槛、餐位覆盖，支持一晚订多处住宿 |
| `scripts/run_travelplanner_eval_v2.py` | prepare / run / summarize 三个阶段、缓存、失败记账、judge 质量对比 |
| `scripts/mine_travelplanner_judge_cases.py` | 从已有的 judge 结果里挖可能判错的案例，作为校准和 few-shot 的来源 |
| `tests/test_travelplanner_eval_v2.py` | 8 个单元测试：引文校验、baseline 判定标准校验、团队分住、最低入住、餐位覆盖、few-shot 排除同一 instance、Hard Success 两个分支 |
| `annotation/data/travelplanner_judge_calibration_v1.json` | 校准集和 few-shot 来源（草稿标签） |
| `annotation/data/exports/travelplanner_v4/gold_audit_v1.json` | gold 审核层，**不在仓库里**（2026-09-24 清理 git 历史时移除，只保存在作者本地）；代码在它不存在时照常运行 |
| `baseline/`、`action/`、`intention/` | 每次 judge 调用的原始输出（共 3,680 个文件），**不在仓库里**（2026-09-24 清理 git 历史时移除，只保存在作者本地） |
| `metrics.json`、`tables.md` | 汇总指标（含分子、分母、excluded）和表格。逐轮的 `scored_rows.json` 不在仓库里，需要有 judge 缓存才能用 `summarize` 重建 |
| `annotation_issues.json` | 公共基准标出的标注问题 |

**花费（OpenRouter 实际扣费）**：v2 全部约 $6.10，包括一次因为改规则而作废的试跑。

## 12. v2.2 的修改（代码已改，尚未重跑）

分析 v2.1 的结果时发现了两个问题，已经在代码和规则里改好，但**还没有重新调用 judge**。重跑需要一个新的 `--out` 目录，baseline 和 action 大约 $5；intention 的 prompt 没有变。本报告 §0–§11 的判定结果仍然来自 v2.1。

| 问题 | v2.1 的情况 | v2.2 的修改 |
|---|---|---|
| 约束的理解没有真正冻结 | baseline 只冻结了排除范围、活动要求、原子拆分和可行性；**每条约束具体怎么理解，仍然由 action judge 在判每个模型时自己读对话决定**，11 个模型可能拿到不同的理解 | baseline 为每条在审核范围内的约束（除 budget）写一条 `criteria`（判定标准），把适用范围（哪几天、哪一餐、哪个城市、哪位旅客）结合对话写清楚，并附用户原文；**action judge 不再看到对话**，只按判定标准来判 |
| baseline 的引文可能是编的 | 例如 test_0144 t1 的活动要求"不要在 Gunnison 观光"是 judge 推断出来的，附的"原文"用户根本没说过。在 293 条活动要求里，有 16 条引文和用户原话对不上，其中多数只是用了省略号 | 代码校验活动要求和判定标准的每条引文，必须能在用户原话里找到（允许用 "..." 省略中间的字），找不到就判为输出不合格、让 judge 重试 |
| 说明率口径 | 按轮统计，"一轮里所有 Must 违反都说明了"才算，会低估 | 改为按约束统计的"Must 违反说明率（含 budget）"，另外单独给出"budget 违反说明率"。这一项只改了计算方法，**已经用 v2.1 的判定重新算过**，§0 和 §7.2 的数字就是新口径 |

其他相关改动：
- 规则版本升到 `travelplanner-rules-v2.2`；
- `run` 阶段发现 baseline 的规则版本与当前代码不一致时会直接报错，防止新旧混用；
- 校准案例 cal-06 的要点改成"按判定标准的范围来判"，因为 action judge 已经看不到对话；
- 新增 2 个单元测试。

重跑命令：

```bash
OUT=annotation/reports/travelplanner_v4_eval_v2_2_gpt6luna_<date>
.venv/bin/python scripts/run_travelplanner_eval_v2.py prepare   --run-dir "$RUN" --out $OUT --timeout 300
.venv/bin/python scripts/run_travelplanner_eval_v2.py run       --run-dir "$RUN" --out $OUT --parallelism 32 --timeout 300
.venv/bin/python scripts/run_travelplanner_eval_v2.py summarize --run-dir "$RUN" --out $OUT
```

