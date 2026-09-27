# Priority-aware Action Success：shard 1 直接评定

本次由当前会话的 Assistant 阅读原始方案、人工约束和环境记录逐项评定，未调用外部 Judge API。Python 仅负责计数、词典序比较和汇总。这是一次可复核的直接评定，不是新增人工标注或多评审一致性实验。

| 数据 | Case | 总轮数 | 成功 | 失败 | 暂无法判定 | 可判定轮次成功率 |
|---|---:|---:|---:|---:|---:|---:|
| WebShop shard 001 | 8 | 40 | 20 | 5 | 15 | **20/25 = 80.00%** |
| TravelPlanner shard 1 | 6 | 32 | 12 | 11 | 9 | **12/23 = 52.17%** |

成功率分母仅包含能够确定成功或失败的轮次，不是全部轮次。全部轮次中的已确定成功占比分别为 20/40 和 12/32；其余未知结果不能默认为失败或成功。

## 评估范围及规则

- WebShop 的 8 个 ID 来自 `annotation/data/webshop_shard1_5_reviewed.zip::shard1_5/shard_001_human_annotated.json`，对应 GPT-5.6 Sol 输出实际在 **bundle2.json**，不是最初给出的 bundle1。筛选后的 40 轮对话和当前人工意图均与源标注逐轮核对一致。
- TravelPlanner 使用指定的 `gpt-5.6-sol__shard1.json`。原标注从 Git HEAD 只读获取，原始文件 SHA256 与 Agent 输出记录的 dataset_sha256 完全一致；没有恢复或覆盖用户已删除的标注文件。
- Agent 和 Gold 使用同一套人工 active constraints。一个人工约束字段计一项；复合字段必须整体满足。high→Must、medium→Preferred；Optional 不参与计分，不做穷尽审计。未合并人工分别列出的 `ready_eat` / `ready_to_eat` 等字段。
- 比较 `(Must满足数, Preferred满足数)`，前者优先；不加权。Agent 结果无效，或已知 Must 违反没有明确承认，即使数量占优也失败。
- 未知属性不是满足，也不直接推断为违反。明示要求“明确声明某属性”时，缺少声明本身可判违反。不能确定的满足数表示为区间；仅当区间内所有可能结果都给出相同胜负时判定。
- 没有 Gold 的轮次不以 `(0,0)` 代替。未标优先级或重复标层级不擅自补标；但若该项在双方均已满足，分到任何层都给双方相同增量，不影响比较，可以判定。
- 商品事实来自实际商品记录及已选 options，不把 Agent 自述或候选排序理由当作属性证据。环境商品价格采用冻结记录，不推断现实商店的变体价格。
- 旅行费用采用仓库口径：机票及餐费按人数、住宿按夜计算。自备餐等缺少价格时，不把小计误当作完整总价；小计已经超预算则可确定违反。另核对跨城路线、入住容量和连续晚数。
- 住宿有效性沿用仓库按行程连续入住晚数检查 minimum nights 的规则。`0144 t2` 和 `0873 t0` 虽在 rationale 表示愿意支付额外空置晚数，行程入住晚数仍不足，因此主结果判无效；若采用“支付空置晚数即视为有效完整预订”的不同规则，这两轮需要改判。
- Gold 的 `confirmed=false` 不当作“已验证全满足”；实际核对其属性、日期、费用和容量。Gold 无效问题单独保留，不要求 Agent 复现 Gold 的缺陷。

## 暂无法判定的轮次

WebShop 15 轮：

- **10 轮没有 Gold 商品**：02449 的 t1、t2；02702 的 t3；05038 的全部 t0–t6。
- **5 轮属性不足以确定比较**：01181 的 t1–t3（后跟覆盖、品牌、尺寸等）；02449 的 t3、t4（哑光表面等）。

TravelPlanner 9 轮：

- **2 轮没有有效 Gold 行程**：0081 t3、0873 t6。
- **3 轮预算证据不足**：0126 t2–t4，双方有未报价餐食，预算是否满足会影响比较。
- **4 轮实体约束缺优先级且无法抵消**：0357 t1–t4。已保存已标层级部分的计数，但不把它冒充完整指标。

这里的 ID 为完整 instance_id 的末四/五位，t 使用文件中的 **0-based turn_id**。

## WebShop 每个 case

| instance_id | 成功 | 失败 | 暂无法判定 |
|---|---:|---:|---:|
| webshop_goal_00000 | 2 | 2 | 0 |
| webshop_goal_00318 | 3 | 0 | 0 |
| webshop_goal_01181 | 1 | 0 | 3 |
| webshop_goal_02449 | 2 | 0 | 4 |
| webshop_goal_02702 | 4 | 1 | 1 |
| webshop_goal_05038 | 0 | 0 | 7 |
| webshop_goal_05121 | 4 | 1 | 0 |
| webshop_goal_06824 | 4 | 1 | 0 |

5 个失败轮次：

1. **00000 t2**：Agent `(2,0)`、Gold `(1,0)`，数量本可通过；但商品明确不含枕芯，未承认 Must 违反，判失败。
2. **00000 t3**：Agent 仍选 botanical/floral，Gold 要排除 floral 并要求 abstract/geometric。Agent `(1,1)`、Gold `(3,1)`，且未承认违反。
3. **02702 t0**：Agent 选 egg-white chips/crisps，Gold 是 puffed corn；当前 Must 是 puffed snacks。Agent `(0,0)`、Gold `(1,0)`，承认妥协也不能补足数量差距。
4. **05121 t2**：双方都是 vegetarian Pasta Fagioli，均不满足 Gold 已标为 Must 的 chicken/beef，Agent 未承认。**此轮存在标注提前一轮引入要求的问题**：用户 t3 才提出鸡肉/牛肉。按原始标注判失败；修正标注后会变为成功。
5. **06824 t4**：Agent 71.5 英寸、$279.99，Gold 62 英寸、$65.34。双方 Must 均满足 2 项，但 Agent 未承认当前 $100 上限被违反；其 Preferred 也只有 1 项，低于 Gold 的 3 项。

## TravelPlanner 每个 case

| instance_id | 成功 | 失败 | 暂无法判定 |
|---|---:|---:|---:|
| travelplanner_test_0081 | 2 | 3 | 1 |
| travelplanner_test_0126 | 1 | 1 | 3 |
| travelplanner_test_0129 | 3 | 3 | 0 |
| travelplanner_test_0144 | 0 | 3 | 0 |
| travelplanner_test_0357 | 1 | 0 | 4 |
| travelplanner_test_0873 | 5 | 1 | 1 |

关键例子：

- **0357 t0 成功**：Agent 超预算但明确承认，并满足四人住宿容量；Gold 预算内但容量不足。双方 Must 均满足 3 项，Agent 自身方案有效，故成功。这体现了指标允许不同的妥协方案。
- **0129 t5 成功**：Agent 在两城都安排了数据库标注为 Chinese 的餐厅，所有餐费均低于 $50/人；Gold 含 $96 的 Domino's。Agent Must 2 > Gold 1。
- **0081 t2 失败**：双方住宿评分 Must 都满足，但 Agent 仍安排 Day 1 餐厅午餐，Preferred 为 2，Gold 为 3。
- **0129 t3、t4 失败受标注冲突影响**：人工字段写的是严格 `<2 attractions/day`，但同时要求同一天两个指定景点。Agent 和 Gold 都安排两处，Agent 未承认这个字面冲突，因此按原始标注判失败；若本意是 `≤2`，两轮会改变。

## 可复核文件

- `report.md`：全部 72 轮的 Must/Preferred 数量或区间、判定和原因。
- `results.json`：逐约束的双方状态、证据、违反说明原文、有效性及标注问题。
- `jobs/`：原始约束、方案及相关环境记录；保留源文件位置。
- `judgments/`：本会话直接判定的逐轮记录，judge 标为 `assistant-direct-review`。
- `manifest.json`：输入文件哈希及范围核对。
- `scripts/record_shard1_direct_review.py`：可重放的直接评定决定；没有任何外部模型调用。

词典序、不可补偿的 Must、披露/有效性门槛、缺 Gold、未知项区间及共同未标层级抵消，共 7 项逻辑检查已通过。当前 Python 环境没有 pytest，因此以直接执行测试函数的方式运行。
