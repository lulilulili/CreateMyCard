# CC 训练数据集设计说明（v1）

> 交付物：`training/datasets/cc_train_v1.jsonl`（messages 三元组，可直接微调）
> 配套：`cc_train_v1.audit.jsonl`（逐行审计，绝不进模型输入）
> 契约版本：`training/prompts.py` 的 PROMPT_MANIFEST（改提示词必须升版本，否则本数据集作废）

## 1. 训练目标

端侧 3B 模型（qwen2.5:3b 基线 → 盘古 3B）承接 CC 模板路由的**两次调用**，
契约为 3B 友好改造版：

| 任务 | 输入 | 输出 |
|---|---|---|
| first_layer | 能力目录（字段/主题/动作候选）+ 用户 query | `{"theme","capability","fields","action"}` |
| second_layer | 确定性枚举的方案列表 + 动作数 + 同一 query | `{"plan": n}` |

**两次调用 = 两个独立任务，混在同一数据集里训练**（system prompt 天然区分任务）。
中间的确定性 Search 不训练、不出现在任何样本中：构建数据时由
`training/validators.search_templates` 从**金标**推导第二层输入（teacher forcing）。
推理时第一层答对 → 第二层输入分布与训练一致；第一层答错 → 方案列表轻微不同，
选择题任务对此天然鲁棒。

## 2. 数据来源（只源于 CC 自身）

- **场景与标签**：provider gallery 118 例的结构化元数据（businessId / candidateOutputFields /
  eventCandidates / targetTemplateId / expectsFusionBall）——**标签全部确定性推导，LLM 从不碰答案**；
  theme 标签按融球标志匹配主题表，layout-scoped 主题（如 2x2-two-support）按云侧首层规则排除。
- **query 措辞**：工作流生成（详见 §3），每个训练组合约 20 条中文自然语言变体。
- **不参与训练**：6 个整组合留出（22 例）+ 20 条 taskspec 人写用例 = 冻结评测集。

## 3. 生成管线（全自动，可复跑）

```text
emit_workitems.py     96 个训练组合 → 工作项（业务描述/字段释义/动作释义/配额）
     ↓
Workflow（每项两代理）
  生成代理：按配额写三组 query（见 §4），要求口语化一半、含省略句/错别字/长句/间接表达
  审校代理：删除【提及不允许字段 / 动作意图缺失或多余 / 非自然中文 / 组内近重复】，
            错别字与口语是刻意的不修正；结果写分片 shards/cc/<caseId>.json
     ↓
expand_queries.py     分片 → 渲染 messages 样本（复刻生产提示词逐字节）
                      + 归一化去重 + 与 20 条 taskspec 冻结措辞碰撞检查
```

## 4. 三种 query 变体（教什么）

| 变体 | 配额/组合 | 标签 fields | 教会模型 |
|---|---|---|---|
| full | 12 | 全部候选字段 | 点名即全选 |
| subset | 6（字段≥3 的组合） | 前 2 个字段 | **字段选择**：候选注入不变，只选用户点名的 |
| domainOnly | 2 | `[]` | "只提领域不提字段 → 空数组"规则 |

second_layer 样本仅由 full 变体派生（subset/domainOnly 下"最优方案"有歧义，硬标会教错），
且仅当枚举方案 ≥2 且金标在枚举内。

## 5. 泄漏防线

1. 留出组合在 emit_workitems 阶段整体排除（query 生成都不发生）；
2. 渲染时与 20 条 taskspec 冻结措辞做归一化精确碰撞检查；
3. 全集归一化去重；审计文件记录每行的 caseId/variant/query 便于追溯。

## 6. 规模与统计（2026-09-12 生成）

| 指标 | 数值 |
|---|---|
| 总样本 | **3086** |
| first_layer / second_layer | 2265 / 821 |
| first_layer 按变体 | full 1691 · subset 383 · domainOnly 191 |
| 训练组合 | 96（gallery 118 − 留出 22） |
| 工作流保留 query | 1698（审校代理已过滤越界/近重复） |
| 归一化去重丢弃 / 冻结措辞碰撞 | 4 / 0 |
| 生成方式 | 444 代理（96 生成 + 96 审校 + GD 侧共用一次工作流），零失败 |

明细见 `cc_train_v1.stats.json`；逐行溯源见 audit 文件。

## 7. 微调建议

- 格式即 messages 三元组；**answer-only loss**（只对 assistant 计损失）；
- 起点超参：LoRA r=8~16（attn+MLP），lr 1e-4~2e-4，2 epoch，温度 0 评测；
- 两任务混训不需配平（second_layer 天然少且更简单）；
- 换盘古：按其 chat template 重新包装 messages 即可，内容不变；
- 训后验收：`python -m training.eval_harness --model <微调模型>`，
  与裸模型基线同表对比；**taskspec 桶（人写措辞）分数才是有效信号**，
  gallery 留出桶只是同分布 sanity。

## 8. 已知边界（诚实声明）

- 模板命中率 ~30% 是模板库覆盖度属性，本数据集不改变它；训练目标是
  "模板路由适用时两次决策的正确率与一次通过率"；
- query 由 LLM 反写并经审校代理过滤，仍建议抽 5% 人工复核（审计文件按
  caseId 抽即可）；
- 未包含拒绝/边界类样本（CC 第一层契约无 reject 输出，场景门禁在主 Agent）。
