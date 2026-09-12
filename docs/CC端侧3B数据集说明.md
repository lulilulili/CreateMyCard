# CC 端侧 3B 数据集说明

## 目标

这套数据用于替换 CC 模板流程中的两次云侧模型调用。两次调用被视为两个独立的监督任务：

1. **第一层**：从当前能力、字段、主题和动作候选中选择用户真正要求的内容。
2. **第二层**：在代码根据能力和动作数确定性枚举出的候选方案中，只输出方案序号。

中间的 Search 不进入模型输入，也不作为模型输出标签。训练和线上推理都由代码执行 Search；训练数据中的第二层输入使用第一层金标推导出的候选列表，这是 teacher forcing，保证训练分布稳定。

## 产物

运行 `training/build_3b_dataset.bat` 后，默认生成 `training/datasets/cc-3b-v1/`：

- `cc_train.jsonl`、`cc_dev.jsonl`、`cc_test.jsonl`：可直接转换到目标模型 chat template 的 messages 数据。
- `cc_all.audit.jsonl`：样本来源、案例、变体、金标方案号和契约信息，不应喂给模型。
- `cc_dataset_card.json`：数量、切分和限制。

当前生成规模为 4,200 条：第一层 3,360 条，第二层 840 条。train/dev/test 按 `(businessId, scenarioId)` 的稳定哈希整组切分；被冻结的 TaskSpec 不进入训练。目标样本按 gallery 合法标签确定性生成，动作和方案标签来自 CC 注册资产。

## 为什么第二层少于 3,000 条

第二层只有在以下条件同时满足时才产生：请求是 full 变体、真实 Search 至少产生两个候选方案、目标模板确实在候选中。无法满足时硬造方案号会产生错误监督，因此第二层保持 840 条。总 CC SFT 样本数超过 3,000，且两个任务都有独立样本。

## 训练边界

只对 assistant 消息计算 loss。不要把 `source`、`caseId`、`audit` 或 Search 结果加入消息。第一层和第二层建议使用不同 task tag 或分别训练两个 LoRA adapter；如果使用一个 adapter，保留 `task` 字段用于采样统计，但不要把它额外塞入 prompt。

这是一版可运行的 bootstrap 数据工厂：表达由确定性句式扩增，适合先跑通训练和端到端链路；正式指标前建议对每个案例抽审，并逐步用真实用户表达替换句式扩增样本。
