# ⚠ 定位声明：synthetic_bootstrap（L0 工程回归数据）

本目录数据由固定句式模板确定性生成（详见评审结论，docs/数据三层计划与双机交接.md
@ GenUI_DeviceSideLLM claude 分支）：

- **仅用于**：训练管线冒烟（LoRA/QLoRA 跑通、JSON 约束、后处理、格式回归）；
- **禁止**：作为正式 SFT 数据单独训练后对外宣称模型能力；
- **禁止**：以本目录 dev/test 分数汇报"模型准确率"——同分布模板数据，
  分数包含句式与答案组合记忆（实测跨 split 重复 train/dev 34、train/test 17、dev/test 17）。

正式训练集（L1）：claude 分支 training/datasets/cc_train_v1.jsonl（LLM 反写+审校）。
正式评测（L2）：taskspec-20 + 人工冻结条目；gallery 留出仅作同分布回归桶。
