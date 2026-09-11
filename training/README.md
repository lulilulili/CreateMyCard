# CC 端侧模型适配 · 训练与评测工具链

目标：保持小艺输入不变，用端侧 3B 模型（qwen2.5:3b 基线，目标盘古 3B）替换
模板路由的两次云端调用。契约为 3B 友好改造版（见 `playground/`）：

- 第一层：`{"theme","capability","fields","action"}`（小 JSON 标定，非最终选择）
- 第二层：`{"plan": n}`（服务端枚举合法方案后的选择题）

中间的确定性 Search 在**造数据和推理时都真实运行**——两次调用各自独立成样本，
模型永远不看中间代码（详见 build_dataset.py 头注释）。

## 数据来源（只用 CC 现成资产，标签零人工）

- `provider_gallery` 118 例：businessId/scenarioId/candidateOutputFields/
  eventCandidates/targetTemplateId/expectsFusionBall 即全套标签；
- `testdata/taskspec` 20 例：唯一人写措辞，全部进冻结测试集（needsReview）。

## 使用

```bash
pip install -r widget_service/requirements.txt   # gallery 生成依赖
python -m training.export_gallery                # 0.1  118 例 → cases_gallery.jsonl
python -m training.build_dataset                 # 1.1/1.2  train/frozen 切分（防泄漏：整组合留出）
python -m training.eval_harness --gold           # 评分器自测，应全 1.0
python -m training.eval_harness --model qwen2.5:3b --base-url http://127.0.0.1:11434/v1   # 1.3 裸模型基线
```

产物在 `training/generated/`（可再生，不入库）。当前一次跑通的数字：
训练 96 case / 166 样本（二层 26 例因候选<2 或金标不在枚举内跳过）；
冻结 = 22 gallery（6 个整组合留出）+ 20 taskspec。

## 换模型（盘古 3B）

训练集是 messages 三元组，模型无关；换模型只需按其 chat template 重新包装 +
复测 `--gold` 与基线。提示词冻结在 `training/prompts.py`（PROMPT_MANIFEST 版本化，
改动必须升版本，否则数据作废）。

## 诚实边界

- 95 案例为模板化措辞，只教任务形状与格式，不证明口语泛化；
- 评测分两桶：taskspec（人写）分数才是有效信号，gallery 留出分数只是同分布 sanity；
- 模板命中率 ~30% 是模板库覆盖度属性，训练不改变它——训练目标是
  "模板路由适用时，两次决策的正确率与一次通过率"。
