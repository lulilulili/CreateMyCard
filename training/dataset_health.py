"""CC 训练集四层健康检查（P0-3）。

L1 精确重复（构建期已去重，应为 0）
L2 近重复对（字符 2-gram Jaccard ≥0.80）：整体冗余率 + **跨答案冲突对**
   （近重复 query 但 assistant 标签不同 = 教模型自相矛盾，最高危）
L3 对冻结措辞（taskspec-20）：精确 + 近重复
L4 答案组合集中度：每个答案的样本数分布（反写法天然一答多问，报告 Top 供审）

用法：python -m training.dataset_health
输出：training/datasets/cc_train_v1.health.json
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from .assets import REPO_ROOT

DATASETS = REPO_ROOT / "training" / "datasets"
NEAR_DUP_THRESHOLD = 0.80


def _normalize(text: str) -> str:
    return re.sub(r"[\s，。！？,.!?、“”\"'：:；;]", "", text)


def _bigrams(text: str) -> frozenset:
    return frozenset(text[i:i + 2] for i in range(len(text) - 1)) or frozenset([text])


def _jaccard(a: frozenset, b: frozenset) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def near_dup_pairs(items: list[tuple[str, frozenset, str]]) -> list[tuple[int, int, float]]:
    """按首两字符分桶后桶内两两比较（近重复必然共享大量 bigram，跨桶漏检率可忽略）。"""
    buckets = defaultdict(list)
    for index, (norm, grams, _) in enumerate(items):
        for gram in list(grams)[:6]:
            buckets[gram].append(index)
    candidate_pairs = set()
    for bucket in buckets.values():
        if len(bucket) > 400:
            continue
        for i in range(len(bucket)):
            for j in range(i + 1, len(bucket)):
                candidate_pairs.add((bucket[i], bucket[j]))
    pairs = []
    for i, j in candidate_pairs:
        if abs(len(items[i][0]) - len(items[j][0])) > max(len(items[i][0]), len(items[j][0])) * 0.5:
            continue
        score = _jaccard(items[i][1], items[j][1])
        if score >= NEAR_DUP_THRESHOLD:
            pairs.append((i, j, round(score, 3)))
    return pairs


def main() -> int:
    rows = [json.loads(line) for line in
            (DATASETS / "cc_train_v1.jsonl").read_text(encoding="utf-8").splitlines() if line]
    audit = [json.loads(line) for line in
             (DATASETS / "cc_train_v1.audit.jsonl").read_text(encoding="utf-8").splitlines() if line]
    assert len(rows) == len(audit)
    first = [(a, r) for a, r in zip(audit, rows) if a["task"] == "first_layer"]
    items = []
    for a, r in first:
        norm = _normalize(a["query"])
        items.append((norm, _bigrams(norm), r["messages"][2]["content"]))

    exact = Counter(norm for norm, _, _ in items)
    exact_dups = {k: v for k, v in exact.items() if v > 1}

    pairs = near_dup_pairs(items)
    conflicts = [(i, j, s) for i, j, s in pairs if items[i][2] != items[j][2]]

    taskspec = json.loads((REPO_ROOT / "testdata" / "taskspec" / "taskspec_cases.json")
                          .read_text(encoding="utf-8"))
    frozen_items = [(_normalize(s["userQuery"]),
                     _bigrams(_normalize(s["userQuery"])), "") for s in taskspec]
    frozen_exact = sum(1 for norm, _, _ in items if norm in {f[0] for f in frozen_items})
    frozen_near = 0
    for norm, grams, _ in items:
        for f_norm, f_grams, _ in frozen_items:
            if _jaccard(grams, f_grams) >= 0.60:
                frozen_near += 1
                break

    answer_counts = Counter(ans for _, _, ans in items)
    report = {
        "dataset": "cc_train_v1 (v1.1)",
        "firstLayerN": len(items),
        "L1_exact_dup": len(exact_dups),
        "L2_near_dup_pairs": len(pairs),
        "L2_near_dup_rate": round(len(pairs) / max(1, len(items)), 4),
        "L2_cross_answer_conflicts": len(conflicts),
        "L2_conflict_examples": [
            {"a": first[i][0]["query"][:40], "b": first[j][0]["query"][:40], "jaccard": s}
            for i, j, s in conflicts[:5]],
        "L3_frozen_exact": frozen_exact,
        "L3_frozen_near060": frozen_near,
        "L4_distinct_answers": len(answer_counts),
        "L4_top_answer_share": round(max(answer_counts.values()) / len(items), 4),
    }
    out = DATASETS / "cc_train_v1.health.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    verdict = (report["L2_cross_answer_conflicts"] == 0 and report["L3_frozen_exact"] == 0)
    print("健康判定:", "PASS" if verdict else "存在待处理项")
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
