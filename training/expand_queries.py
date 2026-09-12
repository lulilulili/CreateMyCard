"""把工作流回收的 query 变体渲染成 CC 训练样本。

输入：training/generated/cc_queries.json
      [{caseId, full:[...], subset:[...], domainOnly:[...]}, ...]
输出：training/datasets/cc_train_v1.jsonl（messages 三元组）
      training/datasets/cc_train_v1.audit.jsonl（审计，绝不进模型输入）

每条 query 产出：
- 1 条 first_layer 样本（标签按变体：full=全字段 / subset=子集 / domainOnly=[]）
- 1 条 second_layer 样本（仅当该 case 枚举方案≥2 且金标在内；query 复用）

泄漏防线：冻结组合在 emit_workitems 已排除；此处再对 20 条 taskspec 人写措辞
做精确+归一化去重。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .assets import REPO_ROOT, load_registry
from .prompts import (PROMPT_MANIFEST, SYSTEM_FIRST_LAYER, SYSTEM_SECOND_LAYER,
                      build_first_layer_user, build_second_layer_user)
from .validators import search_templates

GENERATED = REPO_ROOT / "training" / "generated"
DATASETS = REPO_ROOT / "training" / "datasets"


def _normalize(text: str) -> str:
    return re.sub(r"[\s，。！？,.!?、“”\"']", "", text)


# 确定性风格轴（只动表达不动语义；仅用于 full 变体的 first_layer 扩增）
STYLES = (lambda q: ("麻烦" + q + "，谢谢") if q.startswith(("帮我", "给我", "替我"))
          else "麻烦帮我" + q + "，谢谢",
          lambda q: "哎，" + q + "呗",
          lambda q: "我赶时间，" + q,
          lambda q: q + "，现在就要")


def main() -> int:
    registry = load_registry()
    cases = {json.loads(line)["caseId"]: json.loads(line) for line in
             (GENERATED / "cases_gallery.jsonl").read_text(encoding="utf-8").splitlines() if line}
    payload = json.loads((GENERATED / "cc_queries.json").read_text(encoding="utf-8"))
    frozen_prompts = set()
    taskspec = json.loads((REPO_ROOT / "testdata" / "taskspec" / "taskspec_cases.json")
                          .read_text(encoding="utf-8"))
    for spec in taskspec:
        frozen_prompts.add(_normalize(spec["userQuery"]))

    DATASETS.mkdir(parents=True, exist_ok=True)
    out = DATASETS / "cc_train_v1.jsonl"
    audit_out = DATASETS / "cc_train_v1.audit.jsonl"
    seen, written, stats = set(), 0, {"first_layer": 0, "second_layer": 0,
                                      "dedup_dropped": 0, "frozen_collision": 0}
    with out.open("w", encoding="utf-8") as sft, audit_out.open("w", encoding="utf-8") as audit:
        for row in payload:
            case = cases[row["caseId"]]
            theme = (case.get("themeExpected") or [None])[0]
            plans = search_templates(registry, case["businessId"], case["actionCount"])
            gold_plan = next((i for i, p in enumerate(plans)
                              if p["templateId"] == case["targetTemplateId"]), None)
            for variant, fields_label in (("full", case["candidateFields"]),
                                          ("subset", (case["candidateFields"][:2]
                                                      if len(case["candidateFields"]) >= 3 else None)),
                                          ("domainOnly", [])):
                queries = row.get(variant) or []
                if fields_label is None:
                    continue
                for query_index, query in enumerate(queries):
                    query = str(query).strip()
                    texts = [query]
                    if variant == "full" and query_index % 2 == 0:
                        texts.append(STYLES[query_index % len(STYLES)](query))
                    for style_index, text in enumerate(texts):
                        key = _normalize(text)
                        if not key or len(text) < 4:
                            continue
                        if key in frozen_prompts:
                            stats["frozen_collision"] += 1
                            continue
                        if key in seen:
                            stats["dedup_dropped"] += 1
                            continue
                        seen.add(key)
                        target = {"theme": theme, "capability": case["capabilityId"],
                                  "fields": fields_label, "action": case["eventIds"]}
                        case_for_prompt = dict(case, userQuery=text)
                        sft.write(json.dumps({"messages": [
                            {"role": "system", "content": SYSTEM_FIRST_LAYER},
                            {"role": "user", "content": build_first_layer_user(case_for_prompt, registry)},
                            {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)}]},
                            ensure_ascii=False) + "\n")
                        written += 1
                        stats["first_layer"] += 1
                        audit.write(json.dumps({"line": written, "task": "first_layer",
                                                "caseId": case["caseId"], "variant": variant,
                                                "styled": style_index > 0, "query": text,
                                                "contract": PROMPT_MANIFEST}, ensure_ascii=False) + "\n")
                        if (gold_plan is not None and len(plans) >= 2 and variant == "full"
                                and style_index == 0):
                            sft.write(json.dumps({"messages": [
                                {"role": "system", "content": SYSTEM_SECOND_LAYER},
                                {"role": "user",
                                 "content": build_second_layer_user(case_for_prompt, plans)},
                                {"role": "assistant",
                                 "content": json.dumps({"plan": gold_plan + 1})}]},
                                ensure_ascii=False) + "\n")
                            written += 1
                            stats["second_layer"] += 1
                            audit.write(json.dumps({"line": written, "task": "second_layer",
                                                    "caseId": case["caseId"], "query": text,
                                                    "contract": PROMPT_MANIFEST},
                                                   ensure_ascii=False) + "\n")
    stats["total"] = written
    print(json.dumps(stats, ensure_ascii=False))
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
