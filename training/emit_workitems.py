"""为 query 反写工作流发射 CC 侧工作项。

每个训练侧 gallery case 产出一个工作项，含三种变体的生成配额：
- full       : 点名全部候选字段（标签 = 全部 candidateFields）
- subset     : 只点名前 2 个字段（标签 = 该子集；候选注入不变，训练字段选择）
- domainOnly : 只提业务不提字段（标签 fields=[]，训练"未点名给空数组"规则）

用法：python -m training.emit_workitems > training/generated/cc_workitems.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from .assets import REPO_ROOT, load_registry, resolve_field_description

GENERATED = REPO_ROOT / "training" / "generated"
FULL_N, SUBSET_N, DOMAIN_N = 12, 6, 2


def main() -> int:
    registry = load_registry()
    card = json.loads((GENERATED / "dataset_card.json").read_text(encoding="utf-8"))
    held = set(card["frozen"]["heldCombos"])
    cases = [json.loads(line) for line in
             (GENERATED / "cases_gallery.jsonl").read_text(encoding="utf-8").splitlines() if line]
    items = []
    for case in cases:
        if "%s|%s" % (case["businessId"], case["scenarioId"]) in held:
            continue  # 冻结组合绝不参与训练 query 生成
        capability = registry["dataCapabilities"].get(case["capabilityId"], {})
        schema = capability.get("outputSchema", {})
        field_descs = {path: resolve_field_description(schema, path)
                       for path in case["candidateFields"]}
        events = registry["eventCapabilities"]
        action_descs = [str(events.get(event_id, {}).get("description", event_id)).split("。")[0]
                        for event_id in case["eventIds"]]
        subset = case["candidateFields"][:2] if len(case["candidateFields"]) >= 3 else None
        items.append({
            "caseId": case["caseId"],
            "businessName": case.get("providerId", "").split(".")[-2] if case.get("providerId") else "",
            "businessDesc": str(capability.get("description", "")).split("。")[0][:40],
            "fieldDescs": field_descs,
            "actionDescs": action_descs,
            "actionCount": case["actionCount"],
            "subset": subset,
            "quota": {"full": FULL_N, "subset": SUBSET_N if subset else 0, "domainOnly": DOMAIN_N},
        })
    out = GENERATED / "cc_workitems.json"
    out.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    print("items: %d -> %s" % (len(items), out), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
