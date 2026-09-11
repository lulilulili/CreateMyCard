"""Steps 1.1/1.2: labels for the 20 TaskSpec cases, chat samples, leak-safe split.

Usage:  python -m training.build_dataset
Inputs: training/generated/cases_gallery.jsonl (from export_gallery)
        testdata/taskspec/taskspec_cases.json
Output: training/generated/{train.jsonl, frozen_set.jsonl, dataset_card.json}

Split rule (leak-safe): entire (businessId, scenarioId) groups are held out —
a combo never appears in both sides. All 20 human-written TaskSpec cases go to
the frozen set (they carry the only natural phrasing we have).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .assets import REPO_ROOT, load_registry
from .prompts import (PROMPT_MANIFEST, SYSTEM_FIRST_LAYER, SYSTEM_SECOND_LAYER,
                      build_first_layer_user, build_second_layer_user)
from .validators import search_templates

GENERATED = REPO_ROOT / "training" / "generated"
HOLDOUT_TARGET = 20  # gallery cases held out for eval


def _load_gallery() -> list[dict]:
    path = GENERATED / "cases_gallery.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _taskspec_cases(registry: dict) -> list[dict]:
    """Derive first-layer labels for the 20 human-written TaskSpecs.

    capability: matched by dataModelSchema data-root vs defaultWriteResultTo;
    fields: all schema leaves relative to that root; actions: eventCandidates.
    Cases without data capability become boundary/entry records (first layer
    is not scored on them). Every record is flagged needsReview=True.
    """
    raw = json.loads((REPO_ROOT / "testdata" / "taskspec" / "taskspec_cases.json")
                     .read_text(encoding="utf-8"))
    root_to_capability = {}
    for capability_id, capability in registry["dataCapabilities"].items():
        root = capability.get("defaultWriteResultTo")
        if root:
            root_to_capability[root.rstrip("/")] = capability_id

    def leaves(node, prefix=""):
        if isinstance(node, dict):
            if "sampleValue" in node or ("type" in node and "description" in node):
                yield prefix
                return
            for key, value in node.items():
                yield from leaves(value, prefix + "/" + key)
        elif isinstance(node, list) and node:
            yield from leaves(node[0], prefix)

    records = []
    for index, spec in enumerate(raw):
        data = spec.get("dataModelSchema", {}).get("data", {})
        roots = list(data.keys())
        capability_id = None
        fields = []
        if len(roots) == 1:
            capability_id = root_to_capability.get("/data/" + roots[0])
            if capability_id:
                fields = sorted(leaves(data[roots[0]]))
        event_ids = [e.get("id") for e in spec.get("eventCandidates", []) if e.get("id")]
        records.append({
            "source": "taskspec", "caseId": "taskspec-%02d" % (index + 1),
            "userQuery": spec["userQuery"], "size": spec.get("size", "2x2"),
            "capabilityId": capability_id, "candidateFields": fields,
            "eventIds": event_ids, "actionCount": len(event_ids),
            "themeExpected": [], "targetTemplateId": None,
            "businessId": None, "scenarioId": "taskspec",
            "needsReview": True,
            "boundary": capability_id is None,
        })
    return records


def _holdout_combos(gallery: list[dict]) -> set[tuple]:
    combos = sorted({(r["businessId"], r["scenarioId"]) for r in gallery})
    sized = {combo: sum(1 for r in gallery if (r["businessId"], r["scenarioId"]) == combo)
             for combo in combos}
    ranked = sorted(combos, key=lambda c: hashlib.md5(("|".join(c)).encode()).hexdigest())
    held, count = [], 0
    for combo in ranked:
        if count >= HOLDOUT_TARGET:
            break
        held.append(combo)
        count += sized[combo]
    return set(held)


def _first_layer_target(case: dict) -> dict:
    theme = (case.get("themeExpected") or [None])[0]
    return {"theme": theme, "capability": case["capabilityId"],
            "fields": case["candidateFields"], "action": case["eventIds"]}


def main() -> int:
    registry = load_registry()
    gallery = _load_gallery()
    taskspec = _taskspec_cases(registry)
    held = _holdout_combos(gallery)
    train_cases = [r for r in gallery if (r["businessId"], r["scenarioId"]) not in held]
    frozen_gallery = [r for r in gallery if (r["businessId"], r["scenarioId"]) in held]

    train_samples = []
    skipped_plans = 0
    for case in train_cases:
        train_samples.append({
            "task": "first_layer", "caseId": case["caseId"],
            "messages": [
                {"role": "system", "content": SYSTEM_FIRST_LAYER},
                {"role": "user", "content": build_first_layer_user(case, registry)},
                {"role": "assistant",
                 "content": json.dumps(_first_layer_target(case), ensure_ascii=False)}]})
        plans = search_templates(registry, case["businessId"], case["actionCount"])
        gold = [i for i, p in enumerate(plans) if p["templateId"] == case["targetTemplateId"]]
        if not gold or len(plans) < 2:
            skipped_plans += 1
            continue
        train_samples.append({
            "task": "second_layer", "caseId": case["caseId"],
            "messages": [
                {"role": "system", "content": SYSTEM_SECOND_LAYER},
                {"role": "user", "content": build_second_layer_user(case, plans)},
                {"role": "assistant",
                 "content": json.dumps({"plan": gold[0] + 1}, ensure_ascii=False)}]})

    frozen = frozen_gallery + taskspec
    GENERATED.mkdir(parents=True, exist_ok=True)
    with (GENERATED / "train.jsonl").open("w", encoding="utf-8") as handle:
        for sample in train_samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
    with (GENERATED / "frozen_set.jsonl").open("w", encoding="utf-8") as handle:
        for record in frozen:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    # 泄漏检查：冻结 gallery 组合不得出现在训练侧
    train_combos = {(r["businessId"], r["scenarioId"]) for r in train_cases}
    assert not (train_combos & held), "泄漏：留出组合出现在训练集"

    card = {"promptManifest": PROMPT_MANIFEST,
            "train": {"cases": len(train_cases), "samples": len(train_samples),
                      "secondLayerSkipped": skipped_plans},
            "frozen": {"galleryCases": len(frozen_gallery), "taskspecCases": len(taskspec),
                       "heldCombos": sorted(["%s|%s" % combo for combo in held])},
            "splitRule": "hold out whole (businessId, scenarioId) combos by md5 order; "
                         "all 20 taskspec cases frozen"}
    (GENERATED / "dataset_card.json").write_text(
        json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(card["train"], ensure_ascii=False))
    print(json.dumps(card["frozen"] | {"heldCombos": len(held)}, ensure_ascii=False,
                     default=str))
    boundary = sum(1 for r in taskspec if r["boundary"])
    print("taskspec: %d 条（其中 %d 条为无数据边界/入口卡，不计第一层分）" % (len(taskspec), boundary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
