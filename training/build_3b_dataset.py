"""Build a reproducible, leak-aware CC 3B SFT dataset.

The cloud gallery supplies labels.  This builder creates natural-language
requests from those labels, then emits the two production calls separately:
first-layer capability selection and second-layer numbered template choice.
The deterministic template search is run here for labels and is not part of
either model input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from .assets import REPO_ROOT, load_registry, resolve_field_description
from .export_gallery import main as export_gallery
from .prompts import (PROMPT_MANIFEST, SYSTEM_FIRST_LAYER, SYSTEM_SECOND_LAYER,
                      build_first_layer_user, build_second_layer_user)
from .validators import search_templates


def _normalize(text: str) -> str:
    return re.sub(r"[\s，。！？,.!?、“”\"']", "", text)


def _held_combos(cases: list[dict], target: int = 20) -> set[tuple[str, str]]:
    combos = sorted({(case["businessId"], case["scenarioId"]) for case in cases})
    sizes = Counter((case["businessId"], case["scenarioId"]) for case in cases)
    ranked = sorted(combos, key=lambda combo: hashlib.md5(("|".join(combo)).encode()).hexdigest())
    held, count = set(), 0
    for combo in ranked:
        if count >= target:
            break
        held.add(combo)
        count += sizes[combo]
    return held


def _field_name(path: str, description: str) -> str:
    leaf = path.rstrip("/").split("/")[-1]
    replacements = {
        "appName": "应用名称", "durationText": "使用时长", "temperatureText": "当前温度",
        "condition": "天气情况", "batterySOCText": "剩余电量", "chargingStatusDesc": "充电状态",
        "title": "标题", "dtStart": "开始时间", "countdownDays": "倒计时天数",
    }
    return replacements.get(leaf, description.split("，")[0].split("。")[0][:18] or leaf)


def _domain_name(case: dict, registry: dict) -> str:
    capability = registry["dataCapabilities"].get(case["capabilityId"], {})
    return str(capability.get("description", "服务"))[:18] or "服务"


def _request_variants(case: dict, registry: dict, count: int) -> list[tuple[str, str]]:
    schema = registry["dataCapabilities"].get(case["capabilityId"], {}).get("outputSchema", {})
    names = [_field_name(path, resolve_field_description(schema, path))
             for path in case["candidateFields"]]
    all_fields = "、".join(names) if names else "相关信息"
    subset_fields = "、".join(names[:2]) if len(names) >= 2 else all_fields
    action_names = []
    for event_id in case["eventIds"]:
        event = registry["eventCapabilities"].get(event_id, {})
        action_names.append(str(event.get("description", event_id)).split("。")[0][:22])
    actions = "，并提供" + "、".join(action_names) if action_names else ""
    domain = _domain_name(case, registry)
    patterns = [
        ("full", f"做一张{domain}卡片，显示{all_fields}{actions}。"),
        ("full", f"我想在桌面看看{all_fields}{actions}，帮我做成卡片。"),
        ("full", f"请把{all_fields}放到一张卡片里{actions}。"),
        ("full", f"给我一个能看到{all_fields}的卡片{actions}。"),
        ("full", f"桌面卡片展示{all_fields}{actions}，信息清楚一点。"),
        ("full", f"帮我整理一下{all_fields}，做成方便查看的小卡片{actions}。"),
        ("full", f"我只想快速看到{all_fields}{actions}，生成一张卡片。"),
        ("full", f"能不能做个卡片，让我知道{all_fields}{actions}？"),
        ("full", f"麻烦做一张卡片，重点展示{all_fields}{actions}。"),
        ("full", f"我要一个{domain}小组件，内容包括{all_fields}{actions}。"),
        ("full", f"不用复杂布局，把{all_fields}放一起{actions}即可。"),
        ("full", f"请生成桌面卡片：{all_fields}{actions}。"),
        ("subset", f"做一张{domain}卡片，先显示{subset_fields}{actions}。"),
        ("subset", f"我只关心{subset_fields}{actions}，请放进卡片。"),
        ("subset", f"桌面上给我看看{subset_fields}{actions}。"),
        ("subset", f"卡片里保留{subset_fields}就够了{actions}。"),
        ("subset", f"帮我快速查看{subset_fields}{actions}，做成小卡片。"),
        ("subset", f"生成一个只展示{subset_fields}的{domain}卡片{actions}。"),
        ("domainOnly", f"做一张{domain}卡片。"),
        ("domainOnly", f"我想在桌面放个{domain}小组件。"),
        ("domainOnly", f"帮我看看{domain}相关信息。"),
        ("domainOnly", f"给我生成一个{domain}服务卡片。"),
        ("domainOnly", f"桌面上来一个{domain}卡片，简洁一点。"),
        ("domainOnly", f"我需要一个方便查看{domain}的卡片。"),
        ("domainOnly", f"做个{domain}卡片放桌面上。"),
        ("domainOnly", f"请把{domain}做成一个桌面卡片。"),
        ("domainOnly", f"想快速了解{domain}，生成小卡片即可。"),
        ("domainOnly", f"帮我安排一个{domain}卡片。"),
        ("domainOnly", f"我要一个{domain}信息卡。"),
        ("domainOnly", f"给我来张{domain}卡片看看。"),
        ("domainOnly", f"请生成{domain}桌面卡片，不用展开太多。"),
        ("domainOnly", f"把{domain}放到桌面，做成卡片。"),
        ("domainOnly", f"我想要一个{domain}快捷卡片。"),
        ("domainOnly", f"做个{domain}卡片，便于我随时查看。"),
        ("domainOnly", f"桌面卡片显示{domain}就好。"),
    ]
    return patterns[:count]


def _emit_row(task: str, case: dict, user: str, answer: dict, source: str) -> dict:
    return {"task": task, "caseId": case["caseId"], "source": source,
            "messages": [{"role": "system", "content": SYSTEM_FIRST_LAYER if task == "first_layer" else SYSTEM_SECOND_LAYER},
                         {"role": "user", "content": user},
                         {"role": "assistant", "content": json.dumps(answer, ensure_ascii=False, separators=(",", ":"))}]}


def build_dataset(output_dir: Path, variants_per_case: int = 36) -> dict:
    generated = REPO_ROOT / "training" / "generated"
    gallery_path = generated / "cases_gallery.jsonl"
    if not gallery_path.exists():
        export_gallery()
    registry = load_registry()
    cases = [json.loads(line) for line in gallery_path.read_text(encoding="utf-8").splitlines() if line]
    held = _held_combos(cases)
    taskspec = json.loads((REPO_ROOT / "testdata" / "taskspec" / "taskspec_cases.json").read_text(encoding="utf-8"))
    frozen_prompts = {_normalize(str(row["userQuery"])) for row in taskspec}
    rows = []
    audits = []
    for case in cases:
        combo = (case["businessId"], case["scenarioId"])
        if combo in held or not case.get("capabilityId"):
            continue
        plans = search_templates(registry, case["businessId"], case["actionCount"])
        gold_plan = next((index + 1 for index, plan in enumerate(plans)
                          if plan["templateId"] == case["targetTemplateId"]), None)
        for variant, query in _request_variants(case, registry, variants_per_case):
            if _normalize(query) in frozen_prompts:
                continue
            fields = case["candidateFields"] if variant == "full" else (
                case["candidateFields"][:2] if variant == "subset" else [])
            prompt_case = dict(case, userQuery=query, candidateFields=fields)
            theme = (case.get("themeExpected") or [None])[0]
            target = {"theme": theme, "capability": case["capabilityId"],
                      "fields": fields, "action": case["eventIds"]}
            user1 = build_first_layer_user(prompt_case, registry)
            rows.append(_emit_row("first_layer", prompt_case, user1, target, "cc_gallery"))
            audits.append({"line": len(rows), "task": "first_layer", "caseId": case["caseId"],
                          "variant": variant, "query": query, "contract": PROMPT_MANIFEST})
            if variant == "full" and gold_plan is not None and len(plans) >= 2:
                user2 = build_second_layer_user(prompt_case, plans)
                rows.append(_emit_row("second_layer", prompt_case, user2, {"plan": gold_plan}, "cc_gallery"))
                audits.append({"line": len(rows), "task": "second_layer", "caseId": case["caseId"],
                              "variant": variant, "query": query, "plan": gold_plan,
                              "candidateCount": len(plans), "contract": PROMPT_MANIFEST})
    output_dir.mkdir(parents=True, exist_ok=True)
    split_rows = {"train": [], "dev": [], "test": []}
    case_combos = {(case["caseId"]): (case["businessId"], case["scenarioId"])
                   for case in cases}
    for row in rows:
        combo = case_combos[row["caseId"]]
        digest = int(hashlib.md5(("|".join(combo)).encode()).hexdigest()[:2], 16) % 10
        split = "test" if digest == 0 else "dev" if digest == 1 else "train"
        split_rows[split].append(row)
    for split, values in split_rows.items():
        (output_dir / f"cc_{split}.jsonl").write_text(
            "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values), encoding="utf-8")
    (output_dir / "cc_all.audit.jsonl").write_text(
        "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in audits), encoding="utf-8")
    counts = Counter(row["task"] for row in rows)
    card = {"dataset": "cc-3b-sft-v1", "source": "CreateMyCard provider gallery",
            "contract": PROMPT_MANIFEST, "generation": "deterministic labels + natural-language variants",
            "middle_search": "run for gold plan only; never included in model input",
            "split": {name: len(values) for name, values in split_rows.items()},
            "counts": {"total": len(rows), **dict(counts)}, "held_combos": sorted("|".join(combo) for combo in held),
            "frozen_taskspec": len(taskspec), "limitations": ["synthetic paraphrases need human spot review", "gallery labels are template-domain data"]}
    (output_dir / "cc_dataset_card.json").write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    return card


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "training" / "datasets" / "cc-3b-v1")
    parser.add_argument("--variants-per-case", type=int, default=36)
    args = parser.parse_args()
    print(json.dumps(build_dataset(args.output, args.variants_per_case), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
