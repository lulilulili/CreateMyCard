"""Deterministic validators / search / scorers for the 3B-adapted flow.

These are the training-side ports of the playground's JS validators, driven by
the real cloud assets. They serve three roles: data filter, ground-truth plan
enumerator, and automatic scorer for the eval harness.
"""
from __future__ import annotations

# 布局后缀策略（与云侧 2x2 组合策略一致）：动作数 -> 允许的业务模板后缀
SUFFIXES_BY_ACTION_COUNT = {0: ("Full",), 1: ("Hero", "Full"), 2: ("Compact",)}


def search_templates(registry: dict, business_id: str, action_count: int,
                     fields: list[str] | None = None) -> list[dict]:
    """Deterministic candidate list, ordered by templateId for stability."""
    suffixes = SUFFIXES_BY_ACTION_COUNT.get(action_count, ("Full",))
    candidates = [t for t in registry["businessTemplates"].get(business_id, [])
                  if t["suffix"] in suffixes]
    if fields:
        covered = [t for t in candidates if set(fields).issubset(set(t["shows"]))]
        if covered:
            candidates = covered
    return sorted(candidates, key=lambda t: t["templateId"])


def validate_first_layer(output: dict, case: dict, registry: dict) -> list[str]:
    errors = []
    if not isinstance(output, dict):
        return ["输出不是 JSON 对象"]
    if output.get("capability") != case["capabilityId"]:
        errors.append("capability 应为 %s，实际 %r" % (case["capabilityId"], output.get("capability")))
    fields = output.get("fields")
    if not isinstance(fields, list):
        errors.append("fields 必须是数组")
    else:
        allowed = set(case["candidateFields"])
        for field in fields:
            if field not in allowed:
                errors.append("字段 %r 不在候选内" % field)
    actions = output.get("action")
    if not isinstance(actions, list) or len(actions) > 2:
        errors.append("action 必须是 ≤2 的数组")
    else:
        for action in actions:
            if action not in case["eventIds"]:
                errors.append("action %r 不在候选内" % action)
    theme_ids = set(registry["themes"])
    if output.get("theme") not in theme_ids:
        errors.append("theme %r 不在主题表" % output.get("theme"))
    return errors


def score_first_layer(output: dict, case: dict, registry: dict) -> dict:
    """Semantic scoring against gallery ground truth."""
    result = {"structValid": not validate_first_layer(output, case, registry)}
    result["capabilityAcc"] = int(output.get("capability") == case["capabilityId"])
    gold_fields = set(case["candidateFields"])
    pred_fields = set(output.get("fields") or [])
    tp = len(gold_fields & pred_fields)
    precision = tp / len(pred_fields) if pred_fields else (1.0 if not gold_fields else 0.0)
    recall = tp / len(gold_fields) if gold_fields else 1.0
    result["fieldP"], result["fieldR"] = precision, recall
    result["fieldF1"] = (2 * precision * recall / (precision + recall)
                         if precision + recall else 0.0)
    gold_actions = set(case["eventIds"])
    pred_actions = set(output.get("action") or [])
    result["actionAcc"] = int(pred_actions == gold_actions)
    expected_themes = set(case.get("themeExpected") or [])
    result["themeAcc"] = int(output.get("theme") in expected_themes) if expected_themes else None
    return result


def score_second_layer(output: dict, plans: list[dict], target_template_id: str) -> dict:
    try:
        index = int(output.get("plan"))
    except (TypeError, ValueError):
        return {"structValid": False, "planAcc": 0}
    if not 1 <= index <= len(plans):
        return {"structValid": False, "planAcc": 0}
    return {"structValid": True,
            "planAcc": int(plans[index - 1]["templateId"] == target_template_id)}
