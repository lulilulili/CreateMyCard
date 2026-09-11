"""Frozen prompts for the 3B-adapted CreateMyCard flow (playground contracts).

Any change to these strings MUST bump PROMPT_MANIFEST versions — training data
and eval results are bound to these bytes.
"""
from __future__ import annotations

from .assets import resolve_field_description, themes_for_capability

PROMPT_MANIFEST = {
    "firstLayerPrompt": "cmc-3b-first-layer/1.0",
    "secondLayerPrompt": "cmc-3b-second-layer/1.0",
    "agentPrompt": "cmc-3b-agent/1.0",
    "datasetSchema": "cmc-training-case/1.0",
}

SYSTEM_FIRST_LAYER = """You plan a small info card. Answer with ONE JSON object only, no explanation:
{"theme":"<theme-id>","capability":"<capability-id>","fields":["</path>", ...],"action":["<event-id>", ...]}
Rules:
- capability: the ONE capability the user asks about.
- fields: only paths of that capability the user explicitly wants to see; [] if the user names the domain without specific fields.
- theme: one theme id that matches the capability.
- action: [] unless the user asks to tap / open / jump; then 1-2 matching event ids."""

SYSTEM_SECOND_LAYER = """You choose one card composition. Answer with ONE JSON object only:
{"plan": <number>}
Pick the plan whose displayed fields and action count best match the user request."""

SYSTEM_AGENT = """你是手机助手的卡片主 Agent。根据用户需求判断场景并从能力概述中选择候选能力，只输出一个 JSON 对象：
{"suitable":true,"capabilities":["<capabilityId>"],"title":"<不超过8个字的卡片标题>","description":"<一句话卡片描述>"}
规则：
- capabilities 只能从 CAPABILITIES 列表选 1-2 个，且只选与用户需求直接相关的；
- 需求明显不适合做成桌面小卡片（长文章、完整页面、复杂表单、闲聊）时输出 {"suitable":false,"reason":"<一句话边界说明>"}；
- 不输出解释、Markdown 或额外字段。"""


def build_first_layer_user(case: dict, registry: dict) -> str:
    capability = registry["dataCapabilities"].get(case["capabilityId"], {})
    schema = capability.get("outputSchema", {})
    field_lines = "\n".join(
        "  %s — %s" % (path, resolve_field_description(schema, path))
        for path in case["candidateFields"])
    cap_desc = str(capability.get("description", "")).split("。")[0][:50]
    themes = themes_for_capability(registry, case["capabilityId"]) or list(
        registry["themes"].values())[:2]
    theme_lines = "\n".join("  %s — %s" % (t["themeId"], t["description"]) for t in themes)
    events = registry["eventCapabilities"]
    action_lines = "\n".join(
        "  %s — %s" % (event_id, str(events.get(event_id, {}).get("description", ""))[:50])
        for event_id in case["eventIds"]) or "  (none)"
    return ("CAPABILITIES:\n%s (%s):\n%s\nTHEMES:\n%s\nACTIONS:\n%s\n\nUSER REQUEST: %s"
            % (case["capabilityId"], cap_desc, field_lines, theme_lines, action_lines,
               case["userQuery"]))


def build_second_layer_user(case: dict, plans: list[dict]) -> str:
    lines = []
    for index, plan in enumerate(plans, 1):
        shows = ", ".join(p.split("/")[-1] for p in plan["shows"][:6])
        desc = plan["description"].split("。")[0][:45]
        lines.append("%d. template=%s  (%s)  shows: %s" % (index, plan["templateId"], desc, shows))
    return "PLANS:\n%s\n\nACTIONS WANTED: %d\nUSER REQUEST: %s" % (
        "\n".join(lines), case["actionCount"], case["userQuery"])
