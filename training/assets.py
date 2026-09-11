"""Load CreateMyCard cloud assets into one registry dict for training tooling.

Single source of truth = the same files the provider gallery reads:
- data/capabilities/app-11.7.5.205_rom-6.0/{data,event}_capabilities.json
- services/template_generation/resources/source/providers/*/provider.json
- services/template_generation/resources/source/themes/*/theme.json
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CLOUD = REPO_ROOT / "widget_service" / "cloud"
CAPABILITY_ROOT = CLOUD / "data" / "capabilities" / "app-11.7.5.205_rom-6.0"
SOURCE = CLOUD / "services" / "template_generation" / "resources" / "source"

_SUFFIXES = ("WideFull", "WideHero", "Compact", "Support", "Full", "Hero")


def template_suffix(template_id: str) -> str:
    name = template_id.split("@", 1)[0]
    for suffix in _SUFFIXES:
        if name.endswith(suffix):
            return suffix
    return ""


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_registry() -> dict:
    data_caps = {item["id"]: item for item in _read(CAPABILITY_ROOT / "data_capabilities.json")}
    event_caps = {item["id"]: item for item in _read(CAPABILITY_ROOT / "event_capabilities.json")}
    providers, templates, business_templates = {}, {}, {}
    for manifest in sorted((SOURCE / "providers").glob("*/provider.json")):
        bundle = _read(manifest)
        providers[bundle["providerId"]] = bundle
        for template in bundle.get("templates", []):
            if "businessId" not in template or "capabilityId" not in template:
                continue  # layout/action 模板不进业务检索索引
            template_id = template["templateId"]
            shows = list(dict.fromkeys(
                [*template.get("primaryData", []), *template.get("secondaryData", [])]))
            record = {"templateId": template_id, "businessId": template["businessId"],
                      "capabilityId": template["capabilityId"],
                      "description": template.get("description", ""),
                      "primaryData": template.get("primaryData", []),
                      "shows": shows, "suffix": template_suffix(template_id)}
            templates[template_id] = record
            business_templates.setdefault(template["businessId"], []).append(record)
    themes = {}
    for theme_path in sorted((SOURCE / "themes").glob("*/theme.json")):
        theme = _read(theme_path)
        themes[theme["themeProfileId"]] = {
            "themeId": theme["themeProfileId"],
            "description": theme.get("description", ""),
            "supportedCapabilityIds": theme.get("supportedCapabilityIds", []),
            "isFusion": "fusionBallStyle" in theme}
    return {"dataCapabilities": data_caps, "eventCapabilities": event_caps,
            "providers": providers, "templates": templates,
            "businessTemplates": business_templates, "themes": themes}


def resolve_field_description(output_schema: dict, pointer: str) -> str:
    """Walk outputSchema by JSON pointer; arrays follow items."""
    current = output_schema
    for raw in pointer.strip("/").split("/"):
        part = raw.replace("~1", "/").replace("~0", "~")
        while isinstance(current, dict) and current.get("type") == "array":
            current = current.get("items", {})
        if not isinstance(current, dict):
            return part
        props = current.get("properties", {})
        current = props.get(part, {}) if part in props else current.get("items", {}).get(
            "properties", {}).get(part, {})
    if isinstance(current, dict) and current.get("description"):
        return str(current["description"]).split("。")[0][:40]
    return pointer.split("/")[-1]


def themes_for_capability(registry: dict, capability_id: str) -> list[dict]:
    return [theme for theme in registry["themes"].values()
            if capability_id in theme["supportedCapabilityIds"]]
