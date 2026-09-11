"""Step 0.1: regenerate the 118-case provider gallery and flatten to JSONL.

Usage:  python -m training.export_gallery
Output: training/generated/cases_gallery.jsonl  (one labelled case per line)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from .assets import CLOUD, REPO_ROOT, load_registry, themes_for_capability

GENERATED = REPO_ROOT / "training" / "generated"


def main() -> int:
    sys.path.insert(0, str(CLOUD))
    from services.template_generation.test_support.provider_gallery import (  # noqa: E402
        load_gallery_input_manifest, write_gallery_input_dataset)

    gallery_root = GENERATED / "gallery_input"
    gallery_root.mkdir(parents=True, exist_ok=True)
    write_gallery_input_dataset(gallery_root)
    manifest = load_gallery_input_manifest(gallery_root)
    registry = load_registry()

    records = []
    for provider in manifest.providers:
        for case in provider.cases:
            request_path = gallery_root / case.requestFile
            request = json.loads(request_path.read_text(encoding="utf-8"))
            content = request.get("content", request)
            bindings = content.get("candidateDataBindings", [])
            events = content.get("candidateEventCandidates", [])
            binding = bindings[0] if bindings else {}
            capability_id = binding.get("capabilityId")
            themes = themes_for_capability(registry, capability_id) if capability_id else []
            expected_themes = [t["themeId"] for t in themes
                               if t["isFusion"] == bool(case.expectsFusionBall)] or \
                              [t["themeId"] for t in themes]
            records.append({
                "source": "gallery", "caseId": case.caseId,
                "userQuery": content.get("userQuery", ""),
                "size": "2x2",
                "providerId": case.providerId, "businessId": case.businessId,
                "scenarioId": case.scenarioId,
                "capabilityId": capability_id,
                "writeResultTo": binding.get("writeResultTo"),
                "candidateFields": list(binding.get("candidateOutputFields", [])),
                "eventIds": [e.get("capabilityId") or e.get("id") for e in events],
                "actionCount": len(events),
                "expectsFusionBall": bool(case.expectsFusionBall),
                "themeExpected": expected_themes,
                "targetTemplateId": case.targetTemplateId,
                "expectedTemplateSuffix": case.expectedTemplateSuffix,
            })

    out = GENERATED / "cases_gallery.jsonl"
    with out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    by_provider = {}
    for record in records:
        by_provider[record["providerId"]] = by_provider.get(record["providerId"], 0) + 1
    print("providers:", len(by_provider))
    for provider_id, count in sorted(by_provider.items()):
        print(" ", provider_id, count)
    print("total:", len(records), "->", out)
    missing_capability = [r["caseId"] for r in records if not r["capabilityId"]]
    if missing_capability:
        print("warn: cases without data binding:", len(missing_capability))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
