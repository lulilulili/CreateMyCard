"""Step 1.3: run a model over the frozen set and score with deterministic rules.

Usage:
  python -m training.eval_harness --gold                    # scorer self-test (should be ~100%)
  python -m training.eval_harness --base-url http://127.0.0.1:11434/v1 --model qwen2.5:3b
  python -m training.eval_harness ... --limit 10            # smoke run

Report: per-task metrics, gallery vs taskspec buckets reported separately —
only the taskspec bucket carries human phrasing; treat gallery scores as
in-distribution sanity numbers, not generalization evidence.
"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path

from .assets import REPO_ROOT, load_registry
from .prompts import (SYSTEM_FIRST_LAYER, SYSTEM_SECOND_LAYER,
                      build_first_layer_user, build_second_layer_user)
from .validators import (score_first_layer, score_second_layer, search_templates,
                         validate_first_layer)

GENERATED = REPO_ROOT / "training" / "generated"


def chat(base_url: str, model: str, api_key: str, system: str, user: str,
         json_mode: bool = True, timeout: float = 180.0) -> str:
    body = {"model": model, "temperature": 0, "max_tokens": 400, "stream": False,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}]}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"), headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def extract_json(text: str) -> dict:
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("no JSON object in output")
    return json.loads(match.group(0))


def mean(values: list) -> float:
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 4) if values else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--gold", action="store_true", help="用标准答案自测评分器")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-json-mode", action="store_true")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    registry = load_registry()
    cases = [json.loads(line) for line in
             (GENERATED / "frozen_set.jsonl").read_text(encoding="utf-8").splitlines() if line]
    if args.limit:
        cases = cases[:args.limit]

    rows = []
    started = time.time()
    for case in cases:
        row = {"caseId": case["caseId"], "source": case["source"]}
        if case.get("capabilityId"):
            gold1 = {"theme": (case.get("themeExpected")
                               or [next(iter(registry["themes"]))])[0],
                     "capability": case["capabilityId"],
                     "fields": case["candidateFields"], "action": case["eventIds"]}
            if args.gold:
                output, raw_error = gold1, None
            else:
                try:
                    output = extract_json(chat(args.base_url, args.model, args.api_key,
                                               SYSTEM_FIRST_LAYER,
                                               build_first_layer_user(case, registry),
                                               json_mode=not args.no_json_mode))
                    raw_error = None
                except Exception as exc:  # noqa: BLE001
                    output, raw_error = {}, str(exc)
            score = score_first_layer(output, case, registry)
            score["onePass"] = int(raw_error is None and
                                   not validate_first_layer(output, case, registry))
            row["firstLayer"] = score
            if raw_error:
                row["firstLayerError"] = raw_error[:200]
        if case.get("targetTemplateId") and case.get("businessId"):
            plans = search_templates(registry, case["businessId"], case["actionCount"])
            gold_index = next((i for i, p in enumerate(plans)
                               if p["templateId"] == case["targetTemplateId"]), None)
            if gold_index is not None and len(plans) >= 2:
                if args.gold:
                    output2 = {"plan": gold_index + 1}
                else:
                    try:
                        output2 = extract_json(chat(args.base_url, args.model, args.api_key,
                                                    SYSTEM_SECOND_LAYER,
                                                    build_second_layer_user(case, plans),
                                                    json_mode=not args.no_json_mode))
                    except Exception:  # noqa: BLE001
                        output2 = {}
                row["secondLayer"] = score_second_layer(output2, plans, case["targetTemplateId"])
        rows.append(row)
        done = len(rows)
        if done % 10 == 0 or done == len(cases):
            print("progress %d/%d (%.0fs)" % (done, len(cases), time.time() - started))

    report = {"model": "GOLD" if args.gold else args.model, "cases": len(rows)}
    for bucket_name, bucket in (("all", rows),
                                ("gallery", [r for r in rows if r["source"] == "gallery"]),
                                ("taskspec", [r for r in rows if r["source"] == "taskspec"])):
        first = [r["firstLayer"] for r in bucket if "firstLayer" in r]
        second = [r["secondLayer"] for r in bucket if "secondLayer" in r]
        report[bucket_name] = {
            "firstLayerN": len(first),
            "capabilityAcc": mean([s["capabilityAcc"] for s in first]),
            "fieldF1": mean([s["fieldF1"] for s in first]),
            "actionAcc": mean([s["actionAcc"] for s in first]),
            "themeAcc": mean([s["themeAcc"] for s in first]),
            "onePass": mean([s.get("onePass") for s in first]),
            "secondLayerN": len(second),
            "planAcc": mean([s["planAcc"] for s in second]),
        }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    out = Path(args.output) if args.output else GENERATED / (
        "eval_%s.json" % ("gold" if args.gold else args.model.replace(":", "-").replace("/", "-")))
    out.write_text(json.dumps({"report": report, "rows": rows}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print("saved:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
