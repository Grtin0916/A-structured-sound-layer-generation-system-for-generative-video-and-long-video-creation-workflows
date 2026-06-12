#!/usr/bin/env python3
"""
Build Week13 Candidate Audio Bank V1 stage-gate transition feedback.

Purpose:
- Consume Cloud Friday stage gate input.
- Consume Mainbase platform promotion feedback index.
- Produce a Mainbase-side transition record for Week15 Temporal Alignment Eval.

This does not regenerate audio, placement, mix preview, or semantic quality results.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def find_key(obj: Any, names: list[str]) -> Any:
    targets = {n.lower() for n in names}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower() in targets:
                return v
        for v in obj.values():
            found = find_key(v, names)
            if found is not None:
                return found
    if isinstance(obj, list):
        for item in obj:
            found = find_key(item, names)
            if found is not None:
                return found
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cloud", required=True)
    ap.add_argument("--mainbase", default=".")
    ap.add_argument(
        "--out",
        default="artifacts/manifests/week13_candidate_bank_v1_stage_gate_feedback.json",
    )
    args = ap.parse_args()

    mainbase = Path(args.mainbase).expanduser().resolve()
    cloud = Path(args.cloud).expanduser().resolve()

    mainbase_feedback_path = mainbase / "artifacts/manifests/week13_platform_promotion_feedback_index.json"
    cloud_stage_gate_path = cloud / "loadtest/reports/week13_friday_stage_gate_input.json"

    blockers: list[str] = []

    try:
        mainbase_feedback = read_json(mainbase_feedback_path)
    except Exception as exc:
        mainbase_feedback = {}
        blockers.append(f"cannot read mainbase feedback: {mainbase_feedback_path}: {exc}")

    try:
        cloud_stage_gate = read_json(cloud_stage_gate_path)
    except Exception as exc:
        cloud_stage_gate = {}
        blockers.append(f"cannot read cloud friday stage gate input: {cloud_stage_gate_path}: {exc}")

    mainbase_status = find_key(mainbase_feedback, ["status"])
    mainbase_decision = find_key(mainbase_feedback, ["platformPromotionDecision", "promotionDecision", "decision"])

    cloud_status = cloud_stage_gate.get("status")
    cloud_decision = cloud_stage_gate.get("stageGateDecision")
    cloud_summary = cloud_stage_gate.get("summary", {})
    cloud_blockers = cloud_stage_gate.get("blockers", [])

    if mainbase_status != "PASS":
        blockers.append(f"mainbase feedback status expected PASS, got {mainbase_status}")
    if mainbase_decision != "PROMOTE_TO_WEEK13_DEMO_READY":
        blockers.append(f"mainbase decision unexpected: {mainbase_decision}")
    if cloud_status != "PASS":
        blockers.append(f"cloud friday stage gate status expected PASS, got {cloud_status}")
    if cloud_decision != "READY_FOR_FRIDAY_STAGE_GATE":
        blockers.append(f"cloud stageGateDecision unexpected: {cloud_decision}")
    if cloud_blockers:
        blockers.append(f"cloud stage gate blockers not empty: {cloud_blockers}")

    expected_summary = {
        "candidateCount": 10,
        "workerSuccessCount": 10,
        "drilldownReadyCount": 10,
        "javaApiTestsRun": 1,
        "javaFailures": 0,
        "javaErrors": 0,
        "negativeTarget": "procedural_v0_0002",
    }

    for key, expected in expected_summary.items():
        actual = cloud_summary.get(key)
        if actual != expected:
            blockers.append(f"summary.{key} expected {expected!r}, got {actual!r}")

    status = "PASS" if not blockers else "FAIL"

    report = {
        "schemaVersion": "week13.candidate_bank_v1_stage_gate_feedback.v1",
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "transitionDecision": "ENTER_WEEK15_TEMPORAL_ALIGNMENT_EVAL_INPUT" if status == "PASS" else "BLOCKED",
        "sourceInputs": {
            "mainbasePlatformPromotionFeedbackIndex": str(mainbase_feedback_path),
            "cloudFridayStageGateInput": str(cloud_stage_gate_path),
        },
        "verifiedScope": {
            "completedStage": "Week13 Candidate Audio Bank V1 local demo-ready input",
            "nextStage": "Week15 Temporal Alignment Eval",
            "allowedClaims": [
                "10 candidate audio records have local demo-ready gate evidence.",
                "Java readiness API contract and Cloud promotion gate evidence are aggregated.",
                "Negative regression target procedural_v0_0002 is represented.",
                "This is a transition input for temporal alignment evaluation."
            ],
            "forbiddenClaims": [
                "semantic audio quality is solved",
                "human audition passed",
                "final mix readiness",
                "production Kubernetes Job",
                "live Grafana import",
                "cloud object storage readiness",
                "production SLO"
            ],
        },
        "stageGateSummary": cloud_summary,
        "blockers": blockers,
        "nextActions": [
            {
                "name": "Week15 temporal alignment eval input",
                "goal": "use existing candidate placement and mix evidence to score event timestamp vs audio energy/onset alignment",
                "expectedArtifacts": [
                    "artifacts/evals/week15_temporal_alignment_input_index.json",
                    "artifacts/evals/week15_temporal_alignment.csv"
                ],
            },
            {
                "name": "human audition rubric draft",
                "goal": "separate timing correctness from semantic/audio-quality judgment",
                "expectedArtifacts": [
                    "docs/evals/week15_human_audition_rubric.md"
                ],
            },
        ],
    }

    out = mainbase / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps({
        "output": str(out),
        "status": status,
        "transitionDecision": report["transitionDecision"],
        "blockerCount": len(blockers),
        "blockers": blockers,
    }, indent=2, ensure_ascii=False))

    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())