#!/usr/bin/env python3
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path.cwd()

READINESS_JSON = ROOT / "artifacts/evals/week16_s3_to_w17_layer_mix_input.json"

OUT_JSON = ROOT / "artifacts/evals/week17_layer_mix_plan_v0.json"
OUT_CSV = ROOT / "artifacts/evals/week17_layer_mix_plan_v0.csv"
OUT_DOC = ROOT / "docs/evals/week17_layer_mix_plan_v0.md"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def find_candidate_artifacts(candidate_id: str) -> list[str]:
    search_roots = [
        ROOT / "artifacts/audio",
        ROOT / "artifacts/wav",
        ROOT / "artifacts/evals",
        ROOT / "results",
    ]
    hits: list[str] = []
    for base in search_roots:
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if p.is_file() and candidate_id in p.name:
                hits.append(str(p))
    return sorted(set(hits))


def require_counts(records: list[dict[str, Any]]) -> None:
    role_counts: dict[str, int] = {}
    mix_counts: dict[str, int] = {}
    for r in records:
        role = str(r.get("fixtureRole"))
        mix = str(r.get("mixEligibility"))
        role_counts[role] = role_counts.get(role, 0) + 1
        mix_counts[mix] = mix_counts.get(mix, 0) + 1

    expected_roles = {
        "P1_PAIRED_REGRESSION_FIXTURE": 2,
        "P2_THRESHOLD_MARGIN_FIXTURE": 1,
        "P4_NUMERIC_MARGIN_CONTROL": 7,
    }
    expected_mix = {
        "BLOCK_AUTOMIX_REGRESSION_ONLY": 2,
        "MONITOR_ONLY_THRESHOLD_MARGIN": 1,
        "ELIGIBLE_CONTROL_ONLY": 7,
    }

    if role_counts != expected_roles:
        raise RuntimeError(f"Unexpected fixtureRoleCounts: {role_counts}")
    if mix_counts != expected_mix:
        raise RuntimeError(f"Unexpected mixEligibilityCounts: {mix_counts}")


def main() -> int:
    if not READINESS_JSON.exists():
        raise FileNotFoundError(f"Missing readiness input: {READINESS_JSON}")

    source = json.loads(READINESS_JSON.read_text(encoding="utf-8"))
    records = source.get("records", [])
    if not isinstance(records, list) or not records:
        raise RuntimeError("readiness records missing or empty")

    require_counts(records)

    selected_controls: list[dict[str, Any]] = []
    blocked_regression: list[dict[str, Any]] = []
    monitor_only: list[dict[str, Any]] = []

    for r in records:
        cid = str(r.get("candidateId"))
        role = str(r.get("fixtureRole"))
        mix = str(r.get("mixEligibility"))
        artifact_hits = find_candidate_artifacts(cid)

        item = {
            "candidateId": cid,
            "fixtureRole": role,
            "mixEligibility": mix,
            "nextAction": r.get("nextAction"),
            "requiredSafeguard": r.get("requiredSafeguard"),
            "artifactPathStatus": "FOUND" if artifact_hits else "NOT_FOUND",
            "artifactPaths": artifact_hits,
        }

        if mix == "ELIGIBLE_CONTROL_ONLY":
            selected_controls.append(item)
        elif mix == "BLOCK_AUTOMIX_REGRESSION_ONLY":
            blocked_regression.append(item)
        elif mix == "MONITOR_ONLY_THRESHOLD_MARGIN":
            monitor_only.append(item)
        else:
            raise RuntimeError(f"Unknown mixEligibility for {cid}: {mix}")

    all_plan_items = selected_controls + blocked_regression + monitor_only
    audio_artifact_available_total = sum(
        1 for item in all_plan_items if item.get("artifactPathStatus") == "FOUND"
    )
    audio_artifact_missing_total = len(all_plan_items) - audio_artifact_available_total

    selected_control_artifact_available_total = sum(
        1 for item in selected_controls if item.get("artifactPathStatus") == "FOUND"
    )
    selected_control_artifact_missing_total = (
        len(selected_controls) - selected_control_artifact_available_total
    )

    mix_execution_readiness_decision = (
        "READY_WEEK17_LAYER_MIX_EXECUTION_SELECTED_CONTROL_INPUTS_AVAILABLE"
        if selected_control_artifact_missing_total == 0
        else "BLOCKED_WEEK17_LAYER_MIX_EXECUTION_SELECTED_CONTROL_AUDIO_ARTIFACTS_MISSING"
    )

    plan = {
        "decision": "PASS_WEEK17_LAYER_MIX_PLAN_V0",
        "mixExecutionReadinessDecision": mix_execution_readiness_decision,
        "audioArtifactAvailableTotal": audio_artifact_available_total,
        "audioArtifactMissingTotal": audio_artifact_missing_total,
        "selectedControlArtifactAvailableTotal": selected_control_artifact_available_total,
        "selectedControlArtifactMissingTotal": selected_control_artifact_missing_total,
        "generatedAtUtc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "schemaVersion": "week17.layer_mix_plan.v0",
        "sourceReadinessArtifact": str(READINESS_JSON),
        "sourceReadinessSha256": sha256(READINESS_JSON),
        "candidateTotal": len(records),
        "selectedControlInputTotal": len(selected_controls),
        "blockedRegressionFixtureTotal": len(blocked_regression),
        "monitorOnlyFixtureTotal": len(monitor_only),
        "realMixerExecuted": False,
        "realMixerExecutionAllowed": selected_control_artifact_missing_total == 0,
        "wavExported": False,
        "finalMixReadinessClaimed": False,
        "semanticAudioQualityClaimed": False,
        "selectedControlInputs": selected_controls,
        "blockedRegressionFixtures": blocked_regression,
        "monitorOnlyFixtures": monitor_only,
        "blockedClaims": [
            "real layer mixer executed",
            "wav export completed",
            "final mix readiness",
            "semantic audio quality pass",
            "human review pass",
            "production mixer availability",
        ],
    }

    OUT_JSON.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "candidateId",
            "planBucket",
            "fixtureRole",
            "mixEligibility",
            "artifactPathStatus",
            "nextAction",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for bucket_name, rows in [
            ("selected_control_input", selected_controls),
            ("blocked_regression_fixture", blocked_regression),
            ("monitor_only_fixture", monitor_only),
        ]:
            for r in rows:
                writer.writerow({
                    "candidateId": r["candidateId"],
                    "planBucket": bucket_name,
                    "fixtureRole": r["fixtureRole"],
                    "mixEligibility": r["mixEligibility"],
                    "artifactPathStatus": r["artifactPathStatus"],
                    "nextAction": r["nextAction"],
                })

    OUT_DOC.write_text(
        "# Week17 Layer Mix Plan V0\n\n"
        "## Purpose\n\n"
        "Convert the S3-to-W17 readiness gate into a concrete layer-mix input plan.\n\n"
        "## Result\n\n"
        f"- Selected control inputs: {len(selected_controls)}\n"
        f"- Blocked regression fixtures: {len(blocked_regression)}\n"
        f"- Monitor-only threshold fixtures: {len(monitor_only)}\n"
        f"- Audio artifact available inputs: {audio_artifact_available_total}\n"
        f"- Audio artifact missing inputs: {audio_artifact_missing_total}\n"
        f"- Selected control artifact available inputs: {selected_control_artifact_available_total}\n"
        f"- Selected control artifact missing inputs: {selected_control_artifact_missing_total}\n"
        f"- Mix execution readiness: {mix_execution_readiness_decision}\n\n"
        "## Boundary\n\n"
        "- This plan does not execute a real layer mixer.\n"
        "- This plan does not export wav files.\n"
        "- This plan does not claim final mix readiness.\n"
        "- This plan does not claim semantic audio quality pass.\n",
        encoding="utf-8",
    )

    print(json.dumps({
        "decision": plan["decision"],
        "candidateTotal": plan["candidateTotal"],
        "selectedControlInputTotal": plan["selectedControlInputTotal"],
        "blockedRegressionFixtureTotal": plan["blockedRegressionFixtureTotal"],
        "monitorOnlyFixtureTotal": plan["monitorOnlyFixtureTotal"],
        "mixExecutionReadinessDecision": plan["mixExecutionReadinessDecision"],
        "audioArtifactAvailableTotal": plan["audioArtifactAvailableTotal"],
        "audioArtifactMissingTotal": plan["audioArtifactMissingTotal"],
        "selectedControlArtifactAvailableTotal": plan["selectedControlArtifactAvailableTotal"],
        "selectedControlArtifactMissingTotal": plan["selectedControlArtifactMissingTotal"],
        "realMixerExecuted": plan["realMixerExecuted"],
        "realMixerExecutionAllowed": plan["realMixerExecutionAllowed"],
        "wavExported": plan["wavExported"],
        "outJson": str(OUT_JSON),
        "outCsv": str(OUT_CSV),
        "outDoc": str(OUT_DOC),
    }, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())