#!/usr/bin/env python3
"""
Check Week15 temporal alignment regression gate.

This gate preserves the important behavior discovered on 2026-06-12:
- the original temporal alignment score must retain real drift evidence
- the remediation plan must identify and fix drifted event_local candidates
- the remediated score must have no failed rows and all event_local rows passing

This is a regression guard, not a semantic audio quality evaluator.
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


def get_summary(doc: dict[str, Any]) -> dict[str, Any]:
    obj = doc.get("summary", {})
    return obj if isinstance(obj, dict) else {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mainbase", default=".")
    ap.add_argument("--original", default="artifacts/evals/week15_temporal_alignment_summary.json")
    ap.add_argument("--plan", default="artifacts/evals/week15_temporal_alignment_remediation_plan.json")
    ap.add_argument("--remediated", default="artifacts/evals/week15_temporal_alignment_remediated_summary.json")
    ap.add_argument("--out", default="artifacts/evals/week15_temporal_alignment_regression_gate.json")
    args = ap.parse_args()

    root = Path(args.mainbase).expanduser().resolve()
    original_path = root / args.original
    plan_path = root / args.plan
    remediated_path = root / args.remediated
    out_path = root / args.out

    blockers: list[str] = []

    try:
        original = read_json(original_path)
    except Exception as exc:
        original = {}
        blockers.append(f"cannot read original summary: {original_path}: {exc}")

    try:
        plan = read_json(plan_path)
    except Exception as exc:
        plan = {}
        blockers.append(f"cannot read remediation plan: {plan_path}: {exc}")

    try:
        remediated = read_json(remediated_path)
    except Exception as exc:
        remediated = {}
        blockers.append(f"cannot read remediated summary: {remediated_path}: {exc}")

    original_summary = get_summary(original)
    remediated_summary = get_summary(remediated)
    actions = plan.get("actions", [])
    actions = actions if isinstance(actions, list) else []

    action_ids = sorted(str(a.get("candidateId")) for a in actions if a.get("candidateId"))
    expected_action_ids = ["procedural_v0_0004", "procedural_v0_0010"]

    checks = {
        "originalStatusFail": original.get("status") == "FAIL",
        "originalCandidateCountTen": original_summary.get("candidateCount") == 10,
        "originalFailCountTwo": original_summary.get("failCount") == 2,
        "originalEventLocalPassCountThree": original_summary.get("eventLocalPassCount") == 3,
        "planStatusPass": plan.get("status") == "PASS",
        "planHasTwoActions": len(actions) == 2,
        "planTargetsExpectedCandidates": action_ids == expected_action_ids,
        "remediatedStatusPass": remediated.get("status") == "PASS",
        "remediatedCandidateCountTen": remediated_summary.get("candidateCount") == 10,
        "remediatedFailCountZero": remediated_summary.get("failCount") == 0,
        "remediatedEventLocalPassCountFive": remediated_summary.get("eventLocalPassCount") == 5,
        "remediatedImprovesEventLocalPassCount": (
            isinstance(original_summary.get("eventLocalPassCount"), int)
            and isinstance(remediated_summary.get("eventLocalPassCount"), int)
            and remediated_summary.get("eventLocalPassCount") > original_summary.get("eventLocalPassCount")
        ),
        "remediatedDoesNotIncreaseFailCount": (
            isinstance(original_summary.get("failCount"), int)
            and isinstance(remediated_summary.get("failCount"), int)
            and remediated_summary.get("failCount") < original_summary.get("failCount")
        ),
    }

    failed_checks = [name for name, ok in checks.items() if not ok]
    for name in failed_checks:
        blockers.append(f"failed check: {name}")

    status = "PASS" if not blockers else "FAIL"

    report = {
        "schemaVersion": "week15.temporal_alignment_regression_gate.v1",
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "gateDecision": "TEMPORAL_ALIGNMENT_REMEDIATION_REGRESSION_GUARDED" if status == "PASS" else "BLOCKED",
        "inputs": {
            "originalSummary": str(original_path),
            "remediationPlan": str(plan_path),
            "remediatedSummary": str(remediated_path),
        },
        "checks": checks,
        "failedChecks": failed_checks,
        "summary": {
            "original": original_summary,
            "remediated": remediated_summary,
            "remediatedCandidateIds": action_ids,
            "eventLocalPassDelta": (
                remediated_summary.get("eventLocalPassCount", 0)
                - original_summary.get("eventLocalPassCount", 0)
                if isinstance(remediated_summary.get("eventLocalPassCount"), int)
                and isinstance(original_summary.get("eventLocalPassCount"), int)
                else None
            ),
            "failCountDelta": (
                remediated_summary.get("failCount", 0)
                - original_summary.get("failCount", 0)
                if isinstance(remediated_summary.get("failCount"), int)
                and isinstance(original_summary.get("failCount"), int)
                else None
            ),
        },
        "blockers": blockers,
        "boundary": [
            "regression_gate_only",
            "preserves_original_failure_as_baseline",
            "does_not_claim_semantic_audio_quality",
            "does_not_claim_human_audition_passed",
            "does_not_claim_final_mix_readiness",
        ],
        "nextSystemStep": {
            "java": "Expose temporal alignment remediation status and artifact links through a focused API contract.",
            "cloud": "Consume the regression gate as platform readiness input without claiming production SLO.",
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps({
        "output": str(out_path),
        "status": status,
        "gateDecision": report["gateDecision"],
        "failedChecks": failed_checks,
        "blockers": blockers,
        "summary": report["summary"],
    }, indent=2, ensure_ascii=False))

    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())