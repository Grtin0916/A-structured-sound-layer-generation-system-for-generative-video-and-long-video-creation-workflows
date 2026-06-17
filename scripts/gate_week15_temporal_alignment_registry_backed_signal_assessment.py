#!/usr/bin/env python3
import json
from datetime import datetime, timezone
from pathlib import Path

ASSESSMENT = Path("artifacts/reviews/week15_temporal_alignment_registry_backed_signal_audition_assessment.json")
STRICT_QUEUE = Path("artifacts/reviews/week15_temporal_alignment_registry_backed_audition_queue_strict.json")
OUT = Path("artifacts/reviews/week15_temporal_alignment_registry_backed_signal_assessment_gate.json")

def first_burst_start(item, role):
    for a in item.get("audioReports", []):
        if a.get("role") == role:
            intervals = a.get("burstIntervals") or []
            if intervals:
                return intervals[0].get("startSec")
    return None

def burst_count(item, role):
    for a in item.get("audioReports", []):
        if a.get("role") == role:
            return a.get("burstCount")
    return None

def peak_abs(item, role):
    for a in item.get("audioReports", []):
        if a.get("role") == role:
            return a.get("peakAbs")
    return None

def main():
    if not ASSESSMENT.exists():
        raise SystemExit(f"missing assessment: {ASSESSMENT}")
    if not STRICT_QUEUE.exists():
        raise SystemExit(f"missing strict queue: {STRICT_QUEUE}")

    d = json.loads(ASSESSMENT.read_text(encoding="utf-8"))
    q = json.loads(STRICT_QUEUE.read_text(encoding="utf-8"))

    items = {x.get("candidateId"): x for x in d.get("items", [])}
    q_items = {x.get("candidateId"): x for x in q.get("items", [])}

    i4 = items.get("procedural_v0_0004", {})
    b0 = first_burst_start(i4, "baseline")
    r0 = first_burst_start(i4, "remediated")
    b_count = burst_count(i4, "baseline")
    r_count = burst_count(i4, "remediated")
    b_peak = peak_abs(i4, "baseline")
    r_peak = peak_abs(i4, "remediated")

    checks = {
        "assessment_decision_partial": d.get("decision") == "SIGNAL_ASSESSMENT_PARTIAL_NOT_HUMAN_REVIEW_PASS",
        "no_human_review_pass_claim": "HUMAN_REVIEW_PASS is not claimed" in d.get("claimBoundary", ""),
        "visual_risk_preserved": d.get("visualAssessment", {}).get("risk") == "POSSIBLE_ONSET_PROXY_ANNOTATION_BUG_OR_OVER_SENSITIVE_THRESHOLD",
        "strict_queue_boundary_preserved": "HUMAN_REVIEW_PASS is not claimed" in q.get("claimBoundary", ""),
        "four_items_present": set(items) == {"procedural_v0_0002", "procedural_v0_0003", "procedural_v0_0004", "procedural_v0_0007"},
        "strict_queue_four_items_present": set(q_items) == {"procedural_v0_0002", "procedural_v0_0003", "procedural_v0_0004", "procedural_v0_0007"},
        "0004_signal_supported": i4.get("signalJudgement") == "PASS_SIGNAL_SUPPORTED",
        "0004_timing_supported": i4.get("timingJudgement") == "PASS_SIGNAL_SUPPORTED",
        "0004_semantic_not_promoted": i4.get("semanticJudgement") == "UNCERTAIN",
        "0004_human_not_performed": i4.get("humanReviewStatus") == "NOT_PERFORMED",
        "0004_baseline_late_burst": b0 is not None and b0 >= 1.0,
        "0004_remediated_early_burst": r0 is not None and r0 <= 0.10,
        "0004_burst_count_preserved": b_count == r_count == 2,
        "0004_peak_preserved": b_peak is not None and r_peak is not None and abs(b_peak - r_peak) <= 0.001,
    }

    for cid in ["procedural_v0_0002", "procedural_v0_0003", "procedural_v0_0007"]:
        item = items.get(cid, {})
        checks[f"{cid}_signal_not_promoted"] = item.get("signalJudgement") == "UNCERTAIN"
        checks[f"{cid}_timing_not_promoted"] = item.get("timingJudgement") == "UNCERTAIN"
        checks[f"{cid}_semantic_not_promoted"] = item.get("semanticJudgement") == "UNCERTAIN"
        checks[f"{cid}_human_not_performed"] = item.get("humanReviewStatus") == "NOT_PERFORMED"

    decision = "PASS_WEEK15_REGISTRY_BACKED_SIGNAL_ASSESSMENT_GATE" if all(checks.values()) else "FAIL_WEEK15_REGISTRY_BACKED_SIGNAL_ASSESSMENT_GATE"

    report = {
        "schemaVersion": "week15.mainbase.temporal-alignment.registry-backed-signal-assessment-gate.v1",
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "assessmentPath": str(ASSESSMENT),
        "strictQueuePath": str(STRICT_QUEUE),
        "visualRisk": d.get("visualAssessment", {}).get("risk"),
        "claimBoundary": "Gate enforces signal-only partial assessment. It must not promote HUMAN_REVIEW_PASS or semantic PASS.",
        "checks": checks,
        "0004SignalDelta": {
            "baselineFirstBurstSec": b0,
            "remediatedFirstBurstSec": r0,
            "baselineBurstCount": b_count,
            "remediatedBurstCount": r_count,
            "baselinePeakAbs": b_peak,
            "remediatedPeakAbs": r_peak,
        },
        "nextAllowedAction": (
            "Manual human audition may be recorded only from explicit user-provided semantic/timing/usable judgements. "
            "Do not use this gate to claim human review pass."
        ),
    }

    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if decision != "PASS_WEEK15_REGISTRY_BACKED_SIGNAL_ASSESSMENT_GATE":
        raise SystemExit(4)

if __name__ == "__main__":
    main()
