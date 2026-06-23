from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

SEED_JSON = ROOT / "artifacts/evals/week16_temporal_alignment_failure_taxonomy_seed.json"
SEED_CSV = ROOT / "artifacts/evals/week16_temporal_alignment_failure_taxonomy_seed.csv"

REPORT_JSON = ROOT / "artifacts/evals/week16_temporal_alignment_failure_regression_report.json"
REPORT_CSV = ROOT / "artifacts/evals/week16_temporal_alignment_failure_regression_report.csv"
REGISTRY_JSON = ROOT / "artifacts/evals/week16_temporal_alignment_failure_fixture_registry.json"

EXPECTED_P1_IDS = {"procedural_v0_0004", "procedural_v0_0010"}
EXPECTED_P2_IDS = {"procedural_v0_0007"}

GLOBAL_BLOCKED_CLAIMS = [
    "semantic_audio_quality_pass_not_verified",
    "human_review_pass_not_verified",
    "final_mix_readiness_not_verified",
    "live_java_service_availability_not_verified",
    "live_prometheus_or_grafana_import_not_verified",
    "production_slo_or_real_cloud_deployment_not_verified",
]

ID_KEYS = [
    "candidateId",
    "candidate_id",
    "caseId",
    "case_id",
    "sampleId",
    "sample_id",
    "id",
]


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def get_str(row: dict[str, Any], key: str, default: str = "") -> str:
    value = row.get(key, default)
    if value is None:
        return default
    return str(value).strip()


def load_json_candidates(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    data = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]

    if not isinstance(data, dict):
        return []

    for key in [
        "candidates",
        "records",
        "rows",
        "items",
        "results",
        "taxonomy",
        "failureTaxonomy",
        "candidateTaxonomy",
    ]:
        value = data.get(key)
        if isinstance(value, list) and all(isinstance(x, dict) for x in value):
            return list(value)

    # Fallback for nested summary structures, but avoid treating the whole report as one row.
    out: list[dict[str, Any]] = []

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            has_candidate_key = any(k in obj for k in ID_KEYS)
            has_seed_fields = any(
                k in obj
                for k in [
                    "originalStatus",
                    "remediatedStatus",
                    "w16FailureBucket",
                    "severity",
                ]
            )
            if has_candidate_key and has_seed_fields:
                out.append(obj)
                return
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(data)
    return out


def load_csv_candidates(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def candidate_id(row: dict[str, Any], index: int) -> str:
    for key in ID_KEYS:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return f"unknown_candidate_{index:04d}"


def normalize_seed_row(row: dict[str, Any], index: int) -> dict[str, Any]:
    cid = candidate_id(row, index)
    original_status = get_str(row, "originalStatus").upper()
    remediated_status = get_str(row, "remediatedStatus").upper()
    seed_bucket = get_str(row, "w16FailureBucket")
    seed_severity = get_str(row, "severity").upper()

    return {
        "candidateId": cid,
        "originalStatus": original_status,
        "remediatedStatus": remediated_status,
        "originalOnsetDeltaSec": parse_float(row.get("originalOnsetDeltaSec")),
        "originalAbsOnsetDeltaSec": parse_float(row.get("originalAbsOnsetDeltaSec")),
        "remediatedOnsetDeltaSec": parse_float(row.get("remediatedOnsetDeltaSec")),
        "remediatedAbsOnsetDeltaSec": parse_float(row.get("remediatedAbsOnsetDeltaSec")),
        "actionable": parse_bool(row.get("actionable")),
        "alertEligible": parse_bool(row.get("alertEligible")),
        "hasWaveformEvidence": parse_bool(row.get("hasWaveformEvidence")),
        "seedFailureBucket": seed_bucket,
        "seedSeverity": seed_severity,
        "recommendedNextAction": get_str(row, "recommendedNextAction"),
        "sourceFiles": get_str(row, "sourceFiles"),
        "rawSeedRow": row,
    }


def classify_from_evidence(seed: dict[str, Any]) -> dict[str, Any]:
    original = seed["originalStatus"]
    remediated = seed["remediatedStatus"]
    bucket = seed["seedFailureBucket"].lower()
    severity = seed["seedSeverity"]
    actionable = seed["actionable"]
    alert_eligible = seed["alertEligible"]

    is_actionable_remediated_drift = (
        severity == "P1"
        or "timing_drift_actionable_remediated" in bucket
        or (
            original.startswith("FAIL")
            and remediated == "PASS"
            and actionable
            and alert_eligible
        )
    )

    if is_actionable_remediated_drift:
        return {
            "failureBucket": "P1_ACTIONABLE_REMEDIATED_TIMING_DRIFT",
            "severity": "P1",
            "fixtureRole": "paired_regression_fixture",
            "eligibleForRerun": True,
            "rerunReason": "evidence shows actionable timing drift that was remediated; preserve original/remediated pair",
            "nextAction": "Use as paired fixture for rerun-plan and future regression checks.",
            "blockedClaims": GLOBAL_BLOCKED_CLAIMS,
            "classificationEvidence": {
                "rule": "severity=P1 OR timing_drift_actionable_remediated OR FAIL->PASS actionable+alertEligible",
                "originalStatus": original,
                "remediatedStatus": remediated,
                "seedFailureBucket": seed["seedFailureBucket"],
                "seedSeverity": severity,
                "actionable": actionable,
                "alertEligible": alert_eligible,
            },
        }

    is_threshold_margin = (
        severity == "P2"
        or "warn_near_miss" in bucket
        or original == "WARN_NEAR_MISS"
        or remediated == "WARN_NEAR_MISS"
    )

    if is_threshold_margin:
        return {
            "failureBucket": "P2_WARN_NEAR_MISS_THRESHOLD_MARGIN",
            "severity": "P2",
            "fixtureRole": "threshold_margin_fixture",
            "eligibleForRerun": False,
            "rerunReason": "evidence shows near-threshold margin; monitor boundary instead of triggering actionable rerun",
            "nextAction": "Use as threshold boundary guard; do not alert as actionable failure.",
            "blockedClaims": GLOBAL_BLOCKED_CLAIMS,
            "classificationEvidence": {
                "rule": "severity=P2 OR warn_near_miss bucket/status",
                "originalStatus": original,
                "remediatedStatus": remediated,
                "seedFailureBucket": seed["seedFailureBucket"],
                "seedSeverity": severity,
                "actionable": actionable,
                "alertEligible": alert_eligible,
            },
        }

    is_pass_control = (
        severity == "P4"
        or "pass_low_risk" in bucket
        or (original == "PASS" and remediated == "PASS")
    )

    if is_pass_control:
        return {
            "failureBucket": "P4_PASS_NUMERIC_MARGIN_CONTROL",
            "severity": "P4",
            "fixtureRole": "numeric_margin_control",
            "eligibleForRerun": False,
            "rerunReason": "evidence shows pass control; no rerun unless future regression flips decision",
            "nextAction": "Keep as numeric margin baseline and suppress false positive alerting.",
            "blockedClaims": GLOBAL_BLOCKED_CLAIMS,
            "classificationEvidence": {
                "rule": "severity=P4 OR pass_low_risk bucket OR PASS->PASS",
                "originalStatus": original,
                "remediatedStatus": remediated,
                "seedFailureBucket": seed["seedFailureBucket"],
                "seedSeverity": severity,
                "actionable": actionable,
                "alertEligible": alert_eligible,
            },
        }

    return {
        "failureBucket": "P3_UNCLASSIFIED_EVIDENCE_GAP",
        "severity": "P3",
        "fixtureRole": "evidence_gap_fixture",
        "eligibleForRerun": False,
        "rerunReason": "insufficient evidence for deterministic rerun eligibility",
        "nextAction": "Inspect seed evidence and extend taxonomy rules before downstream consumption.",
        "blockedClaims": GLOBAL_BLOCKED_CLAIMS,
        "classificationEvidence": {
            "rule": "fallback_unclassified",
            "originalStatus": original,
            "remediatedStatus": remediated,
            "seedFailureBucket": seed["seedFailureBucket"],
            "seedSeverity": severity,
            "actionable": actionable,
            "alertEligible": alert_eligible,
        },
    }


def unique_by_candidate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for i, row in enumerate(rows):
        seed = normalize_seed_row(row, i)
        out[seed["candidateId"]] = seed
    return [out[k] for k in sorted(out)]


def main() -> None:
    json_rows = load_json_candidates(SEED_JSON)
    csv_rows = load_csv_candidates(SEED_CSV)

    rows = unique_by_candidate(json_rows)
    source_used = "json"
    if len(rows) < 10 and csv_rows:
        rows = unique_by_candidate(csv_rows)
        source_used = "csv"

    if not rows:
        raise SystemExit("No candidate rows loaded from seed JSON or CSV.")

    records: list[dict[str, Any]] = []
    for seed in rows:
        cls = classify_from_evidence(seed)
        records.append(
            {
                "candidateId": seed["candidateId"],
                **cls,
                "seedEvidence": {
                    k: v
                    for k, v in seed.items()
                    if k != "rawSeedRow"
                },
            }
        )

    p1 = [r["candidateId"] for r in records if r["severity"] == "P1"]
    p2 = [r["candidateId"] for r in records if r["severity"] == "P2"]
    p4 = [r["candidateId"] for r in records if r["severity"] == "P4"]
    p3 = [r["candidateId"] for r in records if r["severity"] == "P3"]

    decision_errors: list[str] = []
    if len(records) != 10:
        decision_errors.append(f"candidateTotal expected 10, got {len(records)}")
    if set(p1) != EXPECTED_P1_IDS:
        decision_errors.append(f"P1 ids expected {sorted(EXPECTED_P1_IDS)}, got {sorted(p1)}")
    if set(p2) != EXPECTED_P2_IDS:
        decision_errors.append(f"P2 ids expected {sorted(EXPECTED_P2_IDS)}, got {sorted(p2)}")
    if len(p4) != 7:
        decision_errors.append(f"P4 control count expected 7, got {len(p4)}")
    if p3:
        decision_errors.append(f"unclassified P3 evidence gaps found: {p3}")

    decision = (
        "PASS_WEEK16_FAILURE_REGRESSION_REPORT_V0_EVIDENCE_DRIVEN"
        if not decision_errors
        else "FAIL_WEEK16_FAILURE_REGRESSION_REPORT_V0_EVIDENCE_DRIVEN"
    )

    report = {
        "schemaVersion": "week16.failure_regression_report.v0",
        "classificationMode": "evidence_driven_with_expected_fixture_consistency_guard",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "decisionErrors": decision_errors,
        "sourceMode": source_used,
        "sourceFiles": [
            str(SEED_JSON.relative_to(ROOT)),
            str(SEED_CSV.relative_to(ROOT)),
        ],
        "summary": {
            "candidateTotal": len(records),
            "p1RegressionFixtureTotal": len(p1),
            "thresholdFixtureTotal": len(p2),
            "passControlTotal": len(p4),
            "evidenceGapFixtureTotal": len(p3),
            "p1RegressionFixtureIds": p1,
            "thresholdFixtureIds": p2,
            "passControlIds": p4,
            "evidenceGapFixtureIds": p3,
            "blockedClaimTotal": len(GLOBAL_BLOCKED_CLAIMS),
            "blockedClaims": GLOBAL_BLOCKED_CLAIMS,
        },
        "records": records,
        "downstreamContract": {
            "javaRerunPlanRequiredFields": [
                "candidateId",
                "failureBucket",
                "eligibleForRerun",
                "rerunReason",
                "idempotencyKeyPolicy",
                "previousAttemptPreserved",
            ],
            "cloudFaultDrillInputs": [
                "failureBucket",
                "severity",
                "fixtureRole",
                "blockedClaims",
                "classificationEvidence",
            ],
            "blockedClaimsBoundary": "Do not claim semantic quality pass, human review pass, final mix readiness, live Java availability, live Grafana/Prometheus import, production SLO, or real cloud deployment.",
        },
    }

    registry = {
        "schemaVersion": "week16.failure_fixture_registry.v0",
        "classificationMode": report["classificationMode"],
        "generatedAt": report["generatedAt"],
        "decision": decision,
        "decisionErrors": decision_errors,
        "fixtures": [
            {
                "candidateId": r["candidateId"],
                "failureBucket": r["failureBucket"],
                "severity": r["severity"],
                "fixtureRole": r["fixtureRole"],
                "eligibleForRerun": r["eligibleForRerun"],
                "rerunReason": r["rerunReason"],
                "nextAction": r["nextAction"],
                "blockedClaims": r["blockedClaims"],
                "classificationEvidence": r["classificationEvidence"],
            }
            for r in records
        ],
    }

    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    REGISTRY_JSON.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with REPORT_CSV.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "candidateId",
            "failureBucket",
            "severity",
            "fixtureRole",
            "eligibleForRerun",
            "originalStatus",
            "remediatedStatus",
            "originalAbsOnsetDeltaSec",
            "remediatedAbsOnsetDeltaSec",
            "seedFailureBucket",
            "seedSeverity",
            "actionable",
            "alertEligible",
            "hasWaveformEvidence",
            "rerunReason",
            "nextAction",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            seed = r["seedEvidence"]
            writer.writerow(
                {
                    "candidateId": r["candidateId"],
                    "failureBucket": r["failureBucket"],
                    "severity": r["severity"],
                    "fixtureRole": r["fixtureRole"],
                    "eligibleForRerun": r["eligibleForRerun"],
                    "originalStatus": seed["originalStatus"],
                    "remediatedStatus": seed["remediatedStatus"],
                    "originalAbsOnsetDeltaSec": seed["originalAbsOnsetDeltaSec"],
                    "remediatedAbsOnsetDeltaSec": seed["remediatedAbsOnsetDeltaSec"],
                    "seedFailureBucket": seed["seedFailureBucket"],
                    "seedSeverity": seed["seedSeverity"],
                    "actionable": seed["actionable"],
                    "alertEligible": seed["alertEligible"],
                    "hasWaveformEvidence": seed["hasWaveformEvidence"],
                    "rerunReason": r["rerunReason"],
                    "nextAction": r["nextAction"],
                }
            )

    print(f"decision={decision}")
    print(f"decisionErrors={decision_errors}")
    print(f"classificationMode={report['classificationMode']}")
    print(f"sourceMode={source_used}")
    print(f"candidateTotal={len(records)}")
    print(f"p1RegressionFixtureIds={p1}")
    print(f"thresholdFixtureIds={p2}")
    print(f"passControlTotal={len(p4)}")
    print(f"evidenceGapFixtureTotal={len(p3)}")
    print(f"blockedClaimTotal={len(GLOBAL_BLOCKED_CLAIMS)}")

    if decision_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
