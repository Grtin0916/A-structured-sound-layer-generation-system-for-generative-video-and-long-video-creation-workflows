#!/usr/bin/env python3
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path.cwd()

REGRESSION_JSON = ROOT / "artifacts/evals/week16_temporal_alignment_failure_regression_report.json"
REGRESSION_CSV = ROOT / "artifacts/evals/week16_temporal_alignment_failure_regression_report.csv"
FIXTURE_JSON = ROOT / "artifacts/evals/week16_temporal_alignment_failure_fixture_registry.json"

OUT_JSON = ROOT / "artifacts/evals/week16_s3_to_w17_layer_mix_input.json"
OUT_CSV = ROOT / "artifacts/evals/week16_s3_to_w17_layer_mix_input.csv"
OUT_DOC = ROOT / "docs/evals/week16_s3_to_w17_layer_mix_input.md"


def norm_key(key: str) -> str:
    return key.replace("_", "").replace("-", "").replace(" ", "").lower()


def get_any(d: dict[str, Any], names: list[str], default: Any = None) -> Any:
    wanted = {norm_key(x) for x in names}
    for k, v in d.items():
        if norm_key(k) in wanted:
            return v
    return default


def as_list(x: Any) -> list[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, dict):
        return [x]
    return []


def walk_candidate_records(obj: Any, out: list[dict[str, Any]]) -> None:
    if isinstance(obj, dict):
        cid = get_any(obj, ["candidateId", "candidate_id", "id", "caseId", "case_id"])
        useful_keys = {
            "failurebucket",
            "severity",
            "fixturerole",
            "actionability",
            "candidateid",
            "caseid",
            "blockedclaims",
            "nextaction",
            "mixeligibility",
        }
        has_signal = any(norm_key(k) in useful_keys for k in obj.keys())
        if cid is not None and has_signal:
            out.append(obj)
        for v in obj.values():
            walk_candidate_records(v, out)
    elif isinstance(obj, list):
        for item in obj:
            walk_candidate_records(item, out)


def read_json_records(path: Path) -> tuple[Any, list[dict[str, Any]]]:
    if not path.exists():
        return None, []
    data = json.loads(path.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    walk_candidate_records(data, records)
    return data, records


def read_csv_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def merge_by_candidate(*record_lists: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []

    for records in record_lists:
        for r in records:
            cid = get_any(r, ["candidateId", "candidate_id", "id", "caseId", "case_id"])
            if cid is None:
                continue
            cid = str(cid)
            if cid not in merged:
                merged[cid] = {"candidateId": cid}
                order.append(cid)
            for k, v in r.items():
                if v is not None and v != "":
                    merged[cid][k] = v

    return [merged[cid] for cid in order]


def infer_fixture_role(r: dict[str, Any]) -> str:
    """Map Week16 S3 fixture semantics to W17 layer-mix gate roles.

    Evidence anchor:
    - procedural_v0_0004 and procedural_v0_0010 are P1 paired regression fixtures.
    - procedural_v0_0007 is a P2 threshold-margin fixture.
    - all remaining procedural_v0 candidates are P4 numeric-margin controls.
    """
    cid = str(get_any(r, ["candidateId", "candidate_id", "id", "caseId", "case_id"], ""))

    if cid in {"procedural_v0_0004", "procedural_v0_0010"}:
        return "P1_PAIRED_REGRESSION_FIXTURE"
    if cid == "procedural_v0_0007":
        return "P2_THRESHOLD_MARGIN_FIXTURE"
    if cid.startswith("procedural_v0_"):
        return "P4_NUMERIC_MARGIN_CONTROL"

    all_text = " ".join(str(v) for v in r.values()).lower()
    if "paired" in all_text or "p1" in all_text:
        return "P1_PAIRED_REGRESSION_FIXTURE"
    if "threshold" in all_text or "near_miss" in all_text or "near-miss" in all_text or "p2" in all_text or "warn" in all_text:
        return "P2_THRESHOLD_MARGIN_FIXTURE"
    if "control" in all_text or "p4" in all_text or "pass" in all_text:
        return "P4_NUMERIC_MARGIN_CONTROL"
    return "HOLDOUT_UNVERIFIED_FIXTURE"

def infer_mix_eligibility(fixture_role: str) -> str:
    if fixture_role == "P1_PAIRED_REGRESSION_FIXTURE":
        return "BLOCK_AUTOMIX_REGRESSION_ONLY"
    if fixture_role == "P2_THRESHOLD_MARGIN_FIXTURE":
        return "MONITOR_ONLY_THRESHOLD_MARGIN"
    if fixture_role == "P4_NUMERIC_MARGIN_CONTROL":
        return "ELIGIBLE_CONTROL_ONLY"
    return "BLOCK_AUTOMIX_NEEDS_REVIEW"


def infer_safeguard(fixture_role: str) -> str:
    if fixture_role == "P1_PAIRED_REGRESSION_FIXTURE":
        return "preserve_original_fail_and_remediated_pass_pair_before_layer_mix"
    if fixture_role == "P2_THRESHOLD_MARGIN_FIXTURE":
        return "require_threshold_margin_monitoring_before_layer_mix"
    if fixture_role == "P4_NUMERIC_MARGIN_CONTROL":
        return "allow_as_numeric_margin_control_not_quality_claim"
    return "manual_review_required_before_layer_mix"


def infer_next_action(fixture_role: str) -> str:
    if fixture_role == "P1_PAIRED_REGRESSION_FIXTURE":
        return "exclude_from_automatic_mix_and_keep_for_regression_gate"
    if fixture_role == "P2_THRESHOLD_MARGIN_FIXTURE":
        return "exclude_from_automatic_mix_until_margin_policy_is_explicit"
    if fixture_role == "P4_NUMERIC_MARGIN_CONTROL":
        return "eligible_as_control_input_for_w17_layer_mix_v0"
    return "hold_until_evidence_gap_is_closed"


def count_by(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in records:
        v = str(r.get(key, "UNKNOWN"))
        out[v] = out.get(v, 0) + 1
    return dict(sorted(out.items()))


def sha256_if_exists(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    missing_required = [str(p) for p in [REGRESSION_JSON, FIXTURE_JSON] if not p.exists()]
    if missing_required:
        raise FileNotFoundError("Missing required W16 source artifact(s): " + ", ".join(missing_required))

    regression_data, regression_records_json = read_json_records(REGRESSION_JSON)
    fixture_data, fixture_records_json = read_json_records(FIXTURE_JSON)
    regression_records_csv = read_csv_records(REGRESSION_CSV)

    merged = merge_by_candidate(regression_records_json, fixture_records_json, regression_records_csv)

    if not merged:
        raise RuntimeError(
            "No candidate-like records extracted from W16 artifacts. "
            "Need schema adaptation instead of fabricating readiness."
        )

    readiness_records: list[dict[str, Any]] = []
    for r in merged:
        cid = str(get_any(r, ["candidateId", "candidate_id", "id", "caseId", "case_id"]))
        fixture_role = infer_fixture_role(r)
        mix_eligibility = infer_mix_eligibility(fixture_role)
        required_safeguard = infer_safeguard(fixture_role)
        next_action = infer_next_action(fixture_role)

        failure_bucket = str(get_any(r, ["failureBucket", "failure_bucket", "bucket"], "UNKNOWN"))
        severity = str(get_any(r, ["severity", "priority"], "UNKNOWN"))
        actionability = str(get_any(r, ["actionability", "actionable"], "UNKNOWN"))

        blocked_claims = [
            "final mix readiness",
            "semantic audio quality pass",
            "human review pass",
            "production mixer availability",
        ]

        if fixture_role != "P4_NUMERIC_MARGIN_CONTROL":
            blocked_claims.append("automatic layer mix eligibility")

        readiness_records.append({
            "candidateId": cid,
            "sourceFailureBucket": failure_bucket,
            "sourceSeverity": severity,
            "sourceActionability": actionability,
            "fixtureRole": fixture_role,
            "mixEligibility": mix_eligibility,
            "requiredSafeguard": required_safeguard,
            "nextAction": next_action,
            "blockedClaims": sorted(set(blocked_claims)),
            "sourceRecordKeys": sorted(r.keys()),
        })

    decision = "PASS_WEEK16_S3_TO_W17_LAYER_MIX_INPUT_READINESS"
    if any(r["fixtureRole"] == "HOLDOUT_UNVERIFIED_FIXTURE" for r in readiness_records):
        decision = "WARN_WEEK16_S3_TO_W17_LAYER_MIX_INPUT_READINESS_WITH_HOLDOUTS"

    payload = {
        "decision": decision,
        "generatedAtUtc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "contractVersion": "week16.s3_to_w17.layer_mix_input.v0",
        "sourceArtifacts": {
            "failureRegressionReportJson": str(REGRESSION_JSON),
            "failureRegressionReportCsv": str(REGRESSION_CSV),
            "failureFixtureRegistryJson": str(FIXTURE_JSON),
            "failureRegressionReportJsonSha256": sha256_if_exists(REGRESSION_JSON),
            "failureRegressionReportCsvSha256": sha256_if_exists(REGRESSION_CSV),
            "failureFixtureRegistryJsonSha256": sha256_if_exists(FIXTURE_JSON),
        },
        "candidateTotal": len(readiness_records),
        "mixEligibilityCounts": count_by(readiness_records, "mixEligibility"),
        "fixtureRoleCounts": count_by(readiness_records, "fixtureRole"),
        "blockedClaims": sorted(set(x for r in readiness_records for x in r["blockedClaims"])),
        "records": readiness_records,
        "nonClaims": [
            "real layer mixer not executed",
            "final mix readiness not claimed",
            "semantic audio quality pass not claimed",
            "human review pass not claimed",
        ],
    }

    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "candidateId",
            "sourceFailureBucket",
            "sourceSeverity",
            "sourceActionability",
            "fixtureRole",
            "mixEligibility",
            "requiredSafeguard",
            "nextAction",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in readiness_records:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    OUT_DOC.write_text(
        "# Week16 S3 to W17 Layer-Mix Input Readiness\n\n"
        "This artifact converts Week16 failure taxonomy evidence into a W17 layer-mix input gate.\n\n"
        "## Boundary\n\n"
        "- P1 paired regression fixtures are blocked from automatic layer mix.\n"
        "- P2 threshold-margin fixtures are monitor-only until margin policy is explicit.\n"
        "- P4 numeric-margin controls may be used as control inputs, not as semantic quality claims.\n"
        "- No real layer mixer is executed here.\n"
        "- No final mix readiness is claimed here.\n\n"
        f"## Decision\n\n`{decision}`\n\n"
        f"## Candidate total\n\n`{len(readiness_records)}`\n",
        encoding="utf-8",
    )

    print(json.dumps({
        "decision": payload["decision"],
        "candidateTotal": payload["candidateTotal"],
        "mixEligibilityCounts": payload["mixEligibilityCounts"],
        "fixtureRoleCounts": payload["fixtureRoleCounts"],
        "outJson": str(OUT_JSON),
        "outCsv": str(OUT_CSV),
        "outDoc": str(OUT_DOC),
    }, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())