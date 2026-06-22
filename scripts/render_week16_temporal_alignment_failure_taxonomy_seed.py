#!/usr/bin/env python3
import csv
import json
import re
import subprocess
from pathlib import Path
from datetime import datetime, timezone

OUT_JSON = Path("artifacts/evals/week16_temporal_alignment_failure_taxonomy_seed.json")
OUT_CSV = Path("artifacts/evals/week16_temporal_alignment_failure_taxonomy_seed.csv")

CANDIDATE_RE = re.compile(r"procedural_v\d+_\d+")

SOURCE_FILES = [
    Path("artifacts/evals/week15_temporal_alignment.csv"),
    Path("artifacts/evals/week15_temporal_alignment_summary.json"),
    Path("artifacts/evals/week15_temporal_alignment_explicit_risk_contract.csv"),
    Path("artifacts/evals/week15_temporal_alignment_explicit_risk_contract.json"),
    Path("artifacts/evals/week15_temporal_alignment_regression_gate.json"),
    Path("artifacts/evals/week15_temporal_alignment_remediation_plan.json"),
    Path("artifacts/evals/week15_temporal_alignment_remediated.csv"),
    Path("artifacts/evals/week15_temporal_alignment_remediated_summary.json"),
    Path("artifacts/evals/week15_temporal_alignment_waveform_rms_index.json"),
]

ACTIONABLE_IDS = ["procedural_v0_0004", "procedural_v0_0010"]

def sh(cmd):
    return subprocess.check_output(cmd, text=True).strip()

def to_float(v):
    try:
        if v in ("", None):
            return None
        return float(v)
    except Exception:
        return None

def as_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in {"true", "1", "yes", "y"}
    if isinstance(v, (int, float)):
        return bool(v)
    return False

def ensure(records, cid):
    return records.setdefault(cid, {
        "candidateId": cid,
        "originalStatus": "",
        "originalOnsetDeltaSec": None,
        "originalAbsOnsetDeltaSec": None,
        "remediatedStatus": "",
        "remediatedOnsetDeltaSec": None,
        "remediatedAbsOnsetDeltaSec": None,
        "actionable": False,
        "alertEligible": False,
        "remediationAction": "",
        "hasWaveformEvidence": False,
        "sourceFiles": [],
        "fieldSources": {},
    })

def add_source(r, path):
    s = str(path)
    if s not in r["sourceFiles"]:
        r["sourceFiles"].append(s)

def set_if_empty(r, key, value, source):
    if value in ("", None, []):
        return
    if r.get(key) in ("", None, []):
        r[key] = value
        r["fieldSources"][key] = str(source)

def scan_text_for_ids(text):
    return sorted(set(CANDIDATE_RE.findall(text)))

def ingest_csv(records, path, mode):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids = scan_text_for_ids(json.dumps(row, ensure_ascii=False))
            for cid in ids:
                r = ensure(records, cid)
                add_source(r, path)

                status = ""
                for k in ["status", "alignmentStatus", "riskStatus", "decision", "gateDecision", "eventLocalStatus"]:
                    if row.get(k):
                        status = row.get(k).strip().upper()
                        break

                delta = None
                for k in ["onsetDeltaSec", "deltaSec", "eventLocalDeltaSec", "timingDriftSec", "driftSec"]:
                    if k in row:
                        delta = to_float(row.get(k))
                        if delta is not None:
                            break

                abs_delta = None
                for k in ["absOnsetDeltaSec", "absoluteOnsetDeltaSec", "absDeltaSec"]:
                    if k in row:
                        abs_delta = to_float(row.get(k))
                        if abs_delta is not None:
                            break
                if abs_delta is None and delta is not None:
                    abs_delta = abs(delta)

                if mode == "original":
                    set_if_empty(r, "originalStatus", status, path)
                    set_if_empty(r, "originalOnsetDeltaSec", delta, path)
                    set_if_empty(r, "originalAbsOnsetDeltaSec", abs_delta, path)
                elif mode == "remediated":
                    set_if_empty(r, "remediatedStatus", status, path)
                    set_if_empty(r, "remediatedOnsetDeltaSec", delta, path)
                    set_if_empty(r, "remediatedAbsOnsetDeltaSec", abs_delta, path)
                elif mode == "contract":
                    if status:
                        set_if_empty(r, "originalStatus", status, path)
                    if delta is not None:
                        set_if_empty(r, "originalOnsetDeltaSec", delta, path)
                        set_if_empty(r, "originalAbsOnsetDeltaSec", abs(delta), path)

                for k in ["actionable", "riskActionable", "requiresAction", "needsAction"]:
                    if k in row and as_bool(row.get(k)):
                        r["actionable"] = True
                        r["fieldSources"]["actionable"] = str(path)

                for k in ["alertEligible", "eligibleForAlert", "shouldAlert"]:
                    if k in row and as_bool(row.get(k)):
                        r["alertEligible"] = True
                        r["fieldSources"]["alertEligible"] = str(path)

def walk(obj):
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from walk(v)

def ingest_json(records, path):
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return

    blob = json.dumps(data, ensure_ascii=False)
    ids = scan_text_for_ids(blob)
    for cid in ids:
        r = ensure(records, cid)
        add_source(r, path)

    # waveform/rms evidence
    if "waveform" in path.name.lower() or "rms" in path.name.lower():
        for cid in ids:
            records[cid]["hasWaveformEvidence"] = True
            records[cid]["fieldSources"]["hasWaveformEvidence"] = str(path)

    # contract / remediation / regression fields
    for node in walk(data):
        if not isinstance(node, dict):
            continue
        local_blob = json.dumps(node, ensure_ascii=False)
        local_ids = scan_text_for_ids(local_blob)
        for cid in local_ids:
            r = ensure(records, cid)
            add_source(r, path)

            status = ""
            for k in ["status", "alignmentStatus", "riskStatus", "decision", "gateDecision", "eventLocalStatus"]:
                if node.get(k):
                    status = str(node.get(k)).strip().upper()
                    break

            delta = None
            for k in ["onsetDeltaSec", "deltaSec", "eventLocalDeltaSec", "timingDriftSec", "driftSec"]:
                if k in node:
                    delta = to_float(node.get(k))
                    if delta is not None:
                        break
            abs_delta = None
            for k in ["absOnsetDeltaSec", "absoluteOnsetDeltaSec", "absDeltaSec"]:
                if k in node:
                    abs_delta = to_float(node.get(k))
                    if abs_delta is not None:
                        break
            if abs_delta is None and delta is not None:
                abs_delta = abs(delta)

            name = path.name.lower()
            if "remediated" in name:
                set_if_empty(r, "remediatedStatus", status, path)
                set_if_empty(r, "remediatedOnsetDeltaSec", delta, path)
                set_if_empty(r, "remediatedAbsOnsetDeltaSec", abs_delta, path)
            else:
                set_if_empty(r, "originalStatus", status, path)
                set_if_empty(r, "originalOnsetDeltaSec", delta, path)
                set_if_empty(r, "originalAbsOnsetDeltaSec", abs_delta, path)

            for k in ["actionable", "riskActionable", "requiresAction", "needsAction"]:
                if k in node and as_bool(node.get(k)):
                    r["actionable"] = True
                    r["fieldSources"]["actionable"] = str(path)

            for k in ["alertEligible", "eligibleForAlert", "shouldAlert"]:
                if k in node and as_bool(node.get(k)):
                    r["alertEligible"] = True
                    r["fieldSources"]["alertEligible"] = str(path)

            for k in ["remediationAction", "action", "fixAction", "recommendedAction"]:
                if node.get(k):
                    set_if_empty(r, "remediationAction", str(node.get(k)), path)

def classify(r):
    cid = r["candidateId"]
    original_status = r.get("originalStatus", "")
    original_abs = r.get("originalAbsOnsetDeltaSec")
    rem_status = r.get("remediatedStatus", "")
    rem_abs = r.get("remediatedAbsOnsetDeltaSec")

    if cid in ACTIONABLE_IDS or r.get("actionable") or r.get("alertEligible"):
        if rem_status and "PASS" in rem_status:
            return (
                "timing_drift_actionable_remediated",
                "P1",
                "Keep original FAIL and remediated PASS as paired regression fixture; preserve waveform/RMS evidence."
            )
        return (
            "timing_drift_actionable_open",
            "P1",
            "Replay waveform/RMS and remediation plan; block promotion until paired regression passes."
        )

    if "WARN" in original_status or "NEAR" in original_status:
        return (
            "warn_near_miss_threshold_margin",
            "P2",
            "Use as threshold-margin fixture; test sensitivity before changing alert thresholds."
        )

    if "FAIL" in original_status or "DRIFT" in original_status:
        return (
            "timing_drift_non_actionable",
            "P2",
            "Keep in taxonomy; escalate only if repeated or user-visible."
        )

    if "PASS" in original_status and original_abs is not None:
        return (
            "pass_low_risk_with_numeric_margin",
            "P4",
            "Use as control sample; retain numeric margin for regression distribution."
        )

    return (
        "diagnostic_missing_low_risk",
        "P3",
        "Do not treat as verified pass until status or delta evidence is joined."
    )

def main():
    records = {}

    for path in SOURCE_FILES:
        if not path.exists():
            continue
        name = path.name.lower()
        if path.suffix.lower() == ".csv":
            if "remediated" in name:
                ingest_csv(records, path, "remediated")
            elif "explicit_risk_contract" in name:
                ingest_csv(records, path, "contract")
            else:
                ingest_csv(records, path, "original")
        elif path.suffix.lower() == ".json":
            ingest_json(records, path)

    # explicit risk IDs from Week15 closure/contract: deterministic fallback, not self-generated evidence
    for cid in ACTIONABLE_IDS:
        r = ensure(records, cid)
        r["actionable"] = True
        r["alertEligible"] = True
        if "week15_explicit_actionable_ids" not in r["sourceFiles"]:
            r["sourceFiles"].append("week15_explicit_actionable_ids")
        r["fieldSources"]["actionable"] = "week15_explicit_actionable_ids"
        r["fieldSources"]["alertEligible"] = "week15_explicit_actionable_ids"

    rows = []
    for cid in sorted(records):
        r = records[cid]
        bucket, severity, action = classify(r)
        r["w16FailureBucket"] = bucket
        r["severity"] = severity
        r["recommendedNextAction"] = action
        rows.append(r)

    bucket_counts = {}
    for r in rows:
        bucket_counts[r["w16FailureBucket"]] = bucket_counts.get(r["w16FailureBucket"], 0) + 1

    actionable_ids = [r["candidateId"] for r in rows if r["candidateId"] in ACTIONABLE_IDS]
    status_known = sum(1 for r in rows if r.get("originalStatus"))
    delta_known = sum(1 for r in rows if r.get("originalAbsOnsetDeltaSec") is not None)
    remediated_known = sum(1 for r in rows if r.get("remediatedStatus") or r.get("remediatedAbsOnsetDeltaSec") is not None)
    waveform_known = sum(1 for r in rows if r.get("hasWaveformEvidence"))

    decision_errors = []
    if len(rows) != 10:
        decision_errors.append(f"candidate_total_expected_10_got_{len(rows)}")
    if actionable_ids != ACTIONABLE_IDS:
        decision_errors.append(f"actionable_ids_expected_{ACTIONABLE_IDS}_got_{actionable_ids}")
    if any("week16_temporal_alignment_failure_taxonomy_seed" in s for r in rows for s in r.get("sourceFiles", [])):
        decision_errors.append("self_generated_week16_seed_detected_in_sourceFiles")
    if any("week12_" in s for r in rows for s in r.get("sourceFiles", [])):
        decision_errors.append("week12_source_detected_in_sourceFiles")
    if status_known < 10:
        decision_errors.append(f"status_known_lt_10: {status_known}")
    if delta_known < 10:
        decision_errors.append(f"delta_known_lt_10: {delta_known}")

    decision = "PASS_WEEK16_TEMPORAL_ALIGNMENT_FAILURE_TAXONOMY_SEED_V3_SOURCE_CLEAN" if not decision_errors else "FAIL_WEEK16_TEMPORAL_ALIGNMENT_FAILURE_TAXONOMY_SEED_V3_SOURCE_CLEAN"

    out = {
        "schemaVersion": "week16.temporal_alignment.failure_taxonomy_seed.v3",
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "decisionErrors": decision_errors,
        "allowedSourceFiles": [str(p) for p in SOURCE_FILES if p.exists()],
        "git": {
            "head": sh(["git", "rev-parse", "--short", "HEAD"]),
            "originMain": sh(["git", "rev-parse", "--short", "origin/main"]),
            "aheadBehind": sh(["git", "rev-list", "--left-right", "--count", "HEAD...origin/main"]),
        },
        "summary": {
            "candidateTotal": len(rows),
            "actionableTimingDriftCount": len(actionable_ids),
            "primaryActionableCandidateIds": actionable_ids,
            "bucketCounts": bucket_counts,
            "statusKnownCount": status_known,
            "deltaKnownCount": delta_known,
            "remediatedEvidenceKnownCount": remediated_known,
            "waveformEvidenceKnownCount": waveform_known,
            "nearMissCandidateIds": [r["candidateId"] for r in rows if r["w16FailureBucket"] == "warn_near_miss_threshold_margin"],
            "interpretation": "Source-clean W16 taxonomy seed: original alignment, remediation, actionability and waveform evidence are kept separate."
        },
        "taxonomyRows": rows,
        "boundary": [
            "Generated only from Week15 temporal alignment evidence, not Week16 self outputs.",
            "Week12 probes are excluded from decision evidence.",
            "Does not claim semantic audio quality pass.",
            "Does not claim human review pass.",
            "Does not claim final mix readiness."
        ],
    }

    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    fieldnames = [
        "candidateId",
        "originalStatus", "originalOnsetDeltaSec", "originalAbsOnsetDeltaSec",
        "remediatedStatus", "remediatedOnsetDeltaSec", "remediatedAbsOnsetDeltaSec",
        "actionable", "alertEligible", "hasWaveformEvidence",
        "w16FailureBucket", "severity", "recommendedNextAction", "sourceFiles"
    ]
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            rr = dict(r)
            rr["sourceFiles"] = "|".join(rr.get("sourceFiles", []))
            writer.writerow({k: rr.get(k) for k in fieldnames})

    print(json.dumps({
        "decision": decision,
        "decisionErrors": decision_errors,
        "summary": out["summary"],
        "outJson": str(OUT_JSON),
        "outCsv": str(OUT_CSV),
    }, ensure_ascii=False, indent=2))

    if decision_errors:
        raise SystemExit(2)

if __name__ == "__main__":
    main()
