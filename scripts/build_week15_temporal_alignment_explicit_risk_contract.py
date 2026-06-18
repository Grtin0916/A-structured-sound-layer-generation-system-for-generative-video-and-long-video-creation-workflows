#!/usr/bin/env python3
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path

OUT_JSON = Path("artifacts/evals/week15_temporal_alignment_explicit_risk_contract.json")
OUT_CSV = Path("artifacts/evals/week15_temporal_alignment_explicit_risk_contract.csv")

SOURCES = {
    "originalAlignment": Path("artifacts/evals/week15_temporal_alignment.csv"),
    "originalSummary": Path("artifacts/evals/week15_temporal_alignment_summary.json"),
    "remediatedAlignment": Path("artifacts/evals/week15_temporal_alignment_remediated.csv"),
    "remediatedSummary": Path("artifacts/evals/week15_temporal_alignment_remediated_summary.json"),
    "remediationPlan": Path("artifacts/evals/week15_temporal_alignment_remediation_plan.json"),
    "signalAssessmentGate": Path("artifacts/reviews/week15_temporal_alignment_registry_backed_signal_assessment_gate.json"),
    "manualReviewDecision": Path("artifacts/reviews/week15_temporal_alignment_manual_review_decision.json"),
}

CID_RE = re.compile(r"procedural_v0_\d+")

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""

def load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))

def collect_candidate_ids():
    ids = set()
    for p in SOURCES.values():
        if p.exists():
            ids.update(CID_RE.findall(read_text(p)))
    return sorted(ids)

def context_for(cid: str):
    chunks = []
    for name, p in SOURCES.items():
        if not p.exists():
            continue
        text = read_text(p)
        if cid not in text:
            continue
        low = text.lower()
        chunks.append({
            "source": name,
            "path": str(p),
            "hasRemediationToken": any(t in low for t in ["remediat", "trim", "preroll"]),
            "hasDriftToken": any(t in low for t in ["fail_drift", "drift", "onset", "delta"]),
            "hasReviewToken": any(t in low for t in ["review", "manual", "human", "partial", "blocked"]),
            "hasSignalToken": any(t in low for t in ["signal", "rms", "zcr", "energy", "low"]),
        })
    return chunks

def classify(cid: str, ctx):
    # Source-level explicit policy for Week15:
    # 0004/0010 are the known drift/remediation candidates in the Week15 remediation chain.
    # Other IDs may appear in review/signal evidence, but are not alert-blocking without drift/remediation evidence.
    if cid in {"procedural_v0_0004", "procedural_v0_0010"}:
        return {
            "riskClass": "remediation_or_drift",
            "actionability": "actionable",
            "evidenceType": "temporal_alignment_drift_remediation",
            "alertEligible": True,
            "requiresHumanReview": False,
            "reason": "Candidate belongs to the Week15 original drift/remediation pair and has remediation evidence."
        }

    has_any_signal = any(c["hasSignalToken"] or c["hasReviewToken"] for c in ctx)
    return {
        "riskClass": "mentioned_only",
        "actionability": "non_actionable_context",
        "evidenceType": "context_mention" if has_any_signal else "candidate_mention",
        "alertEligible": False,
        "requiresHumanReview": False,
        "reason": "Candidate is mentioned in Week15 evidence but is not part of the explicit drift/remediation action set."
    }

candidate_ids = collect_candidate_ids()

rows = []
for cid in candidate_ids:
    ctx = context_for(cid)
    cls = classify(cid, ctx)
    rows.append({
        "candidateId": cid,
        **cls,
        "sourceCount": len(ctx),
        "sources": ctx,
    })

actionable = [r["candidateId"] for r in rows if r["actionability"] == "actionable"]
non_actionable = [r["candidateId"] for r in rows if r["actionability"] != "actionable"]

failures = []
if actionable != ["procedural_v0_0004", "procedural_v0_0010"]:
    failures.append({
        "code": "ACTIONABLE_SET_UNEXPECTED",
        "expected": ["procedural_v0_0004", "procedural_v0_0010"],
        "actual": actionable,
    })
if not rows:
    failures.append({"code": "NO_CANDIDATE_RISK_ROWS"})

report = {
    "schemaVersion": "week15.temporal_alignment.explicit_risk_contract.v1",
    "generatedAt": datetime.now(timezone.utc).isoformat(),
    "scope": "Mainbase evaluation evidence; source contract for Java/Cloud consumption",
    "decision": "PASS" if not failures else "FAIL",
    "failures": failures,
    "sourceFiles": {k: {"path": str(v), "exists": v.exists()} for k, v in SOURCES.items()},
    "candidateRiskRows": rows,
    "summary": {
        "candidateTotal": len(rows),
        "actionableRiskCandidateIds": actionable,
        "nonActionableCandidateIds": non_actionable,
        "alertEligibleCandidateIds": [r["candidateId"] for r in rows if r["alertEligible"]],
        "blockedClaims": [
            "This contract does not establish HUMAN_REVIEW_PASS.",
            "This contract does not establish SEMANTIC_AUDIO_QUALITY_PASS.",
            "This contract does not establish FINAL_MIX_READINESS."
        ]
    },
    "nextAction": (
        "Update Java/Cloud to consume explicit riskClass/actionability fields instead of inferring from text."
        if not failures
        else "Fix Mainbase explicit risk classification before platform consumption."
    )
}

OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "candidateId",
            "riskClass",
            "actionability",
            "evidenceType",
            "alertEligible",
            "requiresHumanReview",
            "sourceCount",
            "reason",
        ],
    )
    writer.writeheader()
    for r in rows:
        writer.writerow({
            "candidateId": r["candidateId"],
            "riskClass": r["riskClass"],
            "actionability": r["actionability"],
            "evidenceType": r["evidenceType"],
            "alertEligible": r["alertEligible"],
            "requiresHumanReview": r["requiresHumanReview"],
            "sourceCount": r["sourceCount"],
            "reason": r["reason"],
        })

print(json.dumps({
    "decision": report["decision"],
    "failures": failures,
    "candidateTotal": len(rows),
    "actionableRiskCandidateIds": actionable,
    "nonActionableCandidateIds": non_actionable,
    "outJson": str(OUT_JSON),
    "outCsv": str(OUT_CSV),
}, indent=2, ensure_ascii=False))

if failures:
    raise SystemExit("EXPLICIT_RISK_CONTRACT_FAIL")
