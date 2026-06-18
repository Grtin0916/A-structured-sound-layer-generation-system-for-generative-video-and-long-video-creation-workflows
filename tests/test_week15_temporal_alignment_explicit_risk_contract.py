import json
from pathlib import Path

CONTRACT = Path("artifacts/evals/week15_temporal_alignment_explicit_risk_contract.json")

def test_explicit_risk_contract_exists_and_passes():
    assert CONTRACT.exists()
    data = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert data["decision"] == "PASS"
    assert data["failures"] == []

def test_actionable_risk_set_is_explicit_and_precise():
    data = json.loads(CONTRACT.read_text(encoding="utf-8"))
    summary = data["summary"]
    assert summary["actionableRiskCandidateIds"] == [
        "procedural_v0_0004",
        "procedural_v0_0010",
    ]
    assert summary["alertEligibleCandidateIds"] == [
        "procedural_v0_0004",
        "procedural_v0_0010",
    ]

def test_all_candidate_rows_have_platform_contract_fields():
    data = json.loads(CONTRACT.read_text(encoding="utf-8"))
    rows = data["candidateRiskRows"]
    assert rows
    required = {
        "candidateId",
        "riskClass",
        "actionability",
        "evidenceType",
        "alertEligible",
        "requiresHumanReview",
        "reason",
    }
    for row in rows:
        assert required.issubset(row.keys())
        assert row["actionability"] in {"actionable", "non_actionable_context"}
        assert isinstance(row["alertEligible"], bool)
        assert isinstance(row["requiresHumanReview"], bool)

def test_non_actionable_mentions_do_not_alert():
    data = json.loads(CONTRACT.read_text(encoding="utf-8"))
    rows = {r["candidateId"]: r for r in data["candidateRiskRows"]}
    for cid in ["procedural_v0_0002", "procedural_v0_0003", "procedural_v0_0007"]:
        assert rows[cid]["riskClass"] == "mentioned_only"
        assert rows[cid]["actionability"] == "non_actionable_context"
        assert rows[cid]["alertEligible"] is False
