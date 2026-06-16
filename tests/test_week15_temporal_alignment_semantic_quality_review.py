from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_week15_semantic_quality_review_packet_keeps_claim_boundaries():
    subprocess.run(
        [sys.executable, "scripts/build_week15_temporal_alignment_semantic_quality_review.py"],
        check=True,
    )

    path = Path("artifacts/reviews/week15_temporal_alignment_semantic_quality_review_v0.json")
    assert path.exists()

    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["schemaVersion"] == "week15.semantic-quality-review.v0"
    assert data["humanReviewStatus"] == "HUMAN_REVIEW_PARTIAL"
    assert data["auditionStatus"] == "NOT_PERFORMED"
    assert data["semanticQualityReviewStatus"] == "NOT_PERFORMED"
    assert data["finalMixReadiness"] == "NOT_CLAIMED"
    assert data["qualityGateLiteStatus"] in {
        "SEMANTIC_REVIEW_READY",
        "BLOCKED_MISSING_MANUAL_REVIEW_INPUT",
        "BLOCKED_NO_CANDIDATES",
    }

    blocked = " ".join(data["blockedClaims"])
    assert "Do not claim HUMAN_REVIEW_PASS" in blocked
    assert "Do not claim semantic audio quality PASS" in blocked
    assert "Do not claim final mix readiness" in blocked

    candidates = {c["candidateId"]: c for c in data["candidates"]}
    assert "procedural_v0_0004" in candidates
    assert "procedural_v0_0010" in candidates

    for cid in ["procedural_v0_0004", "procedural_v0_0010"]:
        c = candidates[cid]
        assert "riskFlags" in c
        assert c["riskFlags"]
        assert "original" in c["audioPaths"]
        assert "remediated" in c["audioPaths"]
        assert c["lightweightAudioMetrics"]["original"]["status"] == "OK"
        assert c["lightweightAudioMetrics"]["remediated"]["status"] == "OK"
        assert "riskFlags" in c["lightweightAudioMetrics"]["original"]
        assert "riskFlags" in c["lightweightAudioMetrics"]["remediated"]
