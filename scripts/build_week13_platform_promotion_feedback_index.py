#!/usr/bin/env python3
"""
Build Week13 platform promotion feedback index in Mainbase.

This script consumes Cloud's platform promotion gate and records whether
Mainbase Candidate Audio Bank V1 has been accepted by the platform chain.

Boundary:
- local demo promotion feedback only
- does not claim semantic audio quality
- does not claim human audition pass
- does not claim final mix readiness
- does not claim live Grafana import
- does not claim production Kubernetes Job
- does not claim S3/MinIO/CSI/cloud object storage
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"missing required json: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mainbase-demo-index",
        type=Path,
        default=Path("artifacts/manifests/week13_candidate_bank_demo_index.json"),
    )
    ap.add_argument(
        "--cloud-promotion-gate",
        type=Path,
        default=Path("../ai-job-platform-cloud/loadtest/reports/week13_candidate_bank_platform_promotion_gate.json"),
    )
    ap.add_argument(
        "--cloud-ci-check-log",
        type=Path,
        default=None,
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/manifests/week13_platform_promotion_feedback_index.json"),
    )
    args = ap.parse_args()

    demo = read_json(args.mainbase_demo_index)
    promotion = read_json(args.cloud_promotion_gate)

    positive = promotion.get("positivePath", {})
    negative = promotion.get("negativePath", {})
    hard = promotion.get("hardChecks", {})

    ci_log_exists = args.cloud_ci_check_log.exists() if args.cloud_ci_check_log else False

    checks = {
        "mainbaseDemoIndexPass": demo.get("status") == "PASS",
        "cloudPromotionGatePass": promotion.get("status") == "PASS",
        "cloudPromotionDecisionReady": promotion.get("promotionDecision") == "PROMOTE_TO_WEEK13_DEMO_READY",
        "candidateCountIsTen": positive.get("candidateCount") == 10,
        "workerSuccessCountIsTen": positive.get("workerSuccessCount") == 10,
        "drilldownReadyCountIsTen": positive.get("drilldownReadyCount") == 10,
        "fullClipLikeCountIsFive": positive.get("fullClipLikeCount") == 5,
        "eventLocalLikeCountIsFive": positive.get("eventLocalLikeCount") == 5,
        "negativeRegressionTargetExpected": negative.get("targetCandidate") == "procedural_v0_0002",
        "negativeRegressionDetectedFail": negative.get("failureSummaryStatus") == "FAIL",
        "negativeRegressionLocatedBlocker": any(
            "procedural_v0_0002:worker_smoke_not_success:FAILED" in str(x)
            for x in negative.get("failureSummaryBlockers", [])
        ),
        "cloudGateNoBlockers": promotion.get("blockers") == [],
        "allCloudHardChecksTrue": all(v is True for v in hard.values()),
    }

    status = "PASS" if all(checks.values()) else "FAIL"

    payload = {
        "schemaVersion": "week13.mainbase_platform_promotion_feedback_index.v1",
        "generatedAt": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "scope": "mainbase-platform-promotion-feedback-only",
        "sourceMainbaseDemoIndex": str(args.mainbase_demo_index),
        "sourceCloudPromotionGate": str(args.cloud_promotion_gate),
        "sourceCloudCiCheckLog": str(args.cloud_ci_check_log) if args.cloud_ci_check_log else None,
        "sourceCloudCiCheckLogExists": ci_log_exists,
        "platformPromotionDecision": promotion.get("promotionDecision"),
        "positivePath": positive,
        "negativePath": negative,
        "checks": checks,
        "blockers": [] if status == "PASS" else [k for k, v in checks.items() if not v],
        "boundary": [
            "does_not_claim_semantic_audio_quality",
            "does_not_claim_human_audition_pass",
            "does_not_claim_final_mix_readiness",
            "does_not_claim_live_grafana_import",
            "does_not_claim_production_slo",
            "does_not_claim_production_kubernetes_job",
            "does_not_claim_s3_minio_csi_or_cloud_object_storage",
        ],
        "feedbackDecision": (
            "PASS: Mainbase Candidate Audio Bank V1 has platform promotion feedback for Week13 local demo readiness."
            if status == "PASS"
            else "FAIL: Mainbase cannot accept platform promotion feedback; inspect checks and blockers."
        ),
        "nextRecommendedStep": (
            "Use this feedback index as the source-side anchor for Friday stage gate or engineering closure."
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps({
        "out": str(args.out),
        "status": status,
        "platformPromotionDecision": payload["platformPromotionDecision"],
        "positivePath": positive,
        "negativePath": negative,
        "failedChecks": payload["blockers"],
        "sourceCloudCiCheckLogExists": ci_log_exists,
    }, indent=2, ensure_ascii=False))

    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())