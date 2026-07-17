#!/usr/bin/env python3
"""
Build Week12 audio candidate platform feedback index.

Purpose:
- Close the Mainbase -> Java -> Cloud feedback loop.
- Record that Mainbase enriched audio candidates were consumed by Java API and Cloud runtime index.
- Do not claim human audition passed, semantic audio quality passed, final mix readiness, or production storage.

Outputs:
- artifacts/manifests/week12_audio_candidate_platform_feedback_index.json
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(".").resolve()
JAVA = Path(os.environ.get("JAVA", str(Path.home() / "work" / "grt_work" / "media-task-platform-java"))).resolve()
CLOUD = Path(os.environ.get("CLOUD", str(Path.home() / "work" / "grt_work" / "ai-job-platform-cloud"))).resolve()

MAINBASE_QUEUE = ROOT / "artifacts/evals/week12_audio_audition_review_queue_v0.json"
JAVA_IMPORT_SUMMARY = JAVA / "artifacts/runtime/week12_audio_candidate_api_import_summary.json"
JAVA_HTTP_SUMMARY = JAVA / "artifacts/runtime/week12_audio_candidate_api_http_it_summary.json"
JAVA_HTTP_BODY = JAVA / "artifacts/runtime/week12_audio_candidate_api_http_it_body.json"
CLOUD_RUNTIME_INDEX = CLOUD / "loadtest/reports/week12_audio_candidate_runtime_index.json"
CLOUD_DASHBOARD = CLOUD / "observability/grafana/dashboards/week12_audio_candidate_dashboard.json"

OUT_INDEX = ROOT / "artifacts/manifests/week12_audio_candidate_platform_feedback_index.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"ERROR: missing required file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def require(obj: Dict[str, Any], key: str, expected: Any, label: str) -> None:
    actual = obj.get(key)
    if actual != expected:
        raise SystemExit(f"ERROR: {label}.{key}: expected={expected!r}, actual={actual!r}")


def main() -> int:
    mainbase_queue = read_json(MAINBASE_QUEUE)
    java_import = read_json(JAVA_IMPORT_SUMMARY)
    java_summary = read_json(JAVA_HTTP_SUMMARY)
    java_body = read_json(JAVA_HTTP_BODY)
    cloud_index = read_json(CLOUD_RUNTIME_INDEX)
    cloud_dashboard = read_json(CLOUD_DASHBOARD)

    require(mainbase_queue, "status", "PASS", "mainbase_queue")
    require(mainbase_queue, "candidateCount", 10, "mainbase_queue")
    require(mainbase_queue, "audioProbeOkCount", 10, "mainbase_queue")
    require(mainbase_queue, "semanticFidelityClaimedAny", False, "mainbase_queue")
    require(mainbase_queue, "mixReadyClaimedAny", False, "mainbase_queue")

    require(java_import, "status", "PASS", "java_import")
    require(java_import, "candidateCount", 10, "java_import")
    require(java_import, "audioProbeOkCount", 10, "java_import")
    require(java_import, "semanticFidelityClaimedAny", False, "java_import")
    require(java_import, "mixReadyClaimedAny", False, "java_import")

    require(java_summary, "status", "PASS", "java_summary")
    require(java_summary, "httpCode", 200, "java_summary")
    require(java_summary, "candidateCount", 10, "java_summary")
    require(java_summary, "audioProbeOkCount", 10, "java_summary")
    require(java_summary, "semanticFidelityClaimedAny", False, "java_summary")
    require(java_summary, "mixReadyClaimedAny", False, "java_summary")

    require(java_body, "status", "PASS", "java_body")
    require(java_body, "qualityGateStatus", "HUMAN_AUDITION_REQUIRED", "java_body")
    require(java_body, "candidateCount", 10, "java_body")
    require(java_body, "audioProbeOkCount", 10, "java_body")
    require(java_body, "audioProbeFailedCount", 0, "java_body")
    require(java_body, "durationMissingCount", 0, "java_body")
    require(java_body, "sampleRateMissingCount", 0, "java_body")
    require(java_body, "eventIdMissingCount", 0, "java_body")
    require(java_body, "semanticFidelityClaimedAny", False, "java_body")
    require(java_body, "mixReadyClaimedAny", False, "java_body")

    require(cloud_index, "status", "PASS", "cloud_index")
    require(cloud_index, "candidateCount", 10, "cloud_index")
    require(cloud_index, "audioProbeOkCount", 10, "cloud_index")
    require(cloud_index, "audioProbeFailedCount", 0, "cloud_index")
    require(cloud_index, "qualityGateStatus", "HUMAN_AUDITION_REQUIRED", "cloud_index")
    require(cloud_index, "semanticFidelityClaimedAny", False, "cloud_index")
    require(cloud_index, "mixReadyClaimedAny", False, "cloud_index")

    candidates: List[Dict[str, Any]] = java_body.get("candidates") or []
    if len(candidates) != 10:
        raise SystemExit(f"ERROR: expected 10 Java API candidates, got {len(candidates)}")

    required_candidate_fields = [
        "candidateId",
        "caseId",
        "sceneId",
        "eventId",
        "eventLabel",
        "layer",
        "candidateUri",
        "durationSec",
        "sampleRateHz",
        "channels",
        "sampleWidthBytes",
        "rmsDbfs",
        "peakDbfs",
        "formatOk",
        "reviewStatus",
        "failureTags",
    ]

    missing_records = []
    for item in candidates:
        missing = [k for k in required_candidate_fields if item.get(k) in (None, "")]
        if missing:
            missing_records.append({
                "candidateId": item.get("candidateId"),
                "missing": missing,
            })

    blockers = []
    if missing_records:
        blockers.append("CANDIDATE_FIELD_MISSING")
    if cloud_index.get("blockers"):
        blockers.append("CLOUD_BLOCKERS_NOT_EMPTY")
    if java_body.get("qualityGateStatus") != "HUMAN_AUDITION_REQUIRED":
        blockers.append("QUALITY_GATE_SHOULD_REMAIN_HUMAN_AUDITION_REQUIRED")

    dashboard_panels = cloud_dashboard.get("panels") or []
    if len(dashboard_panels) < 4:
        blockers.append("DASHBOARD_PANEL_COUNT_TOO_LOW")

    first = candidates[0]
    index = {
        "schemaVersion": "week12.mainbase.audio_candidate_platform_feedback_index.v0",
        "generatedAt": utc_now(),
        "status": "PASS" if not blockers else "FAIL",
        "closedLoop": {
            "mainbaseProducedEnrichedReviewQueue": True,
            "javaImportedReviewQueue": True,
            "javaExposedAudioCandidateApi": True,
            "javaVerifiedByRandomPortHttpIT": True,
            "cloudConsumedJavaApiEvidence": True,
            "cloudBuiltRuntimeIndex": True,
            "cloudBuiltDashboardStub": True,
        },
        "evidence": {
            "mainbaseReviewQueueUri": str(MAINBASE_QUEUE),
            "javaImportSummaryUri": str(JAVA_IMPORT_SUMMARY),
            "javaHttpSummaryUri": str(JAVA_HTTP_SUMMARY),
            "javaHttpBodyUri": str(JAVA_HTTP_BODY),
            "cloudRuntimeIndexUri": str(CLOUD_RUNTIME_INDEX),
            "cloudDashboardUri": str(CLOUD_DASHBOARD),
        },
        "counts": {
            "candidateCount": java_body.get("candidateCount"),
            "audioProbeOkCount": java_body.get("audioProbeOkCount"),
            "audioProbeFailedCount": java_body.get("audioProbeFailedCount"),
            "durationMissingCount": java_body.get("durationMissingCount"),
            "sampleRateMissingCount": java_body.get("sampleRateMissingCount"),
            "eventIdMissingCount": java_body.get("eventIdMissingCount"),
            "formatFailedCount": java_body.get("formatFailedCount"),
            "dashboardPanelCount": len(dashboard_panels),
        },
        "platformGate": {
            "qualityGateStatus": java_body.get("qualityGateStatus"),
            "semanticFidelityClaimedAny": java_body.get("semanticFidelityClaimedAny"),
            "mixReadyClaimedAny": java_body.get("mixReadyClaimedAny"),
            "cloudBlockers": cloud_index.get("blockers"),
            "mainbaseBlockers": mainbase_queue.get("blockers"),
        },
        "cloudStats": {
            "layerCounts": cloud_index.get("layerCounts"),
            "durationSec": cloud_index.get("durationSec"),
            "rmsDbfs": cloud_index.get("rmsDbfs"),
            "peakDbfs": cloud_index.get("peakDbfs"),
        },
        "firstCandidateRoundTrip": {
            "candidateId": first.get("candidateId"),
            "caseId": first.get("caseId"),
            "sceneId": first.get("sceneId"),
            "eventId": first.get("eventId"),
            "eventLabel": first.get("eventLabel"),
            "layer": first.get("layer"),
            "candidateUri": first.get("candidateUri"),
            "durationSec": first.get("durationSec"),
            "sampleRateHz": first.get("sampleRateHz"),
            "channels": first.get("channels"),
            "sampleWidthBytes": first.get("sampleWidthBytes"),
            "rmsDbfs": first.get("rmsDbfs"),
            "peakDbfs": first.get("peakDbfs"),
            "formatOk": first.get("formatOk"),
            "reviewStatus": first.get("reviewStatus"),
            "failureTags": first.get("failureTags"),
        },
        "remainingGaps": [
            "human_audition_not_performed",
            "semantic_audio_quality_not_verified",
            "expected_event_timing_window_not_bound",
            "final_mix_not_ready",
            "production_object_storage_not_verified",
            "real_prometheus_datasource_panel_not_verified",
        ],
        "nextAction": {
            "preferred": "build human audition decision artifact or bind expected event timing windows before claiming alignment",
            "avoid": [
                "do not claim semantic pass from procedural fallback",
                "do not claim final mix readiness",
                "do not keep adding API wrappers without resolving expected timing windows",
            ],
        },
        "doesNotClaim": [
            "semantic_audio_quality_passed",
            "human_audition_passed",
            "final_mix_readiness",
            "production_asset_storage",
            "production_slo_or_alerting",
        ],
        "missingRecords": missing_records,
        "blockers": blockers,
    }

    OUT_INDEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_INDEX.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": index["status"],
        "candidateCount": index["counts"]["candidateCount"],
        "audioProbeOkCount": index["counts"]["audioProbeOkCount"],
        "dashboardPanelCount": index["counts"]["dashboardPanelCount"],
        "qualityGateStatus": index["platformGate"]["qualityGateStatus"],
        "semanticFidelityClaimedAny": index["platformGate"]["semanticFidelityClaimedAny"],
        "mixReadyClaimedAny": index["platformGate"]["mixReadyClaimedAny"],
        "blockers": index["blockers"],
        "output": str(OUT_INDEX),
        "remainingGaps": index["remainingGaps"],
    }, ensure_ascii=False, indent=2))

    return 0 if index["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())