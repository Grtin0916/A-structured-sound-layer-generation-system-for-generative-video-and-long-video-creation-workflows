#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_short_head(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
        text=True,
    ).strip()


def git_remote(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "remote", "get-url", "origin"],
        text=True,
    ).strip()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    mainbase = Path.home() / "work" / "audio_engineering_repo_skeleton_v1"
    java = Path.home() / "work" / "media-task-platform-java"
    cloud = Path.home() / "work" / "ai-job-platform-cloud"

    blueprint_manifest = mainbase / "artifacts/manifests/week12_blueprint_v1_manifest.json"
    timeline_jsonl = mainbase / "artifacts/manifests/week12_event_timeline.jsonl"
    timeline_csv = mainbase / "artifacts/manifests/week12_event_timeline.csv"
    contact_sheet = mainbase / "artifacts/visuals/week12_event_timeline_contact_sheet.png"

    cloud_index_rel = Path("loadtest/reports/week12_asset_blueprint_runtime_index.json")
    cloud_index = cloud / cloud_index_rel

    java_summary_rel = Path("artifacts/logs/week12_asset_blueprint_http_runtime_smoke_summary.json")
    java_summary = java / java_summary_rel

    required_files = [
        blueprint_manifest,
        timeline_jsonl,
        timeline_csv,
        contact_sheet,
        cloud_index,
        java_summary,
    ]
    missing_files = [str(p) for p in required_files if not p.exists()]
    if missing_files:
        raise SystemExit("MISSING_REQUIRED_FILES=" + json.dumps(missing_files, ensure_ascii=False))

    cloud_data = load_json(cloud_index)
    java_data = load_json(java_summary)

    consumed = cloud_data.get("consumedAssetBlueprint") or {}
    java_body = java_data.get("body") or {}

    expected_blueprint_uri = "artifacts/manifests/week12_blueprint_v1_manifest.json"
    expected_timeline_uri = "artifacts/manifests/week12_event_timeline.jsonl"

    checks = {
        "cloudRuntimePass": cloud_data.get("status") == "PASS",
        "cloudBlockersEmpty": cloud_data.get("blockers") == [],
        "javaRuntimePass": java_data.get("status") == "PASS",
        "javaHttp200": java_data.get("httpCode") == "200",
        "cloudJavaEvidenceCommitMatchesJavaHead": (
            (cloud_data.get("javaRuntimeEvidence") or {}).get("commit") == git_short_head(java)
        ),
        "blueprintArtifactLinked": consumed.get("blueprintManifestLinked") is True,
        "timelineArtifactLinked": consumed.get("timelineArtifactLinked") is True,
        "qualityGatePassed": consumed.get("qualityGatePassed") is True,
        "blueprintUriPointsToMainbaseManifest": expected_blueprint_uri in str(consumed.get("blueprintArtifactUri", "")),
        "timelineUriPointsToMainbaseTimeline": consumed.get("timelineArtifactUri") == expected_timeline_uri,
        "javaAndCloudTaskIdMatch": consumed.get("taskId") == java_body.get("taskId"),
        "noLocalAbsolutePathInCloudIndex": "/home/GRT/" not in json.dumps(cloud_data, ensure_ascii=False),
    }

    status = "PASS" if all(checks.values()) else "BLOCKED"
    blockers = [k for k, v in checks.items() if not v]

    feedback = {
        "schemaVersion": "week12.blueprint-runtime-feedback-index.v1",
        "generatedAt": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": status,
        "businessChain": [
            "mainbase.blueprint_v1_handoff",
            "java.asset_blueprint_runtime_endpoint",
            "cloud.asset_blueprint_runtime_index",
            "mainbase.runtime_feedback"
        ],
        "mainbaseEvidence": {
            "repo": git_remote(mainbase),
            "commit": git_short_head(mainbase),
            "blueprintManifestPath": "artifacts/manifests/week12_blueprint_v1_manifest.json",
            "blueprintManifestSha256": sha256_file(blueprint_manifest),
            "timelineJsonlPath": "artifacts/manifests/week12_event_timeline.jsonl",
            "timelineJsonlSha256": sha256_file(timeline_jsonl),
            "timelineCsvPath": "artifacts/manifests/week12_event_timeline.csv",
            "timelineCsvSha256": sha256_file(timeline_csv),
            "contactSheetPath": "artifacts/visuals/week12_event_timeline_contact_sheet.png",
            "contactSheetSha256": sha256_file(contact_sheet),
        },
        "javaRuntimeEvidence": {
            "repo": git_remote(java),
            "commit": git_short_head(java),
            "summaryPath": str(java_summary_rel),
            "summarySha256": sha256_file(java_summary),
            "status": java_data.get("status"),
            "httpCode": java_data.get("httpCode"),
            "selectedEndpoint": java_data.get("selectedEndpoint"),
            "redisHealthDisabledForSmoke": java_data.get("redisHealthDisabledForSmoke"),
        },
        "cloudRuntimeEvidence": {
            "repo": git_remote(cloud),
            "commit": git_short_head(cloud),
            "runtimeIndexPath": str(cloud_index_rel),
            "runtimeIndexSha256": sha256_file(cloud_index),
            "status": cloud_data.get("status"),
            "javaRuntimeStatus": cloud_data.get("javaRuntimeStatus"),
            "javaRuntimeHttpCode": cloud_data.get("javaRuntimeHttpCode"),
            "blockers": cloud_data.get("blockers"),
        },
        "consumedAssetBlueprint": consumed,
        "checks": checks,
        "blockers": blockers,
        "doesNotClaim": [
            "audio waveform generation",
            "candidate audio bank",
            "production deployment",
            "production SLO",
            "full Redis-backed readiness",
            "durable object storage"
        ]
    }

    out = mainbase / "artifacts/manifests/week12_blueprint_runtime_feedback_index.json"
    out.write_text(json.dumps(feedback, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(feedback, indent=2, ensure_ascii=False))
    return 0 if status == "PASS" else 4


if __name__ == "__main__":
    raise SystemExit(main())