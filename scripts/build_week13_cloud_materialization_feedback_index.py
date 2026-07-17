#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_CLOUD = os.environ.get("CLOUD", str(Path.home() / "work/grt_work/ai-job-platform-cloud"))
DEFAULT_JAVA = os.environ.get("JAVA_REPO", str(Path.home() / "work/grt_work/media-task-platform-java"))

DEFAULT_MAINBASE_PLACEMENT = "artifacts/manifests/week13_mix_placement_manifest.json"
DEFAULT_MAINBASE_PREVIEW = "artifacts/audio_mix/week13_mix_preview_manifest.json"

DEFAULT_CLOUD_MATERIALIZED = "loadtest/reports/week13_materialized_audio_artifact_manifest.json"
DEFAULT_CLOUD_MOUNT = "loadtest/reports/week13_mount_read_contract.json"
DEFAULT_CLOUD_POD_READ = "loadtest/reports/week13_pod_audio_read_simulation_report.json"

DEFAULT_JAVA_READINESS = "artifacts/manifests/week13_java_materialized_audio_registry_readiness_report.json"
DEFAULT_JAVA_API = "artifacts/manifests/week13_materialized_readiness_api_contract_report.json"

DEFAULT_OUT = "artifacts/manifests/week13_cloud_materialization_feedback_index.json"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def iter_dicts(x: Any):
    if isinstance(x, dict):
        yield x
        for v in x.values():
            yield from iter_dicts(v)
    elif isinstance(x, list):
        for v in x:
            yield from iter_dicts(v)


def cid_of(d: dict[str, Any]) -> str | None:
    for k in ["candidateId", "candidate_id", "id", "audioCandidateId", "audio_candidate_id"]:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def extract_records(obj: Any) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for d in iter_dicts(obj):
        cid = cid_of(d)
        if not cid:
            continue
        useful = any(k in d for k in [
            "assetTimeMode", "expectedStartSec", "placementRequired",
            "materializedStorageStatus", "podPath", "localObjectPath",
            "audioReadable", "sampleRateHz", "durationSec",
            "objectKey", "sha256", "localSha256", "sourceSha256",
            "sourceType"
        ])
        if useful:
            out[cid] = dict(d)
    return out


def pick(d: dict[str, Any] | None, keys: list[str], default=None):
    if not d:
        return default
    for k in keys:
        if k in d:
            return d.get(k)
    return default


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cloud-root", default=DEFAULT_CLOUD)
    ap.add_argument("--java-root", default=DEFAULT_JAVA)
    ap.add_argument("--placement", default=DEFAULT_MAINBASE_PLACEMENT)
    ap.add_argument("--preview", default=DEFAULT_MAINBASE_PREVIEW)
    ap.add_argument("--cloud-materialized", default=DEFAULT_CLOUD_MATERIALIZED)
    ap.add_argument("--cloud-mount", default=DEFAULT_CLOUD_MOUNT)
    ap.add_argument("--cloud-pod-read", default=DEFAULT_CLOUD_POD_READ)
    ap.add_argument("--java-readiness", default=DEFAULT_JAVA_READINESS)
    ap.add_argument("--java-api", default=DEFAULT_JAVA_API)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    main_root = Path.cwd().resolve()
    cloud_root = Path(args.cloud_root).expanduser().resolve()
    java_root = Path(args.java_root).expanduser().resolve()

    placement = load_json(main_root / args.placement)
    preview = load_json(main_root / args.preview)
    cloud_materialized = load_json(cloud_root / args.cloud_materialized)
    cloud_mount = load_json(cloud_root / args.cloud_mount)
    cloud_pod_read = load_json(cloud_root / args.cloud_pod_read)
    java_readiness = load_json(java_root / args.java_readiness)
    java_api = load_json(java_root / args.java_api)

    blockers: list[str] = []

    sources = [
        ("mainbase_placement", placement),
        ("mainbase_preview", preview),
        ("cloud_materialized", cloud_materialized),
        ("cloud_mount", cloud_mount),
        ("cloud_pod_read", cloud_pod_read),
        ("java_readiness", java_readiness),
        ("java_api", java_api),
    ]

    for name, obj in sources:
        if isinstance(obj, dict) and obj.get("status") != "PASS":
            blockers.append(f"{name}_status={obj.get('status')}")

    placement_records = extract_records(placement)
    preview_records = extract_records(preview)
    cloud_mat_records = extract_records(cloud_materialized)
    cloud_mount_records = extract_records(cloud_mount)
    cloud_pod_records = extract_records(cloud_pod_read)
    java_ready_records = extract_records(java_readiness)

    ids = set(placement_records) | set(preview_records)
    if not ids:
        blockers.append("no_mainbase_candidate_records_found")

    missing_cloud_pod = sorted(ids - set(cloud_pod_records))
    missing_java_ready = sorted(ids - set(java_ready_records))

    if missing_cloud_pod:
        blockers.append(f"missing_cloud_pod_read={missing_cloud_pod}")
    if missing_java_ready:
        blockers.append(f"missing_java_readiness={missing_java_ready}")

    records = []
    for cid in sorted(ids):
        pr = placement_records.get(cid)
        pv = preview_records.get(cid)
        cm = cloud_mat_records.get(cid)
        mt = cloud_mount_records.get(cid)
        cp = cloud_pod_records.get(cid)
        jr = java_ready_records.get(cid)

        materialized = bool(pick(cm, ["materialized"], False))
        cloud_audio_readable = bool(pick(cp, ["audioReadable"], False))
        cloud_hash_ok = bool(pick(cp, ["sha256Verified"], False))
        cloud_size_ok = bool(pick(cp, ["sizeVerified"], False))
        pod_mapped = bool(pick(cp, ["podPathMapped"], False))
        java_ready = pick(jr, ["materializedStorageStatus"]) == "READY"

        feedback_status = "READY_FOR_PLATFORM_CONSUMPTION" if (
            materialized and cloud_audio_readable and cloud_hash_ok and cloud_size_ok and pod_mapped and java_ready
        ) else "NOT_READY"

        if feedback_status != "READY_FOR_PLATFORM_CONSUMPTION":
            blockers.append(f"candidate_not_ready:{cid}")

        records.append({
            "candidateId": cid,
            "feedbackStatus": feedback_status,
            "sourceType": pick(jr, ["sourceType"], pick(pv, ["sourceType"])),
            "assetTimeMode": pick(jr, ["assetTimeMode"], pick(pr, ["assetTimeMode"], pick(cp, ["assetTimeMode"]))),
            "expectedStartSec": pick(jr, ["expectedStartSec"], pick(pr, ["expectedStartSec"], pick(cp, ["expectedStartSec"]))),
            "placementRequired": pick(jr, ["placementRequired"], pick(pr, ["placementRequired"], pick(cp, ["placementRequired"]))),
            "objectKey": pick(cp, ["objectKey"], pick(cm, ["objectKey"])),
            "podPath": pick(jr, ["podPath"], pick(cp, ["podPath"], pick(mt, ["podPath"]))),
            "localObjectPath": pick(jr, ["localObjectPath"], pick(cp, ["localObjectPath"], pick(cm, ["localObjectPath"]))),
            "sampleRateHz": pick(jr, ["sampleRateHz"], pick(cp, ["sampleRateHz"])),
            "channels": pick(jr, ["channels"], pick(cp, ["channels"])),
            "durationSec": pick(jr, ["durationSec"], pick(cp, ["durationSec"])),
            "sha256": pick(jr, ["sha256"], pick(cp, ["sha256"], pick(cm, ["localSha256"]))),
            "sizeBytes": pick(jr, ["sizeBytes"], pick(cp, ["sizeBytes"], pick(cm, ["localSizeBytes"]))),
            "cloudMaterialized": materialized,
            "cloudAudioReadable": cloud_audio_readable,
            "cloudSha256Verified": cloud_hash_ok,
            "cloudSizeVerified": cloud_size_ok,
            "cloudPodPathMapped": pod_mapped,
            "javaMaterializedStorageStatus": pick(jr, ["materializedStorageStatus"]),
        })

    ready_count = sum(1 for r in records if r["feedbackStatus"] == "READY_FOR_PLATFORM_CONSUMPTION")

    mode_counts: dict[str, int] = {}
    for r in records:
        mode = r.get("assetTimeMode") or "UNKNOWN"
        mode_counts[mode] = mode_counts.get(mode, 0) + 1

    api_endpoint = java_api.get("endpoint")
    api_test_class = java_api.get("testClass")

    status = "PASS" if len(records) == 10 and ready_count == 10 and not blockers else "FAIL"

    report = {
        "status": status,
        "scope": "week13_mainbase_cloud_java_materialization_feedback_index_v1",
        "generatedAt": dt.datetime.now(dt.timezone.utc).isoformat(),
        "candidateCount": len(records),
        "readyForPlatformConsumptionCount": ready_count,
        "assetTimeModeCounts": mode_counts,
        "javaApiEndpoint": api_endpoint,
        "javaApiTestClass": api_test_class,
        "sourceFiles": {
            "mainbasePlacement": str(main_root / args.placement),
            "mainbasePreview": str(main_root / args.preview),
            "cloudMaterialized": str(cloud_root / args.cloud_materialized),
            "cloudMount": str(cloud_root / args.cloud_mount),
            "cloudPodRead": str(cloud_root / args.cloud_pod_read),
            "javaReadiness": str(java_root / args.java_readiness),
            "javaApiContract": str(java_root / args.java_api),
        },
        "blockers": blockers,
        "boundary": [
            "feedback index for Mainbase candidate lifecycle",
            "does not rebuild or regenerate audio",
            "does not claim semantic audio quality",
            "does not claim final mix readiness",
            "confirms platform-consumable storage/readiness for current procedural baseline candidates",
        ],
        "records": records,
    }

    write_json(main_root / args.out, report)

    print(json.dumps({
        "report": str(main_root / args.out),
        "status": status,
        "candidateCount": len(records),
        "readyForPlatformConsumptionCount": ready_count,
        "assetTimeModeCounts": mode_counts,
        "javaApiEndpoint": api_endpoint,
        "blockers": blockers,
    }, ensure_ascii=False, indent=2))

    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
