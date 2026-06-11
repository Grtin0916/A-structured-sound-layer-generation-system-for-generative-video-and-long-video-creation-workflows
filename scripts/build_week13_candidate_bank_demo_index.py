#!/usr/bin/env python3
"""
Build Week13 candidate bank demo index.

Purpose:
- Merge Mainbase placement evidence, Java readiness evidence, and Cloud worker smoke evidence.
- Produce a single demo-ready index for the Candidate Audio Bank chain.
- This is not a semantic quality claim, not a production Kubernetes Job claim, and not a durable registry claim.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def read_json(path: Path, required: bool = True) -> dict[str, Any]:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"missing required json: {path}")
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return sum(1 for _ in reader)


def pick_first_int(obj: dict[str, Any], keys: list[str], default: int = 0) -> int:
    for key in keys:
        val = obj.get(key)
        if isinstance(val, int):
            return val
        if isinstance(val, float) and val.is_integer():
            return int(val)
    return default


def pick_status(obj: dict[str, Any]) -> str:
    val = obj.get("status")
    return val if isinstance(val, str) else "UNKNOWN"


def pick_blockers(obj: dict[str, Any]) -> list[Any]:
    val = obj.get("blockers")
    return val if isinstance(val, list) else []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mainbase", type=Path, default=Path("."))
    ap.add_argument("--java", type=Path, default=Path("../media-task-platform-java"))
    ap.add_argument("--cloud", type=Path, default=Path("../ai-job-platform-cloud"))
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/manifests/week13_candidate_bank_demo_index.json"),
    )
    args = ap.parse_args()

    mainbase = args.mainbase.resolve()
    java = args.java.resolve()
    cloud = args.cloud.resolve()

    placement_manifest_path = mainbase / "artifacts/manifests/week13_mix_placement_manifest.json"
    placement_table_path = mainbase / "artifacts/evals/week13_mix_global_placement_table.csv"
    mix_preview_path = mainbase / "artifacts/audio_mix/week13_mix_preview_manifest.json"
    feedback_path = mainbase / "artifacts/manifests/week13_cloud_materialization_feedback_index.json"

    java_registry_report_path = java / "artifacts/manifests/week13_java_audio_artifact_registry_contract_report.json"
    java_materialized_report_path = java / "artifacts/manifests/week13_java_materialized_audio_registry_readiness_report.json"
    java_api_report_path = java / "artifacts/manifests/week13_materialized_readiness_api_contract_report.json"

    cloud_materialized_path = cloud / "loadtest/reports/week13_materialized_audio_artifact_manifest.json"
    cloud_mount_path = cloud / "loadtest/reports/week13_mount_read_contract.json"
    cloud_worker_input_path = cloud / "loadtest/reports/week13_ready_candidate_worker_input_manifest.json"
    cloud_worker_smoke_path = cloud / "loadtest/reports/week13_local_audio_worker_smoke_result.json"

    placement = read_json(placement_manifest_path)
    feedback = read_json(feedback_path)
    java_registry = read_json(java_registry_report_path)
    java_materialized = read_json(java_materialized_report_path, required=False)
    java_api = read_json(java_api_report_path, required=False)
    cloud_materialized = read_json(cloud_materialized_path)
    cloud_mount = read_json(cloud_mount_path)
    cloud_worker_input = read_json(cloud_worker_input_path)
    cloud_worker_smoke = read_json(cloud_worker_smoke_path)

    placement_count = pick_first_int(placement, ["candidateCount", "candidate_count"])
    placement_table_rows = count_csv_rows(placement_table_path)
    worker_success_count = pick_first_int(cloud_worker_smoke, ["workerSuccessCount", "successCount"])
    worker_ready_count = pick_first_int(cloud_worker_input, ["workerReadyCount", "readyCount"])
    materialized_count = pick_first_int(cloud_materialized, ["materializedCount", "candidateCount"])
    mount_readable_count = pick_first_int(cloud_mount, ["readableCount", "candidateCount"])
    feedback_ready_count = pick_first_int(
        feedback,
        ["readyForPlatformConsumptionCount", "candidateCount"],
    )

    statuses = {
        "mainbasePlacement": pick_status(placement),
        "mainbaseCloudFeedback": pick_status(feedback),
        "javaRegistryContract": pick_status(java_registry),
        "javaMaterializedReadiness": pick_status(java_materialized) if java_materialized else "OPTIONAL_NOT_FOUND",
        "javaReadinessApiContract": pick_status(java_api) if java_api else "OPTIONAL_NOT_FOUND",
        "cloudMaterializedArtifacts": pick_status(cloud_materialized),
        "cloudMountReadContract": pick_status(cloud_mount),
        "cloudWorkerInput": pick_status(cloud_worker_input),
        "cloudWorkerSmoke": pick_status(cloud_worker_smoke),
    }

    blockers: list[str] = []
    for name, obj in [
        ("mainbasePlacement", placement),
        ("mainbaseCloudFeedback", feedback),
        ("javaRegistryContract", java_registry),
        ("javaMaterializedReadiness", java_materialized),
        ("javaReadinessApiContract", java_api),
        ("cloudMaterializedArtifacts", cloud_materialized),
        ("cloudMountReadContract", cloud_mount),
        ("cloudWorkerInput", cloud_worker_input),
        ("cloudWorkerSmoke", cloud_worker_smoke),
    ]:
        for b in pick_blockers(obj):
            blockers.append(f"{name}:{b}")

    hard_checks = {
        "placementStatusPass": statuses["mainbasePlacement"] == "PASS",
        "feedbackStatusPass": statuses["mainbaseCloudFeedback"] == "PASS",
        "javaRegistryStatusPass": statuses["javaRegistryContract"] == "PASS",
        "cloudMaterializedStatusPass": statuses["cloudMaterializedArtifacts"] == "PASS",
        "cloudMountStatusPass": statuses["cloudMountReadContract"] == "PASS",
        "cloudWorkerInputStatusPass": statuses["cloudWorkerInput"] == "PASS",
        "cloudWorkerSmokeStatusPass": statuses["cloudWorkerSmoke"] == "PASS",
        "candidateCountIsTen": placement_count == 10,
        "workerSuccessCountIsTen": worker_success_count == 10,
        "noBlockers": len(blockers) == 0,
    }

    status = "PASS" if all(hard_checks.values()) else "FAIL"

    index = {
        "schemaVersion": "week13.candidate_bank_demo_index.v1",
        "generatedAt": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "demoName": "Week13 Candidate Audio Bank worker-consumable demo",
        "scope": "local-demo-index-only",
        "boundary": [
            "does_not_claim_semantic_audio_quality",
            "does_not_claim_human_audition_pass",
            "does_not_claim_final_mix_readiness",
            "does_not_claim_production_kubernetes_job",
            "does_not_claim_s3_minio_csi_or_cloud_object_storage",
            "does_not_claim_durable_java_registry",
        ],
        "counts": {
            "candidateCount": placement_count,
            "placementTableRows": placement_table_rows,
            "materializedCount": materialized_count,
            "mountReadableCount": mount_readable_count,
            "workerReadyCount": worker_ready_count,
            "workerSuccessCount": worker_success_count,
            "feedbackReadyForPlatformConsumptionCount": feedback_ready_count,
        },
        "statuses": statuses,
        "hardChecks": hard_checks,
        "blockers": blockers,
        "chain": [
            {
                "step": "mainbase_mix_placement",
                "evidence": str(placement_manifest_path.relative_to(mainbase)),
                "status": statuses["mainbasePlacement"],
            },
            {
                "step": "mainbase_global_placement_table",
                "evidence": str(placement_table_path.relative_to(mainbase)),
                "rowCount": placement_table_rows,
            },
            {
                "step": "cloud_materialized_audio",
                "evidence": str(cloud_materialized_path),
                "status": statuses["cloudMaterializedArtifacts"],
            },
            {
                "step": "cloud_mount_read_contract",
                "evidence": str(cloud_mount_path),
                "status": statuses["cloudMountReadContract"],
            },
            {
                "step": "java_artifact_registry_contract",
                "evidence": str(java_registry_report_path),
                "status": statuses["javaRegistryContract"],
            },
            {
                "step": "cloud_worker_input",
                "evidence": str(cloud_worker_input_path),
                "status": statuses["cloudWorkerInput"],
            },
            {
                "step": "cloud_worker_smoke",
                "evidence": str(cloud_worker_smoke_path),
                "status": statuses["cloudWorkerSmoke"],
            },
            {
                "step": "mainbase_cloud_materialization_feedback",
                "evidence": str(feedback_path.relative_to(mainbase)),
                "status": statuses["mainbaseCloudFeedback"],
            },
        ],
        "humanReadableConclusion": (
            "PASS: 10 candidate audio artifacts have placement evidence, local materialization evidence, "
            "mount-read evidence, Java registry contract evidence, worker input evidence, and local worker smoke evidence."
            if status == "PASS"
            else "FAIL: demo chain is incomplete; inspect hardChecks and blockers."
        ),
        "nextRecommendedStep": (
            "Expose or consume this demo index from Java/Cloud dashboard only after keeping the boundary local-demo-index-only."
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps({
        "out": str(args.out),
        "status": status,
        "counts": index["counts"],
        "failedChecks": [k for k, v in hard_checks.items() if not v],
        "blockers": blockers,
    }, indent=2, ensure_ascii=False))

    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())