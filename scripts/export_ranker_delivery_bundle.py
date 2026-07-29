#!/usr/bin/env python3
"""Export a status-consistent, checksummed preference-ranker delivery bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import shutil
import subprocess
from pathlib import Path
from typing import Any


BLOCKED = "DATA_BLOCKED"
MODEL_STATUSES = {"EXPLORATORY_ONLY", "CANDIDATE"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def git_head(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def artifact_ref(path: Path, required: bool = True) -> dict[str, Any]:
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return {
        "relativePath": path.name,
        "sha256": sha256(path),
        "sizeBytes": path.stat().st_size,
        "mediaType": media_type,
        "requiredForStatus": required,
    }


def validate_status(
    status: str,
    model_present: bool,
    oof_available: bool,
    recommendation_count: int,
) -> None:
    if status == BLOCKED:
        if model_present or oof_available or recommendation_count:
            raise ValueError("DATA_BLOCKED forbids model, OOF, and recommendations")
        return
    if status not in MODEL_STATUSES:
        raise ValueError(f"unsupported promotion status: {status}")
    if not model_present or not oof_available:
        raise ValueError(f"{status} requires model and real OOF artifacts")
    if status == "CANDIDATE" and recommendation_count < 1:
        raise ValueError("CANDIDATE requires at least one recommendation")


def export_bundle(args: argparse.Namespace) -> dict[str, Any]:
    root = Path.cwd().resolve()
    quality_gate = load_json(Path(args.quality_gate))
    promotion_gate = load_json(Path(args.promotion_gate))
    active_gate = load_json(Path(args.active_gate))
    model_dir = Path(args.model_dir).resolve()
    feature_snapshot = Path(args.feature_snapshot).resolve()
    output_dir = Path(args.output_dir).resolve()
    report_path = Path(args.report).resolve()

    status = promotion_gate["promotionStatus"]
    review_count = int(quality_gate["metrics"]["judgmentCount"])
    human_review_completed = quality_gate["status"] == "TRAINING_ELIGIBLE"
    model_source = model_dir / "ranker.json"
    oof_source = root / "reports/preference_ranker_oof_20260728.csv"
    recommendation_source = root / "reports/ranker_recommendations_20260729.csv"
    model_present = status in MODEL_STATUSES and model_source.is_file()
    oof_available = (
        status in MODEL_STATUSES
        and oof_source.is_file()
        and len(oof_source.read_text(encoding="utf-8").splitlines()) > 1
    )
    recommendation_count = 0
    if status in MODEL_STATUSES and recommendation_source.is_file():
        recommendation_count = max(
            0, len(recommendation_source.read_text(encoding="utf-8").splitlines()) - 1
        )
    validate_status(status, model_present, oof_available, recommendation_count)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    model_card_source = model_dir / "model_card.json"
    feature_schema_source = model_dir / "feature_schema.json"
    shutil.copy2(model_card_source, output_dir / "model-card.json")
    shutil.copy2(feature_schema_source, output_dir / "feature-schema.json")

    claim_boundary = {
        "schemaVersion": "ranker-claim-boundary/v1",
        "proxyOnly": True,
        "humanGateRequired": True,
        "autoFinalForbidden": True,
        "humanPairPreferenceOnly": True,
        "rankerRecommendationIsPublishDecision": False,
        "finalSelectedMutationCount": 0,
    }
    write_json(output_dir / "claim-boundary.json", claim_boundary)

    conditional: list[tuple[Path, str]] = []
    if status in MODEL_STATUSES:
        conditional.extend(
            [
                (model_source, "ranker.json"),
                (oof_source, "oof-predictions.csv"),
            ]
        )
        if recommendation_count:
            conditional.append((recommendation_source, "recommendations.csv"))
    for source, name in conditional:
        shutil.copy2(source, output_dir / name)

    payload_names = ["model-card.json", "feature-schema.json", "claim-boundary.json"]
    payload_names.extend(name for _, name in conditional)
    artifacts = [artifact_ref(output_dir / name) for name in sorted(payload_names)]

    feature_schema = load_json(feature_schema_source)
    training_dataset = root / "reports/preference_training_dataset_20260728.csv"
    manifest_without_digest = {
        "schemaVersion": "ranker-delivery-bundle/v1",
        "rankerName": "preference-ranker",
        "rankerVersion": "preference-ranker-v1-20260730",
        "promotionStatus": status,
        "modelPresent": model_present,
        "oofAvailable": oof_available,
        "recommendationCount": recommendation_count,
        "featureSchemaVersion": feature_schema.get(
            "schemaVersion", "preference-features-v1"
        ),
        "featureSnapshotDigest": sha256(feature_snapshot),
        "trainingDatasetDigest": sha256(training_dataset),
        "trainingCodeCommit": git_head(root),
        "sourceGitHead": git_head(root),
        "reviewSubmittedCount": review_count,
        "humanReviewCompleted": human_review_completed,
        "finalSelectedMutationCount": 0,
        "blockedReason": (
            promotion_gate.get("reason") if status == BLOCKED else None
        ),
        "activeLearningStatus": active_gate["status"],
        "artifacts": artifacts,
        "claimBoundary": {
            "proxyOnly": True,
            "humanGateRequired": True,
            "autoFinalForbidden": True,
        },
    }
    canonical = json.dumps(
        manifest_without_digest, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    manifest = dict(manifest_without_digest)
    manifest["bundleDigest"] = hashlib.sha256(canonical).hexdigest()
    write_json(output_dir / "manifest.json", manifest)

    checksummed_names = sorted(payload_names + ["manifest.json"])
    checksums = "".join(
        f"{sha256(output_dir / name)}  {name}\n" for name in checksummed_names
    )
    (output_dir / "checksums.sha256").write_text(checksums, encoding="utf-8")

    actual_members = sorted(path.name for path in output_dir.iterdir() if path.is_file())
    expected_members = sorted(checksummed_names + ["checksums.sha256"])
    report = {
        "schemaVersion": "ranker-delivery-report/v1",
        "status": "DELIVERY_READY",
        "promotionStatus": status,
        "bundleDigest": manifest["bundleDigest"],
        "bundleRelativePath": output_dir.relative_to(root).as_posix(),
        "artifactCount": len(artifacts),
        "directoryMembers": actual_members,
        "memberSetMatches": actual_members == expected_members,
        "checksumEntryCount": len(checksummed_names),
        "checksumVerifiedCount": sum(
            sha256(output_dir / name)
            == next(
                item["sha256"]
                for item in artifacts
                if item["relativePath"] == name
            )
            for name in payload_names
        )
        + 1,
        "modelPresent": model_present,
        "oofAvailable": oof_available,
        "recommendationCount": recommendation_count,
        "reviewSubmittedCount": review_count,
        "humanReviewCompleted": human_review_completed,
        "finalSelectedMutationCount": 0,
        "blockedReason": manifest["blockedReason"],
    }
    if not report["memberSetMatches"]:
        raise ValueError("delivery directory member set does not match manifest contract")
    write_json(report_path, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality-gate", required=True)
    parser.add_argument("--promotion-gate", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--feature-snapshot", required=True)
    parser.add_argument("--ablation", required=True)
    parser.add_argument("--active-gate", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(export_bundle(parse_args()), indent=2))
