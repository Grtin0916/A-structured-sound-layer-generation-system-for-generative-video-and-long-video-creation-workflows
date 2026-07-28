"""Canonicalize human pair labels and build case-grouped feature differences."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from .leakage_guard import assert_feature_safe
from .pairwise_features import FEATURE_NAMES, difference, extract_features


def canonical_pair_key(case_id, digest_a, digest_b):
    left, right = sorted((digest_a, digest_b))
    return f"{case_id}:{left}:{right}"


def load_labels(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def hydrate_labels(labels, private_pairs):
    truth = {row["pair_id"]: row for row in private_pairs}
    return [{**truth.get(row.get("pair_id", ""), {}), **row} for row in labels]


def submitted(value):
    return str(value).strip().lower() in {"1", "true", "yes"}


def feature_snapshot(root: Path, inventory: dict):
    rows = []
    for candidate in inventory["candidates"]:
        dss = root / "cases" / candidate["source_case_id"] / "director_sound_script.yaml"
        features = extract_features(root / candidate["artifact_path"], dss, candidate)
        rows.append(
            {
                "case_id": candidate["case_id"],
                "strategy_id": candidate["strategy_id"],
                "artifact_digest": candidate["artifact_digest"],
                "feature_schema_version": "preference-features-v1",
                **features,
            }
        )
    return rows


def build_dataset(labels, private_pairs, snapshot, quality_gate):
    assert_feature_safe(FEATURE_NAMES)
    labels = hydrate_labels(labels, private_pairs)
    feature_map = {
        (row["case_id"], row["artifact_digest"]): row for row in snapshot
    }
    exclusion = {
        "not_submitted": 0,
        "hidden_repeat": 0,
        "audit": 0,
        "tie_or_unjudgeable": 0,
        "canonical_duplicate": 0,
        "missing_feature": 0,
        "same_digest": 0,
    }
    rows = []
    seen = set()
    for label in labels:
        if not submitted(label.get("submitted")):
            exclusion["not_submitted"] += 1
            continue
        if str(label.get("is_hidden_repeat", "")).lower() == "true":
            exclusion["hidden_repeat"] += 1
            continue
        if str(label.get("is_audit_pair", "")).lower() == "true":
            exclusion["audit"] += 1
            continue
        preference = label.get("overall_preference", "").upper()
        if preference not in {"LEFT", "RIGHT"}:
            exclusion["tie_or_unjudgeable"] += 1
            continue
        if label["left_digest"] == label["right_digest"]:
            exclusion["same_digest"] += 1
            continue
        key = canonical_pair_key(
            label["case_id"], label["left_digest"], label["right_digest"]
        )
        if key in seen:
            exclusion["canonical_duplicate"] += 1
            continue
        seen.add(key)
        left = feature_map.get((label["case_id"], label["left_digest"]))
        right = feature_map.get((label["case_id"], label["right_digest"]))
        if left is None or right is None:
            exclusion["missing_feature"] += 1
            continue
        # Orient each unordered pair by digest so reversed UI presentation cannot
        # change X or y.
        if label["left_digest"] < label["right_digest"]:
            first, second = left, right
            y = 1 if preference == "LEFT" else 0
        else:
            first, second = right, left
            y = 1 if preference == "RIGHT" else 0
        rows.append(
            {
                "case_id": label["case_id"],
                "canonical_pair_key": key,
                "label": y,
                **difference(first, second),
            }
        )
    training_allowed = quality_gate.get("status") == "TRAINING_ELIGIBLE"
    if not training_allowed:
        rows = []
    digest = hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return rows, {
        "schemaVersion": "preference-training-dataset/v1",
        "status": "READY" if training_allowed and len(rows) >= 30 else "DATA_BLOCKED",
        "trainingAllowedByQualityGate": training_allowed,
        "independentPairCount": len(rows),
        "caseCount": len({row["case_id"] for row in rows}),
        "canonicalDuplicateCount": exclusion["canonical_duplicate"],
        "hiddenRepeatTrainingCount": 0,
        "auditTrainingCount": 0,
        "missingCoreFeatureCount": exclusion["missing_feature"],
        "datasetDigest": "sha256:" + digest,
        "exclusionCounts": exclusion,
    }
