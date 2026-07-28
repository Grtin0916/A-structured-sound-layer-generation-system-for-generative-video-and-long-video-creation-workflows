#!/usr/bin/env python3
"""Train and export a grouped pairwise ranker, or emit an explicit DATA_BLOCKED model card."""

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from soundlayer.ranking.pairwise_features import FEATURE_NAMES
from soundlayer.ranking.pairwise_logistic import (
    fit_logistic,
    fit_scaler,
    leave_one_case_out,
    predict,
    reverse_augment,
    scale,
    symmetry_error,
)


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def load_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def metrics(rows):
    if not rows:
        return {}
    labels = [int(row["label"]) for row in rows]
    probabilities = [float(row["probability"]) for row in rows]
    predictions = [int(value >= 0.5) for value in probabilities]
    accuracy = sum(a == b for a, b in zip(labels, predictions)) / len(labels)
    recalls = []
    for label in (0, 1):
        indexes = [index for index, value in enumerate(labels) if value == label]
        if indexes:
            recalls.append(sum(predictions[index] == label for index in indexes) / len(indexes))
    eps = 1e-12
    return {
        "accuracy": accuracy,
        "balancedAccuracy": sum(recalls) / len(recalls) if recalls else None,
        "logLoss": -sum(
            y * math.log(max(eps, p)) + (1 - y) * math.log(max(eps, 1 - p))
            for y, p in zip(labels, probabilities)
        )
        / len(labels),
        "brierScore": sum((p - y) ** 2 for y, p in zip(labels, probabilities))
        / len(labels),
    }


def checksums(directory):
    rows = []
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.name != "checksums.sha256":
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            rows.append(f"{digest}  {path.name}")
    (directory / "checksums.sha256").write_text("\n".join(rows) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--group-column", default="case_id")
    parser.add_argument("--outer-cv", default="leave-one-group-out")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--oof-csv", required=True)
    parser.add_argument("--metrics-json", required=True)
    parser.add_argument("--promotion-gate", default="reports/preference_ranker_promotion_gate_20260728.json")
    parser.add_argument("--symmetry-audit", default="reports/preference_ranker_symmetry_audit_20260728.json")
    parser.add_argument("--seed", type=int, default=20260728)
    args = parser.parse_args()

    rows = load_csv(resolve(args.dataset))
    config = json.loads(resolve(args.config).read_text())
    output = resolve(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    environment = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "sklearnAvailable": importlib.util.find_spec("sklearn") is not None,
        "joblibAvailable": importlib.util.find_spec("joblib") is not None,
    }
    dump(output / "environment.json", environment)
    dump(
        output / "feature_schema.json",
        {
            "schemaVersion": "preference-features-v1",
            "featureNames": list(FEATURE_NAMES),
            "identityFeaturesAllowed": False,
        },
    )
    if len(rows) < config["minimum_independent_pairs"]:
        reason = (
            f"independent pair count {len(rows)} is below "
            f"{config['minimum_independent_pairs']}; human quality gate remains blocked"
        )
        result = {
            "schemaVersion": "preference-ranker-metrics/v1",
            "status": "DATA_BLOCKED",
            "promotionStatus": "DATA_BLOCKED",
            "independentPairCount": len(rows),
            "oofTestCaseCount": 0,
            "metrics": {},
            "blockedReasons": [reason],
            "rankerRecommendationCount": 0,
            "finalSelectedMutationCount": 0,
        }
        dump(resolve(args.metrics_json), result)
        dump(
            resolve(args.promotion_gate),
            {
                "schemaVersion": "preference-ranker-promotion/v1",
                "promotionStatus": "DATA_BLOCKED",
                "checks": {"qualityGate": False, "independentPairs": False},
                "reason": reason,
                "finalSelectedMutationCount": 0,
            },
        )
        dump(
            resolve(args.symmetry_audit),
            {
                "schemaVersion": "preference-ranker-symmetry/v1",
                "status": "NOT_RUN_DATA_BLOCKED",
                "meanSymmetryError": None,
                "maxSymmetryError": None,
                "violationCount": None,
            },
        )
        dump(
            output / "model_card.json",
            {
                "rankerVersion": "preference-ranker-v1",
                "algorithm": "L2 pairwise logistic regression",
                "promotionStatus": "DATA_BLOCKED",
                "trainingPerformed": False,
                "knownLimitations": [
                    reason,
                    "scikit-learn and joblib are unavailable in the current environment",
                ],
                "modelArtifactProduced": False,
                "portableContractProduced": False,
                "finalSelectedMutationCount": 0,
            },
        )
        oof_path = resolve(args.oof_csv)
        oof_path.parent.mkdir(parents=True, exist_ok=True)
        with oof_path.open("w", newline="", encoding="utf-8") as stream:
            csv.writer(stream, lineterminator="\n").writerow(
                ("case_id", "canonical_pair_key", "label", "probability", "fold_status")
            )
        checksums(output)
        print(json.dumps(result, indent=2))
        return

    oof = leave_one_case_out(rows, FEATURE_NAMES, args.seed)
    oof_path = resolve(args.oof_csv)
    oof_path.parent.mkdir(parents=True, exist_ok=True)
    with oof_path.open("w", newline="", encoding="utf-8") as stream:
        fields = ("case_id", "canonical_pair_key", "label", "probability", "fold_status")
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in oof)
    vectors = [[float(row[name]) for name in FEATURE_NAMES] for row in rows]
    labels = [int(row["label"]) for row in rows]
    means, scales = fit_scaler(vectors)
    scaled = [scale(vector, means, scales) for vector in vectors]
    augmented, augmented_labels = reverse_augment(scaled, labels)
    weights, intercept = fit_logistic(augmented, augmented_labels)
    symmetry = [symmetry_error(vector, weights, intercept) for vector in scaled]
    ranker = {
        "format": "linear-pairwise-ranker-v1",
        "featureNames": list(FEATURE_NAMES),
        "scalerMean": means,
        "scalerScale": scales,
        "coefficients": weights,
        "intercept": intercept,
        "decisionThreshold": 0.5,
        "featureSchemaVersion": "preference-features-v1",
    }
    dump(output / "ranker.json", ranker)
    result = {
        "schemaVersion": "preference-ranker-metrics/v1",
        "status": "TRAINED",
        "promotionStatus": "EXPLORATORY_ONLY",
        "independentPairCount": len(rows),
        "oofTestCaseCount": len({row["case_id"] for row in oof}),
        "metrics": metrics(oof),
        "rankerRecommendationCount": 0,
        "finalSelectedMutationCount": 0,
    }
    dump(resolve(args.metrics_json), result)
    dump(
        resolve(args.symmetry_audit),
        {
            "status": "PASS" if max(symmetry, default=0.0) <= 1e-6 else "FAIL",
            "meanSymmetryError": sum(symmetry) / len(symmetry),
            "maxSymmetryError": max(symmetry),
            "violationCount": sum(value > 1e-6 for value in symmetry),
        },
    )
    dump(
        output / "model_card.json",
        {
            "rankerVersion": "preference-ranker-v1",
            "algorithm": "dependency-free L2 pairwise logistic regression",
            "promotionStatus": "EXPLORATORY_ONLY",
            "trainingPerformed": True,
            "modelArtifactProduced": False,
            "portableContractProduced": True,
            "knownLimitations": ["single reviewer", "small case count"],
            "finalSelectedMutationCount": 0,
        },
    )
    dump(
        resolve(args.promotion_gate),
        {
            "promotionStatus": "EXPLORATORY_ONLY",
            "checks": {"qualityGate": True, "independentPairs": True},
            "finalSelectedMutationCount": 0,
        },
    )
    checksums(output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
