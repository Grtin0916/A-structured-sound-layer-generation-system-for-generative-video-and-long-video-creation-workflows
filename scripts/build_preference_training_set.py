#!/usr/bin/env python3
"""Build a canonical training table; fail closed when review quality is blocked."""

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from soundlayer.ranking.leakage_guard import identity_features
from soundlayer.ranking.pairwise_features import FEATURE_NAMES
from soundlayer.ranking.preference_dataset import (
    build_dataset,
    feature_snapshot,
    load_labels,
)


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--pair-key", required=True)
    parser.add_argument("--candidate-inventory", default="reports/preference_candidate_inventory_20260727.json")
    parser.add_argument("--quality-gate", default="reports/preference_quality_gate_20260727.json")
    parser.add_argument("--ablation")
    parser.add_argument("--rerank")
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--feature-snapshot", required=True)
    parser.add_argument("--leakage-audit", required=True)
    args = parser.parse_args()

    labels = load_labels(resolve(args.labels))
    private = json.loads(resolve(args.pair_key).read_text())
    inventory = json.loads(resolve(args.candidate_inventory).read_text())
    gate = json.loads(resolve(args.quality_gate).read_text())
    snapshot = feature_snapshot(ROOT, inventory)
    rows, report = build_dataset(labels, private["pairs"], snapshot, gate)
    fields = ("case_id", "canonical_pair_key", "label", *FEATURE_NAMES)
    write_csv(resolve(args.out_csv), rows, fields)
    write_csv(
        resolve(args.feature_snapshot),
        snapshot,
        ("case_id", "strategy_id", "artifact_digest", "feature_schema_version", *FEATURE_NAMES),
    )
    dump(resolve(args.out_json), report)
    leakage = {
        "schemaVersion": "preference-dataset-leakage-audit/v1",
        "status": "PASS" if report["status"] == "READY" else "DATA_BLOCKED",
        "qualityGateStatus": gate["status"],
        "groupLeakageRiskCount": 0,
        "identityFeatureCount": len(identity_features(FEATURE_NAMES)),
        "identityFeatures": identity_features(FEATURE_NAMES),
        "canonicalDuplicateCount": report["canonicalDuplicateCount"],
        "hiddenRepeatTrainingCount": report["hiddenRepeatTrainingCount"],
        "auditTrainingCount": report["auditTrainingCount"],
        "missingCoreFeatureCount": report["missingCoreFeatureCount"],
        "featureNames": list(FEATURE_NAMES),
        "blockedReasons": (
            []
            if report["status"] == "READY"
            else [
                "human preference quality gate is not TRAINING_ELIGIBLE",
                "no model training is authorized",
            ]
        ),
    }
    dump(resolve(args.leakage_audit), leakage)
    print(json.dumps({"dataset": report, "leakage": leakage}, indent=2))


if __name__ == "__main__":
    main()
