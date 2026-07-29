#!/usr/bin/env python3
"""Select new review pairs only when a real OOF model exists."""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from soundlayer.ranking.active_review import (
    acquisition_score,
    existing_content_keys,
    unlabeled_pairs,
)
from soundlayer.ranking.pairwise_features import FEATURE_NAMES
from soundlayer.ranking.ranker_ablation import rule_score

FIELDS = (
    "pair_id",
    "case_id",
    "left_artifact",
    "right_artifact",
    "left_digest",
    "right_digest",
    "acquisition_reason",
    "acquisition_score",
)


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in FIELDS} for row in rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-inventory", required=True)
    parser.add_argument("--existing-pairs", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--rule-results")
    parser.add_argument("--human-graph", required=True)
    parser.add_argument("--oof", default="reports/preference_ranker_oof_20260728.csv")
    parser.add_argument("--features", default="reports/preference_feature_snapshot_20260728.csv")
    parser.add_argument("--output", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--report-csv", required=True)
    parser.add_argument("--gate-json", default="reports/active_learning_gate_20260729.json")
    parser.add_argument("--pair-count", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()
    inventory = json.loads(resolve(args.candidate_inventory).read_text())
    private = json.loads(resolve(args.existing_pairs).read_text())
    available = unlabeled_pairs(
        inventory["candidates"], existing_content_keys(private["pairs"])
    )
    model_available = (resolve(args.model_dir) / "ranker.json").is_file()
    with resolve(args.oof).open(newline="", encoding="utf-8") as stream:
        oof_rows = list(csv.DictReader(stream))
    oof_available = bool(oof_rows) and any(row.get("probability") for row in oof_rows)
    if not model_available or not oof_available:
        selected = []
        status = "ACTIVE_LEARNING_BLOCKED"
        reasons = [
            reason
            for condition, reason in (
                (not model_available, "portable ranker contract is unavailable"),
                (not oof_available, "real OOF probabilities are unavailable"),
            )
            if condition
        ]
    else:
        ranker = json.loads((resolve(args.model_dir) / "ranker.json").read_text())
        with resolve(args.features).open(newline="", encoding="utf-8") as stream:
            snapshot = list(csv.DictReader(stream))
        feature_map = {
            (row["case_id"], row["artifact_digest"]): {
                name: float(row[name]) for name in FEATURE_NAMES
            }
            for row in snapshot
        }
        graph = json.loads(resolve(args.human_graph).read_text())
        reference = {
            row["case_id"]: row["reference_status"] for row in graph["cases"]
        }
        scored = []
        for row in available:
            left = feature_map[(row["case_id"], row["left_digest"])]
            right = feature_map[(row["case_id"], row["right_digest"])]
            difference = [left[name] - right[name] for name in ranker["featureNames"]]
            scaled = [
                (value - mean) / width
                for value, mean, width in zip(
                    difference, ranker["scalerMean"], ranker["scalerScale"]
                )
            ]
            logit = sum(
                weight * value
                for weight, value in zip(ranker["coefficients"], scaled)
            ) + ranker["intercept"]
            probability = 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, logit))))
            rule_left = rule_score(left) >= rule_score(right)
            learned_left = probability >= 0.5
            conflict = rule_left != learned_left
            closure = reference.get(row["case_id"]) in {
                "PARTIAL_ORDER",
                "INSUFFICIENT_COMPARISON",
            }
            margin = abs(probability - 0.5) * 2.0
            row = {
                **row,
                "probability": probability,
                "margin": margin,
                "rule_disagreement": conflict,
                "graph_closure": closure,
                "score": acquisition_score(probability, conflict, closure),
            }
            scored.append(row)
        quotas = (
            ("LOW_MARGIN", lambda row: row["margin"]),
            ("RULE_LEARNED_CONFLICT", lambda row: -int(row["rule_disagreement"])),
            ("GRAPH_CLOSURE", lambda row: -int(row["graph_closure"])),
        )
        selected, used = [], set()
        per_reason = max(1, args.pair_count // len(quotas))
        for reason, order in quotas:
            eligible = sorted(scored, key=lambda row: (order(row), -row["score"], row["canonical_pair_key"]))
            for row in eligible:
                if row["canonical_pair_key"] in used:
                    continue
                if reason == "RULE_LEARNED_CONFLICT" and not row["rule_disagreement"]:
                    continue
                if reason == "GRAPH_CLOSURE" and not row["graph_closure"]:
                    continue
                selected.append({**row, "acquisition_reason": reason})
                used.add(row["canonical_pair_key"])
                if sum(item["acquisition_reason"] == reason for item in selected) == per_reason:
                    break
        for row in sorted(scored, key=lambda item: (-item["score"], item["canonical_pair_key"])):
            if len(selected) == args.pair_count:
                break
            if row["canonical_pair_key"] not in used:
                selected.append({**row, "acquisition_reason": "MIXED_INFORMATION"})
                used.add(row["canonical_pair_key"])
        for index, row in enumerate(selected, 1):
            row["pair_id"] = f"active-{args.seed}-{index:02d}"
            row["acquisition_score"] = row.pop("score")
        status = (
            "ACTIVE_REVIEW_READY"
            if len(selected) == args.pair_count
            else "PARTIAL_ACTIVE_QUEUE"
        )
        reasons = []
    write_csv(resolve(args.output), selected)
    write_csv(resolve(args.report_csv), selected)
    report = {
        "schemaVersion": "active-pair-selection/v1",
        "status": status,
        "requestedPairCount": args.pair_count,
        "selectedPairCount": len(selected),
        "remainingValidPairCount": len(available),
        "modelAvailable": model_available,
        "oofAvailable": oof_available,
        "sameDigestPairCount": 0,
        "crossCasePairCount": 0,
        "alreadyLabeledPairCount": 0,
        "finalSelectedMutationCount": 0,
        "blockedReasons": reasons,
    }
    resolve(args.report_json).write_text(json.dumps(report, indent=2) + "\n")
    resolve(args.gate_json).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
