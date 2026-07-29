#!/usr/bin/env python3
"""Materialize the 48-row ablation contract while failing closed without OOF."""

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from soundlayer.ranking.pairwise_features import FEATURE_NAMES
from soundlayer.ranking.ranker_ablation import (
    dss_interactions,
    hard_guard,
    rule_score,
    selective_hybrid,
)


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--rule-results")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--human-reference", required=True)
    parser.add_argument("--candidate-inventory", default="reports/preference_candidate_inventory_20260727.json")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()
    features = load_csv(resolve(args.features))
    references = {
        row["case_id"]: row for row in load_csv(resolve(args.human_reference))
    }
    inventory = json.loads(resolve(args.candidate_inventory).read_text())
    decisions = {
        (row["case_id"], row["strategy_id"]): row["publish_decision"]
        for row in inventory["candidates"]
    }
    model_available = (resolve(args.model_dir) / "ranker.json").is_file()
    prepared = []
    for feature_row in sorted(features, key=lambda row: (row["case_id"], row["strategy_id"])):
        values = {name: float(feature_row[name]) for name in FEATURE_NAMES}
        guard = hard_guard(
            values,
            decisions.get((feature_row["case_id"], feature_row["strategy_id"]), ""),
        )
        prepared.append(
            {
                "feature_row": feature_row,
                "values": values,
                "guard": guard,
                "rule_score": rule_score(values),
            }
        )
    rule_reference = {}
    for case_id in sorted({row["feature_row"]["case_id"] for row in prepared}):
        eligible = sorted(
            (
                row
                for row in prepared
                if row["feature_row"]["case_id"] == case_id and row["guard"]["passed"]
            ),
            key=lambda row: (
                -row["rule_score"],
                row["feature_row"]["strategy_id"],
            ),
        )
        rule_reference[case_id] = {
            "selected": eligible[0]["feature_row"]["artifact_digest"] if eligible else "",
            "runner_up_score": eligible[1]["rule_score"] if len(eligible) > 1 else "",
            "margin": (
                eligible[0]["rule_score"] - eligible[1]["rule_score"]
                if len(eligible) > 1
                else ""
            ),
        }
    rows = []
    for prepared_row in prepared:
        feature_row = prepared_row["feature_row"]
        values = prepared_row["values"]
        guard = prepared_row["guard"]
        hybrid = selective_hybrid(
            rule_reference[feature_row["case_id"]]["selected"] or None,
            None,
            0.0,
            guard,
            model_available=model_available,
        )
        reference = references.get(feature_row["case_id"], {})
        rows.append(
            {
                "case_id": feature_row["case_id"],
                "strategy_id": feature_row["strategy_id"],
                "candidate_digest": feature_row["artifact_digest"],
                "r0_rule_score": prepared_row["rule_score"],
                "r1_learned_score": "",
                "r2_interaction_score": "",
                "r3_hybrid_score": "",
                "selected_candidate": rule_reference[feature_row["case_id"]]["selected"],
                "recommendation_status": hybrid["recommendation_status"],
                "runner_up_score": rule_reference[feature_row["case_id"]]["runner_up_score"],
                "margin": rule_reference[feature_row["case_id"]]["margin"],
                "guard_status": "PASS" if guard["passed"] else "BLOCKED",
                "defer_reason": hybrid["defer_reason"],
                "human_support": (
                    "UNOBSERVED"
                    if reference.get("reference_status") == "INSUFFICIENT_COMPARISON"
                    else "AMBIGUOUS"
                ),
                "interaction_feature_count": len(dss_interactions(values)),
                "final_selected_mutation": 0,
            }
        )
    fields = tuple(rows[0])
    out_csv = resolve(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "schemaVersion": "ranker-ablation/v1",
        "status": "RANKER_ABLATION_BLOCKED" if not model_available else "RANKER_ABLATION_READY",
        "summary": {
            "rowCount": len(rows),
            "caseCount": len({row["case_id"] for row in rows}),
            "strategyCount": len({row["strategy_id"] for row in rows}),
            "ruleDiagnosticCount": len(rows),
            "ruleDiagnosticWinnerCaseCount": sum(
                bool(value["selected"]) for value in rule_reference.values()
            ),
            "learnedScoreCount": sum(bool(row["r1_learned_score"]) for row in rows),
            "rankerRecommendationCount": sum(
                row["recommendation_status"] == "RANKER_RECOMMENDED" for row in rows
            ),
            "publishBlockedCount": sum(
                row["recommendation_status"] == "PUBLISH_BLOCKED" for row in rows
            ),
            "dataBlockedCount": sum(
                row["recommendation_status"] == "ABLATION_DATA_BLOCKED"
                for row in rows
            ),
            "finalSelectedMutationCount": 0,
        },
        "modelAvailable": model_available,
        "humanReferenceAvailable": any(
            row.get("reference_status") != "INSUFFICIENT_COMPARISON"
            for row in references.values()
        ),
        "blockedReasons": (
            []
            if model_available
            else ["real OOF model is unavailable because human labels are not submitted"]
        ),
    }
    resolve(args.out_json).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
