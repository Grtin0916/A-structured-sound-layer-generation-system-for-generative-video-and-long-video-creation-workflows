#!/usr/bin/env python3
"""Summarize ablation availability and keep empty error tables explicit."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def empty_csv(path, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream, lineterminator="\n").writerow(fields)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", required=True)
    parser.add_argument("--human-graph", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--disagreements", required=True)
    parser.add_argument("--risk-coverage", required=True)
    parser.add_argument("--high-confidence-errors", required=True)
    args = parser.parse_args()
    with resolve(args.ablation).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    graph = json.loads(resolve(args.human_graph).read_text())
    learned = [row for row in rows if row.get("r1_learned_score")]
    summary = {
        "schemaVersion": "ranker-ablation-summary/v1",
        "status": "RANKER_ABLATION_BLOCKED" if not learned else "EVALUATED",
        "rowCount": len(rows),
        "caseCount": len({row["case_id"] for row in rows}),
        "strategyCounts": dict(Counter(row["strategy_id"] for row in rows)),
        "recommendationStatusCounts": dict(
            Counter(row["recommendation_status"] for row in rows)
        ),
        "humanReferenceStatusCounts": dict(
            Counter(row["reference_status"] for row in graph["cases"])
        ),
        "learnedMetricAvailable": bool(learned),
        "riskCoverageAvailable": bool(learned),
        "highConfidenceErrorCount": 0,
        "finalSelectedMutationCount": 0,
        "blockedReasons": (
            [] if learned else ["no real OOF probabilities; learned ablation is not authorized"]
        ),
    }
    resolve(args.out_json).write_text(json.dumps(summary, indent=2) + "\n")
    empty_csv(
        resolve(args.disagreements),
        ("case_id", "rule_candidate", "learned_candidate", "human_support"),
    )
    empty_csv(
        resolve(args.risk_coverage),
        ("coverage", "selective_accuracy", "defer_rate"),
    )
    empty_csv(
        resolve(args.high_confidence_errors),
        ("case_id", "pair_key", "probability", "human_label", "error_type"),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
