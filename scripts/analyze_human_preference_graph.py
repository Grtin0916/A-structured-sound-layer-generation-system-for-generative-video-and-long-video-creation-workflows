#!/usr/bin/env python3
"""Create per-case partial-order references without inventing winners."""

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from soundlayer.ranking.human_preference_graph import build_human_graph


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--pair-key", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--cycles", required=True)
    parser.add_argument("--references", required=True)
    args = parser.parse_args()
    with resolve(args.labels).open(newline="", encoding="utf-8") as stream:
        labels = list(csv.DictReader(stream))
    private = json.loads(resolve(args.pair_key).read_text())
    graph = build_human_graph(labels, private["pairs"])
    resolve(args.out_json).write_text(json.dumps(graph, indent=2) + "\n")
    fields = (
        "case_id",
        "candidate_count",
        "observed_edge_count",
        "decisive_edge_count",
        "tie_count",
        "unjudgeable_count",
        "cycle_count",
        "transitivity_violation_count",
        "reference_status",
        "human_top_candidates",
    )
    flat = [
        {
            **{field: row.get(field, "") for field in fields},
            "human_top_candidates": json.dumps(row["human_top_candidates"]),
        }
        for row in graph["cases"]
    ]
    write_csv(resolve(args.out_csv), fields, flat)
    write_csv(
        resolve(args.references),
        ("case_id", "reference_status", "human_top_candidates"),
        [
            {
                "case_id": row["case_id"],
                "reference_status": row["reference_status"],
                "human_top_candidates": json.dumps(row["human_top_candidates"]),
            }
            for row in graph["cases"]
        ],
    )
    write_csv(
        resolve(args.cycles),
        ("case_id", "cycle_count", "edges"),
        [
            {
                "case_id": row["case_id"],
                "cycle_count": row["cycle_count"],
                "edges": json.dumps(row["edges"]),
            }
            for row in graph["cases"]
            if row["cycle_count"]
        ],
    )
    print(json.dumps(graph["summary"], indent=2))


if __name__ == "__main__":
    main()
