#!/usr/bin/env python3
"""Validate exported blind-review labels and decide training eligibility."""

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from soundlayer.preference.review_summary import summarize


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-key", required=True)
    parser.add_argument("--public-pairs", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--pair-graph", default="reports/preference_pair_graph_20260727.json")
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--quality-gate", required=True)
    parser.add_argument("--disagreements", required=True)
    args = parser.parse_args()

    private = json.loads(resolve(args.private_key).read_text())
    public = json.loads(resolve(args.public_pairs).read_text())
    graph = json.loads(resolve(args.pair_graph).read_text())
    with resolve(args.labels).open(newline="", encoding="utf-8") as stream:
        exported = list(csv.DictReader(stream))
    truth_by_id = {row["pair_id"]: row for row in private["pairs"]}
    labels = [
        {**truth_by_id.get(row.get("pair_id", ""), {}), **row}
        for row in exported
    ]
    gate = summarize(private["pairs"], public["pairs"], labels, graph["summary"])
    summary = {
        "schemaVersion": "preference-review-summary/v1",
        "status": gate["status"],
        "metrics": gate["metrics"],
        "claimBoundary": gate["claimBoundary"],
    }
    dump(resolve(args.summary_json), summary)
    dump(resolve(args.quality_gate), gate)

    summary_csv = resolve(args.summary_csv)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("metric", "value"))
        for key, value in gate["metrics"].items():
            writer.writerow((key, "" if value is None else value))

    disagreement_path = resolve(args.disagreements)
    disagreement_path.parent.mkdir(parents=True, exist_ok=True)
    with disagreement_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("pair_id", "case_id", "issue"))
        for item in gate["invalidJudgments"]:
            writer.writerow((item["pair_id"], "", "|".join(item["failures"])))
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
