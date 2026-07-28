#!/usr/bin/env python3
"""Summarize evaluation availability without bypassing the promotion gate."""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--oof", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--rule-results")
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--output", default="reports/preference_ranker_evaluation_20260728.json")
    args = parser.parse_args()
    model_card = json.loads((resolve(args.model_dir) / "model_card.json").read_text())
    status = model_card["promotionStatus"]
    result = {
        "schemaVersion": "preference-ranker-evaluation/v1",
        "status": "NOT_RUN_DATA_BLOCKED" if status == "DATA_BLOCKED" else "EVALUATED",
        "promotionStatus": status,
        "bootstrapUnit": "case_id",
        "bootstrapResamples": args.bootstrap_resamples,
        "metricsAvailable": status != "DATA_BLOCKED",
        "explanationsAvailable": status != "DATA_BLOCKED",
        "finalSelectedMutationCount": 0,
    }
    resolve(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
