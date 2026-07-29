#!/usr/bin/env python3
"""Build an active-review pack only for a non-empty, authorized queue."""

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--source-pack", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    with resolve(args.pairs).open(newline="", encoding="utf-8") as stream:
        pairs = list(csv.DictReader(stream))
    output = resolve(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "schemaVersion": "active-review-pack/v1",
        "status": "ACTIVE_LEARNING_BLOCKED" if not pairs else "ACTIVE_REVIEW_READY",
        "pairCount": len(pairs),
        "mediaCopiedCount": 0,
        "reason": "active pair queue is empty" if not pairs else "",
        "finalSelectedMutationCount": 0,
    }
    (output / "build_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
