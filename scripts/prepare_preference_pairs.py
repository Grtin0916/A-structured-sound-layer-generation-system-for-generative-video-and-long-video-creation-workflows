#!/usr/bin/env python3
"""Prepare private truth, public blind manifests, and an empty label sheet."""

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from soundlayer.preference.pair_sampler import sample_pairs
from soundlayer.preference.schema import LABEL_FIELDS


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--pair-graph", required=True)
    parser.add_argument("--private-key", required=True)
    parser.add_argument("--public-pairs", required=True)
    parser.add_argument("--inventory-json", required=True)
    parser.add_argument("--inventory-csv", required=True)
    parser.add_argument(
        "--labels", default="annotations/preference_labels_20260727.csv"
    )
    parser.add_argument("--unique-pairs", type=int, default=36)
    parser.add_argument("--repeat-pairs", type=int, default=8)
    parser.add_argument("--audit-pairs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260727)
    args = parser.parse_args()

    inventory = json.loads(resolve(args.inventory).read_text())
    graph = json.loads(resolve(args.pair_graph).read_text())
    private, public, summary = sample_pairs(
        inventory,
        graph,
        args.unique_pairs,
        args.repeat_pairs,
        args.audit_pairs,
        args.seed,
    )
    private_payload = {
        "schemaVersion": "preference-private-key/v1",
        "gitHead": inventory["summary"]["gitHead"],
        "audioFeatureOrigin": inventory["summary"]["audioFeatureOrigin"],
        "artifactOrigin": inventory["summary"]["artifactOrigin"],
        "summary": summary,
        "pairs": private,
    }
    public_payload = {
        "schemaVersion": "blind-review-pairs/v1",
        "protocolVersion": "preference-v1",
        "pairs": public,
    }
    report = {
        "schemaVersion": "preference-pair-inventory/v1",
        "status": (
            "PAIR_POOL_READY"
            if summary["uniquePairShortfall"] == 0 and summary["caseCoverage"] == 12
            else "PAIR_POOL_INSUFFICIENT"
        ),
        "summary": {
            **summary,
            "connectedCaseGraphCount": graph["summary"][
                "connectedCaseGraphCount"
            ],
            "uniqueContentPairCount": graph["summary"]["uniqueContentPairCount"],
            "duplicateContentComparisonCount": graph["summary"][
                "duplicateContentComparisonCount"
            ],
            "insufficientVariationCaseCount": graph["summary"][
                "insufficientVariationCaseCount"
            ],
            "finalSelectedMutationCount": 0,
        },
    }
    dump(resolve(args.private_key), private_payload)
    dump(resolve(args.public_pairs), public_payload)
    dump(resolve(args.inventory_json), report)

    csv_path = resolve(args.inventory_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "pair_id",
        "case_id",
        "kind",
        "left_strategy",
        "right_strategy",
        "left_digest",
        "right_digest",
        "block_id",
        "display_index",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in private)

    label_path = resolve(args.labels)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with label_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=LABEL_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in private:
            writer.writerow(
                {
                    "pair_id": row["pair_id"],
                    "case_id": row["case_id"],
                    "protocol_version": row["protocol_version"],
                    "review_session_id": row["review_session_id"],
                    "left_artifact": row["left_artifact"],
                    "right_artifact": row["right_artifact"],
                    "left_digest": row["left_digest"],
                    "right_digest": row["right_digest"],
                    "presentation_order": row["presentation_order"],
                    "is_hidden_repeat": row["is_hidden_repeat"],
                    "repeat_group_id": row["repeat_group_id"],
                    "is_audit_pair": row["is_audit_pair"],
                    "submitted": False,
                }
            )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
