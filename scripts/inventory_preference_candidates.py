#!/usr/bin/env python3
"""Inventory frozen W20 candidates and construct the connected pair graph."""

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from soundlayer.preference.candidate_graph import build_pair_graph, load_inventory


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-matrix", required=True)
    parser.add_argument("--ablation", required=True)
    parser.add_argument("--handoff", required=True)
    parser.add_argument(
        "--repair-handoff",
        default="artifacts/manifests/repair_handoff_20260715.json",
    )
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--graph-json", required=True)
    args = parser.parse_args()

    inventory = load_inventory(
        ROOT,
        resolve(args.candidate_matrix),
        resolve(args.ablation),
        resolve(args.handoff),
        resolve(args.repair_handoff),
    )
    graph = build_pair_graph(inventory)
    write_json(resolve(args.out_json), inventory)
    write_json(resolve(args.graph_json), graph)
    csv_path = resolve(args.out_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "case_id",
        "source_case_id",
        "strategy_id",
        "artifact_path",
        "artifact_digest",
        "video_path",
        "duration_sec",
        "sample_rate",
        "channels",
        "publish_decision",
        "repair_decision",
        "git_head",
        "audio_feature_origin",
        "artifact_origin",
        "candidate_role",
        "ablation_materialized",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in inventory["candidates"])
    print(json.dumps({"inventory": inventory["summary"], "graph": graph["summary"]}, indent=2))


if __name__ == "__main__":
    main()
