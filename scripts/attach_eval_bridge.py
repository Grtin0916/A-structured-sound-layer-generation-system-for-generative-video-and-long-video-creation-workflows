#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from datetime import datetime, timezone

ROW_KEYS = ("results", "rows", "samples", "items", "content", "records", "data")

def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def dump_json(path: Path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def attach_to_records(obj, bridge):
    row_count = 0

    def mark_row(row):
        row["external_task_id"] = bridge["external_task_id"]
        row["external_task_status"] = bridge.get("external_task_status", "CREATED")
        row["java_repo_commit"] = bridge["java_repo_commit"]
        row["cloud_repo_commit"] = bridge["cloud_repo_commit"]
        row["java_query_path"] = bridge["java_query_path"]
        return row

    if isinstance(obj, list):
        for row in obj:
            if isinstance(row, dict):
                mark_row(row)
                row_count += 1
        return obj, row_count

    if isinstance(obj, dict):
        obj["cross_repo_bridge"] = bridge
        for key in ROW_KEYS:
            if isinstance(obj.get(key), list):
                for row in obj[key]:
                    if isinstance(row, dict):
                        mark_row(row)
                        row_count += 1
        if row_count == 0:
            # Some eval outputs are dict-only summaries; still make the bridge explicit.
            obj["external_task_id"] = bridge["external_task_id"]
            obj["java_repo_commit"] = bridge["java_repo_commit"]
            obj["cloud_repo_commit"] = bridge["cloud_repo_commit"]
            row_count = 1
        return obj, row_count

    raise TypeError(f"Unsupported JSON root type: {type(obj)!r}")

def attach_csv(path: Path, bridge):
    if not path.exists():
        return 0

    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            rows.append(row)

    if not fieldnames:
        return 0

    extra_fields = [
        "external_task_id",
        "external_task_status",
        "java_repo_commit",
        "cloud_repo_commit",
        "java_query_path",
    ]

    for field in extra_fields:
        if field not in fieldnames:
            fieldnames.append(field)

    for row in rows:
        row["external_task_id"] = bridge["external_task_id"]
        row["external_task_status"] = bridge.get("external_task_status", "CREATED")
        row["java_repo_commit"] = bridge["java_repo_commit"]
        row["cloud_repo_commit"] = bridge["cloud_repo_commit"]
        row["java_query_path"] = bridge["java_query_path"]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)

def update_metrics(path: Path, bridge, json_rows, csv_rows):
    metrics = {}
    if path.exists():
        metrics = load_json(path)

    metrics["cross_repo_bridge_external_task_id_present"] = 1
    metrics["cross_repo_bridge_json_rows_with_task_id"] = int(json_rows)
    metrics["cross_repo_bridge_csv_rows_with_task_id"] = int(csv_rows)
    metrics["cross_repo_bridge_java_commit_present"] = 1
    metrics["cross_repo_bridge_cloud_commit_present"] = 1
    metrics["cross_repo_bridge_checked_at_epoch"] = int(datetime.now(timezone.utc).timestamp())

    dump_json(path, metrics)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bridge", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-metrics", required=True)
    args = parser.parse_args()

    bridge_path = Path(args.bridge)
    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)
    out_metrics = Path(args.out_metrics)

    bridge = load_json(bridge_path)

    data = load_json(out_json)
    data, json_rows = attach_to_records(data, bridge)
    dump_json(out_json, data)

    csv_rows = attach_csv(out_csv, bridge)
    update_metrics(out_metrics, bridge, json_rows, csv_rows)

    print(json.dumps({
        "bridge_ok": True,
        "external_task_id": bridge["external_task_id"],
        "json_rows_with_task_id": json_rows,
        "csv_rows_with_task_id": csv_rows,
        "metrics": str(out_metrics)
    }, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
