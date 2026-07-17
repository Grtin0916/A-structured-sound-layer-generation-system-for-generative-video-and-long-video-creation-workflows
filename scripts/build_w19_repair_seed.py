#!/usr/bin/env python3
"""Prepare unique W18 failures as honest W19 repair-engine inputs."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


FIELDS = [
    "seed_id", "failure_id", "failure_type", "proposed_repair_action",
    "source_audio", "copied_source_audio", "before_metrics", "priority",
    "case_id", "candidate", "variant", "artifact_exists", "micro_probe_available",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--failure-bank", type=Path, required=True)
    parser.add_argument("--repair-priority", type=Path, required=True)
    parser.add_argument("--micro-repair-report", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--target-count", type=int, default=20)
    args = parser.parse_args()

    failures = read_csv(args.failure_bank)
    priorities = {row["candidate_key"]: row for row in read_csv(args.repair_priority)}
    probes = {row["failure_id"]: row for row in read_csv(args.micro_repair_report)}
    args.out_dir.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    rows: list[dict[str, str]] = []
    missing_sources: list[str] = []
    for index, failure in enumerate(failures, 1):
        key = failure["candidate_key"]
        if key in seen:
            continue
        seen.add(key)
        source = Path(failure["audio_path"])
        exists = source.is_file()
        copied = ""
        if exists:
            destination = args.out_dir / f'{failure["failure_id"]}__{source.name}'
            shutil.copy2(source, destination)
            copied = destination.as_posix()
        else:
            missing_sources.append(source.as_posix())
        priority_row = priorities.get(key, {})
        metrics = {
            "duration_sec": failure.get("duration_sec"),
            "peak_abs": failure.get("peak_abs"),
            "rms_mean": failure.get("rms_mean"),
            "onset_proxy_count": failure.get("onset_proxy_count"),
        }
        rows.append(
            {
                "seed_id": f"w19_{index:03d}",
                "failure_id": failure["failure_id"],
                "failure_type": failure["failure_category"],
                "proposed_repair_action": failure["next_action"],
                "source_audio": failure["audio_path"],
                "copied_source_audio": copied,
                "before_metrics": json.dumps(metrics, sort_keys=True),
                "priority": priority_row.get("selector_v2_rank", failure.get("selector_v2_rank", "")),
                "case_id": failure["case_id"],
                "candidate": key,
                "variant": failure["variant"],
                "artifact_exists": str(exists).lower(),
                "micro_probe_available": str(failure["failure_id"] in probes).lower(),
            }
        )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    required_complete = all(
        row["failure_type"] and row["proposed_repair_action"] and row["source_audio"]
        and row["before_metrics"] and row["priority"] and row["case_id"] and row["variant"]
        for row in rows
    )
    summary = {
        "targetCount": args.target_count,
        "actualCount": len(rows),
        "missingCount": max(args.target_count - len(rows), 0),
        "uniqueCandidateCount": len(seen),
        "artifactExistsCount": sum(row["artifact_exists"] == "true" for row in rows),
        "missingSourceAudio": missing_sources,
        "requiredFieldsComplete": required_complete,
        "actualCountGe12": len(rows) >= 12,
        "notClaimFullRepairEngine": True,
        "gateStatus": "PASS" if len(rows) >= 12 and required_complete and not missing_sources else "FAIL",
    }
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["gateStatus"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
