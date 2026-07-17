#!/usr/bin/env python3
"""Validate W19 seed identity, lineage, metrics, and action coverage."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()

    with args.seed_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    source_summary = json.loads(args.summary_json.read_text(encoding="utf-8"))

    failure_counts = Counter(row.get("failure_id", "") for row in rows)
    candidate_counts = Counter(row.get("candidate", "") for row in rows)
    duplicate_failure_ids = sorted(key for key, count in failure_counts.items() if key and count > 1)
    duplicate_candidates = sorted(key for key, count in candidate_counts.items() if key and count > 1)
    missing_sources: list[str] = []
    invalid_metrics: list[str] = []
    missing_actions: list[str] = []

    for row in rows:
        source = Path(row.get("source_audio", ""))
        if not source.is_file():
            missing_sources.append(source.as_posix())
        try:
            metrics = json.loads(row.get("before_metrics", ""))
            for key in ("duration_sec", "peak_abs", "rms_mean", "onset_proxy_count"):
                float(metrics[key])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            invalid_metrics.append(row.get("failure_id", ""))
        if not row.get("proposed_repair_action"):
            missing_actions.append(row.get("failure_id", ""))

    result = {
        "gateStatus": "PASS",
        "seedCount": len(rows),
        "uniqueFailureCount": len(failure_counts),
        "uniqueCandidateCount": len(candidate_counts),
        "duplicateFailureIdCount": len(duplicate_failure_ids),
        "duplicateCandidateCount": len(duplicate_candidates),
        "missingSourceCount": len(missing_sources),
        "invalidMetricCount": len(invalid_metrics),
        "missingTargetActionCount": len(missing_actions),
        "targetCount": source_summary.get("targetCount"),
        "missingCount": source_summary.get("missingCount"),
        "duplicates": {
            "failureIds": duplicate_failure_ids,
            "candidates": duplicate_candidates,
        },
        "missingSources": missing_sources,
        "invalidMetrics": invalid_metrics,
        "missingActions": missing_actions,
    }
    passed = (
        len(rows) == 12
        and result["uniqueFailureCount"] == 12
        and result["uniqueCandidateCount"] == 12
        and not duplicate_failure_ids
        and not duplicate_candidates
        and not missing_sources
        and not invalid_metrics
        and not missing_actions
        and source_summary.get("missingCount") == 8
    )
    result["gateStatus"] = "PASS" if passed else "FAIL"
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
