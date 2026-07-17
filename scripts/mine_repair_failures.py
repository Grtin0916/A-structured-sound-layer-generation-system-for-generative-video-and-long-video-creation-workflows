#!/usr/bin/env python3
"""Mine only threshold-backed W18 hard negatives not already in the repair seed."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = [
    "failure_id", "case_id", "candidate", "variant", "failure_type",
    "proposed_repair_action", "source_audio", "before_metrics", "priority",
    "artifact_exists", "evidence", "threshold_rule",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--existing-seed", type=Path, required=True)
    parser.add_argument("--failure-bank", type=Path, required=True)
    parser.add_argument("--selector-scores", type=Path, required=True)
    parser.add_argument("--selector-rejections", type=Path)
    parser.add_argument("--repair-priority", type=Path)
    parser.add_argument("--audio-metrics", type=Path)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--target-new-count", type=int, default=8)
    args = parser.parse_args()

    seed = read_csv(args.existing_seed)
    seeded = {row["candidate"] for row in seed}
    rows = read_csv(args.selector_scores)
    mined: list[dict[str, str]] = []

    for row in rows:
        candidate = row["candidate_key"]
        if candidate in seeded or row.get("selector_v2_decision") == "winner":
            continue
        clip = float(row["metric_clipped_ratio"])
        silence = float(row["metric_silence_ratio"])
        rms_dbfs = float(row["metric_rms_dbfs"])
        rules: list[str] = []
        failure_type = ""
        action = ""
        if clip >= 0.00001 or rms_dbfs >= -12.0:
            failure_type = "clipping"
            action = "clipping_attenuation"
            if clip >= 0.00001:
                rules.append(f"clipped_ratio={clip}>=0.00001")
            if rms_dbfs >= -12.0:
                rules.append(f"rms_dbfs={rms_dbfs}>=-12")
        elif silence >= 0.12:
            failure_type = "excessive_silence"
            action = "trim_or_event_local_gain"
            rules.append(f"silence_ratio={silence}>=0.12")
        else:
            continue

        source = Path(row["audio_path"])
        metrics = {
            "duration_sec": float(row["metric_duration_sec"]),
            "peak_dbfs": float(row["metric_peak_dbfs"]),
            "rms_dbfs": rms_dbfs,
            "clipped_ratio": clip,
            "silence_ratio": silence,
            "onset_proxy_count": int(float(row["metric_onset_count_proxy"])),
        }
        mined.append(
            {
                "failure_id": f"hn_{len(mined) + 1:03d}",
                "case_id": row["case_id"],
                "candidate": candidate,
                "variant": row["variant"],
                "failure_type": failure_type,
                "proposed_repair_action": action,
                "source_audio": source.as_posix(),
                "before_metrics": json.dumps(metrics, sort_keys=True),
                "priority": row["selector_v2_rank"],
                "artifact_exists": str(source.is_file()).lower(),
                "evidence": ";".join(rules),
                "threshold_rule": "fixed_w19_preflight_v1",
            }
        )
        if len(mined) >= args.target_new_count:
            break

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(mined)

    summary = {
        "targetNewCount": args.target_new_count,
        "newUniqueFailureCount": len(mined),
        "missingCount": max(args.target_new_count - len(mined), 0),
        "existingSeedCount": len(seed),
        "overlapWithSeedCount": sum(row["candidate"] in seeded for row in mined),
        "missingSourceCount": sum(row["artifact_exists"] != "true" for row in mined),
        "thresholds": {
            "clippedRatioMin": 0.00001,
            "loudRmsDbfsMin": -12.0,
            "excessiveSilenceRatioMin": 0.12,
        },
        "thresholdsLoweredToFillTarget": False,
        "status": "HONEST_PARTIAL" if len(mined) < args.target_new_count else "TARGET_MET",
    }
    args.out_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
