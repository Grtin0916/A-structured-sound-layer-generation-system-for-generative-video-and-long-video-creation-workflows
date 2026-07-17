#!/usr/bin/env python3
"""Build an evidence-backed repair bank and one diagnostic plot per failure."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from repair_audio_utils import audio_metrics, read_pcm16, write_diagnostic_png


FIELDS = [
    "failure_id", "case_id", "candidate", "variant", "failure_type",
    "proposed_repair_action", "source_audio", "before_metrics", "priority",
    "artifact_exists", "evidence", "duration_sec", "sample_rate", "channels",
    "sample_width", "event_id", "event_start_sec", "event_end_sec",
    "window_source", "window_confidence", "target_start_sec", "target_end_sec",
    "has_stems", "plot_path",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def event_window(case_id: str, duration: float) -> tuple[str, float, float, str, str]:
    dss_path = Path("cases") / case_id / "director_sound_script.yaml"
    if not dss_path.is_file():
        return "", 0.0, duration, "audio_duration_fallback", "low"
    payload = json.loads(dss_path.read_text(encoding="utf-8"))
    events = payload.get("events", [])
    if not events:
        return "", 0.0, duration, "audio_duration_fallback", "low"
    event = max(events, key=lambda item: (int(item.get("priority", 0)), -float(item.get("time_s", 0))))
    start = max(0.0, min(float(event.get("time_s", 0.0)), duration))
    end = max(start, min(start + float(event.get("duration_s", 0.0)), duration))
    return str(event.get("event_id", "")), start, end, dss_path.as_posix(), "high"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=Path, required=True)
    parser.add_argument("--hard-negatives", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--out-failures", "--failures-json", dest="out_failures", type=Path, required=True)
    parser.add_argument("--plot-dir", type=Path, required=True)
    parser.add_argument("--target-count", type=int, default=20)
    args = parser.parse_args()

    inputs = read_csv(args.seed) + read_csv(args.hard_negatives)
    bank: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    seen: set[str] = set()

    for source_row in inputs:
        candidate = source_row["candidate"]
        if candidate in seen:
            failures.append({"candidate": candidate, "reason": "duplicate_candidate"})
            continue
        seen.add(candidate)
        source = Path(source_row["source_audio"])
        try:
            params, samples = read_pcm16(source)
            measured = audio_metrics(params, samples)
            duration = float(measured["duration_sec"])
            event_id, event_start, event_end, window_source, confidence = event_window(
                source_row["case_id"], duration
            )
            failure_type = source_row["failure_type"]
            if failure_type in {"clipping", "silence", "excessive_silence"}:
                target_start, target_end = 0.0, duration
            else:
                target_start, target_end = event_start, event_end
            plot_path = args.plot_dir / f"{source_row['failure_id']}.png"
            write_diagnostic_png(
                plot_path, samples, params.framerate, params.nchannels,
                event_start, event_end, target_start, target_end,
            )
            bank.append({
                "failure_id": source_row["failure_id"],
                "case_id": source_row["case_id"],
                "candidate": candidate,
                "variant": source_row["variant"],
                "failure_type": failure_type,
                "proposed_repair_action": source_row["proposed_repair_action"],
                "source_audio": source.as_posix(),
                "before_metrics": source_row["before_metrics"],
                "priority": source_row["priority"],
                "artifact_exists": str(source.is_file()).lower(),
                "evidence": source_row.get("evidence", "w18_failure_bank"),
                "duration_sec": f"{duration:.6f}",
                "sample_rate": params.framerate,
                "channels": params.nchannels,
                "sample_width": params.sampwidth,
                "event_id": event_id,
                "event_start_sec": f"{event_start:.6f}",
                "event_end_sec": f"{event_end:.6f}",
                "window_source": window_source,
                "window_confidence": confidence,
                "target_start_sec": f"{target_start:.6f}",
                "target_end_sec": f"{target_end:.6f}",
                "has_stems": "false",
                "plot_path": plot_path.as_posix(),
            })
        except Exception as exc:
            failures.append({
                "failure_id": source_row.get("failure_id", ""),
                "candidate": candidate,
                "source_audio": source.as_posix(),
                "reason": f"{type(exc).__name__}: {exc}",
            })

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(bank)

    categories: dict[str, int] = {}
    for row in bank:
        categories[row["failure_type"]] = categories.get(row["failure_type"], 0) + 1
    summary = {
        "targetCount": args.target_count,
        "actualCount": len(bank),
        "uniqueFailureCount": len({row["candidate"] for row in bank}),
        "missingCount": max(args.target_count - len(bank), 0),
        "sourceExistsCount": sum(row["artifact_exists"] == "true" for row in bank),
        "missingSourceCount": sum(row["artifact_exists"] != "true" for row in bank),
        "plotCount": sum(Path(row["plot_path"]).is_file() for row in bank),
        "categoryCounts": categories,
        "failureCount": len(failures),
        "thresholdsLoweredToFillTarget": False,
        "gateStatus": "PASS" if len(bank) >= 20 and not failures else "YELLOW_HONEST_PARTIAL",
    }
    args.out_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.out_failures.write_text(json.dumps(failures, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
