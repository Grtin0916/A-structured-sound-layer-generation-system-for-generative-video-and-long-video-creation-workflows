#!/usr/bin/env python3
"""Run a bounded deterministic repair preflight on real failure audio."""

from __future__ import annotations

from array import array
import argparse
import csv
import json
from pathlib import Path
import shutil

from repair_audio_utils import audio_metrics, read_pcm16, write_pcm16


FIELDS = [
    "failure_id", "case_id", "candidate", "action", "source_audio", "before_audio",
    "after_audio", "before_metrics", "after_metrics", "target_metric_improved",
    "severe_regression", "readable_after",
]


def apply_action(plan: dict, params, samples: array) -> array:
    action = plan["action"]
    if action == "attenuate_limit":
        gain = float(plan["parameters"]["gain"])
        ceiling = int(float(plan["parameters"]["peak_ceiling"]) * 32767)
        return array("h", (max(-ceiling, min(ceiling, round(value * gain))) for value in samples))
    if action == "trim":
        threshold = int(float(plan["parameters"]["silence_threshold"]) * 32768)
        channels = params.nchannels
        active_frames = [
            index // channels for index, value in enumerate(samples) if abs(value) > threshold
        ]
        if not active_frames:
            return array("h", samples)
        padding = int(int(plan["parameters"]["padding_ms"]) / 1000 * params.framerate)
        first = max(min(active_frames) - padding, 0) * channels
        last = min(max(active_frames) + padding + 1, params.nframes) * channels
        return array("h", samples[first:last])
    raise ValueError(f"unsupported preflight action: {action}")


def target_improved(action: str, before: dict, after: dict) -> bool:
    if action == "attenuate_limit":
        return after["peak_abs"] < before["peak_abs"] and after["clipped_ratio"] <= before["clipped_ratio"]
    if action == "trim":
        return after["silence_ratio"] < before["silence_ratio"] or after["duration_sec"] < before["duration_sec"]
    return False


def severe_regression(plan: dict, before: dict, after: dict) -> bool:
    max_duration_loss = float(plan["max_regression"].get("duration_ratio", 1.0))
    duration_loss = 1.0 - after["duration_sec"] / max(before["duration_sec"], 1.0e-9)
    rms_loss = before["rms_dbfs"] - after["rms_dbfs"]
    return duration_loss > max_duration_loss + 1.0e-6 or rms_loss > float(plan["max_regression"].get("rms_db", 99.0))


def select_diverse(plans: list[dict], maximum: int) -> list[dict]:
    selected: list[dict] = []
    seen_actions: set[str] = set()
    seen_cases: set[str] = set()
    for plan in plans:
        if plan["action"] not in seen_actions:
            selected.append(plan)
            seen_actions.add(plan["action"])
            seen_cases.add(plan["case_id"])
    for plan in plans:
        if len(selected) >= maximum:
            break
        if plan not in selected and plan["case_id"] not in seen_cases:
            selected.append(plan)
            seen_cases.add(plan["case_id"])
    for plan in plans:
        if len(selected) >= maximum:
            break
        if plan not in selected:
            selected.append(plan)
    return selected[:maximum]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", "--execution-manifest", dest="manifest", type=Path, required=True
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--out-metrics", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--out-failures", "--failures-json", dest="out_failures", type=Path, required=True)
    parser.add_argument("--max-items", type=int, default=4)
    parser.add_argument("--allowed-actions", default="shift_left,delay_or_pad,attenuate_limit,trim")
    args = parser.parse_args()

    allowed = set(args.allowed_actions.split(","))
    all_plans = [
        json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    eligible = [plan for plan in all_plans if plan["action"] in allowed]
    plans = select_diverse(eligible, args.max_items)
    rows: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []

    for plan in plans:
        failure_dir = args.out_dir / plan["failure_id"]
        before_path = failure_dir / "before.wav"
        after_path = failure_dir / "after.wav"
        try:
            failure_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(plan["source_audio"], before_path)
            params, samples = read_pcm16(before_path)
            before = audio_metrics(params, samples)
            repaired = apply_action(plan, params, samples)
            write_pcm16(after_path, params, repaired)
            after_params, after_samples = read_pcm16(after_path)
            after = audio_metrics(after_params, after_samples)
            improved = target_improved(plan["action"], before, after)
            regression = severe_regression(plan, before, after)
            rows.append({
                "failure_id": plan["failure_id"],
                "case_id": plan["case_id"],
                "candidate": plan["candidate"],
                "action": plan["action"],
                "source_audio": plan["source_audio"],
                "before_audio": before_path.as_posix(),
                "after_audio": after_path.as_posix(),
                "before_metrics": json.dumps(before, sort_keys=True),
                "after_metrics": json.dumps(after, sort_keys=True),
                "target_metric_improved": str(improved).lower(),
                "severe_regression": str(regression).lower(),
                "readable_after": "true",
            })
        except Exception as exc:
            failures.append({
                "failure_id": plan["failure_id"],
                "reason": f"{type(exc).__name__}: {exc}",
            })

    args.out_metrics.parent.mkdir(parents=True, exist_ok=True)
    with args.out_metrics.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    improved_count = sum(row["target_metric_improved"] == "true" for row in rows)
    severe_count = sum(row["severe_regression"] == "true" for row in rows)
    readable_count = sum(row["readable_after"] == "true" for row in rows)
    non_improving = [
        {
            "failure_id": row["failure_id"],
            "action": row["action"],
            "reason": "target_metric_not_improved_in_preflight",
            "severity": "non_fatal",
        }
        for row in rows
        if row["target_metric_improved"] != "true"
    ]
    runner_ready = len(rows) == args.max_items and readable_count == len(rows) and improved_count >= 2 and severe_count == 0
    summary = {
        "requestedCount": args.max_items,
        "preflightCount": len(rows),
        "readableAfterCount": readable_count,
        "targetMetricImprovedCount": improved_count,
        "targetMetricNotImprovedCount": len(non_improving),
        "targetMetricNotImprovedFailureIds": [item["failure_id"] for item in non_improving],
        "severeRegressionCount": severe_count,
        "executionFailureCount": len(failures),
        "runnerInputReady": runner_ready,
        "allowedActions": sorted(allowed),
        "availableFailureTypes": sorted({plan["failure_type"] for plan in eligible}),
        "onsetBoundaryCaseAvailable": any(
            plan["failure_type"] in {"onset_late", "onset_early"} for plan in eligible
        ),
        "limitation": "No real onset_late/onset_early failure exists in the W18 evidence; none was fabricated.",
        "gateStatus": "PASS" if runner_ready else "FAIL",
    }
    args.out_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.out_failures.write_text(
        json.dumps(
            {"executionFailures": failures, "nonImprovingCases": non_improving},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))
    return 0 if runner_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
