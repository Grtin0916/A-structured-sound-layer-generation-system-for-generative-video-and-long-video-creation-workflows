#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soundlayer.models.mmaudio_runner import run_queue_csv


ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "experiments" / "mmaudio_baseline_2026_06_30" / "candidate_run_queue.csv"
REPORTS = ROOT / "reports"
OUT_DIR = ROOT / "artifacts" / "model_runs" / "week17_mmaudio"


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not QUEUE.exists():
        raise SystemExit(f"QUEUE_NOT_FOUND: {QUEUE}")

    results = run_queue_csv(QUEUE, OUT_DIR, max_candidates=12)

    metrics_csv = REPORTS / "mmaudio_baseline_metrics.csv"
    failures_json = REPORTS / "mmaudio_baseline_failures.json"
    summary_json = REPORTS / "mmaudio_baseline_summary.json"

    fieldnames = [
        "candidate_id",
        "case_id",
        "model",
        "prompt_variant",
        "status",
        "fallback_used",
        "video_conditioned",
        "blocked_reason",
        "output_wav",
        "readable",
        "duration_sec",
        "sample_rate",
        "channels",
        "rms",
        "peak",
        "clip_rate",
        "silence_ratio",
        "runtime_sec",
    ]

    with metrics_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    failures = [
        r for r in results
        if r.get("fallback_used") or not r.get("readable") or r.get("status") != "generated"
    ]
    failures_json.write_text(json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8")

    readable_count = sum(1 for r in results if r.get("readable"))
    fallback_count = sum(1 for r in results if r.get("fallback_used"))
    generated_count = sum(1 for r in results if r.get("status") == "generated")
    wav_count = len([p for p in (ROOT / "experiments" / "mmaudio_baseline_2026_06_30" / "candidates").glob("*.wav")])

    decision = (
        "GREEN_LOCAL_MMAUDIO_GENERATED"
        if generated_count >= 8 and readable_count >= 8
        else "YELLOW_FALLBACK_CONTROL_AUDIO_READY_MMAUDIO_BLOCKED"
        if readable_count >= 8
        else "RED_NO_USABLE_AUDIO"
    )

    summary = {
        "decision": decision,
        "candidate_count": len(results),
        "readable_count": readable_count,
        "generated_count": generated_count,
        "fallback_count": fallback_count,
        "wav_count": wav_count,
        "metrics_csv": str(metrics_csv),
        "failures_json": str(failures_json),
        "boundary": {
            "local_mmaudio_success_claimed": generated_count > 0,
            "fallback_control_is_not_video_conditioned_v2a": fallback_count > 0,
            "do_not_claim_synchronized_v2a_for_fallback": fallback_count > 0,
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if readable_count >= 8 else 1


if __name__ == "__main__":
    raise SystemExit(main())