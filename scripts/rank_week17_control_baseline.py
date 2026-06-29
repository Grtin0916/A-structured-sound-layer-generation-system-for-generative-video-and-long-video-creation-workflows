#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rank Week17 control-rule Foley baseline against DSS event windows.

This is not a learned perceptual metric.
It is a deterministic model-race seed:
- reads DSS events
- reads control WAV candidates
- scores event-window energy coverage
- estimates basic signal quality
- outputs provisional winners and repair queue

Outputs:
- artifacts/model_race/week17_control_seed/event_window_scores.csv
- artifacts/model_race/week17_control_seed/control_seed_ranking.csv/json
- artifacts/model_race/week17_control_seed/repair_queue_seed.csv/json
- reports/week17_control_seed_summary.json
- reports/week17_java_model_race_seed_payload.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import struct
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def read_wav_mono(path: Path) -> tuple[list[float], int, float]:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sr = wf.getframerate()
        frames = wf.getnframes()
        raw = wf.readframes(frames)

    if sampwidth != 2:
        raise ValueError(f"Only 16-bit PCM WAV is supported, got sampwidth={sampwidth}: {path}")

    values = struct.unpack("<" + "h" * (len(raw) // 2), raw)
    if channels == 1:
        samples = [v / 32768.0 for v in values]
    else:
        samples = []
        for i in range(0, len(values), channels):
            samples.append(sum(values[i:i + channels]) / (channels * 32768.0))

    duration_s = len(samples) / sr if sr > 0 else 0.0
    return samples, sr, duration_s


def rms(xs: list[float]) -> float:
    if not xs:
        return 0.0
    return math.sqrt(sum(x * x for x in xs) / len(xs))


def clip_rate(xs: list[float]) -> float:
    if not xs:
        return 0.0
    return sum(1 for x in xs if abs(x) >= 0.98) / len(xs)


def silence_ratio(xs: list[float], threshold: float = 0.003) -> float:
    if not xs:
        return 1.0
    return sum(1 for x in xs if abs(x) < threshold) / len(xs)


def slice_samples(samples: list[float], sr: int, start_s: float, end_s: float) -> list[float]:
    start = max(0, min(len(samples), int(start_s * sr)))
    end = max(start, min(len(samples), int(end_s * sr)))
    return samples[start:end]


def event_threshold(layer_role: str, priority: int) -> float:
    if layer_role == "ambience":
        return 0.004
    if layer_role in {"foley", "fx"}:
        return max(0.010, 0.006 + priority * 0.002)
    return 0.008


def score_candidate(case_dir: Path, wav_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    dss = read_json(case_dir / "director_sound_script.yaml")
    case_id = dss.get("case_id", case_dir.name)
    samples, sr, wav_duration = read_wav_mono(wav_path)
    global_rms = rms(samples)
    global_peak = max((abs(x) for x in samples), default=0.0)
    global_clip = clip_rate(samples)
    global_silence = silence_ratio(samples)

    events = dss.get("events", [])
    event_rows: list[dict[str, Any]] = []
    repair_rows: list[dict[str, Any]] = []

    weighted_hit = 0.0
    weighted_total = 0.0
    priority_miss_count = 0

    for ev in events:
        event_id = ev.get("event_id", "")
        layer_role = ev.get("layer_role", "")
        priority = int(ev.get("priority", 3))
        time_s = float(ev.get("time_s", 0.0))
        duration_s = float(ev.get("duration_s", 0.3))
        tolerance_ms = int(ev.get("tolerance_ms", 200))

        pad = max(0.05, min(0.25, tolerance_ms / 1000.0))
        win_start = max(0.0, time_s - pad)
        win_end = min(wav_duration, time_s + duration_s + pad)
        event_samples = slice_samples(samples, sr, win_start, win_end)

        event_rms = rms(event_samples)
        energy_ratio = event_rms / max(global_rms, 1e-8)
        threshold = event_threshold(layer_role, priority)

        if layer_role == "ambience":
            covered = event_rms >= threshold and global_silence < 0.40
        else:
            covered = event_rms >= threshold and energy_ratio >= 0.65

        event_score = min(1.0, event_rms / max(threshold, 1e-8)) * (1.0 if covered else 0.55)
        weighted_hit += priority * (1.0 if covered else 0.0)
        weighted_total += priority

        if not covered and priority >= 4:
            priority_miss_count += 1
            repair_rows.append(
                {
                    "case_id": case_id,
                    "candidate_id": f"{case_id}__control_rule_foley_v0",
                    "event_id": event_id,
                    "repair_bucket": "priority_event_weak_or_missing",
                    "suggested_action": "increase event-region gain or regenerate/replace this event candidate",
                    "priority": priority,
                    "event_rms": round(event_rms, 6),
                    "threshold": round(threshold, 6),
                }
            )

        event_rows.append(
            {
                "case_id": case_id,
                "candidate_id": f"{case_id}__control_rule_foley_v0",
                "event_id": event_id,
                "layer_role": layer_role,
                "priority": priority,
                "time_s": time_s,
                "duration_s": duration_s,
                "tolerance_ms": tolerance_ms,
                "window_start_s": round(win_start, 3),
                "window_end_s": round(win_end, 3),
                "event_rms": round(event_rms, 6),
                "global_rms": round(global_rms, 6),
                "energy_ratio": round(energy_ratio, 6),
                "threshold": round(threshold, 6),
                "covered": covered,
                "event_score": round(event_score, 6),
            }
        )

    weighted_coverage = weighted_hit / weighted_total if weighted_total else 0.0
    duration_expected = float(dss.get("video", {}).get("duration_s", wav_duration))
    duration_error = abs(wav_duration - duration_expected)
    duration_score = max(0.0, 1.0 - duration_error / max(duration_expected, 1e-6))

    quality_score = 1.0
    quality_score -= min(0.35, global_clip * 200.0)
    quality_score -= min(0.30, max(0.0, global_silence - 0.35))
    quality_score -= min(0.20, max(0.0, 0.015 - global_rms) * 5.0)
    quality_score = max(0.0, min(1.0, quality_score))

    total_score = 0.65 * weighted_coverage + 0.25 * quality_score + 0.10 * duration_score

    if priority_miss_count > 0:
        status = "repair_required"
        reason = f"{priority_miss_count} high-priority event(s) weak or missing"
    elif total_score >= 0.75:
        status = "winner_seed"
        reason = "control baseline covers DSS event windows with acceptable signal quality"
    elif total_score >= 0.60:
        status = "usable_but_repair_recommended"
        reason = "usable baseline, but event coverage or quality is not strong"
    else:
        status = "repair_required"
        reason = "low aggregate event-window coverage or weak signal quality"

    if global_clip > 0.001:
        repair_rows.append(
            {
                "case_id": case_id,
                "candidate_id": f"{case_id}__control_rule_foley_v0",
                "event_id": "",
                "repair_bucket": "clip_risk",
                "suggested_action": "lower gain or apply peak normalization",
                "priority": "",
                "event_rms": "",
                "threshold": "",
            }
        )

    if global_silence > 0.35:
        repair_rows.append(
            {
                "case_id": case_id,
                "candidate_id": f"{case_id}__control_rule_foley_v0",
                "event_id": "",
                "repair_bucket": "high_silence_ratio",
                "suggested_action": "boost ambience bed or fill missing event regions",
                "priority": "",
                "event_rms": "",
                "threshold": "",
            }
        )

    ranking_row = {
        "case_id": case_id,
        "candidate_id": f"{case_id}__control_rule_foley_v0",
        "model_family": "control_rule_foley_v0",
        "candidate_path": str(wav_path),
        "status": status,
        "selection_reason": reason,
        "event_count": len(events),
        "weighted_event_coverage": round(weighted_coverage, 6),
        "quality_score": round(quality_score, 6),
        "duration_score": round(duration_score, 6),
        "total_score": round(total_score, 6),
        "wav_duration_s": round(wav_duration, 3),
        "expected_duration_s": round(duration_expected, 3),
        "global_rms": round(global_rms, 6),
        "global_peak": round(global_peak, 6),
        "clip_rate": round(global_clip, 8),
        "silence_ratio": round(global_silence, 6),
        "priority_miss_count": priority_miss_count,
    }

    return ranking_row, event_rows, repair_rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases-root", default="cases")
    ap.add_argument("--control-root", default="artifacts/model_runs/week17_control_baseline")
    ap.add_argument("--out-root", default="artifacts/model_race/week17_control_seed")
    ap.add_argument("--reports-root", default="reports")
    args = ap.parse_args()

    repo = Path.cwd()
    cases_root = repo / args.cases_root
    control_root = repo / args.control_root
    out_root = repo / args.out_root
    reports_root = repo / args.reports_root
    out_root.mkdir(parents=True, exist_ok=True)
    reports_root.mkdir(parents=True, exist_ok=True)

    ranking_rows: list[dict[str, Any]] = []
    event_rows_all: list[dict[str, Any]] = []
    repair_rows_all: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []

    case_dirs = sorted([p for p in cases_root.iterdir() if p.is_dir()]) if cases_root.exists() else []

    for case_dir in case_dirs:
        case_id = case_dir.name
        wav_path = control_root / case_id / f"{case_id}__control_rule_foley_v0.wav"
        if not wav_path.exists():
            missing_rows.append(
                {
                    "case_id": case_id,
                    "missing": str(wav_path),
                    "impact": "cannot rank control baseline candidate",
                }
            )
            continue

        ranking_row, event_rows, repair_rows = score_candidate(case_dir, wav_path)
        ranking_rows.append(ranking_row)
        event_rows_all.extend(event_rows)
        repair_rows_all.extend(repair_rows)

    ranking_rows = sorted(ranking_rows, key=lambda r: (r["case_id"], -float(r["total_score"])))

    winner_count = sum(1 for r in ranking_rows if r["status"] == "winner_seed")
    usable_count = sum(1 for r in ranking_rows if r["status"] in {"winner_seed", "usable_but_repair_recommended"})
    repair_case_count = len(set(r["case_id"] for r in repair_rows_all))
    missing_count = len(missing_rows)

    decision = "PASS_CONTROL_RANKING_SEED_READY"
    if missing_count > 0:
        decision = "FAIL_MISSING_CONTROL_CANDIDATES"
    elif usable_count < len(ranking_rows):
        decision = "PASS_WITH_REPAIR_QUEUE"

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "case_count": len(case_dirs),
        "ranked_candidate_count": len(ranking_rows),
        "winner_seed_count": winner_count,
        "usable_candidate_count": usable_count,
        "repair_case_count": repair_case_count,
        "repair_item_count": len(repair_rows_all),
        "missing_candidate_count": missing_count,
        "limitation": "This is event-window energy scoring for control-rule Foley only; it is not perceptual quality judgment and not V2A semantic validation.",
        "next_use": "Feed Java/Cloud as a model-race seed payload and compare MMAudio/FoleyCrafter candidates against this baseline later.",
    }

    write_csv(
        out_root / "control_seed_ranking.csv",
        ranking_rows,
        [
            "case_id",
            "candidate_id",
            "model_family",
            "candidate_path",
            "status",
            "selection_reason",
            "event_count",
            "weighted_event_coverage",
            "quality_score",
            "duration_score",
            "total_score",
            "wav_duration_s",
            "expected_duration_s",
            "global_rms",
            "global_peak",
            "clip_rate",
            "silence_ratio",
            "priority_miss_count",
        ],
    )
    write_json(out_root / "control_seed_ranking.json", {"summary": summary, "ranking": ranking_rows})

    write_csv(
        out_root / "event_window_scores.csv",
        event_rows_all,
        [
            "case_id",
            "candidate_id",
            "event_id",
            "layer_role",
            "priority",
            "time_s",
            "duration_s",
            "tolerance_ms",
            "window_start_s",
            "window_end_s",
            "event_rms",
            "global_rms",
            "energy_ratio",
            "threshold",
            "covered",
            "event_score",
        ],
    )

    write_csv(
        out_root / "repair_queue_seed.csv",
        repair_rows_all,
        [
            "case_id",
            "candidate_id",
            "event_id",
            "repair_bucket",
            "suggested_action",
            "priority",
            "event_rms",
            "threshold",
        ],
    )
    write_json(out_root / "repair_queue_seed.json", {"summary": summary, "repair_queue": repair_rows_all, "missing": missing_rows})

    write_json(reports_root / "week17_control_seed_summary.json", summary)

    java_payload = {
        "schema_version": "model_race_seed.v0.1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "mainbase",
        "decision": decision,
        "summary": summary,
        "results": [
            {
                "case_id": r["case_id"],
                "winner_candidate_id": r["candidate_id"],
                "winner_path": r["candidate_path"],
                "score": r["total_score"],
                "status": r["status"],
                "selection_reason": r["selection_reason"],
                "repair_required": r["status"] != "winner_seed",
            }
            for r in ranking_rows
        ],
    }
    write_json(reports_root / "week17_java_model_race_seed_payload.json", java_payload)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if decision.startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())