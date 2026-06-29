#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Repair Week17 control-rule Foley candidates from repair_queue_seed.

Current repair actions:
- priority_event_weak_or_missing:
  add event-region pulse/noise reinforcement around DSS event window
- clip_risk:
  peak normalize
- high_silence_ratio:
  add low-level ambience bed

Outputs:
- artifacts/model_runs/week17_control_repaired/<case_id>/<candidate>__repair_v1.wav
- artifacts/model_race/week17_repair_seed/repair_before_after.csv/json
- reports/week17_control_repair_summary.json
- reports/week17_java_repair_seed_payload.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import struct
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SAMPLE_RATE_FALLBACK = 48000
MAX_INT16 = 32767


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


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_wav_mono(path: Path) -> tuple[list[float], int]:
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

    return samples, sr


def write_wav(path: Path, samples: list[float], sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    peak = max((abs(x) for x in samples), default=1e-9)
    scale = 0.95 / peak if peak > 0.95 else 1.0

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        frames = bytearray()
        for x in samples:
            y = max(-1.0, min(1.0, x * scale))
            frames.extend(struct.pack("<h", int(y * MAX_INT16)))
        wf.writeframes(frames)


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


def event_threshold(layer_role: str, priority: int) -> float:
    if layer_role == "ambience":
        return 0.004
    if layer_role in {"foley", "fx"}:
        return max(0.010, 0.006 + priority * 0.002)
    return 0.008


def slice_samples(samples: list[float], sr: int, start_s: float, end_s: float) -> list[float]:
    start = max(0, min(len(samples), int(start_s * sr)))
    end = max(start, min(len(samples), int(end_s * sr)))
    return samples[start:end]


def find_event(dss: dict[str, Any], event_id: str) -> dict[str, Any] | None:
    for ev in dss.get("events", []):
        if ev.get("event_id") == event_id:
            return ev
    return None


def add_noise_burst(samples: list[float], sr: int, start_s: float, dur_s: float, amp: float, seed: int) -> None:
    rng = random.Random(seed)
    start = max(0, min(len(samples), int(start_s * sr)))
    n = max(1, int(dur_s * sr))
    end = min(len(samples), start + n)

    for i in range(start, end):
        t = (i - start) / sr
        attack = min(1.0, t / 0.02)
        release = min(1.0, max(0.0, (dur_s - t) / 0.06))
        env = max(0.0, min(attack, release))
        samples[i] += amp * env * rng.uniform(-1.0, 1.0)


def add_tonal_click(samples: list[float], sr: int, start_s: float, dur_s: float, freq: float, amp: float) -> None:
    start = max(0, min(len(samples), int(start_s * sr)))
    n = max(1, int(dur_s * sr))
    end = min(len(samples), start + n)

    for i in range(start, end):
        t = (i - start) / sr
        env = math.exp(-18.0 * t)
        samples[i] += amp * env * math.sin(2.0 * math.pi * freq * t)


def reinforce_event(samples: list[float], sr: int, dss: dict[str, Any], ev: dict[str, Any]) -> dict[str, Any]:
    event_id = str(ev.get("event_id", ""))
    time_s = float(ev.get("time_s", 0.0))
    duration_s = max(0.1, float(ev.get("duration_s", 0.3)))
    priority = int(ev.get("priority", 3))
    layer_role = str(ev.get("layer_role", "foley"))
    threshold = event_threshold(layer_role, priority)

    # For rhythmic knife/footstep-like events, add multiple short transients.
    if "chop" in event_id or "knife" in event_id or "footstep" in event_id:
        pulses = max(4, int(duration_s * 5))
        step = duration_s / pulses
        for j in range(pulses):
            t = time_s + j * step
            add_noise_burst(samples, sr, t, min(0.07, step * 0.55), amp=0.30 + 0.04 * priority, seed=9000 + j)
            add_tonal_click(samples, sr, t, min(0.08, step * 0.60), freq=260.0 + 20.0 * j, amp=0.12)
        action = f"added {pulses} transient pulses in event window"
    else:
        add_noise_burst(samples, sr, time_s, duration_s, amp=0.22 + 0.04 * priority, seed=777)
        action = "added event-region noise reinforcement"

    # Make local event region stronger if still too weak.
    win_start = max(0.0, time_s - 0.10)
    win_end = time_s + duration_s + 0.10
    region = slice_samples(samples, sr, win_start, win_end)
    region_rms = rms(region)
    target = max(threshold * 1.75, 0.035)

    if region_rms < target and region_rms > 1e-8:
        gain = min(3.0, target / region_rms)
        start = max(0, int(win_start * sr))
        end = min(len(samples), int(win_end * sr))
        for i in range(start, end):
            samples[i] *= gain
        action += f"; local gain x{gain:.2f}"
    elif region_rms <= 1e-8:
        action += "; local gain skipped due to zero rms"

    return {
        "event_id": event_id,
        "repair_action": action,
        "target_event_threshold": round(threshold, 6),
    }


def normalize_peak(samples: list[float], ceiling: float = 0.92) -> None:
    peak = max((abs(x) for x in samples), default=0.0)
    if peak <= ceiling or peak <= 1e-9:
        return
    scale = ceiling / peak
    for i, x in enumerate(samples):
        samples[i] = x * scale


def add_ambience_bed(samples: list[float], amp: float = 0.006, seed: int = 1234) -> None:
    rng = random.Random(seed)
    for i in range(len(samples)):
        samples[i] += amp * rng.uniform(-1.0, 1.0)


def score_candidate(case_dir: Path, wav_path: Path) -> dict[str, Any]:
    dss = read_json(case_dir / "director_sound_script.yaml")
    samples, sr = read_wav_mono(wav_path)
    wav_duration = len(samples) / sr if sr else 0.0

    global_rms = rms(samples)
    global_peak = max((abs(x) for x in samples), default=0.0)
    global_clip = clip_rate(samples)
    global_silence = silence_ratio(samples)

    weighted_hit = 0.0
    weighted_total = 0.0
    priority_miss_count = 0

    for ev in dss.get("events", []):
        layer_role = str(ev.get("layer_role", ""))
        priority = int(ev.get("priority", 3))
        time_s = float(ev.get("time_s", 0.0))
        duration_s = float(ev.get("duration_s", 0.3))
        tolerance_ms = int(ev.get("tolerance_ms", 200))

        pad = max(0.05, min(0.25, tolerance_ms / 1000.0))
        event_samples = slice_samples(samples, sr, max(0.0, time_s - pad), min(wav_duration, time_s + duration_s + pad))
        event_rms = rms(event_samples)
        energy_ratio = event_rms / max(global_rms, 1e-8)
        threshold = event_threshold(layer_role, priority)

        if layer_role == "ambience":
            covered = event_rms >= threshold and global_silence < 0.40
        else:
            covered = event_rms >= threshold and energy_ratio >= 0.65

        weighted_hit += priority * (1.0 if covered else 0.0)
        weighted_total += priority

        if not covered and priority >= 4:
            priority_miss_count += 1

    weighted_coverage = weighted_hit / weighted_total if weighted_total else 0.0
    expected_duration = float(dss.get("video", {}).get("duration_s", wav_duration))
    duration_score = max(0.0, 1.0 - abs(wav_duration - expected_duration) / max(expected_duration, 1e-6))

    quality_score = 1.0
    quality_score -= min(0.35, global_clip * 200.0)
    quality_score -= min(0.30, max(0.0, global_silence - 0.35))
    quality_score -= min(0.20, max(0.0, 0.015 - global_rms) * 5.0)
    quality_score = max(0.0, min(1.0, quality_score))

    total_score = 0.65 * weighted_coverage + 0.25 * quality_score + 0.10 * duration_score

    status = "winner_seed" if priority_miss_count == 0 and total_score >= 0.75 else "repair_required"

    return {
        "candidate_path": str(wav_path),
        "status": status,
        "weighted_event_coverage": round(weighted_coverage, 6),
        "quality_score": round(quality_score, 6),
        "duration_score": round(duration_score, 6),
        "total_score": round(total_score, 6),
        "global_rms": round(global_rms, 6),
        "global_peak": round(global_peak, 6),
        "clip_rate": round(global_clip, 8),
        "silence_ratio": round(global_silence, 6),
        "priority_miss_count": priority_miss_count,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases-root", default="cases")
    ap.add_argument("--control-root", default="artifacts/model_runs/week17_control_baseline")
    ap.add_argument("--repair-queue", default="artifacts/model_race/week17_control_seed/repair_queue_seed.csv")
    ap.add_argument("--out-root", default="artifacts/model_runs/week17_control_repaired")
    ap.add_argument("--race-out-root", default="artifacts/model_race/week17_repair_seed")
    ap.add_argument("--reports-root", default="reports")
    args = ap.parse_args()

    repo = Path.cwd()
    cases_root = repo / args.cases_root
    control_root = repo / args.control_root
    out_root = repo / args.out_root
    race_out_root = repo / args.race_out_root
    reports_root = repo / args.reports_root

    queue_rows = read_csv(repo / args.repair_queue)
    event_queue = [r for r in queue_rows if r.get("repair_bucket") == "priority_event_weak_or_missing"]

    before_after_rows: list[dict[str, Any]] = []
    java_results: list[dict[str, Any]] = []
    repaired_case_ids = sorted(set(r["case_id"] for r in event_queue if r.get("case_id")))

    for case_id in repaired_case_ids:
        case_dir = cases_root / case_id
        dss_path = case_dir / "director_sound_script.yaml"
        src_wav = control_root / case_id / f"{case_id}__control_rule_foley_v0.wav"
        dst_wav = out_root / case_id / f"{case_id}__control_rule_foley_v0_repair_v1.wav"

        if not dss_path.exists() or not src_wav.exists():
            before_after_rows.append(
                {
                    "case_id": case_id,
                    "repair_status": "failed_missing_input",
                    "before_score": "",
                    "after_score": "",
                    "score_delta": "",
                    "before_status": "",
                    "after_status": "",
                    "repaired_path": str(dst_wav),
                    "repair_actions": "missing dss or source wav",
                }
            )
            continue

        dss = read_json(dss_path)
        samples, sr = read_wav_mono(src_wav)
        before = score_candidate(case_dir, src_wav)

        actions = []
        for row in event_queue:
            if row.get("case_id") != case_id:
                continue
            ev = find_event(dss, row.get("event_id", ""))
            if ev is None:
                actions.append({"event_id": row.get("event_id", ""), "repair_action": "event not found"})
                continue
            actions.append(reinforce_event(samples, sr, dss, ev))

        if silence_ratio(samples) > 0.35:
            add_ambience_bed(samples)

        normalize_peak(samples, ceiling=0.92)
        write_wav(dst_wav, samples, sr)

        after = score_candidate(case_dir, dst_wav)
        score_delta = float(after["total_score"]) - float(before["total_score"])
        repair_status = "repaired_to_winner" if after["status"] == "winner_seed" else "repair_attempted_still_required"

        before_after_rows.append(
            {
                "case_id": case_id,
                "repair_status": repair_status,
                "before_score": before["total_score"],
                "after_score": after["total_score"],
                "score_delta": round(score_delta, 6),
                "before_status": before["status"],
                "after_status": after["status"],
                "before_coverage": before["weighted_event_coverage"],
                "after_coverage": after["weighted_event_coverage"],
                "before_priority_miss_count": before["priority_miss_count"],
                "after_priority_miss_count": after["priority_miss_count"],
                "before_rms": before["global_rms"],
                "after_rms": after["global_rms"],
                "before_peak": before["global_peak"],
                "after_peak": after["global_peak"],
                "before_clip_rate": before["clip_rate"],
                "after_clip_rate": after["clip_rate"],
                "before_silence_ratio": before["silence_ratio"],
                "after_silence_ratio": after["silence_ratio"],
                "repaired_path": str(dst_wav),
                "repair_actions": json.dumps(actions, ensure_ascii=False),
            }
        )

        java_results.append(
            {
                "case_id": case_id,
                "source_candidate_id": f"{case_id}__control_rule_foley_v0",
                "repaired_candidate_id": f"{case_id}__control_rule_foley_v0_repair_v1",
                "repaired_path": str(dst_wav),
                "repair_status": repair_status,
                "before_score": before["total_score"],
                "after_score": after["total_score"],
                "score_delta": round(score_delta, 6),
                "winner_after_repair": after["status"] == "winner_seed",
            }
        )

    repaired_to_winner_count = sum(1 for r in before_after_rows if r.get("repair_status") == "repaired_to_winner")
    still_required_count = sum(1 for r in before_after_rows if r.get("repair_status") == "repair_attempted_still_required")
    failed_count = sum(1 for r in before_after_rows if str(r.get("repair_status", "")).startswith("failed"))

    decision = "PASS_REPAIR_APPLIED"
    if failed_count > 0:
        decision = "FAIL_REPAIR_INPUT_MISSING"
    elif still_required_count > 0:
        decision = "PASS_REPAIR_ATTEMPTED_WITH_REMAINING_QUEUE"
    elif repaired_to_winner_count == len(repaired_case_ids) and repaired_case_ids:
        decision = "PASS_REPAIR_CLOSED_QUEUE"

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "input_repair_item_count": len(event_queue),
        "repaired_case_count": len(repaired_case_ids),
        "repaired_to_winner_count": repaired_to_winner_count,
        "still_required_count": still_required_count,
        "failed_count": failed_count,
        "limitation": "Repair is deterministic event-region reinforcement for control baseline only; it is not a generative model quality claim.",
        "before_after": before_after_rows,
    }

    write_csv(
        race_out_root / "repair_before_after.csv",
        before_after_rows,
        [
            "case_id",
            "repair_status",
            "before_score",
            "after_score",
            "score_delta",
            "before_status",
            "after_status",
            "before_coverage",
            "after_coverage",
            "before_priority_miss_count",
            "after_priority_miss_count",
            "before_rms",
            "after_rms",
            "before_peak",
            "after_peak",
            "before_clip_rate",
            "after_clip_rate",
            "before_silence_ratio",
            "after_silence_ratio",
            "repaired_path",
            "repair_actions",
        ],
    )
    write_json(race_out_root / "repair_before_after.json", summary)
    write_json(reports_root / "week17_control_repair_summary.json", summary)

    java_payload = {
        "schema_version": "repair_seed.v0.1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "mainbase",
        "decision": decision,
        "summary": {k: summary[k] for k in summary if k != "before_after"},
        "results": java_results,
    }
    write_json(reports_root / "week17_java_repair_seed_payload.json", java_payload)

    print(json.dumps({k: summary[k] for k in summary if k != "before_after"}, ensure_ascii=False, indent=2))
    return 0 if decision.startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())