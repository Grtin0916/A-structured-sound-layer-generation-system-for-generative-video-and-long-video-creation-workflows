"""Extract small, interpretable audio/DSS features without model dependencies."""

from __future__ import annotations

import array
import json
import math
import statistics
import wave
from pathlib import Path

FEATURE_NAMES = (
    "event_coverage",
    "priority_weighted_coverage",
    "onset_error_ms",
    "tolerance_violation_count",
    "event_window_energy",
    "outside_window_energy",
    "clip_ratio",
    "silence_ratio",
    "peak_abs",
    "duration_error_ms",
    "changed_sample_ratio",
    "repair_action_count",
    "source_reliability",
)


def load_pcm(path: Path):
    with wave.open(str(path), "rb") as handle:
        if handle.getsampwidth() != 2:
            raise ValueError(f"only 16-bit PCM is supported: {path}")
        channels = handle.getnchannels()
        rate = handle.getframerate()
        samples = array.array("h", handle.readframes(handle.getnframes()))
    if channels > 1:
        samples = array.array(
            "h",
            (
                round(sum(samples[index : index + channels]) / channels)
                for index in range(0, len(samples), channels)
            ),
        )
    return rate, [sample / 32768.0 for sample in samples]


def _rms(values):
    return math.sqrt(sum(value * value for value in values) / len(values)) if values else 0.0


def _frame_rms(samples, frame_size):
    return [
        _rms(samples[index : index + frame_size])
        for index in range(0, len(samples), frame_size)
    ]


def _onsets(samples, rate):
    frame_size = max(1, round(rate * 0.02))
    energy = _frame_rms(samples, frame_size)
    if not energy:
        return []
    baseline = statistics.median(energy)
    spread = statistics.pstdev(energy)
    threshold = max(0.003, baseline + 1.5 * spread)
    points = []
    active = False
    for index, value in enumerate(energy):
        if value >= threshold and not active:
            points.append(index * frame_size / rate)
            active = True
        elif value < threshold * 0.65:
            active = False
    return points


def extract_features(audio_path: Path, dss_path: Path, candidate: dict) -> dict:
    dss = json.loads(dss_path.read_text())
    rate, samples = load_pcm(audio_path)
    duration = len(samples) / rate if rate else 0.0
    target_duration = float(dss["video"]["duration_s"])
    events = [
        event for event in dss.get("events", []) if event.get("layer_role") != "ambience"
    ]
    onsets = _onsets(samples, rate)
    covered = []
    onset_errors = []
    priorities = []
    event_mask = [False] * len(samples)
    event_values = []
    for event in events:
        start = float(event["time_s"])
        end = start + float(event.get("duration_s", 0.0))
        tolerance_ms = float(event.get("tolerance_ms", 250))
        error_ms = (
            min(abs(onset - start) for onset in onsets) * 1000.0
            if onsets
            else target_duration * 1000.0
        )
        is_covered = error_ms <= tolerance_ms
        covered.append(is_covered)
        onset_errors.append(error_ms)
        priorities.append(float(event.get("priority", 1)))
        lo = max(0, min(len(samples), round(start * rate)))
        hi = max(lo, min(len(samples), round(end * rate)))
        event_values.extend(samples[lo:hi])
        for index in range(lo, hi):
            event_mask[index] = True
    outside_values = [
        value for index, value in enumerate(samples) if not event_mask[index]
    ]
    abs_values = [abs(value) for value in samples]
    metrics = candidate.get("proxy_metrics", {})
    role = candidate.get("candidate_role", "")
    return {
        "event_coverage": sum(covered) / len(covered) if covered else 0.0,
        "priority_weighted_coverage": (
            sum(priority for priority, ok in zip(priorities, covered) if ok)
            / sum(priorities)
            if priorities
            else 0.0
        ),
        "onset_error_ms": statistics.mean(onset_errors)
        if onset_errors
        else target_duration * 1000.0,
        "tolerance_violation_count": sum(not value for value in covered),
        "event_window_energy": _rms(event_values),
        "outside_window_energy": _rms(outside_values),
        "clip_ratio": sum(value >= 0.999 for value in abs_values) / len(abs_values)
        if abs_values
        else 0.0,
        "silence_ratio": sum(value < 0.001 for value in abs_values) / len(abs_values)
        if abs_values
        else 1.0,
        "peak_abs": max(abs_values, default=0.0),
        "duration_error_ms": abs(duration - target_duration) * 1000.0,
        "changed_sample_ratio": float(metrics.get("changed_sample_ratio", 0.0)),
        "repair_action_count": 1.0
        if "REPAIR" in role or candidate.get("strategy_id") == "D"
        else 0.0,
        "source_reliability": 1.0
        if candidate.get("ablation_materialized", False)
        else 0.8,
    }


def difference(left: dict, right: dict) -> dict:
    return {name: float(left[name]) - float(right[name]) for name in FEATURE_NAMES}
