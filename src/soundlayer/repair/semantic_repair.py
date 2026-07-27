"""Capability-aware metrics and mixed-only semantic repair helpers."""

from __future__ import annotations

from array import array
import math
from typing import Any

from .repair_metrics import region_rms
from .repair_search import edit_cost
from .signal_repair import smooth_region_gain


def classify_source_mode(stem_paths: list[str], source_path: str) -> tuple[str, str]:
    unique = {path for path in stem_paths if path and path != source_path}
    if len(unique) >= 2:
        return "true_stems", "two or more distinct stem paths declared; runtime validation still required"
    return "mixed_only", "no independently traceable stem set"


def boundary_jump(samples: array, sample_rate: int, channels: int, seconds: float) -> float:
    frame = round(seconds * sample_rate)
    if frame <= 0 or frame >= len(samples) // channels:
        return 0.0
    jumps = [
        abs(samples[frame * channels + channel] - samples[(frame - 1) * channels + channel])
        / 32768.0
        for channel in range(channels)
    ]
    return max(jumps, default=0.0)


def outside_window_rms(
    samples: array, sample_rate: int, channels: int, start_sec: float, end_sec: float
) -> float:
    start = max(0, round(start_sec * sample_rate)) * channels
    end = min(len(samples), round(end_sec * sample_rate) * channels)
    return math.sqrt(
        sum((value / 32768.0) ** 2 for value in samples[:start] + samples[end:])
        / max(len(samples[:start]) + len(samples[end:]), 1)
    )


def db_delta(after: float, before: float) -> float:
    return 20.0 * math.log10(max(after, 1.0e-6) / max(before, 1.0e-6))


def apply_mixed_only_repair(
    samples: array,
    sample_rate: int,
    channels: int,
    start_sec: float,
    end_sec: float,
    intent: str,
) -> tuple[array, dict[str, Any]]:
    if intent == "reduce_masking":
        gain = 0.82
    elif intent == "strengthen_expected_event":
        gain = 1.20
    else:
        raise ValueError(f"unsupported semantic intent: {intent}")
    output, metadata = smooth_region_gain(
        samples, sample_rate, channels, start_sec, end_sec, gain, fade_ms=50
    )
    before_target = region_rms(samples, sample_rate, channels, start_sec, end_sec)
    after_target = region_rms(output, sample_rate, channels, start_sec, end_sec)
    before_outside = outside_window_rms(samples, sample_rate, channels, start_sec, end_sec)
    after_outside = outside_window_rms(output, sample_rate, channels, start_sec, end_sec)
    changed_ratio, mean_delta = edit_cost(samples, output)
    metadata.update({
        "target_window_rms_delta_db": db_delta(after_target, before_target),
        "outside_window_rms_delta_db": db_delta(after_outside, before_outside),
        "boundary_jump_before": max(
            boundary_jump(samples, sample_rate, channels, start_sec),
            boundary_jump(samples, sample_rate, channels, end_sec),
        ),
        "boundary_jump_after": max(
            boundary_jump(output, sample_rate, channels, start_sec),
            boundary_jump(output, sample_rate, channels, end_sec),
        ),
        "changed_sample_ratio": changed_ratio,
        "edit_cost": changed_ratio + mean_delta,
        "semantic_intent": intent,
    })
    return output, metadata


def semantic_gate(metadata: dict[str, Any], peak_after: float, duration_error_ms: float) -> tuple[bool, str]:
    direction_ok = (
        metadata["target_window_rms_delta_db"] < -0.1
        if metadata["semantic_intent"] == "reduce_masking"
        else metadata["target_window_rms_delta_db"] > 0.1
    )
    guards_ok = (
        abs(metadata["outside_window_rms_delta_db"]) <= 0.05
        and peak_after < 0.999
        and abs(duration_error_ms) <= 20.0
        and metadata["boundary_jump_after"] <= metadata["boundary_jump_before"] + 0.02
    )
    if not direction_ok:
        return False, "semantic target metric did not move in the intended direction"
    if not guards_ok:
        return False, "one or more acoustic guards failed"
    return True, "proxy target and acoustic guards passed; human semantic review still required"
