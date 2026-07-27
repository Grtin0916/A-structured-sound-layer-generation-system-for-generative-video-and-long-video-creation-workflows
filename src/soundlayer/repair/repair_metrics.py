"""Signal diagnostics used by the adaptive repair policy.

The implementation intentionally uses only the Python standard library so the
repair batch remains reproducible in the repository's thin runtime.
"""

from __future__ import annotations

from array import array
import math
from typing import Any, Iterable


def _frames(samples: array, channels: int) -> Iterable[tuple[int, ...]]:
    for offset in range(0, len(samples), channels):
        yield tuple(samples[offset:offset + channels])


def _rms(values: Iterable[int]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return math.sqrt(sum((value / 32768.0) ** 2 for value in values) / len(values))


def _dbfs(value: float) -> float:
    return 20.0 * math.log10(max(value, 1.0e-6))


def region_rms(
    samples: array, sample_rate: int, channels: int, start_sec: float, end_sec: float
) -> float:
    start = max(0, int(start_sec * sample_rate)) * channels
    end = min(len(samples), int(end_sec * sample_rate) * channels)
    return _rms(samples[start:end])


def edge_silence(
    samples: array, sample_rate: int, channels: int, threshold: float = 0.01
) -> tuple[float, float]:
    frame_list = list(_frames(samples, channels))
    active = [
        index
        for index, frame in enumerate(frame_list)
        if max((abs(value) / 32768.0 for value in frame), default=0.0) > threshold
    ]
    if not active:
        duration_ms = len(frame_list) / max(sample_rate, 1) * 1000.0
        return duration_ms, duration_ms
    leading_ms = active[0] / sample_rate * 1000.0
    trailing_ms = (len(frame_list) - active[-1] - 1) / sample_rate * 1000.0
    return leading_ms, trailing_ms


def flat_top_runs(samples: array, threshold: float = 0.995) -> list[int]:
    """Return contiguous near-ceiling run lengths measured in interleaved samples."""
    limit = int(threshold * 32767)
    runs: list[int] = []
    current = 0
    for value in samples:
        if abs(value) >= limit:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return runs


def diagnose_audio(
    samples: array,
    sample_rate: int,
    channels: int,
    event: dict[str, Any] | None = None,
    silence_threshold: float = 0.01,
    flat_top_threshold: float = 0.995,
) -> dict[str, Any]:
    if sample_rate <= 0 or channels <= 0:
        raise ValueError("sample_rate and channels must be positive")
    if any(not isinstance(value, int) for value in samples):
        raise ValueError("PCM16 samples must be integers")
    absolute = [abs(value) for value in samples]
    peak = max(absolute, default=0) / 32768.0
    rms = _rms(samples)
    leading_ms, trailing_ms = edge_silence(
        samples, sample_rate, channels, silence_threshold
    )
    runs = flat_top_runs(samples, flat_top_threshold)
    duration = len(samples) / (sample_rate * channels)
    event = event or {}
    event_start = max(0.0, float(event.get("start_sec", 0.0)))
    event_end = min(duration, float(event.get("end_sec", duration)))
    event_rms = region_rms(samples, sample_rate, channels, event_start, event_end)
    context_rms = _rms(samples)
    return {
        "duration_sec": duration,
        "sample_peak_abs": peak,
        "headroom_db": -20.0 * math.log10(max(peak, 1.0e-6)),
        "rms": rms,
        "rms_dbfs": _dbfs(rms),
        "silence_ratio": (
            sum(value <= int(silence_threshold * 32768) for value in absolute)
            / max(len(absolute), 1)
        ),
        "leading_silence_ms": leading_ms,
        "trailing_silence_ms": trailing_ms,
        "event_window_rms": event_rms,
        "event_context_ratio": event_rms / max(context_rms, 1.0e-6),
        "flat_top_sample_count": sum(runs),
        "flat_top_max_run": max(runs, default=0),
        "flat_top_run_count": len(runs),
        "clipped_ratio": sum(value >= 32760 for value in absolute) / max(len(absolute), 1),
    }


def detect_subtype(failure_type: str, metrics: dict[str, Any]) -> str:
    if failure_type == "clipping":
        longest = int(metrics["flat_top_max_run"])
        if longest <= 1:
            return "peak_near_ceiling"
        if longest <= 6:
            return "short_flat_top"
        return "long_flat_top"
    if failure_type in {"silence", "excessive_silence"}:
        edge_ms = max(metrics["leading_silence_ms"], metrics["trailing_silence_ms"])
        if edge_ms >= 80.0:
            return "leading_trailing_silence"
        return "weak_event_window"
    if failure_type == "layer_conflict_or_repairable":
        return "mixed_region_only"
    return "unsupported"


def action_compatibility(failure_type: str, subtype: str, action: str) -> tuple[bool, str]:
    allowed = {
        ("clipping", "peak_near_ceiling"): {"attenuate_limit", "peak_normalize_only"},
        ("clipping", "short_flat_top"): {"micro_declip", "attenuate_limit"},
        ("silence", "leading_trailing_silence"): {"trim"},
        ("silence", "weak_event_window"): {"event_local_gain"},
        ("excessive_silence", "leading_trailing_silence"): {"trim"},
        ("excessive_silence", "weak_event_window"): {"event_local_gain"},
        ("layer_conflict_or_repairable", "mixed_region_only"): {
            "mixed_region_attenuation"
        },
    }
    if subtype == "long_flat_top":
        return False, "long flat-top cannot be reconstructed deterministically"
    compatible = action in allowed.get((failure_type, subtype), set())
    return compatible, (
        "action matches detected signal pathology"
        if compatible
        else f"{action} does not target {subtype}"
    )
