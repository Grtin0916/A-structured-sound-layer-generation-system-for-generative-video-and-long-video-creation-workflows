"""Bounded PCM16 repair primitives with explicit capability limits."""

from __future__ import annotations

from array import array
import math


def _validate(samples: array) -> None:
    if samples.typecode != "h":
        raise ValueError("expected signed PCM16 array")


def adaptive_headroom(samples: array, ceiling: float = 0.95) -> tuple[array, dict]:
    _validate(samples)
    if not 0 < ceiling <= 1:
        raise ValueError("ceiling must be in (0, 1]")
    observed = max((abs(value) / 32768.0 for value in samples), default=0.0)
    gain = min(1.0, ceiling / max(observed, 1.0e-12))
    output = array("h", (round(value * gain) for value in samples))
    return output, {
        "repair_kind": "HEADROOM_ONLY",
        "gain": gain,
        "recovered_clipped_waveform": False,
    }


def conservative_micro_declip(
    samples: array, threshold: float = 0.995, max_run: int = 6
) -> tuple[array, dict]:
    """Interpolate only tiny flat-top runs; this is a probe, not waveform recovery."""
    _validate(samples)
    limit = int(threshold * 32767)
    output = array("h", samples)
    changed = 0
    blocked_runs = 0
    index = 0
    while index < len(output):
        if abs(output[index]) < limit:
            index += 1
            continue
        start = index
        while index < len(output) and abs(output[index]) >= limit:
            index += 1
        end = index
        run = end - start
        if run > max_run or start == 0 or end >= len(output):
            blocked_runs += 1
            continue
        left, right = output[start - 1], output[end]
        for offset in range(run):
            value = round(left + (right - left) * ((offset + 1) / (run + 1)))
            if value != output[start + offset]:
                output[start + offset] = value
                changed += 1
    return output, {
        "repair_kind": "MICRO_DECLIP_LINEAR_PROBE",
        "changed_sample_ratio": changed / max(len(output), 1),
        "blocked_long_runs": blocked_runs,
        "recovered_clipped_waveform": False,
    }


def silence_aware_trim(
    samples: array,
    sample_rate: int,
    channels: int,
    silence_threshold: float = 0.01,
    padding_ms: int = 80,
) -> tuple[array, dict]:
    _validate(samples)
    if sample_rate <= 0 or channels <= 0:
        raise ValueError("invalid audio geometry")
    threshold = int(silence_threshold * 32768)
    active_frames = [
        index // channels
        for index, value in enumerate(samples)
        if abs(value) > threshold
    ]
    if not active_frames:
        return array("h", samples), {
            "repair_kind": "SILENCE_AWARE_TRIM",
            "trimmed_leading_frames": 0,
            "trimmed_trailing_frames": 0,
            "all_silent_preserved": True,
        }
    padding = round(padding_ms / 1000 * sample_rate)
    frame_count = len(samples) // channels
    first = max(min(active_frames) - padding, 0)
    last = min(max(active_frames) + padding + 1, frame_count)
    return array("h", samples[first * channels:last * channels]), {
        "repair_kind": "SILENCE_AWARE_TRIM",
        "trimmed_leading_frames": first,
        "trimmed_trailing_frames": frame_count - last,
        "all_silent_preserved": False,
    }


def smooth_region_gain(
    samples: array,
    sample_rate: int,
    channels: int,
    start_sec: float,
    end_sec: float,
    gain: float,
    fade_ms: int = 30,
) -> tuple[array, dict]:
    _validate(samples)
    if not 0 < gain <= 2:
        raise ValueError("gain must be in (0, 2]")
    frame_count = len(samples) // channels
    start = max(0, min(frame_count, round(start_sec * sample_rate)))
    end = max(start, min(frame_count, round(end_sec * sample_rate)))
    fade = min(round(fade_ms / 1000 * sample_rate), max((end - start) // 2, 0))
    output = array("h", samples)
    for frame in range(start, end):
        if fade and frame < start + fade:
            progress = (frame - start + 1) / fade
            envelope = 1.0 + (gain - 1.0) * (0.5 - 0.5 * math.cos(math.pi * progress))
        elif fade and frame >= end - fade:
            progress = (end - frame) / fade
            envelope = 1.0 + (gain - 1.0) * (0.5 - 0.5 * math.cos(math.pi * progress))
        else:
            envelope = gain
        for channel in range(channels):
            index = frame * channels + channel
            output[index] = max(-32768, min(32767, round(output[index] * envelope)))
    changed = sum(a != b for a, b in zip(samples, output))
    return output, {
        "repair_kind": "SMOOTH_REGION_GAIN",
        "changed_sample_ratio": changed / max(len(output), 1),
        "start_sec": start / sample_rate,
        "end_sec": end / sample_rate,
        "fade_ms": fade_ms,
    }
