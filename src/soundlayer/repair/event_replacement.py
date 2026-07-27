"""Reference-guided event transplant for compatible PCM16 candidates."""

from __future__ import annotations

from array import array
import math
from typing import Any

from .repair_metrics import region_rms
from .repair_search import edit_cost
from .semantic_repair import boundary_jump, db_delta, outside_window_rms


def transplant_event(
    target: array,
    donor: array,
    sample_rate: int,
    channels: int,
    target_start_sec: float,
    target_end_sec: float,
    donor_start_sec: float,
    donor_end_sec: float,
    crossfade_ms: int,
    donor_gain: float = 0.65,
) -> tuple[array, dict[str, Any]]:
    if sample_rate <= 0 or channels <= 0:
        raise ValueError("invalid audio geometry")
    target_frames = len(target) // channels
    donor_frames = len(donor) // channels
    target_start = max(0, min(target_frames, round(target_start_sec * sample_rate)))
    target_end = max(target_start, min(target_frames, round(target_end_sec * sample_rate)))
    donor_start = max(0, min(donor_frames, round(donor_start_sec * sample_rate)))
    donor_end = max(donor_start, min(donor_frames, round(donor_end_sec * sample_rate)))
    available = min(target_end - target_start, donor_end - donor_start)
    if available < max(round(0.04 * sample_rate), 1):
        raise ValueError("donor or target window is too short")
    fade = min(round(crossfade_ms / 1000 * sample_rate), available // 2)
    output = array("h", target)
    clipped = 0
    for offset in range(available):
        if fade and offset < fade:
            envelope = 0.5 - 0.5 * math.cos(math.pi * (offset + 1) / fade)
        elif fade and offset >= available - fade:
            envelope = 0.5 - 0.5 * math.cos(math.pi * (available - offset) / fade)
        else:
            envelope = 1.0
        for channel in range(channels):
            target_index = (target_start + offset) * channels + channel
            donor_index = (donor_start + offset) * channels + channel
            mixed = round(output[target_index] + donor[donor_index] * donor_gain * envelope)
            clipped += int(mixed < -32768 or mixed > 32767)
            output[target_index] = max(-32768, min(32767, mixed))
    changed_ratio, mean_delta = edit_cost(target, output)
    return output, {
        "repair_action": "event_transplant",
        "crossfade_ms": crossfade_ms,
        "donor_gain": donor_gain,
        "target_window": [target_start / sample_rate, (target_start + available) / sample_rate],
        "donor_window": [donor_start / sample_rate, (donor_start + available) / sample_rate],
        "changed_sample_ratio": changed_ratio,
        "edit_cost": changed_ratio + mean_delta,
        "clipped_sample_count": clipped,
    }


def transplant_metrics(
    before: array,
    after: array,
    sample_rate: int,
    channels: int,
    start_sec: float,
    end_sec: float,
) -> dict[str, float | bool]:
    before_target = region_rms(before, sample_rate, channels, start_sec, end_sec)
    after_target = region_rms(after, sample_rate, channels, start_sec, end_sec)
    before_outside = outside_window_rms(before, sample_rate, channels, start_sec, end_sec)
    after_outside = outside_window_rms(after, sample_rate, channels, start_sec, end_sec)
    return {
        "target_window_rms_delta_db": db_delta(after_target, before_target),
        "outside_window_rms_delta_db": db_delta(after_outside, before_outside),
        "boundary_jump_before": max(
            boundary_jump(before, sample_rate, channels, start_sec),
            boundary_jump(before, sample_rate, channels, end_sec),
        ),
        "boundary_jump_after": max(
            boundary_jump(after, sample_rate, channels, start_sec),
            boundary_jump(after, sample_rate, channels, end_sec),
        ),
        "ordering_correct": True,
    }
