"""Candidate enumeration and constrained minimal-edit selection."""

from __future__ import annotations

from array import array
import math
from typing import Any


def enumerate_variants(plan: dict[str, Any], diagnosis: dict[str, Any]) -> list[dict[str, Any]]:
    subtype = diagnosis["detected_subtype"]
    if subtype == "peak_near_ceiling":
        return [{"action": "attenuate_limit", "ceiling": value} for value in (0.98, 0.95, 0.92)]
    if subtype == "short_flat_top":
        return [{"action": "micro_declip", "threshold": value, "max_run": 6} for value in (0.999, 0.997, 0.995)]
    if subtype == "leading_trailing_silence":
        return [
            {"action": "trim", "silence_threshold": threshold, "padding_ms": padding}
            for threshold, padding in ((0.005, 120), (0.01, 80), (0.02, 50))
        ]
    if subtype == "weak_event_window":
        return [
            {"action": "event_local_gain", "gain": gain, "fade_ms": 30}
            for gain in (1.10, 1.20, 1.35)
        ]
    if subtype == "mixed_region_only":
        return [
            {"action": "mixed_region_attenuation", "gain": gain, "fade_ms": fade}
            for gain, fade in ((0.90, 40), (0.82, 30), (0.75, 50))
        ]
    return []


def edit_cost(before: array, after: array) -> tuple[float, float]:
    compared = min(len(before), len(after))
    changed = sum(before[index] != after[index] for index in range(compared))
    changed += abs(len(before) - len(after))
    denominator = max(len(before), len(after), 1)
    mean_delta = (
        sum(abs(before[index] - after[index]) for index in range(compared))
        / max(compared, 1)
        / 32768.0
    )
    return changed / denominator, mean_delta


def select_minimal_valid_edit(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    feasible = [
        row
        for row in rows
        if row["target_improved"]
        and not row["severe_regression"]
        and row["output_readable"]
        and row["lineage_complete"]
        and row["action_metric_compatible"]
        and math.isfinite(float(row["edit_cost"]))
    ]
    return min(feasible, key=lambda row: (row["edit_cost"], row["variant_index"])) if feasible else None
