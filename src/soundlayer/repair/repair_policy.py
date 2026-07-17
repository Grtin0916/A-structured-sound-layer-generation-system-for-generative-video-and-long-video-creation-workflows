"""Compile diagnosed audio failures into bounded deterministic repair plans."""

from __future__ import annotations

from typing import Any


POLICIES: dict[str, dict[str, Any]] = {
    "clipping": {
        "action": "attenuate_limit",
        "parameters": {"gain": 0.88, "peak_ceiling": 0.95},
        "target_metric": "peak_abs",
        "target_direction": "decrease",
        "guard_metrics": ["duration_sec", "rms_dbfs", "silence_ratio"],
        "max_regression": {"duration_ratio": 0.02, "rms_db": 3.0},
        "fallback_action": "peak_normalize_only",
        "requires_stems": False,
    },
    "silence": {
        "action": "trim",
        "parameters": {"silence_threshold": 0.01, "padding_ms": 80},
        "target_metric": "silence_ratio",
        "target_direction": "decrease",
        "guard_metrics": ["duration_sec", "peak_abs", "rms_dbfs"],
        "max_regression": {"duration_ratio": 0.30, "rms_db": 2.0},
        "fallback_action": "event_local_gain",
        "requires_stems": False,
    },
    "excessive_silence": {
        "action": "trim",
        "parameters": {"silence_threshold": 0.01, "padding_ms": 80},
        "target_metric": "silence_ratio",
        "target_direction": "decrease",
        "guard_metrics": ["duration_sec", "peak_abs", "rms_dbfs"],
        "max_regression": {"duration_ratio": 0.30, "rms_db": 2.0},
        "fallback_action": "event_local_gain",
        "requires_stems": False,
    },
    "layer_conflict_or_repairable": {
        "action": "mixed_region_attenuation",
        "parameters": {"gain": 0.82, "fade_ms": 30},
        "target_metric": "region_rms",
        "target_direction": "decrease",
        "guard_metrics": ["duration_sec", "peak_abs", "outside_region_rms"],
        "max_regression": {"duration_ratio": 0.02, "outside_region_rms_db": 1.0},
        "fallback_action": "candidate_replace",
        "requires_stems": False,
    },
    "naive_less_controllable": {
        "action": "candidate_replace",
        "parameters": {"replacement_family": "dss"},
        "target_metric": "event_coverage",
        "target_direction": "increase",
        "guard_metrics": ["forbidden_leakage", "duration_sec", "peak_abs"],
        "max_regression": {"forbidden_leakage": 0.0},
        "fallback_action": "manual_review",
        "requires_stems": False,
        "deterministic_execution_supported": False,
    },
    "onset_late": {
        "action": "shift_left",
        "parameters": {"max_shift_ms": 500},
        "target_metric": "absolute_onset_error_ms",
        "target_direction": "decrease",
        "guard_metrics": ["duration_sec", "event_order", "peak_abs"],
        "max_regression": {"duration_ratio": 0.02, "event_order_violations": 0},
        "fallback_action": "manual_review",
        "requires_stems": False,
    },
    "onset_early": {
        "action": "delay_or_pad",
        "parameters": {"max_delay_ms": 500},
        "target_metric": "absolute_onset_error_ms",
        "target_direction": "decrease",
        "guard_metrics": ["duration_sec", "event_order", "peak_abs"],
        "max_regression": {"duration_ratio": 0.02, "event_order_violations": 0},
        "fallback_action": "manual_review",
        "requires_stems": False,
    },
    "event_missing": {
        "action": "candidate_replace",
        "parameters": {"replacement_family": "dss"},
        "target_metric": "event_coverage",
        "target_direction": "increase",
        "guard_metrics": ["forbidden_leakage", "duration_sec", "peak_abs"],
        "max_regression": {"forbidden_leakage": 0.0},
        "fallback_action": "manual_review",
        "requires_stems": False,
        "deterministic_execution_supported": False,
    },
    "insertion_hallucination": {
        "action": "mixed_region_attenuation",
        "parameters": {"gain": 0.75, "fade_ms": 30},
        "target_metric": "unexpected_region_rms",
        "target_direction": "decrease",
        "guard_metrics": ["duration_sec", "peak_abs", "outside_region_rms"],
        "max_regression": {"duration_ratio": 0.02, "outside_region_rms_db": 1.0},
        "fallback_action": "candidate_replace",
        "requires_stems": False,
    },
}


def compile_policy(failure_type: str, has_stems: bool = False) -> dict[str, Any] | None:
    policy = POLICIES.get(failure_type)
    if policy is None:
        return None
    result = dict(policy)
    result["parameters"] = dict(policy["parameters"])
    result["guard_metrics"] = list(policy["guard_metrics"])
    result["max_regression"] = dict(policy["max_regression"])
    result.setdefault("deterministic_execution_supported", True)
    if result["requires_stems"] and not has_stems:
        result["blocked_reason"] = "action_requires_stems"
        result["execution_ready"] = False
    elif not result["deterministic_execution_supported"]:
        result["blocked_reason"] = "deterministic_action_not_available"
        result["execution_ready"] = False
    else:
        result["blocked_reason"] = ""
        result["execution_ready"] = True
    return result
