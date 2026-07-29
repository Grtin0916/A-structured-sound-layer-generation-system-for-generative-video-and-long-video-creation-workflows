"""Fixed DSS interactions and selective hybrid decision semantics."""

from __future__ import annotations

DSS_INTERACTION_NAMES = (
    "priority_coverage_x_event_energy",
    "priority_coverage_x_inverse_onset",
    "tolerance_violation_x_priority",
    "outside_energy_x_repair_count",
)


def dss_interactions(features):
    return {
        "priority_coverage_x_event_energy": features["priority_weighted_coverage"]
        * features["event_window_energy"],
        "priority_coverage_x_inverse_onset": features[
            "priority_weighted_coverage"
        ]
        / (1.0 + max(0.0, features["onset_error_ms"])),
        "tolerance_violation_x_priority": features[
            "tolerance_violation_count"
        ]
        * features["priority_weighted_coverage"],
        "outside_energy_x_repair_count": features["outside_window_energy"]
        * features["repair_action_count"],
    }


def rule_score(features):
    """Transparent safety-oriented baseline; higher is preferred."""
    return (
        2.0 * features["event_coverage"]
        + 2.5 * features["priority_weighted_coverage"]
        - 0.001 * features["onset_error_ms"]
        - 0.5 * features["tolerance_violation_count"]
        + 0.6 * features["event_window_energy"]
        - 0.4 * features["outside_window_energy"]
        - 8.0 * features["clip_ratio"]
        - 0.5 * features["silence_ratio"]
        - 0.0001 * features["duration_error_ms"]
        - 0.1 * features["repair_action_count"]
        + 0.2 * features["source_reliability"]
    )


def hard_guard(features, publish_decision=""):
    reasons = []
    if publish_decision in {"BLOCKED", "REPAIR_REJECTED"}:
        reasons.append("PUBLISH_DECISION_BLOCKED")
    if features["clip_ratio"] > 0.001:
        reasons.append("CLIP_RATIO")
    if features["duration_error_ms"] > 1500:
        reasons.append("DURATION_MISMATCH")
    return {"passed": not reasons, "reasons": reasons}


def selective_hybrid(
    rule_candidate,
    learned_candidate,
    learned_margin,
    guard,
    *,
    model_available=True,
    ood=False,
    minimum_margin=0.12,
):
    if not guard["passed"]:
        return {
            "recommendation_status": "PUBLISH_BLOCKED",
            "selected_candidate": None,
            "defer_reason": "|".join(guard["reasons"]),
        }
    if not model_available:
        return {
            "recommendation_status": "ABLATION_DATA_BLOCKED",
            "selected_candidate": None,
            "defer_reason": "REAL_OOF_MODEL_UNAVAILABLE",
        }
    if ood or learned_margin < minimum_margin:
        return {
            "recommendation_status": "NEEDS_HUMAN_REVIEW",
            "selected_candidate": None,
            "defer_reason": "OOD" if ood else "LOW_MARGIN",
        }
    if rule_candidate != learned_candidate:
        return {
            "recommendation_status": "RULE_FALLBACK",
            "selected_candidate": rule_candidate,
            "defer_reason": "RULE_LEARNED_DISAGREEMENT",
        }
    return {
        "recommendation_status": "RANKER_RECOMMENDED",
        "selected_candidate": learned_candidate,
        "defer_reason": "",
    }
