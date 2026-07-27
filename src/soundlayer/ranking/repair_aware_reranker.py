"""Constraint-first decisions for repaired audio candidates."""

from __future__ import annotations

from typing import Any


FINAL_STATES = {
    "FINAL_SELECTED", "RUNNER_UP", "MANUAL_REVIEW", "REPAIR_REJECTED", "REPAIR_BLOCKED"
}


def decide_candidate(candidate: dict[str, Any]) -> tuple[str, str]:
    if not candidate.get("output_readable"):
        return "REPAIR_BLOCKED", "after artifact is missing or unreadable"
    if candidate.get("severe_regression"):
        return "REPAIR_REJECTED", "severe acoustic regression"
    if not candidate.get("lineage_complete"):
        return "REPAIR_BLOCKED", "artifact or donor lineage incomplete"
    if not candidate.get("ordering_correct", True):
        return "REPAIR_REJECTED", "DSS event ordering guard failed"
    if candidate.get("manual_reject"):
        return "REPAIR_REJECTED", "human reviewer rejected audible result"
    if candidate.get("forbidden_event_status") == "present":
        return "REPAIR_REJECTED", "human label found a forbidden extra event"
    if not candidate.get("semantic_target_satisfied"):
        return "REPAIR_REJECTED", "semantic target proxy not satisfied"
    if candidate.get("forbidden_event_status", "unknown") == "unknown":
        return "MANUAL_REVIEW", "forbidden-event detector unavailable; human label required"
    if not candidate.get("human_review_complete"):
        return "MANUAL_REVIEW", "blind listening preference is pending"
    return "FINAL_SELECTED", "hard gates, semantic target, and human review passed"
