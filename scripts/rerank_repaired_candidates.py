#!/usr/bin/env python3
"""Constraint-first reranking and Java-facing repair handoff export."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from soundlayer.ranking.repair_aware_reranker import decide_candidate


FIELDS = [
    "candidate_id", "failure_id", "parent_candidate_id", "repair_action",
    "source_mode", "before_artifact", "after_artifact", "donor_lineage",
    "target_delta", "outside_window_delta_db", "edit_cost",
    "baseline_selector_score", "output_readable", "severe_regression",
    "lineage_complete", "ordering_correct", "semantic_target_satisfied",
    "forbidden_event_status", "human_review_complete", "manual_reject",
    "decision", "reason",
]


def as_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return default
    return str(value).lower() == "true"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-selector", type=Path, required=True)
    parser.add_argument("--tuesday-promotions", type=Path, required=True)
    parser.add_argument("--listening-sheet", type=Path, required=True)
    parser.add_argument("--layer-metrics", type=Path, required=True)
    parser.add_argument("--replacement-metrics", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--handoff-json", type=Path, required=True)
    args = parser.parse_args()

    baseline = {
        f"{row['case_id']}|{row['variant']}": row["selector_score"]
        for row in csv.DictReader(args.baseline_selector.open())
    }
    listening = list(csv.DictReader(args.listening_sheet.open()))
    listening_by_failure = {row["failure_id"]: row for row in listening}
    records: list[dict] = []

    for row in csv.DictReader(args.tuesday_promotions.open()):
        if not row["selected_after"]:
            continue
        failure_id = row["failure_id"]
        after = row["selected_after"]
        before = str(Path(after).with_name("before.wav"))
        listen = listening_by_failure.get(failure_id, {})
        human_complete = listen.get("review_status") == "COMPLETED"
        candidate = {
            "candidate_id": f"{failure_id}_tuesday_repair",
            "failure_id": failure_id,
            "parent_candidate_id": row["candidate"],
            "repair_action": row["repair_kind"],
            "source_mode": "mixed_only",
            "before_artifact": before,
            "after_artifact": after,
            "donor_lineage": {},
            "target_delta": row["target_delta"],
            "outside_window_delta_db": "",
            "edit_cost": row["edit_cost"],
            "baseline_selector_score": baseline.get(row["candidate"], ""),
            "output_readable": (ROOT / after).is_file(),
            "severe_regression": False,
            "lineage_complete": (ROOT / before).is_file() and (ROOT / after).is_file(),
            "ordering_correct": True,
            "semantic_target_satisfied": row["guard_status"] == "PASS",
            "forbidden_event_status": "unknown",
            "human_review_complete": human_complete,
            "manual_reject": False,
        }
        decision, reason = decide_candidate(candidate)
        candidate.update(decision=decision, reason=reason)
        records.append(candidate)

    for row in csv.DictReader(args.layer_metrics.open()):
        candidate = {
            "candidate_id": f"{row['failure_id']}_semantic_repair",
            "failure_id": row["failure_id"],
            "parent_candidate_id": row["parent_candidate_id"],
            "repair_action": row["repair_action"],
            "source_mode": row["source_mode"],
            "before_artifact": row["before_artifact"],
            "after_artifact": row["after_artifact"],
            "donor_lineage": {},
            "target_delta": row["target_window_rms_delta_db"],
            "outside_window_delta_db": row["outside_window_rms_delta_db"],
            "edit_cost": row["changed_sample_ratio"],
            "baseline_selector_score": baseline.get(row["parent_candidate_id"], ""),
            "output_readable": as_bool(row["output_readable"]),
            "severe_regression": as_bool(row["severe_regression"]),
            "lineage_complete": as_bool(row["lineage_complete"]),
            "ordering_correct": as_bool(row["ordering_correct"]),
            "semantic_target_satisfied": as_bool(row["semantic_target_satisfied"]),
            "forbidden_event_status": row["forbidden_event_status"],
            "human_review_complete": False,
            "manual_reject": False,
        }
        decision, reason = decide_candidate(candidate)
        candidate.update(decision=decision, reason=reason)
        records.append(candidate)

    for row in csv.DictReader(args.replacement_metrics.open()):
        candidate = {
            "candidate_id": row["candidate_id"],
            "failure_id": row["failure_id"],
            "parent_candidate_id": row["parent_candidate_id"],
            "repair_action": row["repair_action"],
            "source_mode": row["source_mode"],
            "before_artifact": row["before_artifact"],
            "after_artifact": row["after_artifact"],
            "donor_lineage": {
                "donor_candidate": row["donor_candidate"],
                "donor_audio": row["donor_audio"],
                "source_window": [
                    float(row["donor_start_sec"]), float(row["donor_end_sec"])
                ],
                "crossfade_ms": int(row["crossfade_ms"]),
            },
            "target_delta": row["target_window_rms_delta_db"],
            "outside_window_delta_db": row["outside_window_rms_delta_db"],
            "edit_cost": row["changed_sample_ratio"],
            "baseline_selector_score": baseline.get(row["parent_candidate_id"], ""),
            "output_readable": as_bool(row["output_readable"]),
            "severe_regression": as_bool(row["severe_regression"]),
            "lineage_complete": as_bool(row["lineage_complete"]),
            "ordering_correct": as_bool(row["ordering_correct"]),
            "semantic_target_satisfied": as_bool(row["semantic_target_satisfied"]),
            "forbidden_event_status": row["forbidden_event_status"],
            "human_review_complete": False,
            "manual_reject": False,
        }
        if row["decision"] == "REJECTED_DIAGNOSTIC":
            decision = "REPAIR_REJECTED"
            reason = row["reason"]
        else:
            decision, reason = decide_candidate(candidate)
        candidate.update(decision=decision, reason=reason)
        records.append(candidate)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in records:
            serialized = dict(row)
            serialized["donor_lineage"] = json.dumps(
                serialized["donor_lineage"], sort_keys=True
            )
            writer.writerow({field: serialized.get(field, "") for field in FIELDS})

    handoff = {
        "schema_version": "repair.handoff.v1",
        "automatic_forbidden_event_detection_available": False,
        "human_listening_completed_count": sum(
            row.get("review_status") == "COMPLETED" for row in listening
        ),
        "records": [
            {
                "repair_id": row["candidate_id"],
                "parent_candidate_id": row["parent_candidate_id"],
                "before_artifact": row["before_artifact"],
                "after_artifact": row["after_artifact"],
                "repair_action": row["repair_action"],
                "source_mode": row["source_mode"],
                "donor_lineage": row["donor_lineage"],
                "metrics": {
                    "target_delta": row["target_delta"],
                    "outside_window_delta_db": row["outside_window_delta_db"],
                    "edit_cost": row["edit_cost"],
                    "severe_regression": row["severe_regression"],
                    "ordering_correct": row["ordering_correct"],
                },
                "manual_review": {
                    "required": row["decision"] == "MANUAL_REVIEW",
                    "completed": row["human_review_complete"],
                    "forbidden_event_status": row["forbidden_event_status"],
                },
                "decision": row["decision"],
                "reason": row["reason"],
            }
            for row in records
        ],
    }
    args.handoff_json.parent.mkdir(parents=True, exist_ok=True)
    args.handoff_json.write_text(
        json.dumps(handoff, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    counts = {
        state: sum(row["decision"] == state for row in records)
        for state in sorted({row["decision"] for row in records})
    }
    print(json.dumps({"records": len(records), "decisions": counts}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
