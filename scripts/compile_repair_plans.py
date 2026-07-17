#!/usr/bin/env python3
"""Compile repair-bank rows into bounded, auditable repair plans."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soundlayer.repair.repair_policy import compile_policy  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repair-bank", type=Path, required=True)
    parser.add_argument("--cases-root", type=Path, default=Path("cases"))
    parser.add_argument("--out-plans", "--out-jsonl", dest="out_plans", type=Path, required=True)
    parser.add_argument("--out-coverage", "--coverage-json", dest="out_coverage", type=Path, required=True)
    parser.add_argument("--out-blocked", "--blocked-json", dest="out_blocked", type=Path, required=True)
    args = parser.parse_args()

    with args.repair_bank.open(encoding="utf-8", newline="") as handle:
        bank = list(csv.DictReader(handle))
    plans: list[dict] = []
    blocked: list[dict] = []
    for row in bank:
        policy = compile_policy(row["failure_type"], row["has_stems"] == "true")
        if policy is None:
            blocked.append({"failure_id": row["failure_id"], "reason": "unknown_failure_type"})
            continue
        plan = {
            "schema_version": "repair.plan.v1",
            "failure_id": row["failure_id"],
            "case_id": row["case_id"],
            "candidate": row["candidate"],
            "failure_type": row["failure_type"],
            "source_audio": row["source_audio"],
            "source_metrics": json.loads(row["before_metrics"]),
            "event": {
                "event_id": row["event_id"],
                "start_sec": float(row["event_start_sec"]),
                "end_sec": float(row["event_end_sec"]),
            },
            "window": {
                "start_sec": float(row["target_start_sec"]),
                "end_sec": float(row["target_end_sec"]),
                "source": row["window_source"],
                "confidence": row["window_confidence"],
            },
            "duration_sec": float(row["duration_sec"]),
            "action": policy["action"],
            "parameters": policy["parameters"],
            "target_metric": policy["target_metric"],
            "target_direction": policy["target_direction"],
            "guard_metrics": policy["guard_metrics"],
            "max_regression": policy["max_regression"],
            "fallback_action": policy["fallback_action"],
            "has_stems": row["has_stems"] == "true",
            "execution_ready": policy["execution_ready"],
            "blocked_reason": policy["blocked_reason"],
            "manual_review_required": True,
            "lineage": {
                "repair_bank": args.repair_bank.as_posix(),
                "diagnostic_plot": row["plot_path"],
            },
        }
        plans.append(plan)
        if not plan["execution_ready"]:
            blocked.append({
                "failure_id": plan["failure_id"],
                "candidate": plan["candidate"],
                "action": plan["action"],
                "reason": plan["blocked_reason"],
            })

    args.out_plans.parent.mkdir(parents=True, exist_ok=True)
    args.out_plans.write_text(
        "".join(json.dumps(plan, sort_keys=True) + "\n" for plan in plans), encoding="utf-8"
    )
    total = len(bank)
    coverage = {
        "repairBankCount": total,
        "compiledPlanCount": len(plans),
        "policyCoverageRatio": len(plans) / total if total else 0.0,
        "targetMetricCoverageRatio": sum(bool(p["target_metric"]) for p in plans) / total if total else 0.0,
        "guardMetricCoverageRatio": sum(bool(p["guard_metrics"]) for p in plans) / total if total else 0.0,
        "fallbackCoverageRatio": sum(bool(p["fallback_action"]) for p in plans) / total if total else 0.0,
        "executionReadyCount": sum(p["execution_ready"] for p in plans),
        "blockedCount": len(blocked),
        "gateStatus": "PASS" if total and len(plans) / total >= 0.8 else "FAIL",
    }
    args.out_coverage.write_text(json.dumps(coverage, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.out_blocked.write_text(json.dumps(blocked, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(coverage, sort_keys=True))
    return 0 if coverage["gateStatus"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
