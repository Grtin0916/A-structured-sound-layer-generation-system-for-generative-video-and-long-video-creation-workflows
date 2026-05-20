#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any


ROOT = Path(".")
CLOUD_ROOT = Path.home() / "work/ai-job-platform-cloud"

BRIDGE = ROOT / "artifacts/manifests/week11_crossrepo_task_bridge.json"
METRICS = ROOT / "artifacts/evals/week11_eval_v0_metrics.json"

CLOUD_SUMMARY = CLOUD_ROOT / "loadtest/reports/week11_k6_eval_artifact_link_summary.json"
CLOUD_SCRIPT = CLOUD_ROOT / "loadtest/k6_eval_artifact_link.js"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"[FAIL] missing JSON: {path}")
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise SystemExit(f"[FAIL] JSON root must be object: {path}")
    return obj


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def git_output(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def require_clean_cloud() -> str:
    status = git_output(CLOUD_ROOT, "status", "-sb")
    if status.strip() != "## main...origin/main":
        raise SystemExit(f"[FAIL] Cloud repo is not clean:\n{status}")
    return git_output(CLOUD_ROOT, "rev-parse", "--short", "HEAD")


def main() -> None:
    cloud_commit = require_clean_cloud()
    summary = load_json(CLOUD_SUMMARY)
    bridge = load_json(BRIDGE)
    metrics = load_json(METRICS)

    if cloud_commit != "08e075c":
        raise SystemExit(f"[FAIL] expected Cloud commit 08e075c, got {cloud_commit}")

    if summary.get("passed") is not True:
        raise SystemExit(f"[FAIL] Cloud eval artifact link summary did not pass: {summary}")

    required_fields = summary.get("required_fields") or []
    for field in ["artifactUri", "evalSummaryUri", "qualityGateStatus"]:
        if field not in required_fields:
            raise SystemExit(f"[FAIL] missing required field in summary: {field}")

    checks_rate = float(summary.get("checks_rate"))
    failed_rate = float(summary.get("http_req_failed_rate"))
    p95 = float(summary.get("http_req_duration_p95_ms"))

    if checks_rate != 1.0:
        raise SystemExit(f"[FAIL] checks_rate != 1.0: {checks_rate}")
    if failed_rate != 0.0:
        raise SystemExit(f"[FAIL] http_req_failed_rate != 0.0: {failed_rate}")
    if p95 >= 200.0:
        raise SystemExit(f"[FAIL] p95 >= 200ms: {p95}")

    payload = {
        "schema_version": summary.get("schema_version"),
        "cloud_commit": cloud_commit,
        "summary_json": str(CLOUD_SUMMARY),
        "k6_script": str(CLOUD_SCRIPT),
        "bound_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "scope": "Cloud k6 external consumer smoke verifying Java MediaTask eval artifact link fields.",
        "passed": True,
        "checks_rate": checks_rate,
        "http_req_failed_rate": failed_rate,
        "http_req_duration_p95_ms": p95,
        "required_fields": required_fields,
        "expected_task_id": summary.get("expected_task_id"),
        "expected_artifact_uri": summary.get("expected_artifact_uri"),
        "expected_eval_summary_uri": summary.get("expected_eval_summary_uri"),
        "expected_quality_gate_status": summary.get("expected_quality_gate_status"),
        "source_report": summary.get("source_report"),
        "source_log": summary.get("source_log"),
    }

    bridge["cloud_week11_eval_artifact_link_consumer_gate"] = payload
    bridge["cloud_week11_eval_artifact_link_consumer_passed"] = True
    bridge["cloud_week11_eval_artifact_link_consumer_commit"] = cloud_commit
    bridge["cloud_week11_eval_artifact_link_consumer_bound_at_epoch"] = int(time.time())

    metrics.update(
        {
            "cross_repo_bridge_cloud_eval_artifact_link_consumer_present": 1,
            "cross_repo_bridge_cloud_eval_artifact_link_consumer_passed": 1,
            "cross_repo_bridge_cloud_eval_artifact_link_consumer_bound_commit_is_08e075c": int(cloud_commit == "08e075c"),
            "cross_repo_bridge_cloud_eval_artifact_link_checks_rate": checks_rate,
            "cross_repo_bridge_cloud_eval_artifact_link_failed_rate": failed_rate,
            "cross_repo_bridge_cloud_eval_artifact_link_p95_ms": p95,
            "cross_repo_bridge_cloud_eval_artifact_link_required_fields_count": len(required_fields),
            "cross_repo_bridge_cloud_eval_artifact_link_bound_at_epoch": int(time.time()),
        }
    )

    write_json(BRIDGE, bridge)
    write_json(METRICS, metrics)

    print("[OK] bound Cloud eval artifact link consumer evidence into Mainbase")
    print("cloud_commit=", cloud_commit)
    print("checks_rate=", checks_rate)
    print("http_req_failed_rate=", failed_rate)
    print("http_req_duration_p95_ms=", p95)
    print("required_fields=", required_fields)


if __name__ == "__main__":
    main()