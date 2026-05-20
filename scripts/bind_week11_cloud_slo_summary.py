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

CLOUD_SUMMARY = CLOUD_ROOT / "loadtest/reports/week11_k6_slo_summary.json"
CLOUD_DOC = CLOUD_ROOT / "docs/benchmarks/cloud_k6_week11.md"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"[FAIL] missing required JSON: {path}")
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"[FAIL] invalid JSON: {path}: {exc}") from exc
    if not isinstance(obj, dict):
        raise SystemExit(f"[FAIL] JSON root must be object: {path}")
    return obj


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def git_commit(repo: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            text=True,
        ).strip()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"[FAIL] cannot read git commit for {repo}: {exc}") from exc
    return out


def git_status(repo: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo), "status", "-sb"],
            text=True,
        ).strip()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"[FAIL] cannot read git status for {repo}: {exc}") from exc
    return out


def require_clean_cloud() -> None:
    status = git_status(CLOUD_ROOT)
    first_line = status.splitlines()[0] if status else ""
    if "main...origin/main" not in first_line:
        raise SystemExit(f"[FAIL] Cloud is not aligned with origin/main:\n{status}")
    dirty = [line for line in status.splitlines()[1:] if line.strip()]
    if dirty:
        raise SystemExit(f"[FAIL] Cloud has local dirty files:\n{status}")


def main() -> None:
    require_clean_cloud()

    bridge = load_json(BRIDGE)
    metrics = load_json(METRICS)
    cloud_summary = load_json(CLOUD_SUMMARY)

    cloud_commit = git_commit(CLOUD_ROOT)
    aggregate = cloud_summary.get("aggregate", {})
    reports = cloud_summary.get("reports", [])

    if cloud_commit != "65c838e":
        raise SystemExit(f"[FAIL] expected Cloud commit 65c838e, got {cloud_commit}")

    if aggregate.get("report_count") != 2:
        raise SystemExit(f"[FAIL] expected report_count=2, got {aggregate.get('report_count')}")

    if aggregate.get("all_reports_passed_local_smoke_gate") is not True:
        raise SystemExit("[FAIL] Cloud all_reports_passed_local_smoke_gate is not True")

    for report in reports:
        if report.get("http_req_failed_rate") != 0.0:
            raise SystemExit(f"[FAIL] bad failed rate in report: {report}")
        if report.get("http_req_duration_p95_ms") is None:
            raise SystemExit(f"[FAIL] missing p95 in report: {report}")
        if report.get("missing_metrics"):
            raise SystemExit(f"[FAIL] report has missing metrics: {report}")
        if report.get("passed_local_smoke_gate") is not True:
            raise SystemExit(f"[FAIL] report did not pass local gate: {report}")

    cloud_slo_gate = {
        "schema_version": cloud_summary.get("schema_version"),
        "cloud_commit": cloud_commit,
        "summary_json": str(CLOUD_SUMMARY),
        "summary_doc": str(CLOUD_DOC),
        "bound_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "scope": "Cloud Week11 local k6 SLO summary. Local smoke gate only; not production SLO.",
        "aggregate": {
            "report_count": aggregate.get("report_count"),
            "report_kinds": aggregate.get("report_kinds"),
            "all_reports_passed_local_smoke_gate": aggregate.get("all_reports_passed_local_smoke_gate"),
            "max_p95_ms": aggregate.get("max_p95_ms"),
            "max_http_req_failed_rate": aggregate.get("max_http_req_failed_rate"),
            "min_checks_rate": aggregate.get("min_checks_rate"),
        },
        "reports": [
            {
                "file": r.get("file"),
                "kind": r.get("kind"),
                "related_log": r.get("related_log"),
                "http_req_failed_rate": r.get("http_req_failed_rate"),
                "http_req_duration_p95_ms": r.get("http_req_duration_p95_ms"),
                "checks_rate": r.get("checks_rate"),
                "missing_metrics": r.get("missing_metrics"),
                "passed_local_smoke_gate": r.get("passed_local_smoke_gate"),
                "failed_reasons": r.get("failed_reasons"),
            }
            for r in reports
        ],
    }

    bridge["cloud_week11_k6_slo_summary_gate"] = cloud_slo_gate
    bridge["cloud_commit"] = cloud_commit
    bridge["cloud_week11_k6_slo_summary_json"] = str(CLOUD_SUMMARY)
    bridge["cloud_week11_k6_slo_summary_passed"] = True
    bridge["cloud_week11_k6_slo_summary_bound_at_epoch"] = int(time.time())

    metrics.update(
        {
            "cross_repo_bridge_cloud_commit_present": 1,
            "cross_repo_bridge_cloud_slo_summary_present": 1,
            "cross_repo_bridge_cloud_slo_summary_passed": 1,
            "cross_repo_bridge_cloud_slo_summary_schema_version_present": int(
                bool(cloud_summary.get("schema_version"))
            ),
            "cross_repo_bridge_cloud_slo_summary_report_count": int(aggregate.get("report_count") or 0),
            "cross_repo_bridge_cloud_slo_summary_report_kinds_count": len(aggregate.get("report_kinds") or {}),
            "cross_repo_bridge_cloud_slo_summary_max_p95_ms": float(aggregate.get("max_p95_ms")),
            "cross_repo_bridge_cloud_slo_summary_failed_rate": float(
                aggregate.get("max_http_req_failed_rate")
            ),
            "cross_repo_bridge_cloud_slo_summary_min_checks_rate": float(
                aggregate.get("min_checks_rate")
            ),
            "cross_repo_bridge_cloud_slo_summary_missing_metrics_count": sum(
                len(r.get("missing_metrics") or []) for r in reports
            ),
            "cross_repo_bridge_cloud_slo_summary_bound_commit_is_65c838e": int(cloud_commit == "65c838e"),
            "cross_repo_bridge_cloud_slo_summary_bound_at_epoch": int(time.time()),
        }
    )

    write_json(BRIDGE, bridge)
    write_json(METRICS, metrics)

    print("[OK] bound Cloud SLO summary into Mainbase bridge")
    print("cloud_commit=", cloud_commit)
    print("summary_json=", CLOUD_SUMMARY)
    print("report_count=", aggregate.get("report_count"))
    print("all_reports_passed_local_smoke_gate=", aggregate.get("all_reports_passed_local_smoke_gate"))
    print("max_p95_ms=", aggregate.get("max_p95_ms"))
    print("max_http_req_failed_rate=", aggregate.get("max_http_req_failed_rate"))
    print("min_checks_rate=", aggregate.get("min_checks_rate"))


if __name__ == "__main__":
    main()