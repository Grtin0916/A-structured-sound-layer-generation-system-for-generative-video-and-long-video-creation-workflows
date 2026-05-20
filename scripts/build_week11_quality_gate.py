#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(".")
EVAL_JSON = ROOT / "artifacts/evals/week11_eval_v0.json"
EVAL_CSV = ROOT / "artifacts/evals/week11_eval_v0.csv"
EVAL_METRICS = ROOT / "artifacts/evals/week11_eval_v0_metrics.json"
BRIDGE_MANIFEST = ROOT / "artifacts/manifests/week11_crossrepo_task_bridge.json"

OUT_JSON = ROOT / "artifacts/evals/week11_eval_quality_gate_v0.json"
OUT_CSV = ROOT / "artifacts/evals/week11_eval_quality_gate_v0.csv"


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"[FAIL] invalid json: {path}: {exc}") from exc


def flatten_values(obj: Any) -> list[Any]:
    values: list[Any] = []
    if isinstance(obj, dict):
        for v in obj.values():
            values.extend(flatten_values(v))
    elif isinstance(obj, list):
        for v in obj:
            values.extend(flatten_values(v))
    else:
        values.append(obj)
    return values


def recursive_find_keys(obj: Any, key_fragments: tuple[str, ...]) -> list[Any]:
    hits: list[Any] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            lk = str(k).lower()
            if all(fragment.lower() in lk for fragment in key_fragments):
                hits.append(v)
            hits.extend(recursive_find_keys(v, key_fragments))
    elif isinstance(obj, list):
        for item in obj:
            hits.extend(recursive_find_keys(item, key_fragments))
    return hits


def as_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(x) or math.isinf(x):
        return default
    return x


def metric(metrics: dict[str, Any], name: str, default: Any = None) -> Any:
    return metrics.get(name, default)


def load_eval_rows(eval_json: dict[str, Any]) -> list[dict[str, Any]]:
    rows = eval_json.get("rows", [])
    if not isinstance(rows, list):
        raise SystemExit("[FAIL] artifacts/evals/week11_eval_v0.json has no list field: rows")
    normalized = []
    for row in rows:
        if isinstance(row, dict):
            normalized.append(row)
    return normalized


def csv_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    return max(len(rows) - 1, 0)


def derive_failure_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    c = Counter()
    for row in rows:
        reason = str(row.get("failure_reason", "")).strip()
        if not reason:
            reason = "missing_failure_reason"
        c[reason] += 1
    return dict(sorted(c.items()))


VALID_TASK_ID_RE = re.compile(r"^week\d+[-_][A-Za-z0-9][A-Za-z0-9_.:-]*$")

INVALID_TASK_ID_TOKENS = {
    "CREATED",
    "QUEUED",
    "RUNNING",
    "SUCCEEDED",
    "FAILED",
    "PASS",
    "FAIL",
    "WARN",
    "NONE",
    "TRUE",
    "FALSE",
}

DIRECT_TASK_ID_KEYS = {
    "external_task_id",
    "media_task_id",
    "task_id",
}


def looks_like_external_task_id(value: Any) -> bool:
    s = str(value).strip()
    if not s:
        return False
    if s.upper() in INVALID_TASK_ID_TOKENS:
        return False
    if VALID_TASK_ID_RE.match(s):
        return True
    # Conservative fallback: allow non-uppercase IDs with digits and separators.
    # Reject plain states such as CREATED/RUNNING.
    return (
        any(ch.isdigit() for ch in s)
        and any(sep in s for sep in ("-", "_", ":"))
        and not s.isupper()
    )


def derive_external_task_ids(rows: list[dict[str, Any]], bridge: dict[str, Any]) -> list[str]:
    ids: set[str] = set()

    def add_candidate(value: Any) -> None:
        if isinstance(value, (str, int, float)) and looks_like_external_task_id(value):
            ids.add(str(value).strip())

    # Eval rows: only exact task-id fields are allowed.
    for row in rows:
        for key in DIRECT_TASK_ID_KEYS:
            if key in row:
                add_candidate(row.get(key))

    # Bridge manifest: exact task-id fields, plus "id" only under a task-like parent path.
    def walk(obj: Any, path: tuple[str, ...] = ()) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                key_l = str(key).lower()
                path2 = path + (key_l,)

                exact_task_key = key_l in DIRECT_TASK_ID_KEYS
                task_parent_id = key_l == "id" and any("task" in part for part in path)

                if exact_task_key or task_parent_id:
                    add_candidate(value)

                walk(value, path2)
        elif isinstance(obj, list):
            for item in obj:
                walk(item, path)

    walk(bridge)
    return sorted(ids)


def main() -> None:
    eval_json = load_json(EVAL_JSON, {})
    metrics = load_json(EVAL_METRICS, {})
    bridge = load_json(BRIDGE_MANIFEST, {})

    if not EVAL_JSON.exists():
        raise SystemExit(f"[FAIL] missing {EVAL_JSON}")
    if not EVAL_METRICS.exists():
        raise SystemExit(f"[FAIL] missing {EVAL_METRICS}")
    if not BRIDGE_MANIFEST.exists():
        raise SystemExit(f"[FAIL] missing {BRIDGE_MANIFEST}")

    rows = load_eval_rows(eval_json)
    failure_reason_counts = derive_failure_counts(rows)
    external_task_ids = derive_external_task_ids(rows, bridge)

    invalid_external_task_ids = [
        x for x in external_task_ids
        if str(x).upper() in INVALID_TASK_ID_TOKENS or not looks_like_external_task_id(x)
    ]
    if invalid_external_task_ids:
        raise SystemExit(f"[FAIL] invalid external_task_ids detected: {invalid_external_task_ids}")

    sample_count = int(metric(metrics, "sample_count", len(rows)))
    pass_count = int(metric(metrics, "pass_count", 0))
    pass_rate = as_float(metric(metrics, "pass_rate"), 0.0) or 0.0
    semantic_proxy_mean = as_float(metric(metrics, "semantic_proxy_mean"), 0.0) or 0.0
    temporal_proxy_mean = as_float(metric(metrics, "temporal_proxy_mean"), 0.0) or 0.0

    bridge_external_task_id_present = int(metric(metrics, "cross_repo_bridge_external_task_id_present", 0) or 0)
    bridge_json_rows_with_task_id = int(metric(metrics, "cross_repo_bridge_json_rows_with_task_id", 0) or 0)
    bridge_csv_rows_with_task_id = int(metric(metrics, "cross_repo_bridge_csv_rows_with_task_id", 0) or 0)
    java_commit_present = int(metric(metrics, "cross_repo_bridge_java_commit_present", 0) or 0)
    cloud_commit_present = int(metric(metrics, "cross_repo_bridge_cloud_commit_present", 0) or 0)

    cloud_k6_gate_present = int(metric(metrics, "cross_repo_bridge_cloud_k6_gate_present", 0) or 0)
    cloud_k6_gate_passed = int(metric(metrics, "cross_repo_bridge_cloud_k6_gate_passed", 0) or 0)
    cloud_k6_p95_ms = as_float(metric(metrics, "cross_repo_bridge_cloud_k6_p95_ms"), None)
    cloud_k6_failed_rate = as_float(metric(metrics, "cross_repo_bridge_cloud_k6_failed_rate"), None)
    cloud_k6_checks_rate = as_float(metric(metrics, "cross_repo_bridge_cloud_k6_checks_rate"), None)

    checks = {
        "eval_rows_present": sample_count > 0 and len(rows) > 0,
        "eval_pass_rate_ok": pass_rate >= 1.0,
        "semantic_proxy_ok": semantic_proxy_mean >= 0.75,
        "temporal_proxy_ok": temporal_proxy_mean >= 0.95,
        "failure_reason_taxonomy_present": bool(failure_reason_counts),
        "external_task_id_present": bool(external_task_ids) or bridge_external_task_id_present == 1,
        "json_rows_bound_to_task": bridge_json_rows_with_task_id >= sample_count > 0,
        "csv_rows_bound_to_task": bridge_csv_rows_with_task_id >= sample_count > 0,
        "java_commit_present": java_commit_present == 1,
        "cloud_commit_present": cloud_commit_present == 1,
        "cloud_k6_gate_present": cloud_k6_gate_present == 1,
        "cloud_k6_gate_passed": cloud_k6_gate_passed == 1,
        "cloud_k6_p95_under_200ms": cloud_k6_p95_ms is not None and cloud_k6_p95_ms < 200.0,
        "cloud_k6_failed_rate_zero": cloud_k6_failed_rate is not None and cloud_k6_failed_rate == 0.0,
        "cloud_k6_checks_rate_one": cloud_k6_checks_rate is not None and cloud_k6_checks_rate >= 1.0,
    }

    failed_checks = [k for k, ok in checks.items() if not ok]

    if not failed_checks:
        gate_status = "PASS"
    elif all(checks.get(k, False) for k in ("eval_rows_present", "external_task_id_present")):
        gate_status = "WARN"
    else:
        gate_status = "FAIL"

    payload = {
        "schema_version": "week11_quality_gate_v0",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "gate_status": gate_status,
        "scope": "Mainbase quality gate over Week11 eval proxy, cross-repo bridge, and Cloud k6 gate. This is not a generated-audio quality claim.",
        "inputs": {
            "eval_json": str(EVAL_JSON),
            "eval_csv": str(EVAL_CSV),
            "eval_metrics": str(EVAL_METRICS),
            "bridge_manifest": str(BRIDGE_MANIFEST),
        },
        "eval_summary": {
            "eval_rows": len(rows),
            "csv_rows": csv_row_count(EVAL_CSV),
            "sample_count": sample_count,
            "pass_count": pass_count,
            "pass_rate": pass_rate,
            "semantic_proxy_mean": semantic_proxy_mean,
            "temporal_proxy_mean": temporal_proxy_mean,
            "failure_reason_counts": failure_reason_counts,
        },
        "bridge_summary": {
            "external_task_ids": external_task_ids,
            "external_task_id_rule": "exact task-id keys only; state tokens such as CREATED/RUNNING are rejected",
            "external_task_id_present": checks["external_task_id_present"],
            "json_rows_with_task_id": bridge_json_rows_with_task_id,
            "csv_rows_with_task_id": bridge_csv_rows_with_task_id,
            "java_commit_present": bool(java_commit_present),
            "cloud_commit_present": bool(cloud_commit_present),
        },
        "cloud_k6_summary": {
            "gate_present": bool(cloud_k6_gate_present),
            "gate_passed": bool(cloud_k6_gate_passed),
            "p95_ms": cloud_k6_p95_ms,
            "failed_rate": cloud_k6_failed_rate,
            "checks_rate": cloud_k6_checks_rate,
            "threshold_interpretation": "V0 local smoke gate: failed_rate==0, checks_rate>=1, p95<200ms. Not a production SLO.",
        },
        "checks": checks,
        "failed_checks": failed_checks,
        "next_recommendation": (
            "Proceed to Cloud k6 SLO summary if PASS; if WARN/FAIL, inspect failed_checks before adding more features."
        ),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "gate_status",
                "check_name",
                "passed",
                "sample_count",
                "pass_rate",
                "semantic_proxy_mean",
                "temporal_proxy_mean",
                "cloud_k6_p95_ms",
                "cloud_k6_failed_rate",
                "cloud_k6_checks_rate",
            ],
        )
        writer.writeheader()
        for name, ok in checks.items():
            writer.writerow(
                {
                    "gate_status": gate_status,
                    "check_name": name,
                    "passed": int(bool(ok)),
                    "sample_count": sample_count,
                    "pass_rate": pass_rate,
                    "semantic_proxy_mean": semantic_proxy_mean,
                    "temporal_proxy_mean": temporal_proxy_mean,
                    "cloud_k6_p95_ms": cloud_k6_p95_ms,
                    "cloud_k6_failed_rate": cloud_k6_failed_rate,
                    "cloud_k6_checks_rate": cloud_k6_checks_rate,
                }
            )

    print(f"[OK] wrote {OUT_JSON}")
    print(f"[OK] wrote {OUT_CSV}")
    print(f"gate_status={gate_status}")
    if failed_checks:
        print("failed_checks=" + ",".join(failed_checks))
    else:
        print("failed_checks=none")


if __name__ == "__main__":
    main()