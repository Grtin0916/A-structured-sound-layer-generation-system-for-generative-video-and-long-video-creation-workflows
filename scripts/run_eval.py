#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_BRIDGE_PATH = Path("artifacts/manifests/week11_crossrepo_task_bridge.json")


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"missing JSON file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_samples(path: Path) -> list[dict[str, Any]]:
    data = load_json(path)
    if isinstance(data, list):
        samples = data
    elif isinstance(data, dict):
        for key in ("samples", "items", "rows", "data"):
            value = data.get(key)
            if isinstance(value, list):
                samples = value
                break
        else:
            samples = [data]
    else:
        raise ValueError(f"unsupported samples JSON root type: {type(data).__name__}")

    out: list[dict[str, Any]] = []
    for i, sample in enumerate(samples):
        if not isinstance(sample, dict):
            sample = {"sample_id": f"sample_{i:04d}", "scene": str(sample)}
        out.append(sample)
    return out


def load_bridge(path: Path | None) -> dict[str, Any]:
    if path is None:
        if DEFAULT_BRIDGE_PATH.exists():
            path = DEFAULT_BRIDGE_PATH
        else:
            return {}
    if not path.exists():
        raise FileNotFoundError(f"missing bridge JSON file: {path}")
    data = load_json(path)
    if not isinstance(data, dict):
        raise ValueError("bridge JSON root must be object")
    return data


def event_list(sample: dict[str, Any]) -> list[dict[str, Any]]:
    events = sample.get("events")
    if isinstance(events, list):
        return [e if isinstance(e, dict) else {"label": str(e)} for e in events]

    timeline = sample.get("timeline")
    if isinstance(timeline, list):
        return [e if isinstance(e, dict) else {"label": str(e)} for e in timeline]

    labels = sample.get("labels")
    if isinstance(labels, list):
        return [{"label": str(x)} for x in labels]

    return []


def expected_layers(sample: dict[str, Any], events: list[dict[str, Any]]) -> list[str]:
    raw = sample.get("expected_layers")
    if isinstance(raw, list):
        return [str(x) for x in raw]
    layers = sorted({str(e.get("layer")) for e in events if e.get("layer") is not None})
    return layers


def score_sample(sample: dict[str, Any], idx: int) -> dict[str, Any]:
    events = event_list(sample)
    expected = expected_layers(sample, events)

    event_count = len(events)
    event_layers = {str(e.get("layer")) for e in events if e.get("layer") is not None}
    expected_set = set(expected)

    if expected_set:
        layer_recall = len(event_layers & expected_set) / max(len(expected_set), 1)
    else:
        layer_recall = 1.0 if event_count > 0 else 0.0

    temporal_ok = True
    timed_events = 0
    for e in events:
        t = e.get("t", e.get("time", e.get("timestamp")))
        if isinstance(t, (int, float)) and t >= 0:
            timed_events += 1
        elif t is not None:
            temporal_ok = False

    temporal_proxy = 1.0 if temporal_ok else 0.0
    semantic_proxy = min(1.0, 0.45 + 0.12 * event_count + 0.25 * layer_recall)
    pass_flag = semantic_proxy >= 0.50 and temporal_proxy >= 1.0 and event_count >= 1

    failure_reason = "none"
    if event_count < 1:
        failure_reason = "no_events"
    elif temporal_proxy < 1.0:
        failure_reason = "invalid_event_time"
    elif semantic_proxy < 0.50:
        failure_reason = "semantic_proxy_below_threshold"

    return {
        "sample_id": str(sample.get("sample_id", sample.get("id", f"sample_{idx:04d}"))),
        "scene": str(sample.get("scene", sample.get("description", ""))),
        "event_count": event_count,
        "expected_layer_count": len(expected),
        "layer_recall": round(layer_recall, 6),
        "semantic_proxy": round(semantic_proxy, 6),
        "temporal_proxy": round(temporal_proxy, 6),
        "pass": bool(pass_flag),
        "failure_reason": failure_reason,
    }


def apply_bridge(row: dict[str, Any], bridge: dict[str, Any]) -> dict[str, Any]:
    if not bridge:
        return row

    mapped = dict(row)
    bridge_fields = [
        "external_task_id",
        "java_repo_commit",
        "cloud_repo_commit",
        "cloud_k6_gate_schema_version",
        "cloud_k6_gate_type",
        "cloud_k6_gate_passed",
        "cloud_k6_gate_summary_path",
        "cloud_k6_gate_report_path",
        "cloud_k6_source_summary_json",
        "cloud_k6_http_req_failed_rate",
        "cloud_k6_http_req_duration_p95_ms",
        "cloud_k6_checks_rate",
        "cloud_k6_checks_passes",
        "cloud_k6_checks_fails",
        "cloud_k6_native_threshold_note",
    ]

    for key in bridge_fields:
        if key in bridge:
            mapped[key] = bridge.get(key)

    return mapped


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    preferred = [
        "sample_id",
        "scene",
        "event_count",
        "semantic_proxy",
        "temporal_proxy",
        "pass",
        "failure_reason",
        "external_task_id",
        "java_repo_commit",
        "cloud_repo_commit",
        "cloud_k6_gate_schema_version",
        "cloud_k6_gate_type",
        "cloud_k6_gate_passed",
        "cloud_k6_http_req_duration_p95_ms",
        "cloud_k6_http_req_failed_rate",
        "cloud_k6_checks_rate",
        "cloud_k6_native_threshold_note",
    ]

    all_keys: list[str] = []
    for key in preferred:
        if any(key in r for r in rows):
            all_keys.append(key)
    for row in rows:
        for key in row:
            if key not in all_keys:
                all_keys.append(key)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def build_metrics(rows: list[dict[str, Any]], bridge: dict[str, Any]) -> dict[str, Any]:
    sample_count = len(rows)
    pass_count = sum(1 for r in rows if r.get("pass") is True)
    rows_with_task = sum(1 for r in rows if r.get("external_task_id"))

    metrics: dict[str, Any] = {
        "sample_count": sample_count,
        "pass_count": pass_count,
        "pass_rate": round(pass_count / sample_count, 6) if sample_count else 0.0,
        "semantic_proxy_mean": round(mean([float(r.get("semantic_proxy", 0.0)) for r in rows]), 6),
        "temporal_proxy_mean": round(mean([float(r.get("temporal_proxy", 0.0)) for r in rows]), 6),
        "cross_repo_bridge_external_task_id_present": 1 if bridge.get("external_task_id") else 0,
        "cross_repo_bridge_json_rows_with_task_id": rows_with_task,
        "cross_repo_bridge_csv_rows_with_task_id": rows_with_task,
        "cross_repo_bridge_java_commit_present": 1 if bridge.get("java_repo_commit") else 0,
        "cross_repo_bridge_cloud_commit_present": 1 if bridge.get("cloud_repo_commit") else 0,
        "cross_repo_bridge_checked_at_epoch": int(time.time()),
    }

    if bridge:
        metrics.update({
            "cross_repo_bridge_cloud_k6_gate_present": 1 if bridge.get("cloud_k6_gate_schema_version") else 0,
            "cross_repo_bridge_cloud_k6_gate_passed": 1 if bridge.get("cloud_k6_gate_passed") is True else 0,
            "cross_repo_bridge_cloud_k6_gate_schema_version_present": 1 if bridge.get("cloud_k6_gate_schema_version") else 0,
            "cross_repo_bridge_cloud_k6_p95_ms": bridge.get("cloud_k6_http_req_duration_p95_ms"),
            "cross_repo_bridge_cloud_k6_failed_rate": bridge.get("cloud_k6_http_req_failed_rate"),
            "cross_repo_bridge_cloud_k6_checks_rate": bridge.get("cloud_k6_checks_rate"),
            "cross_repo_bridge_cloud_k6_native_threshold_note_present": 1 if bridge.get("cloud_k6_native_threshold_note") else 0,
        })

    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Week11 SoundLayer eval and optionally attach cross-repo bridge evidence.")
    parser.add_argument("--samples", required=True)
    parser.add_argument("--bridge", default=None)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-metrics", required=True)
    args = parser.parse_args()

    samples_path = Path(args.samples)
    bridge_path = Path(args.bridge) if args.bridge else None
    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)
    out_metrics = Path(args.out_metrics)

    samples = load_samples(samples_path)
    bridge = load_bridge(bridge_path)

    rows = [apply_bridge(score_sample(sample, i), bridge) for i, sample in enumerate(samples)]
    metrics = build_metrics(rows, bridge)

    payload = {
        "schema_version": "week11_eval_v0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "samples_path": str(samples_path),
        "bridge_path": str(bridge_path or DEFAULT_BRIDGE_PATH if bridge else ""),
        "cross_repo_bridge": bridge,
        "rows": rows,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_metrics.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(out_csv, rows)
    out_metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "rows": len(rows),
        "pass_count": metrics.get("pass_count"),
        "pass_rate": metrics.get("pass_rate"),
        "bridge_external_task_id": bridge.get("external_task_id"),
        "bridge_cloud_repo_commit": bridge.get("cloud_repo_commit"),
        "bridge_cloud_k6_gate_passed": bridge.get("cloud_k6_gate_passed"),
        "out_json": str(out_json),
        "out_csv": str(out_csv),
        "out_metrics": str(out_metrics),
    }, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
