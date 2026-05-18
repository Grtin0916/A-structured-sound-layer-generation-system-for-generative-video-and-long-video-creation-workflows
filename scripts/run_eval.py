#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path
from typing import Any


VALID_LAYERS = {"foley", "ambience", "music", "dialogue"}


def load_samples(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"no samples found in {path}")
    return samples


def validate_sample(sample: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    sample_id = sample.get("sample_id")
    duration = sample.get("duration_sec")
    events = sample.get("events")
    expected_layers = sample.get("expected_layers")

    if not isinstance(sample_id, str) or not sample_id:
        errors.append("missing_sample_id")

    if not isinstance(duration, (int, float)) or duration <= 0:
        errors.append("invalid_duration")

    if not isinstance(events, list) or not events:
        errors.append("missing_events")
        return errors

    if not isinstance(expected_layers, list) or not expected_layers:
        errors.append("missing_expected_layers")

    for idx, event in enumerate(events):
        t = event.get("t")
        label = event.get("label")
        layer = event.get("layer")
        priority = event.get("priority", 0.5)

        if not isinstance(t, (int, float)) or t < 0:
            errors.append(f"event_{idx}_invalid_time")
        elif isinstance(duration, (int, float)) and t > duration:
            errors.append(f"event_{idx}_time_out_of_range")

        if not isinstance(label, str) or not label.strip():
            errors.append(f"event_{idx}_missing_label")

        if layer not in VALID_LAYERS:
            errors.append(f"event_{idx}_invalid_layer")

        if not isinstance(priority, (int, float)) or not (0 <= priority <= 1):
            errors.append(f"event_{idx}_invalid_priority")

    return errors


def score_sample(sample: dict[str, Any]) -> dict[str, Any]:
    errors = validate_sample(sample)
    events = sample.get("events", [])
    expected_layers = set(sample.get("expected_layers", []))
    observed_layers = {e.get("layer") for e in events if e.get("layer") in VALID_LAYERS}

    event_count = len(events)
    layer_recall = len(observed_layers & expected_layers) / max(len(expected_layers), 1)

    valid_time_count = 0
    duration = sample.get("duration_sec", 0)
    for event in events:
        t = event.get("t")
        if isinstance(t, (int, float)) and isinstance(duration, (int, float)) and 0 <= t <= duration:
            valid_time_count += 1
    temporal_proxy = valid_time_count / max(event_count, 1)

    priority_values = [
        float(e.get("priority", 0.5))
        for e in events
        if isinstance(e.get("priority", 0.5), (int, float))
    ]
    priority_mean = statistics.mean(priority_values) if priority_values else 0.0

    scene_text = str(sample.get("scene", "")).lower()
    keywords = [str(x).lower() for x in sample.get("expected_keywords", [])]
    keyword_hit = sum(1 for kw in keywords if kw in scene_text) / max(len(keywords), 1)

    semantic_proxy = min(
        1.0,
        0.25 * keyword_hit + 0.35 * layer_recall + 0.20 * min(event_count / 3.0, 1.0) + 0.20 * priority_mean,
    )

    if errors:
        failure_reason = ";".join(errors)
    elif semantic_proxy < 0.70:
        failure_reason = "weak_semantic_or_layer_coverage"
    elif temporal_proxy < 1.0:
        failure_reason = "invalid_or_incomplete_timeline"
    else:
        failure_reason = "none"

    return {
        "sample_id": sample["sample_id"],
        "scene": sample["scene"],
        "duration_sec": round(float(sample["duration_sec"]), 4),
        "event_count": event_count,
        "layers": ";".join(sorted(observed_layers)),
        "semantic_proxy": round(semantic_proxy, 4),
        "temporal_proxy": round(temporal_proxy, 4),
        "layer_recall_proxy": round(layer_recall, 4),
        "priority_mean": round(priority_mean, 4),
        "failure_reason": failure_reason,
        "boundary": "V0 proxy rubric only; not a generated-audio quality claim",
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", default="artifacts/manifests/week11_soundlayer_eval_samples.json")
    parser.add_argument("--out-json", default="artifacts/evals/week11_eval_v0.json")
    parser.add_argument("--out-csv", default="artifacts/evals/week11_eval_v0.csv")
    parser.add_argument("--out-metrics", default="artifacts/evals/week11_eval_v0_metrics.json")
    args = parser.parse_args()

    sample_path = Path(args.samples)
    json_path = Path(args.out_json)
    csv_path = Path(args.out_csv)
    metrics_path = Path(args.out_metrics)

    rows = [score_sample(s) for s in load_samples(sample_path)]

    semantic_values = [r["semantic_proxy"] for r in rows]
    temporal_values = [r["temporal_proxy"] for r in rows]
    pass_count = sum(1 for r in rows if r["failure_reason"] == "none")

    metrics = {
        "sample_count": len(rows),
        "pass_count": pass_count,
        "pass_rate": round(pass_count / max(len(rows), 1), 4),
        "semantic_proxy_mean": round(statistics.mean(semantic_values), 4),
        "temporal_proxy_mean": round(statistics.mean(temporal_values), 4),
    }

    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "eval_name": "week11_eval_v0",
        "scope": "SoundLayer Blueprint proxy eval; no model generation claim",
        "input_samples": str(sample_path),
        "metrics": metrics,
        "rows": rows,
    }

    json_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(rows, csv_path)

    print(json.dumps({
        "out_json": str(json_path),
        "out_csv": str(csv_path),
        "out_metrics": str(metrics_path),
        "metrics": metrics
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
