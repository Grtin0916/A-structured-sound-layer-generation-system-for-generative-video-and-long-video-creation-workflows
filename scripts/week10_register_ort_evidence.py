from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

BENCHMARK_CSV = ROOT / "artifacts/benchmarks/ort_week09.csv"
RUNTIME_MATRIX_CSV = ROOT / "artifacts/benchmarks/ort_week09_runtime_matrix.csv"
PROFILE_JSON = ROOT / "artifacts/benchmarks/ort_week09_profile_20260507.json"

METRICS_OUT = ROOT / "artifacts/benchmarks/week10_ort_evidence_metrics.json"
MANIFEST_OUT = ROOT / "artifacts/benchmarks/week10_ort_evidence_manifest.json"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"missing required CSV: {path}")

    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def to_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default

    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def flatten_numbers(obj: Any, prefix: str = "") -> dict[str, float]:
    found: dict[str, float] = {}

    if isinstance(obj, dict):
        for key, value in obj.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            found.update(flatten_numbers(value, next_prefix))
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            next_prefix = f"{prefix}.{idx}" if prefix else str(idx)
            found.update(flatten_numbers(value, next_prefix))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        found[prefix] = float(obj)

    return found


def main() -> None:
    benchmark_rows = read_csv_rows(BENCHMARK_CSV)
    runtime_rows = read_csv_rows(RUNTIME_MATRIX_CSV)

    if not benchmark_rows:
        raise RuntimeError(f"empty benchmark CSV: {BENCHMARK_CSV}")

    if not runtime_rows:
        raise RuntimeError(f"empty runtime matrix CSV: {RUNTIME_MATRIX_CSV}")

    if not PROFILE_JSON.exists():
        raise FileNotFoundError(f"missing required JSON: {PROFILE_JSON}")

    profile_data = json.loads(PROFILE_JSON.read_text(encoding="utf-8"))

    benchmark_first = benchmark_rows[0]

    runtime_rows_with_p50 = [
        row for row in runtime_rows
        if to_float(row.get("p50_ms")) is not None
    ]

    if not runtime_rows_with_p50:
        raise RuntimeError("runtime matrix CSV has no numeric p50_ms column")

    best_runtime_by_p50 = min(
        runtime_rows_with_p50,
        key=lambda row: float(row["p50_ms"]),
    )

    profile_numbers = flatten_numbers(profile_data)

    metrics = {
        "week10_ort_evidence": {
            "benchmark_samples": to_int(benchmark_first.get("samples"), 0),
            "benchmark_ok_samples": to_int(benchmark_first.get("ok_samples"), 0),
            "benchmark_mean_p50_ms": to_float(benchmark_first.get("mean_p50_ms"), 0.0),
            "benchmark_mean_p95_ms": to_float(benchmark_first.get("mean_p95_ms"), 0.0),
            "benchmark_throughput_samples_per_sec": to_float(
                benchmark_first.get("throughput_samples_per_sec"), 0.0
            ),
            "runtime_best_p50_ms": to_float(best_runtime_by_p50.get("p50_ms"), 0.0),
            "runtime_best_p95_ms": to_float(best_runtime_by_p50.get("p95_ms"), 0.0),
            "runtime_best_mean_ms": to_float(best_runtime_by_p50.get("mean_ms"), 0.0),
            "runtime_best_throughput_samples_per_sec": to_float(
                best_runtime_by_p50.get("throughput_samples_per_sec"), 0.0
            ),
            "profile_numeric_field_count": len(profile_numbers),
        }
    }

    manifest = {
        "date": str(date.today()),
        "scope": "Week10 local DVC-first evidence registration only",
        "tracking_route": "DVC-first",
        "fallback_route": "MLflow local tracking",
        "inputs": [
            str(BENCHMARK_CSV.relative_to(ROOT)),
            str(RUNTIME_MATRIX_CSV.relative_to(ROOT)),
            str(PROFILE_JSON.relative_to(ROOT)),
        ],
        "outputs": [
            str(METRICS_OUT.relative_to(ROOT)),
            str(MANIFEST_OUT.relative_to(ROOT)),
        ],
        "runtime_best_case_id": best_runtime_by_p50.get("case_id", ""),
        "boundary": [
            "local CPU / ONNX Runtime evidence only",
            "no remote DVC storage",
            "no MLflow tracking server",
            "no CUDAExecutionProvider benchmark",
            "no TensorRT / vLLM / SGLang comparison",
            "no production runtime selection",
        ],
    }

    METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)
    METRICS_OUT.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    MANIFEST_OUT.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"[OK] wrote {METRICS_OUT.relative_to(ROOT)}")
    print(f"[OK] wrote {MANIFEST_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
