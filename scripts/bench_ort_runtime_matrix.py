#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Week09 ORT graph optimization and thread config runtime matrix."
    )
    parser.add_argument("--onnx", default="artifacts/onnx/baseline_v1.onnx")
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260506)
    parser.add_argument("--out-csv", default="artifacts/benchmarks/ort_week09_runtime_matrix.csv")
    return parser.parse_args()


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    if not values:
        return float("nan")
    k = (len(values) - 1) * q / 100.0
    lo = int(k)
    hi = min(lo + 1, len(values) - 1)
    frac = k - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def graph_level(name: str):
    mapping = {
        "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }
    return mapping[name]


def make_session(onnx_path: str, graph_opt: str, intra_threads: int, inter_threads: int):
    options = ort.SessionOptions()
    options.graph_optimization_level = graph_level(graph_opt)

    if intra_threads >= 0:
        options.intra_op_num_threads = intra_threads
    if inter_threads >= 0:
        options.inter_op_num_threads = inter_threads

    return ort.InferenceSession(
        onnx_path,
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


def make_input(session: ort.InferenceSession, seed: int):
    model_input = session.get_inputs()[0]
    shape = []
    for dim in model_input.shape:
        if isinstance(dim, int) and dim > 0:
            shape.append(dim)
        else:
            shape.append(1)

    rng = np.random.default_rng(seed)
    x = rng.standard_normal(shape).astype(np.float32)
    return model_input.name, x, tuple(shape)


def run_one(session, input_name: str, x: np.ndarray, warmup: int, runs: int):
    for _ in range(warmup):
        session.run(None, {input_name: x})

    latencies = []
    for _ in range(runs):
        start = time.perf_counter()
        session.run(None, {input_name: x})
        end = time.perf_counter()
        latencies.append((end - start) * 1000.0)

    return {
        "min_ms": min(latencies),
        "p50_ms": statistics.median(latencies),
        "p90_ms": percentile(latencies, 90),
        "p95_ms": percentile(latencies, 95),
        "p99_ms": percentile(latencies, 99),
        "max_ms": max(latencies),
        "mean_ms": statistics.mean(latencies),
        "std_ms": statistics.pstdev(latencies),
        "throughput_samples_per_sec": 1000.0 / statistics.mean(latencies),
    }


def main() -> None:
    args = parse_args()

    onnx_path = Path(args.onnx)
    if not onnx_path.exists():
        raise SystemExit(f"ONNX not found: {onnx_path}")

    matrix = [
        {"case_id": "graph_disable_threads_default", "graph_optimization": "disable", "intra_op_threads": 0, "inter_op_threads": 0},
        {"case_id": "graph_basic_threads_default", "graph_optimization": "basic", "intra_op_threads": 0, "inter_op_threads": 0},
        {"case_id": "graph_extended_threads_default", "graph_optimization": "extended", "intra_op_threads": 0, "inter_op_threads": 0},
        {"case_id": "graph_all_threads_default", "graph_optimization": "all", "intra_op_threads": 0, "inter_op_threads": 0},
        {"case_id": "graph_all_threads_1_1", "graph_optimization": "all", "intra_op_threads": 1, "inter_op_threads": 1},
        {"case_id": "graph_all_threads_2_1", "graph_optimization": "all", "intra_op_threads": 2, "inter_op_threads": 1},
        {"case_id": "graph_all_threads_4_1", "graph_optimization": "all", "intra_op_threads": 4, "inter_op_threads": 1},
    ]

    rows = []
    input_shape = None

    for cfg in matrix:
        session = make_session(
            str(onnx_path),
            cfg["graph_optimization"],
            cfg["intra_op_threads"],
            cfg["inter_op_threads"],
        )
        input_name, x, input_shape = make_input(session, args.seed)
        stats = run_one(session, input_name, x, args.warmup, args.runs)

        row = {
            "date": "2026-05-06",
            "model": str(onnx_path),
            "provider": "CPUExecutionProvider",
            "input_mode": "synthetic_fixed_shape",
            "input_name": input_name,
            "input_shape": ",".join(map(str, input_shape)),
            "warmup": args.warmup,
            "runs": args.runs,
            **cfg,
            **{k: f"{v:.4f}" for k, v in stats.items()},
            "notes": "Week09 graph optimization and thread config matrix; synthetic input, session.run only",
        }
        rows.append(row)
        print(row)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fields = list(rows[0].keys())
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
