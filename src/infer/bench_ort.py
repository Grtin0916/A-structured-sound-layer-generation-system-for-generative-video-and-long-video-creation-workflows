#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

# File: src/infer/bench_ort.py
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.infer.infer_ort import create_input_array, build_session  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark ONNX Runtime latency for baseline_v1."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train/baseline.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--onnx",
        type=str,
        default="artifacts/onnx/baseline_v1.onnx",
        help="Path to ONNX model.",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="CPUExecutionProvider",
        help="ORT execution provider.",
    )
    parser.add_argument(
        "--input-mode",
        type=str,
        default="real",
        choices=["real", "random"],
        help="Use a real sample from manifest or a random tensor.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="",
        help="Optional manifest path. If empty, prefer valid_manifest in config.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Sample index used when --input-mode=real.",
    )
    parser.add_argument(
        "--input-shape",
        type=str,
        default="1,1,64,313",
        help="4D shape used when --input-mode=random.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when --input-mode=random.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Number of warmup runs.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=200,
        help="Number of timed runs.",
    )
    parser.add_argument(
        "--intra-op-threads",
        type=int,
        default=0,
        help="Set ORT intra_op_num_threads if > 0.",
    )
    parser.add_argument(
        "--inter-op-threads",
        type=int,
        default=0,
        help="Set ORT inter_op_num_threads if > 0.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="artifacts/logs/bench_ort_cpu.csv",
        help="Path to summary CSV output.",
    )
    parser.add_argument(
        "--markdown",
        type=str,
        default="docs/benchmarks/ort_cpu_baseline.md",
        help="Path to markdown benchmark report.",
    )
    return parser.parse_args()


def benchmark_latency(
    session,
    input_name: str,
    output_name: str,
    input_array: np.ndarray,
    warmup: int,
    runs: int,
) -> Tuple[np.ndarray, Tuple[int, ...]]:
    if warmup < 0:
        raise ValueError(f"--warmup must be >= 0, got {warmup}")
    if runs <= 0:
        raise ValueError(f"--runs must be > 0, got {runs}")

    feed = {input_name: input_array}

    for _ in range(warmup):
        _ = session.run([output_name], feed)

    latencies_ms = np.empty(runs, dtype=np.float64)
    output_shape: Tuple[int, ...] | None = None

    for i in range(runs):
        t0 = time.perf_counter()
        output = session.run([output_name], feed)[0]
        t1 = time.perf_counter()

        latencies_ms[i] = (t1 - t0) * 1000.0
        if output_shape is None:
            output_shape = tuple(np.asarray(output).shape)

    if output_shape is None:
        output_shape = tuple()

    return latencies_ms, output_shape


def summarize(latencies_ms: np.ndarray, batch_size: int) -> Dict[str, Any]:
    mean_ms = float(np.mean(latencies_ms))
    throughput = float(batch_size * 1000.0 / mean_ms) if mean_ms > 0 else float("inf")

    return {
        "batch_size": int(batch_size),
        "runs": int(latencies_ms.size),
        "min_ms": float(np.min(latencies_ms)),
        "p50_ms": float(np.percentile(latencies_ms, 50)),
        "p90_ms": float(np.percentile(latencies_ms, 90)),
        "p95_ms": float(np.percentile(latencies_ms, 95)),
        "p99_ms": float(np.percentile(latencies_ms, 99)),
        "max_ms": float(np.max(latencies_ms)),
        "mean_ms": mean_ms,
        "std_ms": float(np.std(latencies_ms)),
        "throughput_samples_per_sec": throughput,
    }


def write_summary_csv(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "onnx",
        "provider_request",
        "provider_actual",
        "input_mode",
        "input_source",
        "input_shape",
        "output_shape",
        "warmup",
        "runs",
        "batch_size",
        "intra_op_threads",
        "inter_op_threads",
        "min_ms",
        "p50_ms",
        "p90_ms",
        "p95_ms",
        "p99_ms",
        "max_ms",
        "mean_ms",
        "std_ms",
        "throughput_samples_per_sec",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def write_markdown(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    md = f"""# ORT Benchmark Report

## Summary
- ONNX: `{row["onnx"]}`
- Provider request: `{row["provider_request"]}`
- Provider actual: `{row["provider_actual"]}`
- Input mode: `{row["input_mode"]}`
- Input source: `{row["input_source"]}`
- Input shape: `{row["input_shape"]}`
- Output shape: `{row["output_shape"]}`

## Benchmark setup
- Warmup runs: `{row["warmup"]}`
- Timed runs: `{row["runs"]}`
- Batch size: `{row["batch_size"]}`
- Intra-op threads: `{row["intra_op_threads"]}`
- Inter-op threads: `{row["inter_op_threads"]}`

## Latency (ms)
- min: `{float(row["min_ms"]):.4f}`
- p50: `{float(row["p50_ms"]):.4f}`
- p90: `{float(row["p90_ms"]):.4f}`
- p95: `{float(row["p95_ms"]):.4f}`
- p99: `{float(row["p99_ms"]):.4f}`
- max: `{float(row["max_ms"]):.4f}`
- mean: `{float(row["mean_ms"]):.4f}`
- std: `{float(row["std_ms"]):.4f}`

## Throughput
- samples / sec: `{float(row["throughput_samples_per_sec"]):.4f}`

## Conclusion
当前基准测试只统计 `session.run()` 的执行耗时，不包含 session 初始化与样本读取。
这一版结果可作为周四主线的第一版 ORT 延迟证据。
"""
    path.write_text(md, encoding="utf-8")


def main() -> None:
    args = parse_args()

    input_array, input_source = create_input_array(
        input_mode=args.input_mode,
        config_path=args.config,
        manifest_path=args.manifest,
        sample_index=args.sample_index,
        input_shape=args.input_shape,
        seed=args.seed,
    )

    session = build_session(
        onnx_path=Path(args.onnx),
        provider=args.provider,
        intra_op_threads=args.intra_op_threads,
        inter_op_threads=args.inter_op_threads,
        enable_profiling=False,
    )

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    latencies_ms, output_shape = benchmark_latency(
        session=session,
        input_name=input_name,
        output_name=output_name,
        input_array=input_array,
        warmup=args.warmup,
        runs=args.runs,
    )

    stats = summarize(latencies_ms=latencies_ms, batch_size=int(input_array.shape[0]))

    row = {
        "onnx": args.onnx,
        "provider_request": args.provider,
        "provider_actual": str(session.get_providers()),
        "input_mode": args.input_mode,
        "input_source": input_source,
        "input_shape": str(tuple(input_array.shape)),
        "output_shape": str(output_shape),
        "warmup": int(args.warmup),
        "runs": int(args.runs),
        "batch_size": stats["batch_size"],
        "intra_op_threads": int(args.intra_op_threads),
        "inter_op_threads": int(args.inter_op_threads),
        "min_ms": stats["min_ms"],
        "p50_ms": stats["p50_ms"],
        "p90_ms": stats["p90_ms"],
        "p95_ms": stats["p95_ms"],
        "p99_ms": stats["p99_ms"],
        "max_ms": stats["max_ms"],
        "mean_ms": stats["mean_ms"],
        "std_ms": stats["std_ms"],
        "throughput_samples_per_sec": stats["throughput_samples_per_sec"],
    }

    write_summary_csv(Path(args.csv), row)
    write_markdown(Path(args.markdown), row)

    print("=" * 80)
    print("[OK] ORT benchmark finished.")
    print(f"[Info] ONNX path       : {args.onnx}")
    print(f"[Info] Provider req    : {args.provider}")
    print(f"[Info] Provider actual : {session.get_providers()}")
    print(f"[Info] Input source    : {input_source}")
    print(f"[Info] Input shape     : {tuple(input_array.shape)}")
    print(f"[Info] Output shape    : {output_shape}")
    print(f"[Info] Warmup / runs   : {args.warmup} / {args.runs}")
    print(f"[Info] P50 ms          : {stats['p50_ms']:.4f}")
    print(f"[Info] P95 ms          : {stats['p95_ms']:.4f}")
    print(f"[Info] P99 ms          : {stats['p99_ms']:.4f}")
    print(f"[Info] Mean ms         : {stats['mean_ms']:.4f}")
    print(f"[Info] Std ms          : {stats['std_ms']:.4f}")
    print(f"[Info] Throughput      : {stats['throughput_samples_per_sec']:.4f} samples/s")
    print(f"[Info] CSV             : {args.csv}")
    print(f"[Info] Markdown        : {args.markdown}")
    print("=" * 80)


if __name__ == "__main__":
    main()