#!/usr/bin/env python3
import argparse
import json
import resource
import statistics
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


def dtype_from_ort_type(type_str):
    if "float16" in type_str:
        return np.float16
    if "float" in type_str:
        return np.float32
    if "double" in type_str:
        return np.float64
    if "int64" in type_str:
        return np.int64
    if "int32" in type_str:
        return np.int32
    if "uint8" in type_str:
        return np.uint8
    if "int8" in type_str:
        return np.int8
    return np.float32


def safe_shape(shape):
    fixed = []
    for d in shape:
        if isinstance(d, int) and d > 0:
            fixed.append(d)
        else:
            fixed.append(1)
    return fixed or [1]


def graph_opt_level(name):
    table = {
        "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }
    return table[name]


def p95(values):
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[94]


def max_rss_mb():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_maxrss / 1024.0


def cpu_times():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime, usage.ru_stime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="artifacts/onnx/baseline_v1.onnx")
    parser.add_argument("--out", default="artifacts/benchmarks/ort_week09_profile_20260507.json")
    parser.add_argument("--profile-prefix", default="artifacts/profiles/ort_week09_profile_20260507")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--graph-opt", choices=["disable", "basic", "extended", "all"], default="disable")
    args = parser.parse_args()

    model = Path(args.model)
    if not model.exists():
        raise FileNotFoundError(f"model not found: {model}")

    np.random.seed(20260507)

    so = ort.SessionOptions()
    so.enable_profiling = True
    so.profile_file_prefix = args.profile_prefix
    so.intra_op_num_threads = args.threads
    so.inter_op_num_threads = args.threads
    so.graph_optimization_level = graph_opt_level(args.graph_opt)
    so.enable_mem_pattern = False
    so.enable_cpu_mem_arena = False

    sess = ort.InferenceSession(str(model), sess_options=so, providers=["CPUExecutionProvider"])

    feeds = {}
    input_meta = []
    for x in sess.get_inputs():
        shape = safe_shape(x.shape)
        dtype = dtype_from_ort_type(x.type)
        if np.issubdtype(dtype, np.floating):
            arr = np.random.randn(*shape).astype(dtype)
        else:
            arr = np.zeros(shape, dtype=dtype)
        feeds[x.name] = arr
        input_meta.append({
            "name": x.name,
            "type": x.type,
            "declared_shape": list(x.shape),
            "used_shape": list(arr.shape),
            "dtype": str(arr.dtype),
        })

    output_meta = []
    for y in sess.get_outputs():
        output_meta.append({
            "name": y.name,
            "type": y.type,
            "declared_shape": list(y.shape),
        })

    for _ in range(args.warmup):
        sess.run(None, feeds)

    cpu_user_before, cpu_sys_before = cpu_times()
    wall_before = time.perf_counter()
    rss_before_mb = max_rss_mb()

    latencies = []
    output_shapes = None

    for _ in range(args.runs):
        t0 = time.perf_counter()
        outputs = sess.run(None, feeds)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)
        output_shapes = [list(o.shape) for o in outputs if hasattr(o, "shape")]

    wall_after = time.perf_counter()
    cpu_user_after, cpu_sys_after = cpu_times()
    rss_after_mb = max_rss_mb()
    profile_file = sess.end_profiling()

    result = {
        "date": "2026-05-07",
        "mode": "week09_profile_cpu_memory",
        "model": str(model),
        "model_size_mb": round(model.stat().st_size / 1024 / 1024, 4),
        "onnxruntime_version": ort.__version__,
        "providers_used": sess.get_providers(),
        "graph_optimization": args.graph_opt,
        "threads": args.threads,
        "warmup": args.warmup,
        "runs": args.runs,
        "inputs": input_meta,
        "outputs": output_meta,
        "runtime_output_shapes": output_shapes,
        "latency_ms_mean": round(statistics.mean(latencies), 4),
        "latency_ms_median": round(statistics.median(latencies), 4),
        "latency_ms_p95": round(p95(latencies), 4),
        "latency_ms_min": round(min(latencies), 4),
        "latency_ms_max": round(max(latencies), 4),
        "wall_time_sec": round(wall_after - wall_before, 4),
        "cpu_user_sec": round(cpu_user_after - cpu_user_before, 4),
        "cpu_system_sec": round(cpu_sys_after - cpu_sys_before, 4),
        "max_rss_before_mb": round(rss_before_mb, 4),
        "max_rss_after_mb": round(rss_after_mb, 4),
        "profile_file": profile_file,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
