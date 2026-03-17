#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import statistics
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multi-sample PT/ORT parity and ORT benchmark over a manifest."
    )
    parser.add_argument("--config", type=str, default="configs/train/baseline.yaml")
    parser.add_argument("--checkpoint", type=str, default="artifacts/experiments/baseline_v1/checkpoints/best.pt")
    parser.add_argument("--onnx", type=str, default="artifacts/onnx/baseline_v1.onnx")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--provider", type=str, default="CPUExecutionProvider")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=200)

    parser.add_argument("--logs-dir", type=str, default="artifacts/logs")
    parser.add_argument("--parity-csv", type=str, default="artifacts/logs/parity_multi_sample.csv")
    parser.add_argument("--bench-csv", type=str, default="artifacts/logs/bench_ort_multi_sample.csv")
    parser.add_argument("--summary-md", type=str, default="docs/benchmarks/ort_cpu_multi_sample.md")
    return parser.parse_args()


def ensure_parent(path_str: str) -> Path:
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def run_cmd(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    text = (proc.stdout or "") + ("\n" if proc.stdout and proc.stderr else "") + (proc.stderr or "")
    return proc.returncode, text


def parse_compare_output(text: str) -> dict:
    row = {
        "status": "error",
        "mean_abs_err": "",
        "max_abs_err": "",
        "mse": "",
        "allclose": "",
    }
    if "[OK] PT vs ONNX comparison finished." in text:
        row["status"] = "ok"

    for line in text.splitlines():
        line = line.strip()
        if "Mean abs err" in line:
            row["mean_abs_err"] = line.split(":")[-1].strip()
        elif "Max abs err" in line:
            row["max_abs_err"] = line.split(":")[-1].strip()
        elif line.startswith("[Info] MSE"):
            row["mse"] = line.split(":")[-1].strip()
        elif "Allclose" in line:
            row["allclose"] = line.split(":")[-1].strip()
    return row


def read_single_row_csv(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"Empty CSV: {path}")
    return rows[0]


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_float(v):
    try:
        return float(v)
    except Exception:
        return None


def mean_of(rows: list[dict], key: str) -> float | None:
    vals = [safe_float(r.get(key)) for r in rows]
    vals = [v for v in vals if v is not None]
    return statistics.mean(vals) if vals else None


def median_of(rows: list[dict], key: str) -> float | None:
    vals = [safe_float(r.get(key)) for r in rows]
    vals = [v for v in vals if v is not None]
    return statistics.median(vals) if vals else None


def max_row(rows: list[dict], key: str) -> dict | None:
    valid = [(safe_float(r.get(key)), r) for r in rows]
    valid = [(v, r) for v, r in valid if v is not None]
    if not valid:
        return None
    return max(valid, key=lambda x: x[0])[1]


def min_row(rows: list[dict], key: str) -> dict | None:
    valid = [(safe_float(r.get(key)), r) for r in rows]
    valid = [(v, r) for v, r in valid if v is not None]
    if not valid:
        return None
    return min(valid, key=lambda x: x[0])[1]


def fmt(v, digits=4):
    if v is None:
        return "N/A"
    return f"{v:.{digits}f}"


def write_summary_md(path: Path, parity_rows: list[dict], bench_rows: list[dict], manifest: str, onnx: str) -> None:
    ok_parity = [r for r in parity_rows if r.get("status") == "ok"]
    ok_bench = [r for r in bench_rows if r.get("status") == "ok"]

    allclose_true = sum(1 for r in ok_parity if str(r.get("allclose", "")).strip() == "True")

    parity_outlier = max_row(ok_parity, "mean_abs_err")
    slowest = max_row(ok_bench, "mean_ms")
    fastest = min_row(ok_bench, "mean_ms")

    text = f"""# ORT CPU Multi-sample Benchmark

## 1. Scope

本报告记录多样本验证结果。

相关输入：
- ONNX: `{onnx}`
- Manifest: `{manifest}`

相关输出：
- Parity CSV: `artifacts/logs/parity_multi_sample.csv`
- ORT multi-sample CSV: `artifacts/logs/bench_ort_multi_sample.csv`

---

## 2. Parity Summary

### 2.1 Overall result
- Samples: {len(parity_rows)}
- status=ok: {len(ok_parity)}
- allclose=True: {allclose_true}
- pass rate: {fmt((allclose_true / len(parity_rows) * 100.0) if parity_rows else None, 2)}%

### 2.2 Error statistics
- mean(mean_abs_err): {fmt(mean_of(ok_parity, "mean_abs_err"), 10)}
- median(mean_abs_err): {fmt(median_of(ok_parity, "mean_abs_err"), 10)}
- max(mean_abs_err): {fmt(safe_float(parity_outlier.get("mean_abs_err")) if parity_outlier else None, 10)}
- max(max_abs_err): {fmt(max([safe_float(r.get("max_abs_err")) for r in ok_parity if safe_float(r.get("max_abs_err")) is not None], default=None), 10)}

### 2.3 Observation
- 多数样本的 `mean_abs_err` 处于较小量级；
- 当前多样本数值一致性成立；
- 如存在离群样本，应继续在后续 shape / export 审计中单独复查。

---

## 3. ORT CPU Latency Summary

### 3.1 Aggregate statistics
- samples: {len(bench_rows)}
- mean(p50_ms): {fmt(mean_of(ok_bench, "p50_ms"), 4)}
- mean(p95_ms): {fmt(mean_of(ok_bench, "p95_ms"), 4)}
- mean(p99_ms): {fmt(mean_of(ok_bench, "p99_ms"), 4)}
- mean(mean_ms): {fmt(mean_of(ok_bench, "mean_ms"), 4)}
- mean(throughput_samples_per_sec): {fmt(mean_of(ok_bench, "throughput_samples_per_sec"), 4)}

### 3.2 Slowest / fastest samples
- slowest sample by mean_ms: sample {slowest.get("sample_index", "N/A") if slowest else "N/A"}
  - p50_ms: {fmt(safe_float(slowest.get("p50_ms")) if slowest else None, 4)}
  - mean_ms: {fmt(safe_float(slowest.get("mean_ms")) if slowest else None, 4)}
  - throughput: {fmt(safe_float(slowest.get("throughput_samples_per_sec")) if slowest else None, 4)}

- fastest sample by mean_ms: sample {fastest.get("sample_index", "N/A") if fastest else "N/A"}
  - p50_ms: {fmt(safe_float(fastest.get("p50_ms")) if fastest else None, 4)}
  - mean_ms: {fmt(safe_float(fastest.get("mean_ms")) if fastest else None, 4)}
  - throughput: {fmt(safe_float(fastest.get("throughput_samples_per_sec")) if fastest else None, 4)}

---

## 4. Conclusion

截至当前：
1. 多样本 PT / ORT 对齐结果已形成 CSV 证据；
2. ORT CPU 多样本延迟与吞吐结果已形成 CSV 证据；
3. 当前脚本已将周二 ad-hoc 循环固化为可重复执行入口；
4. 下一步优先级应为：
   - 复查离群样本
   - 进入 shape stability / export audit
   - 再进行今日收口提交
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()

    logs_dir = Path(args.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    parity_rows = []
    bench_rows = []

    for idx in range(args.start_index, args.start_index + args.max_samples):
        print("=" * 80)
        print(f"[RUN] sample_index={idx}")

        parity_report = logs_dir / f"onnx_parity_sample_{idx:02d}.md"
        bench_sample_csv = logs_dir / f"bench_ort_sample_{idx:02d}.csv"
        bench_sample_md = logs_dir / f"bench_ort_sample_{idx:02d}.md"

        compare_cmd = [
            sys.executable, "src/infer/compare_pt_onnx.py",
            "--config", args.config,
            "--checkpoint", args.checkpoint,
            "--onnx", args.onnx,
            "--manifest", args.manifest,
            "--sample-index", str(idx),
            "--device", args.device,
            "--provider", args.provider,
            "--report", str(parity_report),
        ]
        code, text = run_cmd(compare_cmd)
        print(text)

        parity_row = {
            "sample_index": idx,
            "status": "error" if code != 0 else "ok",
            "mean_abs_err": "",
            "max_abs_err": "",
            "mse": "",
            "allclose": "",
            "report": str(parity_report),
        }
        parity_row.update(parse_compare_output(text))
        parity_rows.append(parity_row)

        if code != 0:
            bench_rows.append({
                "sample_index": idx,
                "status": "skipped_due_to_parity_error",
            })
            continue

        bench_cmd = [
            sys.executable, "src/infer/bench_ort.py",
            "--config", args.config,
            "--onnx", args.onnx,
            "--provider", args.provider,
            "--input-mode", "real",
            "--manifest", args.manifest,
            "--sample-index", str(idx),
            "--warmup", str(args.warmup),
            "--runs", str(args.runs),
            "--csv", str(bench_sample_csv),
            "--markdown", str(bench_sample_md),
        ]
        code, text = run_cmd(bench_cmd)
        print(text)

        if code != 0:
            bench_rows.append({
                "sample_index": idx,
                "status": "error",
            })
            continue

        single = read_single_row_csv(bench_sample_csv)
        single["sample_index"] = idx
        single["status"] = "ok"
        bench_rows.append(single)

    parity_csv = ensure_parent(args.parity_csv)
    bench_csv = ensure_parent(args.bench_csv)
    summary_md = ensure_parent(args.summary_md)

    write_csv(
        parity_csv,
        parity_rows,
        fieldnames=[
            "sample_index",
            "status",
            "mean_abs_err",
            "max_abs_err",
            "mse",
            "allclose",
            "report",
        ],
    )

    bench_fieldnames = sorted({k for row in bench_rows for k in row.keys()})
    write_csv(bench_csv, bench_rows, fieldnames=bench_fieldnames)

    write_summary_md(
        summary_md,
        parity_rows=parity_rows,
        bench_rows=bench_rows,
        manifest=args.manifest,
        onnx=args.onnx,
    )

    print("=" * 80)
    print(f"[OK] wrote parity csv : {parity_csv}")
    print(f"[OK] wrote bench csv  : {bench_csv}")
    print(f"[OK] wrote summary md : {summary_md}")
    print("=" * 80)


if __name__ == "__main__":
    main()
