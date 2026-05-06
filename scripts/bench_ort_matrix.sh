#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/train/baseline.yaml}"
CHECKPOINT="${CHECKPOINT:-artifacts/experiments/baseline_v1/checkpoints/best.pt}"
ONNX="${ONNX:-artifacts/onnx/baseline_v1.onnx}"
MANIFEST="${MANIFEST:-data/manifests/esc10_miniset_valid.jsonl}"
PROVIDER="${PROVIDER:-CPUExecutionProvider}"
DEVICE="${DEVICE:-cpu}"
MAX_SAMPLES="${MAX_SAMPLES:-10}"
WARMUP="${WARMUP:-20}"
RUNS="${RUNS:-200}"

RUN_DATE="${RUN_DATE:-$(date +%F)}"
DATE_TAG="$(date +%Y%m%d_%H%M%S)"

RUN_DIR="artifacts/logs/week09_ort_matrix_${DATE_TAG}"
LOG_FILE="${RUN_DIR}/run.log"

PARITY_CSV="artifacts/logs/week09_parity_multi_sample.csv"
BENCH_CSV="artifacts/logs/week09_bench_ort_multi_sample.csv"
SUMMARY_MD="docs/benchmarks/ort_week09.md"
WEEK09_CSV="artifacts/benchmarks/ort_week09.csv"

mkdir -p "${RUN_DIR}" artifacts/logs artifacts/benchmarks docs/benchmarks

{
  echo "===== Week09 ORT benchmark matrix ====="
  echo "date: $(date -Iseconds)"
  echo "config: ${CONFIG}"
  echo "checkpoint: ${CHECKPOINT}"
  echo "onnx: ${ONNX}"
  echo "manifest: ${MANIFEST}"
  echo "provider: ${PROVIDER}"
  echo "device: ${DEVICE}"
  echo "max_samples: ${MAX_SAMPLES}"
  echo "warmup: ${WARMUP}"
  echo "runs: ${RUNS}"
  echo "run_dir: ${RUN_DIR}"
  echo

  for path in "${CONFIG}" "${CHECKPOINT}" "${ONNX}" "${MANIFEST}" "scripts/bench_ort_dataset.py"; do
    if [ ! -f "${path}" ]; then
      echo "ERROR: required file not found: ${path}"
      exit 2
    fi
  done

  python scripts/bench_ort_dataset.py \
    --config "${CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --onnx "${ONNX}" \
    --manifest "${MANIFEST}" \
    --start-index 0 \
    --max-samples "${MAX_SAMPLES}" \
    --device "${DEVICE}" \
    --provider "${PROVIDER}" \
    --warmup "${WARMUP}" \
    --runs "${RUNS}" \
    --logs-dir "${RUN_DIR}" \
    --parity-csv "${PARITY_CSV}" \
    --bench-csv "${BENCH_CSV}" \
    --summary-md "${SUMMARY_MD}"

  RUN_DATE="${RUN_DATE}" \
  BENCH_CSV="${BENCH_CSV}" \
  WEEK09_CSV="${WEEK09_CSV}" \
  python - <<'PY'
import csv
import os
from pathlib import Path
from statistics import mean

bench_csv = Path(os.environ["BENCH_CSV"])
out_csv = Path(os.environ["WEEK09_CSV"])
run_date = os.environ["RUN_DATE"]

out_csv.parent.mkdir(parents=True, exist_ok=True)

if not bench_csv.exists():
    raise SystemExit(f"missing bench csv: {bench_csv}")

with bench_csv.open("r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

ok_rows = [r for r in rows if r.get("status") == "ok"]

def fnum(row, key):
    try:
        return float(row.get(key, ""))
    except Exception:
        return None

def avg(key):
    vals = [fnum(r, key) for r in ok_rows]
    vals = [v for v in vals if v is not None]
    return mean(vals) if vals else None

summary = {
    "date": run_date,
    "model": "artifacts/onnx/baseline_v1.onnx",
    "provider": "CPUExecutionProvider",
    "manifest": "data/manifests/esc10_miniset_valid.jsonl",
    "samples": str(len(rows)),
    "ok_samples": str(len(ok_rows)),
    "warmup": "20",
    "runs": "200",
    "mean_p50_ms": f"{avg('p50_ms'):.4f}" if avg("p50_ms") is not None else "N/A",
    "mean_p95_ms": f"{avg('p95_ms'):.4f}" if avg("p95_ms") is not None else "N/A",
    "mean_p99_ms": f"{avg('p99_ms'):.4f}" if avg("p99_ms") is not None else "N/A",
    "mean_ms": f"{avg('mean_ms'):.4f}" if avg("mean_ms") is not None else "N/A",
    "throughput_samples_per_sec": f"{avg('throughput_samples_per_sec'):.4f}" if avg("throughput_samples_per_sec") is not None else "N/A",
    "notes": "Week09 initial CPUExecutionProvider matrix based on existing bench_ort_dataset.py"
}

fields = list(summary.keys())
with out_csv.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerow(summary)

print(f"wrote {out_csv}")
print(summary)
PY

  SUMMARY_MD="${SUMMARY_MD}" \
  PARITY_CSV="${PARITY_CSV}" \
  BENCH_CSV="${BENCH_CSV}" \
  RUN_DATE="${RUN_DATE}" \
  python - <<'PY'
import os
from pathlib import Path

p = Path(os.environ["SUMMARY_MD"])
text = p.read_text(encoding="utf-8")

text = text.replace(
    "# ORT CPU Multi-sample Benchmark",
    "# Week09 ORT CPU Multi-sample Benchmark",
    1,
)

text = text.replace(
    "本报告记录多样本验证结果。",
    "本报告记录 Week09 主基地 ORT CPU 多样本 benchmark 首轮验证结果。\n\n"
    f"- Run date: `{os.environ['RUN_DATE']}`\n"
    "- Provider: `CPUExecutionProvider`\n"
    "- Warmup / runs: `20 / 200`",
    1,
)

text = text.replace(
    "artifacts/logs/parity_multi_sample.csv",
    os.environ["PARITY_CSV"],
)

text = text.replace(
    "artifacts/logs/bench_ort_multi_sample.csv",
    os.environ["BENCH_CSV"],
)

if "## 5. Evidence Boundary" not in text:
    text += """

---

## 5. Evidence Boundary

- 当前结果只覆盖 `CPUExecutionProvider`，没有覆盖 CUDA / TensorRT / ORT GPU provider。
- 当前输入合同为固定 shape `(1, 1, 64, 313)`，不能外推为动态 shape 已验证。
- 当前统计范围只包含 `session.run()` 相关推理路径，不代表端到端音频预处理 + 推理 + 后处理总延迟。
- 后续 Week09 可继续补 graph optimization / thread config 对照，但本轮证据已经可作为 Week09 baseline。
"""

p.write_text(text, encoding="utf-8")
PY

  echo
  echo "===== Week09 CSV ====="
  cat "${WEEK09_CSV}"

  echo
  echo "===== Week09 summary ====="
  sed -n '1,260p' "${SUMMARY_MD}"

} 2>&1 | tee "${LOG_FILE}"

echo
echo "log saved to ${LOG_FILE}"
