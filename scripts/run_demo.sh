#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG="configs/train/baseline.yaml"
TRAIN_MANIFEST="data/manifests/esc10_miniset_train.jsonl"
VALID_MANIFEST="data/manifests/esc10_miniset_valid.jsonl"
CHECKPOINT="artifacts/experiments/baseline_v1/checkpoints/best.pt"
ONNX_PATH="artifacts/onnx/baseline_v1.onnx"
PARITY_REPORT="artifacts/logs/onnx_parity.md"
BENCH_CSV="artifacts/logs/bench_ort_cpu.csv"
BENCH_MD="docs/benchmarks/ort_cpu_baseline.md"

usage() {
  cat <<EOF
Usage:
  bash scripts/run_demo.sh prepare
  bash scripts/run_demo.sh train
  bash scripts/run_demo.sh export
  bash scripts/run_demo.sh compare
  bash scripts/run_demo.sh infer
  bash scripts/run_demo.sh bench
  bash scripts/run_demo.sh demo
  bash scripts/run_demo.sh full
EOF
}

prepare() {
  echo "[STEP] prepare dataset miniset"
  python scripts/prepare_esc50_miniset.py \
    --src-root data/interim/esc50_raw \
    --dst-root data/processed/esc10_miniset \
    --train-per-class 8 \
    --valid-per-class 2 \
    --valid-fold 5 \
    --target-sr 16000 \
    --clip-seconds 5.0 \
    --seed 42

  echo "[STEP] build manifests"
  python scripts/build_manifest.py \
    --input-dir data/processed/esc10_miniset/train \
    --output-path "$TRAIN_MANIFEST"

  python scripts/build_manifest.py \
    --input-dir data/processed/esc10_miniset/valid \
    --output-path "$VALID_MANIFEST"
}

train() {
  echo "[STEP] train baseline model"
  export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"
  python src/engine/train.py \
    --config "$CONFIG" \
    --device cuda
}

export_onnx() {
  echo "[STEP] export onnx"
  require_file "$CHECKPOINT"
  python src/export/export_onnx.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --output "$ONNX_PATH" \
    --device cpu \
    --verify
}

compare() {
  echo "[STEP] compare pytorch and onnxruntime"
  require_file "$CHECKPOINT"
  require_file "$ONNX_PATH"
  python src/infer/compare_pt_onnx.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --onnx "$ONNX_PATH" \
    --report "$PARITY_REPORT" \
    --device cpu \
    --provider CPUExecutionProvider
}

infer() {
  echo "[STEP] run onnxruntime inference"
  require_file "$ONNX_PATH"
  python src/infer/infer_ort.py \
    --config "$CONFIG" \
    --onnx "$ONNX_PATH" \
    --provider CPUExecutionProvider \
    --input-mode random \
    --input-shape 1,1,64,313 \
    --seed 42
}

bench() {
  echo "[STEP] benchmark onnxruntime"
  require_file "$ONNX_PATH"
  python src/infer/bench_ort.py \
    --config "$CONFIG" \
    --onnx "$ONNX_PATH" \
    --provider CPUExecutionProvider \
    --input-mode random \
    --input-shape 1,1,64,313 \
    --warmup 20 \
    --runs 200 \
    --csv "$BENCH_CSV" \
    --markdown "$BENCH_MD"
}

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "[ERROR] Missing file: $path"
    exit 1
  fi
}

main() {
  if [[ $# -lt 1 ]]; then
    usage
    exit 1
  fi

  case "$1" in
        demo)
      export_onnx
      compare
      infer
      bench
      ;;
    full)
      prepare
      train
      export_onnx
      compare
      infer
      bench
      ;;
  esac
}

main "$@"