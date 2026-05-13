#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LOG_DIR="artifacts/logs"
LOG_FILE="${LOG_DIR}/week10_dvc_repro_20260514.log"
mkdir -p "$LOG_DIR"

{
  echo "===== Week10 DVC repro check ====="
  date
  hostname
  pwd

  echo
  echo "===== git state before repro ====="
  git rev-list --left-right --count HEAD...origin/main || true
  git status -sb --untracked-files=all
  git log --oneline --decorate -5

  echo
  echo "===== dvc command availability ====="
  if ! command -v dvc >/dev/null 2>&1; then
    echo "ERROR: dvc command not found in current environment."
    echo "Hint: activate the environment that was used for Week10 DVC evidence, or install dvc in this env."
    exit 2
  fi
  dvc --version

  echo
  echo "===== dvc status ====="
  dvc status

  echo
  echo "===== dvc repro week10_ort_evidence ====="
  dvc repro week10_ort_evidence

  echo
  echo "===== dvc metrics show ====="
  dvc metrics show

  echo
  echo "===== evidence files after repro ====="
  for f in \
    artifacts/benchmarks/week10_ort_evidence_metrics.json \
    artifacts/benchmarks/week10_ort_evidence_manifest.json \
    artifacts/logs/week10_dvc_ort_evidence_20260511.log
  do
    if [ -e "$f" ]; then
      echo "OK      $f"
    else
      echo "MISSING $f"
    fi
  done

  echo
  echo "===== git state after repro ====="
  git status -sb --untracked-files=all
  git diff --stat
  git diff --name-only

  echo
  echo "===== DONE ====="
} 2>&1 | tee "$LOG_FILE"
