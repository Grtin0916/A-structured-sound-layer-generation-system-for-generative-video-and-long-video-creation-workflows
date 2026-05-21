#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import subprocess
import time
from pathlib import Path
from typing import Any


ROOT = Path(".")
OUT = Path("artifacts/manifests/week11_e2e_demo_index.json")


def sh(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"unknown: {exc}"


def read_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_read_error": str(exc), "_path": str(p)}


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    if not p.exists():
        return []
    try:
        with p.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def file_info(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "exists": False}
    return {
        "path": str(p),
        "exists": True,
        "size_bytes": p.stat().st_size,
        "modified_at_epoch": int(p.stat().st_mtime),
    }


def glob_files(patterns: list[str]) -> list[str]:
    files: list[str] = []
    for pat in patterns:
        files.extend(str(p) for p in sorted(ROOT.glob(pat)) if p.is_file())
    return sorted(set(files))


def infer_gate_status(eval_json: dict[str, Any], eval_rows: list[dict[str, str]], bridge: dict[str, Any]) -> str:
    for key in ("gate_status", "quality_gate_status", "status"):
        value = eval_json.get(key) or bridge.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    row_failure_reasons = []
    for row in eval_rows:
        value = row.get("failure_reason", "").strip().lower()
        if value and value != "none":
            row_failure_reasons.append(value)

    if eval_rows and not row_failure_reasons:
        return "PASS"
    if eval_rows and row_failure_reasons:
        return "CHECK_FAILURE_REASONS"
    return "UNKNOWN"


def main() -> None:
    eval_json_path = Path("artifacts/evals/week11_eval_v0.json")
    eval_csv_path = Path("artifacts/evals/week11_eval_v0.csv")
    bridge_path = Path("artifacts/manifests/week11_crossrepo_task_bridge.json")

    eval_json = read_json(eval_json_path)
    eval_rows = read_csv_rows(eval_csv_path)
    bridge = read_json(bridge_path)

    metrics_candidates = glob_files([
        "artifacts/evals/*metrics*.json",
        "artifacts/metrics/*week11*.json",
        "artifacts/benchmarks/*week11*.json",
    ])

    manifest_candidates = glob_files([
        "artifacts/manifests/week11*.json",
    ])

    log_candidates = glob_files([
        "artifacts/logs/week11*.log",
    ])

    dvc_status = sh(["dvc", "status"])
    tracked_git_status = sh(["git", "status", "-sb", "--untracked-files=no"])
    full_git_status_at_build = sh(["git", "status", "-sb"])
    mainbase_commit = sh(["git", "rev-parse", "--short", "HEAD"])

    sample_count = 0
    if isinstance(eval_json.get("rows"), list):
        sample_count = len(eval_json["rows"])
    elif eval_rows:
        sample_count = len(eval_rows)

    external_task_id = (
        bridge.get("external_task_id")
        or bridge.get("task_id")
        or eval_json.get("external_task_id")
        or "week11-k6-seed-created-001"
    )

    generated_outputs = [
        "scripts/build_week11_e2e_demo_index.py",
        "docs/evals/eval_runner.md",
        "artifacts/manifests/week11_e2e_demo_index.json",
    ]

    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "index_name": "week11_e2e_demo_index",
        "purpose": "Machine-readable Week11 E2E demo index: media task -> SoundLayer eval -> Java artifact link API -> Cloud k6 consumer gate -> DVC accepted state.",
        "boundary": "V0 evidence product only. Not a production artifact registry, not a production SLO, not generated-audio quality benchmark, not real cloud load testing.",
        "repo": {
            "name": "audio_engineering_repo_skeleton_v1",
            "role": "Mainbase: SoundLayer eval and evidence root",
            "commit": mainbase_commit,
            "tracked_git_status": tracked_git_status,
            "full_git_status_at_build": full_git_status_at_build,
            "dvc_status": dvc_status,
            "status_interpretation": "tracked_git_status ignores untracked generated outputs; generated outputs are listed explicitly below and must be git-added before commit.",
        },
        "week11_known_commits": {
            "mainbase": ["7813e0d", "8050479", "327cd7a"],
            "java": ["cee70e7", "1fc9b80", "c9debb1"],
            "cloud": ["08e075c", "65c838e", "6eb961f"],
        },
        "generated_outputs": [file_info(p) for p in generated_outputs],
        "business_chain": {
            "external_task_id": external_task_id,
            "mainbase_eval": {
                "eval_json": file_info(eval_json_path),
                "eval_csv": file_info(eval_csv_path),
                "sample_count": sample_count,
                "gate_status": infer_gate_status(eval_json, eval_rows, bridge),
                "scope": eval_json.get("scope") or "manual SoundLayer Blueprint proxy eval",
            },
            "java_api_contract": {
                "expected_fields": ["artifactUri", "evalSummaryUri", "qualityGateStatus"],
                "meaning": "V0 evidence-link fields consumed by Cloud k6; not a complete artifact registry.",
                "known_commits": ["cee70e7", "1fc9b80", "c9debb1"],
            },
            "cloud_consumer_gate": {
                "expected_fields": ["artifactUri", "evalSummaryUri", "qualityGateStatus"],
                "expected_report_roles": ["query_slo_smoke", "eval_artifact_link_consumer"],
                "known_commits": ["08e075c", "65c838e", "6eb961f"],
            },
        },
        "evidence_files": {
            "self": file_info(OUT),
            "bridge_manifest": file_info(bridge_path),
            "metrics_candidates": metrics_candidates,
            "manifest_candidates": sorted(set(manifest_candidates + [str(OUT)])),
            "log_candidates": log_candidates[-20:],
        },
        "quality_notes": [
            "DVC status must remain clean after index generation.",
            "Do not run dvc repro here; existing manual bridge evidence has already been accepted.",
            "Proxy eval scores are schema and evidence-chain checks, not generated audio perceptual quality claims.",
            "Cloud k6 evidence should be interpreted through thresholds/pass-fail semantics, not as production SLO proof.",
            "The index separates tracked_git_status from generated_outputs to avoid self-polluting the evidence root.",
        ],
        "next_recommended_edges": [
            {
                "repo": "media-task-platform-java",
                "action": "Add minimal evidence-link contract boundary for artifactUri/evalSummaryUri/qualityGateStatus if absent.",
            },
            {
                "repo": "ai-job-platform-cloud",
                "action": "Generate week11_k6_evidence_index.json by merging query/SLO and artifact-link consumer reports.",
            },
        ],
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "out": str(OUT),
        "mainbase_commit": mainbase_commit,
        "external_task_id": external_task_id,
        "sample_count": sample_count,
        "gate_status": payload["business_chain"]["mainbase_eval"]["gate_status"],
        "tracked_git_status": tracked_git_status,
        "dvc_clean": dvc_status.strip() == "Data and pipelines are up to date.",
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()