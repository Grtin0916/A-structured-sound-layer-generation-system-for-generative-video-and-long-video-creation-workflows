#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path.cwd()

INPUTS = {
    "e2e_demo_index": ROOT / "artifacts/manifests/week11_e2e_demo_index.json",
    "eval_payload": ROOT / "artifacts/evals/week11_eval_v0.json",
    "eval_metrics": ROOT / "artifacts/evals/week11_eval_v0_metrics.json",
    "crossrepo_bridge": ROOT / "artifacts/manifests/week11_crossrepo_task_bridge.json",
}

OUT = ROOT / "artifacts/manifests/week11_to_w12_handoff_manifest.json"
LOG_DIR = ROOT / "artifacts/logs"


def git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"UNAVAILABLE: {exc}"


def load_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def flatten(obj: Any, prefix: str = "") -> list[tuple[str, Any]]:
    rows: list[tuple[str, Any]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            rows.append((key, v))
            rows.extend(flatten(v, key))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            key = f"{prefix}[{i}]"
            rows.append((key, v))
            rows.extend(flatten(v, key))
    return rows


def find_values(obj: Any, key_fragments: list[str]) -> list[Any]:
    out: list[Any] = []
    fragments = [x.lower() for x in key_fragments]
    for key, value in flatten(obj):
        lk = key.lower()
        if any(f in lk for f in fragments):
            out.append(value)
    return out


def count_candidate_samples(obj: Any) -> int | None:
    if obj is None:
        return None

    direct_values = find_values(obj, ["sample_count", "case_count", "eval_count"])
    for value in direct_values:
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)

    best = 0
    for key, value in flatten(obj):
        lk = key.lower()
        if any(tag in lk for tag in ["samples", "cases", "items", "rows", "eval_rows", "examples"]):
            if isinstance(value, list):
                best = max(best, len(value))
    return best if best > 0 else None


def detect_gate_status(*objects: Any) -> dict[str, Any]:
    values: list[Any] = []
    for obj in objects:
        if obj is None:
            continue
        values.extend(find_values(obj, ["gate_status", "qualityGateStatus", "gate_passed", "quality_gate", "status"]))

    normalized = []
    passed = False
    for value in values:
        if isinstance(value, bool):
            normalized.append(value)
            passed = passed or value
        elif isinstance(value, str):
            s = value.strip().lower()
            normalized.append(value)
            if s in {"pass", "passed", "success", "ok", "true"}:
                passed = True

    return {
        "detected_values": normalized[:30],
        "passed_detected": passed,
    }


def detect_required_terms(*objects: Any) -> dict[str, bool]:
    text_parts: list[str] = []
    for obj in objects:
        if obj is not None:
            text_parts.append(json.dumps(obj, ensure_ascii=False).lower())
    text = "\n".join(text_parts)

    terms = {
        "artifactUri": "artifacturi",
        "evalSummaryUri": "evalsummaryuri",
        "qualityGateStatus": "qualitygatestatus",
        "cloud_k6": "k6",
        "non_production_boundary": "not production",
        "local_boundary": "local",
    }
    return {name: (needle in text) for name, needle in terms.items()}


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)

    loaded = {name: load_json(path) for name, path in INPUTS.items()}

    missing = [name for name, path in INPUTS.items() if not path.exists()]
    present = [name for name, path in INPUTS.items() if path.exists()]

    e2e = loaded["e2e_demo_index"]
    eval_payload = loaded["eval_payload"]
    eval_metrics = loaded["eval_metrics"]
    bridge = loaded["crossrepo_bridge"]

    sample_count_candidates = [
        count_candidate_samples(e2e),
        count_candidate_samples(eval_payload),
        count_candidate_samples(eval_metrics),
        count_candidate_samples(bridge),
    ]
    sample_count = max([x for x in sample_count_candidates if x is not None], default=None)

    gate = detect_gate_status(e2e, eval_payload, eval_metrics, bridge)
    terms = detect_required_terms(e2e, eval_payload, eval_metrics, bridge)

    hard_failures: list[str] = []
    warnings: list[str] = []

    if "e2e_demo_index" in missing:
        hard_failures.append("missing week11_e2e_demo_index.json")
    if "eval_payload" in missing:
        hard_failures.append("missing week11_eval_v0.json")
    if sample_count is None or sample_count < 3:
        warnings.append("sample_count is missing or below 3; W12 blueprint seed coverage may be too weak")
    if not gate["passed_detected"]:
        warnings.append("no explicit PASS-like gate status detected; keep W12 handoff conservative")
    for required in ["artifactUri", "evalSummaryUri", "qualityGateStatus"]:
        if not terms[required]:
            warnings.append(f"{required} not detected in Week11 handoff inputs")

    handoff_status = "PASS" if not hard_failures else "FAIL"
    if handoff_status == "PASS" and warnings:
        handoff_status = "PASS_WITH_WARNINGS"

    manifest = {
        "schema_version": "week11_to_w12_handoff_v0",
        "generated_at_local": datetime.now().isoformat(timespec="seconds"),
        "repo": {
            "role": "Mainbase: SoundLayer eval root and W12 blueprint handoff producer",
            "git_status": git("status", "-sb"),
            "head": git("rev-parse", "--short", "HEAD"),
            "branch": git("branch", "--show-current"),
        },
        "input_files": {
            name: {
                "path": str(path.relative_to(ROOT)),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else None,
            }
            for name, path in INPUTS.items()
        },
        "week11_summary": {
            "present_inputs": present,
            "missing_inputs": missing,
            "sample_count": sample_count,
            "gate_status": gate,
            "required_crossrepo_terms": terms,
        },
        "w12_entry": {
            "main_problem": "Turn Week11 eval evidence into SoundLayer Blueprint schema v0 and task-state-consumable artifacts.",
            "entry_items": [
                {
                    "id": "w12_mainbase_blueprint_schema_v0",
                    "owner_repo": "mainbase",
                    "objective": "Define minimal SoundLayer Blueprint schema v0 from Week11 eval samples.",
                    "expected_output": "docs/design/soundlayer_blueprint_schema_v0.md and artifacts/manifests/week12_blueprint_seed_manifest.json",
                    "depends_on": ["week11_eval_v0", "week11_e2e_demo_index"],
                },
                {
                    "id": "w12_java_task_state_machine",
                    "owner_repo": "java",
                    "objective": "Consume evidence-link semantics as task lifecycle states instead of static response fields.",
                    "expected_output": "minimal create -> queued/running/succeeded/failed transition tests",
                    "depends_on": ["artifactUri", "evalSummaryUri", "qualityGateStatus"],
                },
                {
                    "id": "w12_cloud_dashboard_or_gate",
                    "owner_repo": "cloud",
                    "objective": "Use Cloud k6 evidence as local reliability gate input, not production SLO.",
                    "expected_output": "dashboard JSON or local gate summary from existing k6 reports",
                    "depends_on": ["week11_k6_evidence_index", "local_non_production_boundary"],
                },
            ],
            "non_goals": [
                "No production SLO claim.",
                "No real cloud-provider deployment claim.",
                "No generated-audio perceptual-quality claim from proxy eval alone.",
                "No full artifact registry or signed URL lifecycle claim.",
            ],
        },
        "handoff_status": handoff_status,
        "hard_failures": hard_failures,
        "warnings": warnings,
    }

    OUT.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"week11_to_w12_handoff_manifest_{stamp}.log"
    log_path.write_text(
        "\n".join(
            [
                "===== Week11 -> Week12 handoff manifest =====",
                f"status={handoff_status}",
                f"output={OUT.relative_to(ROOT)}",
                f"sample_count={sample_count}",
                f"hard_failures={hard_failures}",
                f"warnings={warnings}",
                f"present_inputs={present}",
                f"missing_inputs={missing}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"[{handoff_status}] Week11 -> Week12 handoff manifest")
    print(f"output={OUT.relative_to(ROOT)}")
    print(f"log={log_path.relative_to(ROOT)}")
    print(f"sample_count={sample_count}")
    print(f"warnings={len(warnings)}")
    if hard_failures:
        for item in hard_failures:
            print(f"HARD_FAILURE: {item}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()