#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path.cwd()

INPUT_JSON = ROOT / "artifacts/evals/week11_eval_v0.json"
INPUT_CSV = ROOT / "artifacts/evals/week11_eval_v0.csv"
HANDOFF = ROOT / "artifacts/manifests/week11_to_w12_handoff_manifest.json"

OUT_JSON = ROOT / "artifacts/manifests/week12_blueprint_seed_manifest.json"
OUT_CSV = ROOT / "artifacts/manifests/week12_blueprint_seed_manifest.csv"
LOG_DIR = ROOT / "artifacts/logs"


def git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"UNAVAILABLE: {exc}"


def load_json(path: Path) -> Any:
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


def find_case_lists(obj: Any) -> list[list[dict[str, Any]]]:
    candidates: list[list[dict[str, Any]]] = []

    if isinstance(obj, list) and all(isinstance(x, dict) for x in obj):
        candidates.append(obj)

    if isinstance(obj, dict):
        for key, value in obj.items():
            lk = key.lower()
            if isinstance(value, list) and all(isinstance(x, dict) for x in value):
                if any(token in lk for token in ["sample", "case", "row", "eval", "item", "result"]):
                    candidates.append(value)

        for _, value in obj.items():
            if isinstance(value, (dict, list)):
                candidates.extend(find_case_lists(value))

    return candidates


def score_case_list(rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0
    keys = {str(k).lower() for row in rows for k in row.keys()}
    score = 0
    for token in ["case", "sample", "task", "semantic", "temporal", "quality", "failure", "status", "score", "artifact", "summary"]:
        if any(token in key for key in keys):
            score += 1
    return score + min(len(rows), 10)


def normalize_id(raw: Any, fallback: str) -> str:
    if raw is None:
        return fallback
    s = str(raw).strip()
    if not s:
        return fallback
    return s.replace(" ", "_").replace("/", "_")


def get_first(row: dict[str, Any], fragments: list[str], default: Any = None) -> Any:
    lowered = {str(k).lower(): k for k in row.keys()}
    for frag in fragments:
        frag_l = frag.lower()
        for lk, original in lowered.items():
            if frag_l in lk:
                return row.get(original)
    return default


def bounded_float(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
    except Exception:
        return default
    if x < 0:
        return 0.0
    if x > 1:
        return 1.0
    return x


def make_blueprint_seed(row: dict[str, Any], idx: int) -> dict[str, Any]:
    case_id = normalize_id(
        get_first(row, ["case_id", "sample_id", "task_id", "id"], None),
        f"week12_seed_{idx:03d}",
    )

    semantic_score = bounded_float(get_first(row, ["semantic", "semantic_score"], 0.0))
    temporal_score = bounded_float(get_first(row, ["temporal", "temporal_score"], 0.0))
    quality_score = bounded_float(get_first(row, ["quality", "quality_score"], 0.0))

    failure_reason = get_first(row, ["failure_reason", "failure", "failure_tag"], "")
    gate_status = get_first(row, ["gate_status", "qualityGateStatus", "status"], "")

    scene_hint = get_first(row, ["scene", "prompt", "description", "input"], "")
    artifact_uri = get_first(row, ["artifactUri", "artifact_uri", "artifact", "manifest"], "")
    eval_summary_uri = get_first(row, ["evalSummaryUri", "summary", "eval_uri"], "")

    # V0 不伪造音频；只把 eval case 映射成可消费的声音层蓝图种子。
    events = [
        {
            "event_id": f"{case_id}_event_001",
            "layer": "ambience",
            "time_start_sec": 0.0,
            "time_end_sec": 3.0,
            "sound_intent": "background_scene_context",
            "control_source": "week11_eval_seed",
            "confidence_proxy": semantic_score,
        },
        {
            "event_id": f"{case_id}_event_002",
            "layer": "foley",
            "time_start_sec": 0.0,
            "time_end_sec": 3.0,
            "sound_intent": "visible_or_implied_action",
            "control_source": "week11_eval_seed",
            "confidence_proxy": temporal_score,
        },
    ]

    return {
        "blueprint_id": f"bp_{case_id}",
        "source_case_id": case_id,
        "schema_version": "soundlayer_blueprint_seed_v0",
        "source": {
            "from": "week11_eval_v0",
            "scene_hint": str(scene_hint),
            "artifactUri": str(artifact_uri),
            "evalSummaryUri": str(eval_summary_uri),
            "qualityGateStatus": str(gate_status),
        },
        "scores_proxy": {
            "semantic": semantic_score,
            "temporal": temporal_score,
            "quality": quality_score,
            "note": "Proxy scores are W11 eval evidence only; not human perceptual audio quality.",
        },
        "events": events,
        "layers_required": sorted({event["layer"] for event in events}),
        "failure_reason": str(failure_reason),
        "w12_action": "Use as seed input for SoundLayer Blueprint schema v0 and candidate-generation planning.",
        "non_goals": [
            "No generated audio is claimed by this seed.",
            "No perceptual quality claim is made from proxy score alone.",
            "No production artifact registry or signed URL lifecycle is claimed.",
        ],
    }


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    hard_failures: list[str] = []
    warnings: list[str] = []

    if not INPUT_JSON.exists():
        hard_failures.append(f"missing {INPUT_JSON.relative_to(ROOT)}")
    if not HANDOFF.exists():
        hard_failures.append(f"missing {HANDOFF.relative_to(ROOT)}")
    if hard_failures:
        for item in hard_failures:
            print(f"HARD_FAILURE: {item}")
        raise SystemExit(1)

    payload = load_json(INPUT_JSON)
    handoff = load_json(HANDOFF)
    csv_rows = read_csv_rows(INPUT_CSV)

    lists = find_case_lists(payload)
    chosen: list[dict[str, Any]] = []

    if lists:
        chosen = max(lists, key=score_case_list)

    if not chosen and csv_rows:
        chosen = csv_rows

    if not chosen:
        hard_failures.append("cannot find eval case rows in week11_eval_v0.json or week11_eval_v0.csv")
        for item in hard_failures:
            print(f"HARD_FAILURE: {item}")
        raise SystemExit(1)

    seeds = [make_blueprint_seed(row, idx + 1) for idx, row in enumerate(chosen)]

    if len(seeds) < 3:
        warnings.append("blueprint seed count below 3; W12 coverage is weak")

    handoff_status = handoff.get("handoff_status")
    if handoff_status not in {"PASS", "PASS_WITH_WARNINGS"}:
        warnings.append(f"handoff_status is not clean PASS: {handoff_status}")

    manifest = {
        "schema_version": "week12_blueprint_seed_manifest_v0",
        "generated_at_local": datetime.now().isoformat(timespec="seconds"),
        "repo": {
            "role": "Mainbase: SoundLayer Blueprint seed producer",
            "git_status": git("status", "-sb"),
            "head": git("rev-parse", "--short", "HEAD"),
            "branch": git("branch", "--show-current"),
        },
        "inputs": {
            "week11_eval_v0_json": str(INPUT_JSON.relative_to(ROOT)),
            "week11_eval_v0_csv": str(INPUT_CSV.relative_to(ROOT)) if INPUT_CSV.exists() else None,
            "week11_to_w12_handoff": str(HANDOFF.relative_to(ROOT)),
        },
        "summary": {
            "seed_count": len(seeds),
            "layers_observed": sorted({layer for seed in seeds for layer in seed["layers_required"]}),
            "has_artifact_links": any(seed["source"].get("artifactUri") for seed in seeds),
            "has_eval_summary_links": any(seed["source"].get("evalSummaryUri") for seed in seeds),
            "handoff_status": handoff_status,
        },
        "blueprint_seeds": seeds,
        "hard_failures": hard_failures,
        "warnings": warnings,
        "boundary": [
            "This is a W12 seed manifest derived from Week11 eval evidence.",
            "It is not generated audio.",
            "It is not a perceptual quality benchmark.",
            "It is not a production artifact registry.",
        ],
    }

    OUT_JSON.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "blueprint_id",
            "source_case_id",
            "semantic",
            "temporal",
            "quality",
            "layers_required",
            "qualityGateStatus",
            "artifactUri",
            "evalSummaryUri",
            "failure_reason",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for seed in seeds:
            writer.writerow(
                {
                    "blueprint_id": seed["blueprint_id"],
                    "source_case_id": seed["source_case_id"],
                    "semantic": seed["scores_proxy"]["semantic"],
                    "temporal": seed["scores_proxy"]["temporal"],
                    "quality": seed["scores_proxy"]["quality"],
                    "layers_required": "|".join(seed["layers_required"]),
                    "qualityGateStatus": seed["source"]["qualityGateStatus"],
                    "artifactUri": seed["source"]["artifactUri"],
                    "evalSummaryUri": seed["source"]["evalSummaryUri"],
                    "failure_reason": seed["failure_reason"],
                }
            )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"week12_blueprint_seed_manifest_{stamp}.log"
    status = "PASS" if not hard_failures else "FAIL"
    if status == "PASS" and warnings:
        status = "PASS_WITH_WARNINGS"

    log_path.write_text(
        "\n".join(
            [
                "===== Week12 Blueprint Seed Manifest =====",
                f"status={status}",
                f"output_json={OUT_JSON.relative_to(ROOT)}",
                f"output_csv={OUT_CSV.relative_to(ROOT)}",
                f"seed_count={len(seeds)}",
                f"layers={manifest['summary']['layers_observed']}",
                f"handoff_status={handoff_status}",
                f"warnings={warnings}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"[{status}] Week12 blueprint seed manifest")
    print(f"output_json={OUT_JSON.relative_to(ROOT)}")
    print(f"output_csv={OUT_CSV.relative_to(ROOT)}")
    print(f"log={log_path.relative_to(ROOT)}")
    print(f"seed_count={len(seeds)}")
    print(f"layers={manifest['summary']['layers_observed']}")
    if warnings:
        for item in warnings:
            print(f"WARNING: {item}")


if __name__ == "__main__":
    main()