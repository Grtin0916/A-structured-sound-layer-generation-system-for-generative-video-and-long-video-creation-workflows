#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(".")
REPORTS = ROOT / "reports"
OUT_DIR = ROOT / "experiments" / "model_race_2026_07_01"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DECISION_JSON = REPORTS / "week17_model_race_decision_20260701.json"
RERANKER_JSON = REPORTS / "week17_reranker_input_for_0702.json"
RERANKER_CSV = REPORTS / "week17_reranker_input_for_0702.csv"
BOUNDARY_JSON = OUT_DIR / "source_boundary_20260701.json"


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_load_error": str(exc), "_path": str(path)}


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def as_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    return str(v).strip().lower() in {"1", "true", "yes", "y", "ok", "pass", "readable"}


def first(row: dict[str, Any], keys: list[str], default: str = "") -> str:
    for k in keys:
        if k in row and row[k] not in (None, ""):
            return str(row[k])
    return default


def infer_case_id(row: dict[str, Any], artifact_path: str = "") -> str:
    explicit = first(row, ["case_id", "case", "video_id", "sample_id"])
    if explicit:
        return explicit
    text = artifact_path or first(row, ["candidate_id", "id", "name"])
    if "__" in text:
        return Path(text).name.split("__", 1)[0]
    parts = Path(text).parts
    for p in reversed(parts):
        if re.search(r"_[0-9]{3}$", p):
            return p
    return "unknown_case"


def infer_candidate_id(row: dict[str, Any], source_label: str, artifact_path: str = "") -> str:
    explicit = first(row, ["candidate_id", "id", "candidate", "name"])
    if explicit:
        return explicit
    if artifact_path:
        return Path(artifact_path).stem
    case_id = infer_case_id(row, artifact_path)
    return f"{case_id}__{source_label}__unknown"


def normalize_candidate(row: dict[str, Any], source_label: str, eligible_default: bool) -> dict[str, Any]:
    artifact_path = first(
        row,
        [
            "artifact_path",
            "wav_path",
            "audio_path",
            "candidate_path",
            "path",
            "file",
            "output_path",
        ],
    )
    candidate_id = infer_candidate_id(row, source_label, artifact_path)
    case_id = infer_case_id(row, artifact_path or candidate_id)

    readable = (
        as_bool(first(row, ["readable", "is_readable", "wav_readable", "valid_wav"]))
        or (artifact_path.endswith(".wav") and Path(artifact_path).exists())
    )

    source_is_true = source_label == "true_mmaudio_attempt"
    video_conditioned = source_is_true and readable

    eligible = eligible_default and readable
    if source_is_true:
        eligible = readable and video_conditioned

    reject_reason = first(row, ["reject_reason", "failure_reason", "error", "reason"])
    if source_is_true and not readable and not reject_reason:
        reject_reason = "true_mmaudio_attempt_not_readable_or_not_generated"

    return {
        "case_id": case_id,
        "candidate_id": candidate_id,
        "source_label": source_label,
        "video_conditioned": str(video_conditioned).lower(),
        "readable": str(readable).lower(),
        "eligible_for_rerank": str(eligible).lower(),
        "winner_hint": first(row, ["winner", "is_winner", "selected", "rank1"], "false"),
        "reject_reason": reject_reason,
        "repair_action": first(row, ["repair_action", "action", "recommended_repair"]),
        "artifact_path": artifact_path,
        "duration_sec": first(row, ["duration_sec", "duration", "audio_duration_sec"]),
        "rms": first(row, ["rms", "rms_db", "rms_mean"]),
        "peak": first(row, ["peak", "peak_abs", "max_abs"]),
        "silence_ratio": first(row, ["silence_ratio", "silent_ratio"]),
        "score": first(row, ["score", "rank_score", "total_score", "quality_score"]),
    }


def discover_baseline_wavs() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for base in [
        ROOT / "experiments" / "mmaudio_baseline_2026_06_30" / "candidates",
        ROOT / "artifacts" / "model_runs" / "week17_control_baseline",
        ROOT / "artifacts" / "model_runs" / "week17_control_repaired",
    ]:
        if not base.exists():
            continue
        for wav in sorted(base.rglob("*.wav")):
            source = "fallback_mmaudio_baseline"
            if "control" in str(wav).lower():
                source = "control_or_repair_candidate"
            rows.append(
                normalize_candidate(
                    {
                        "artifact_path": str(wav),
                        "readable": "true",
                    },
                    source,
                    eligible_default=True,
                )
            )
    return rows


def flatten_failures(obj: Any) -> list[str]:
    if obj is None:
        return []
    if isinstance(obj, dict):
        vals = []
        for k, v in obj.items():
            if isinstance(v, (str, int, float, bool)):
                vals.append(f"{k}={v}")
            elif isinstance(v, list):
                vals.append(f"{k}[{len(v)}]")
            elif isinstance(v, dict):
                vals.append(f"{k}{{{len(v)}}}")
        return vals
    if isinstance(obj, list):
        out = []
        for item in obj[:8]:
            if isinstance(item, dict):
                out.append("; ".join(flatten_failures(item)[:6]))
            else:
                out.append(str(item))
        return out
    return [str(obj)]


def main() -> None:
    true_summary = load_json(REPORTS / "mmaudio_true_one_attempt_summary.json", {})
    true_failures = load_json(REPORTS / "mmaudio_true_one_attempt_failures.json", {})
    true_metrics = load_csv(REPORTS / "mmaudio_true_one_attempt_metrics.csv")

    candidate_rows: list[dict[str, Any]] = []

    input_tables = [
        (REPORTS / "mmaudio_baseline_ranking.csv", "fallback_mmaudio_baseline", True),
        (REPORTS / "mmaudio_baseline_metrics.csv", "fallback_mmaudio_metrics", True),
        (ROOT / "artifacts" / "model_race" / "week17_control_seed" / "control_seed_ranking.csv", "control_seed", True),
        (ROOT / "artifacts" / "model_race" / "week17_repair_seed" / "repair_before_after.csv", "repair_seed", True),
        (REPORTS / "mmaudio_true_one_attempt_metrics.csv", "true_mmaudio_attempt", False),
    ]

    for path, label, eligible_default in input_tables:
        for row in load_csv(path):
            candidate_rows.append(normalize_candidate(row, label, eligible_default))

    candidate_rows.extend(discover_baseline_wavs())

    true_generated_count = 0
    for key in [
        "true_mmaudio_generated_count",
        "generated_count",
        "readable_count",
        "success_count",
    ]:
        if isinstance(true_summary, dict) and key in true_summary:
            try:
                true_generated_count = max(true_generated_count, int(true_summary[key]))
            except Exception:
                pass

    true_readable_rows = [
        r for r in candidate_rows
        if r["source_label"] == "true_mmaudio_attempt" and r["readable"] == "true"
    ]
    if true_readable_rows:
        true_generated_count = max(true_generated_count, len(true_readable_rows))

    if true_generated_count <= 0 and (
        (REPORTS / "mmaudio_true_one_attempt_summary.json").exists()
        or (REPORTS / "mmaudio_true_one_attempt_failures.json").exists()
    ):
        failure_note = " | ".join(flatten_failures(true_failures)[:10])
        candidate_rows.append(
            {
                "case_id": first(true_summary if isinstance(true_summary, dict) else {}, ["case_id"], "glass_drop_room_001"),
                "candidate_id": "true_mmaudio_attempt__blocked__20260701",
                "source_label": "true_mmaudio_attempt",
                "video_conditioned": "false",
                "readable": "false",
                "eligible_for_rerank": "false",
                "winner_hint": "false",
                "reject_reason": failure_note or "true_mmaudio_runtime_blocked_or_no_output",
                "repair_action": "keep_as_blocker_then_use_labeled_fallback_race",
                "artifact_path": "reports/mmaudio_true_one_attempt_failures.json",
                "duration_sec": "",
                "rms": "",
                "peak": "",
                "silence_ratio": "",
                "score": "",
            }
        )

    seen = set()
    deduped = []
    for r in candidate_rows:
        key = (r["candidate_id"], r["source_label"], r["artifact_path"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)

    eligible_rows = [r for r in deduped if r["eligible_for_rerank"] == "true"]
    blocked_rows = [r for r in deduped if r["eligible_for_rerank"] != "true"]
    case_count = len({r["case_id"] for r in eligible_rows if r["case_id"] != "unknown_case"})
    source_count = len({r["source_label"] for r in eligible_rows})

    if true_generated_count > 0:
        decision_status = "GREEN_TRUE_REPLACEMENT_READY"
        next_action = "run_true_vs_fallback_comparison_then_java_cloud_conditional_consume"
    elif len(eligible_rows) >= 12 and case_count >= 6:
        decision_status = "YELLOW_FALLBACK_AWARE_RERANKER_READY"
        next_action = "use_labeled_fallback_control_candidates_for_temporal_reranker_keep_true_mmaudio_blocker"
    else:
        decision_status = "RED_INSUFFICIENT_RERANKER_INPUT"
        next_action = "inspect_metrics_and_discover_candidate_wavs_before_platform_consume"

    fieldnames = [
        "case_id",
        "candidate_id",
        "source_label",
        "video_conditioned",
        "readable",
        "eligible_for_rerank",
        "winner_hint",
        "reject_reason",
        "repair_action",
        "artifact_path",
        "duration_sec",
        "rms",
        "peak",
        "silence_ratio",
        "score",
    ]

    with RERANKER_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{k: r.get(k, "") for k in fieldnames} for r in deduped])

    RERANKER_JSON.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "decision_status": decision_status,
                "next_action": next_action,
                "eligible_candidate_count": len(eligible_rows),
                "blocked_candidate_count": len(blocked_rows),
                "case_count": case_count,
                "source_count": source_count,
                "rows": deduped,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    decision = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "decision_status": decision_status,
        "next_action": next_action,
        "true_mmaudio_generated_count": true_generated_count,
        "true_attempt_summary_exists": (REPORTS / "mmaudio_true_one_attempt_summary.json").exists(),
        "true_attempt_failures_exists": (REPORTS / "mmaudio_true_one_attempt_failures.json").exists(),
        "eligible_candidate_count": len(eligible_rows),
        "blocked_candidate_count": len(blocked_rows),
        "case_count": case_count,
        "source_count": source_count,
        "outputs": {
            "decision_json": str(DECISION_JSON),
            "reranker_json": str(RERANKER_JSON),
            "reranker_csv": str(RERANKER_CSV),
            "source_boundary_json": str(BOUNDARY_JSON),
        },
        "source_boundary": {
            "true_mmaudio_attempt": "Only rows with readable=true and video_conditioned=true may be called true V2A.",
            "fallback_mmaudio_baseline": "Fallback/control baseline candidates must not be described as true video-conditioned MMAudio.",
            "control_seed": "Rule/control candidates are valid system candidates but not model-generated V2A.",
            "repair_seed": "Repair candidates are valid for repair/reranker testing, not proof of true V2A.",
        },
    }

    DECISION_JSON.write_text(json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8")
    BOUNDARY_JSON.write_text(json.dumps(decision["source_boundary"], indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(decision, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()