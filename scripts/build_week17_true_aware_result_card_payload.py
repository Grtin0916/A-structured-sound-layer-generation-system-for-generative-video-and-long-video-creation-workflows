from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(".").resolve()

SOURCE_FILES = {
    "true_aware_ranking": Path("reports/week17_true_aware_ranking_20260701.json"),
    "true_aware_winners": Path("reports/week17_true_aware_winners_20260701.json"),
    "true_single_evidence_report": Path("reports/week17_true_mmaudio_single_candidate_evidence_20260701.json"),
    "true_single_evidence_artifact": Path(
        "artifacts/model_race/week17_true_mmaudio_single/true_mmaudio_single_candidate_evidence_20260701.json"
    ),
    "true_aware_reranker_summary": Path(
        "artifacts/model_race/week17_true_aware_reranker/true_aware_reranker_summary_20260701.json"
    ),
    "true_aware_gallery": Path(
        "artifacts/model_race/week17_true_aware_reranker/true_aware_gallery_20260701.md"
    ),
    "true_replacement_audio": Path(
        "experiments/mmaudio_true_replacement_2026_06_30/candidates/"
        "glass_drop_room_001__mmaudio__true_replacement_v0.flac"
    ),
}

OUT_PAYLOAD = Path("reports/week17_true_aware_result_card_payload_20260702.json")
OUT_REGISTRY = Path("reports/week17_true_aware_candidate_registry_20260702.csv")
OUT_CARD = Path(
    "artifacts/model_race/week17_true_aware_reranker/true_aware_result_card_20260702.json"
)


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Any:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_records(obj: Any) -> list[dict[str, Any]]:
    if obj is None:
        return []
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        for key in (
            "ranking",
            "candidates",
            "ranked_candidates",
            "winners",
            "items",
            "rows",
            "results",
            "records",
        ):
            value = obj.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]
        return [obj]
    return []


def deep_contains_true_mmaudio(obj: Any) -> bool:
    text = json.dumps(obj, ensure_ascii=False, sort_keys=True).lower()
    return ("true" in text and "mmaudio" in text) or "true_replacement" in text


def first_value(record: dict[str, Any], keys: tuple[str, ...], default: Any = "") -> Any:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return default


def as_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def main() -> None:
    loaded = {name: load_json(path) for name, path in SOURCE_FILES.items() if path.suffix == ".json"}

    ranking_records = collect_records(loaded.get("true_aware_ranking"))
    winner_records = collect_records(loaded.get("true_aware_winners"))
    evidence_records = collect_records(loaded.get("true_single_evidence_report")) + collect_records(
        loaded.get("true_single_evidence_artifact")
    )

    all_candidate_like_records = ranking_records + winner_records + evidence_records
    true_mmaudio_present = any(deep_contains_true_mmaudio(r) for r in all_candidate_like_records)

    candidate_rows: list[dict[str, Any]] = []
    for i, record in enumerate(all_candidate_like_records, start=1):
        candidate_rows.append(
            {
                "row_index": i,
                "case_id": first_value(record, ("case_id", "case", "demo_case"), "unknown_case"),
                "candidate_id": first_value(
                    record,
                    ("candidate_id", "id", "audio_id", "name"),
                    f"candidate_{i:03d}",
                ),
                "model": first_value(record, ("model", "model_name", "source_model"), "unknown_model"),
                "audio_path": first_value(
                    record,
                    ("audio_path", "path", "candidate_path", "wav_path", "flac_path"),
                    "",
                ),
                "score": as_float(first_value(record, ("score", "total_score", "rank_score"), None)),
                "selected": bool(first_value(record, ("selected", "winner", "is_winner"), False)),
                "contains_true_mmaudio_signal": deep_contains_true_mmaudio(record),
            }
        )

    registry_rows = []
    for artifact_type, path in SOURCE_FILES.items():
        registry_rows.append(
            {
                "artifact_type": artifact_type,
                "path": str(path),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() and path.is_file() else 0,
                "sha256": sha256_file(path),
                "commit_policy": "include_if_small_and_project_evidence"
                if path.suffix.lower() not in {".pt", ".pth", ".ckpt", ".safetensors", ".bin"}
                else "never_include_weight",
            }
        )

    boundary = {
        "true_mmaudio_single_success": true_mmaudio_present and SOURCE_FILES["true_replacement_audio"].exists(),
        "batch_true_mmaudio_success": False,
        "full_28_candidate_ranking_available": False,
        "production_slo_verified": False,
        "k6_threshold_pass_verified": False,
        "hf_cache_or_model_weight_included": False,
    }

    result_card = {
        "title": "Week17 true-aware model race result card",
        "status": "true_single_available" if boundary["true_mmaudio_single_success"] else "true_single_not_confirmed",
        "headline": (
            "At least one true MMAudio video-conditioned candidate is available for platform consumption."
            if boundary["true_mmaudio_single_success"]
            else "True-aware evidence exists, but true MMAudio audio artifact was not confirmed."
        ),
        "case_count_observed": len({r["case_id"] for r in candidate_rows if r["case_id"]}),
        "candidate_record_count": len(candidate_rows),
        "winner_record_count": len(winner_records),
        "true_mmaudio_record_count": sum(1 for r in candidate_rows if r["contains_true_mmaudio_signal"]),
        "primary_audio_artifact": str(SOURCE_FILES["true_replacement_audio"]),
        "next_consumer": "Java result-card API, then Cloud demo gate seed",
        "explicit_non_claims": [
            "No batch true MMAudio success claim.",
            "No full 28-candidate ranking claim.",
            "No production SLO claim.",
            "No k6 threshold pass claim.",
            "No model weight or HF cache included.",
        ],
    }

    payload = {
        "schema_version": "week17.true_aware.result_card.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_repo_role": "mainbase",
        "boundary": boundary,
        "result_card": result_card,
        "source_artifacts": registry_rows,
        "candidate_rows": candidate_rows,
    }

    OUT_PAYLOAD.parent.mkdir(parents=True, exist_ok=True)
    OUT_CARD.parent.mkdir(parents=True, exist_ok=True)

    OUT_PAYLOAD.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_CARD.write_text(json.dumps(result_card, ensure_ascii=False, indent=2), encoding="utf-8")

    with OUT_REGISTRY.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "artifact_type",
                "path",
                "exists",
                "size_bytes",
                "sha256",
                "commit_policy",
            ],
        )
        writer.writeheader()
        writer.writerows(registry_rows)

    print("WROTE", OUT_PAYLOAD)
    print("WROTE", OUT_REGISTRY)
    print("WROTE", OUT_CARD)
    print("TRUE_MMAUDIO_SINGLE_SUCCESS=", boundary["true_mmaudio_single_success"])
    print("CANDIDATE_RECORD_COUNT=", result_card["candidate_record_count"])
    print("WINNER_RECORD_COUNT=", result_card["winner_record_count"])
    print("TRUE_MMAUDIO_RECORD_COUNT=", result_card["true_mmaudio_record_count"])


if __name__ == "__main__":
    main()