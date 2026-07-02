from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(".").resolve()

RAW_PAYLOAD = Path("reports/week17_true_aware_result_card_payload_20260702.json")
TRUE_AUDIO = Path(
    "experiments/mmaudio_true_replacement_2026_06_30/candidates/"
    "glass_drop_room_001__mmaudio__true_replacement_v0.flac"
)

OUT_BRIDGE = Path("reports/week17_true_aware_platform_bridge_payload_20260702.json")
OUT_GUARD = Path("reports/week17_true_aware_claim_guard_20260702.json")
OUT_ARTIFACT = Path(
    "artifacts/model_race/week17_true_aware_reranker/"
    "true_aware_platform_bridge_20260702.json"
)


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return data


def main() -> None:
    if not RAW_PAYLOAD.exists():
        raise FileNotFoundError(f"Missing raw payload: {RAW_PAYLOAD}")

    raw = load_json(RAW_PAYLOAD)

    raw_result_card = raw.get("result_card", {})
    raw_boundary = raw.get("boundary", {})
    raw_candidate_rows = raw.get("candidate_rows", [])
    raw_source_artifacts = raw.get("source_artifacts", [])

    true_audio_exists = TRUE_AUDIO.exists()
    true_audio_size = TRUE_AUDIO.stat().st_size if true_audio_exists else 0

    strict_boundary = {
        "true_mmaudio_single_success": bool(true_audio_exists and raw_boundary.get("true_mmaudio_single_success")),
        "true_mmaudio_audio_artifact_count": 1 if true_audio_exists else 0,
        "true_mmaudio_case_count": 1 if true_audio_exists else 0,
        "true_mmaudio_batch_success": False,
        "full_candidate_ranking_available": False,
        "production_slo_verified": False,
        "k6_threshold_pass_verified": False,
        "hf_cache_or_model_weight_included": False,
    }

    claim_guard = {
        "claim_level": "single_true_v2a_candidate_available"
        if strict_boundary["true_mmaudio_single_success"]
        else "true_v2a_candidate_not_confirmed",
        "allowed_claims": [
            "One true MMAudio video-conditioned audio artifact exists.",
            "The artifact can be used as a result-card input for Java and Cloud demo-gate seeding.",
            "The system can compare this true candidate against existing fallback-aware model-race evidence.",
        ],
        "forbidden_claims": [
            "Do not claim true MMAudio batch success.",
            "Do not claim all model-race candidates are true MMAudio outputs.",
            "Do not claim full 28-candidate ranking availability.",
            "Do not claim production SLO verification.",
            "Do not claim k6 threshold pass.",
            "Do not include model weights or Hugging Face cache in Git.",
        ],
        "why_this_guard_exists": (
            "The raw payload contains multiple candidate/winner records, but only one confirmed "
            "true MMAudio audio artifact. Platform consumers must not inflate this into batch success."
        ),
    }

    bridge = {
        "schema_version": "week17.true_aware.platform_bridge.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_repo_role": "mainbase",
        "consumer_order": ["java_result_card_api", "cloud_demo_gate_seed"],
        "strict_boundary": strict_boundary,
        "claim_guard": claim_guard,
        "platform_result_card": {
            "title": "Week17 true-aware result card bridge",
            "status": "consumer_ready"
            if strict_boundary["true_mmaudio_single_success"]
            else "blocked_missing_true_audio",
            "primary_case_id": "glass_drop_room_001",
            "primary_model": "MMAudio",
            "primary_audio_artifact": str(TRUE_AUDIO),
            "primary_audio_exists": true_audio_exists,
            "primary_audio_size_bytes": true_audio_size,
            "primary_audio_sha256": sha256_file(TRUE_AUDIO),
            "raw_candidate_record_count": len(raw_candidate_rows)
            if isinstance(raw_candidate_rows, list)
            else None,
            "raw_winner_record_count": raw_result_card.get("winner_record_count"),
            "safe_true_mmaudio_record_count": strict_boundary["true_mmaudio_audio_artifact_count"],
            "raw_payload_path": str(RAW_PAYLOAD),
            "recommended_java_behavior": (
                "Expose this as a result-card endpoint and keep raw fallback-aware counts as context, "
                "not as true-MMAudio batch evidence."
            ),
            "recommended_cloud_behavior": (
                "Use this bridge as a Friday demo gate seed with single=true, batch=false, "
                "fullRanking=false, productionSlo=false."
            ),
        },
        "source_artifact_count": len(raw_source_artifacts)
        if isinstance(raw_source_artifacts, list)
        else None,
    }

    OUT_BRIDGE.parent.mkdir(parents=True, exist_ok=True)
    OUT_GUARD.parent.mkdir(parents=True, exist_ok=True)
    OUT_ARTIFACT.parent.mkdir(parents=True, exist_ok=True)

    OUT_BRIDGE.write_text(json.dumps(bridge, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_GUARD.write_text(json.dumps(claim_guard, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_ARTIFACT.write_text(json.dumps(bridge["platform_result_card"], ensure_ascii=False, indent=2), encoding="utf-8")

    if not strict_boundary["true_mmaudio_single_success"]:
        raise RuntimeError("Bridge generated, but true_mmaudio_single_success is false")

    print("WROTE", OUT_BRIDGE)
    print("WROTE", OUT_GUARD)
    print("WROTE", OUT_ARTIFACT)
    print("STATUS=", bridge["platform_result_card"]["status"])
    print("SAFE_TRUE_MMAUDIO_RECORD_COUNT=", bridge["platform_result_card"]["safe_true_mmaudio_record_count"])
    print("RAW_CANDIDATE_RECORD_COUNT=", bridge["platform_result_card"]["raw_candidate_record_count"])


if __name__ == "__main__":
    main()