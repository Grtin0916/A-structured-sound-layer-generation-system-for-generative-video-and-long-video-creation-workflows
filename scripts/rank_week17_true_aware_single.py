import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone

FALLBACK_RANKING = Path("reports/week17_fallback_aware_ranking_20260701.json")
FALLBACK_WINNERS = Path("reports/week17_fallback_aware_winners_20260701.json")
EVIDENCE = Path("reports/week17_true_mmaudio_single_candidate_evidence_20260701.json")

OUT_DIR = Path("artifacts/model_race/week17_true_aware_reranker")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANKING_OUT = Path("reports/week17_true_aware_ranking_20260701.json")
WINNERS_OUT = Path("reports/week17_true_aware_winners_20260701.json")
SUMMARY_OUT = OUT_DIR / "true_aware_reranker_summary_20260701.json"
GALLERY_OUT = OUT_DIR / "true_aware_gallery_20260701.md"

def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"missing required file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))

def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

fallback_ranking = load_json(FALLBACK_RANKING)
fallback_winners_payload = load_json(FALLBACK_WINNERS)
evidence = load_json(EVIDENCE)

base_winners = fallback_winners_payload.get("winners") or fallback_ranking.get("winners") or []
if len(base_winners) != 6:
    raise RuntimeError(f"expected 6 fallback-aware winners, got {len(base_winners)}")

audio_path = Path(evidence["audio_path"])
if not audio_path.exists():
    raise FileNotFoundError(f"true candidate audio missing: {audio_path}")

if evidence.get("status") != "GREEN_TRUE_MMAUDIO_SINGLE_CANDIDATE_READY":
    raise RuntimeError(f"unexpected evidence status: {evidence.get('status')}")

if not evidence.get("readable") or not evidence.get("video_conditioned") or not evidence.get("eligible_for_rerank"):
    raise RuntimeError("true evidence is not rerank eligible")

case_id = evidence["case_id"]
candidate_id = evidence["candidate_id"]

true_item = {
    "case_id": case_id,
    "candidate_id": candidate_id,
    "source_label": "true_mmaudio_attempt",
    "source": "true_mmaudio_attempt",
    "canonical_source": "true_mmaudio",
    "video_conditioned": True,
    "readable": True,
    "eligible_for_rerank": True,
    "winner_hint": True,
    "artifact_path": str(audio_path),
    "audio_path": str(audio_path),
    "sha256": sha256_file(audio_path),
    "bytes": audio_path.stat().st_size,
    "samplerate": evidence.get("samplerate"),
    "channels": evidence.get("channels"),
    "frames": evidence.get("frames"),
    "duration_sec": evidence.get("duration_sec"),
    "format": evidence.get("format"),
    "subtype": evidence.get("subtype"),
    "rank_score": 1.05,
    "calibrated_score": 1.05,
    "rank_status": "true_single_promoted",
    "rank_reason": "true video-conditioned MMAudio single candidate; readable FLAC; generated from input video and prompt; score is source-priority, not human preference",
    "selection": "winner",
    "claim_boundary": {
        "true_mmaudio_single_success": True,
        "batch_true_mmaudio_success": False,
        "fallback_candidate": False,
        "production_slo_claim": False
    }
}

compact_candidates = []
winners = []
displaced = []

for item in base_winners:
    item = dict(item)
    if item.get("case_id") == case_id:
        item["selection"] = "displaced_by_true_mmaudio_single"
        item["displaced_by"] = candidate_id
        displaced.append(item)
        compact_candidates.append(item)
        compact_candidates.append(true_item)
        winners.append(true_item)
    else:
        item["selection"] = "winner"
        compact_candidates.append(item)
        winners.append(item)

case_set = {x.get("case_id") for x in compact_candidates if x.get("case_id")}
winner_case_set = {x.get("case_id") for x in winners if x.get("case_id")}

if len(case_set) != 6:
    raise RuntimeError(f"expected compact case_count=6, got {len(case_set)}")
if len(winners) != 6 or len(winner_case_set) != 6:
    raise RuntimeError(f"expected 6 winners and 6 winner cases, got winners={len(winners)}, cases={len(winner_case_set)}")
if sum(1 for x in winners if x.get("source") == "true_mmaudio_attempt") != 1:
    raise RuntimeError("expected exactly one true_mmaudio_attempt winner")

generated_at = datetime.now(timezone.utc).isoformat()

claim_boundary = {
    "true_mmaudio_single_success": True,
    "batch_true_mmaudio_success": False,
    "fallback_aware_reranker_ready": True,
    "true_aware_compact_reranker_ready": True,
    "full_candidate_ranking_available": False,
    "production_slo_claim": False
}

ranking_payload = {
    "artifact_type": "week17_true_aware_compact_model_race_ranking",
    "generated_at": generated_at,
    "status": "GREEN_TRUE_AWARE_SINGLE_COMPACT_RERANKER_READY",
    "ranking_mode": "compact_winner_level_injection",
    "input_ranking": str(FALLBACK_RANKING),
    "input_winners": str(FALLBACK_WINNERS),
    "input_evidence": str(EVIDENCE),
    "source_raw_row_count": fallback_ranking.get("raw_row_count"),
    "source_canonical_candidate_count": fallback_ranking.get("canonical_candidate_count"),
    "source_case_count": fallback_ranking.get("case_count"),
    "compact_candidates": compact_candidates,
    "claim_boundary": claim_boundary
}

winners_payload = {
    "artifact_type": "week17_true_aware_compact_model_race_winners",
    "generated_at": generated_at,
    "status": "GREEN_TRUE_AWARE_SINGLE_COMPACT_RERANKER_READY",
    "winners": winners,
    "displaced": displaced,
    "claim_boundary": claim_boundary
}

summary = {
    "generated_at": generated_at,
    "status": "GREEN_TRUE_AWARE_SINGLE_COMPACT_RERANKER_READY",
    "ranking_mode": "compact_winner_level_injection",
    "input_ranking": str(FALLBACK_RANKING),
    "input_winners": str(FALLBACK_WINNERS),
    "input_evidence": str(EVIDENCE),
    "source_raw_row_count": fallback_ranking.get("raw_row_count"),
    "source_canonical_candidate_count": fallback_ranking.get("canonical_candidate_count"),
    "source_case_count": fallback_ranking.get("case_count"),
    "input_winner_count": len(base_winners),
    "compact_candidate_count": len(compact_candidates),
    "case_count": len(case_set),
    "winner_count": len(winners),
    "true_mmaudio_single_count": 1,
    "true_mmaudio_winner_count": sum(1 for x in winners if x.get("source") == "true_mmaudio_attempt"),
    "displaced_fallback_winner_count": len(displaced),
    "true_mmaudio_case": case_id,
    "true_mmaudio_candidate_id": candidate_id,
    "true_mmaudio_audio_path": str(audio_path),
    "claim_boundary": claim_boundary,
    "outputs": {
        "ranking_json": str(RANKING_OUT),
        "winners_json": str(WINNERS_OUT),
        "summary_json": str(SUMMARY_OUT),
        "gallery_md": str(GALLERY_OUT)
    }
}

RANKING_OUT.write_text(json.dumps(ranking_payload, ensure_ascii=False, indent=2), encoding="utf-8")
WINNERS_OUT.write_text(json.dumps(winners_payload, ensure_ascii=False, indent=2), encoding="utf-8")
SUMMARY_OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

lines = [
    "# Week17 True-aware Compact Reranker",
    "",
    f"- Status: `{summary['status']}`",
    f"- Ranking mode: `{summary['ranking_mode']}`",
    f"- Source canonical candidate count: `{summary['source_canonical_candidate_count']}`",
    f"- Compact candidate count: `{summary['compact_candidate_count']}`",
    f"- Case count: `{summary['case_count']}`",
    f"- Winner count: `{summary['winner_count']}`",
    f"- True MMAudio winner count: `{summary['true_mmaudio_winner_count']}`",
    f"- True case: `{case_id}`",
    f"- True candidate: `{candidate_id}`",
    f"- Audio: `{audio_path}`",
    "",
    "## Winners",
    ""
]
for w in winners:
    lines.append(f"- `{w.get('case_id')}` -> `{w.get('candidate_id')}` | source=`{w.get('source') or w.get('canonical_source')}` | score=`{w.get('calibrated_score', w.get('rank_score', w.get('score')))}"`)

lines.extend(["", "## Displaced", ""])
for d in displaced:
    lines.append(f"- `{d.get('case_id')}` old winner `{d.get('candidate_id')}` displaced by `{candidate_id}`")

GALLERY_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")

print(json.dumps(summary, ensure_ascii=False, indent=2))
