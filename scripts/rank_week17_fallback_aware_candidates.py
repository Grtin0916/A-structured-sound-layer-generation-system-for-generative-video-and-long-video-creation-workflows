#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import wave
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(".")
IN_CSV = ROOT / "reports" / "week17_reranker_input_for_0702.csv"
DECISION_JSON = ROOT / "reports" / "week17_model_race_decision_20260701.json"

OUT_DIR = ROOT / "artifacts" / "model_race" / "week17_fallback_aware_reranker"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANKING_CSV = ROOT / "reports" / "week17_fallback_aware_ranking_20260701.csv"
RANKING_JSON = ROOT / "reports" / "week17_fallback_aware_ranking_20260701.json"
WINNERS_JSON = ROOT / "reports" / "week17_fallback_aware_winners_20260701.json"
REPAIR_CSV = ROOT / "reports" / "week17_fallback_aware_repair_queue_20260701.csv"
REPAIR_JSON = ROOT / "reports" / "week17_fallback_aware_repair_queue_20260701.json"
JAVA_PAYLOAD_JSON = ROOT / "reports" / "week17_model_race_java_payload_20260701.json"
GALLERY_MD = OUT_DIR / "fallback_aware_gallery_20260701.md"
SUMMARY_JSON = OUT_DIR / "fallback_aware_reranker_summary_20260701.json"


SOURCE_CANONICAL = {
    "fallback_mmaudio_baseline": "fallback_mmaudio",
    "fallback_mmaudio_metrics": "fallback_mmaudio",
    "control_seed": "control_rule_foley",
    "repair_seed": "repair_candidate",
    "control_or_repair_candidate": "control_or_repair_candidate",
    "true_mmaudio_attempt": "true_mmaudio_blocker",
}

SOURCE_WEIGHT = {
    "fallback_mmaudio": 0.72,
    "control_rule_foley": 0.62,
    "control_or_repair_candidate": 0.58,
    "repair_candidate": 0.48,
    "true_mmaudio_blocker": 0.00,
}


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_error": str(exc)}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def as_bool(v: Any) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "y", "ok", "pass"}


def safe_float(v: Any, default: float | None = None) -> float | None:
    try:
        if v is None or str(v).strip() == "":
            return default
        return float(v)
    except Exception:
        return default


def wav_stats(path_str: str) -> dict[str, Any]:
    if not path_str:
        return {
            "path_exists": False,
            "duration_sec_probe": "",
            "audio_probe_error": "empty_path",
        }

    path = Path(path_str)
    if not path.exists():
        return {
            "path_exists": False,
            "duration_sec_probe": "",
            "audio_probe_error": "path_missing",
        }

    if path.suffix.lower() != ".wav":
        return {
            "path_exists": True,
            "duration_sec_probe": "",
            "audio_probe_error": "not_wav",
        }

    try:
        with wave.open(str(path), "rb") as wf:
            nframes = wf.getnframes()
            framerate = wf.getframerate()
            nchannels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            duration = nframes / float(framerate) if framerate else 0.0

            # 只采样前 10 秒，避免大文件拖慢。
            max_frames = min(nframes, int(framerate * 10))
            raw = wf.readframes(max_frames)

        if sampwidth == 2 and raw:
            import array
            arr = array.array("h")
            arr.frombytes(raw)
            if nchannels > 1:
                vals = arr[::nchannels]
            else:
                vals = arr
            if len(vals) == 0:
                peak = 0.0
                rms = 0.0
                silence_ratio = 1.0
            else:
                norm = 32768.0
                peak = max(abs(x) for x in vals) / norm
                rms = math.sqrt(sum((x / norm) ** 2 for x in vals) / len(vals))
                silence_ratio = sum(1 for x in vals if abs(x) / norm < 1e-4) / len(vals)
        else:
            peak = ""
            rms = ""
            silence_ratio = ""

        return {
            "path_exists": True,
            "duration_sec_probe": round(duration, 3),
            "peak_probe": peak if peak == "" else round(float(peak), 6),
            "rms_probe": rms if rms == "" else round(float(rms), 6),
            "silence_ratio_probe": silence_ratio if silence_ratio == "" else round(float(silence_ratio), 6),
            "audio_probe_error": "",
        }
    except Exception as exc:
        return {
            "path_exists": True,
            "duration_sec_probe": "",
            "audio_probe_error": type(exc).__name__ + ": " + str(exc)[:160],
        }


def canonical_key(row: dict[str, str]) -> tuple[str, str, str]:
    case_id = row.get("case_id", "").strip() or "unknown_case"
    candidate_id = row.get("candidate_id", "").strip()
    source = SOURCE_CANONICAL.get(row.get("source_label", "").strip(), row.get("source_label", "").strip())

    # fallback_mmaudio_baseline 和 fallback_mmaudio_metrics 是同一候选的不同视图，必须合并。
    if source == "fallback_mmaudio":
        return case_id, candidate_id, source

    # repair/control 有时 candidate_id 不稳定，artifact_path 更可信。
    artifact = row.get("artifact_path", "").strip()
    if artifact:
        return case_id, artifact, source

    return case_id, candidate_id, source


def merge_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[canonical_key(row)].append(row)

    merged = []
    for (_, _, canonical_source), items in grouped.items():
        base: dict[str, Any] = {}
        for item in items:
            for k, v in item.items():
                if str(v).strip() and not str(base.get(k, "")).strip():
                    base[k] = v

        base["canonical_source"] = canonical_source
        base["merge_count"] = len(items)

        # 只要任一视图有真实路径，就补上。
        for item in items:
            p = item.get("artifact_path", "").strip()
            if p and Path(p).exists():
                base["artifact_path"] = p
                break

        merged.append(base)

    return merged


def compute_score(row: dict[str, Any]) -> tuple[float, str, str]:
    canonical_source = row.get("canonical_source", "")
    readable = as_bool(row.get("readable"))
    eligible = as_bool(row.get("eligible_for_rerank"))
    video_conditioned = as_bool(row.get("video_conditioned"))

    if canonical_source == "true_mmaudio_blocker":
        return 0.0, "blocked", "true_mmaudio_runtime_blocked_not_eligible"

    if not eligible or not readable:
        return 0.0, "rejected", row.get("reject_reason") or "not_readable_or_not_eligible"

    stats = wav_stats(str(row.get("artifact_path", "")))

    duration = safe_float(row.get("duration_sec"), None)
    if duration is None:
        duration = safe_float(stats.get("duration_sec_probe"), None)

    peak = safe_float(row.get("peak"), None)
    if peak is None:
        peak = safe_float(stats.get("peak_probe"), None)

    rms = safe_float(row.get("rms"), None)
    if rms is None:
        rms = safe_float(stats.get("rms_probe"), None)

    silence = safe_float(row.get("silence_ratio"), None)
    if silence is None:
        silence = safe_float(stats.get("silence_ratio_probe"), None)

    score = SOURCE_WEIGHT.get(canonical_source, 0.35)

    if stats.get("path_exists"):
        score += 0.10
    if duration is not None:
        if 4.0 <= duration <= 16.0:
            score += 0.08
        else:
            score -= 0.05
    if peak is not None:
        if 0.02 <= peak <= 0.98:
            score += 0.08
        elif peak > 0.98:
            score -= 0.10
        else:
            score -= 0.04
    if rms is not None:
        if 0.002 <= rms <= 0.35:
            score += 0.08
        else:
            score -= 0.05
    if silence is not None:
        if silence <= 0.25:
            score += 0.08
        elif silence > 0.60:
            score -= 0.12

    if canonical_source == "repair_candidate":
        score -= 0.05
    if video_conditioned:
        score += 0.20

    score = round(max(0.0, min(1.0, score)), 4)

    reason_parts = [
        f"source={canonical_source}",
        f"readable={readable}",
        f"path_exists={stats.get('path_exists')}",
        f"duration={duration}",
        f"peak={peak}",
        f"rms={rms}",
        f"silence={silence}",
        f"merge_count={row.get('merge_count')}",
    ]

    return score, "ranked", "; ".join(reason_parts)


def main() -> None:
    raw_rows = read_csv(IN_CSV)
    merged = merge_rows(raw_rows)

    ranked_rows = []
    for row in merged:
        score, status, reason = compute_score(row)
        out = dict(row)
        out["rank_score"] = score
        out["rank_status"] = status
        out["rank_reason"] = reason

        stats = wav_stats(str(out.get("artifact_path", "")))
        for k, v in stats.items():
            out.setdefault(k, v)

        ranked_rows.append(out)

    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in ranked_rows:
        by_case[str(r.get("case_id", "unknown_case"))].append(r)

    winners = []
    rejected = []
    repair_queue = []

    for case_id, items in sorted(by_case.items()):
        eligible = [r for r in items if r.get("rank_status") == "ranked"]
        eligible.sort(key=lambda x: float(x.get("rank_score", 0.0)), reverse=True)

        winner = eligible[0] if eligible else None
        if winner:
            winner["selection"] = "winner"
            winners.append(winner)

        for r in items:
            if winner is not None and r is winner:
                continue
            r["selection"] = "rejected"
            rejected.append(r)

        if winner is None:
            repair_queue.append({
                "case_id": case_id,
                "repair_priority": "P0",
                "repair_action": "generate_or_recover_candidate",
                "reason": "no_eligible_candidate",
            })
        else:
            low_quality = float(winner.get("rank_score", 0.0)) < 0.72
            if low_quality:
                repair_queue.append({
                    "case_id": case_id,
                    "candidate_id": winner.get("candidate_id", ""),
                    "repair_priority": "P1",
                    "repair_action": "event_gain_or_loudness_normalize_then_recheck",
                    "reason": f"winner_score_below_threshold:{winner.get('rank_score')}",
                })

    fieldnames = [
        "case_id",
        "candidate_id",
        "canonical_source",
        "source_label",
        "video_conditioned",
        "readable",
        "eligible_for_rerank",
        "rank_score",
        "rank_status",
        "selection",
        "rank_reason",
        "reject_reason",
        "repair_action",
        "artifact_path",
        "duration_sec",
        "duration_sec_probe",
        "rms",
        "rms_probe",
        "peak",
        "peak_probe",
        "silence_ratio",
        "silence_ratio_probe",
        "path_exists",
        "audio_probe_error",
        "merge_count",
    ]

    ranked_rows.sort(
        key=lambda r: (
            str(r.get("case_id", "")),
            -float(r.get("rank_score", 0.0)),
            str(r.get("candidate_id", "")),
        )
    )

    with RANKING_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ranked_rows)

    ranking_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_csv": str(IN_CSV),
        "raw_row_count": len(raw_rows),
        "canonical_candidate_count": len(ranked_rows),
        "case_count": len(by_case),
        "winner_count": len(winners),
        "rejected_count": len(rejected),
        "repair_queue_count": len(repair_queue),
        "claim_boundary": {
            "true_mmaudio_success": False,
            "fallback_aware_reranker_ready": len(winners) >= 6,
            "can_platform_consume": len(winners) >= 6,
        },
        "winners": winners,
        "outputs": {
            "ranking_csv": str(RANKING_CSV),
            "ranking_json": str(RANKING_JSON),
            "winners_json": str(WINNERS_JSON),
            "repair_csv": str(REPAIR_CSV),
            "repair_json": str(REPAIR_JSON),
            "java_payload_json": str(JAVA_PAYLOAD_JSON),
            "gallery_md": str(GALLERY_MD),
        },
    }

    RANKING_JSON.write_text(json.dumps(ranking_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    WINNERS_JSON.write_text(json.dumps({"winners": winners}, indent=2, ensure_ascii=False), encoding="utf-8")

    with REPAIR_CSV.open("w", encoding="utf-8", newline="") as f:
        repair_fields = ["case_id", "candidate_id", "repair_priority", "repair_action", "reason"]
        writer = csv.DictWriter(f, fieldnames=repair_fields)
        writer.writeheader()
        writer.writerows(repair_queue)

    REPAIR_JSON.write_text(json.dumps({"repair_queue": repair_queue}, indent=2, ensure_ascii=False), encoding="utf-8")

    java_payload = {
        "artifact_type": "week17_fallback_aware_model_race_result",
        "generated_at": ranking_payload["generated_at"],
        "source": "mainbase",
        "true_mmaudio_status": "blocked_by_torch_torchaudio_abi",
        "case_count": len(by_case),
        "winner_count": len(winners),
        "canonical_candidate_count": len(ranked_rows),
        "ranking_csv": str(RANKING_CSV),
        "winners_json": str(WINNERS_JSON),
        "repair_queue_json": str(REPAIR_JSON),
        "items": [
            {
                "case_id": w.get("case_id", ""),
                "winner_candidate_id": w.get("candidate_id", ""),
                "source": w.get("canonical_source", ""),
                "score": w.get("rank_score", 0.0),
                "artifact_path": w.get("artifact_path", ""),
                "why_selected": w.get("rank_reason", ""),
            }
            for w in winners
        ],
    }
    JAVA_PAYLOAD_JSON.write_text(json.dumps(java_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    gallery_lines = [
        "# Week17 fallback-aware model race gallery",
        "",
        f"- generated_at: {ranking_payload['generated_at']}",
        f"- raw_row_count: {len(raw_rows)}",
        f"- canonical_candidate_count: {len(ranked_rows)}",
        f"- winner_count: {len(winners)}",
        f"- true_mmaudio_success: false",
        "",
        "## Winners",
        "",
    ]
    for w in winners:
        gallery_lines.append(
            f"- `{w.get('case_id')}` -> `{w.get('candidate_id')}` "
            f"source=`{w.get('canonical_source')}` score=`{w.get('rank_score')}` "
            f"path=`{w.get('artifact_path')}`"
        )

    gallery_lines.extend(["", "## Repair queue", ""])
    for r in repair_queue:
        gallery_lines.append(
            f"- `{r.get('case_id')}` priority=`{r.get('repair_priority')}` "
            f"action=`{r.get('repair_action')}` reason=`{r.get('reason')}`"
        )

    GALLERY_MD.write_text("\n".join(gallery_lines) + "\n", encoding="utf-8")

    SUMMARY_JSON.write_text(
        json.dumps(
            {
                "generated_at": ranking_payload["generated_at"],
                "status": "GREEN_FALLBACK_AWARE_RERANKER_READY" if len(winners) >= 6 else "RED_NO_6_WINNERS",
                "raw_row_count": len(raw_rows),
                "canonical_candidate_count": len(ranked_rows),
                "case_count": len(by_case),
                "winner_count": len(winners),
                "repair_queue_count": len(repair_queue),
                "outputs": ranking_payload["outputs"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(json.dumps(json.loads(SUMMARY_JSON.read_text(encoding="utf-8")), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()