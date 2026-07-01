#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(".")
IN_CSV = ROOT / "reports/week17_fallback_aware_ranking_20260701.csv"

OUT_DIR = ROOT / "artifacts/model_race/week17_calibrated_reranker"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = ROOT / "reports/week17_calibrated_ranking_20260701.csv"
OUT_JSON = ROOT / "reports/week17_calibrated_ranking_20260701.json"
WINNERS_JSON = ROOT / "reports/week17_calibrated_winners_20260701.json"
REJECTED_JSON = ROOT / "reports/week17_calibrated_rejections_20260701.json"
SUMMARY_JSON = OUT_DIR / "calibrated_reranker_summary_20260701.json"
GALLERY_MD = OUT_DIR / "calibrated_gallery_20260701.md"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def f(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        v = row.get(key, "")
        if v is None or str(v).strip() == "":
            return default
        return float(v)
    except Exception:
        return default


def b(row: dict[str, str], key: str) -> bool:
    return str(row.get(key, "")).strip().lower() in {"1", "true", "yes", "ok", "pass"}


def calibrated_score(row: dict[str, str]) -> tuple[float, list[str]]:
    source = row.get("canonical_source", "")
    candidate = row.get("candidate_id", "")
    reasons: list[str] = []

    if source == "true_mmaudio_blocker":
        return 0.0, ["true_mmaudio_blocked_by_torch_torchaudio_abi"]

    if not b(row, "readable") or not b(row, "eligible_for_rerank"):
        return 0.05, ["not_readable_or_not_eligible"]

    # 非饱和基础分：避免所有 fallback 直接顶到 1.0。
    source_base = {
        "fallback_mmaudio": 0.66,
        "control_rule_foley": 0.54,
        "control_or_repair_candidate": 0.50,
        "repair_candidate": 0.46,
    }.get(source, 0.40)

    score = source_base
    reasons.append(f"source_base={source_base}")

    # prompt variant：avoid_priority 代表更强 DSS 约束，但不能给满。
    if "avoid_priority" in candidate:
        score += 0.065
        reasons.append("dss_avoid_priority_bonus=0.065")
    elif "dss_compact" in candidate:
        score += 0.035
        reasons.append("dss_compact_bonus=0.035")
    elif "control_rule" in candidate:
        score += 0.020
        reasons.append("control_rule_interpretable_bonus=0.020")

    path_exists = b(row, "path_exists")
    if path_exists:
        score += 0.045
        reasons.append("artifact_exists_bonus=0.045")
    else:
        score -= 0.100
        reasons.append("artifact_missing_penalty=-0.100")

    duration = f(row, "duration_sec_probe", f(row, "duration_sec", 0.0))
    if 5.0 <= duration <= 15.5:
        score += 0.050
        reasons.append(f"duration_in_demo_range_bonus=0.050,duration={duration}")
    elif duration > 0:
        score -= 0.030
        reasons.append(f"duration_out_of_range_penalty=-0.030,duration={duration}")
    else:
        score -= 0.040
        reasons.append("duration_unknown_penalty=-0.040")

    peak = f(row, "peak_probe", f(row, "peak", -1.0))
    if 0.05 <= peak <= 0.95:
        score += 0.045
        reasons.append(f"peak_healthy_bonus=0.045,peak={peak}")
    elif peak > 0.95:
        score -= 0.070
        reasons.append(f"near_clip_penalty=-0.070,peak={peak}")
    elif peak >= 0:
        score -= 0.030
        reasons.append(f"weak_peak_penalty=-0.030,peak={peak}")
    else:
        score -= 0.030
        reasons.append("peak_unknown_penalty=-0.030")

    rms = f(row, "rms_probe", f(row, "rms", -1.0))
    if 0.005 <= rms <= 0.25:
        score += 0.045
        reasons.append(f"rms_healthy_bonus=0.045,rms={rms}")
    elif rms >= 0:
        score -= 0.035
        reasons.append(f"rms_outlier_penalty=-0.035,rms={rms}")
    else:
        score -= 0.025
        reasons.append("rms_unknown_penalty=-0.025")

    silence = f(row, "silence_ratio_probe", f(row, "silence_ratio", -1.0))
    if 0 <= silence <= 0.18:
        score += 0.035
        reasons.append(f"low_silence_bonus=0.035,silence={silence}")
    elif silence > 0.55:
        score -= 0.090
        reasons.append(f"high_silence_penalty=-0.090,silence={silence}")
    elif silence >= 0:
        reasons.append(f"silence_neutral={silence}")
    else:
        score -= 0.020
        reasons.append("silence_unknown_penalty=-0.020")

    # 让分数分布保留不确定性，不再 1.0 饱和。
    score = max(0.0, min(0.93, score))
    return round(score, 4), reasons


def main() -> None:
    rows = read_csv(IN_CSV)
    calibrated = []

    for row in rows:
        score, reasons = calibrated_score(row)
        out = dict(row)
        out["calibrated_score"] = score
        out["calibrated_reason"] = " | ".join(reasons)
        calibrated.append(out)

    by_case = defaultdict(list)
    for row in calibrated:
        by_case[row.get("case_id", "unknown_case")].append(row)

    winners = []
    rejections = []

    for case_id, items in sorted(by_case.items()):
        ranked = sorted(items, key=lambda r: float(r.get("calibrated_score", 0.0)), reverse=True)
        for rank, item in enumerate(ranked, start=1):
            item["calibrated_rank"] = rank
            if rank == 1:
                item["calibrated_selection"] = "winner"
                winners.append(item)
            elif rank == 2:
                item["calibrated_selection"] = "runner_up"
                rejections.append({
                    "case_id": case_id,
                    "candidate_id": item.get("candidate_id", ""),
                    "source": item.get("canonical_source", ""),
                    "calibrated_score": item.get("calibrated_score"),
                    "rejected_as": "runner_up",
                    "reject_reason": "lower_calibrated_score_than_winner",
                    "candidate_reason": item.get("calibrated_reason", ""),
                })
            else:
                item["calibrated_selection"] = "rejected"
                rejections.append({
                    "case_id": case_id,
                    "candidate_id": item.get("candidate_id", ""),
                    "source": item.get("canonical_source", ""),
                    "calibrated_score": item.get("calibrated_score"),
                    "rejected_as": "rejected",
                    "reject_reason": "lower_source_or_audio_health_score",
                    "candidate_reason": item.get("calibrated_reason", ""),
                })

    fieldnames = list(calibrated[0].keys()) if calibrated else []
    preferred = [
        "case_id", "candidate_id", "canonical_source", "calibrated_rank",
        "calibrated_selection", "calibrated_score", "calibrated_reason",
        "rank_score", "source_label", "artifact_path", "duration_sec_probe",
        "peak_probe", "rms_probe", "silence_ratio_probe", "path_exists"
    ]
    fieldnames = preferred + [x for x in fieldnames if x not in preferred]

    calibrated.sort(key=lambda r: (r.get("case_id", ""), int(r.get("calibrated_rank", 999))))

    with OUT_CSV.open("w", encoding="utf-8", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(calibrated)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "GREEN_CALIBRATED_RERANKER_READY" if len(winners) == 6 else "RED_CALIBRATED_RERANKER_INCOMPLETE",
        "input_csv": str(IN_CSV),
        "candidate_count": len(calibrated),
        "case_count": len(by_case),
        "winner_count": len(winners),
        "score_min": min(float(r["calibrated_score"]) for r in calibrated) if calibrated else None,
        "score_max": max(float(r["calibrated_score"]) for r in calibrated) if calibrated else None,
        "score_unique_count": len({r["calibrated_score"] for r in calibrated}),
        "true_mmaudio_success": False,
        "claim_boundary": {
            "true_mmaudio_v2a_success": False,
            "calibrated_reranker_ready": len(winners) == 6,
            "scores_are_relative_not_human_preference": True,
        },
        "winners": [
            {
                "case_id": w.get("case_id"),
                "candidate_id": w.get("candidate_id"),
                "source": w.get("canonical_source"),
                "calibrated_score": w.get("calibrated_score"),
                "artifact_path": w.get("artifact_path"),
                "why_selected": w.get("calibrated_reason"),
            }
            for w in winners
        ],
        "outputs": {
            "ranking_csv": str(OUT_CSV),
            "ranking_json": str(OUT_JSON),
            "winners_json": str(WINNERS_JSON),
            "rejections_json": str(REJECTED_JSON),
            "gallery_md": str(GALLERY_MD),
        },
    }

    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    WINNERS_JSON.write_text(json.dumps({"winners": payload["winners"]}, indent=2, ensure_ascii=False), encoding="utf-8")
    REJECTED_JSON.write_text(json.dumps({"rejections": rejections}, indent=2, ensure_ascii=False), encoding="utf-8")
    SUMMARY_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Week17 calibrated fallback-aware reranker",
        "",
        f"- status: {payload['status']}",
        f"- candidate_count: {payload['candidate_count']}",
        f"- case_count: {payload['case_count']}",
        f"- winner_count: {payload['winner_count']}",
        f"- score_range: {payload['score_min']} ~ {payload['score_max']}",
        f"- score_unique_count: {payload['score_unique_count']}",
        f"- true_mmaudio_v2a_success: false",
        "",
        "## Winners",
        "",
    ]
    for w in payload["winners"]:
        lines.append(
            f"- `{w['case_id']}` -> `{w['candidate_id']}` "
            f"source=`{w['source']}` score=`{w['calibrated_score']}`"
        )
    GALLERY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()