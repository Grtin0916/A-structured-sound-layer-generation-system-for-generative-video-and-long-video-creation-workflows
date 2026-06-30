#!/usr/bin/env python3
from __future__ import annotations

import csv
import html
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
EXP = ROOT / "experiments" / "mmaudio_baseline_2026_06_30"
CANDIDATES = EXP / "candidates"
OUT = ROOT / "artifacts" / "model_runs" / "week17_mmaudio"


def rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def fnum(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def boolish(x: Any) -> bool:
    return str(x).strip().lower() in {"1", "true", "yes", "y"}


def score_candidate(row: Dict[str, str]) -> tuple[float, List[str]]:
    reasons: List[str] = []
    score = 0.0

    readable = boolish(row.get("readable"))
    fallback = boolish(row.get("fallback_used"))
    video_conditioned = boolish(row.get("video_conditioned"))

    rms = fnum(row.get("rms"))
    peak = fnum(row.get("peak"))
    clip_rate = fnum(row.get("clip_rate"))
    silence_ratio = fnum(row.get("silence_ratio"))
    duration_sec = fnum(row.get("duration_sec"))

    if readable:
        score += 40.0
        reasons.append("readable_wav")
    else:
        score -= 100.0
        reasons.append("unreadable_wav")

    if 5.0 <= duration_sec <= 15.0:
        score += 10.0
        reasons.append("duration_in_demo_range")
    else:
        score -= 10.0
        reasons.append("duration_out_of_demo_range")

    if 0.008 <= rms <= 0.08:
        score += 15.0
        reasons.append("rms_reasonable")
    else:
        score -= 8.0
        reasons.append("rms_outlier")

    if 0.08 <= peak <= 0.90:
        score += 15.0
        reasons.append("peak_reasonable")
    else:
        score -= 8.0
        reasons.append("peak_outlier")

    if clip_rate <= 0.001:
        score += 10.0
        reasons.append("no_clipping")
    else:
        score -= 20.0
        reasons.append("clipping_detected")

    if silence_ratio <= 0.20:
        score += 10.0
        reasons.append("not_mostly_silent")
    else:
        score -= 20.0
        reasons.append("too_much_silence")

    # Do not pretend fallback is V2A. Penalize it lightly so future true MMAudio wins naturally.
    if fallback:
        score -= 5.0
        reasons.append("fallback_control_not_true_v2a")
    if not video_conditioned:
        score -= 5.0
        reasons.append("video_conditioned_false")

    # Prefer the more structured variant only if it did not break quality.
    if row.get("prompt_variant") == "dss_avoid_priority":
        score += 1.0
        reasons.append("structured_dss_variant")

    return round(score, 4), reasons


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    metrics_path = REPORTS / "mmaudio_baseline_metrics.csv"
    summary_path = REPORTS / "mmaudio_baseline_summary.json"
    prompt_path = REPORTS / "mmaudio_prompt_manifest.json"

    if not metrics_path.exists():
        raise SystemExit(f"METRICS_NOT_FOUND: {metrics_path}")

    rows = list(csv.DictReader(metrics_path.open("r", encoding="utf-8")))
    if not rows:
        raise SystemExit("METRICS_EMPTY")

    prompt_manifest = json.loads(prompt_path.read_text(encoding="utf-8")) if prompt_path.exists() else {"prompts": []}
    prompt_by_id = {p.get("candidate_id"): p for p in prompt_manifest.get("prompts", [])}

    bad_prompts = []
    for p in prompt_manifest.get("prompts", []):
        text = p.get("prompt", "")
        if '"avoid": [' in text or "Avoid: [" in text:
            bad_prompts.append(p.get("candidate_id"))

    ranked_rows: List[Dict[str, Any]] = []
    by_case: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for row in rows:
        score, reasons = score_candidate(row)
        item: Dict[str, Any] = dict(row)
        item["score"] = score
        item["score_reasons"] = "|".join(reasons)
        item["prompt"] = prompt_by_id.get(row.get("candidate_id"), {}).get("prompt", "")
        item["output_wav_rel"] = rel(row.get("output_wav", ""))
        by_case[row["case_id"]].append(item)

    winners: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    repair_queue: List[Dict[str, Any]] = []

    for case_id, items in sorted(by_case.items()):
        items_sorted = sorted(items, key=lambda x: x["score"], reverse=True)
        winner = items_sorted[0]
        winner["rank"] = 1
        winner["selection_status"] = "provisional_winner_control_baseline"
        winners.append(winner)

        for idx, item in enumerate(items_sorted[1:], start=2):
            item["rank"] = idx
            item["selection_status"] = "rejected_lower_score"
            item["rejected_reason"] = (
                "Lower baseline score than winner; keep as runner-up for future MMAudio comparison."
            )
            rejected.append(item)

        # Because all current outputs are fallback, every case remains a true-model repair target.
        if boolish(winner.get("fallback_used")):
            repair_queue.append({
                "case_id": case_id,
                "winner_candidate_id": winner["candidate_id"],
                "repair_reason": "Current winner is fallback control audio; rerun with true local/remote MMAudio when available.",
                "priority": "high",
            })

        ranked_rows.extend(items_sorted)

    ranking_csv = REPORTS / "mmaudio_baseline_ranking.csv"
    with ranking_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "case_id",
            "candidate_id",
            "rank",
            "selection_status",
            "score",
            "prompt_variant",
            "status",
            "readable",
            "fallback_used",
            "video_conditioned",
            "duration_sec",
            "rms",
            "peak",
            "clip_rate",
            "silence_ratio",
            "output_wav_rel",
            "score_reasons",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ranked_rows)

    winners_json = REPORTS / "mmaudio_baseline_winners.json"
    repair_json = REPORTS / "mmaudio_baseline_repair_queue.json"
    payload_json = REPORTS / "mmaudio_baseline_java_seed_payload.json"
    boundary_json = REPORTS / "mmaudio_baseline_boundary.json"

    winners_json.write_text(json.dumps(winners, indent=2, ensure_ascii=False), encoding="utf-8")
    repair_json.write_text(json.dumps(repair_queue, indent=2, ensure_ascii=False), encoding="utf-8")

    local_summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}

    boundary = {
        "decision": local_summary.get("decision"),
        "case_count": len(by_case),
        "candidate_count": len(rows),
        "winner_count": len(winners),
        "rejected_count": len(rejected),
        "repair_queue_count": len(repair_queue),
        "bad_prompt_count": len(bad_prompts),
        "all_outputs_are_fallback": all(boolish(r.get("fallback_used")) for r in rows),
        "true_mmaudio_generated_count": sum(1 for r in rows if r.get("status") == "generated"),
        "claim_boundary": {
            "can_claim_readable_candidate_audio": True,
            "can_claim_dss_conditioned_control_baseline": True,
            "can_claim_true_mmaudio_v2a_success": False,
            "can_claim_video_synchronized_quality": False,
        },
        "bad_prompts": bad_prompts,
    }
    boundary_json.write_text(json.dumps(boundary, indent=2, ensure_ascii=False), encoding="utf-8")

    payload = {
        "name": "week17_mmaudio_baseline_java_seed_payload",
        "source": "mainbase",
        "boundary": boundary,
        "winners": [
            {
                "case_id": w["case_id"],
                "candidate_id": w["candidate_id"],
                "rank": w["rank"],
                "score": w["score"],
                "status": w["status"],
                "fallback_used": boolish(w["fallback_used"]),
                "video_conditioned": boolish(w["video_conditioned"]),
                "output_wav": w["output_wav_rel"],
                "score_reasons": w["score_reasons"],
            }
            for w in winners
        ],
        "repair_queue": repair_queue,
    }
    payload_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    index_path = OUT / "audio_index.html"
    cards = []
    for w in winners:
        cards.append(
            f"""
            <section style="border:1px solid #ccc;padding:12px;margin:12px 0;">
              <h3>{html.escape(w['case_id'])}</h3>
              <p><b>Winner:</b> {html.escape(w['candidate_id'])}</p>
              <p><b>Score:</b> {w['score']} | <b>Status:</b> {html.escape(w['status'])} |
                 <b>Fallback:</b> {html.escape(str(w['fallback_used']))} |
                 <b>Video-conditioned:</b> {html.escape(str(w['video_conditioned']))}</p>
              <p><b>Reasons:</b> {html.escape(w['score_reasons'])}</p>
              <audio controls src="../../../../{html.escape(w['output_wav_rel'])}"></audio>
            </section>
            """
        )

    html_text = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Week17 MMAudio Baseline Audio Index</title>
</head>
<body>
  <h1>Week17 MMAudio Baseline Audio Index</h1>
  <p>
    This page lists provisional winners from the current baseline run.
    Boundary: current winners are fallback control audio, not true MMAudio V2A outputs.
  </p>
  <pre>{html.escape(json.dumps(boundary, indent=2, ensure_ascii=False))}</pre>
  {''.join(cards)}
</body>
</html>
"""
    index_path.write_text(html_text, encoding="utf-8")

    result = {
        "decision": "PASS_BASELINE_RESULT_PACK_READY" if len(winners) == 6 and len(bad_prompts) == 0 else "PARTIAL_BASELINE_RESULT_PACK",
        "case_count": len(by_case),
        "candidate_count": len(rows),
        "winner_count": len(winners),
        "rejected_count": len(rejected),
        "repair_queue_count": len(repair_queue),
        "bad_prompt_count": len(bad_prompts),
        "outputs": {
            "ranking_csv": rel(ranking_csv),
            "winners_json": rel(winners_json),
            "repair_queue_json": rel(repair_json),
            "java_seed_payload": rel(payload_json),
            "boundary_json": rel(boundary_json),
            "audio_index_html": rel(index_path),
        },
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["decision"] == "PASS_BASELINE_RESULT_PACK_READY" else 1


if __name__ == "__main__":
    raise SystemExit(main())