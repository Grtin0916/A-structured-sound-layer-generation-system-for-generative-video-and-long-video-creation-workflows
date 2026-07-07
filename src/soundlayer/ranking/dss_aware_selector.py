from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


DSS_VARIANTS = {"dss_global", "dss_event_timeline", "dss_layer_avoid"}
BASELINE_VARIANTS = {"naive", "naive_rich"}

VARIANT_PRIOR = {
    "naive": 0.0,
    "naive_rich": 1.0,
    "dss_global": 0.5,
    "dss_event_timeline": 2.0,
    "dss_layer_avoid": 2.5,
}


@dataclass
class CandidateScore:
    case_id: str
    variant: str
    audio_path: str
    acoustic_score: float
    selector_score: float
    decision: str
    reasons: List[str]


def _fnum(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def acoustic_score_from_metrics(row: Dict[str, Any], expected_event_count: int = 0) -> tuple[float, List[str]]:
    """
    Return a conservative 0-100 acoustic sanity score from raw metrics.
    This is not semantic quality or human preference.
    """
    rms = _fnum(row.get("rms_dbfs"))
    peak = _fnum(row.get("peak_dbfs"))
    clip = _fnum(row.get("clipped_ratio"))
    silence = _fnum(row.get("silence_ratio"))
    active = _fnum(row.get("active_ratio"))
    onsets = _fnum(row.get("onset_count_proxy"))

    score = 100.0
    reasons: List[str] = []

    if clip >= 0.001:
        score -= 30.0
        reasons.append(f"clipping_ratio_high:{clip}")
    elif clip > 0:
        penalty = min(10.0, clip * 10000.0)
        score -= penalty
        reasons.append(f"minor_clip:-{round(penalty, 4)}")

    if peak >= -0.1:
        score -= 8.0
        reasons.append(f"near_full_scale_peak:{peak}")
    elif peak > -1.0:
        score -= 3.0
        reasons.append(f"hot_peak:{peak}")

    if rms > -12.0:
        score -= 14.0
        reasons.append(f"very_loud_rms:{rms}")
    elif rms > -16.0:
        score -= 6.0
        reasons.append(f"loud_rms:{rms}")
    elif rms < -42.0:
        score -= 12.0
        reasons.append(f"very_quiet_rms:{rms}")
    elif rms < -36.0:
        score -= 5.0
        reasons.append(f"quiet_rms:{rms}")

    if silence > 0.12:
        score -= 8.0
        reasons.append(f"very_high_silence_ratio:{silence}")
    elif silence > 0.08:
        score -= 5.0
        reasons.append(f"high_silence_ratio:{silence}")
    elif silence > 0.03:
        score -= 2.0
        reasons.append(f"moderate_silence_ratio:{silence}")

    if active < 0.70:
        score -= 8.0
        reasons.append(f"low_active_ratio:{active}")

    if expected_event_count > 0:
        if onsets == 0:
            score -= 10.0
            reasons.append("no_onset_proxy")
        elif onsets > expected_event_count * 8:
            score -= 5.0
            reasons.append(f"over_dense_onset_proxy:{onsets}_vs_expected:{expected_event_count}")

    status = row.get("status", "ok")
    if status != "ok":
        score -= 18.0
        reasons.append(f"status:{status}")

    return round(max(0.0, min(100.0, score)), 4), reasons


def score_candidate(row: Dict[str, Any], expected_event_count: int = 0) -> CandidateScore:
    variant = row["variant"]
    acoustic_score, reasons = acoustic_score_from_metrics(row, expected_event_count)

    score = acoustic_score

    prior = VARIANT_PRIOR.get(variant, 0.0)
    score += prior
    if prior:
        reasons.append(f"variant_prior:+{prior}")

    is_repaired = str(row.get("is_repaired_selected", "False")).lower() == "true"
    if is_repaired:
        score -= 3.0
        reasons.append("repaired_candidate_penalty:-3.0")

    clip = _fnum(row.get("clipped_ratio"))
    rms = _fnum(row.get("rms_dbfs"))
    peak = _fnum(row.get("peak_dbfs"))

    if clip >= 0.001 or rms > -10.0:
        decision = "repair_required"
    elif score >= 92.0:
        decision = "selector_candidate"
    elif score >= 84.0:
        decision = "listen_review"
    else:
        decision = "rejected"

    # Near-full-scale peak should not force repair by itself, but it should never be silently hidden.
    if peak >= -0.1 and decision == "selector_candidate":
        decision = "listen_review"
        reasons.append("downgrade_selector_candidate_due_to_near_full_scale_peak")

    return CandidateScore(
        case_id=row["case_id"],
        variant=variant,
        audio_path=row.get("selected_wav_path", ""),
        acoustic_score=round(acoustic_score, 4),
        selector_score=round(score, 4),
        decision=decision,
        reasons=reasons,
    )


def select_case(case_id: str, rows: List[Dict[str, Any]], expected_event_count: int = 0) -> Dict[str, Any]:
    scored = [score_candidate(r, expected_event_count) for r in rows]

    scored_sorted = sorted(
        scored,
        key=lambda x: (
            -x.selector_score,
            0 if x.variant in DSS_VARIANTS else 1,
            x.variant,
        ),
    )

    winner = scored_sorted[0]
    runner_up = scored_sorted[1] if len(scored_sorted) > 1 else None

    repair_queue = [x for x in scored_sorted if x.decision == "repair_required"]
    listen_review = [x for x in scored_sorted if x.decision == "listen_review"]
    rejected = [x for x in scored_sorted if x.decision == "rejected"]

    best_dss = next((x for x in scored_sorted if x.variant in DSS_VARIANTS), None)
    best_baseline = next((x for x in scored_sorted if x.variant in BASELINE_VARIANTS), None)

    dss_delta = None
    if best_dss and best_baseline:
        dss_delta = round(best_dss.selector_score - best_baseline.selector_score, 4)

    if dss_delta is None:
        case_classification = "blocked"
    elif dss_delta >= 5.0:
        case_classification = "dss_improves"
    elif dss_delta <= -5.0:
        case_classification = "dss_worse"
    else:
        case_classification = "dss_neutral"

    return {
        "case_id": case_id,
        "winner": winner.__dict__,
        "runner_up": runner_up.__dict__ if runner_up else None,
        "best_dss_variant": best_dss.variant if best_dss else None,
        "best_baseline_variant": best_baseline.variant if best_baseline else None,
        "dss_delta_vs_best_baseline": dss_delta,
        "case_classification": case_classification,
        "repair_queue": [x.__dict__ for x in repair_queue],
        "listen_review": [x.__dict__ for x in listen_review],
        "rejected": [x.__dict__ for x in rejected],
        "all_candidates": [x.__dict__ for x in scored_sorted],
    }
