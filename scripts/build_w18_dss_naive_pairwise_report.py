#!/usr/bin/env python3
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

BASELINE_VARIANTS = {"naive", "naive_rich"}
DSS_VARIANTS = {"dss_global", "dss_event_timeline", "dss_layer_avoid"}

VARIANT_ORDER = {
    "naive": 0,
    "naive_rich": 1,
    "dss_global": 2,
    "dss_event_timeline": 3,
    "dss_layer_avoid": 4,
}

def fnum(x, default=0.0):
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default

def load_expected_event_count(case_id: str) -> int:
    p = Path("cases") / case_id / "expected_events.csv"
    if not p.exists():
        return 0
    try:
        rows = list(csv.DictReader(p.open(encoding="utf-8")))
        return len(rows)
    except Exception:
        lines = [x for x in p.read_text(encoding="utf-8", errors="replace").splitlines() if x.strip()]
        return max(0, len(lines) - 1)

def metric_score(row, expected_events: int):
    """
    Conservative acoustic sanity score.
    This is not semantic preference and not final synchrony.
    Higher is better.
    """
    rms = fnum(row.get("rms_dbfs"))
    peak = fnum(row.get("peak_dbfs"))
    clip = fnum(row.get("clipped_ratio"))
    silence = fnum(row.get("silence_ratio"))
    active = fnum(row.get("active_ratio"))
    onsets = fnum(row.get("onset_count_proxy"))

    score = 100.0
    reasons = []

    if clip >= 0.001:
        score -= 30.0
        reasons.append(f"clipping_ratio_high:{clip}")
    elif clip > 0:
        score -= min(10.0, clip * 10000.0)
        reasons.append(f"minor_clip:{clip}")

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

    if silence > 0.08:
        score -= 10.0
        reasons.append(f"high_silence_ratio:{silence}")
    elif silence > 0.03:
        score -= 4.0
        reasons.append(f"moderate_silence_ratio:{silence}")

    if active < 0.70:
        score -= 8.0
        reasons.append(f"low_active_ratio:{active}")

    # Onset is a weak proxy. Use only as a soft penalty for extreme mismatch.
    if expected_events > 0:
        if onsets == 0:
            score -= 12.0
            reasons.append("no_onset_proxy")
        elif onsets > expected_events * 8:
            score -= 5.0
            reasons.append(f"over_dense_onset_proxy:{onsets}_vs_expected:{expected_events}")

    status = row.get("status", "ok")
    if status != "ok":
        score -= 20.0
        reasons.append(f"status:{status}")

    return round(score, 4), reasons

def classify_delta(delta):
    if delta >= 5.0:
        return "improves"
    if delta <= -5.0:
        return "worse"
    return "neutral"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-csv", default="reports/w18_audio_metrics_eval_20260707.csv")
    ap.add_argument("--out-json", default="reports/w18_dss_vs_naive_pairwise_report_20260707.json")
    ap.add_argument("--out-csv", default="reports/w18_dss_vs_naive_pairwise_report_20260707.csv")
    ap.add_argument("--selector-seed-json", default="reports/w18_repair_aware_selector_seed_20260707.json")
    args = ap.parse_args()

    rows = list(csv.DictReader(Path(args.metrics_csv).open(encoding="utf-8")))
    by_case = defaultdict(list)
    for r in rows:
        by_case[r["case_id"]].append(r)

    pairwise_rows = []
    case_summaries = []
    repair_targets = []
    selector_seed = []

    for case_id, case_rows in sorted(by_case.items()):
        expected_count = load_expected_event_count(case_id)

        scored = []
        for r in case_rows:
            score, reasons = metric_score(r, expected_count)
            rr = dict(r)
            rr["metric_score"] = score
            rr["score_reasons"] = reasons
            rr["expected_event_count"] = expected_count
            scored.append(rr)

        baselines = [r for r in scored if r["variant"] in BASELINE_VARIANTS]
        dss_rows = [r for r in scored if r["variant"] in DSS_VARIANTS]

        if not baselines or not dss_rows:
            case_summaries.append({
                "case_id": case_id,
                "status": "blocked",
                "reason": "missing_baseline_or_dss",
            })
            continue

        baseline = sorted(
            baselines,
            key=lambda r: (-float(r["metric_score"]), VARIANT_ORDER.get(r["variant"], 99))
        )[0]

        best_dss = sorted(
            dss_rows,
            key=lambda r: (-float(r["metric_score"]), VARIANT_ORDER.get(r["variant"], 99))
        )[0]

        for dss in sorted(dss_rows, key=lambda r: VARIANT_ORDER.get(r["variant"], 99)):
            delta = round(float(dss["metric_score"]) - float(baseline["metric_score"]), 4)
            label = classify_delta(delta)

            pairwise_rows.append({
                "case_id": case_id,
                "expected_event_count": expected_count,
                "baseline_variant": baseline["variant"],
                "baseline_score": baseline["metric_score"],
                "baseline_rms_dbfs": baseline["rms_dbfs"],
                "baseline_peak_dbfs": baseline["peak_dbfs"],
                "baseline_clipped_ratio": baseline["clipped_ratio"],
                "baseline_onset_count_proxy": baseline["onset_count_proxy"],
                "dss_variant": dss["variant"],
                "dss_score": dss["metric_score"],
                "dss_rms_dbfs": dss["rms_dbfs"],
                "dss_peak_dbfs": dss["peak_dbfs"],
                "dss_clipped_ratio": dss["clipped_ratio"],
                "dss_onset_count_proxy": dss["onset_count_proxy"],
                "score_delta_vs_baseline": delta,
                "classification": label,
                "baseline_reasons": "|".join(baseline["score_reasons"]),
                "dss_reasons": "|".join(dss["score_reasons"]),
                "dss_audio_path": dss["selected_wav_path"],
            })

        case_delta = round(float(best_dss["metric_score"]) - float(baseline["metric_score"]), 4)
        case_label = classify_delta(case_delta)

        ordered = sorted(scored, key=lambda r: (-float(r["metric_score"]), VARIANT_ORDER.get(r["variant"], 99)))
        winner = ordered[0]
        runner_up = ordered[1] if len(ordered) > 1 else None

        rejected = []
        for r in ordered[2:]:
            reason = r["score_reasons"] or ["lower_metric_score"]
            rejected.append({
                "variant": r["variant"],
                "score": r["metric_score"],
                "reasons": reason,
                "audio_path": r["selected_wav_path"],
            })

        for r in scored:
            reasons = list(r["score_reasons"])
            priority = "normal"
            if any("clipping" in x or "near_full_scale" in x or "very_loud" in x for x in reasons):
                priority = "repair_target"
            elif any("very_quiet" in x or "over_dense_onset" in x or "no_onset" in x for x in reasons):
                priority = "listen_review"

            if priority != "normal":
                repair_targets.append({
                    "case_id": case_id,
                    "variant": r["variant"],
                    "priority": priority,
                    "score": r["metric_score"],
                    "reasons": reasons,
                    "audio_path": r["selected_wav_path"],
                })

        case_summary = {
            "case_id": case_id,
            "expected_event_count": expected_count,
            "baseline_variant": baseline["variant"],
            "baseline_score": baseline["metric_score"],
            "best_dss_variant": best_dss["variant"],
            "best_dss_score": best_dss["metric_score"],
            "best_dss_delta": case_delta,
            "case_classification": case_label,
            "winner_variant": winner["variant"],
            "winner_score": winner["metric_score"],
            "winner_audio_path": winner["selected_wav_path"],
            "runner_up_variant": runner_up["variant"] if runner_up else None,
            "runner_up_score": runner_up["metric_score"] if runner_up else None,
            "runner_up_audio_path": runner_up["selected_wav_path"] if runner_up else None,
            "rejected_count": len(rejected),
            "rejected": rejected,
        }
        case_summaries.append(case_summary)

        selector_seed.append({
            "case_id": case_id,
            "winner": {
                "variant": winner["variant"],
                "score": winner["metric_score"],
                "audio_path": winner["selected_wav_path"],
            },
            "runner_up": {
                "variant": runner_up["variant"] if runner_up else None,
                "score": runner_up["metric_score"] if runner_up else None,
                "audio_path": runner_up["selected_wav_path"] if runner_up else None,
            },
            "classification": case_label,
            "baseline_variant": baseline["variant"],
            "best_dss_variant": best_dss["variant"],
            "best_dss_delta": case_delta,
        })

    label_counts = defaultdict(int)
    for c in case_summaries:
        label_counts[c.get("case_classification", "blocked")] += 1

    summary = {
        "date": "2026-07-07",
        "scope": "w18_dss_vs_naive_pairwise_report",
        "status": "success" if len(case_summaries) == 6 and len(pairwise_rows) == 18 else "review_required",
        "candidate_count": len(rows),
        "case_count": len(case_summaries),
        "pairwise_count": len(pairwise_rows),
        "classification_counts": dict(label_counts),
        "repair_target_count": len(repair_targets),
        "case_summaries": case_summaries,
        "repair_targets": repair_targets,
        "metric_boundary": [
            "Classification is based on acoustic sanity and onset proxy only.",
            "This does not prove DSS semantic superiority.",
            "Cases labeled worse or neutral are still useful for repair bank and selector design.",
        ],
        "outputs": {
            "json": args.out_json,
            "csv": args.out_csv,
            "selector_seed_json": args.selector_seed_json,
        }
    }

    Path(args.out_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    with Path(args.out_csv).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(pairwise_rows[0].keys()))
        writer.writeheader()
        writer.writerows(pairwise_rows)

    Path(args.selector_seed_json).write_text(json.dumps({
        "date": "2026-07-07",
        "scope": "w18_repair_aware_selector_seed",
        "status": "success" if len(selector_seed) == 6 else "review_required",
        "case_count": len(selector_seed),
        "selector_seed": selector_seed,
        "repair_targets": repair_targets,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "status": summary["status"],
        "candidate_count": summary["candidate_count"],
        "case_count": summary["case_count"],
        "pairwise_count": summary["pairwise_count"],
        "classification_counts": summary["classification_counts"],
        "repair_target_count": summary["repair_target_count"],
        "outputs": summary["outputs"],
    }, ensure_ascii=False, indent=2))

    return 0 if summary["status"] == "success" else 2

if __name__ == "__main__":
    raise SystemExit(main())
