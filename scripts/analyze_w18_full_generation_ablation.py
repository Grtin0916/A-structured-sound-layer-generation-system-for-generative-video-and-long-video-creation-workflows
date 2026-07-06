#!/usr/bin/env python3
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

MATRIX = Path("reports/w18_full_30job_generation_matrix_20260706.csv")
OUT_JSON = Path("reports/w18_full_30job_ablation_analysis_20260706.json")
OUT_CSV = Path("reports/w18_full_30job_ablation_analysis_20260706.csv")
REVIEW_JSON = Path("reports/w18_candidate_review_queue_20260706.json")
REVIEW_CSV = Path("reports/w18_candidate_review_queue_20260706.csv")

VARIANT_ORDER = {
    "naive": 0,
    "naive_rich": 1,
    "dss_global": 2,
    "dss_event_timeline": 3,
    "dss_layer_avoid": 4,
}

def fnum(x):
    if x in {None, "", "None"}:
        return None
    try:
        return float(x)
    except Exception:
        return None

rows = list(csv.DictReader(MATRIX.open(encoding="utf-8")))

for r in rows:
    for k in [
        "generation_duration_sec",
        "prompt_chars",
        "flac_size_bytes",
        "wav_size_bytes",
        "sample_rate",
        "duration_sec",
        "rms_dbfs",
        "peak_dbfs",
        "clipped_ratio",
        "active_ratio",
    ]:
        r[k] = fnum(r.get(k))
    r["flac_exists"] = str(r.get("flac_exists")).lower() == "true"
    r["wav_exists"] = str(r.get("wav_exists")).lower() == "true"

variant_groups = defaultdict(list)
case_groups = defaultdict(list)

for r in rows:
    variant_groups[r["variant"]].append(r)
    case_groups[r["case_id"]].append(r)

variant_summary = []
for variant, vs in sorted(variant_groups.items(), key=lambda kv: VARIANT_ORDER.get(kv[0], 99)):
    rms_vals = [r["rms_dbfs"] for r in vs if r["rms_dbfs"] is not None]
    peak_vals = [r["peak_dbfs"] for r in vs if r["peak_dbfs"] is not None]
    clip_vals = [r["clipped_ratio"] for r in vs if r["clipped_ratio"] is not None]
    active_vals = [r["active_ratio"] for r in vs if r["active_ratio"] is not None]
    prompt_vals = [r["prompt_chars"] for r in vs if r["prompt_chars"] is not None]

    variant_summary.append({
        "variant": variant,
        "n": len(vs),
        "generated": sum(1 for r in vs if r["flac_exists"] or r["wav_exists"]),
        "ok_count": sum(1 for r in vs if r["status"] == "ok"),
        "review_count": sum(1 for r in vs if r["status"] != "ok"),
        "mean_prompt_chars": round(mean(prompt_vals), 2) if prompt_vals else None,
        "mean_rms_dbfs": round(mean(rms_vals), 4) if rms_vals else None,
        "max_peak_dbfs": round(max(peak_vals), 4) if peak_vals else None,
        "max_clipped_ratio": round(max(clip_vals), 8) if clip_vals else None,
        "mean_active_ratio": round(mean(active_vals), 8) if active_vals else None,
    })

case_summary = []
review_items = []

for case_id, cs in sorted(case_groups.items()):
    cs = sorted(cs, key=lambda r: VARIANT_ORDER.get(r["variant"], 99))
    generated = sum(1 for r in cs if r["flac_exists"] or r["wav_exists"])
    review = [r for r in cs if r["status"] != "ok"]

    # Conservative candidate: prefer ok, non-clipping, moderate peak, then richer prompt.
    ok_rows = [r for r in cs if r["status"] == "ok"]
    ranked = sorted(
        ok_rows,
        key=lambda r: (
            abs((r["peak_dbfs"] or -99.0) - (-6.0)),   # avoid hard 0 dBFS and too quiet
            -1 * (r["active_ratio"] or 0.0),
            -1 * (r["prompt_chars"] or 0.0),
        )
    )
    candidate = ranked[0] if ranked else None

    case_summary.append({
        "case_id": case_id,
        "generated": generated,
        "variant_count": len(cs),
        "review_count": len(review),
        "candidate_variant_for_listening": candidate["variant"] if candidate else None,
        "candidate_audio_path": candidate["wav_path"] if candidate else None,
        "review_variants": [r["variant"] for r in review],
    })

    for r in cs:
        reason = []
        if r["status"] != "ok":
            reason.append(r["status"])
        if r["peak_dbfs"] is not None and r["peak_dbfs"] >= -0.1:
            reason.append("near_full_scale_peak")
        if r["clipped_ratio"] is not None and r["clipped_ratio"] > 0.001:
            reason.append("clipping_ratio_gt_0.001")
        if r["rms_dbfs"] is not None and r["rms_dbfs"] > -12:
            reason.append("very_loud_rms")
        if r["rms_dbfs"] is not None and r["rms_dbfs"] < -45:
            reason.append("very_quiet_rms")

        priority = "normal"
        if "clipping_ratio_gt_0.001" in reason or "clipping_review" in reason:
            priority = "repair_or_regenerate"
        elif reason:
            priority = "listen_review"

        review_items.append({
            "case_id": case_id,
            "variant": r["variant"],
            "job_id": r["job_id"],
            "priority": priority,
            "reason": "|".join(reason) if reason else "",
            "status": r["status"],
            "rms_dbfs": r["rms_dbfs"],
            "peak_dbfs": r["peak_dbfs"],
            "clipped_ratio": r["clipped_ratio"],
            "active_ratio": r["active_ratio"],
            "wav_path": r["wav_path"],
            "flac_path": r["flac_path"],
        })

summary = {
    "date": "2026-07-06",
    "scope": "w18_full_30job_ablation_analysis",
    "source_matrix": str(MATRIX),
    "status": "success",
    "job_count": len(rows),
    "case_count": len(case_groups),
    "variant_count": len(variant_groups),
    "generated_count": sum(1 for r in rows if r["flac_exists"] or r["wav_exists"]),
    "review_item_count": sum(1 for r in review_items if r["priority"] != "normal"),
    "repair_or_regenerate_count": sum(1 for r in review_items if r["priority"] == "repair_or_regenerate"),
    "variant_summary": variant_summary,
    "case_summary": case_summary,
    "claim_boundary": [
        "This is acoustic sanity and candidate review analysis.",
        "Candidate selection is for listening/review only, not final semantic ranking.",
        "No DSS superiority claim is made without listening or stronger semantic/synchrony metrics.",
    ],
}

OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(variant_summary[0].keys()))
    writer.writeheader()
    writer.writerows(variant_summary)

REVIEW_JSON.write_text(json.dumps({
    "date": "2026-07-06",
    "scope": "w18_candidate_review_queue",
    "item_count": len(review_items),
    "non_normal_count": sum(1 for r in review_items if r["priority"] != "normal"),
    "repair_or_regenerate_count": sum(1 for r in review_items if r["priority"] == "repair_or_regenerate"),
    "items": review_items,
}, ensure_ascii=False, indent=2), encoding="utf-8")

with REVIEW_CSV.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(review_items[0].keys()))
    writer.writeheader()
    writer.writerows(review_items)

print(json.dumps(summary, ensure_ascii=False, indent=2))
