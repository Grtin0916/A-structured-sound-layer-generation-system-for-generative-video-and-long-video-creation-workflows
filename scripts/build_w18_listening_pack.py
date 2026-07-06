#!/usr/bin/env python3
import csv
import html
import json
from pathlib import Path

ANALYSIS = Path("reports/w18_full_30job_ablation_analysis_20260706.json")
REPAIR = Path("reports/w18_clipping_repair_report_20260706.json")
MATRIX = Path("reports/w18_full_30job_generation_matrix_20260706.csv")

OUT_JSON = Path("reports/w18_repair_aware_listening_pack_20260706.json")
OUT_CSV = Path("reports/w18_repair_aware_listening_pack_20260706.csv")
OUT_M3U = Path("artifacts/model_runs/w18_dss_ablation/w18_repair_aware_listening_pack_20260706.m3u")
OUT_HTML = Path("artifacts/model_runs/w18_dss_ablation/w18_repair_aware_listening_pack_20260706.html")

VARIANT_ORDER = {
    "naive": 0,
    "naive_rich": 1,
    "dss_global": 2,
    "dss_event_timeline": 3,
    "dss_layer_avoid": 4,
}

analysis = json.loads(ANALYSIS.read_text(encoding="utf-8"))
repair = json.loads(REPAIR.read_text(encoding="utf-8"))

rows = list(csv.DictReader(MATRIX.open(encoding="utf-8")))

repair_by_job = {
    r["job_id"]: r
    for r in repair.get("rows", [])
    if r.get("status") == "success"
}

rows_by_case = {}
for r in rows:
    rows_by_case.setdefault(r["case_id"], []).append(r)

case_candidates = []
playlist_items = []

for case in analysis.get("case_summary", []):
    case_id = case["case_id"]
    candidate_path = case.get("candidate_audio_path")
    candidate_variant = case.get("candidate_variant_for_listening")

    case_rows = sorted(
        rows_by_case.get(case_id, []),
        key=lambda x: VARIANT_ORDER.get(x["variant"], 99),
    )

    # Add all generated variants for A/B listening.
    variants = []
    for r in case_rows:
        job_id = r["job_id"]
        wav_path = r["wav_path"]
        selected_path = wav_path
        repaired = repair_by_job.get(job_id)

        if repaired:
            selected_path = repaired["repaired_wav"]

        reason = []
        if r.get("status") != "ok":
            reason.append(r.get("status", "review"))
        if repaired:
            reason.append("uses_peak_repaired_version")

        variants.append({
            "case_id": case_id,
            "variant": r["variant"],
            "job_id": job_id,
            "original_wav_path": wav_path,
            "selected_wav_path": selected_path,
            "is_default_candidate": wav_path == candidate_path,
            "is_repaired": bool(repaired),
            "rms_dbfs": r.get("rms_dbfs"),
            "peak_dbfs": r.get("peak_dbfs"),
            "clipped_ratio": r.get("clipped_ratio"),
            "status": r.get("status"),
            "review_reason": "|".join(reason),
        })

        playlist_items.append({
            "title": f"{case_id}__{r['variant']}" + ("__REPAIRED" if repaired else ""),
            "path": selected_path,
        })

    case_candidates.append({
        "case_id": case_id,
        "default_candidate_variant": candidate_variant,
        "default_candidate_audio_path": candidate_path,
        "review_variants": case.get("review_variants", []),
        "variants": variants,
    })

summary = {
    "date": "2026-07-06",
    "scope": "w18_repair_aware_listening_pack",
    "status": "success" if len(case_candidates) == 6 and len(playlist_items) == 30 else "review_required",
    "case_count": len(case_candidates),
    "playlist_item_count": len(playlist_items),
    "repair_applied_count": sum(1 for c in case_candidates for v in c["variants"] if v["is_repaired"]),
    "source_analysis": str(ANALYSIS),
    "source_repair": str(REPAIR),
    "outputs": {
        "json": str(OUT_JSON),
        "csv": str(OUT_CSV),
        "m3u": str(OUT_M3U),
        "html": str(OUT_HTML),
    },
    "claim_boundary": [
        "This is a repair-aware listening/review pack.",
        "It does not claim semantic superiority of DSS variants.",
        "It selects paths for listening and review, not final production mix.",
        "Repaired files are local artifacts and remain ignored by Git.",
    ],
    "cases": case_candidates,
}

OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

flat_rows = []
for c in case_candidates:
    for v in c["variants"]:
        flat_rows.append({
            "case_id": c["case_id"],
            "default_candidate_variant": c["default_candidate_variant"],
            **v,
        })

with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()))
    writer.writeheader()
    writer.writerows(flat_rows)

OUT_M3U.parent.mkdir(parents=True, exist_ok=True)
with OUT_M3U.open("w", encoding="utf-8") as f:
    f.write("#EXTM3U\n")
    for item in playlist_items:
        f.write(f"#EXTINF:-1,{item['title']}\n")
        f.write(f"{item['path']}\n")

html_rows = []
for item in playlist_items:
    title = html.escape(item["title"])
    path = html.escape(item["path"])
    html_rows.append(
        f"<tr><td>{title}</td><td><audio controls src='{path}'></audio></td><td><code>{path}</code></td></tr>"
    )

OUT_HTML.write_text(
    "<!doctype html><html><head><meta charset='utf-8'>"
    "<title>W18 Repair-aware Listening Pack</title>"
    "<style>body{font-family:Arial,sans-serif;margin:24px}table{border-collapse:collapse;width:100%}"
    "td,th{border:1px solid #ccc;padding:8px;vertical-align:top}code{font-size:12px}</style>"
    "</head><body>"
    "<h1>W18 Repair-aware Listening Pack</h1>"
    "<p>Local review index for 6 cases × 5 variants. Audio files are local artifacts.</p>"
    "<table><thead><tr><th>Item</th><th>Player</th><th>Path</th></tr></thead><tbody>"
    + "\n".join(html_rows)
    + "</tbody></table></body></html>",
    encoding="utf-8",
)

print(json.dumps(summary, ensure_ascii=False, indent=2))
raise SystemExit(0 if summary["status"] == "success" else 2)
