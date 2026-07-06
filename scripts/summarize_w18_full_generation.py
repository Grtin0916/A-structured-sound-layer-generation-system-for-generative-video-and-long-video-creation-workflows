#!/usr/bin/env python3
import csv
import json
from pathlib import Path

CASE_REPORTS = [
    ("forest_bird_branch_001", "reports/w18_forest_5_variant_generation_20260706.json"),
    ("glass_drop_room_001", "reports/w18_glass_5_variant_generation_20260706.json"),
    ("kitchen_chop_sizzle_001", "reports/w18_kitchen_chop_sizzle_5_variant_generation_20260706.json"),
    ("robot_warehouse_pick_001", "reports/w18_robot_warehouse_pick_5_variant_generation_20260706.json"),
    ("street_rain_crosswalk_001", "reports/w18_street_rain_crosswalk_5_variant_generation_20260706.json"),
    ("subway_arrival_door_001", "reports/w18_subway_arrival_door_5_variant_generation_20260706.json"),
]

OUT_JSON = Path("reports/w18_full_30job_generation_summary_20260706.json")
OUT_CSV = Path("reports/w18_full_30job_generation_matrix_20260706.csv")

all_rows = []
case_summaries = []
missing_reports = []

for case_id, report_path in CASE_REPORTS:
    p = Path(report_path)
    if not p.exists():
        missing_reports.append(report_path)
        continue

    obj = json.loads(p.read_text(encoding="utf-8"))
    case_summaries.append({
        "case_id": case_id,
        "report": report_path,
        "status": obj.get("status"),
        "variant_count": obj.get("variant_count"),
        "generated_count": obj.get("generated_count"),
        "duration_values": obj.get("duration_values"),
        "sample_rates": obj.get("sample_rates"),
        "review_flags": obj.get("review_flags", []),
    })

    for row in obj.get("rows", []):
        all_rows.append({
            "case_id": row.get("case_id"),
            "variant": row.get("variant"),
            "job_id": row.get("job_id"),
            "generation_duration_sec": row.get("generation_duration_sec", row.get("duration_sec")),
            "prompt_chars": row.get("prompt_chars"),
            "flac_exists": row.get("flac_exists"),
            "flac_size_bytes": row.get("flac_size_bytes"),
            "wav_exists": row.get("wav_exists"),
            "wav_size_bytes": row.get("wav_size_bytes"),
            "sample_rate": row.get("sample_rate"),
            "duration_sec": row.get("duration_sec"),
            "rms_dbfs": row.get("rms_dbfs"),
            "peak_dbfs": row.get("peak_dbfs"),
            "clipped_ratio": row.get("clipped_ratio"),
            "active_ratio": row.get("active_ratio"),
            "status": row.get("status"),
            "flac_path": row.get("flac_path"),
            "wav_path": row.get("wav_path"),
        })

variant_counts = {}
for row in all_rows:
    variant_counts[row["variant"]] = variant_counts.get(row["variant"], 0) + 1

review_rows = [
    row for row in all_rows
    if row.get("status") not in {"ok", None}
]

generated_count = sum(
    1 for row in all_rows
    if row.get("flac_exists") or row.get("wav_exists")
)

summary = {
    "date": "2026-07-06",
    "scope": "w18_full_30job_generation",
    "status": "success" if len(all_rows) == 30 and generated_count == 30 and not missing_reports else "review_required",
    "case_count": len(case_summaries),
    "expected_case_count": 6,
    "job_count": len(all_rows),
    "expected_job_count": 30,
    "generated_count": generated_count,
    "variant_counts": variant_counts,
    "missing_reports": missing_reports,
    "case_summaries": case_summaries,
    "review_row_count": len(review_rows),
    "review_rows": review_rows,
    "claim_boundary": [
        "This records generation coverage and basic audio sanity for 6 cases x 5 variants.",
        "This does not claim DSS superiority over baselines.",
        "Audio files are local artifacts and are intentionally ignored by Git.",
        "Semantic quality, event timing, and preference ranking require later listening or metric evaluation.",
    ],
}

OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
if all_rows:
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

print(json.dumps(summary, ensure_ascii=False, indent=2))
raise SystemExit(0 if summary["status"] in {"success", "review_required"} and generated_count == 30 else 2)
