#!/usr/bin/env python3
import csv
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf

REVIEW = Path("reports/w18_candidate_review_queue_20260706.csv")
OUT_JSON = Path("reports/w18_clipping_repair_report_20260706.json")
OUT_CSV = Path("reports/w18_clipping_repair_report_20260706.csv")

rows = list(csv.DictReader(REVIEW.open(encoding="utf-8")))
targets = [
    r for r in rows
    if r["priority"] == "repair_or_regenerate"
    and r["wav_path"]
    and Path(r["wav_path"]).exists()
]

repair_rows = []

def dbfs(x):
    return 20.0 * math.log10(max(float(x), 1e-12))

for r in targets:
    src = Path(r["wav_path"])
    dst = src.with_name(src.stem + "_repaired_peak_m3db.wav")

    audio, sr = sf.read(src, always_2d=True)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0

    target_peak = 10 ** (-3.0 / 20.0)
    gain = target_peak / max(peak, 1e-12)
    repaired = np.clip(audio * gain, -1.0, 1.0)

    sf.write(dst, repaired, sr)

    abs_rep = np.abs(repaired)
    new_peak = float(abs_rep.max()) if repaired.size else 0.0
    new_rms = float(np.sqrt(np.mean(repaired ** 2))) if repaired.size else 0.0
    new_clip = float(np.mean(abs_rep >= 0.999)) if repaired.size else 0.0

    repair_rows.append({
        "case_id": r["case_id"],
        "variant": r["variant"],
        "job_id": r["job_id"],
        "source_wav": str(src),
        "repaired_wav": str(dst),
        "source_peak_dbfs": r["peak_dbfs"],
        "source_clipped_ratio": r["clipped_ratio"],
        "gain": round(gain, 8),
        "new_peak_dbfs": round(dbfs(new_peak), 4),
        "new_rms_dbfs": round(dbfs(new_rms), 4),
        "new_clipped_ratio": round(new_clip, 8),
        "status": "success" if dst.exists() and new_clip < 0.001 else "review_required",
    })

summary = {
    "date": "2026-07-06",
    "scope": "w18_clipping_review_peak_repair",
    "target_count": len(targets),
    "repaired_count": len(repair_rows),
    "status": "success" if repair_rows and all(r["status"] == "success" for r in repair_rows) else "review_required",
    "policy": [
        "Repair is non-destructive: original generated audio is preserved.",
        "Peak-normalized repair targets -3 dBFS peak.",
        "This is an engineering repair candidate, not semantic quality improvement.",
    ],
    "rows": repair_rows,
}

OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

if repair_rows:
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(repair_rows[0].keys()))
        writer.writeheader()
        writer.writerows(repair_rows)

print(json.dumps(summary, ensure_ascii=False, indent=2))
