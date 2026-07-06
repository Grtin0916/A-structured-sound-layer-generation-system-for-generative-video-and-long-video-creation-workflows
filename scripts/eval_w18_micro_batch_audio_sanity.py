#!/usr/bin/env python3
import csv
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf

REPORT = Path("reports/w18_micro_batch_forest_naive_vs_rich_20260706.json")
OUT_JSON = Path("reports/w18_micro_batch_audio_sanity_20260706.json")
OUT_CSV = Path("reports/w18_micro_batch_audio_sanity_20260706.csv")

obj = json.loads(REPORT.read_text(encoding="utf-8"))
rows = []

for item in obj.get("rows", []):
    audio_path = Path(item["wav_path"]) if Path(item["wav_path"]).exists() else Path(item["flac_path"])
    exists = audio_path.exists()

    if not exists:
        rows.append({
            "job_id": item["job_id"],
            "variant": item["variant"],
            "audio_path": str(audio_path),
            "exists": False,
            "status": "missing_audio",
        })
        continue

    audio, sr = sf.read(audio_path, always_2d=True)
    samples = audio.shape[0]
    channels = audio.shape[1]
    duration = samples / float(sr)

    abs_audio = np.abs(audio)
    peak = float(abs_audio.max()) if samples else 0.0
    rms = float(np.sqrt(np.mean(audio ** 2))) if samples else 0.0
    peak_dbfs = 20.0 * math.log10(max(peak, 1e-12))
    rms_dbfs = 20.0 * math.log10(max(rms, 1e-12))
    clipped_ratio = float(np.mean(abs_audio >= 0.999)) if samples else 0.0
    active_ratio = float(np.mean(abs_audio >= 1e-4)) if samples else 0.0

    rows.append({
        "job_id": item["job_id"],
        "case_id": item["case_id"],
        "variant": item["variant"],
        "audio_path": str(audio_path),
        "exists": True,
        "sample_rate": sr,
        "channels": channels,
        "samples": samples,
        "duration_sec": round(duration, 4),
        "prompt_chars": item.get("prompt_chars"),
        "peak": round(peak, 8),
        "rms": round(rms, 8),
        "peak_dbfs": round(peak_dbfs, 4),
        "rms_dbfs": round(rms_dbfs, 4),
        "clipped_ratio": round(clipped_ratio, 8),
        "active_ratio": round(active_ratio, 8),
        "status": "ok" if rms > 1e-6 and clipped_ratio < 0.01 else "suspicious",
    })

durations = [r["duration_sec"] for r in rows if r.get("exists")]
sample_rates = [r["sample_rate"] for r in rows if r.get("exists")]
statuses = [r["status"] for r in rows]

summary = {
    "source_report": str(REPORT),
    "status": "success" if len(rows) == 2 and all(s == "ok" for s in statuses) else "review_required",
    "job_count": len(rows),
    "generated_count": sum(1 for r in rows if r.get("exists")),
    "duration_min_sec": min(durations) if durations else None,
    "duration_max_sec": max(durations) if durations else None,
    "duration_delta_sec": round(max(durations) - min(durations), 4) if len(durations) >= 2 else None,
    "sample_rates": sorted(set(sample_rates)),
    "rows": rows,
    "claim_boundary": [
        "This is audio sanity checking for a 2-job micro-batch only.",
        "It checks existence, duration, sample rate, RMS, peak, clipping, and active sample ratio.",
        "It does not claim semantic quality or DSS superiority."
    ],
}

OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print(json.dumps(summary, ensure_ascii=False, indent=2))
