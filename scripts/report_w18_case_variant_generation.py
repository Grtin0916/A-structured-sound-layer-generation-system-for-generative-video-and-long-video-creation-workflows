#!/usr/bin/env python3
import argparse
import csv
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

ORDER = {
    "naive": 0,
    "naive_rich": 1,
    "dss_global": 2,
    "dss_event_timeline": 3,
    "dss_layer_avoid": 4,
}

def audio_stats(path: Path):
    if not path.exists():
        return {
            "exists": False,
            "sample_rate": None,
            "duration_sec": None,
            "rms_dbfs": None,
            "peak_dbfs": None,
            "clipped_ratio": None,
            "active_ratio": None,
            "status": "missing",
        }

    audio, sr = sf.read(path, always_2d=True)
    abs_audio = np.abs(audio)
    peak = float(abs_audio.max()) if audio.size else 0.0
    rms = float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0
    peak_dbfs = 20.0 * math.log10(max(peak, 1e-12))
    rms_dbfs = 20.0 * math.log10(max(rms, 1e-12))
    clipped_ratio = float(np.mean(abs_audio >= 0.999)) if audio.size else 0.0
    active_ratio = float(np.mean(abs_audio >= 1e-4)) if audio.size else 0.0
    duration = audio.shape[0] / float(sr)

    status = "ok"
    if rms <= 1e-6:
        status = "near_silent"
    elif clipped_ratio >= 0.001:
        status = "clipping_review"

    return {
        "exists": True,
        "sample_rate": sr,
        "duration_sec": round(duration, 4),
        "rms_dbfs": round(rms_dbfs, 4),
        "peak_dbfs": round(peak_dbfs, 4),
        "clipped_ratio": round(clipped_ratio, 8),
        "active_ratio": round(active_ratio, 8),
        "status": status,
    }

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-id", required=True)
    ap.add_argument("--queue-jsonl", default="artifacts/model_runs/w18_dss_ablation/generation_queue_duration_prompt_aligned_20260706.jsonl")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    jobs = [json.loads(x) for x in Path(args.queue_jsonl).read_text(encoding="utf-8").splitlines() if x.strip()]
    jobs = [j for j in jobs if j["case_id"] == args.case_id]
    jobs = sorted(jobs, key=lambda x: ORDER.get(x["variant"], 99))

    rows = []
    for j in jobs:
        flac = Path(j["expected_output_flac"])
        wav = Path(j["expected_output_wav"])
        chosen = wav if wav.exists() else flac
        stats = audio_stats(chosen)

        rows.append({
            "job_id": j["job_id"],
            "case_id": j["case_id"],
            "variant": j["variant"],
            "generation_duration_sec": j["duration_sec"],
            "prompt_chars": len(j["prompt"]),
            "flac_path": str(flac),
            "flac_exists": flac.exists(),
            "flac_size_bytes": flac.stat().st_size if flac.exists() else None,
            "wav_path": str(wav),
            "wav_exists": wav.exists(),
            "wav_size_bytes": wav.stat().st_size if wav.exists() else None,
            **stats,
        })

    summary = {
        "date": "2026-07-06",
        "scope": f"w18_{args.case_id}_5_variant_generation",
        "case_id": args.case_id,
        "status": "success" if len(rows) == 5 and all(r["exists"] and r["status"] in {"ok", "clipping_review"} for r in rows) else "review_required",
        "variant_count": len(rows),
        "generated_count": sum(1 for r in rows if r["flac_exists"] or r["wav_exists"]),
        "model_variant": "small_44k",
        "duration_values": sorted(set(r["duration_sec"] for r in rows if r["duration_sec"] is not None)),
        "sample_rates": sorted(set(r["sample_rate"] for r in rows if r["sample_rate"])),
        "review_flags": [
            {
                "variant": r["variant"],
                "status": r["status"],
                "rms_dbfs": r["rms_dbfs"],
                "peak_dbfs": r["peak_dbfs"],
                "clipped_ratio": r["clipped_ratio"],
            }
            for r in rows if r["status"] != "ok"
        ],
        "rows": rows,
        "claim_boundary": [
            "This is one case-level five-variant generation report.",
            "It validates generation coverage and audio sanity, not semantic superiority.",
            "Audio artifacts remain local and are intentionally ignored by Git.",
        ],
    }

    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["generated_count"] == 5 else 2

if __name__ == "__main__":
    raise SystemExit(main())
