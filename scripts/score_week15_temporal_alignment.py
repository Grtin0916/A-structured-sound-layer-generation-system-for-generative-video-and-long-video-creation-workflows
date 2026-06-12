#!/usr/bin/env python3
"""
Score Week15 temporal alignment from candidate audio WAV files.

Metric idea:
- Read candidate WAV.
- Convert to mono float waveform.
- Compute short-window RMS envelope.
- Estimate first onset frame above adaptive threshold.
- Estimate peak-energy frame.
- Convert local onset to global onset using globalStartSec.
- Compare global onset against expectedStartSec.

Boundary:
- Does not judge semantic audio quality.
- Does not replace human audition.
- Does not claim final mix readiness.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_pcm_wav(path: Path) -> tuple[np.ndarray, int]:
    """
    Return mono float32 waveform in [-1, 1] approximately and sample rate.
    Supports common PCM sample widths: 8/16/24/32-bit integer PCM.
    """
    with wave.open(str(path), "rb") as wf:
        nch = wf.getnchannels()
        sr = wf.getframerate()
        sw = wf.getsampwidth()
        nframes = wf.getnframes()
        raw = wf.readframes(nframes)

    if nframes <= 0:
        raise ValueError("empty wav")

    if sw == 1:
        arr = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        arr = (arr - 128.0) / 128.0
    elif sw == 2:
        arr = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sw == 3:
        b = np.frombuffer(raw, dtype=np.uint8)
        if len(b) % 3 != 0:
            raise ValueError("invalid 24-bit PCM byte length")
        b = b.reshape(-1, 3)
        signed = (
            b[:, 0].astype(np.int32)
            | (b[:, 1].astype(np.int32) << 8)
            | (b[:, 2].astype(np.int32) << 16)
        )
        signed = np.where(signed & 0x800000, signed - 0x1000000, signed)
        arr = signed.astype(np.float32) / 8388608.0
    elif sw == 4:
        arr = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported sample width: {sw}")

    if nch <= 0:
        raise ValueError(f"invalid channel count: {nch}")
    if arr.size % nch != 0:
        raise ValueError(f"sample count {arr.size} not divisible by channels {nch}")

    arr = arr.reshape(-1, nch).mean(axis=1)
    return arr.astype(np.float32), int(sr)


def rms_envelope(x: np.ndarray, sr: int, frame_ms: float, hop_ms: float) -> tuple[np.ndarray, np.ndarray]:
    frame = max(1, int(round(sr * frame_ms / 1000.0)))
    hop = max(1, int(round(sr * hop_ms / 1000.0)))

    if x.size < frame:
        pad = np.zeros(frame - x.size, dtype=np.float32)
        x = np.concatenate([x, pad])

    vals = []
    times = []
    for start in range(0, x.size - frame + 1, hop):
        chunk = x[start:start + frame]
        vals.append(float(np.sqrt(np.mean(chunk * chunk) + 1e-12)))
        times.append((start + frame / 2.0) / sr)

    return np.asarray(vals, dtype=np.float32), np.asarray(times, dtype=np.float32)


def estimate_onset_and_peak(
    x: np.ndarray,
    sr: int,
    frame_ms: float,
    hop_ms: float,
    threshold_ratio: float,
    min_abs_rms: float,
) -> dict[str, Any]:
    rms, times = rms_envelope(x, sr, frame_ms, hop_ms)
    if rms.size == 0:
        raise ValueError("empty RMS envelope")

    peak_idx = int(np.argmax(rms))
    peak_rms = float(rms[peak_idx])
    median = float(np.median(rms))
    p90 = float(np.percentile(rms, 90))
    threshold = max(min_abs_rms, median + threshold_ratio * max(0.0, peak_rms - median))

    onset_idx = None
    for i, val in enumerate(rms):
        if float(val) >= threshold:
            onset_idx = i
            break

    if onset_idx is None:
        onset_sec = None
        onset_rms = None
    else:
        onset_sec = float(times[onset_idx])
        onset_rms = float(rms[onset_idx])

    return {
        "durationSec": float(x.size / sr),
        "sampleRate": sr,
        "rmsMedian": median,
        "rmsP90": p90,
        "rmsPeak": peak_rms,
        "adaptiveThreshold": float(threshold),
        "localOnsetSec": onset_sec,
        "localOnsetRms": onset_rms,
        "localPeakSec": float(times[peak_idx]),
        "localPeakRms": peak_rms,
    }


def classify(delta: float | None, tolerance_sec: float) -> str:
    if delta is None or not math.isfinite(delta):
        return "FAIL_NO_ONSET"
    if abs(delta) <= tolerance_sec:
        return "PASS"
    if abs(delta) <= tolerance_sec * 2:
        return "WARN_NEAR_MISS"
    return "FAIL_DRIFT"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mainbase", default=".")
    ap.add_argument("--input", default="artifacts/evals/week15_temporal_alignment_input_index.json")
    ap.add_argument("--csv-out", default="artifacts/evals/week15_temporal_alignment.csv")
    ap.add_argument("--json-out", default="artifacts/evals/week15_temporal_alignment_summary.json")
    ap.add_argument("--frame-ms", type=float, default=50.0)
    ap.add_argument("--hop-ms", type=float, default=10.0)
    ap.add_argument("--threshold-ratio", type=float, default=0.20)
    ap.add_argument("--min-abs-rms", type=float, default=1e-4)
    ap.add_argument("--tolerance-sec", type=float, default=0.25)
    args = ap.parse_args()

    root = Path(args.mainbase).expanduser().resolve()
    input_path = root / args.input
    idx = read_json(input_path)

    blockers: list[str] = []
    rows: list[dict[str, Any]] = []

    if idx.get("status") != "PASS":
        blockers.append(f"input index status is not PASS: {idx.get('status')}")

    eval_inputs = idx.get("evalInputs", [])
    for item in eval_inputs:
        cid = item.get("candidateId")
        audio_uri = item.get("audioUri")
        expected_start = item.get("expectedStartSec")
        global_start = item.get("globalStartSec")
        asset_time_mode = item.get("assetTimeMode")

        row: dict[str, Any] = {
            "candidateId": cid,
            "assetTimeMode": asset_time_mode,
            "audioUri": audio_uri,
            "expectedStartSec": expected_start,
            "globalStartSec": global_start,
            "alignmentStatus": "UNSCORED",
            "error": "",
        }

        try:
            if not audio_uri:
                raise ValueError("missing audioUri")
            audio_path = root / str(audio_uri)
            if not audio_path.exists():
                raise FileNotFoundError(str(audio_path))

            x, sr = read_pcm_wav(audio_path)
            score = estimate_onset_and_peak(
                x=x,
                sr=sr,
                frame_ms=args.frame_ms,
                hop_ms=args.hop_ms,
                threshold_ratio=args.threshold_ratio,
                min_abs_rms=args.min_abs_rms,
            )

            local_onset = score["localOnsetSec"]
            local_peak = score["localPeakSec"]

            if global_start is None or expected_start is None or local_onset is None:
                global_onset = None
                onset_delta = None
            else:
                global_onset = float(global_start) + float(local_onset)
                onset_delta = global_onset - float(expected_start)

            if global_start is None or expected_start is None or local_peak is None:
                global_peak = None
                peak_delta = None
            else:
                global_peak = float(global_start) + float(local_peak)
                peak_delta = global_peak - float(expected_start)

            status = classify(onset_delta, args.tolerance_sec)

            row.update(score)
            row.update({
                "globalOnsetSec": global_onset,
                "onsetDeltaSec": onset_delta,
                "globalPeakSec": global_peak,
                "peakDeltaSec": peak_delta,
                "toleranceSec": args.tolerance_sec,
                "alignmentStatus": status,
            })
        except Exception as exc:
            row["alignmentStatus"] = "FAIL_ERROR"
            row["error"] = str(exc)
            blockers.append(f"{cid}: {exc}")

        rows.append(row)

    total = len(rows)
    pass_count = sum(1 for r in rows if r.get("alignmentStatus") == "PASS")
    warn_count = sum(1 for r in rows if r.get("alignmentStatus") == "WARN_NEAR_MISS")
    fail_count = sum(1 for r in rows if str(r.get("alignmentStatus", "")).startswith("FAIL"))
    event_local_rows = [r for r in rows if r.get("assetTimeMode") == "event_local"]
    event_local_pass = sum(1 for r in event_local_rows if r.get("alignmentStatus") == "PASS")

    if total != 10:
        blockers.append(f"expected 10 rows, got {total}")
    if fail_count > 0:
        blockers.append(f"alignment scoring has {fail_count} failed rows")

    status = "PASS" if not blockers else "FAIL"

    csv_out = root / args.csv_out
    json_out = root / args.json_out
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "candidateId",
        "assetTimeMode",
        "audioUri",
        "sampleRate",
        "durationSec",
        "expectedStartSec",
        "globalStartSec",
        "localOnsetSec",
        "globalOnsetSec",
        "onsetDeltaSec",
        "localPeakSec",
        "globalPeakSec",
        "peakDeltaSec",
        "rmsMedian",
        "rmsP90",
        "rmsPeak",
        "adaptiveThreshold",
        "toleranceSec",
        "alignmentStatus",
        "error",
    ]

    with csv_out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    summary = {
        "schemaVersion": "week15.temporal_alignment_score.v1",
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "inputIndex": str(input_path),
        "csv": str(csv_out),
        "summary": {
            "candidateCount": total,
            "passCount": pass_count,
            "warnNearMissCount": warn_count,
            "failCount": fail_count,
            "eventLocalCount": len(event_local_rows),
            "eventLocalPassCount": event_local_pass,
            "toleranceSec": args.tolerance_sec,
            "frameMs": args.frame_ms,
            "hopMs": args.hop_ms,
            "thresholdRatio": args.threshold_ratio,
            "minAbsRms": args.min_abs_rms,
        },
        "blockers": blockers,
        "boundary": [
            "energy_onset_proxy_only",
            "does_not_score_semantic_audio_quality",
            "does_not_claim_human_audition_passed",
            "does_not_claim_final_mix_readiness",
        ],
        "rows": rows,
    }

    json_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": status,
        "csv": str(csv_out),
        "json": str(json_out),
        "summary": summary["summary"],
        "blockers": blockers,
    }, indent=2, ensure_ascii=False))

    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())