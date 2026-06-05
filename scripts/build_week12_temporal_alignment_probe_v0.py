#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import statistics
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]

INPUT_QUEUE_CSV = ROOT / "artifacts/evals/week12_audio_candidate_timing_bound_queue_v2.csv"

OUT_CSV = ROOT / "artifacts/evals/week12_temporal_alignment_probe_v0.csv"
OUT_JSON = ROOT / "artifacts/evals/week12_temporal_alignment_probe_v0.json"
OUT_REPORT = ROOT / "artifacts/manifests/week12_temporal_alignment_probe_report_v0.json"


def as_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def norm_text(v: Any) -> str:
    return str(v or "").strip().lower()


def decode_pcm_sample(raw: bytes, sample_width: int) -> int:
    if sample_width == 1:
        return raw[0] - 128
    if sample_width == 2:
        return int.from_bytes(raw, byteorder="little", signed=True)
    if sample_width == 3:
        x = int.from_bytes(raw + (b"\xff" if raw[-1] & 0x80 else b"\x00"), byteorder="little", signed=True)
        return x
    if sample_width == 4:
        return int.from_bytes(raw, byteorder="little", signed=True)
    raise ValueError(f"unsupported sample width: {sample_width}")


def read_wav_energy(path: Path, window_sec: float = 0.10) -> Dict[str, Any]:
    with wave.open(str(path), "rb") as wf:
        nchannels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        framerate = wf.getframerate()
        nframes = wf.getnframes()
        frames = wf.readframes(nframes)

    if nchannels <= 0 or framerate <= 0 or nframes <= 0:
        raise ValueError("invalid wav metadata")

    frame_size = nchannels * sample_width
    window_frames = max(1, int(round(window_sec * framerate)))
    sample_max = float((2 ** (8 * sample_width - 1)) - 1) if sample_width > 1 else 128.0

    envelope: List[Dict[str, float]] = []
    cur_energy_sum = 0.0
    cur_count = 0
    cur_start_frame = 0

    for frame_idx in range(nframes):
        base = frame_idx * frame_size
        channel_values = []
        for ch in range(nchannels):
            b0 = base + ch * sample_width
            b1 = b0 + sample_width
            channel_values.append(decode_pcm_sample(frames[b0:b1], sample_width) / sample_max)

        mono = sum(channel_values) / len(channel_values)
        cur_energy_sum += mono * mono
        cur_count += 1

        is_window_end = cur_count >= window_frames or frame_idx == nframes - 1
        if is_window_end:
            center_frame = cur_start_frame + cur_count / 2.0
            center_sec = center_frame / framerate
            rms = math.sqrt(cur_energy_sum / max(1, cur_count))
            envelope.append({"centerSec": center_sec, "rms": rms})
            cur_energy_sum = 0.0
            cur_count = 0
            cur_start_frame = frame_idx + 1

    duration_sec = nframes / framerate
    rms_values = [x["rms"] for x in envelope]
    peak_rms = max(rms_values) if rms_values else 0.0
    median_rms = statistics.median(rms_values) if rms_values else 0.0
    mean_rms = sum(rms_values) / len(rms_values) if rms_values else 0.0

    threshold = max(peak_rms * 0.20, median_rms + 0.25 * max(0.0, peak_rms - median_rms), 1e-6)

    peak_item = max(envelope, key=lambda x: x["rms"]) if envelope else {"centerSec": None, "rms": None}
    active_items = [x for x in envelope if x["rms"] >= threshold]
    first_active_sec = active_items[0]["centerSec"] if active_items else None
    last_active_sec = active_items[-1]["centerSec"] if active_items else None

    return {
        "durationSec": duration_sec,
        "channels": nchannels,
        "sampleWidthBytes": sample_width,
        "sampleRate": framerate,
        "frameCount": nframes,
        "windowSec": window_sec,
        "peakSec": peak_item["centerSec"],
        "peakRms": peak_item["rms"],
        "medianRms": median_rms,
        "meanRms": mean_rms,
        "activeThresholdRms": threshold,
        "firstActiveSec": first_active_sec,
        "lastActiveSec": last_active_sec,
        "envelope": envelope,
    }


def ratio_active_in_window(envelope: List[Dict[str, float]], start: float, end: float, threshold: float) -> Tuple[float, float, int, int]:
    inside = [x for x in envelope if start <= x["centerSec"] <= end]
    outside = [x for x in envelope if not (start <= x["centerSec"] <= end)]

    inside_active = sum(1 for x in inside if x["rms"] >= threshold)
    outside_active = sum(1 for x in outside if x["rms"] >= threshold)

    inside_ratio = inside_active / len(inside) if inside else 0.0
    outside_ratio = outside_active / len(outside) if outside else 0.0
    return inside_ratio, outside_ratio, len(inside), len(outside)


def evaluate_row(row: Dict[str, str]) -> Dict[str, Any]:
    audio_uri = row.get("audioUri") or ""
    audio_path = ROOT / audio_uri

    expected_start = as_float(row.get("expectedStartSec"))
    expected_end = as_float(row.get("expectedEndSec"))
    expected_clip_duration = as_float(row.get("expectedDurationSec"))
    expected_window_duration = None
    if expected_start is not None and expected_end is not None:
        expected_window_duration = max(0.0, expected_end - expected_start)

    out: Dict[str, Any] = {
        "candidateRowId": row.get("candidateRowId"),
        "candidateId": row.get("candidateId"),
        "audioUri": audio_uri,
        "sceneId": row.get("sceneId"),
        "eventId": row.get("eventId"),
        "layer": row.get("layer"),
        "label": row.get("label"),
        "expectedStartSec": expected_start,
        "expectedEndSec": expected_end,
        "expectedClipDurationSec": expected_clip_duration,
        "expectedWindowDurationSec": expected_window_duration,
        "audioExists": audio_path.exists(),
        "audioReadable": False,
        "actualDurationSec": None,
        "durationAbsErrorSec": None,
        "peakSec": None,
        "firstActiveSec": None,
        "lastActiveSec": None,
        "peakInsideExpectedWindow": False,
        "firstActiveInsideExpectedWindow": False,
        "activeRatioInsideWindow": None,
        "activeRatioOutsideWindow": None,
        "alignmentDecision": "FAIL",
        "alignmentReason": "",
    }

    if not audio_path.exists():
        out["alignmentReason"] = "AUDIO_MISSING"
        return out

    if expected_start is None or expected_end is None:
        out["alignmentReason"] = "TIMING_WINDOW_MISSING"
        return out

    try:
        info = read_wav_energy(audio_path)
    except Exception as e:
        out["alignmentReason"] = f"AUDIO_READ_ERROR:{type(e).__name__}:{e}"
        return out

    out["audioReadable"] = True
    out["actualDurationSec"] = round(info["durationSec"], 6)

    if expected_clip_duration is not None:
        out["durationAbsErrorSec"] = round(abs(info["durationSec"] - expected_clip_duration), 6)

    peak_sec = info["peakSec"]
    first_active_sec = info["firstActiveSec"]
    last_active_sec = info["lastActiveSec"]

    out["peakSec"] = None if peak_sec is None else round(float(peak_sec), 6)
    out["firstActiveSec"] = None if first_active_sec is None else round(float(first_active_sec), 6)
    out["lastActiveSec"] = None if last_active_sec is None else round(float(last_active_sec), 6)

    peak_inside = peak_sec is not None and expected_start <= peak_sec <= expected_end
    first_inside = first_active_sec is not None and expected_start <= first_active_sec <= expected_end

    inside_ratio, outside_ratio, inside_n, outside_n = ratio_active_in_window(
        info["envelope"],
        expected_start,
        expected_end,
        float(info["activeThresholdRms"]),
    )

    out["peakInsideExpectedWindow"] = bool(peak_inside)
    out["firstActiveInsideExpectedWindow"] = bool(first_inside)
    out["activeRatioInsideWindow"] = round(inside_ratio, 6)
    out["activeRatioOutsideWindow"] = round(outside_ratio, 6)
    out["insideWindowFrameCount"] = inside_n
    out["outsideWindowFrameCount"] = outside_n

    layer = norm_text(row.get("layer"))
    duration_close = out["durationAbsErrorSec"] is None or out["durationAbsErrorSec"] <= 0.25

    if layer == "ambience":
        if duration_close and inside_ratio > 0.0:
            out["alignmentDecision"] = "PASS"
            out["alignmentReason"] = "AMBIENCE_SPANS_EXPECTED_WINDOW"
        else:
            out["alignmentDecision"] = "FAIL"
            out["alignmentReason"] = "AMBIENCE_DURATION_OR_ACTIVITY_WEAK"
    else:
        if duration_close and peak_inside:
            out["alignmentDecision"] = "PASS"
            out["alignmentReason"] = "PEAK_INSIDE_EXPECTED_WINDOW"
        elif duration_close and first_inside:
            out["alignmentDecision"] = "PARTIAL"
            out["alignmentReason"] = "FIRST_ACTIVE_INSIDE_BUT_PEAK_OUTSIDE"
        elif duration_close and inside_ratio > outside_ratio:
            out["alignmentDecision"] = "PARTIAL"
            out["alignmentReason"] = "MORE_ACTIVITY_INSIDE_THAN_OUTSIDE"
        else:
            out["alignmentDecision"] = "FAIL"
            out["alignmentReason"] = "ENERGY_NOT_ALIGNED_TO_EXPECTED_WINDOW"

    return out


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields = [
        "candidateRowId",
        "candidateId",
        "audioUri",
        "sceneId",
        "eventId",
        "layer",
        "label",
        "expectedStartSec",
        "expectedEndSec",
        "expectedClipDurationSec",
        "expectedWindowDurationSec",
        "audioExists",
        "audioReadable",
        "actualDurationSec",
        "durationAbsErrorSec",
        "peakSec",
        "firstActiveSec",
        "lastActiveSec",
        "peakInsideExpectedWindow",
        "firstActiveInsideExpectedWindow",
        "activeRatioInsideWindow",
        "activeRatioOutsideWindow",
        "insideWindowFrameCount",
        "outsideWindowFrameCount",
        "alignmentDecision",
        "alignmentReason",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    if not INPUT_QUEUE_CSV.exists():
        raise FileNotFoundError(f"missing input queue: {INPUT_QUEUE_CSV}")

    with INPUT_QUEUE_CSV.open("r", encoding="utf-8", errors="replace", newline="") as f:
        input_rows = list(csv.DictReader(f))

    rows = [evaluate_row(row) for row in input_rows]

    candidate_count = len(rows)
    readable_count = sum(1 for r in rows if r["audioReadable"])
    pass_count = sum(1 for r in rows if r["alignmentDecision"] == "PASS")
    partial_count = sum(1 for r in rows if r["alignmentDecision"] == "PARTIAL")
    fail_count = sum(1 for r in rows if r["alignmentDecision"] == "FAIL")
    peak_inside_count = sum(1 for r in rows if r["peakInsideExpectedWindow"])
    first_inside_count = sum(1 for r in rows if r["firstActiveInsideExpectedWindow"])

    foley_rows = [r for r in rows if norm_text(r.get("layer")) == "foley"]
    ambience_rows = [r for r in rows if norm_text(r.get("layer")) == "ambience"]
    foley_pass_count = sum(1 for r in foley_rows if r["alignmentDecision"] == "PASS")
    foley_partial_count = sum(1 for r in foley_rows if r["alignmentDecision"] == "PARTIAL")
    ambience_pass_count = sum(1 for r in ambience_rows if r["alignmentDecision"] == "PASS")

    blockers = []
    warnings = []

    if candidate_count == 0:
        blockers.append("NO_CANDIDATES")
    if readable_count < candidate_count:
        blockers.append("AUDIO_READ_INCOMPLETE")
    if pass_count + partial_count == 0:
        blockers.append("NO_TEMPORAL_ALIGNMENT_SIGNAL")
    if foley_rows and foley_pass_count == 0 and foley_partial_count == 0:
        blockers.append("FOLEY_EVENTS_NOT_ALIGNED")
    if any(r["expectedWindowDurationSec"] is not None and r["expectedClipDurationSec"] is not None and abs(r["expectedWindowDurationSec"] - r["expectedClipDurationSec"]) > 0.25 for r in rows):
        warnings.append("EXPECTED_CLIP_DURATION_DIFFERS_FROM_EVENT_WINDOW_DURATION")

    if blockers:
        status = "FAIL"
    elif fail_count > 0:
        status = "PARTIAL"
    else:
        status = "PASS"

    report = {
        "status": status,
        "candidateCount": candidate_count,
        "audioReadableCount": readable_count,
        "alignmentPassCount": pass_count,
        "alignmentPartialCount": partial_count,
        "alignmentFailCount": fail_count,
        "peakInsideWindowCount": peak_inside_count,
        "firstActiveInsideWindowCount": first_inside_count,
        "foleyCount": len(foley_rows),
        "foleyPassCount": foley_pass_count,
        "foleyPartialCount": foley_partial_count,
        "ambienceCount": len(ambience_rows),
        "ambiencePassCount": ambience_pass_count,
        "blockers": blockers,
        "warnings": warnings,
        "outputs": {
            "alignmentCsv": str(OUT_CSV.relative_to(ROOT)),
            "alignmentJson": str(OUT_JSON.relative_to(ROOT)),
            "alignmentReport": str(OUT_REPORT.relative_to(ROOT)),
        },
        "boundaryStatement": (
            "This probe checks simple RMS energy timing against expected windows. "
            "It is not semantic quality, human audition, final mix readiness, or production readiness."
        ),
    }

    write_csv(OUT_CSV, rows)
    OUT_JSON.write_text(json.dumps({"items": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if status in {"PASS", "PARTIAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())