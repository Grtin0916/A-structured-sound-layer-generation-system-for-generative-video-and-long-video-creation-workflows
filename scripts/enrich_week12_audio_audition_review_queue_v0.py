#!/usr/bin/env python3
"""
Enrich Week12 audio audition review queue v0 with real WAV metadata.

This script does not regenerate audio.
It updates the canonical review queue JSON/CSV with:
- parsed caseId / eventId from sourceRequestId
- WAV durationSec / sampleRateHz / channels / sampleWidthBytes
- rmsDbfs / peakDbfs
- corrected duration-related failure tags

Boundary:
- Still requires human audition.
- Still does not claim semantic fidelity.
- Still does not claim mix readiness.
"""

from __future__ import annotations

import csv
import json
import math
import re
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


QUEUE_JSON = Path("artifacts/evals/week12_audio_audition_review_queue_v0.json")
QUEUE_CSV = Path("artifacts/evals/week12_audio_audition_review_queue_v0.csv")

FIELDNAMES = [
    "candidateId",
    "sourceRequestId",
    "caseId",
    "sceneId",
    "eventId",
    "eventLabel",
    "layer",
    "candidateUri",
    "expectedStartSec",
    "expectedEndSec",
    "expectedDurationSec",
    "durationSec",
    "durationDeltaSec",
    "durationMatchesExpected",
    "sampleRateHz",
    "channels",
    "sampleWidthBytes",
    "rmsDbfs",
    "peakDbfs",
    "formatOk",
    "reviewStatus",
    "humanAuditionRequired",
    "semanticFidelityClaimed",
    "mixReadyClaimed",
    "failureTags",
    "reviewNote",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def to_float(x: Any) -> Optional[float]:
    if x in (None, ""):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def parse_ids(row: Dict[str, Any]) -> None:
    src = str(row.get("sourceRequestId") or "")

    if not row.get("caseId"):
        m = re.search(r"(seed_\d+_case_[A-Za-z0-9]+)", src)
        if m:
            row["caseId"] = m.group(1)

    if not row.get("sceneId") and row.get("caseId"):
        row["sceneId"] = row["caseId"]

    if not row.get("eventId"):
        m = re.search(r"(evt_\d+)", src)
        if m:
            row["eventId"] = m.group(1)


def pcm_samples_from_frames(frames: bytes, sample_width: int) -> Iterable[int]:
    if sample_width == 1:
        for b in frames:
            yield b - 128
        return

    if sample_width == 2:
        for i in range(0, len(frames), 2):
            yield int.from_bytes(frames[i:i + 2], "little", signed=True)
        return

    if sample_width == 3:
        for i in range(0, len(frames), 3):
            chunk = frames[i:i + 3]
            if len(chunk) == 3:
                yield int.from_bytes(chunk, "little", signed=True)
        return

    if sample_width == 4:
        for i in range(0, len(frames), 4):
            yield int.from_bytes(frames[i:i + 4], "little", signed=True)
        return

    raise ValueError(f"unsupported sample width: {sample_width}")


def dbfs_from_linear(x: float) -> float:
    if x <= 0:
        return -120.0
    return round(20.0 * math.log10(x), 3)


def read_wav_metrics(path: Path) -> Dict[str, Any]:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        nframes = wf.getnframes()
        frames = wf.readframes(nframes)

    duration = round(nframes / sample_rate, 6) if sample_rate else None
    max_amp = 128.0 if sample_width == 1 else float(2 ** (8 * sample_width - 1))

    count = 0
    sum_sq = 0.0
    peak = 0.0

    for s in pcm_samples_from_frames(frames, sample_width):
        a = abs(float(s))
        peak = max(peak, a)
        sum_sq += a * a
        count += 1

    if count == 0:
        rms_linear = 0.0
        peak_linear = 0.0
    else:
        rms_linear = math.sqrt(sum_sq / count) / max_amp
        peak_linear = peak / max_amp

    return {
        "durationSec": duration,
        "sampleRateHz": sample_rate,
        "channels": channels,
        "sampleWidthBytes": sample_width,
        "rmsDbfs": dbfs_from_linear(rms_linear),
        "peakDbfs": dbfs_from_linear(peak_linear),
        "formatOk": True,
    }


def duration_match(actual: Optional[float], expected: Optional[float]) -> Optional[bool]:
    if actual is None or expected is None:
        return None
    tol = max(0.25, expected * 0.10)
    return abs(actual - expected) <= tol


def normalize_tags(row: Dict[str, Any], audio_probe_ok: bool) -> str:
    tags = set(
        t.strip()
        for t in str(row.get("failureTags") or "").split("|")
        if t.strip()
    )

    tags.add("human_audition_required")
    tags.add("semantic_unverified")

    if audio_probe_ok:
        tags.discard("duration_missing")
        tags.discard("format_unverified")
        tags.discard("audio_probe_failed")
    else:
        tags.add("audio_probe_failed")
        tags.add("duration_missing")

    duration = to_float(row.get("durationSec"))
    expected = to_float(row.get("expectedDurationSec"))

    if duration is not None and expected is None:
        tags.discard("duration_missing")
        tags.add("expected_timing_unverified")
        tags.discard("duration_unverified")

    if duration is not None and expected is not None:
        tags.discard("duration_missing")
        tags.discard("duration_unverified")
        tags.discard("expected_timing_unverified")
        if duration_match(duration, expected):
            tags.discard("duration_mismatch")
        else:
            tags.add("duration_mismatch")

    ordered = [
        "human_audition_required",
        "semantic_unverified",
        "expected_timing_unverified",
        "duration_unverified",
        "duration_missing",
        "duration_mismatch",
        "format_unverified",
        "audio_probe_failed",
    ]
    rest = sorted(t for t in tags if t not in ordered)
    return "|".join([t for t in ordered if t in tags] + rest)


def write_csv(rows: List[Dict[str, Any]]) -> None:
    QUEUE_CSV.parent.mkdir(parents=True, exist_ok=True)
    with QUEUE_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in FIELDNAMES})


def main() -> int:
    if not QUEUE_JSON.exists():
        raise SystemExit(f"ERROR: missing {QUEUE_JSON}")

    obj = json.loads(QUEUE_JSON.read_text(encoding="utf-8"))
    rows = obj.get("reviewQueue") or []
    if not isinstance(rows, list) or not rows:
        raise SystemExit("ERROR: reviewQueue is empty or invalid")

    audio_probe_ok_count = 0
    audio_probe_failed_count = 0

    for row in rows:
        parse_ids(row)
        candidate_uri = str(row.get("candidateUri") or "")
        path = Path(candidate_uri)

        audio_probe_ok = False
        if candidate_uri and path.exists():
            try:
                metrics = read_wav_metrics(path)
                row.update(metrics)
                audio_probe_ok = True
                audio_probe_ok_count += 1
            except Exception as exc:
                row["formatOk"] = False
                row["audioProbeError"] = f"{type(exc).__name__}: {exc}"
                audio_probe_failed_count += 1
        else:
            row["formatOk"] = False
            row["audioProbeError"] = "candidateUri missing or file not found"
            audio_probe_failed_count += 1

        expected = to_float(row.get("expectedDurationSec"))
        duration = to_float(row.get("durationSec"))

        if duration is not None and expected is not None:
            row["durationDeltaSec"] = round(duration - expected, 6)
            row["durationMatchesExpected"] = duration_match(duration, expected)
        elif duration is not None:
            row["durationDeltaSec"] = None
            row["durationMatchesExpected"] = None

        row["reviewStatus"] = "HUMAN_AUDITION_REQUIRED"
        row["humanAuditionRequired"] = True
        row["semanticFidelityClaimed"] = False
        row["mixReadyClaimed"] = False
        row["failureTags"] = normalize_tags(row, audio_probe_ok)
        row["reviewNote"] = (
            "Procedural fallback candidate with WAV metadata probe. "
            "Requires human audition before semantic pass or mix readiness can be claimed."
        )

    candidate_count = len(rows)
    human_count = sum(1 for r in rows if r.get("humanAuditionRequired") is True)
    format_failed = sum(1 for r in rows if r.get("formatOk") is False)
    duration_missing = sum(1 for r in rows if r.get("durationSec") in (None, ""))
    sample_rate_missing = sum(1 for r in rows if r.get("sampleRateHz") in (None, ""))
    event_id_missing = sum(1 for r in rows if not r.get("eventId"))

    blockers = []
    if candidate_count == 0:
        blockers.append("NO_CANDIDATES")
    if audio_probe_failed_count > 0:
        blockers.append("AUDIO_PROBE_FAILED")
    if duration_missing > 0:
        blockers.append("DURATION_MISSING")
    if sample_rate_missing > 0:
        blockers.append("SAMPLE_RATE_MISSING")
    if event_id_missing > 0:
        blockers.append("EVENT_ID_MISSING")

    obj.update({
        "generatedAt": utc_now(),
        "status": "PASS" if not blockers else "PARTIAL",
        "candidateCount": candidate_count,
        "humanAuditionRequiredCount": human_count,
        "allRequireHumanAudition": human_count == candidate_count and candidate_count > 0,
        "semanticFidelityClaimedAny": False,
        "mixReadyClaimedAny": False,
        "audioProbeOkCount": audio_probe_ok_count,
        "audioProbeFailedCount": audio_probe_failed_count,
        "durationMissingCount": duration_missing,
        "sampleRateMissingCount": sample_rate_missing,
        "eventIdMissingCount": event_id_missing,
        "formatFailedCount": format_failed,
        "blockers": blockers,
        "metadataEnrichment": {
            "wavProbe": True,
            "caseIdParsedFromSourceRequestId": True,
            "eventIdParsedFromSourceRequestId": True,
            "doesNotRegenerateAudio": True,
            "doesNotClaimSemanticQuality": True,
        },
        "doesNotClaim": [
            "semantic_audio_quality_passed",
            "human_audition_passed",
            "final_mix_readiness",
            "production_asset_storage",
            "model_generated_audio_quality",
        ],
        "reviewQueue": rows,
    })

    QUEUE_JSON.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(rows)

    print(json.dumps({
        "status": obj["status"],
        "candidateCount": candidate_count,
        "audioProbeOkCount": audio_probe_ok_count,
        "audioProbeFailedCount": audio_probe_failed_count,
        "durationMissingCount": duration_missing,
        "sampleRateMissingCount": sample_rate_missing,
        "eventIdMissingCount": event_id_missing,
        "formatFailedCount": format_failed,
        "blockers": blockers,
        "json": str(QUEUE_JSON),
        "csv": str(QUEUE_CSV),
    }, ensure_ascii=False, indent=2))

    return 0 if obj["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())