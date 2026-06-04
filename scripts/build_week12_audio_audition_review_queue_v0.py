#!/usr/bin/env python3
"""
Build Week12 audio audition review queue v0.

Purpose:
- Convert procedural audio audition metrics into a reviewable candidate queue.
- Do NOT claim semantic fidelity, human audition pass, mix readiness, or production storage.
- Provide stable JSON/CSV outputs for Java and Cloud consumption.

Input priority:
1. artifacts/manifests/week12_procedural_audio_audition_metrics_v0.jsonl
2. artifacts/manifests/week12_procedural_audio_audition_metrics_v0.csv
3. artifacts/manifests/week12_procedural_audio_candidates_v0.jsonl

Outputs:
- artifacts/evals/week12_audio_audition_review_queue_v0.json
- artifacts/evals/week12_audio_audition_review_queue_v0.csv
"""

from __future__ import annotations

import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


ROOT = Path(".").resolve()

AUDITION_MANIFEST = Path("artifacts/manifests/week12_procedural_audio_audition_manifest_v0.json")
METRICS_JSONL = Path("artifacts/manifests/week12_procedural_audio_audition_metrics_v0.jsonl")
METRICS_CSV = Path("artifacts/manifests/week12_procedural_audio_audition_metrics_v0.csv")
CANDIDATES_JSONL = Path("artifacts/manifests/week12_procedural_audio_candidates_v0.jsonl")

OUT_JSON = Path("artifacts/evals/week12_audio_audition_review_queue_v0.json")
OUT_CSV = Path("artifacts/evals/week12_audio_audition_review_queue_v0.csv")

DEFAULT_VISUAL = Path("artifacts/audition/week12_procedural_baseline_v0/waveform_contact_sheet.svg")
DEFAULT_HTML = Path("artifacts/audition/week12_procedural_baseline_v0/index.html")


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


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"ERROR: invalid JSON: {path}: {exc}") from exc


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"ERROR: invalid JSONL at {path}:{lineno}: {exc}") from exc
            if isinstance(item, dict):
                rows.append(item)
    return rows


def load_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def pick(d: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in d and d[key] not in (None, ""):
            return d[key]
    return default


def to_float(x: Any) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        val = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(val) or math.isinf(val):
        return None
    return val


def to_int(x: Any) -> Optional[int]:
    if x is None or x == "":
        return None
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return None


def to_bool_or_none(x: Any) -> Optional[bool]:
    if isinstance(x, bool):
        return x
    if x is None or x == "":
        return None
    if isinstance(x, (int, float)):
        return bool(x)
    s = str(x).strip().lower()
    if s in {"true", "1", "yes", "y", "ok", "pass", "passed"}:
        return True
    if s in {"false", "0", "no", "n", "fail", "failed"}:
        return False
    return None


def rel_exists(uri: Optional[str]) -> Optional[bool]:
    if not uri:
        return None
    if uri.startswith(("http://", "https://", "s3://", "gs://")):
        return None
    p = Path(uri)
    return p.exists()


def infer_duration_expected(row: Dict[str, Any]) -> Optional[float]:
    explicit = to_float(
        pick(
            row,
            "expectedDurationSec",
            "expected_duration_sec",
            "targetDurationSec",
            "target_duration_sec",
            "requestDurationSec",
            "request_duration_sec",
        )
    )
    if explicit is not None:
        return explicit

    start = to_float(pick(row, "expectedStartSec", "startSec", "start_sec", "eventStartSec"))
    end = to_float(pick(row, "expectedEndSec", "endSec", "end_sec", "eventEndSec"))
    if start is not None and end is not None and end >= start:
        return round(end - start, 6)
    return None


def duration_match(actual: Optional[float], expected: Optional[float]) -> Optional[bool]:
    if actual is None or expected is None:
        return None
    tolerance = max(0.25, expected * 0.10)
    return abs(actual - expected) <= tolerance


def normalize_row(row: Dict[str, Any], index: int) -> Dict[str, Any]:
    candidate_uri = pick(
        row,
        "candidateUri",
        "candidate_uri",
        "audioUri",
        "audio_uri",
        "wavUri",
        "wav_uri",
        "path",
        "audioPath",
        "audio_path",
    )

    duration_sec = to_float(
        pick(
            row,
            "durationSec",
            "duration_sec",
            "actualDurationSec",
            "actual_duration_sec",
            "audioDurationSec",
            "audio_duration_sec",
            "duration",
        )
    )
    expected_duration_sec = infer_duration_expected(row)
    duration_matches = duration_match(duration_sec, expected_duration_sec)

    explicit_format_ok = to_bool_or_none(pick(row, "formatOk", "format_ok", "wavOk", "wav_ok"))
    exists_flag = rel_exists(str(candidate_uri)) if candidate_uri else None
    suffix_ok = str(candidate_uri).lower().endswith(".wav") if candidate_uri else None

    if explicit_format_ok is not None:
        format_ok = explicit_format_ok
    elif exists_flag is not None and suffix_ok is not None:
        format_ok = bool(exists_flag and suffix_ok)
    elif suffix_ok is not None:
        format_ok = bool(suffix_ok)
    else:
        format_ok = None

    failure_tags: List[str] = [
        "human_audition_required",
        "semantic_unverified",
    ]
    if format_ok is False:
        failure_tags.append("format_check_failed")
    elif format_ok is None:
        failure_tags.append("format_unverified")

    if duration_matches is False:
        failure_tags.append("duration_mismatch")
    elif duration_matches is None:
        failure_tags.append("duration_unverified")

    if duration_sec is None:
        failure_tags.append("duration_missing")

    candidate_id = pick(
        row,
        "candidateId",
        "candidate_id",
        "id",
        "audioCandidateId",
        "audio_candidate_id",
        default=f"week12_candidate_{index:04d}",
    )

    expected_start = to_float(pick(row, "expectedStartSec", "startSec", "start_sec", "eventStartSec"))
    expected_end = to_float(pick(row, "expectedEndSec", "endSec", "end_sec", "eventEndSec"))

    duration_delta = None
    if duration_sec is not None and expected_duration_sec is not None:
        duration_delta = round(duration_sec - expected_duration_sec, 6)

    return {
        "candidateId": str(candidate_id),
        "sourceRequestId": pick(row, "sourceRequestId", "requestId", "request_id", "audioRequestId", default=""),
        "caseId": pick(row, "caseId", "case_id", "seedId", "seed_id", default=""),
        "sceneId": pick(row, "sceneId", "scene_id", default=""),
        "eventId": pick(row, "eventId", "event_id", default=""),
        "eventLabel": pick(row, "eventLabel", "event_label", "label", "soundEvent", default="UNKNOWN_EVENT"),
        "layer": pick(row, "layer", "audioLayer", "audio_layer", default="unknown"),
        "candidateUri": str(candidate_uri or ""),
        "expectedStartSec": expected_start,
        "expectedEndSec": expected_end,
        "expectedDurationSec": expected_duration_sec,
        "durationSec": duration_sec,
        "durationDeltaSec": duration_delta,
        "durationMatchesExpected": duration_matches,
        "sampleRateHz": to_int(pick(row, "sampleRateHz", "sample_rate_hz", "sampleRate", "sample_rate")),
        "channels": to_int(pick(row, "channels", "numChannels", "num_channels")),
        "rmsDbfs": to_float(pick(row, "rmsDbfs", "rms_dbfs", "rms")),
        "peakDbfs": to_float(pick(row, "peakDbfs", "peak_dbfs", "peak")),
        "formatOk": format_ok,
        "reviewStatus": "HUMAN_AUDITION_REQUIRED",
        "humanAuditionRequired": True,
        "semanticFidelityClaimed": False,
        "mixReadyClaimed": False,
        "failureTags": "|".join(dict.fromkeys(failure_tags)),
        "reviewNote": "Procedural fallback candidate. Requires human audition before semantic pass or mix readiness can be claimed.",
    }


def load_source_rows() -> tuple[List[Dict[str, Any]], str]:
    rows = load_jsonl(METRICS_JSONL)
    if rows:
        return rows, str(METRICS_JSONL)

    rows = load_csv(METRICS_CSV)
    if rows:
        return rows, str(METRICS_CSV)

    rows = load_jsonl(CANDIDATES_JSONL)
    if rows:
        return rows, str(CANDIDATES_JSONL)

    raise SystemExit(
        "ERROR: no candidate metrics found. Expected one of: "
        f"{METRICS_JSONL}, {METRICS_CSV}, {CANDIDATES_JSONL}"
    )


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in FIELDNAMES})


def main() -> int:
    raw_rows, source_path = load_source_rows()
    manifest = load_json(AUDITION_MANIFEST)

    queue = [normalize_row(row, i + 1) for i, row in enumerate(raw_rows)]
    queue.sort(key=lambda r: (str(r.get("caseId", "")), str(r.get("eventId", "")), str(r.get("candidateId", ""))))

    candidate_count = len(queue)
    expected_count = to_int(
        pick(
            manifest,
            "candidateCount",
            "candidate_count",
            "qaRecordCount",
            "qa_record_count",
            default=None,
        )
    )

    blockers: List[str] = []
    if candidate_count == 0:
        blockers.append("NO_CANDIDATE_ROWS")
    if expected_count is not None and candidate_count != expected_count:
        blockers.append(f"CANDIDATE_COUNT_MISMATCH_EXPECTED_{expected_count}_GOT_{candidate_count}")

    human_required = sum(1 for r in queue if r["humanAuditionRequired"])
    format_failed = sum(1 for r in queue if r["formatOk"] is False)
    duration_mismatch = sum(1 for r in queue if r["durationMatchesExpected"] is False)

    output = {
        "schemaVersion": "week12.audio_audition_review_queue.v0",
        "generatedAt": utc_now(),
        "status": "PASS" if candidate_count > 0 and not blockers else "FAIL",
        "sourceMetricsUri": source_path,
        "sourceManifestUri": str(AUDITION_MANIFEST) if AUDITION_MANIFEST.exists() else None,
        "candidateCount": candidate_count,
        "expectedCandidateCountFromManifest": expected_count,
        "humanAuditionRequiredCount": human_required,
        "formatFailedCount": format_failed,
        "durationMismatchCount": duration_mismatch,
        "allRequireHumanAudition": human_required == candidate_count and candidate_count > 0,
        "semanticFidelityClaimedAny": False,
        "mixReadyClaimedAny": False,
        "doesNotClaim": [
            "semantic_audio_quality_passed",
            "human_audition_passed",
            "final_mix_readiness",
            "production_asset_storage",
            "model_generated_audio_quality",
        ],
        "blockers": blockers,
        "importantVisuals": [
            str(DEFAULT_VISUAL),
        ],
        "auditionHtml": str(DEFAULT_HTML),
        "reviewPolicy": {
            "defaultReviewStatus": "HUMAN_AUDITION_REQUIRED",
            "semanticPassRequiresHumanReview": True,
            "mixReadyRequiresSeparateEval": True,
            "proceduralFallbackIsNotSemanticQualityEvidence": True,
        },
        "reviewQueue": queue,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(OUT_CSV, queue)

    print(
        json.dumps(
            {
                "status": output["status"],
                "candidateCount": candidate_count,
                "humanAuditionRequiredCount": human_required,
                "semanticFidelityClaimedAny": False,
                "mixReadyClaimedAny": False,
                "blockers": blockers,
                "json": str(OUT_JSON),
                "csv": str(OUT_CSV),
                "importantVisual": str(DEFAULT_VISUAL),
                "auditionHtml": str(DEFAULT_HTML),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    return 0 if output["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())