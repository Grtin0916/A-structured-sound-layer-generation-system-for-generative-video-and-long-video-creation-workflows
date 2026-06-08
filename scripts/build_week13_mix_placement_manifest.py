#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


ROOT = Path.cwd()

BINDING_REPORT_PATH = ROOT / "artifacts/manifests/week12_audio_candidate_timing_binding_report_v2.json"
TEMPORAL_REPORT_PATH = ROOT / "artifacts/manifests/week12_temporal_alignment_probe_report_v1.json"
BOUND_QUEUE_PATH = ROOT / "artifacts/evals/week12_audio_candidate_timing_bound_queue_v2.json"

OUT_MANIFEST_PATH = ROOT / "artifacts/manifests/week13_mix_placement_manifest.json"
OUT_TABLE_JSON_PATH = ROOT / "artifacts/evals/week13_mix_global_placement_table.json"
OUT_TABLE_CSV_PATH = ROOT / "artifacts/evals/week13_mix_global_placement_table.csv"
OUT_LOG_PATH = ROOT / "artifacts/logs" / f"week13_mix_placement_manifest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"missing required file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def git_short_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip()
    except Exception:
        return "UNKNOWN"


def dec(value: Any) -> Decimal | None:
    if value is None or value == "":
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None


def fnum(value: Decimal | None, ndigits: int = 6) -> float | None:
    if value is None:
        return None
    return float(round(value, ndigits))


def infer_asset_time_mode(item: dict[str, Any]) -> str:
    explicit = item.get("assetTimeMode") or item.get("asset_time_mode") or item.get("timeMode")
    if explicit:
        return str(explicit).strip()

    layer = str(item.get("expectedLayer") or item.get("layer") or item.get("raw", {}).get("layer") or "").lower()
    if layer == "ambience":
        return "full_clip"
    if layer == "foley":
        return "event_local"
    return "unknown"


def source_type_from_uri(uri: str) -> str:
    if "procedural_baseline" in uri or "procedural" in uri:
        return "procedural_baseline_v0"
    if "musicgen" in uri.lower():
        return "musicgen"
    if "stable" in uri.lower():
        return "stable_audio"
    if "foley" in uri.lower():
        return "foley_source"
    return "unknown"


def build_row(item: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []

    raw = item.get("raw") if isinstance(item.get("raw"), dict) else {}

    candidate_id = item.get("candidateId") or raw.get("candidateId")
    if not candidate_id:
        blockers.append("MISSING_CANDIDATE_ID")

    audio_uri = item.get("audioUri") or raw.get("candidateUri") or raw.get("audioUri")
    if not audio_uri:
        blockers.append("MISSING_AUDIO_URI")

    layer = item.get("expectedLayer") or item.get("layer") or raw.get("layer") or "unknown"
    asset_time_mode = infer_asset_time_mode(item)

    expected_start = dec(item.get("expectedStartSec"))
    expected_end = dec(item.get("expectedEndSec"))

    actual_duration = (
        dec(item.get("actualDurationSec"))
        or dec(item.get("candidateDurationSec"))
        or dec(raw.get("durationSec"))
        or dec(item.get("expectedWindowDurationSec"))
    )

    if actual_duration is None and expected_start is not None and expected_end is not None:
        actual_duration = expected_end - expected_start

    if expected_start is None:
        blockers.append("MISSING_EXPECTED_START_SEC")
    if expected_end is None:
        blockers.append("MISSING_EXPECTED_END_SEC")
    if actual_duration is None:
        blockers.append("MISSING_AUDIO_OR_WINDOW_DURATION_SEC")

    placement_required = asset_time_mode == "event_local"

    if asset_time_mode == "full_clip":
        global_start = Decimal("0")
        placement_offset = Decimal("0")
        placement_reason = "FULL_CLIP_DIRECT_GLOBAL_TIMELINE"
    elif asset_time_mode == "event_local":
        global_start = expected_start
        placement_offset = expected_start
        placement_reason = "EVENT_LOCAL_OFFSET_BY_EXPECTED_START_SEC"
    else:
        global_start = expected_start
        placement_offset = expected_start
        placement_reason = "UNKNOWN_MODE_FALLBACK_TO_EXPECTED_START_SEC"
        blockers.append("UNKNOWN_ASSET_TIME_MODE")

    global_end = None
    if global_start is not None and actual_duration is not None:
        global_end = global_start + actual_duration
    elif expected_end is not None:
        global_end = expected_end

    misplaced_from_zero = False
    if asset_time_mode == "event_local":
        if expected_start is None or global_start is None:
            misplaced_from_zero = True
        elif global_start != expected_start:
            misplaced_from_zero = True
        elif global_start == Decimal("0") and expected_start != Decimal("0"):
            misplaced_from_zero = True

    expected_end_delta = None
    if global_end is not None and expected_end is not None:
        expected_end_delta = abs(global_end - expected_end)

    if expected_end_delta is not None and expected_end_delta > Decimal("0.02"):
        blockers.append(f"GLOBAL_END_EXPECTED_END_MISMATCH_{expected_end_delta}")

    placement_status = "PASS" if not blockers else "FAIL"

    row = {
        "candidateRowId": item.get("candidateRowId"),
        "candidateId": candidate_id,
        "audioUri": audio_uri,
        "sourceType": source_type_from_uri(str(audio_uri or "")),
        "sceneId": item.get("expectedSceneId") or item.get("sceneId") or raw.get("sceneId"),
        "caseId": item.get("expectedCaseId") or item.get("caseId") or raw.get("caseId"),
        "eventId": item.get("expectedEventId") or item.get("eventId") or raw.get("eventId"),
        "layer": layer,
        "label": item.get("expectedLabel") or item.get("label") or raw.get("eventLabel"),
        "assetTimeMode": asset_time_mode,
        "placementRequired": placement_required,
        "expectedStartSec": fnum(expected_start),
        "expectedEndSec": fnum(expected_end),
        "actualDurationSec": fnum(actual_duration),
        "globalStartSec": fnum(global_start),
        "globalEndSec": fnum(global_end),
        "placementOffsetSec": fnum(placement_offset),
        "timingBindingStatus": item.get("timingBindingStatus"),
        "bindingMethod": item.get("bindingMethod"),
        "bindingConfidence": item.get("bindingConfidence"),
        "placementStatus": placement_status,
        "placementReason": placement_reason,
        "misplacedFromZero": misplaced_from_zero,
        "expectedEndDeltaSec": fnum(expected_end_delta),
        "blockers": blockers,
    }
    return row, blockers


def main() -> int:
    binding_report = load_json(BINDING_REPORT_PATH)
    temporal_report = load_json(TEMPORAL_REPORT_PATH)
    bound_queue = load_json(BOUND_QUEUE_PATH)

    items = bound_queue.get("items")
    if not isinstance(items, list):
        raise ValueError(f"{BOUND_QUEUE_PATH} must contain an items list")

    rows: list[dict[str, Any]] = []
    blockers: list[str] = []

    for item in items:
        row, row_blockers = build_row(item)
        rows.append(row)
        for b in row_blockers:
            blockers.append(f"{row.get('candidateId') or row.get('candidateRowId')}: {b}")

    candidate_count = len(rows)
    placement_count = sum(1 for r in rows if r["placementStatus"] == "PASS")
    full_clip_count = sum(1 for r in rows if r["assetTimeMode"] == "full_clip")
    event_local_count = sum(1 for r in rows if r["assetTimeMode"] == "event_local")
    event_local_required_count = sum(1 for r in rows if r["placementRequired"] is True)
    misplaced_from_zero_count = sum(1 for r in rows if r["misplacedFromZero"] is True)
    naive_zero_would_misplace_count = sum(
        1
        for r in rows
        if r["assetTimeMode"] == "event_local"
        and r["expectedStartSec"] not in (None, 0, 0.0)
    )

    status_blockers: list[str] = list(blockers)

    if binding_report.get("status") != "PASS":
        status_blockers.append("WEEK12_TIMING_BINDING_REPORT_NOT_PASS")
    if temporal_report.get("status") != "PASS":
        status_blockers.append("WEEK12_TEMPORAL_ALIGNMENT_REPORT_NOT_PASS")
    if candidate_count != 10:
        status_blockers.append(f"EXPECTED_10_CANDIDATES_GOT_{candidate_count}")
    if placement_count != candidate_count:
        status_blockers.append(f"PLACEMENT_COUNT_MISMATCH_{placement_count}_OF_{candidate_count}")
    if full_clip_count != 5:
        status_blockers.append(f"EXPECTED_5_FULL_CLIP_GOT_{full_clip_count}")
    if event_local_count != 5:
        status_blockers.append(f"EXPECTED_5_EVENT_LOCAL_GOT_{event_local_count}")
    if misplaced_from_zero_count != 0:
        status_blockers.append(f"EVENT_LOCAL_MISPLACED_FROM_ZERO_COUNT_{misplaced_from_zero_count}")

    status = "PASS" if not status_blockers else "FAIL"

    manifest = {
        "status": status,
        "scope": "week13_mix_placement_manifest_v0",
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "sourceRepo": {
            "path": str(ROOT),
            "head": git_short_head(),
        },
        "inputs": {
            "timingBindingReport": str(BINDING_REPORT_PATH.relative_to(ROOT)),
            "temporalAlignmentReport": str(TEMPORAL_REPORT_PATH.relative_to(ROOT)),
            "boundQueueJson": str(BOUND_QUEUE_PATH.relative_to(ROOT)),
        },
        "outputs": {
            "manifest": str(OUT_MANIFEST_PATH.relative_to(ROOT)),
            "globalPlacementTableJson": str(OUT_TABLE_JSON_PATH.relative_to(ROOT)),
            "globalPlacementTableCsv": str(OUT_TABLE_CSV_PATH.relative_to(ROOT)),
            "log": str(OUT_LOG_PATH.relative_to(ROOT)),
        },
        "candidateCount": candidate_count,
        "placementCount": placement_count,
        "assetTimeModeCounts": {
            "full_clip": full_clip_count,
            "event_local": event_local_count,
        },
        "eventLocalPlacementRequiredCount": event_local_required_count,
        "misplacedFromZeroCount": misplaced_from_zero_count,
        "naiveZeroWouldMisplaceCount": naive_zero_would_misplace_count,
        "boundaryStatement": (
            "PASS only means candidates have a deterministic global placement plan. "
            "It does not claim semantic audio quality, human audition, final mix readiness, "
            "production mixer behavior, durable registry, or real cloud storage."
        ),
        "placementRule": {
            "full_clip": "globalStartSec=0; audio is interpreted in the global scene coordinate frame.",
            "event_local": "globalStartSec=expectedStartSec; local audio coordinates are offset into the global timeline.",
        },
        "blockers": status_blockers,
    }

    OUT_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_TABLE_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_TABLE_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    OUT_TABLE_JSON_PATH.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = [
        "candidateRowId",
        "candidateId",
        "audioUri",
        "sourceType",
        "sceneId",
        "caseId",
        "eventId",
        "layer",
        "label",
        "assetTimeMode",
        "placementRequired",
        "expectedStartSec",
        "expectedEndSec",
        "actualDurationSec",
        "globalStartSec",
        "globalEndSec",
        "placementOffsetSec",
        "timingBindingStatus",
        "bindingMethod",
        "bindingConfidence",
        "placementStatus",
        "placementReason",
        "misplacedFromZero",
        "expectedEndDeltaSec",
    ]

    with OUT_TABLE_CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in fieldnames})

    log_text = "\n".join(
        [
            f"status={status}",
            f"candidateCount={candidate_count}",
            f"placementCount={placement_count}",
            f"assetTimeModeCounts={{'full_clip': {full_clip_count}, 'event_local': {event_local_count}}}",
            f"eventLocalPlacementRequiredCount={event_local_required_count}",
            f"misplacedFromZeroCount={misplaced_from_zero_count}",
            f"naiveZeroWouldMisplaceCount={naive_zero_would_misplace_count}",
            f"blockers={status_blockers}",
            f"manifest={OUT_MANIFEST_PATH}",
            f"tableJson={OUT_TABLE_JSON_PATH}",
            f"tableCsv={OUT_TABLE_CSV_PATH}",
        ]
    )
    OUT_LOG_PATH.write_text(log_text + "\n", encoding="utf-8")

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())