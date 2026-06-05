#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT = Path(__file__).resolve().parents[1]

IN_CSV = ROOT / "artifacts/evals/week12_temporal_alignment_probe_v0.csv"

OUT_CSV = ROOT / "artifacts/evals/week12_temporal_alignment_probe_v1.csv"
OUT_JSON = ROOT / "artifacts/evals/week12_temporal_alignment_probe_v1.json"
OUT_REPORT = ROOT / "artifacts/manifests/week12_temporal_alignment_probe_report_v1.json"


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


def as_bool(v: Any) -> bool:
    return str(v).strip().lower() in {"true", "1", "yes", "y"}


def norm(v: Any) -> str:
    return str(v or "").strip().lower()


def close(a: Optional[float], b: Optional[float], tol: float = 0.25) -> bool:
    if a is None or b is None:
        return False
    return abs(a - b) <= tol


def round_opt(v: Optional[float], ndigits: int = 6) -> Optional[float]:
    if v is None:
        return None
    return round(float(v), ndigits)


def classify_asset_time_mode(row: Dict[str, str]) -> str:
    actual = as_float(row.get("actualDurationSec"))
    clip = as_float(row.get("expectedClipDurationSec"))
    window = as_float(row.get("expectedWindowDurationSec"))
    layer = norm(row.get("layer"))

    if close(actual, window) and not close(actual, clip):
        return "event_local"
    if close(actual, clip):
        return "full_clip"
    if layer == "foley" and window is not None and actual is not None and actual <= window + 0.5:
        return "event_local_probable"
    return "unknown"


def repair_row(row: Dict[str, str]) -> Dict[str, Any]:
    expected_start = as_float(row.get("expectedStartSec"))
    expected_end = as_float(row.get("expectedEndSec"))
    expected_clip_duration = as_float(row.get("expectedClipDurationSec"))
    expected_window_duration = as_float(row.get("expectedWindowDurationSec"))
    actual_duration = as_float(row.get("actualDurationSec"))
    peak_sec = as_float(row.get("peakSec"))
    first_active_sec = as_float(row.get("firstActiveSec"))
    last_active_sec = as_float(row.get("lastActiveSec"))

    mode = classify_asset_time_mode(row)
    layer = norm(row.get("layer"))

    if mode.startswith("event_local") and expected_start is not None:
        peak_global = expected_start + peak_sec if peak_sec is not None else None
        first_active_global = expected_start + first_active_sec if first_active_sec is not None else None
        last_active_global = expected_start + last_active_sec if last_active_sec is not None else None
        duration_error_ref = expected_window_duration
        duration_error_name = "expectedWindowDurationSec"
    else:
        peak_global = peak_sec
        first_active_global = first_active_sec
        last_active_global = last_active_sec
        duration_error_ref = expected_clip_duration
        duration_error_name = "expectedClipDurationSec"

    duration_abs_error = None
    if actual_duration is not None and duration_error_ref is not None:
        duration_abs_error = abs(actual_duration - duration_error_ref)

    def inside(v: Optional[float]) -> bool:
        return (
            v is not None
            and expected_start is not None
            and expected_end is not None
            and expected_start <= v <= expected_end
        )

    peak_inside_global = inside(peak_global)
    first_inside_global = inside(first_active_global)
    last_inside_global = inside(last_active_global)

    audio_readable = as_bool(row.get("audioReadable"))
    audio_exists = as_bool(row.get("audioExists"))

    active_inside = as_float(row.get("activeRatioInsideWindow"))
    active_outside = as_float(row.get("activeRatioOutsideWindow"))
    has_energy_signal = (
        peak_sec is not None
        or first_active_sec is not None
        or ((active_inside or 0.0) > 0.0)
    )

    decision = "FAIL"
    reason = "UNCLASSIFIED"

    if not audio_exists:
        reason = "AUDIO_MISSING"
    elif not audio_readable:
        reason = "AUDIO_NOT_READABLE"
    elif mode == "unknown":
        reason = "ASSET_TIME_MODE_UNKNOWN"
    elif duration_abs_error is not None and duration_abs_error > 0.25:
        reason = f"DURATION_MISMATCH_TO_{duration_error_name}"
    elif not has_energy_signal:
        reason = "NO_ENERGY_SIGNAL"
    elif layer == "ambience":
        if mode == "full_clip" and peak_inside_global:
            decision = "PASS"
            reason = "FULL_CLIP_AMBIENCE_ACTIVITY_INSIDE_SCENE_WINDOW"
        elif peak_inside_global or first_inside_global:
            decision = "PARTIAL"
            reason = "AMBIENCE_ACTIVITY_PRESENT_BUT_MODE_NOT_FULL_CLIP"
        else:
            reason = "AMBIENCE_ACTIVITY_NOT_IN_WINDOW"
    else:
        if mode.startswith("event_local") and peak_inside_global:
            decision = "PASS"
            reason = "EVENT_LOCAL_FOLEY_PEAK_ALIGNED_AFTER_WINDOW_OFFSET"
        elif mode.startswith("event_local") and first_inside_global:
            decision = "PARTIAL"
            reason = "EVENT_LOCAL_FOLEY_FIRST_ACTIVE_ALIGNED_AFTER_WINDOW_OFFSET"
        elif peak_inside_global:
            decision = "PARTIAL"
            reason = "FOLEY_PEAK_INSIDE_BUT_ASSET_MODE_WEAK"
        else:
            reason = "FOLEY_ACTIVITY_NOT_ALIGNED_AFTER_OFFSET"

    out = dict(row)
    out.update({
        "assetTimeMode": mode,
        "durationReference": duration_error_name,
        "durationAbsErrorToReferenceSec": round_opt(duration_abs_error),
        "peakGlobalSec": round_opt(peak_global),
        "firstActiveGlobalSec": round_opt(first_active_global),
        "lastActiveGlobalSec": round_opt(last_active_global),
        "peakInsideExpectedWindowGlobal": peak_inside_global,
        "firstActiveInsideExpectedWindowGlobal": first_inside_global,
        "lastActiveInsideExpectedWindowGlobal": last_inside_global,
        "alignmentDecisionV0": row.get("alignmentDecision"),
        "alignmentReasonV0": row.get("alignmentReason"),
        "alignmentDecision": decision,
        "alignmentReason": reason,
    })
    return out


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    base_fields = [
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
        "assetTimeMode",
        "durationReference",
        "durationAbsErrorToReferenceSec",
        "peakSec",
        "peakGlobalSec",
        "firstActiveSec",
        "firstActiveGlobalSec",
        "lastActiveSec",
        "lastActiveGlobalSec",
        "peakInsideExpectedWindowGlobal",
        "firstActiveInsideExpectedWindowGlobal",
        "lastActiveInsideExpectedWindowGlobal",
        "activeRatioInsideWindow",
        "activeRatioOutsideWindow",
        "alignmentDecision",
        "alignmentReason",
        "alignmentDecisionV0",
        "alignmentReasonV0",
    ]

    extra = []
    seen = set(base_fields)
    for row in rows:
        for k in row.keys():
            if k not in seen:
                extra.append(k)
                seen.add(k)

    fields = base_fields + extra
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    if not IN_CSV.exists():
        raise FileNotFoundError(f"missing input csv: {IN_CSV}")

    with IN_CSV.open("r", encoding="utf-8", errors="replace", newline="") as f:
        input_rows = list(csv.DictReader(f))

    rows = [repair_row(r) for r in input_rows]

    candidate_count = len(rows)
    readable_count = sum(1 for r in rows if str(r.get("audioReadable")).lower() == "true")
    pass_count = sum(1 for r in rows if r.get("alignmentDecision") == "PASS")
    partial_count = sum(1 for r in rows if r.get("alignmentDecision") == "PARTIAL")
    fail_count = sum(1 for r in rows if r.get("alignmentDecision") == "FAIL")

    foley_rows = [r for r in rows if norm(r.get("layer")) == "foley"]
    ambience_rows = [r for r in rows if norm(r.get("layer")) == "ambience"]

    foley_pass = sum(1 for r in foley_rows if r.get("alignmentDecision") == "PASS")
    foley_partial = sum(1 for r in foley_rows if r.get("alignmentDecision") == "PARTIAL")
    foley_fail = sum(1 for r in foley_rows if r.get("alignmentDecision") == "FAIL")

    ambience_pass = sum(1 for r in ambience_rows if r.get("alignmentDecision") == "PASS")
    ambience_partial = sum(1 for r in ambience_rows if r.get("alignmentDecision") == "PARTIAL")
    ambience_fail = sum(1 for r in ambience_rows if r.get("alignmentDecision") == "FAIL")

    mode_counts = Counter(r.get("assetTimeMode") for r in rows)
    decision_counts = Counter(r.get("alignmentDecision") for r in rows)
    reason_counts = Counter(r.get("alignmentReason") for r in rows)
    layer_decision_counts = Counter(f"{r.get('layer')}:{r.get('alignmentDecision')}" for r in rows)

    blockers = []
    warnings = []

    if candidate_count == 0:
        blockers.append("NO_CANDIDATES")
    if readable_count < candidate_count:
        blockers.append("AUDIO_READ_INCOMPLETE")
    if fail_count > 0:
        blockers.append("TEMPORAL_ALIGNMENT_FAILURES_REMAIN")
    if foley_rows and foley_fail > 0:
        blockers.append("FOLEY_EVENTS_NOT_ALIGNED_AFTER_COORDINATE_REPAIR")
    if mode_counts.get("unknown", 0) > 0:
        warnings.append("ASSET_TIME_MODE_UNKNOWN_EXISTS")
    if mode_counts.get("event_local", 0) > 0:
        warnings.append("EVENT_LOCAL_ASSETS_REQUIRE_PLACEMENT_OFFSET_IN_MIXER")

    if blockers:
        status = "FAIL"
    elif partial_count > 0:
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
        "foleyCount": len(foley_rows),
        "foleyPassCount": foley_pass,
        "foleyPartialCount": foley_partial,
        "foleyFailCount": foley_fail,
        "ambienceCount": len(ambience_rows),
        "ambiencePassCount": ambience_pass,
        "ambiencePartialCount": ambience_partial,
        "ambienceFailCount": ambience_fail,
        "assetTimeModeCounts": dict(mode_counts),
        "alignmentDecisionCounts": dict(decision_counts),
        "alignmentReasonCounts": dict(reason_counts),
        "layerDecisionCounts": dict(layer_decision_counts),
        "blockers": blockers,
        "warnings": warnings,
        "outputs": {
            "alignmentCsv": str(OUT_CSV.relative_to(ROOT)),
            "alignmentJson": str(OUT_JSON.relative_to(ROOT)),
            "alignmentReport": str(OUT_REPORT.relative_to(ROOT)),
        },
        "boundaryStatement": (
            "This report repairs coordinate-frame interpretation for full-clip ambience and event-local foley assets. "
            "PASS means RMS/onset proxy is compatible with expected timing placement. "
            "It does not mean semantic quality, human audition, mix readiness, or production readiness."
        ),
    }

    write_csv(OUT_CSV, rows)
    OUT_JSON.write_text(json.dumps({"items": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if status in {"PASS", "PARTIAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())