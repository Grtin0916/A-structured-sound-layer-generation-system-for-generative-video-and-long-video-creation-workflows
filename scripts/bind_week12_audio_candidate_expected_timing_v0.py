#!/usr/bin/env python3
"""
Bind expected event timing windows to Week12 audio candidate review queue.

Purpose:
- Consume existing event timeline and enriched audio audition review queue.
- Produce a timing-bound candidate queue.
- Compute durationDeltaSec and durationMatchesExpected.
- Do NOT claim temporal alignment passed, semantic quality passed, human audition passed, or final mix readiness.

Inputs:
- artifacts/evals/week12_audio_audition_review_queue_v0.json
- artifacts/manifests/week12_event_timeline.jsonl or .csv

Outputs:
- artifacts/evals/week12_audio_candidate_timing_bound_queue_v0.json
- artifacts/evals/week12_audio_candidate_timing_bound_queue_v0.csv
- artifacts/manifests/week12_audio_candidate_timing_binding_report_v0.json
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


QUEUE_JSON = Path("artifacts/evals/week12_audio_audition_review_queue_v0.json")
TIMELINE_JSONL = Path("artifacts/manifests/week12_event_timeline.jsonl")
TIMELINE_CSV = Path("artifacts/manifests/week12_event_timeline.csv")

OUT_JSON = Path("artifacts/evals/week12_audio_candidate_timing_bound_queue_v0.json")
OUT_CSV = Path("artifacts/evals/week12_audio_candidate_timing_bound_queue_v0.csv")
OUT_REPORT = Path("artifacts/manifests/week12_audio_candidate_timing_binding_report_v0.json")

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
    "timingBindingStatus",
    "timingBindingMethod",
    "failureTags",
    "reviewNote",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"ERROR: missing required JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"ERROR: invalid JSONL {path}:{lineno}: {exc}") from exc
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def load_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def pick(d: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in d and d[key] not in (None, ""):
            return d[key]
    return default


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


def normalize_text(x: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(x or "").lower()).strip()


def label_overlap(a: Any, b: Any) -> float:
    sa = set(normalize_text(a).split())
    sb = set(normalize_text(b).split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(len(sa | sb), 1)


def parse_evt_index(event_id: Any) -> Optional[int]:
    m = re.search(r"evt[_-]?(\d+)", str(event_id or ""))
    if not m:
        return None
    return int(m.group(1))


def infer_case_id(row: Dict[str, Any]) -> str:
    for key in ["caseId", "case_id", "sceneId", "scene_id", "seedCaseId", "seed_case_id"]:
        if row.get(key):
            return str(row[key])
    src = str(row.get("sourceRequestId") or row.get("requestId") or "")
    m = re.search(r"(seed_\d+_case_[A-Za-z0-9]+)", src)
    return m.group(1) if m else ""


def normalize_timeline_row(row: Dict[str, Any], row_index: int) -> Dict[str, Any]:
    scene_id = str(pick(
        row,
        "sceneId", "scene_id", "caseId", "case_id", "seedCaseId", "seed_case_id",
        default=infer_case_id(row),
    ) or "")

    event_id = str(pick(
        row,
        "eventId", "event_id", "id", "timelineEventId", "timeline_event_id",
        default="",
    ) or "")

    if not event_id:
        event_idx = pick(row, "eventIndex", "event_index", "index", default=None)
        if event_idx not in (None, ""):
            try:
                event_id = f"evt_{int(float(event_idx)):03d}"
            except ValueError:
                event_id = str(event_idx)

    event_label = str(pick(
        row,
        "eventLabel", "event_label", "label", "soundEvent", "sound_event",
        "description", "caption", "event",
        default="",
    ) or "")

    layer = str(pick(row, "layer", "audioLayer", "audio_layer", "type", default="") or "")

    start = to_float(pick(
        row,
        "expectedStartSec", "startSec", "start_sec", "start", "beginSec", "begin_sec", "t0", "onsetSec", "onset_sec",
    ))
    end = to_float(pick(
        row,
        "expectedEndSec", "endSec", "end_sec", "end", "finishSec", "finish_sec", "t1", "offsetSec", "offset_sec",
    ))
    duration = to_float(pick(
        row,
        "expectedDurationSec", "durationSec", "duration_sec", "duration", "eventDurationSec", "event_duration_sec",
    ))

    if duration is None and start is not None and end is not None and end >= start:
        duration = round(end - start, 6)

    if end is None and start is not None and duration is not None:
        end = round(start + duration, 6)

    if start is None and end is not None and duration is not None:
        start = round(end - duration, 6)

    return {
        "rowIndex": row_index,
        "sceneId": scene_id,
        "eventId": event_id,
        "eventLabel": event_label,
        "layer": layer,
        "expectedStartSec": start,
        "expectedEndSec": end,
        "expectedDurationSec": duration,
        "raw": row,
    }


def load_timeline() -> Tuple[List[Dict[str, Any]], str]:
    rows = load_jsonl(TIMELINE_JSONL)
    source = str(TIMELINE_JSONL)
    if not rows:
        rows = load_csv(TIMELINE_CSV)
        source = str(TIMELINE_CSV)
    if not rows:
        raise SystemExit(f"ERROR: no timeline rows found: {TIMELINE_JSONL} or {TIMELINE_CSV}")
    return [normalize_timeline_row(row, i + 1) for i, row in enumerate(rows)], source


def duration_match(actual: Optional[float], expected: Optional[float]) -> Optional[bool]:
    if actual is None or expected is None:
        return None
    tolerance = max(0.25, expected * 0.10)
    return abs(actual - expected) <= tolerance


def update_failure_tags(tags: str, bound: bool, duration_ok: Optional[bool]) -> str:
    parts = [t.strip() for t in str(tags or "").split("|") if t.strip()]
    s = set(parts)

    s.add("human_audition_required")
    s.add("semantic_unverified")

    if bound:
        s.discard("expected_timing_unverified")
        s.add("expected_timing_bound")
    else:
        s.add("expected_timing_unverified")

    if duration_ok is True:
        s.discard("duration_mismatch")
        s.discard("duration_unverified")
    elif duration_ok is False:
        s.add("duration_mismatch")
    else:
        s.add("duration_unverified")

    ordered = [
        "human_audition_required",
        "semantic_unverified",
        "expected_timing_bound",
        "expected_timing_unverified",
        "duration_mismatch",
        "duration_unverified",
    ]
    rest = sorted(t for t in s if t not in ordered)
    return "|".join([t for t in ordered if t in s] + rest)


def build_indexes(timeline: List[Dict[str, Any]]) -> Tuple[Dict[Tuple[str, str], Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    direct = {}
    by_scene = defaultdict(list)

    for row in timeline:
        scene = row.get("sceneId") or ""
        event = row.get("eventId") or ""
        if scene:
            by_scene[scene].append(row)
        if scene and event:
            direct[(scene, event)] = row

    for scene, rows in by_scene.items():
        rows.sort(key=lambda r: (
            r.get("expectedStartSec") if r.get("expectedStartSec") is not None else 999999.0,
            r.get("rowIndex", 999999),
        ))

    return direct, by_scene


def match_candidate(candidate: Dict[str, Any], direct: Dict[Tuple[str, str], Dict[str, Any]], by_scene: Dict[str, List[Dict[str, Any]]]) -> Tuple[Optional[Dict[str, Any]], str]:
    scene = str(candidate.get("sceneId") or candidate.get("caseId") or "")
    event = str(candidate.get("eventId") or "")

    if scene and event and (scene, event) in direct:
        return direct[(scene, event)], "scene_event_id"

    rows = by_scene.get(scene) or []
    if rows:
        idx = parse_evt_index(event)
        if idx is not None and 1 <= idx <= len(rows):
            return rows[idx - 1], "scene_event_index"

        candidate_layer = normalize_text(candidate.get("layer"))
        candidate_label = candidate.get("eventLabel")
        scored = []
        for r in rows:
            layer_score = 1.0 if candidate_layer and candidate_layer == normalize_text(r.get("layer")) else 0.0
            text_score = label_overlap(candidate_label, r.get("eventLabel"))
            score = layer_score + text_score
            scored.append((score, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        if scored and scored[0][0] > 0:
            return scored[0][1], "scene_layer_label_similarity"

    return None, "unmatched"


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in FIELDNAMES})


def main() -> int:
    queue = load_json(QUEUE_JSON)
    review_rows = queue.get("reviewQueue") or []
    if not isinstance(review_rows, list) or not review_rows:
        raise SystemExit("ERROR: reviewQueue is empty")

    timeline, timeline_source = load_timeline()
    direct, by_scene = build_indexes(timeline)

    bound_rows = []
    binding_records = []

    for candidate in review_rows:
        row = dict(candidate)
        match, method = match_candidate(row, direct, by_scene)

        if match:
            row["expectedStartSec"] = match.get("expectedStartSec")
            row["expectedEndSec"] = match.get("expectedEndSec")
            row["expectedDurationSec"] = match.get("expectedDurationSec")
            row["timingBindingStatus"] = "BOUND"
            row["timingBindingMethod"] = method

            duration = to_float(row.get("durationSec"))
            expected = to_float(row.get("expectedDurationSec"))
            ok = duration_match(duration, expected)

            if duration is not None and expected is not None:
                row["durationDeltaSec"] = round(duration - expected, 6)
                row["durationMatchesExpected"] = ok
            else:
                row["durationDeltaSec"] = None
                row["durationMatchesExpected"] = None

            row["failureTags"] = update_failure_tags(row.get("failureTags", ""), True, ok)
            row["reviewNote"] = (
                "Timing window bound from Week12 event timeline. "
                "Still requires human audition and separate temporal alignment evaluation."
            )
        else:
            row["timingBindingStatus"] = "UNMATCHED"
            row["timingBindingMethod"] = method
            row["failureTags"] = update_failure_tags(row.get("failureTags", ""), False, None)
            row["reviewNote"] = (
                "No expected timing window could be matched. "
                "Candidate remains timing-unverified and requires follow-up."
            )

        binding_records.append({
            "candidateId": row.get("candidateId"),
            "sceneId": row.get("sceneId"),
            "eventId": row.get("eventId"),
            "eventLabel": row.get("eventLabel"),
            "layer": row.get("layer"),
            "timingBindingStatus": row.get("timingBindingStatus"),
            "timingBindingMethod": row.get("timingBindingMethod"),
            "expectedStartSec": row.get("expectedStartSec"),
            "expectedEndSec": row.get("expectedEndSec"),
            "expectedDurationSec": row.get("expectedDurationSec"),
            "durationSec": row.get("durationSec"),
            "durationDeltaSec": row.get("durationDeltaSec"),
            "durationMatchesExpected": row.get("durationMatchesExpected"),
        })
        bound_rows.append(row)

    candidate_count = len(bound_rows)
    bound_count = sum(1 for r in bound_rows if r.get("timingBindingStatus") == "BOUND")
    unmatched_count = candidate_count - bound_count
    duration_match_true = sum(1 for r in bound_rows if r.get("durationMatchesExpected") is True)
    duration_match_false = sum(1 for r in bound_rows if r.get("durationMatchesExpected") is False)
    duration_match_unknown = sum(1 for r in bound_rows if r.get("durationMatchesExpected") is None)

    blockers = []
    if candidate_count == 0:
        blockers.append("NO_CANDIDATES")
    if bound_count == 0:
        blockers.append("NO_TIMING_BINDINGS")
    if unmatched_count > 0:
        blockers.append("TIMING_BINDING_INCOMPLETE")

    out = {
        "schemaVersion": "week12.audio_candidate_timing_bound_queue.v0",
        "generatedAt": utc_now(),
        "status": "PASS" if candidate_count > 0 and bound_count == candidate_count else ("PARTIAL" if bound_count > 0 else "FAIL"),
        "sourceReviewQueueUri": str(QUEUE_JSON),
        "sourceTimelineUri": timeline_source,
        "candidateCount": candidate_count,
        "timingBoundCount": bound_count,
        "timingUnmatchedCount": unmatched_count,
        "durationMatchesExpectedTrueCount": duration_match_true,
        "durationMatchesExpectedFalseCount": duration_match_false,
        "durationMatchesExpectedUnknownCount": duration_match_unknown,
        "semanticFidelityClaimedAny": False,
        "mixReadyClaimedAny": False,
        "humanAuditionPassedAny": False,
        "alignmentPassedClaimedAny": False,
        "blockers": blockers,
        "doesNotClaim": [
            "temporal_alignment_passed",
            "semantic_audio_quality_passed",
            "human_audition_passed",
            "final_mix_readiness",
            "production_asset_storage",
        ],
        "reviewQueue": bound_rows,
    }

    report = {
        "schemaVersion": "week12.audio_candidate_timing_binding_report.v0",
        "generatedAt": utc_now(),
        "status": out["status"],
        "sourceReviewQueueUri": str(QUEUE_JSON),
        "sourceTimelineUri": timeline_source,
        "timelineRowCount": len(timeline),
        "candidateCount": candidate_count,
        "timingBoundCount": bound_count,
        "timingUnmatchedCount": unmatched_count,
        "durationMatchesExpectedTrueCount": duration_match_true,
        "durationMatchesExpectedFalseCount": duration_match_false,
        "durationMatchesExpectedUnknownCount": duration_match_unknown,
        "blockers": blockers,
        "bindingRecords": binding_records,
        "nextAction": {
            "preferred": "inspect duration mismatches and define temporal alignment scoring before claiming alignment",
            "avoid": [
                "do not claim temporal alignment from duration match alone",
                "do not claim semantic quality without human audition or model-based semantic eval",
            ],
        },
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    write_csv(OUT_CSV, bound_rows)

    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": out["status"],
        "candidateCount": candidate_count,
        "timingBoundCount": bound_count,
        "timingUnmatchedCount": unmatched_count,
        "durationMatchesExpectedTrueCount": duration_match_true,
        "durationMatchesExpectedFalseCount": duration_match_false,
        "durationMatchesExpectedUnknownCount": duration_match_unknown,
        "blockers": blockers,
        "json": str(OUT_JSON),
        "csv": str(OUT_CSV),
        "report": str(OUT_REPORT),
    }, ensure_ascii=False, indent=2))

    return 0 if out["status"] in {"PASS", "PARTIAL"} else 2


if __name__ == "__main__":
    raise SystemExit(main())