#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]

EVENT_JSONL = ROOT / "artifacts/manifests/week12_event_timeline.jsonl"
EVENT_CSV = ROOT / "artifacts/manifests/week12_event_timeline.csv"
V0_QUEUE = ROOT / "artifacts/evals/week12_audio_candidate_timing_bound_queue_v0.json"
V0_REPORT = ROOT / "artifacts/manifests/week12_audio_candidate_timing_binding_report_v0.json"

OUT_PROBE = ROOT / "artifacts/manifests/week12_event_timeline_schema_probe_v2.json"
OUT_QUEUE_JSON = ROOT / "artifacts/evals/week12_audio_candidate_timing_bound_queue_v2.json"
OUT_QUEUE_CSV = ROOT / "artifacts/evals/week12_audio_candidate_timing_bound_queue_v2.csv"
OUT_REPORT = ROOT / "artifacts/manifests/week12_audio_candidate_timing_binding_report_v2.json"


SCENE_KEYS = [
    "sourceSeedId", "source_seed_id", "seedId", "seed_id",
    "sceneId", "scene_id", "scene", "sceneName", "scene_name",
    "caseId", "case_id", "case", "blueprintId", "blueprint_id",
]
CASE_KEYS = ["sourceSeedId", "source_seed_id", "seedId", "seed_id", "caseId", "case_id", "case", "blueprintId", "blueprint_id", "sampleId", "sample_id"]
EVENT_ID_KEYS = ["eventId", "event_id", "soundEventId", "sound_event_id", "id", "eventKey", "event_key"]
EVENT_INDEX_KEYS = ["eventIndex", "event_index", "index", "idx", "order", "sequence", "seq"]
LAYER_KEYS = ["layer", "layerType", "layer_type", "soundLayer", "sound_layer", "track", "category", "type", "role"]
LABEL_KEYS = [
    "label", "eventLabel", "event_label", "name", "eventName", "event_name",
    "sound", "soundEvent", "sound_event", "description", "prompt", "text",
]
START_KEYS = [
    "expectedStartSec", "expected_start_sec", "startSec", "start_sec",
    "startSecond", "start_second", "start", "beginSec", "begin_sec",
    "begin", "tStart", "t_start", "startSeconds", "start_seconds",
]
END_KEYS = [
    "expectedEndSec", "expected_end_sec", "endSec", "end_sec",
    "endSecond", "end_second", "end", "stopSec", "stop_sec",
    "stop", "tEnd", "t_end", "endSeconds", "end_seconds",
]
DURATION_KEYS = [
    "expectedDurationSec", "expected_duration_sec", "durationSec", "duration_sec",
    "audioDurationSec", "audio_duration_sec", "duration", "lengthSec", "length_sec", "durationSeconds", "duration_seconds",
]
CANDIDATE_ID_KEYS = [
    "candidateId", "candidate_id", "audioCandidateId", "audio_candidate_id",
    "id", "artifactId", "artifact_id",
]
AUDIO_URI_KEYS = [
    "audioUri", "audio_uri", "audioPath", "audio_path", "wavPath", "wav_path",
    "path", "uri", "artifactUri", "artifact_uri", "candidateUri", "candidate_uri",
]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def norm_key(k: Any) -> str:
    return str(k).strip().lower().replace("-", "_")


def norm_text(v: Any) -> str:
    if v is None:
        return ""
    s = str(v).strip().lower()
    s = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def compact_event_id(v: Any) -> Any:
    """Normalize long event ids to the candidate-level event id when possible."""
    if v is None:
        return None
    s = str(v).strip()
    m = re.search(r"(evt[_-]?\d+)$", s, flags=re.IGNORECASE)
    if m:
        return m.group(1).replace("-", "_").lower()
    return s


def as_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if not s:
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def as_int(v: Any) -> Optional[int]:
    f = as_float(v)
    if f is None:
        return None
    return int(f)


def scalar(v: Any) -> bool:
    return v is None or isinstance(v, (str, int, float, bool))


def first_direct(d: Dict[str, Any], aliases: Iterable[str]) -> Any:
    if not isinstance(d, dict):
        return None
    alias_norm = {norm_key(a) for a in aliases}
    for k, v in d.items():
        if norm_key(k) in alias_norm and scalar(v):
            return v
    return None


def first_nested_timing(d: Dict[str, Any], aliases: Iterable[str]) -> Any:
    alias_norm = {norm_key(a) for a in aliases}
    for tk in ["timing", "time", "window", "expectedTiming", "expected_timing", "span"]:
        obj = d.get(tk)
        if isinstance(obj, dict):
            for k, v in obj.items():
                if norm_key(k) in alias_norm and scalar(v):
                    return v
    return None


def first_value(d: Dict[str, Any], aliases: Iterable[str]) -> Any:
    v = first_direct(d, aliases)
    if v is not None:
        return v
    return first_nested_timing(d, aliases)


def direct_key_set(d: Dict[str, Any]) -> set:
    return {norm_key(k) for k in d.keys()}


def update_context_from_dict(ctx: Dict[str, Any], d: Dict[str, Any]) -> Dict[str, Any]:
    new = dict(ctx)
    scene = first_direct(d, SCENE_KEYS)
    case = first_direct(d, CASE_KEYS)
    if scene is not None and "sceneId" not in new:
        new["sceneId"] = scene
    if case is not None and "caseId" not in new:
        new["caseId"] = case
    return new


def looks_like_event(d: Dict[str, Any]) -> bool:
    if not isinstance(d, dict):
        return False
    keys = direct_key_set(d)
    eventish = {norm_key(x) for x in EVENT_ID_KEYS + EVENT_INDEX_KEYS + LABEL_KEYS + LAYER_KEYS}
    timingish = {norm_key(x) for x in START_KEYS + END_KEYS + DURATION_KEYS}
    if keys & eventish and (keys & timingish):
        return True
    if keys & eventish and isinstance(d.get("timing"), dict):
        return True
    if keys & timingish and ("events" not in keys and "timeline" not in keys):
        return True
    return False


def load_json_or_jsonl(path: Path) -> Any:
    if not path.exists():
        return None
    if path.suffix.lower() == ".jsonl":
        rows = []
        for i, line in enumerate(read_text(path).splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise RuntimeError(f"{path}:{i} JSONL parse failed: {e}") from e
        return rows
    return json.loads(read_text(path))


def normalize_event(d: Dict[str, Any], ctx: Dict[str, Any], source: str, order: int) -> Dict[str, Any]:
    source_seed_id = first_value(d, ["sourceSeedId", "source_seed_id", "seedId", "seed_id"])
    scene_id = source_seed_id or first_value(d, SCENE_KEYS) or ctx.get("sceneId") or ctx.get("caseId")
    case_id = source_seed_id or first_value(d, CASE_KEYS) or ctx.get("caseId") or scene_id
    event_id = compact_event_id(first_value(d, EVENT_ID_KEYS))
    event_index = first_value(d, EVENT_INDEX_KEYS)
    layer = first_value(d, LAYER_KEYS)
    label = first_value(d, LABEL_KEYS)

    start = as_float(first_value(d, START_KEYS))
    end = as_float(first_value(d, END_KEYS))
    duration = as_float(first_value(d, DURATION_KEYS))

    if start is not None and end is None and duration is not None:
        end = start + duration
    if duration is None and start is not None and end is not None:
        duration = max(0.0, end - start)

    return {
        "eventRowId": f"event_{order:04d}",
        "source": source,
        "sceneId": str(scene_id) if scene_id is not None else None,
        "caseId": str(case_id) if case_id is not None else None,
        "eventId": str(event_id) if event_id is not None else None,
        "eventIndex": as_int(event_index),
        "layer": str(layer) if layer is not None else None,
        "label": str(label) if label is not None else None,
        "normalizedLabel": norm_text(label),
        "expectedStartSec": start,
        "expectedEndSec": end,
        "expectedDurationSec": duration,
        "hasTiming": start is not None and end is not None,
        "rawKeys": sorted(str(k) for k in d.keys()),
    }


def walk_events(obj: Any, ctx: Dict[str, Any], source: str, rows: List[Dict[str, Any]]) -> None:
    if isinstance(obj, list):
        for item in obj:
            walk_events(item, ctx, source, rows)
        return

    if not isinstance(obj, dict):
        return

    new_ctx = update_context_from_dict(ctx, obj)

    if looks_like_event(obj):
        rows.append(normalize_event(obj, new_ctx, source, len(rows)))

    for k, v in obj.items():
        if isinstance(v, (dict, list)):
            walk_events(v, new_ctx, source, rows)


def read_csv_events(path: Path, start_order: int) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(normalize_event(row, {}, str(path.relative_to(ROOT)), start_order + len(rows)))
    return rows


def dedupe_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for e in events:
        sig = (
            e.get("sceneId"),
            e.get("caseId"),
            e.get("eventId"),
            e.get("eventIndex"),
            e.get("layer"),
            e.get("normalizedLabel"),
            e.get("expectedStartSec"),
            e.get("expectedEndSec"),
        )
        if sig in seen:
            continue
        seen.add(sig)
        e = dict(e)
        e["eventRowId"] = f"event_{len(out):04d}"
        out.append(e)
    return out


def extract_events() -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    obj = load_json_or_jsonl(EVENT_JSONL)
    if obj is not None:
        walk_events(obj, {}, str(EVENT_JSONL.relative_to(ROOT)), events)
    events.extend(read_csv_events(EVENT_CSV, len(events)))
    return dedupe_events(events)


def find_lists_by_key(obj: Any, keys: Iterable[str]) -> List[List[Any]]:
    target = {norm_key(k) for k in keys}
    found = []

    def walk(x: Any) -> None:
        if isinstance(x, dict):
            for k, v in x.items():
                if norm_key(k) in target and isinstance(v, list):
                    found.append(v)
                if isinstance(v, (dict, list)):
                    walk(v)
        elif isinstance(x, list):
            for item in x:
                walk(item)

    walk(obj)
    return found


def looks_like_candidate(d: Dict[str, Any]) -> bool:
    if not isinstance(d, dict):
        return False
    keys = direct_key_set(d)
    summary_keys = {"candidatecount", "timingboundcount", "timingunmatchedcount", "blockers"}
    if keys & summary_keys and not (keys & {norm_key(x) for x in CANDIDATE_ID_KEYS + AUDIO_URI_KEYS}):
        return False

    for k in d.keys():
        lk = norm_key(k)
        if "candidate" in lk or "audio" in lk or "wav" in lk or "artifacturi" in lk or "artifact_uri" in lk:
            return True
    return False


def walk_candidates(obj: Any, rows: List[Dict[str, Any]]) -> None:
    if isinstance(obj, list):
        for item in obj:
            walk_candidates(item, rows)
        return
    if not isinstance(obj, dict):
        return
    if looks_like_candidate(obj):
        rows.append(obj)
    for v in obj.values():
        if isinstance(v, (dict, list)):
            walk_candidates(v, rows)


def normalize_candidate(d: Dict[str, Any], order: int) -> Dict[str, Any]:
    scene_id = first_value(d, SCENE_KEYS)
    case_id = first_value(d, CASE_KEYS)
    event_id = compact_event_id(first_value(d, EVENT_ID_KEYS))
    event_index = first_value(d, EVENT_INDEX_KEYS)
    layer = first_value(d, LAYER_KEYS)
    label = first_value(d, LABEL_KEYS)
    candidate_id = first_value(d, CANDIDATE_ID_KEYS) or f"candidate_{order:04d}"
    audio_uri = first_value(d, AUDIO_URI_KEYS)
    duration = as_float(first_value(d, DURATION_KEYS))

    return {
        "candidateRowId": f"candidate_{order:04d}",
        "candidateId": str(candidate_id) if candidate_id is not None else f"candidate_{order:04d}",
        "audioUri": str(audio_uri) if audio_uri is not None else None,
        "sceneId": str(scene_id) if scene_id is not None else None,
        "caseId": str(case_id) if case_id is not None else None,
        "eventId": str(event_id) if event_id is not None else None,
        "eventIndex": as_int(event_index),
        "layer": str(layer) if layer is not None else None,
        "label": str(label) if label is not None else None,
        "normalizedLabel": norm_text(label),
        "candidateDurationSec": duration,
        "raw": d,
    }


def extract_candidates() -> List[Dict[str, Any]]:
    if not V0_QUEUE.exists():
        raise FileNotFoundError(f"missing v0 queue: {V0_QUEUE}")

    obj = json.loads(read_text(V0_QUEUE))

    preferred_lists = find_lists_by_key(
        obj,
        ["candidates", "queue", "items", "records", "audioCandidates", "audio_candidates", "reviewQueue", "review_queue"],
    )

    raw: List[Dict[str, Any]] = []
    for lst in preferred_lists:
        for item in lst:
            if isinstance(item, dict) and looks_like_candidate(item):
                raw.append(item)

    if not raw:
        walk_candidates(obj, raw)

    seen = set()
    out: List[Dict[str, Any]] = []
    for d in raw:
        sig = json.dumps(d, ensure_ascii=False, sort_keys=True)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(normalize_candidate(d, len(out)))

    return out


def event_match_keys(e: Dict[str, Any]) -> List[Tuple[str, str]]:
    keys = []
    scene = norm_text(e.get("sceneId"))
    case = norm_text(e.get("caseId"))
    eid = norm_text(e.get("eventId"))
    eidx = e.get("eventIndex")
    layer = norm_text(e.get("layer"))
    label = norm_text(e.get("label"))

    if scene and eid:
        keys.append(("scene_event_id", f"{scene}|{eid}"))
    if case and eid:
        keys.append(("case_event_id", f"{case}|{eid}"))
    if scene and eidx is not None:
        keys.append(("scene_event_index", f"{scene}|{eidx}"))
    if case and eidx is not None:
        keys.append(("case_event_index", f"{case}|{eidx}"))
    if scene and layer and label:
        keys.append(("scene_layer_label", f"{scene}|{layer}|{label}"))
    if case and layer and label:
        keys.append(("case_layer_label", f"{case}|{layer}|{label}"))
    if eid:
        keys.append(("global_event_id_unique", eid))
    if layer and label:
        keys.append(("global_layer_label_unique", f"{layer}|{label}"))
    if label:
        keys.append(("global_label_unique", label))
    return keys


def candidate_match_keys(c: Dict[str, Any]) -> List[Tuple[str, str]]:
    keys = []
    scene = norm_text(c.get("sceneId"))
    case = norm_text(c.get("caseId"))
    eid = norm_text(c.get("eventId"))
    eidx = c.get("eventIndex")
    layer = norm_text(c.get("layer"))
    label = norm_text(c.get("label"))

    if scene and eid:
        keys.append(("scene_event_id", f"{scene}|{eid}"))
    if case and eid:
        keys.append(("case_event_id", f"{case}|{eid}"))
    if scene and eidx is not None:
        keys.append(("scene_event_index", f"{scene}|{eidx}"))
    if case and eidx is not None:
        keys.append(("case_event_index", f"{case}|{eidx}"))
    if scene and layer and label:
        keys.append(("scene_layer_label", f"{scene}|{layer}|{label}"))
    if case and layer and label:
        keys.append(("case_layer_label", f"{case}|{layer}|{label}"))
    if eid:
        keys.append(("global_event_id_unique", eid))
    if layer and label:
        keys.append(("global_layer_label_unique", f"{layer}|{label}"))
    if label:
        keys.append(("global_label_unique", label))
    return keys


def build_unique_event_index(events: List[Dict[str, Any]]) -> Dict[Tuple[str, str], Optional[Dict[str, Any]]]:
    idx: Dict[Tuple[str, str], Optional[Dict[str, Any]]] = {}
    counts: Counter = Counter()

    for e in events:
        if not e.get("hasTiming"):
            continue
        for key in event_match_keys(e):
            counts[key] += 1
            if key not in idx:
                idx[key] = e
            else:
                idx[key] = None

    for key, count in counts.items():
        if count != 1:
            idx[key] = None
    return idx


def bind_candidates(candidates: List[Dict[str, Any]], events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    timed_events = [e for e in events if e.get("hasTiming")]
    event_idx = build_unique_event_index(timed_events)
    used_event_ids = set()
    bound = []

    for c in candidates:
        matched = None
        method = None
        confidence = None

        for key in candidate_match_keys(c):
            e = event_idx.get(key)
            if e is not None and e["eventRowId"] not in used_event_ids:
                matched = e
                method = key[0]
                confidence = "HIGH" if key[0] in {"scene_event_id", "case_event_id", "scene_event_index", "case_event_index"} else "MEDIUM"
                break

        record = dict(c)
        if matched is not None:
            used_event_ids.add(matched["eventRowId"])
            record.update({
                "timingBindingStatus": "BOUND",
                "bindingMethod": method,
                "bindingConfidence": confidence,
                "boundEventRowId": matched["eventRowId"],
                "expectedStartSec": matched["expectedStartSec"],
                "expectedEndSec": matched["expectedEndSec"],
                "expectedDurationSec": matched["expectedDurationSec"],
                "expectedLayer": matched.get("layer"),
                "expectedLabel": matched.get("label"),
                "expectedSceneId": matched.get("sceneId"),
                "expectedCaseId": matched.get("caseId"),
                "expectedEventId": matched.get("eventId"),
                "expectedEventIndex": matched.get("eventIndex"),
            })
        else:
            record.update({
                "timingBindingStatus": "UNMATCHED",
                "bindingMethod": None,
                "bindingConfidence": None,
                "boundEventRowId": None,
                "expectedStartSec": None,
                "expectedEndSec": None,
                "expectedDurationSec": None,
                "expectedLayer": None,
                "expectedLabel": None,
                "expectedSceneId": None,
                "expectedCaseId": None,
                "expectedEventId": None,
                "expectedEventIndex": None,
            })
        bound.append(record)

    # Controlled fallback: if explicit keys fail but the event table and candidate queue are aligned 1:1,
    # bind by deterministic order. This is not temporal alignment; it only attaches expected windows.
    unmatched_records = [r for r in bound if r["timingBindingStatus"] == "UNMATCHED"]
    unused_events = [e for e in timed_events if e["eventRowId"] not in used_event_ids]
    if unmatched_records and len(unmatched_records) == len(unused_events):
        for record, event in zip(unmatched_records, unused_events):
            record.update({
                "timingBindingStatus": "BOUND",
                "bindingMethod": "index_order_fallback",
                "bindingConfidence": "LOW",
                "boundEventRowId": event["eventRowId"],
                "expectedStartSec": event["expectedStartSec"],
                "expectedEndSec": event["expectedEndSec"],
                "expectedDurationSec": event["expectedDurationSec"],
                "expectedLayer": event.get("layer"),
                "expectedLabel": event.get("label"),
                "expectedSceneId": event.get("sceneId"),
                "expectedCaseId": event.get("caseId"),
                "expectedEventId": event.get("eventId"),
                "expectedEventIndex": event.get("eventIndex"),
            })

    return bound


def write_queue_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "candidateRowId", "candidateId", "audioUri",
        "sceneId", "caseId", "eventId", "eventIndex", "layer", "label",
        "candidateDurationSec",
        "timingBindingStatus", "bindingMethod", "bindingConfidence",
        "expectedSceneId", "expectedCaseId", "expectedEventId", "expectedEventIndex",
        "expectedLayer", "expectedLabel",
        "expectedStartSec", "expectedEndSec", "expectedDurationSec",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_schema(events: List[Dict[str, Any]], candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    event_key_counter = Counter()
    for e in events:
        for k in e.get("rawKeys", []):
            event_key_counter[k] += 1

    return {
        "inputs": {
            "eventJsonl": str(EVENT_JSONL.relative_to(ROOT)),
            "eventJsonlExists": EVENT_JSONL.exists(),
            "eventCsv": str(EVENT_CSV.relative_to(ROOT)),
            "eventCsvExists": EVENT_CSV.exists(),
            "v0Queue": str(V0_QUEUE.relative_to(ROOT)),
            "v0QueueExists": V0_QUEUE.exists(),
            "v0Report": str(V0_REPORT.relative_to(ROOT)),
            "v0ReportExists": V0_REPORT.exists(),
        },
        "expandedEventCount": len(events),
        "eventWithTimingCount": sum(1 for e in events if e.get("hasTiming")),
        "candidateExtractedCount": len(candidates),
        "eventRawKeyTop30": event_key_counter.most_common(30),
        "eventSamples": events[:5],
        "candidateSamples": [
            {k: v for k, v in c.items() if k != "raw"}
            for c in candidates[:5]
        ],
    }


def main() -> int:
    missing = [p for p in [EVENT_JSONL, EVENT_CSV, V0_QUEUE] if not p.exists()]
    if missing:
        print("[FATAL] Missing required input files:")
        for p in missing:
            print(f"  - {p.relative_to(ROOT)}")
        return 2

    events = extract_events()
    candidates = extract_candidates()
    probe = summarize_schema(events, candidates)
    write_json(OUT_PROBE, probe)

    bound_rows = bind_candidates(candidates, events)
    write_json(OUT_QUEUE_JSON, {"status": "GENERATED", "items": bound_rows})
    write_queue_csv(OUT_QUEUE_CSV, bound_rows)

    timing_bound = sum(1 for r in bound_rows if r.get("timingBindingStatus") == "BOUND")
    timing_unmatched = len(bound_rows) - timing_bound
    method_counts = Counter(r.get("bindingMethod") or "UNMATCHED" for r in bound_rows)
    confidence_counts = Counter(r.get("bindingConfidence") or "UNMATCHED" for r in bound_rows)

    blockers = []
    warnings = []

    if len(events) == 0:
        blockers.append("NO_EXPANDED_EVENTS")
    if sum(1 for e in events if e.get("hasTiming")) == 0:
        blockers.append("NO_EVENTS_WITH_TIMING")
    if len(candidates) == 0:
        blockers.append("NO_CANDIDATES_EXTRACTED")
    if timing_bound == 0:
        blockers.append("NO_TIMING_BINDINGS")
    if timing_unmatched > 0:
        blockers.append("TIMING_BINDING_INCOMPLETE")
    if method_counts.get("index_order_fallback", 0) > 0:
        warnings.append("INDEX_ORDER_FALLBACK_USED_REQUIRES_ID_STABILIZATION")

    if len(candidates) > 0 and timing_bound == len(candidates):
        status = "PASS"
    elif timing_bound > 0:
        status = "PARTIAL"
    else:
        status = "FAIL"

    report = {
        "status": status,
        "candidateCount": len(candidates),
        "expandedEventCount": len(events),
        "eventWithTimingCount": sum(1 for e in events if e.get("hasTiming")),
        "timingBoundCount": timing_bound,
        "timingUnmatchedCount": timing_unmatched,
        "durationMatchesExpectedUnknownCount": sum(
            1 for r in bound_rows
            if r.get("expectedDurationSec") is None
        ),
        "bindingMethodCounts": dict(method_counts),
        "bindingConfidenceCounts": dict(confidence_counts),
        "blockers": blockers,
        "warnings": warnings,
        "outputs": {
            "schemaProbe": str(OUT_PROBE.relative_to(ROOT)),
            "boundQueueJson": str(OUT_QUEUE_JSON.relative_to(ROOT)),
            "boundQueueCsv": str(OUT_QUEUE_CSV.relative_to(ROOT)),
            "bindingReport": str(OUT_REPORT.relative_to(ROOT)),
        },
        "boundaryStatement": (
            "PASS only means expected timing windows are bound to audio candidates. "
            "It does not mean temporal alignment, onset alignment, semantic quality, "
            "human audition, mix readiness, or production readiness."
        ),
    }
    write_json(OUT_REPORT, report)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if status in {"PASS", "PARTIAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())