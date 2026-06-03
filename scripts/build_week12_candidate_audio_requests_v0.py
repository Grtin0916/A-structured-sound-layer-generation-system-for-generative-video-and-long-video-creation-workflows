#!/usr/bin/env python3
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_short_head(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
        text=True,
    ).strip()


def git_remote(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "remote", "get-url", "origin"],
        text=True,
    ).strip()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        obj = json.loads(line)
        if not isinstance(obj, dict):
            raise ValueError(f"{path}:{line_no} is not a JSON object")
        obj["_sourceLine"] = line_no
        rows.append(obj)
    return rows


def flatten(obj: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in obj.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        out[key] = v
        if isinstance(v, dict):
            out.update(flatten(v, key))
    return out


def first_value(flat: dict[str, Any], names: list[str], default: Any = None) -> Any:
    lower_map = {k.lower(): v for k, v in flat.items()}
    for name in names:
        name_l = name.lower()
        if name_l in lower_map and lower_map[name_l] not in (None, ""):
            return lower_map[name_l]
    for key, value in flat.items():
        last = key.split(".")[-1].lower()
        if last in {n.lower() for n in names} and value not in (None, ""):
            return value
    return default


def as_float(value: Any, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def slugify(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_\-]+", "_", value.strip())
    value = re.sub(r"_+", "_", value).strip("_")
    return value.lower() or "event"


def infer_layer(flat: dict[str, Any], event_label: str) -> str:
    layer = first_value(
        flat,
        ["layer", "layerType", "soundLayer", "audioLayer", "category", "sound_category"],
        None,
    )
    if layer:
        return str(layer).lower()

    text = event_label.lower()
    if any(k in text for k in ["rain", "wind", "traffic", "street", "ambience", "ambient"]):
        return "ambience"
    if any(k in text for k in ["step", "walk", "door", "impact", "hit", "vehicle", "object"]):
        return "foley"
    if any(k in text for k in ["music", "bgm", "score"]):
        return "music"
    return "foley"


def build_prompt(event: dict[str, Any], flat: dict[str, Any], event_label: str, layer: str) -> str:
    scene = first_value(flat, ["scene", "sceneText", "sceneDescription", "scene_id", "sceneId"], "")
    object_name = first_value(flat, ["object", "objectName", "actor", "sourceObject"], "")
    action = first_value(flat, ["action", "eventAction", "motion", "verb"], "")
    description = first_value(flat, ["description", "text", "caption", "eventDescription", "eventText"], "")

    parts = [str(x).strip() for x in [scene, object_name, action, description] if str(x).strip()]
    if parts:
        base = "; ".join(parts)
    else:
        base = event_label

    return (
        f"Create or retrieve a {layer} audio candidate for event '{event_label}'. "
        f"Visual/context cues: {base}. "
        "Preserve temporal alignment and avoid over-dominating other sound layers."
    )


def main() -> int:
    mainbase = Path.home() / "work" / "audio_engineering_repo_skeleton_v1"

    timeline_jsonl = mainbase / "artifacts/manifests/week12_event_timeline.jsonl"
    feedback_index = mainbase / "artifacts/manifests/week12_blueprint_runtime_feedback_index.json"

    out_manifest = mainbase / "artifacts/manifests/week12_candidate_audio_request_manifest.json"
    out_jsonl = mainbase / "artifacts/manifests/week12_candidate_audio_requests.jsonl"
    out_csv = mainbase / "artifacts/manifests/week12_candidate_audio_requests.csv"

    for path in [timeline_jsonl, feedback_index]:
        if not path.exists():
            raise SystemExit(f"MISSING_REQUIRED_FILE={path}")

    feedback = load_json(feedback_index)
    if feedback.get("status") != "PASS":
        raise SystemExit(f"RUNTIME_FEEDBACK_NOT_PASS={feedback.get('status')}")

    events = load_jsonl(timeline_jsonl)
    if not events:
        raise SystemExit("EVENT_TIMELINE_EMPTY")

    requests: list[dict[str, Any]] = []
    for idx, event in enumerate(events, start=1):
        flat = flatten(event)

        event_id = str(first_value(flat, ["eventId", "event_id", "id"], f"event_{idx:04d}"))
        event_label = str(
            first_value(
                flat,
                ["eventLabel", "event_label", "label", "name", "event", "eventType", "description", "text"],
                event_id,
            )
        )
        scene_id = str(first_value(flat, ["sceneId", "scene_id", "scene"], "week12_scene"))
        layer = infer_layer(flat, event_label)

        start_sec = as_float(first_value(flat, ["startSec", "start_sec", "start", "startTime", "t_start"], None))
        end_sec = as_float(first_value(flat, ["endSec", "end_sec", "end", "endTime", "t_end"], None))
        duration_sec = as_float(first_value(flat, ["durationSec", "duration_sec", "duration"], None))

        if start_sec is not None and end_sec is None and duration_sec is not None:
            end_sec = start_sec + duration_sec
        if start_sec is None:
            start_sec = 0.0
        if end_sec is None:
            end_sec = start_sec + 1.0
        if end_sec < start_sec:
            end_sec = start_sec

        request_id = f"car_v0_{idx:04d}_{slugify(event_id)}"

        request = {
            "requestId": request_id,
            "status": "REQUESTED_NOT_GENERATED",
            "source": {
                "eventTimelinePath": "artifacts/manifests/week12_event_timeline.jsonl",
                "sourceLine": event.get("_sourceLine"),
                "eventId": event_id,
                "sceneId": scene_id,
                "rawEvent": {k: v for k, v in event.items() if k != "_sourceLine"},
            },
            "timing": {
                "startSec": round(float(start_sec), 3),
                "endSec": round(float(end_sec), 3),
                "durationSec": round(float(max(end_sec - start_sec, 0.0)), 3),
            },
            "audioRequest": {
                "layer": layer,
                "eventLabel": event_label,
                "prompt": build_prompt(event, flat, event_label, layer),
                "candidatePolicy": {
                    "allowRetrieval": True,
                    "allowGeneration": True,
                    "preferredMode": "retrieve_then_generate",
                    "requiresHumanAudition": True,
                },
                "qualityContract": {
                    "mustLinkBlueprint": True,
                    "mustPreserveTiming": True,
                    "mustRecordCandidateSource": True,
                    "mustRecordFailureReasonIfNoCandidate": True,
                },
            },
        }
        requests.append(request)

    out_jsonl.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in requests),
        encoding="utf-8",
    )

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "requestId",
                "eventId",
                "sceneId",
                "layer",
                "eventLabel",
                "startSec",
                "endSec",
                "durationSec",
                "status",
            ],
        )
        writer.writeheader()
        for r in requests:
            writer.writerow(
                {
                    "requestId": r["requestId"],
                    "eventId": r["source"]["eventId"],
                    "sceneId": r["source"]["sceneId"],
                    "layer": r["audioRequest"]["layer"],
                    "eventLabel": r["audioRequest"]["eventLabel"],
                    "startSec": r["timing"]["startSec"],
                    "endSec": r["timing"]["endSec"],
                    "durationSec": r["timing"]["durationSec"],
                    "status": r["status"],
                }
            )

    manifest = {
        "schemaVersion": "week12.candidate-audio-request-manifest.v0",
        "generatedAt": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "PASS",
        "requestCount": len(requests),
        "mainbase": {
            "repo": git_remote(mainbase),
            "runtimeFeedbackCommit": git_short_head(mainbase),
            "runtimeFeedbackPath": "artifacts/manifests/week12_blueprint_runtime_feedback_index.json",
            "runtimeFeedbackSha256": sha256_file(feedback_index),
            "eventTimelinePath": "artifacts/manifests/week12_event_timeline.jsonl",
            "eventTimelineSha256": sha256_file(timeline_jsonl),
        },
        "upstreamRuntimeClosure": {
            "javaCommit": feedback["javaRuntimeEvidence"]["commit"],
            "cloudCommit": feedback["cloudRuntimeEvidence"]["commit"],
            "javaRuntimeHttpCode": feedback["javaRuntimeEvidence"]["httpCode"],
            "cloudRuntimeStatus": feedback["cloudRuntimeEvidence"]["status"],
            "feedbackStatus": feedback["status"],
            "feedbackBlockers": feedback["blockers"],
        },
        "outputs": {
            "jsonl": "artifacts/manifests/week12_candidate_audio_requests.jsonl",
            "jsonlSha256": sha256_file(out_jsonl),
            "csv": "artifacts/manifests/week12_candidate_audio_requests.csv",
            "csvSha256": sha256_file(out_csv),
        },
        "requestStatusSemantics": {
            "REQUESTED_NOT_GENERATED": "Candidate request is ready, but no waveform or retrieved candidate has been produced yet."
        },
        "doesNotClaim": [
            "audio waveform generation",
            "candidate audio bank exists",
            "model inference has run",
            "human audition has completed",
            "production asset storage"
        ],
    }

    out_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())