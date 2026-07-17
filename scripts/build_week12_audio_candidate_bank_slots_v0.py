#!/usr/bin/env python3
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def make_retrieval_query(request: dict[str, Any]) -> str:
    audio = request.get("audioRequest", {})
    timing = request.get("timing", {})
    layer = audio.get("layer", "")
    label = audio.get("eventLabel", "")
    tags = audio.get("semanticTags", "")
    prompt = audio.get("sourcePrompt", "")
    duration = timing.get("durationSec", "")
    return (
        f"layer:{layer} label:{label} tags:{tags} "
        f"duration:{duration}s cue:{prompt}"
    ).strip()


def make_generation_prompt(request: dict[str, Any]) -> str:
    audio = request.get("audioRequest", {})
    timing = request.get("timing", {})
    layer = audio.get("layer", "")
    label = audio.get("eventLabel", "")
    source_prompt = audio.get("sourcePrompt", "")
    semantic_tags = audio.get("semanticTags", "")
    priority = audio.get("priority", "")
    intensity = audio.get("intensity", "")
    start = timing.get("startSec")
    end = timing.get("endSec")
    duration = timing.get("durationSec")
    alignment = timing.get("alignmentRule", "")

    return (
        f"Generate a {duration}s {layer} sound candidate for '{label}', "
        f"aligned to [{start}, {end}] seconds. "
        f"Source cue: {source_prompt}. "
        f"Semantic tags: {semantic_tags}. "
        f"Priority={priority}, intensity={intensity}, alignment={alignment}. "
        "Keep it clean, mixable, and avoid masking higher-priority layers."
    )


def warning_is_clip_duration_only(manifest: dict[str, Any]) -> bool:
    timing = manifest.get("timingPreservation", {})
    mismatches = timing.get("timingMismatches", [])
    if manifest.get("status") == "PASS":
        return True
    if manifest.get("status") != "WARN":
        return False
    if timing.get("zeroToOneFallbackCount") != 0:
        return False
    if not mismatches:
        return True
    return all(
        item.get("reason") == "duration_field_differs_from_end_minus_start"
        and float(item.get("durationField", -1)) >= float(item.get("computedDuration", 999999))
        for item in mismatches
    )


def main() -> int:
    mainbase = Path.home() / "work" / "grt_work" / "audio_engineering_repo_skeleton_v1"

    request_manifest_path = mainbase / "artifacts/manifests/week12_candidate_audio_request_manifest_v1.json"
    request_jsonl_path = mainbase / "artifacts/manifests/week12_candidate_audio_requests_v1.jsonl"

    out_manifest = mainbase / "artifacts/manifests/week12_audio_candidate_bank_slots_manifest_v0.json"
    out_jsonl = mainbase / "artifacts/manifests/week12_audio_candidate_bank_slots_v0.jsonl"
    out_csv = mainbase / "artifacts/manifests/week12_audio_candidate_bank_slots_v0.csv"

    for path in [request_manifest_path, request_jsonl_path]:
        if not path.exists():
            raise SystemExit(f"MISSING_REQUIRED_FILE={path}")

    request_manifest = load_json(request_manifest_path)
    requests = load_jsonl(request_jsonl_path)

    if not requests:
        raise SystemExit("NO_CANDIDATE_AUDIO_REQUESTS")

    timing_warning_accepted = warning_is_clip_duration_only(request_manifest)
    if not timing_warning_accepted:
        raise SystemExit(
            "REQUEST_MANIFEST_TIMING_WARNING_NOT_ACCEPTED="
            + json.dumps(request_manifest.get("timingPreservation", {}), ensure_ascii=False)
        )

    slots: list[dict[str, Any]] = []
    for idx, request in enumerate(requests, start=1):
        request_id = request["requestId"]
        source = request.get("source", {})
        timing = request.get("timing", {})
        audio = request.get("audioRequest", {})

        common = {
            "requestId": request_id,
            "sourceEventId": source.get("eventId"),
            "sceneId": source.get("sceneId"),
            "blueprintId": source.get("blueprintId"),
            "blueprintArtifactUri": source.get("blueprintArtifactUri"),
            "layer": audio.get("layer"),
            "eventLabel": audio.get("eventLabel"),
            "priority": audio.get("priority"),
            "intensity": audio.get("intensity"),
            "timing": {
                "startSec": timing.get("startSec"),
                "endSec": timing.get("endSec"),
                "durationSec": timing.get("durationSec"),
                "alignmentRule": timing.get("alignmentRule"),
            },
            "qualityContract": audio.get("qualityContract", {}),
        }

        slots.append(
            {
                "candidateSlotId": f"slot_v0_{idx:04d}_retrieval_primary",
                "slotType": "retrieval_primary",
                "status": "SLOT_OPEN",
                **common,
                "retrieval": {
                    "query": make_retrieval_query(request),
                    "preferredSources": [
                        "local_curated_foley",
                        "local_ambience_library",
                        "future_audio_dataset_index"
                    ],
                    "candidateUri": None,
                    "retrievalScore": None,
                },
                "generation": None,
                "audit": {
                    "requiresHumanAudition": True,
                    "failureReason": None,
                    "notes": "Retrieval is preferred before model generation for controllability and lower runtime cost."
                },
            }
        )

        slots.append(
            {
                "candidateSlotId": f"slot_v0_{idx:04d}_generation_fallback",
                "slotType": "generation_fallback",
                "status": "SLOT_OPEN",
                **common,
                "retrieval": None,
                "generation": {
                    "prompt": make_generation_prompt(request),
                    "suggestedBaseline": "offline_text_to_audio_or_foley_baseline",
                    "candidateUri": None,
                    "sourceModel": None,
                    "generationSeed": None,
                    "generationScore": None,
                },
                "audit": {
                    "requiresHumanAudition": True,
                    "failureReason": None,
                    "notes": "Generation is a fallback slot; no waveform has been produced in this step."
                },
            }
        )

    write_jsonl(out_jsonl, slots)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "candidateSlotId",
                "requestId",
                "slotType",
                "status",
                "sourceEventId",
                "sceneId",
                "blueprintId",
                "layer",
                "eventLabel",
                "startSec",
                "endSec",
                "durationSec",
                "priority",
                "intensity",
            ],
        )
        writer.writeheader()
        for slot in slots:
            writer.writerow(
                {
                    "candidateSlotId": slot["candidateSlotId"],
                    "requestId": slot["requestId"],
                    "slotType": slot["slotType"],
                    "status": slot["status"],
                    "sourceEventId": slot["sourceEventId"],
                    "sceneId": slot["sceneId"],
                    "blueprintId": slot["blueprintId"],
                    "layer": slot["layer"],
                    "eventLabel": slot["eventLabel"],
                    "startSec": slot["timing"]["startSec"],
                    "endSec": slot["timing"]["endSec"],
                    "durationSec": slot["timing"]["durationSec"],
                    "priority": slot["priority"],
                    "intensity": slot["intensity"],
                }
            )

    retrieval_count = sum(1 for slot in slots if slot["slotType"] == "retrieval_primary")
    generation_count = sum(1 for slot in slots if slot["slotType"] == "generation_fallback")

    manifest = {
        "schemaVersion": "week12.audio-candidate-bank-slots-manifest.v0",
        "generatedAt": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "PASS",
        "requestManifestStatus": request_manifest.get("status"),
        "requestTimingWarningAccepted": timing_warning_accepted,
        "requestTimingWarningInterpretation": (
            "duration_seconds is treated as clip/scene duration when start_seconds and end_seconds exist; "
            "event timing uses end_seconds - start_seconds."
        ),
        "requestCount": len(requests),
        "slotCount": len(slots),
        "retrievalSlotCount": retrieval_count,
        "generationFallbackSlotCount": generation_count,
        "mainbase": {
            "repo": git_remote(mainbase),
            "commit": git_short_head(mainbase),
            "requestManifestPath": "artifacts/manifests/week12_candidate_audio_request_manifest_v1.json",
            "requestManifestSha256": sha256_file(request_manifest_path),
            "requestJsonlPath": "artifacts/manifests/week12_candidate_audio_requests_v1.jsonl",
            "requestJsonlSha256": sha256_file(request_jsonl_path),
        },
        "outputs": {
            "jsonl": "artifacts/manifests/week12_audio_candidate_bank_slots_v0.jsonl",
            "jsonlSha256": sha256_file(out_jsonl),
            "csv": "artifacts/manifests/week12_audio_candidate_bank_slots_v0.csv",
            "csvSha256": sha256_file(out_csv),
        },
        "slotStatusSemantics": {
            "SLOT_OPEN": "A slot is created and ready for retrieval or generation, but no candidate waveform has been attached yet."
        },
        "doesNotClaim": [
            "audio waveform generation",
            "candidate waveform exists",
            "retrieval index has been queried",
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