#!/usr/bin/env python3
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import struct
import wave
from pathlib import Path


ROOT = Path.cwd()

PLAN_JSON = ROOT / "artifacts/evals/week17_layer_mix_plan_v0.json"

AUDIO_DIR = ROOT / "artifacts/audio/week17_control_placeholders"
OUT_JSON = ROOT / "artifacts/evals/week17_control_audio_placeholder_manifest.json"
OUT_CSV = ROOT / "artifacts/evals/week17_control_audio_placeholder_manifest.csv"
OUT_DOC = ROOT / "docs/evals/week17_control_audio_placeholder_manifest.md"

SAMPLE_RATE = 22050
DURATION_SEC = 2.0
CHANNELS = 1
SAMPLE_WIDTH_BYTES = 2
AMPLITUDE = 0.16


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def candidate_frequency(candidate_id: str) -> float:
    suffix = int(candidate_id.rsplit("_", 1)[-1])
    return 220.0 + suffix * 27.5


def write_placeholder_wav(path: Path, candidate_id: str) -> None:
    freq = candidate_frequency(candidate_id)
    total_frames = int(SAMPLE_RATE * DURATION_SEC)
    fade_frames = int(0.08 * SAMPLE_RATE)

    frames = bytearray()
    for i in range(total_frames):
        t = i / SAMPLE_RATE

        carrier = math.sin(2.0 * math.pi * freq * t)
        overtone = 0.35 * math.sin(2.0 * math.pi * freq * 2.01 * t)
        slow_mod = 0.75 + 0.25 * math.sin(2.0 * math.pi * 2.0 * t)

        sample = (carrier + overtone) * slow_mod

        if i < fade_frames:
            sample *= i / fade_frames
        elif i > total_frames - fade_frames:
            sample *= max(0.0, (total_frames - i) / fade_frames)

        sample = max(-1.0, min(1.0, sample * AMPLITUDE))
        frames.extend(struct.pack("<h", int(sample * 32767)))

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH_BYTES)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(bytes(frames))


def main() -> int:
    if not PLAN_JSON.exists():
        raise FileNotFoundError(f"Missing layer mix plan: {PLAN_JSON}")

    plan = json.loads(PLAN_JSON.read_text(encoding="utf-8"))
    selected = plan.get("selectedControlInputs", [])

    if len(selected) != 7:
        raise RuntimeError(f"Expected 7 selected controls, got {len(selected)}")

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for item in selected:
        candidate_id = item["candidateId"]
        wav_path = AUDIO_DIR / f"{candidate_id}_control_placeholder.wav"
        write_placeholder_wav(wav_path, candidate_id)

        rows.append({
            "candidateId": candidate_id,
            "artifactPath": str(wav_path),
            "artifactType": "synthetic_control_placeholder_wav",
            "sampleRate": SAMPLE_RATE,
            "durationSec": DURATION_SEC,
            "channels": CHANNELS,
            "sampleWidthBytes": SAMPLE_WIDTH_BYTES,
            "sha256": sha256(wav_path),
            "source": "deterministic_local_placeholder_generator",
            "semanticAudioQualityClaimed": False,
            "finalMixReadinessClaimed": False,
        })

    manifest = {
        "decision": "PASS_WEEK17_CONTROL_AUDIO_PLACEHOLDERS_MATERIALIZED",
        "generatedAtUtc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "schemaVersion": "week17.control_audio_placeholder_manifest.v0",
        "sourcePlan": str(PLAN_JSON),
        "sourcePlanSha256": sha256(PLAN_JSON),
        "artifactTotal": len(rows),
        "sampleRate": SAMPLE_RATE,
        "durationSec": DURATION_SEC,
        "placeholderOnly": True,
        "realCandidateAudioClaimed": False,
        "semanticAudioQualityClaimed": False,
        "finalMixReadinessClaimed": False,
        "records": rows,
        "blockedClaims": [
            "real generated candidate audio",
            "semantic audio quality pass",
            "human review pass",
            "final mix readiness",
            "production mixer availability",
        ],
    }

    OUT_JSON.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "candidateId",
            "artifactPath",
            "artifactType",
            "sampleRate",
            "durationSec",
            "channels",
            "sampleWidthBytes",
            "sha256",
            "source",
            "semanticAudioQualityClaimed",
            "finalMixReadinessClaimed",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    OUT_DOC.write_text(
        "# Week17 Control Audio Placeholder Manifest\n\n"
        "## Purpose\n\n"
        "Materialize deterministic local WAV placeholders for the 7 selected control inputs in `week17_layer_mix_plan_v0`.\n\n"
        "## Boundary\n\n"
        "- These files are pipeline placeholders, not real generated candidate audio.\n"
        "- They do not claim semantic audio quality.\n"
        "- They do not claim final mix readiness.\n"
        "- They exist to make artifact path resolution and mixer plumbing testable.\n\n"
        f"## Artifact total\n\n`{len(rows)}`\n",
        encoding="utf-8",
    )

    print(json.dumps({
        "decision": manifest["decision"],
        "artifactTotal": manifest["artifactTotal"],
        "audioDir": str(AUDIO_DIR),
        "outJson": str(OUT_JSON),
        "outCsv": str(OUT_CSV),
        "outDoc": str(OUT_DOC),
        "candidateIds": [r["candidateId"] for r in rows],
    }, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())