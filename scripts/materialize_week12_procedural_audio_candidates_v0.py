#!/usr/bin/env python3
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import random
import struct
import subprocess
import wave
from pathlib import Path
from typing import Any


SAMPLE_RATE = 16000
AMPLITUDE = 0.22
RANDOM_SEED = 20260603


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


def clamp_i16(x: float) -> int:
    x = max(-1.0, min(1.0, x))
    return int(x * 32767)


def write_wav(path: Path, samples: list[float], sample_rate: int = SAMPLE_RATE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        frames = b"".join(struct.pack("<h", clamp_i16(s)) for s in samples)
        wf.writeframes(frames)


def lowpass_noise_sample(prev: float, rng: random.Random, alpha: float = 0.985) -> float:
    white = rng.uniform(-1.0, 1.0)
    return alpha * prev + (1.0 - alpha) * white


def synth_ambience(duration_sec: float, label: str, tags: str, seed: int) -> list[float]:
    n = max(1, int(duration_sec * SAMPLE_RATE))
    rng = random.Random(seed)
    samples: list[float] = []
    lp = 0.0

    rainish = "rain" in tags.lower() or "rain" in label.lower()
    urban = "urban" in tags.lower() or "street" in tags.lower() or "bike" in label.lower()
    indoor = "indoor" in tags.lower() or "room" in tags.lower()

    for i in range(n):
        t = i / SAMPLE_RATE
        lp = lowpass_noise_sample(lp, rng, alpha=0.992 if indoor else 0.982)

        bed = lp
        if rainish:
            # fine-grain rain texture: filtered noise + tiny random impacts
            impact = rng.uniform(-1.0, 1.0) if rng.random() < 0.006 else 0.0
            bed = 0.55 * lp + 0.18 * impact
        elif urban:
            hum = math.sin(2 * math.pi * 90 * t) * 0.08 + math.sin(2 * math.pi * 145 * t) * 0.04
            bed = 0.50 * lp + hum
        elif indoor:
            hum = math.sin(2 * math.pi * 60 * t) * 0.04
            bed = 0.35 * lp + hum

        # soft fade-in/out to avoid clicks
        fade = min(1.0, i / max(1, int(0.08 * SAMPLE_RATE)), (n - i - 1) / max(1, int(0.08 * SAMPLE_RATE)))
        samples.append(AMPLITUDE * bed * max(0.0, fade))

    return samples


def click_envelope(x: float) -> float:
    # x in [0, 1]
    return math.exp(-12.0 * x) * math.sin(2.0 * math.pi * (110.0 + 420.0 * (1.0 - x)) * x)


def synth_foley(duration_sec: float, label: str, tags: str, seed: int) -> list[float]:
    n = max(1, int(duration_sec * SAMPLE_RATE))
    rng = random.Random(seed)
    samples = [0.0 for _ in range(n)]

    label_l = label.lower()
    if "footstep" in label_l or "walk" in tags.lower():
        event_count = max(3, int(duration_sec * 1.8))
        base_times = [0.25 + k * max(0.35, duration_sec / max(1, event_count)) for k in range(event_count)]
    elif "keyboard" in label_l:
        event_count = max(10, int(duration_sec * 6))
        base_times = sorted(rng.uniform(0.1, max(0.11, duration_sec - 0.1)) for _ in range(event_count))
    elif "door" in label_l:
        base_times = [duration_sec * 0.48, duration_sec * 0.54]
    elif "bike" in label_l or "bicycle" in label_l:
        base_times = [duration_sec * 0.25, duration_sec * 0.50, duration_sec * 0.75]
    elif "rain" in label_l or "window" in label_l:
        event_count = max(8, int(duration_sec * 5))
        base_times = sorted(rng.uniform(0.05, max(0.06, duration_sec - 0.05)) for _ in range(event_count))
    else:
        base_times = [duration_sec * 0.4, duration_sec * 0.7]

    for et in base_times:
        center = int(et * SAMPLE_RATE)
        length = int((0.10 if "door" not in label_l else 0.18) * SAMPLE_RATE)
        for j in range(length):
            idx = center + j
            if 0 <= idx < n:
                x = j / max(1, length)
                transient = click_envelope(x)
                noise = rng.uniform(-0.2, 0.2) * math.exp(-10.0 * x)
                samples[idx] += AMPLITUDE * (0.65 * transient + noise)

    # add a very quiet bed so isolated clicks are not totally dry
    lp = 0.0
    for i in range(n):
        lp = lowpass_noise_sample(lp, rng, alpha=0.995)
        samples[i] += 0.025 * lp
        fade = min(1.0, i / max(1, int(0.04 * SAMPLE_RATE)), (n - i - 1) / max(1, int(0.04 * SAMPLE_RATE)))
        samples[i] *= max(0.0, fade)

    return samples


def synth_candidate(slot: dict[str, Any], seed: int) -> list[float]:
    duration_sec = float(slot.get("timing", {}).get("durationSec") or 1.0)
    layer = str(slot.get("layer") or "").lower()
    label = str(slot.get("eventLabel") or "")
    tags = ""
    gen = slot.get("generation") or {}
    prompt = str(gen.get("prompt") or "")
    if "Semantic tags:" in prompt:
        tags = prompt.split("Semantic tags:", 1)[1].split(".", 1)[0].strip()

    if layer == "ambience":
        return synth_ambience(duration_sec, label, tags, seed)
    return synth_foley(duration_sec, label, tags, seed)


def main() -> int:
    mainbase = Path.home() / "work" / "grt_work" / "audio_engineering_repo_skeleton_v1"

    slots_manifest_path = mainbase / "artifacts/manifests/week12_audio_candidate_bank_slots_manifest_v0.json"
    slots_jsonl_path = mainbase / "artifacts/manifests/week12_audio_candidate_bank_slots_v0.jsonl"

    out_dir = mainbase / "artifacts/audio_candidates/week12_procedural_baseline_v0"
    out_manifest = mainbase / "artifacts/manifests/week12_procedural_audio_candidates_manifest_v0.json"
    out_jsonl = mainbase / "artifacts/manifests/week12_procedural_audio_candidates_v0.jsonl"
    out_csv = mainbase / "artifacts/manifests/week12_procedural_audio_candidates_v0.csv"

    for path in [slots_manifest_path, slots_jsonl_path]:
        if not path.exists():
            raise SystemExit(f"MISSING_REQUIRED_FILE={path}")

    slots_manifest = load_json(slots_manifest_path)
    slots = load_jsonl(slots_jsonl_path)

    if slots_manifest.get("status") != "PASS":
        raise SystemExit(f"SLOTS_MANIFEST_NOT_PASS={slots_manifest.get('status')}")

    generation_slots = [s for s in slots if s.get("slotType") == "generation_fallback"]
    if not generation_slots:
        raise SystemExit("NO_GENERATION_FALLBACK_SLOTS")

    candidate_rows: list[dict[str, Any]] = []

    for idx, slot in enumerate(generation_slots, start=1):
        slot_id = slot["candidateSlotId"]
        duration = float(slot.get("timing", {}).get("durationSec") or 1.0)
        layer = slot.get("layer")
        label = slot.get("eventLabel")
        seed = RANDOM_SEED + idx

        filename = f"{idx:04d}_{slot_id}.wav"
        wav_path = out_dir / filename
        samples = synth_candidate(slot, seed=seed)
        write_wav(wav_path, samples, SAMPLE_RATE)

        candidate = {
            "candidateId": f"procedural_v0_{idx:04d}",
            "candidateStatus": "CANDIDATE_ATTACHED_PROCEDURAL_BASELINE",
            "sourceSlotId": slot_id,
            "requestId": slot.get("requestId"),
            "sourceEventId": slot.get("sourceEventId"),
            "sceneId": slot.get("sceneId"),
            "blueprintId": slot.get("blueprintId"),
            "blueprintArtifactUri": slot.get("blueprintArtifactUri"),
            "layer": layer,
            "eventLabel": label,
            "timing": slot.get("timing"),
            "candidateUri": str(wav_path.relative_to(mainbase)),
            "candidateSha256": sha256_file(wav_path),
            "format": {
                "container": "wav",
                "encoding": "pcm_s16le",
                "sampleRate": SAMPLE_RATE,
                "channels": 1,
                "durationSec": duration,
            },
            "method": {
                "type": "deterministic_procedural_baseline",
                "seed": seed,
                "usesModelInference": False,
                "usesExternalNetwork": False,
            },
            "qualityBoundary": {
                "requiresHumanAudition": True,
                "semanticFidelityClaimed": False,
                "mixReadyClaimed": False,
                "purpose": "pipeline materialization and candidate attachment smoke"
            },
            "upstreamSlot": {
                "slotType": slot.get("slotType"),
                "statusBeforeAttachment": slot.get("status"),
                "generationPrompt": (slot.get("generation") or {}).get("prompt"),
            },
        }
        candidate_rows.append(candidate)

    write_jsonl(out_jsonl, candidate_rows)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "candidateId",
                "sourceSlotId",
                "requestId",
                "sourceEventId",
                "sceneId",
                "blueprintId",
                "layer",
                "eventLabel",
                "startSec",
                "endSec",
                "durationSec",
                "candidateUri",
                "sampleRate",
                "channels",
                "candidateStatus",
            ],
        )
        writer.writeheader()
        for c in candidate_rows:
            timing = c["timing"] or {}
            writer.writerow(
                {
                    "candidateId": c["candidateId"],
                    "sourceSlotId": c["sourceSlotId"],
                    "requestId": c["requestId"],
                    "sourceEventId": c["sourceEventId"],
                    "sceneId": c["sceneId"],
                    "blueprintId": c["blueprintId"],
                    "layer": c["layer"],
                    "eventLabel": c["eventLabel"],
                    "startSec": timing.get("startSec"),
                    "endSec": timing.get("endSec"),
                    "durationSec": timing.get("durationSec"),
                    "candidateUri": c["candidateUri"],
                    "sampleRate": c["format"]["sampleRate"],
                    "channels": c["format"]["channels"],
                    "candidateStatus": c["candidateStatus"],
                }
            )

    manifest = {
        "schemaVersion": "week12.procedural-audio-candidates-manifest.v0",
        "generatedAt": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "PASS",
        "candidateCount": len(candidate_rows),
        "sourceGenerationFallbackSlotCount": len(generation_slots),
        "allCandidatesAttached": len(candidate_rows) == len(generation_slots),
        "mainbase": {
            "repo": git_remote(mainbase),
            "commit": git_short_head(mainbase),
            "slotsManifestPath": "artifacts/manifests/week12_audio_candidate_bank_slots_manifest_v0.json",
            "slotsManifestSha256": sha256_file(slots_manifest_path),
            "slotsJsonlPath": "artifacts/manifests/week12_audio_candidate_bank_slots_v0.jsonl",
            "slotsJsonlSha256": sha256_file(slots_jsonl_path),
        },
        "outputs": {
            "candidateDir": "artifacts/audio_candidates/week12_procedural_baseline_v0",
            "jsonl": "artifacts/manifests/week12_procedural_audio_candidates_v0.jsonl",
            "jsonlSha256": sha256_file(out_jsonl),
            "csv": "artifacts/manifests/week12_procedural_audio_candidates_v0.csv",
            "csvSha256": sha256_file(out_csv),
        },
        "audioFormat": {
            "container": "wav",
            "encoding": "pcm_s16le",
            "sampleRate": SAMPLE_RATE,
            "channels": 1
        },
        "methodBoundary": {
            "type": "deterministic_procedural_baseline",
            "randomSeedBase": RANDOM_SEED,
            "usesModelInference": False,
            "usesExternalNetwork": False,
            "semanticFidelityClaimed": False,
            "mixReadyClaimed": False
        },
        "doesNotClaim": [
            "text-to-audio model inference",
            "retrieval index has been queried",
            "semantic audio quality",
            "human audition has completed",
            "production asset storage",
            "final mix readiness"
        ],
    }

    out_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())