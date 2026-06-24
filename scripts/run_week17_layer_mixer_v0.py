#!/usr/bin/env python3
"""
Week17 layer mixer v0 placeholder-control dry-run.

Boundary:
- Uses only 7 selected P4 control placeholder WAVs: 0001/0002/0003/0005/0006/0008/0009.
- Does not claim real candidate audio quality.
- Does not claim semantic audio quality pass.
- Does not claim human review pass.
- Does not claim final mix readiness.
- Does not claim production mixer availability.
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

PLACEHOLDER_MANIFEST = ROOT / "artifacts/evals/week17_control_audio_placeholder_manifest.json"
PLAN_JSON = ROOT / "artifacts/evals/week17_layer_mix_plan_v0.json"

OUT_DIR = ROOT / "artifacts/audio/week17_layer_mix_v0"
OUT_WAV = OUT_DIR / "week17_layer_mix_v0_placeholder_control_mix.wav"
OUT_JSON = ROOT / "artifacts/evals/week17_layer_mix_v0_manifest.json"
OUT_CSV = ROOT / "artifacts/evals/week17_layer_mix_v0_manifest.csv"
OUT_DOC = ROOT / "docs/evals/week17_layer_mix_v0.md"

SELECTED_IDS = {"0001", "0002", "0003", "0005", "0006", "0008", "0009"}
BLOCKED_IDS = {"0004", "0007", "0010"}
TARGET_PEAK = 0.8912509381337456  # roughly -1 dBFS
EPS = 1e-12


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"required input missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def walk_values(obj: Any):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from walk_values(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from walk_values(v)
    else:
        yield obj


def extract_candidate_id(text: str) -> str | None:
    m = re.search(r"(?<!\d)(00(?:0[1-9]|10))(?!\d)", text)
    return m.group(1) if m else None


def discover_wavs_from_manifest(obj: Any) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for v in walk_values(obj):
        if not isinstance(v, str):
            continue
        if not v.lower().endswith(".wav"):
            continue
        cid = extract_candidate_id(v)
        if cid not in SELECTED_IDS:
            continue
        p = Path(v)
        if not p.is_absolute():
            p = ROOT / p
        if p.exists():
            found[cid] = p
    return found


def discover_wavs_from_filesystem() -> dict[str, Path]:
    found: dict[str, Path] = {}
    for p in sorted((ROOT / "artifacts").rglob("*.wav")):
        s = str(p)
        cid = extract_candidate_id(s)
        if cid in SELECTED_IDS and cid not in found:
            found[cid] = p
    return found


def read_wav_float_mono(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        frame_count = wf.getnframes()
        compression = wf.getcomptype()
        raw = wf.readframes(frame_count)

    if compression != "NONE":
        raise ValueError(f"{path} is compressed WAV: {compression}")
    if sample_width not in (1, 2, 4):
        raise ValueError(f"{path} unsupported sample width: {sample_width}")

    if sample_width == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        data = (data - 128.0) / 128.0
    elif sample_width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    else:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0

    if channels > 1:
        if data.size % channels != 0:
            raise ValueError(f"{path} frame data is not divisible by channels={channels}")
        data = data.reshape(-1, channels).mean(axis=1)

    meta = {
        "path": str(path.relative_to(ROOT)),
        "channels": channels,
        "sampleWidthBytes": sample_width,
        "sampleRate": sample_rate,
        "frameCount": int(frame_count),
        "durationSec": float(frame_count / sample_rate) if sample_rate else 0.0,
        "inputPeak": float(np.max(np.abs(data))) if data.size else 0.0,
        "inputRms": float(np.sqrt(np.mean(data * data))) if data.size else 0.0,
        "inputSilent": bool(data.size == 0 or np.max(np.abs(data)) <= EPS),
    }
    return data.astype(np.float32), meta


def write_pcm16_mono(path: Path, sample_rate: int, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(data, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_DOC.parent.mkdir(parents=True, exist_ok=True)

    placeholder_manifest = load_json(PLACEHOLDER_MANIFEST)
    plan_json = load_json(PLAN_JSON)

    wavs = discover_wavs_from_manifest(placeholder_manifest)
    fs_wavs = discover_wavs_from_filesystem()
    wavs.update({k: v for k, v in fs_wavs.items() if k not in wavs})

    missing = sorted(SELECTED_IDS - set(wavs))
    blocked_present = sorted(BLOCKED_IDS & set(wavs))
    if missing:
        raise RuntimeError(f"missing selected placeholder wav ids: {missing}")
    if blocked_present:
        raise RuntimeError(f"blocked ids must not enter mix: {blocked_present}")

    tracks = []
    arrays = []
    sample_rates = set()
    max_len = 0

    for cid in sorted(SELECTED_IDS):
        arr, meta = read_wav_float_mono(wavs[cid])
        if meta["inputSilent"]:
            raise RuntimeError(f"silent or empty placeholder input: {cid} {meta['path']}")
        sample_rates.add(meta["sampleRate"])
        max_len = max(max_len, arr.size)
        arrays.append((cid, arr, meta))

    if len(sample_rates) != 1:
        raise RuntimeError(f"sample-rate mismatch: {sorted(sample_rates)}")

    sample_rate = int(next(iter(sample_rates)))
    track_total = len(arrays)
    base_gain = 1.0 / track_total

    mix = np.zeros(max_len, dtype=np.float32)

    for cid, arr, meta in arrays:
        padded = np.zeros(max_len, dtype=np.float32)
        padded[: arr.size] = arr
        contribution = padded * base_gain
        mix += contribution

        tracks.append({
            "candidateId": cid,
            "role": "selected_control_placeholder",
            "inputPath": meta["path"],
            "sampleRate": meta["sampleRate"],
            "channelsOriginal": meta["channels"],
            "sampleWidthBytesOriginal": meta["sampleWidthBytes"],
            "frameCount": meta["frameCount"],
            "durationSec": meta["durationSec"],
            "inputPeak": meta["inputPeak"],
            "inputRms": meta["inputRms"],
            "gain": base_gain,
            "contributionPeak": float(np.max(np.abs(contribution))) if contribution.size else 0.0,
            "contributionRms": float(np.sqrt(np.mean(contribution * contribution))) if contribution.size else 0.0,
        })

    pre_normalize_peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    pre_normalize_rms = float(np.sqrt(np.mean(mix * mix))) if mix.size else 0.0
    pre_clip_rate = float(np.mean(np.abs(mix) >= 1.0)) if mix.size else 0.0

    if pre_normalize_peak <= EPS:
        raise RuntimeError("mixed signal is silent")

    normalize_gain = TARGET_PEAK / pre_normalize_peak
    normalized = mix * normalize_gain

    final_clip_rate_before_clip = float(np.mean(np.abs(normalized) > 1.0)) if normalized.size else 0.0
    final = np.clip(normalized, -1.0, 1.0)
    final_peak = float(np.max(np.abs(final))) if final.size else 0.0
    final_rms = float(np.sqrt(np.mean(final * final))) if final.size else 0.0
    final_duration_sec = float(final.size / sample_rate)

    write_pcm16_mono(OUT_WAV, sample_rate, final)

    decision = "PASS_WEEK17_LAYER_MIX_V0_PLACEHOLDER_CONTROL"
    blocked_claims = [
        "realCandidateAudioClaimed",
        "semanticAudioQualityPassClaimed",
        "humanReviewPassClaimed",
        "finalMixReadinessClaimed",
        "productionMixerAvailabilityClaimed",
    ]

    manifest = {
        "schemaVersion": "week17.layer_mix_v0.placeholder_control.v1",
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "trackTotal": track_total,
        "selectedControlIds": sorted(SELECTED_IDS),
        "blockedInputIds": sorted(BLOCKED_IDS),
        "sampleRate": sample_rate,
        "channels": 1,
        "sampleWidthBytes": 2,
        "mixArtifactPath": str(OUT_WAV.relative_to(ROOT)),
        "jsonManifestPath": str(OUT_JSON.relative_to(ROOT)),
        "csvManifestPath": str(OUT_CSV.relative_to(ROOT)),
        "preNormalizePeak": pre_normalize_peak,
        "preNormalizeRms": pre_normalize_rms,
        "preNormalizeClipRate": pre_clip_rate,
        "normalizeTargetPeak": TARGET_PEAK,
        "normalizeGain": float(normalize_gain),
        "finalPeak": final_peak,
        "finalRms": final_rms,
        "finalClipRateBeforeClip": final_clip_rate_before_clip,
        "finalDurationSec": final_duration_sec,
        "placeholderInputOnly": True,
        "realCandidateAudioClaimed": False,
        "semanticAudioQualityPassClaimed": False,
        "humanReviewPassClaimed": False,
        "finalMixReadinessClaimed": False,
        "productionMixerAvailabilityClaimed": False,
        "blockedClaims": blocked_claims,
        "sourceManifest": str(PLACEHOLDER_MANIFEST.relative_to(ROOT)),
        "sourcePlan": str(PLAN_JSON.relative_to(ROOT)),
        "previousS3EvidencePreserved": True,
        "realMixerV0DryRunExecuted": True,
        "tracks": tracks,
        "sourcePlanDigest": plan_json if isinstance(plan_json, dict) else {"rawType": type(plan_json).__name__},
    }

    OUT_JSON.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "candidateId",
                "role",
                "inputPath",
                "sampleRate",
                "channelsOriginal",
                "sampleWidthBytesOriginal",
                "frameCount",
                "durationSec",
                "inputPeak",
                "inputRms",
                "gain",
                "contributionPeak",
                "contributionRms",
            ],
        )
        writer.writeheader()
        writer.writerows(tracks)

    OUT_DOC.write_text(
        "\n".join([
            "# Week17 Layer Mix V0 Placeholder-Control Dry Run",
            "",
            f"- Decision: `{decision}`",
            f"- Track total: `{track_total}`",
            f"- Selected controls: `{', '.join(sorted(SELECTED_IDS))}`",
            f"- Blocked ids: `{', '.join(sorted(BLOCKED_IDS))}`",
            f"- Output WAV: `{OUT_WAV.relative_to(ROOT)}`",
            f"- Manifest JSON: `{OUT_JSON.relative_to(ROOT)}`",
            f"- Manifest CSV: `{OUT_CSV.relative_to(ROOT)}`",
            f"- Sample rate: `{sample_rate}`",
            f"- Final duration sec: `{final_duration_sec:.6f}`",
            f"- Pre-normalize peak: `{pre_normalize_peak:.9f}`",
            f"- Normalize gain: `{normalize_gain:.9f}`",
            f"- Final peak: `{final_peak:.9f}`",
            f"- Final RMS: `{final_rms:.9f}`",
            f"- Final clip rate before clip: `{final_clip_rate_before_clip:.9f}`",
            "",
            "## Boundary",
            "",
            "- This is a deterministic placeholder-control mix.",
            "- It is not real generated candidate audio.",
            "- It does not claim semantic audio quality pass.",
            "- It does not claim human review pass.",
            "- It does not claim final mix readiness.",
            "- It does not claim production mixer availability.",
            "",
        ]),
        encoding="utf-8",
    )

    print(json.dumps({
        "decision": decision,
        "trackTotal": track_total,
        "mixArtifactPath": str(OUT_WAV.relative_to(ROOT)),
        "jsonManifestPath": str(OUT_JSON.relative_to(ROOT)),
        "csvManifestPath": str(OUT_CSV.relative_to(ROOT)),
        "finalPeak": final_peak,
        "finalRms": final_rms,
        "finalClipRateBeforeClip": final_clip_rate_before_clip,
        "placeholderInputOnly": True,
        "finalMixReadinessClaimed": False,
    }, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        raise