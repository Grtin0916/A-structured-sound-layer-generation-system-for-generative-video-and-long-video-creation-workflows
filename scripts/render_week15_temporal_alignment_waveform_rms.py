#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    import soundfile as sf
except Exception as exc:
    raise SystemExit(
        "Missing dependency: soundfile. Use the existing project environment, or install python-soundfile."
    ) from exc

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:
    raise SystemExit(
        "Missing dependency: matplotlib. This script needs matplotlib to save waveform/RMS PNG files."
    ) from exc


CANDIDATES = ["procedural_v0_0004", "procedural_v0_0010"]
OUT_DIR = Path("artifacts/figures/week15_temporal_alignment")
INDEX_PATH = Path("artifacts/evals/week15_temporal_alignment_waveform_rms_index.json")


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    y, sr = sf.read(path, always_2d=True)
    if y.size == 0:
        raise ValueError(f"empty audio: {path}")
    y = y.astype(np.float32)
    mono = y.mean(axis=1)
    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    if peak > 1.0:
        mono = mono / peak
    return mono, int(sr)


def frame_rms(y: np.ndarray, sr: int, frame_ms: float = 25.0, hop_ms: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    frame = max(1, int(sr * frame_ms / 1000.0))
    hop = max(1, int(sr * hop_ms / 1000.0))

    if len(y) < frame:
        padded = np.zeros(frame, dtype=np.float32)
        padded[: len(y)] = y
        y = padded

    n_frames = 1 + max(0, (len(y) - frame) // hop)
    rms = np.empty(n_frames, dtype=np.float32)
    times = np.empty(n_frames, dtype=np.float32)

    for i in range(n_frames):
        start = i * hop
        chunk = y[start : start + frame]
        rms[i] = math.sqrt(float(np.mean(chunk * chunk)) + 1e-12)
        times[i] = (start + frame / 2.0) / sr

    return times, rms


def onset_proxy_sec(times: np.ndarray, rms: np.ndarray) -> float | None:
    if rms.size == 0:
        return None

    floor = float(np.percentile(rms, 10))
    high = float(np.percentile(rms, 95))
    threshold = floor + 0.20 * max(1e-8, high - floor)

    active = np.where(rms >= threshold)[0]
    if active.size == 0:
        return None
    return float(times[int(active[0])])


def find_candidate_wavs(candidate_id: str) -> dict[str, Path]:
    all_wavs = sorted(Path("artifacts").rglob(f"*{candidate_id}*.wav"))

    remediated = [
        p for p in all_wavs
        if "remediated" in str(p).lower() or "trimmed" in p.name.lower()
    ]
    original = [
        p for p in all_wavs
        if p not in remediated
        and "remediated" not in str(p).lower()
        and "trimmed" not in p.name.lower()
    ]

    if not original:
        # 兜底：如果仓库只保留了 remediated 文件，也不能伪造 original。
        raise FileNotFoundError(
            f"missing original wav for {candidate_id}; found wavs={list(map(str, all_wavs))}"
        )
    if not remediated:
        raise FileNotFoundError(
            f"missing remediated wav for {candidate_id}; found wavs={list(map(str, all_wavs))}"
        )

    return {
        "original": original[0],
        "remediated": remediated[0],
    }


def plot_pair(candidate_id: str, original_path: Path, remediated_path: Path) -> dict[str, Any]:
    y0, sr0 = read_audio(original_path)
    y1, sr1 = read_audio(remediated_path)

    t0 = np.arange(len(y0), dtype=np.float32) / sr0
    t1 = np.arange(len(y1), dtype=np.float32) / sr1

    rms_t0, rms0 = frame_rms(y0, sr0)
    rms_t1, rms1 = frame_rms(y1, sr1)

    onset0 = onset_proxy_sec(rms_t0, rms0)
    onset1 = onset_proxy_sec(rms_t1, rms1)

    out_png = OUT_DIR / f"{candidate_id}_waveform_rms_original_vs_remediated.png"

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=False)

    axes[0].plot(t0, y0, linewidth=0.8, label="original waveform")
    axes[0].plot(rms_t0, rms0, linewidth=1.2, label="original RMS")
    if onset0 is not None:
        axes[0].axvline(onset0, linestyle="--", linewidth=1.0, label=f"onset proxy={onset0:.3f}s")
    axes[0].set_title(f"{candidate_id} original | sr={sr0}, duration={len(y0)/sr0:.3f}s")
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("amplitude / RMS")
    axes[0].legend(loc="upper right")

    axes[1].plot(t1, y1, linewidth=0.8, label="remediated waveform")
    axes[1].plot(rms_t1, rms1, linewidth=1.2, label="remediated RMS")
    if onset1 is not None:
        axes[1].axvline(onset1, linestyle="--", linewidth=1.0, label=f"onset proxy={onset1:.3f}s")
    axes[1].set_title(f"{candidate_id} remediated | sr={sr1}, duration={len(y1)/sr1:.3f}s")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("amplitude / RMS")
    axes[1].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

    duration0 = len(y0) / sr0
    duration1 = len(y1) / sr1

    return {
        "candidateId": candidate_id,
        "originalAudio": str(original_path),
        "remediatedAudio": str(remediated_path),
        "figure": str(out_png),
        "originalSampleRate": sr0,
        "remediatedSampleRate": sr1,
        "originalDurationSec": round(duration0, 6),
        "remediatedDurationSec": round(duration1, 6),
        "durationDeltaSec": round(duration1 - duration0, 6),
        "originalOnsetProxySec": None if onset0 is None else round(onset0, 6),
        "remediatedOnsetProxySec": None if onset1 is None else round(onset1, 6),
        "onsetProxyDeltaSec": None if onset0 is None or onset1 is None else round(onset1 - onset0, 6),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    errors = []

    for candidate_id in CANDIDATES:
        try:
            paths = find_candidate_wavs(candidate_id)
            records.append(plot_pair(candidate_id, paths["original"], paths["remediated"]))
        except Exception as exc:
            errors.append({"candidateId": candidate_id, "error": str(exc)})

    index = {
        "schemaVersion": "week15.temporal_alignment_waveform_rms_index.v1",
        "status": "PASS" if records and not errors else "PARTIAL" if records else "FAIL",
        "purpose": "Explain original FAIL_DRIFT and remediated PASS using waveform and frame-level RMS evidence.",
        "candidates": records,
        "errors": errors,
        "boundary": [
            "visual_explainability_only",
            "does_not_claim_semantic_audio_quality",
            "does_not_claim_human_audition_passed",
            "does_not_claim_final_mix_readiness",
        ],
    }

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(index, ensure_ascii=False, indent=2))
    print(f"WROTE_INDEX={INDEX_PATH}")
    for r in records:
        print(f"WROTE_FIGURE={r['figure']}")

    if index["status"] == "FAIL":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
