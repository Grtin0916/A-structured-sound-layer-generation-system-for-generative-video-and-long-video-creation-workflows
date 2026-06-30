from __future__ import annotations

import math
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np


@dataclass
class AudioMetrics:
    path: str
    readable: bool
    duration_sec: float
    sample_rate: int
    channels: int
    rms: float
    peak: float
    clip_rate: float
    silence_ratio: float


def write_wav_mono(path: Path, audio: np.ndarray, sample_rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(audio, dtype=np.float32)
    x = np.nan_to_num(x)
    x = np.clip(x, -1.0, 1.0)
    pcm = (x * 32767.0).astype("<i2")

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def read_wav_metrics(path: Path) -> AudioMetrics:
    try:
        with wave.open(str(path), "rb") as wf:
            channels = wf.getnchannels()
            sample_rate = wf.getframerate()
            nframes = wf.getnframes()
            frames = wf.readframes(nframes)
        pcm = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
        if channels > 1:
            pcm = pcm.reshape(-1, channels).mean(axis=1)
        duration = float(len(pcm) / sample_rate) if sample_rate else 0.0
        rms = float(np.sqrt(np.mean(np.square(pcm)))) if len(pcm) else 0.0
        peak = float(np.max(np.abs(pcm))) if len(pcm) else 0.0
        clip_rate = float(np.mean(np.abs(pcm) >= 0.98)) if len(pcm) else 0.0
        silence_ratio = float(np.mean(np.abs(pcm) < 1e-4)) if len(pcm) else 1.0
        return AudioMetrics(str(path), True, duration, sample_rate, channels, rms, peak, clip_rate, silence_ratio)
    except Exception:
        return AudioMetrics(str(path), False, 0.0, 0, 0, 0.0, 0.0, 1.0, 1.0)


def synthesize_control_audio(
    duration_sec: float,
    event_times: Iterable[float],
    variant_seed: int = 0,
    sample_rate: int = 16000,
) -> np.ndarray:
    """Deterministic DSS-conditioned control fallback.

    This is not V2A. It creates audible ambience + event impulses so the
    evaluation/ranking pipeline has real audio files while MMAudio is blocked.
    """
    duration_sec = max(1.0, float(duration_sec))
    n = int(duration_sec * sample_rate)
    t = np.arange(n, dtype=np.float32) / sample_rate

    rng = np.random.default_rng(20260630 + variant_seed)

    # Low ambience bed.
    audio = 0.008 * rng.normal(0, 1, size=n).astype(np.float32)
    audio += 0.012 * np.sin(2 * math.pi * (110 + 17 * variant_seed) * t).astype(np.float32)

    for i, et in enumerate(event_times):
        center = int(max(0.0, min(duration_sec - 0.05, et)) * sample_rate)
        length = int((0.055 + 0.018 * ((i + variant_seed) % 3)) * sample_rate)
        end = min(n, center + length)
        if end <= center:
            continue

        local = np.arange(end - center, dtype=np.float32) / sample_rate
        freq = 550 + 190 * ((i + variant_seed) % 5)
        env = np.exp(-local * (28 + 4 * i)).astype(np.float32)
        transient = 0.20 * np.sin(2 * math.pi * freq * local).astype(np.float32) * env
        click = 0.10 * rng.normal(0, 1, size=end - center).astype(np.float32) * env
        audio[center:end] += transient + click

    peak = float(np.max(np.abs(audio))) if len(audio) else 1.0
    if peak > 0:
        audio = 0.85 * audio / max(peak, 1.0)

    return np.clip(audio, -0.95, 0.95).astype(np.float32)