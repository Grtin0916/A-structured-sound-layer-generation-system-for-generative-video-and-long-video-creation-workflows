#!/usr/bin/env python3
"""Standard-library audio and diagnostic helpers for repair experiments."""

from __future__ import annotations

from array import array
import binascii
import math
from pathlib import Path
import struct
import wave
import zlib


def read_pcm16(path: Path) -> tuple[wave._wave_params, array]:
    with wave.open(str(path), "rb") as handle:
        params = handle.getparams()
        if params.sampwidth != 2:
            raise ValueError(f"only PCM16 is supported, got sample_width={params.sampwidth}")
        samples = array("h")
        samples.frombytes(handle.readframes(params.nframes))
    if struct.pack("=h", 1) != struct.pack("<h", 1):
        samples.byteswap()
    return params, samples


def write_pcm16(path: Path, params: wave._wave_params, samples: array) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = array("h", samples)
    if struct.pack("=h", 1) != struct.pack("<h", 1):
        payload.byteswap()
    with wave.open(str(path), "wb") as handle:
        handle.setparams(params._replace(nframes=0))
        handle.writeframes(payload.tobytes())


def audio_metrics(params: wave._wave_params, samples: array) -> dict[str, float | int]:
    if not samples:
        return {
            "duration_sec": 0.0,
            "peak_abs": 0.0,
            "rms": 0.0,
            "rms_dbfs": -120.0,
            "clipped_ratio": 0.0,
            "silence_ratio": 1.0,
        }
    normalized = [abs(value) / 32768.0 for value in samples]
    peak = max(normalized)
    rms = math.sqrt(sum(value * value for value in normalized) / len(normalized))
    return {
        "duration_sec": len(samples) / (params.framerate * params.nchannels),
        "peak_abs": peak,
        "rms": rms,
        "rms_dbfs": 20.0 * math.log10(max(rms, 1.0e-6)),
        "clipped_ratio": sum(value >= 32760 for value in map(abs, samples)) / len(samples),
        "silence_ratio": sum(value <= 328 for value in map(abs, samples)) / len(samples),
    }


def _chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", binascii.crc32(kind + payload) & 0xFFFFFFFF)
    )


def _line(canvas: bytearray, width: int, height: int, x0: int, y0: int, x1: int, y1: int,
          color: tuple[int, int, int]) -> None:
    dx, sx = abs(x1 - x0), 1 if x0 < x1 else -1
    dy, sy = -abs(y1 - y0), 1 if y0 < y1 else -1
    error = dx + dy
    while True:
        if 0 <= x0 < width and 0 <= y0 < height:
            offset = (y0 * width + x0) * 3
            canvas[offset:offset + 3] = bytes(color)
        if x0 == x1 and y0 == y1:
            break
        doubled = 2 * error
        if doubled >= dy:
            error += dy
            x0 += sx
        if doubled <= dx:
            error += dx
            y0 += sy


def write_diagnostic_png(
    path: Path,
    samples: array,
    sample_rate: int,
    channels: int,
    event_start: float,
    event_end: float,
    target_start: float,
    target_end: float,
) -> None:
    width, height = 1000, 360
    canvas = bytearray([248, 249, 251] * width * height)
    duration = max(len(samples) / max(sample_rate * channels, 1), 1.0e-6)

    def xpos(seconds: float) -> int:
        return max(0, min(width - 1, int(seconds / duration * (width - 1))))

    event_x0, event_x1 = xpos(event_start), xpos(event_end)
    for y in range(height):
        for x in range(event_x0, event_x1 + 1):
            offset = (y * width + x) * 3
            canvas[offset:offset + 3] = bytes((233, 244, 255))
    for y in range(height):
        for x in (xpos(target_start), xpos(target_end)):
            offset = (y * width + x) * 3
            canvas[offset:offset + 3] = bytes((210, 55, 55))

    mono = list(samples[::channels])
    bucket = max(len(mono) // width, 1)
    previous_y = 110
    energy_points: list[int] = []
    for x in range(width):
        segment = mono[x * bucket:min((x + 1) * bucket, len(mono))]
        if not segment:
            amplitude = 0.0
            energy = 0.0
        else:
            amplitude = sum(segment) / len(segment) / 32768.0
            energy = math.sqrt(sum((value / 32768.0) ** 2 for value in segment) / len(segment))
        y = int(110 - amplitude * 90)
        _line(canvas, width, height, max(0, x - 1), previous_y, x, y, (35, 83, 135))
        previous_y = y
        energy_points.append(int(330 - min(energy * 6.0, 1.0) * 100))
    for x in range(1, len(energy_points)):
        _line(canvas, width, height, x - 1, energy_points[x - 1], x, energy_points[x], (35, 145, 90))
    _line(canvas, width, height, 0, 110, width - 1, 110, (170, 175, 180))
    _line(canvas, width, height, 0, 330, width - 1, 330, (170, 175, 180))

    raw = b"".join(b"\x00" + bytes(canvas[y * width * 3:(y + 1) * width * 3]) for y in range(height))
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + _chunk(b"IDAT", zlib.compress(raw, 9))
        + _chunk(b"IEND", b"")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)
