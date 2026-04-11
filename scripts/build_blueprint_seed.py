#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


INPUT_PATH = Path("artifacts/logs/week03_video_probe_smoke.json")
OUTPUT_PATH = Path("artifacts/manifests/seed_0001.json")


def load_probe(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"probe json not found: {path}")

    raw_text = path.read_text(encoding="utf-8")

    # 容忍日志文件前面混入 warning / stderr 文本：
    # 从第一个 "{" 开始截取，按 JSON 主体解析
    json_start = raw_text.find("{")
    if json_start == -1:
        raise ValueError(f"no json object found in: {path}")

    cleaned_text = raw_text[json_start:].strip()

    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"failed to decode cleaned json body from {path}: {e}"
        ) from e


def build_seed(probe: dict) -> dict:
    stream_types = probe.get("stream_types", [])
    has_audio = "audio" in stream_types
    has_video = "video" in stream_types

    seed = {
        "blueprint_version": "v0.1",
        "blueprint_id": "seed_0001",
        "task_type": "video_audio_smoke",
        "source_video": {
            "path": probe.get("path"),
            "duration_sec": probe.get("duration_sec_ffprobe"),
            "has_audio": has_audio,
            "has_video": has_video,
        },
        "media_probe": {
            "log_ref": str(INPUT_PATH),
            "format_name": probe.get("format_name"),
            "size_bytes": probe.get("size_bytes"),
            "bit_rate": probe.get("bit_rate"),
            "audio": {
                "codec": probe.get("audio_codec"),
                "sample_rate": probe.get("audio_sr_ffprobe"),
                "channels": probe.get("audio_channels_ffprobe"),
            },
            "video": {
                "codec": probe.get("video_codec"),
                "width": probe.get("width_ffprobe"),
                "height": probe.get("height_ffprobe"),
                "fps": probe.get("video_fps_ffprobe"),
            },
        },
        "timeline": [],
        "output_stems": [],
        "metadata": {
            "segment_id": "seg_0001_full",
            "generator_name": "seed_builder_v0",
            "output_artifact_dir": "artifacts/wav/seed_0001",
            "source_stage": "week05_seed_mapping",
        },
    }
    return seed


def validate_seed(seed: dict) -> None:
    required_top_level = [
        "blueprint_version",
        "blueprint_id",
        "task_type",
        "source_video",
        "media_probe",
        "timeline",
        "output_stems",
        "metadata",
    ]
    for key in required_top_level:
        if key not in seed:
            raise ValueError(f"missing top-level field: {key}")

    if not seed["source_video"].get("path"):
        raise ValueError("source_video.path is empty")

    if seed["media_probe"].get("log_ref") != str(INPUT_PATH):
        raise ValueError("media_probe.log_ref mismatch")

    if not isinstance(seed["timeline"], list):
        raise TypeError("timeline must be a list")

    if not isinstance(seed["output_stems"], list):
        raise TypeError("output_stems must be a list")


def main() -> None:
    probe = load_probe(INPUT_PATH)
    seed = build_seed(probe)
    validate_seed(seed)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(seed, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"[OK] wrote seed to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
