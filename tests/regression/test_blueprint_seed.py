from __future__ import annotations

import json
from pathlib import Path

import pytest


SEED_PATH = Path("artifacts/manifests/seed_0001.json")
PROBE_LOG_PATH = Path("artifacts/logs/week03_video_probe_smoke.json")


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_probe_with_leading_text(path: Path) -> dict:
    raw_text = path.read_text(encoding="utf-8")
    json_start = raw_text.find("{")
    if json_start == -1:
        raise ValueError(f"no json object found in: {path}")
    return json.loads(raw_text[json_start:].strip())


def test_seed_file_exists_and_is_json_loadable() -> None:
    assert SEED_PATH.exists(), f"missing seed file: {SEED_PATH}"
    seed = load_json(SEED_PATH)
    assert isinstance(seed, dict)
    assert seed["blueprint_id"] == "seed_0001"


def test_seed_top_level_schema_is_complete() -> None:
    seed = load_json(SEED_PATH)

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
        assert key in seed, f"missing top-level field: {key}"

    assert seed["blueprint_version"] == "v0.1"
    assert seed["task_type"] == "video_audio_smoke"
    assert isinstance(seed["timeline"], list)
    assert isinstance(seed["output_stems"], list)


def test_seed_matches_probe_core_fields() -> None:
    seed = load_json(SEED_PATH)
    probe = load_probe_with_leading_text(PROBE_LOG_PATH)

    assert seed["source_video"]["path"] == probe["path"]
    assert seed["source_video"]["duration_sec"] == pytest.approx(
        probe["duration_sec_ffprobe"]
    )
    assert seed["source_video"]["has_audio"] is ("audio" in probe["stream_types"])
    assert seed["source_video"]["has_video"] is ("video" in probe["stream_types"])

    assert seed["media_probe"]["log_ref"] == str(PROBE_LOG_PATH)
    assert seed["media_probe"]["format_name"] == probe["format_name"]
    assert seed["media_probe"]["size_bytes"] == probe["size_bytes"]
    assert seed["media_probe"]["bit_rate"] == probe["bit_rate"]

    assert seed["media_probe"]["audio"]["codec"] == probe["audio_codec"]
    assert seed["media_probe"]["audio"]["sample_rate"] == probe["audio_sr_ffprobe"]
    assert seed["media_probe"]["audio"]["channels"] == probe["audio_channels_ffprobe"]

    assert seed["media_probe"]["video"]["codec"] == probe["video_codec"]
    assert seed["media_probe"]["video"]["width"] == probe["width_ffprobe"]
    assert seed["media_probe"]["video"]["height"] == probe["height_ffprobe"]
    assert seed["media_probe"]["video"]["fps"] == pytest.approx(
        probe["video_fps_ffprobe"]
    )


def test_seed_metadata_fields_are_fixed_for_week05() -> None:
    seed = load_json(SEED_PATH)
    metadata = seed["metadata"]

    assert metadata["segment_id"] == "seg_0001_full"
    assert metadata["generator_name"] == "seed_builder_v0"
    assert metadata["output_artifact_dir"] == "artifacts/wav/seed_0001"
    assert metadata["source_stage"] == "week05_seed_mapping"
