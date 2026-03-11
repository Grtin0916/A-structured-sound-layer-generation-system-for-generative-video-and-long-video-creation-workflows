#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a minimal JSONL manifest from a folder of processed audio files.

Expected input layout:
<input_dir>/
  xxx.wav
  xxx.json
  yyy.wav
  yyy.json
  ...

Output:
One JSON object per line, e.g.
{"audio_path":"data/processed/esc10_miniset/train/1-100032-A-0.wav",
 "meta_path":"data/processed/esc10_miniset/train/1-100032-A-0.json",
 "sample_rate":16000,
 "channels":1,
 "num_frames":80000,
 "duration":5.0}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List

import soundfile as sf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a JSONL manifest from a folder containing wav + sidecar json files."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing processed audio files and same-basename .json sidecars.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Output manifest path, e.g. data/manifests/esc10_miniset_train.jsonl",
    )
    parser.add_argument(
        "--audio-ext",
        type=str,
        default=".wav",
        help="Audio file extension to scan. Default: .wav",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan subdirectories under --input-dir",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="If enabled, missing/invalid samples raise an error instead of being skipped.",
    )
    return parser.parse_args()


def to_portable_path(path: Path) -> str:
    """
    Prefer repo-relative paths if possible; otherwise fall back to absolute paths.
    """
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def list_audio_files(input_dir: Path, audio_ext: str, recursive: bool) -> List[Path]:
    pattern = f"*{audio_ext}"
    files: Iterable[Path]
    if recursive:
        files = input_dir.rglob(pattern)
    else:
        files = input_dir.glob(pattern)
    return sorted([p for p in files if p.is_file()])


def inspect_audio(audio_path: Path) -> dict:
    info = sf.info(str(audio_path))
    return {
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "num_frames": int(info.frames),
        "duration": float(info.duration),
    }


def build_record(audio_path: Path, meta_path: Path) -> dict:
    audio_info = inspect_audio(audio_path)
    record = {
        "audio_path": to_portable_path(audio_path),
        "meta_path": to_portable_path(meta_path),
        "sample_rate": audio_info["sample_rate"],
        "channels": audio_info["channels"],
        "num_frames": audio_info["num_frames"],
        "duration": round(audio_info["duration"], 6),
    }
    return record


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output_path)
    audio_ext = args.audio_ext if args.audio_ext.startswith(".") else f".{args.audio_ext}"

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    audio_files = list_audio_files(input_dir, audio_ext, args.recursive)
    if not audio_files:
        raise RuntimeError(f"No audio files found in: {input_dir} (ext={audio_ext})")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    skipped = []

    for audio_path in audio_files:
        meta_path = audio_path.with_suffix(".json")

        if not meta_path.exists():
            msg = f"Missing sidecar json for audio: {audio_path}"
            if args.strict:
                raise FileNotFoundError(msg)
            skipped.append(msg)
            continue

        try:
            record = build_record(audio_path, meta_path)
            records.append(record)
        except Exception as e:
            msg = f"Failed to inspect audio {audio_path}: {e}"
            if args.strict:
                raise RuntimeError(msg) from e
            skipped.append(msg)

    if not records:
        raise RuntimeError("No valid records were built. Check your input directory and sidecar json files.")

    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("=" * 80)
    print("Manifest build finished.")
    print(f"Input dir   : {input_dir}")
    print(f"Output path : {output_path}")
    print(f"Audio files : {len(audio_files)}")
    print(f"Written     : {len(records)}")
    print(f"Skipped     : {len(skipped)}")
    if skipped:
        print("-" * 80)
        print("Skipped samples:")
        for msg in skipped[:20]:
            print(msg)
        if len(skipped) > 20:
            print(f"... and {len(skipped) - 20} more")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)