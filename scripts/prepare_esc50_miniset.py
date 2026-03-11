#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prepare a minimal ESC-50 / ESC-10 training subset for a small AudioCraft-style project.

What this script does:
1. Read ESC-50 metadata from: <src_root>/meta/esc50.csv
2. Filter ESC-10 subset by default
3. Build a balanced train/valid split by category
4. Load WAV, convert to mono, resample to target sample rate, normalize, pad/crop to fixed duration
5. Save processed WAV + same-basename JSON sidecar metadata
6. Write a split summary JSON for quick inspection

Expected source layout:
<src_root>/
  audio/
    *.wav
  meta/
    esc50.csv

Output layout:
<dst_root>/
  train/
    <basename>.wav
    <basename>.json
  valid/
    <basename>.wav
    <basename>.json
  split_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import soundfile as sf
import torch
import torchaudio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a minimal ESC-50 / ESC-10 dataset for training."
    )
    parser.add_argument(
        "--src-root",
        type=str,
        required=True,
        help="Root directory of ESC-50 raw data, containing audio/ and meta/esc50.csv",
    )
    parser.add_argument(
        "--dst-root",
        type=str,
        required=True,
        help="Output directory for processed dataset, e.g. data/processed/esc10_miniset",
    )
    parser.add_argument(
        "--train-per-class",
        type=int,
        default=8,
        help="Number of training samples per class",
    )
    parser.add_argument(
        "--valid-per-class",
        type=int,
        default=2,
        help="Number of validation samples per class",
    )
    parser.add_argument(
        "--valid-fold",
        type=int,
        default=5,
        choices=[1, 2, 3, 4, 5],
        help="ESC-50 fold to draw validation samples from",
    )
    parser.add_argument(
        "--target-sr",
        type=int,
        default=16000,
        help="Target sample rate for processed WAV",
    )
    parser.add_argument(
        "--clip-seconds",
        type=float,
        default=5.0,
        help="Fixed duration (seconds) after pad/crop",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--use-esc10",
        action="store_true",
        default=True,
        help="Use ESC-10 subset only (default: True)",
    )
    parser.add_argument(
        "--no-use-esc10",
        dest="use_esc10",
        action="store_false",
        help="Use full ESC-50 instead of ESC-10",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing processed files",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def str_to_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def read_esc50_metadata(csv_path: Path, use_esc10: bool) -> List[Dict]:
    rows: List[Dict] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["fold"] = int(row["fold"])
            row["target"] = int(row["target"])
            row["esc10"] = str_to_bool(row["esc10"])
            if use_esc10 and not row["esc10"]:
                continue
            rows.append(row)
    if not rows:
        raise RuntimeError(f"No rows found after filtering metadata: {csv_path}")
    return rows


def group_by_category(rows: List[Dict]) -> Dict[str, List[Dict]]:
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        grouped[row["category"]].append(row)
    return grouped


def select_split(
    grouped: Dict[str, List[Dict]],
    train_per_class: int,
    valid_per_class: int,
    valid_fold: int,
    seed: int,
) -> Tuple[List[Dict], List[Dict]]:
    rng = random.Random(seed)
    train_rows: List[Dict] = []
    valid_rows: List[Dict] = []

    for category, items in sorted(grouped.items()):
        valid_pool = [x for x in items if x["fold"] == valid_fold]
        train_pool = [x for x in items if x["fold"] != valid_fold]

        rng.shuffle(valid_pool)
        rng.shuffle(train_pool)

        if len(valid_pool) < valid_per_class:
            raise ValueError(
                f"Category '{category}' has only {len(valid_pool)} items in fold={valid_fold}, "
                f"but valid_per_class={valid_per_class}."
            )
        if len(train_pool) < train_per_class:
            raise ValueError(
                f"Category '{category}' has only {len(train_pool)} items outside fold={valid_fold}, "
                f"but train_per_class={train_per_class}."
            )

        valid_rows.extend(valid_pool[:valid_per_class])
        train_rows.extend(train_pool[:train_per_class])

    return train_rows, valid_rows


def make_text_prompt(category: str) -> str:
    # Minimal text metadata for future conditional training / retrieval / logging.
    human = category.replace("_", " ").replace("-", " ").strip()
    return f"{human} sound"


def load_process_audio(
    wav_path: Path,
    target_sr: int,
    clip_seconds: float,
) -> torch.Tensor:
    waveform, sr = torchaudio.load(str(wav_path))  # [C, T]

    # mono
    if waveform.dim() != 2:
        raise ValueError(f"Unexpected waveform shape for {wav_path}: {tuple(waveform.shape)}")
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # resample
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)

    # peak normalize to avoid clipping / scale mismatch
    max_abs = waveform.abs().max()
    if torch.isfinite(max_abs) and max_abs > 0:
        waveform = 0.99 * waveform / max_abs

    # pad / crop to exact length
    target_num_samples = int(round(target_sr * clip_seconds))
    current_num_samples = waveform.size(1)

    if current_num_samples < target_num_samples:
        pad_amount = target_num_samples - current_num_samples
        waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
    elif current_num_samples > target_num_samples:
        waveform = waveform[:, :target_num_samples]

    return waveform.contiguous()


def save_audio(waveform: torch.Tensor, wav_path: Path, sample_rate: int) -> None:
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    array = waveform.squeeze(0).cpu().numpy()
    sf.write(str(wav_path), array, sample_rate, subtype="PCM_16")


def save_sidecar_json(
    json_path: Path,
    row: Dict,
    target_sr: int,
    clip_seconds: float,
) -> None:
    metadata = {
        "text": make_text_prompt(row["category"]),
        "label": row["category"],
        "category": row["category"],
        "target": row["target"],
        "esc10": row["esc10"],
        "fold": row["fold"],
        "orig_filename": row["filename"],
        "src_file": row.get("src_file", ""),
        "take": row.get("take", ""),
        "sample_rate": target_sr,
        "duration": clip_seconds,
        "source_dataset": "ESC-50",
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def process_split(
    rows: List[Dict],
    split_name: str,
    src_audio_dir: Path,
    dst_root: Path,
    target_sr: int,
    clip_seconds: float,
    overwrite: bool,
) -> Dict:
    split_dir = dst_root / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    per_category_count: Dict[str, int] = defaultdict(int)
    written = 0

    for row in rows:
        src_wav = src_audio_dir / row["filename"]
        if not src_wav.exists():
            raise FileNotFoundError(f"Missing source audio: {src_wav}")

        base = Path(row["filename"]).stem
        out_wav = split_dir / f"{base}.wav"
        out_json = split_dir / f"{base}.json"

        if out_wav.exists() and out_json.exists() and not overwrite:
            per_category_count[row["category"]] += 1
            written += 1
            continue

        waveform = load_process_audio(
            wav_path=src_wav,
            target_sr=target_sr,
            clip_seconds=clip_seconds,
        )
        save_audio(waveform, out_wav, target_sr)
        save_sidecar_json(out_json, row, target_sr, clip_seconds)

        per_category_count[row["category"]] += 1
        written += 1

    return {
        "split": split_name,
        "num_files": written,
        "per_category_count": dict(sorted(per_category_count.items())),
        "output_dir": str(split_dir),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)

    src_audio_dir = src_root / "audio"
    src_meta_csv = src_root / "meta" / "esc50.csv"

    if not src_audio_dir.exists():
        raise FileNotFoundError(f"Expected source audio directory not found: {src_audio_dir}")
    if not src_meta_csv.exists():
        raise FileNotFoundError(f"Expected metadata CSV not found: {src_meta_csv}")

    rows = read_esc50_metadata(src_meta_csv, use_esc10=args.use_esc10)
    grouped = group_by_category(rows)

    train_rows, valid_rows = select_split(
        grouped=grouped,
        train_per_class=args.train_per_class,
        valid_per_class=args.valid_per_class,
        valid_fold=args.valid_fold,
        seed=args.seed,
    )

    train_summary = process_split(
        rows=train_rows,
        split_name="train",
        src_audio_dir=src_audio_dir,
        dst_root=dst_root,
        target_sr=args.target_sr,
        clip_seconds=args.clip_seconds,
        overwrite=args.overwrite,
    )
    valid_summary = process_split(
        rows=valid_rows,
        split_name="valid",
        src_audio_dir=src_audio_dir,
        dst_root=dst_root,
        target_sr=args.target_sr,
        clip_seconds=args.clip_seconds,
        overwrite=args.overwrite,
    )

    final_summary = {
        "source_root": str(src_root),
        "output_root": str(dst_root),
        "use_esc10": args.use_esc10,
        "train_per_class": args.train_per_class,
        "valid_per_class": args.valid_per_class,
        "valid_fold": args.valid_fold,
        "target_sr": args.target_sr,
        "clip_seconds": args.clip_seconds,
        "seed": args.seed,
        "train": train_summary,
        "valid": valid_summary,
    }

    summary_path = dst_root / "split_summary.json"
    dst_root.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(final_summary, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("Done.")
    print(json.dumps(final_summary, ensure_ascii=False, indent=2))
    print("=" * 80)


if __name__ == "__main__":
    main()