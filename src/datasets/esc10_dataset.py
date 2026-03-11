#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torchaudio
from torch.utils.data import Dataset


class Esc10SoundDataset(Dataset):
    """
    Minimal manifest-driven audio dataset for ESC-10/ESC-50 style processed data.

    Expected manifest line format:
    {
        "audio_path": "data/processed/esc10_miniset/train/xxx.wav",
        "meta_path": "data/processed/esc10_miniset/train/xxx.json",
        "sample_rate": 16000,
        "channels": 1,
        "num_frames": 80000,
        "duration": 5.0
    }

    Expected sidecar JSON format (example):
    {
        "text": "dog bark sound",
        "label": "dog",
        "category": "dog",
        "sample_rate": 16000,
        "duration": 5.0
    }
    """

    def __init__(
        self,
        manifest_path: str,
        sample_rate: int = 16000,
        clip_seconds: float = 5.0,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        n_mels: int = 64,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        center: bool = True,
        power: float = 2.0,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()

        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        self.sample_rate = int(sample_rate)
        self.clip_seconds = float(clip_seconds)
        self.num_samples = int(round(self.sample_rate * self.clip_seconds))
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)
        self.n_mels = int(n_mels)
        self.f_min = float(f_min)
        self.f_max = f_max if f_max is None else float(f_max)
        self.center = bool(center)
        self.power = float(power)
        self.eps = float(eps)

        self.records = self._load_manifest(self.manifest_path)

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            f_min=self.f_min,
            f_max=self.f_max,
            pad=0,
            n_mels=self.n_mels,
            window_fn=torch.hann_window,
            power=self.power,
            center=self.center,
            normalized=False,
            norm="slaney",
            mel_scale="htk",
        )

    def _load_manifest(self, manifest_path: Path) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        with manifest_path.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Invalid JSON in manifest at line {line_idx}: {manifest_path}"
                    ) from e

                if "audio_path" not in record:
                    raise KeyError(
                        f"Missing 'audio_path' in manifest line {line_idx}: {manifest_path}"
                    )
                if "meta_path" not in record:
                    raise KeyError(
                        f"Missing 'meta_path' in manifest line {line_idx}: {manifest_path}"
                    )

                records.append(record)

        if not records:
            raise RuntimeError(f"No valid records found in manifest: {manifest_path}")

        return records

    @staticmethod
    def _resolve_path(path_str: str) -> Path:
        path = Path(path_str)
        if path.is_absolute():
            return path
        return (Path.cwd() / path).resolve()

    @staticmethod
    def _ensure_mono(waveform: torch.Tensor) -> torch.Tensor:
        """
        Input: [C, T]
        Output: [1, T]
        """
        if waveform.dim() != 2:
            raise ValueError(f"Expected waveform shape [C, T], got {tuple(waveform.shape)}")
        if waveform.size(0) == 1:
            return waveform
        return waveform.mean(dim=0, keepdim=True)

    def _resample_if_needed(self, waveform: torch.Tensor, orig_sr: int) -> torch.Tensor:
        if int(orig_sr) == self.sample_rate:
            return waveform
        return torchaudio.functional.resample(
            waveform,
            orig_freq=int(orig_sr),
            new_freq=self.sample_rate,
        )

    def _fix_length(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Force waveform to [1, self.num_samples]
        """
        cur_len = waveform.size(-1)
        if cur_len == self.num_samples:
            return waveform

        if cur_len < self.num_samples:
            pad_amount = self.num_samples - cur_len
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
            return waveform

        return waveform[..., : self.num_samples]

    def _load_sidecar(self, meta_path: Path) -> Dict[str, Any]:
        if not meta_path.exists():
            raise FileNotFoundError(f"Sidecar json not found: {meta_path}")
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        if not isinstance(meta, dict):
            raise ValueError(f"Sidecar json must be a dict: {meta_path}")
        return meta

    def _compute_log_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Input waveform: [1, T]
        Output mel: [1, n_mels, frames]
        """
        mel = self.mel_transform(waveform)  # [1, n_mels, frames]
        mel = torch.clamp(mel, min=self.eps).log()
        return mel

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]

        audio_path = self._resolve_path(record["audio_path"])
        meta_path = self._resolve_path(record["meta_path"])

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        waveform, sr = torchaudio.load(str(audio_path))  # [C, T], float32 by default
        waveform = self._ensure_mono(waveform)
        waveform = self._resample_if_needed(waveform, sr)
        waveform = self._fix_length(waveform)

        # Optional safety normalization
        max_abs = waveform.abs().max()
        if torch.isfinite(max_abs) and max_abs > 0:
            waveform = waveform / max_abs

        mel = self._compute_log_mel(waveform)

        meta = self._load_sidecar(meta_path)
        text = str(meta.get("text", ""))
        label = str(meta.get("label", meta.get("category", "")))

        sample: Dict[str, Any] = {
            "waveform": waveform,                 # [1, 80000]
            "mel": mel,                           # [1, 64, frames]
            "text": text,                         # str
            "label": label,                       # str
            "path": str(audio_path),              # str
            "meta": meta,                         # dict
        }
        return sample


if __name__ == "__main__":
    # Minimal manual sanity check:
    # python src/esc10_dataset.py data/manifests/esc10_miniset_train.jsonl
    import sys

    if len(sys.argv) < 2:
        print("Usage: python src/esc10_dataset.py <manifest_path>")
        raise SystemExit(1)

    manifest_path = sys.argv[1]
    dataset = Esc10SoundDataset(manifest_path=manifest_path)

    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"path      : {sample['path']}")
    print(f"text      : {sample['text']}")
    print(f"label     : {sample['label']}")
    print(f"waveform  : {tuple(sample['waveform'].shape)}")
    print(f"mel       : {tuple(sample['mel'].shape)}")