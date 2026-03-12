#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import torch

# File: src/export/export_onnx.py
# Repo root should be two levels up from this file.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.engine.train import load_yaml, build_model, build_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export MinimalMelAutoencoder checkpoint to ONNX."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train/baseline.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="artifacts/experiments/baseline_v1/checkpoints/best.pt",
        help="Path to checkpoint (.pt).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/onnx/baseline_v1.onnx",
        help="Path to output ONNX file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Export device. Use cpu first for stability.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="",
        help="Optional manifest path. If empty, use config.data.valid_manifest first.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Sample index used to build a real example input.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run onnx.load + onnx.checker.check_model after export.",
    )
    return parser.parse_args()


def resolve_device(device_str: str) -> torch.device:
    if device_str == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device(device_str)


def load_checkpoint_state_dict(
    checkpoint_path: Path, device: torch.device
) -> Dict[str, torch.Tensor]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(str(checkpoint_path), map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]

    if isinstance(checkpoint, dict):
        # Fallback: plain state_dict
        return checkpoint

    raise TypeError(
        "Unsupported checkpoint format. Expected a dict or a dict containing "
        "'model_state_dict'."
    )


def build_example_input(
    cfg: Dict[str, Any],
    device: torch.device,
    manifest_path: str,
    sample_index: int,
) -> Tuple[torch.Tensor, str]:
    data_cfg = cfg["data"]

    chosen_manifest = manifest_path.strip()
    if not chosen_manifest:
        chosen_manifest = str(data_cfg.get("valid_manifest", "")).strip()
    if not chosen_manifest:
        chosen_manifest = str(data_cfg.get("train_manifest", "")).strip()
    if not chosen_manifest:
        raise ValueError("No manifest path found in args or config.")

    dataset = build_dataset(chosen_manifest, cfg)
    if not (0 <= sample_index < len(dataset)):
        raise IndexError(
            f"sample_index={sample_index} out of range for dataset size={len(dataset)}"
        )

    sample = dataset[sample_index]
    # Dataset returns mel of shape [1, n_mels, frames]
    mel = sample["mel"].unsqueeze(0).to(device=device, dtype=torch.float32)
    source = str(sample.get("path", f"{chosen_manifest}[{sample_index}]"))
    return mel, source


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    cfg = load_yaml(args.config)
    model = build_model(cfg).to(device)

    state_dict = load_checkpoint_state_dict(Path(args.checkpoint), device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    example_input, example_source = build_example_input(
        cfg=cfg,
        device=device,
        manifest_path=args.manifest,
        sample_index=args.sample_index,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        example_output = model(example_input)

    torch.onnx.export(
        model,
        (example_input,),
        str(output_path),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["mel"],
        output_names=["recon_mel"],
        dynamic_axes=None,  # 先固定输入，先把链路打通
    )

    print("=" * 80)
    print("[OK] ONNX export finished.")
    print(f"[Info] Config      : {args.config}")
    print(f"[Info] Checkpoint  : {args.checkpoint}")
    print(f"[Info] Output ONNX : {output_path}")
    print(f"[Info] Device      : {device}")
    print(f"[Info] Opset       : {args.opset}")
    print(f"[Info] Example src : {example_source}")
    print(f"[Info] Input name  : mel")
    print(f"[Info] Output name : recon_mel")
    print(f"[Info] Input shape : {tuple(example_input.shape)}")
    print(f"[Info] Output shape: {tuple(example_output.shape)}")
    print("=" * 80)

    if args.verify:
        try:
            import onnx

            onnx_model = onnx.load(str(output_path))
            onnx.checker.check_model(onnx_model)
            print("[OK] onnx.checker.check_model passed.")
        except ImportError:
            print("[Warn] onnx package is not installed. Install it with: pip install onnx")


if __name__ == "__main__":
    main()