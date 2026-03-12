#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch

# File: src/infer/infer_ort.py
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.engine.train import load_yaml, build_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single ONNX Runtime inference for baseline_v1."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train/baseline.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--onnx",
        type=str,
        default="artifacts/onnx/baseline_v1.onnx",
        help="Path to ONNX model.",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="CPUExecutionProvider",
        help="ORT execution provider, e.g. CPUExecutionProvider / CUDAExecutionProvider.",
    )
    parser.add_argument(
        "--input-mode",
        type=str,
        default="real",
        choices=["real", "random"],
        help="Use a real sample from manifest or a random tensor.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="",
        help="Optional manifest path. If empty, prefer valid_manifest in config.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Sample index used when --input-mode=real.",
    )
    parser.add_argument(
        "--input-shape",
        type=str,
        default="1,1,64,313",
        help="4D shape used when --input-mode=random, e.g. 1,1,64,313",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when --input-mode=random.",
    )
    parser.add_argument(
        "--intra-op-threads",
        type=int,
        default=0,
        help="Set ORT intra_op_num_threads if > 0.",
    )
    parser.add_argument(
        "--inter-op-threads",
        type=int,
        default=0,
        help="Set ORT inter_op_num_threads if > 0.",
    )
    parser.add_argument(
        "--save-output",
        type=str,
        default="",
        help="Optional .npy path to save ORT output.",
    )
    parser.add_argument(
        "--enable-profiling",
        action="store_true",
        help="Enable ORT profiling.",
    )
    return parser.parse_args()


def parse_shape(shape_str: str) -> Tuple[int, int, int, int]:
    try:
        shape = tuple(int(x.strip()) for x in shape_str.split(","))
    except ValueError as exc:
        raise ValueError(f"Invalid --input-shape: {shape_str}") from exc

    if len(shape) != 4:
        raise ValueError(f"--input-shape must be 4D, got {shape}")
    if any(x <= 0 for x in shape):
        raise ValueError(f"All dimensions in --input-shape must be > 0, got {shape}")
    return shape  # type: ignore[return-value]


def resolve_manifest(cfg: Dict[str, Any], manifest_path: str) -> str:
    data_cfg = cfg["data"]
    chosen_manifest = manifest_path.strip()

    if not chosen_manifest:
        chosen_manifest = str(data_cfg.get("valid_manifest", "")).strip()
    if not chosen_manifest:
        chosen_manifest = str(data_cfg.get("train_manifest", "")).strip()
    if not chosen_manifest:
        raise ValueError("No manifest path found in args or config.")

    return chosen_manifest


def build_real_input(
    cfg: Dict[str, Any],
    manifest_path: str,
    sample_index: int,
) -> Tuple[np.ndarray, str]:
    chosen_manifest = resolve_manifest(cfg, manifest_path)
    dataset = build_dataset(chosen_manifest, cfg)

    if not (0 <= sample_index < len(dataset)):
        raise IndexError(
            f"sample_index={sample_index} out of range for dataset size={len(dataset)}"
        )

    sample = dataset[sample_index]
    # dataset['mel']: [1, n_mels, frames]
    mel = sample["mel"].unsqueeze(0).to(dtype=torch.float32)  # [1, 1, n_mels, frames]
    input_array = np.ascontiguousarray(
        mel.detach().cpu().numpy().astype(np.float32, copy=False)
    )
    source = str(sample.get("path", f"{chosen_manifest}[{sample_index}]"))
    return input_array, source


def build_random_input(
    input_shape: Tuple[int, int, int, int],
    seed: int,
) -> Tuple[np.ndarray, str]:
    rng = np.random.default_rng(seed)
    input_array = rng.standard_normal(size=input_shape).astype(np.float32, copy=False)
    input_array = np.ascontiguousarray(input_array)
    source = f"random(shape={input_shape}, seed={seed})"
    return input_array, source


def create_input_array(
    input_mode: str,
    config_path: str,
    manifest_path: str,
    sample_index: int,
    input_shape: str,
    seed: int,
) -> Tuple[np.ndarray, str]:
    if input_mode == "real":
        cfg = load_yaml(config_path)
        return build_real_input(cfg=cfg, manifest_path=manifest_path, sample_index=sample_index)

    return build_random_input(input_shape=parse_shape(input_shape), seed=seed)


def build_session(
    onnx_path: Path,
    provider: str,
    intra_op_threads: int = 0,
    inter_op_threads: int = 0,
    enable_profiling: bool = False,
):
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError(
            "onnxruntime is not installed. Install with: pip install onnxruntime"
        ) from exc

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    if intra_op_threads > 0:
        sess_options.intra_op_num_threads = intra_op_threads
    if inter_op_threads > 0:
        sess_options.inter_op_num_threads = inter_op_threads

    sess_options.enable_profiling = enable_profiling

    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=sess_options,
        providers=[provider],
    )
    return session


def main() -> None:
    args = parse_args()

    input_array, input_source = create_input_array(
        input_mode=args.input_mode,
        config_path=args.config,
        manifest_path=args.manifest,
        sample_index=args.sample_index,
        input_shape=args.input_shape,
        seed=args.seed,
    )

    session = build_session(
        onnx_path=Path(args.onnx),
        provider=args.provider,
        intra_op_threads=args.intra_op_threads,
        inter_op_threads=args.inter_op_threads,
        enable_profiling=args.enable_profiling,
    )

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    output = session.run([output_name], {input_name: input_array})[0]
    output = np.asarray(output, dtype=np.float32)

    if args.save_output.strip():
        save_path = Path(args.save_output)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(save_path), output)

    print("=" * 80)
    print("[OK] ORT inference finished.")
    print(f"[Info] ONNX path        : {args.onnx}")
    print(f"[Info] Provider req     : {args.provider}")
    print(f"[Info] Provider actual  : {session.get_providers()}")
    print(f"[Info] Input source     : {input_source}")
    print(f"[Info] Input name       : {input_name}")
    print(f"[Info] Output name      : {output_name}")
    print(f"[Info] Input shape      : {tuple(input_array.shape)}")
    print(f"[Info] Output shape     : {tuple(output.shape)}")
    print(f"[Info] Output min / max : {output.min():.6f} / {output.max():.6f}")
    print(f"[Info] Intra threads    : {args.intra_op_threads}")
    print(f"[Info] Inter threads    : {args.inter_op_threads}")
    if args.save_output.strip():
        print(f"[Info] Saved output     : {args.save_output}")
    if args.enable_profiling:
        profile_path = session.end_profiling()
        print(f"[Info] ORT profile      : {profile_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()  