#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch

# File: src/infer/compare_pt_onnx.py
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.engine.train import load_yaml, build_model, build_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare PyTorch output with ONNX Runtime output."
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
        "--onnx",
        type=str,
        default="artifacts/onnx/baseline_v1.onnx",
        help="Path to exported ONNX model.",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="artifacts/logs/onnx_parity.md",
        help="Path to markdown report.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for PyTorch inference. Recommend cpu first.",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="CPUExecutionProvider",
        help="ONNX Runtime provider, e.g. CPUExecutionProvider / CUDAExecutionProvider",
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
        help="Sample index used to build the real input.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-4,
        help="Absolute tolerance for np.allclose.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-4,
        help="Relative tolerance for np.allclose.",
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
    mel = sample["mel"].unsqueeze(0).to(device=device, dtype=torch.float32)
    source = str(sample.get("path", f"{chosen_manifest}[{sample_index}]"))
    return mel, source


def run_pytorch(
    cfg: Dict[str, Any],
    checkpoint_path: Path,
    example_input: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    model = build_model(cfg).to(device)
    state_dict = load_checkpoint_state_dict(checkpoint_path, device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    with torch.no_grad():
        output = model(example_input)

    return output.detach().cpu().numpy().astype(np.float32, copy=False)


def run_onnx(
    onnx_path: Path,
    input_array: np.ndarray,
    provider: str,
) -> Tuple[np.ndarray, str, str, list[str]]:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError(
            "onnxruntime is not installed. Install with: pip install onnxruntime"
        ) from exc

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    sess = ort.InferenceSession(str(onnx_path), providers=[provider])
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    output = sess.run([output_name], {input_name: input_array})[0]
    output = np.asarray(output, dtype=np.float32)

    return output, input_name, output_name, sess.get_providers()


def compute_metrics(
    pt_output: np.ndarray,
    onnx_output: np.ndarray,
    atol: float,
    rtol: float,
) -> Dict[str, Any]:
    if pt_output.shape != onnx_output.shape:
        raise ValueError(
            f"Shape mismatch: pt={pt_output.shape}, onnx={onnx_output.shape}"
        )

    abs_diff = np.abs(pt_output - onnx_output)
    sq_diff = (pt_output - onnx_output) ** 2

    metrics = {
        "shape_equal": True,
        "pt_shape": tuple(pt_output.shape),
        "onnx_shape": tuple(onnx_output.shape),
        "mean_abs_error": float(abs_diff.mean()),
        "max_abs_error": float(abs_diff.max()),
        "mse": float(sq_diff.mean()),
        "allclose": bool(np.allclose(pt_output, onnx_output, atol=atol, rtol=rtol)),
        "atol": float(atol),
        "rtol": float(rtol),
        "pt_min": float(pt_output.min()),
        "pt_max": float(pt_output.max()),
        "onnx_min": float(onnx_output.min()),
        "onnx_max": float(onnx_output.max()),
    }
    return metrics


def write_report(
    report_path: Path,
    args: argparse.Namespace,
    sample_source: str,
    input_shape: Tuple[int, ...],
    input_name: str,
    output_name: str,
    providers: list[str],
    metrics: Dict[str, Any],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)

    passed = metrics["shape_equal"] and metrics["allclose"]
    status = "PASS" if passed else "CHECK"

    md = f"""# ONNX Parity Report

## Summary
- Status: **{status}**
- Config: `{args.config}`
- Checkpoint: `{args.checkpoint}`
- ONNX: `{args.onnx}`
- Report: `{args.report}`

## Input
- Sample source: `{sample_source}`
- Sample index: `{args.sample_index}`
- Input shape: `{input_shape}`
- ONNX input name: `{input_name}`
- ONNX output name: `{output_name}`

## Runtime
- PyTorch device: `{args.device}`
- ONNX Runtime provider request: `{args.provider}`
- ONNX Runtime provider actual: `{providers}`

## Metrics
- Shape equal: `{metrics["shape_equal"]}`
- PyTorch output shape: `{metrics["pt_shape"]}`
- ONNX output shape: `{metrics["onnx_shape"]}`
- Mean abs error: `{metrics["mean_abs_error"]:.10f}`
- Max abs error: `{metrics["max_abs_error"]:.10f}`
- MSE: `{metrics["mse"]:.10f}`
- Allclose: `{metrics["allclose"]}`
- atol: `{metrics["atol"]}`
- rtol: `{metrics["rtol"]}`

## Value range
- PyTorch min/max: `{metrics["pt_min"]:.10f}` / `{metrics["pt_max"]:.10f}`
- ONNX min/max: `{metrics["onnx_min"]:.10f}` / `{metrics["onnx_max"]:.10f}`

## Conclusion
当前样本下，PyTorch 与 ONNX Runtime 输出已经完成逐元素对齐检查。
如 `Allclose=True` 且误差量级很小，可认为周三主线的 parity check 基本通过。
"""

    report_path.write_text(md, encoding="utf-8")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    cfg = load_yaml(args.config)
    example_input, sample_source = build_example_input(
        cfg=cfg,
        device=device,
        manifest_path=args.manifest,
        sample_index=args.sample_index,
    )

    input_array = np.ascontiguousarray(
        example_input.detach().cpu().numpy().astype(np.float32, copy=False)
    )

    pt_output = run_pytorch(
        cfg=cfg,
        checkpoint_path=Path(args.checkpoint),
        example_input=example_input,
        device=device,
    )

    onnx_output, input_name, output_name, providers = run_onnx(
        onnx_path=Path(args.onnx),
        input_array=input_array,
        provider=args.provider,
    )

    metrics = compute_metrics(
        pt_output=pt_output,
        onnx_output=onnx_output,
        atol=args.atol,
        rtol=args.rtol,
    )

    write_report(
        report_path=Path(args.report),
        args=args,
        sample_source=sample_source,
        input_shape=tuple(example_input.shape),
        input_name=input_name,
        output_name=output_name,
        providers=providers,
        metrics=metrics,
    )

    print("=" * 80)
    print("[OK] PT vs ONNX comparison finished.")
    print(f"[Info] Sample source : {sample_source}")
    print(f"[Info] Input shape   : {tuple(example_input.shape)}")
    print(f"[Info] PT shape      : {metrics['pt_shape']}")
    print(f"[Info] ONNX shape    : {metrics['onnx_shape']}")
    print(f"[Info] Mean abs err  : {metrics['mean_abs_error']:.10f}")
    print(f"[Info] Max abs err   : {metrics['max_abs_error']:.10f}")
    print(f"[Info] MSE           : {metrics['mse']:.10f}")
    print(f"[Info] Allclose      : {metrics['allclose']}")
    print(f"[Info] atol / rtol   : {args.atol} / {args.rtol}")
    print(f"[Info] Provider      : {providers}")
    print(f"[Info] Report        : {args.report}")
    print("=" * 80)


if __name__ == "__main__":
    main()