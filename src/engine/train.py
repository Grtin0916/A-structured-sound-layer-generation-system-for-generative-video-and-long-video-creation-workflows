#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Make repo root importable when running: python src/engine/train.py
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.model import MinimalMelAutoencoder, reconstruction_loss, count_parameters  # noqa: E402

# Compatible with either:
# 1) src/datasets/esc10_dataset.py
# 2) src/esc10_dataset.py
try:
    from src.datasets.esc10_dataset import Esc10SoundDataset  # type: ignore  # noqa: E402
except Exception:
    from src.dataset import Esc10SoundDataset  # type: ignore  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train minimal mel autoencoder.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cuda / cpu",
    )
    return parser.parse_args()


def load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid YAML config: {path}")
    return cfg


def save_yaml(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)


def prepare_dirs(cfg: Dict[str, Any]) -> Dict[str, Path]:
    exp_name = cfg.get("experiment", {}).get("name", "baseline_v1")
    root_dir = Path(cfg.get("output", {}).get("root_dir", "artifacts/experiments")) / exp_name

    checkpoint_dir = root_dir / "checkpoints"
    tensorboard_dir = root_dir / "tensorboard"
    root_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    return {
        "root_dir": root_dir,
        "checkpoint_dir": checkpoint_dir,
        "tensorboard_dir": tensorboard_dir,
        "metrics_path": root_dir / "metrics.jsonl",
        "resolved_config_path": root_dir / "config_resolved.yaml",
    }


def build_dataset(manifest_path: str, cfg: Dict[str, Any]) -> Esc10SoundDataset:
    data_cfg = cfg["data"]
    dataset = Esc10SoundDataset(
        manifest_path=manifest_path,
        sample_rate=int(data_cfg.get("sample_rate", 16000)),
        clip_seconds=float(data_cfg.get("clip_seconds", 5.0)),
        n_fft=int(data_cfg.get("n_fft", 1024)),
        hop_length=int(data_cfg.get("hop_length", 256)),
        win_length=int(data_cfg.get("win_length", 1024)),
        n_mels=int(data_cfg.get("n_mels", 64)),
    )
    return dataset


def build_dataloaders(cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    train_manifest = cfg["data"]["train_manifest"]
    valid_manifest = cfg["data"]["valid_manifest"]

    train_dataset = build_dataset(train_manifest, cfg)
    valid_dataset = build_dataset(valid_manifest, cfg)

    batch_size = int(cfg["train"].get("batch_size", 8))
    num_workers = int(cfg["train"].get("num_workers", 0))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        persistent_workers=(num_workers > 0),
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        persistent_workers=(num_workers > 0),
    )
    return train_loader, valid_loader


def build_model(cfg: Dict[str, Any]) -> MinimalMelAutoencoder:
    model_cfg = cfg.get("model", {})
    model = MinimalMelAutoencoder(
        in_channels=int(model_cfg.get("in_channels", 1)),
        base_channels=int(model_cfg.get("base_channels", 32)),
        bottleneck_channels=int(model_cfg.get("bottleneck_channels", 128)),
    )
    return model


def normalize_for_tb(x: torch.Tensor) -> torch.Tensor:
    """
    Convert mel image to [1, H, W] in [0, 1] for TensorBoard.
    """
    x = x.detach().float().cpu()
    if x.ndim == 3 and x.size(0) == 1:
        # [1, H, W]
        pass
    elif x.ndim == 2:
        x = x.unsqueeze(0)
    else:
        raise ValueError(f"Unexpected tensor shape for TB image: {tuple(x.shape)}")

    x_min = x.min()
    x_max = x.max()
    denom = (x_max - x_min).clamp_min(1e-8)
    x = (x - x_min) / denom
    return x


def save_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    best_val_loss: float,
    cfg: Dict[str, Any],
) -> None:
    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": cfg,
    }
    torch.save(checkpoint, checkpoint_path)


def append_metrics(metrics_path: Path, payload: Dict[str, Any]) -> None:
    with metrics_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    mse_weight: float,
    grad_clip: float,
    writer: SummaryWriter,
    epoch: int,
    global_step: int,
    log_every_n_steps: int,
) -> Tuple[Dict[str, float], int]:
    model.train()

    total_loss = 0.0
    total_l1 = 0.0
    total_mse = 0.0
    num_batches = 0

    for step, batch in enumerate(loader):
        mel = batch["mel"].to(device, non_blocking=True)  # [B, 1, 64, T]

        optimizer.zero_grad(set_to_none=True)

        recon = model(mel)
        loss, loss_dict = reconstruction_loss(recon, mel, mse_weight=mse_weight)

        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += float(loss_dict["loss_total"].item())
        total_l1 += float(loss_dict["loss_l1"].item())
        total_mse += float(loss_dict["loss_mse"].item())
        num_batches += 1
        global_step += 1

        if (step + 1) % log_every_n_steps == 0:
            writer.add_scalar("train/loss_step", float(loss.item()), global_step)
            writer.add_scalar("train/l1_step", float(loss_dict["loss_l1"].item()), global_step)
            writer.add_scalar("train/mse_step", float(loss_dict["loss_mse"].item()), global_step)

    metrics = {
        "loss": total_loss / max(num_batches, 1),
        "l1": total_l1 / max(num_batches, 1),
        "mse": total_mse / max(num_batches, 1),
    }
    writer.add_scalar("train/loss_epoch", metrics["loss"], epoch)
    writer.add_scalar("train/l1_epoch", metrics["l1"], epoch)
    writer.add_scalar("train/mse_epoch", metrics["mse"], epoch)

    return metrics, global_step


@torch.no_grad()
def validate_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    mse_weight: float,
    writer: SummaryWriter,
    epoch: int,
) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_l1 = 0.0
    total_mse = 0.0
    num_batches = 0
    saved_preview = False

    for batch in loader:
        mel = batch["mel"].to(device, non_blocking=True)
        recon = model(mel)

        loss, loss_dict = reconstruction_loss(recon, mel, mse_weight=mse_weight)

        total_loss += float(loss_dict["loss_total"].item())
        total_l1 += float(loss_dict["loss_l1"].item())
        total_mse += float(loss_dict["loss_mse"].item())
        num_batches += 1

        if not saved_preview:
            writer.add_image("valid/input_mel", normalize_for_tb(mel[0]), epoch)
            writer.add_image("valid/recon_mel", normalize_for_tb(recon[0]), epoch)
            saved_preview = True

    metrics = {
        "loss": total_loss / max(num_batches, 1),
        "l1": total_l1 / max(num_batches, 1),
        "mse": total_mse / max(num_batches, 1),
    }
    writer.add_scalar("valid/loss_epoch", metrics["loss"], epoch)
    writer.add_scalar("valid/l1_epoch", metrics["l1"], epoch)
    writer.add_scalar("valid/mse_epoch", metrics["mse"], epoch)
    return metrics


def maybe_resume(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: Dict[str, Any],
    device: torch.device,
) -> Tuple[int, int, float]:
    resume_path = str(cfg.get("train", {}).get("resume", "")).strip()
    if not resume_path:
        return 0, 0, float("inf")

    resume_path = str((REPO_ROOT / resume_path).resolve()) if not Path(resume_path).is_absolute() else resume_path
    ckpt_path = Path(resume_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_epoch = int(checkpoint.get("epoch", -1)) + 1
    global_step = int(checkpoint.get("global_step", 0))
    best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))

    print(f"[Resume] Loaded checkpoint from: {ckpt_path}")
    print(f"[Resume] Start epoch: {start_epoch}, global_step: {global_step}, best_val_loss: {best_val_loss:.6f}")
    return start_epoch, global_step, best_val_loss


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    seed = int(cfg.get("seed", 42))
    seed_everything(seed)

    device = torch.device(args.device)
    print(f"[Info] Using device: {device}")

    out_paths = prepare_dirs(cfg)
    save_yaml(out_paths["resolved_config_path"], cfg)

    train_loader, valid_loader = build_dataloaders(cfg)

    # Fail early if batch shape is wrong
    peek_batch = next(iter(train_loader))
    print(f"[Info] Peek waveform shape: {tuple(peek_batch['waveform'].shape)}")
    print(f"[Info] Peek mel shape     : {tuple(peek_batch['mel'].shape)}")
    print(f"[Info] Peek first label   : {peek_batch['label'][0] if len(peek_batch['label']) > 0 else 'N/A'}")
    print(f"[Info] Peek first text    : {peek_batch['text'][0] if len(peek_batch['text']) > 0 else 'N/A'}")

    model = build_model(cfg).to(device)
    print(f"[Info] Model params: {count_parameters(model):,}")

    train_cfg = cfg.get("train", {})
    lr = float(train_cfg.get("lr", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    epochs = int(train_cfg.get("epochs", 10))
    mse_weight = float(train_cfg.get("mse_weight", 0.1))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    log_every_n_steps = int(train_cfg.get("log_every_n_steps", 10))

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    writer = SummaryWriter(log_dir=str(out_paths["tensorboard_dir"]))
    writer.add_text("config/yaml", yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), 0)

    start_epoch, global_step, best_val_loss = maybe_resume(model, optimizer, cfg, device)

    for epoch in range(start_epoch, epochs):
        t0 = time.time()

        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)

        train_metrics, global_step = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            mse_weight=mse_weight,
            grad_clip=grad_clip,
            writer=writer,
            epoch=epoch,
            global_step=global_step,
            log_every_n_steps=log_every_n_steps,
        )

        valid_metrics = validate_one_epoch(
            model=model,
            loader=valid_loader,
            device=device,
            mse_weight=mse_weight,
            writer=writer,
            epoch=epoch,
        )

        epoch_time = time.time() - t0

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_metrics['loss']:.6f} "
            f"valid_loss={valid_metrics['loss']:.6f} "
            f"time={epoch_time:.2f}s"
        )

        metrics_payload = {
            "epoch": epoch,
            "global_step": global_step,
            "train": train_metrics,
            "valid": valid_metrics,
            "time_sec": epoch_time,
        }
        append_metrics(out_paths["metrics_path"], metrics_payload)

        last_ckpt = out_paths["checkpoint_dir"] / "last.pt"
        save_checkpoint(
            checkpoint_path=last_ckpt,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            global_step=global_step,
            best_val_loss=best_val_loss,
            cfg=cfg,
        )

        current_val_loss = valid_metrics["loss"]
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_ckpt = out_paths["checkpoint_dir"] / "best.pt"
            save_checkpoint(
                checkpoint_path=best_ckpt,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                best_val_loss=best_val_loss,
                cfg=cfg,
            )
            print(f"[Info] New best checkpoint saved to: {best_ckpt}")

    writer.close()
    print("[Done] Training finished.")


if __name__ == "__main__":
    main()