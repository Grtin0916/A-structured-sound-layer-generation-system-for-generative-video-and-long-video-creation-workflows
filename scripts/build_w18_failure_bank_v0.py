#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception as exc:
    plt = None
    MATPLOTLIB_IMPORT_ERROR = repr(exc)
else:
    MATPLOTLIB_IMPORT_ERROR = ""


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def to_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or str(x).strip() == "":
            return default
        v = float(str(x).strip())
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def find_numeric(row: Dict[str, Any], needles: List[str], default: float = 0.0) -> float:
    vals = []
    for k, v in row.items():
        lk = k.lower()
        if any(n in lk for n in needles):
            fv = to_float(v, default=None)
            if fv is not None:
                vals.append((lk, fv))
    if not vals:
        return default
    # score-like fields first, otherwise first numeric match
    vals.sort(key=lambda kv: (0 if "score" in kv[0] or "ratio" in kv[0] else 1, kv[0]))
    return vals[0][1]


def safe_slug(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in s)


def read_wav_pcm(path: Path) -> Tuple[int, np.ndarray, str]:
    try:
        with wave.open(str(path), "rb") as wf:
            sr = wf.getframerate()
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)

        if sampwidth == 1:
            y = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
            y = (y - 128.0) / 128.0
        elif sampwidth == 2:
            y = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 4:
            y = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            return 0, np.zeros(0, dtype=np.float32), f"unsupported_sample_width_{sampwidth}"

        if n_channels > 1:
            y = y.reshape(-1, n_channels).mean(axis=1)

        return sr, y, ""
    except Exception as exc:
        return 0, np.zeros(0, dtype=np.float32), f"wav_read_error:{repr(exc)}"


def frame_rms(y: np.ndarray, frame: int, hop: int) -> np.ndarray:
    if y.size == 0:
        return np.zeros(0, dtype=np.float32)
    if y.size < frame:
        pad = np.pad(y, (0, frame - y.size))
        return np.array([float(np.sqrt(np.mean(pad * pad)))], dtype=np.float32)

    vals = []
    for start in range(0, max(1, y.size - frame + 1), hop):
        seg = y[start:start + frame]
        vals.append(float(np.sqrt(np.mean(seg * seg) + 1e-12)))
    return np.asarray(vals, dtype=np.float32)


def onset_proxy_from_rms(rms: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if rms.size == 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.int64)
    diff = np.maximum(0.0, np.diff(rms, prepend=rms[0]))
    if diff.max() > diff.min():
        env = (diff - diff.min()) / (diff.max() - diff.min() + 1e-12)
    else:
        env = diff

    if env.size < 3:
        return env, np.zeros(0, dtype=np.int64)

    threshold = float(env.mean() + 1.0 * env.std())
    peaks = []
    for i in range(1, env.size - 1):
        if env[i] >= threshold and env[i] >= env[i - 1] and env[i] >= env[i + 1]:
            if not peaks or i - peaks[-1] >= 3:
                peaks.append(i)
    return env, np.asarray(peaks, dtype=np.int64)


def classify_failure(row: Dict[str, Any], audio_stats: Dict[str, Any]) -> Tuple[str, str, str]:
    metric_conf = to_float(row.get("metric_confidence"), 0.5)
    repairability = to_float(row.get("repairability"), 0.0)
    variant_family = row.get("variant_family", "")
    rejection = row.get("selector_v2_rejection_reason", "")

    onset_quality = find_numeric(row, ["onset_quality"], 0.5)
    forbidden = find_numeric(row, ["forbidden_penalty"], 0.5)
    clip_quality = find_numeric(row, ["clip_quality"], 0.5)
    silence_quality = find_numeric(row, ["silence_quality"], 0.5)

    peak_abs = float(audio_stats.get("peak_abs", 0.0))
    rms_mean = float(audio_stats.get("rms_mean", 0.0))
    onset_count = int(audio_stats.get("onset_proxy_count", 0))

    if peak_abs >= 0.98 or clip_quality < 0.35:
        return "clipping", "clipping_attenuation", "Peak or clip proxy is high; attenuate clipped regions before reselection."
    if rms_mean < 0.015 or silence_quality < 0.35:
        return "silence", "silence_trim_or_gain", "RMS/silence proxy suggests weak or silent candidate; trim silence and apply event-local gain."
    if onset_quality < 0.35 or onset_count == 0:
        return "missing_or_weak_onset", "event_local_gain", "Onset proxy is weak; boost expected event window or replace Foley layer."
    if forbidden < 0.35:
        return "forbidden_leakage", "reject_or_regenerate_with_avoid", "Forbidden leakage proxy is risky; prefer avoid-prompt regeneration rather than gain repair."
    if "naive" in variant_family:
        return "naive_less_controllable", "replace_with_dss_variant", "Naive prompt has weaker controllability; use DSS variant or repair only if audio quality is much better."
    if metric_conf < 0.45:
        return "low_metric_confidence", "manual_review_then_repair", "Metric confidence is low; route to manual listening before destructive repair."
    if repairability >= 0.70:
        return "layer_conflict_or_repairable", "layer_priority_remix", "Marked repairable; try non-destructive layer gain/remix first."
    return "lower_ranked_candidate", "keep_as_runner_up", "Lower selector-v2 score; preserve as runner-up evidence."


def make_plot(row: Dict[str, Any], y: np.ndarray, sr: int, rms: np.ndarray, onset_env: np.ndarray, peaks: np.ndarray, out_path: Path) -> str:
    if plt is None:
        return f"matplotlib_unavailable:{MATPLOTLIB_IMPORT_ERROR}"
    if y.size == 0 or sr <= 0:
        return "empty_audio"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    duration = y.size / float(sr)
    max_points = 12000
    step = max(1, y.size // max_points)
    yy = y[::step]
    tt = np.arange(yy.size) * step / float(sr)

    hop = 512
    rr_t = np.arange(rms.size) * hop / float(sr)
    oo_t = np.arange(onset_env.size) * hop / float(sr)
    peak_t = peaks * hop / float(sr)

    fig = plt.figure(figsize=(10, 4.5))
    ax1 = fig.add_subplot(211)
    ax1.plot(tt, yy, linewidth=0.6)
    ax1.set_title(f"{row.get('case_id')} | {row.get('variant')} | {row.get('failure_category', '')}")
    ax1.set_ylabel("waveform")
    ax1.set_xlim(0, max(duration, 0.01))

    ax2 = fig.add_subplot(212)
    if rms.size:
        ax2.plot(rr_t, rms, linewidth=1.0, label="RMS")
    if onset_env.size:
        ax2.plot(oo_t, onset_env, linewidth=1.0, label="onset_proxy")
    for t in peak_t[:20]:
        ax2.axvline(float(t), linestyle="--", linewidth=0.8)
    ax2.set_xlabel("time_sec")
    ax2.set_ylabel("proxy")
    ax2.set_xlim(0, max(duration, 0.01))
    ax2.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selector-v2", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-summary", required=True)
    ap.add_argument("--plot-dir", required=True)
    ap.add_argument("--top-k", type=int, default=12)
    args = ap.parse_args()

    rows = read_csv(Path(args.selector_v2))
    non_winners = [r for r in rows if r.get("selector_v2_decision") != "winner"]

    def priority(r: Dict[str, Any]) -> Tuple[float, float, float]:
        repair = to_float(r.get("repairability"), 0.0)
        metric = to_float(r.get("metric_confidence"), 0.5)
        score = to_float(r.get("selector_v2_score"), 0.0)
        return (repair, 1.0 - metric, score)

    selected = sorted(non_winners, key=priority, reverse=True)[:args.top_k]

    out_rows: List[Dict[str, Any]] = []
    plot_dir = Path(args.plot_dir)

    for idx, r in enumerate(selected, 1):
        audio_path = Path(r.get("audio_path", ""))
        sr, y, read_error = read_wav_pcm(audio_path) if audio_path.exists() else (0, np.zeros(0, dtype=np.float32), "audio_path_missing")

        rms = frame_rms(y, frame=1024, hop=512)
        onset_env, peaks = onset_proxy_from_rms(rms)

        audio_stats = {
            "sr": sr,
            "duration_sec": 0.0 if sr <= 0 else round(y.size / float(sr), 4),
            "peak_abs": round(float(np.max(np.abs(y))) if y.size else 0.0, 6),
            "rms_mean": round(float(rms.mean()) if rms.size else 0.0, 6),
            "onset_proxy_count": int(peaks.size),
            "wav_read_error": read_error,
        }

        category, next_action, note = classify_failure(r, audio_stats)

        item = {
            "failure_id": f"fb_{idx:03d}",
            "candidate_key": r.get("candidate_key"),
            "case_id": r.get("case_id"),
            "variant": r.get("variant"),
            "audio_path": r.get("audio_path"),
            "selector_v2_rank": r.get("selector_v2_rank"),
            "selector_v2_score": r.get("selector_v2_score"),
            "metric_confidence": r.get("metric_confidence"),
            "repairability": r.get("repairability"),
            "selector_rejection_reason": r.get("selector_v2_rejection_reason"),
            "failure_category": category,
            "next_action": next_action,
            "repair_note": note,
            **audio_stats,
        }

        plot_path = plot_dir / f"{item['failure_id']}__{safe_slug(item['case_id'])}__{safe_slug(item['variant'])}.png"
        item["plot_path"] = str(plot_path)
        plot_error = make_plot(item, y, sr, rms, onset_env, peaks, plot_path)
        item["plot_ok"] = "true" if not plot_error else "false"
        item["plot_error"] = plot_error

        out_rows.append(item)

    fields = [
        "failure_id", "candidate_key", "case_id", "variant", "audio_path",
        "selector_v2_rank", "selector_v2_score", "metric_confidence", "repairability",
        "selector_rejection_reason", "failure_category", "next_action", "repair_note",
        "sr", "duration_sec", "peak_abs", "rms_mean", "onset_proxy_count",
        "wav_read_error", "plot_path", "plot_ok", "plot_error",
    ]

    write_csv(Path(args.out_csv), out_rows, fields)
    Path(args.out_json).write_text(json.dumps(out_rows, indent=2, ensure_ascii=False), encoding="utf-8")

    category_counts: Dict[str, int] = {}
    action_counts: Dict[str, int] = {}
    for r in out_rows:
        category_counts[r["failure_category"]] = category_counts.get(r["failure_category"], 0) + 1
        action_counts[r["next_action"]] = action_counts.get(r["next_action"], 0) + 1

    summary = {
        "task": "w18_failure_bank_v0",
        "input_selector_v2": args.selector_v2,
        "failure_count": len(out_rows),
        "plot_count": sum(1 for r in out_rows if r["plot_ok"] == "true"),
        "case_count": len({r["case_id"] for r in out_rows}),
        "category_counts": category_counts,
        "action_counts": action_counts,
        "blocked_audio_count": sum(1 for r in out_rows if r["wav_read_error"]),
        "dod": {
            "failure_count_ge_12": len(out_rows) >= 12,
            "plot_count_ge_12": sum(1 for r in out_rows if r["plot_ok"] == "true") >= 12,
            "each_failure_has_category": all(bool(r["failure_category"]) for r in out_rows),
            "each_failure_has_next_action": all(bool(r["next_action"]) for r in out_rows),
            "case_count_ge_6": len({r["case_id"] for r in out_rows}) >= 6,
        },
        "outputs": {
            "csv": args.out_csv,
            "json": args.out_json,
            "summary": args.out_summary,
            "plot_dir": args.plot_dir,
        },
    }

    Path(args.out_summary).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
