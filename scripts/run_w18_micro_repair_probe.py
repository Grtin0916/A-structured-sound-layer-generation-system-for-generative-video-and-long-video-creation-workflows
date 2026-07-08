#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import wave
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception as exc:
    plt = None
    PLOT_IMPORT_ERROR = repr(exc)
else:
    PLOT_IMPORT_ERROR = ""


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


def safe_slug(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(s))


def read_wav_pcm(path: Path) -> Tuple[int, np.ndarray, str]:
    try:
        with wave.open(str(path), "rb") as wf:
            sr = wf.getframerate()
            ch = wf.getnchannels()
            sw = wf.getsampwidth()
            n = wf.getnframes()
            raw = wf.readframes(n)

        if sw == 1:
            y = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
            y = (y - 128.0) / 128.0
        elif sw == 2:
            y = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sw == 4:
            y = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            return 0, np.zeros(0, dtype=np.float32), f"unsupported_sample_width_{sw}"

        if ch > 1:
            y = y.reshape(-1, ch).mean(axis=1)

        return sr, y.astype(np.float32), ""
    except Exception as exc:
        return 0, np.zeros(0, dtype=np.float32), f"wav_read_error:{repr(exc)}"


def write_wav_int16(path: Path, sr: int, y: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    y = np.nan_to_num(y.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    y = np.clip(y, -0.999, 0.999)
    pcm = (y * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def frame_rms(y: np.ndarray, frame: int = 1024, hop: int = 512) -> np.ndarray:
    if y.size == 0:
        return np.zeros(0, dtype=np.float32)
    if y.size < frame:
        z = np.pad(y, (0, frame - y.size))
        return np.array([float(np.sqrt(np.mean(z * z) + 1e-12))], dtype=np.float32)

    vals = []
    for st in range(0, max(1, y.size - frame + 1), hop):
        seg = y[st:st + frame]
        vals.append(float(np.sqrt(np.mean(seg * seg) + 1e-12)))
    return np.asarray(vals, dtype=np.float32)


def onset_proxy_count(rms: np.ndarray) -> int:
    if rms.size < 3:
        return 0
    diff = np.maximum(0.0, np.diff(rms, prepend=rms[0]))
    if diff.max() > diff.min():
        env = (diff - diff.min()) / (diff.max() - diff.min() + 1e-12)
    else:
        env = diff
    th = float(env.mean() + env.std())
    peaks = 0
    last = -999
    for i in range(1, env.size - 1):
        if env[i] >= th and env[i] >= env[i - 1] and env[i] >= env[i + 1] and i - last >= 3:
            peaks += 1
            last = i
    return peaks


def metrics(y: np.ndarray) -> Dict[str, float]:
    rms = frame_rms(y)
    peak_abs = float(np.max(np.abs(y))) if y.size else 0.0
    clip_ratio = float(np.mean(np.abs(y) >= 0.98)) if y.size else 0.0
    rms_mean = float(rms.mean()) if rms.size else 0.0
    silence_ratio = float(np.mean(rms < 0.01)) if rms.size else 1.0
    onset_count = onset_proxy_count(rms)
    return {
        "peak_abs": round(peak_abs, 6),
        "clip_ratio": round(clip_ratio, 8),
        "rms_mean": round(rms_mean, 6),
        "silence_ratio": round(silence_ratio, 6),
        "onset_proxy_count": onset_count,
    }


def soft_limit(y: np.ndarray, drive: float = 0.82) -> np.ndarray:
    # 非破坏性第一版：整体衰减 + tanh 软限制，目标是降低 peak/clip proxy。
    z = y * drive
    return np.tanh(1.15 * z) / np.tanh(1.15)


def trim_and_gain(y: np.ndarray, target_rms: float = 0.065) -> np.ndarray:
    rms = frame_rms(y)
    if rms.size == 0:
        return y
    keep = np.where(rms > max(0.008, float(rms.mean()) * 0.35))[0]
    hop = 512
    if keep.size:
        start = max(0, int(keep[0] * hop))
        end = min(y.size, int((keep[-1] + 2) * hop))
        z = y[start:end]
    else:
        z = y.copy()

    cur = float(np.sqrt(np.mean(z * z) + 1e-12)) if z.size else 0.0
    gain = min(3.0, target_rms / max(cur, 1e-6))
    return np.clip(z * gain, -0.95, 0.95)


def event_local_gain(y: np.ndarray, gain: float = 1.18) -> np.ndarray:
    z = y.copy()
    rms = frame_rms(y)
    if rms.size < 3:
        return np.clip(z * 1.05, -0.95, 0.95)

    diff = np.maximum(0.0, np.diff(rms, prepend=rms[0]))
    candidate = np.argsort(diff)[-3:]
    hop = 512
    win = 4096
    for idx in candidate:
        center = int(idx * hop)
        lo = max(0, center - win // 2)
        hi = min(z.size, center + win // 2)
        z[lo:hi] *= gain
    return np.clip(z, -0.95, 0.95)


def peak_normalize(y: np.ndarray, target_peak: float = 0.88) -> np.ndarray:
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak <= 1e-9:
        return y
    gain = min(2.0, target_peak / peak)
    return np.clip(y * gain, -0.95, 0.95)


def choose_action(row: Dict[str, str]) -> str:
    category = row.get("failure_category", "")
    next_action = row.get("next_action", "")
    if "clipping" in category or "clipping" in next_action:
        return "clipping_attenuation"
    if "silence" in category or "silence" in next_action:
        return "silence_trim_or_gain"
    if "layer" in category or "remix" in next_action:
        return "event_local_gain"
    if "naive" in category:
        return "peak_normalize_only"
    return "event_local_gain"


def apply_repair(y: np.ndarray, action: str) -> np.ndarray:
    if action == "clipping_attenuation":
        return soft_limit(y)
    if action == "silence_trim_or_gain":
        return trim_and_gain(y)
    if action == "event_local_gain":
        return event_local_gain(y)
    if action == "peak_normalize_only":
        return peak_normalize(y)
    return peak_normalize(y)


def improved(before: Dict[str, float], after: Dict[str, float], action: str) -> Tuple[bool, str]:
    reasons = []

    if action == "clipping_attenuation":
        if after["peak_abs"] < before["peak_abs"]:
            reasons.append("peak_abs_down")
        if after["clip_ratio"] <= before["clip_ratio"]:
            reasons.append("clip_ratio_not_worse")
    elif action == "silence_trim_or_gain":
        if after["rms_mean"] > before["rms_mean"]:
            reasons.append("rms_mean_up")
        if after["silence_ratio"] <= before["silence_ratio"]:
            reasons.append("silence_ratio_not_worse")
    elif action == "event_local_gain":
        if after["onset_proxy_count"] >= before["onset_proxy_count"]:
            reasons.append("onset_count_not_worse")
        if after["rms_mean"] >= before["rms_mean"]:
            reasons.append("rms_mean_not_worse")
    else:
        if after["peak_abs"] <= 0.92:
            reasons.append("safe_peak")
        if after["clip_ratio"] <= before["clip_ratio"]:
            reasons.append("clip_ratio_not_worse")

    return len(reasons) > 0, ";".join(reasons)


def plot_before_after(before: np.ndarray, after: np.ndarray, sr: int, title: str, out_path: Path) -> str:
    if plt is None:
        return f"matplotlib_unavailable:{PLOT_IMPORT_ERROR}"
    if before.size == 0 or after.size == 0 or sr <= 0:
        return "empty_audio"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    def downsample(y: np.ndarray, max_points: int = 9000) -> Tuple[np.ndarray, np.ndarray]:
        step = max(1, y.size // max_points)
        yy = y[::step]
        tt = np.arange(yy.size) * step / float(sr)
        return tt, yy

    tb, yb = downsample(before)
    ta, ya = downsample(after)

    rb = frame_rms(before)
    ra = frame_rms(after)
    trb = np.arange(rb.size) * 512 / float(sr)
    tra = np.arange(ra.size) * 512 / float(sr)

    fig = plt.figure(figsize=(10, 5.2))

    ax1 = fig.add_subplot(211)
    ax1.plot(tb, yb, linewidth=0.55, label="before")
    ax1.plot(ta, ya, linewidth=0.55, alpha=0.75, label="after")
    ax1.set_title(title)
    ax1.set_ylabel("waveform")
    ax1.legend(loc="upper right")

    ax2 = fig.add_subplot(212)
    ax2.plot(trb, rb, linewidth=0.9, label="before_rms")
    ax2.plot(tra, ra, linewidth=0.9, label="after_rms")
    ax2.set_xlabel("time_sec")
    ax2.set_ylabel("RMS")
    ax2.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return ""


def select_probe_rows(rows: List[Dict[str, str]], count: int) -> List[Dict[str, str]]:
    # 保证不是只修 clipping：先按类别各拿一条，再用 clipping / repairability 补足。
    selected = []
    used = set()

    preferred_categories = [
        "silence",
        "layer_conflict_or_repairable",
        "naive_less_controllable",
        "clipping",
    ]

    for cat in preferred_categories:
        for r in rows:
            if r.get("failure_id") in used:
                continue
            if r.get("failure_category") == cat:
                selected.append(r)
                used.add(r.get("failure_id"))
                break

    def priority(r: Dict[str, str]) -> Tuple[int, float, float]:
        cat_bonus = 1 if r.get("failure_category") == "clipping" else 0
        repairability = float(r.get("repairability") or 0.0)
        score = float(r.get("selector_v2_score") or 0.0)
        return cat_bonus, repairability, score

    for r in sorted(rows, key=priority, reverse=True):
        if len(selected) >= count:
            break
        if r.get("failure_id") not in used:
            selected.append(r)
            used.add(r.get("failure_id"))

    return selected[:count]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--failure-bank", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-summary", required=True)
    ap.add_argument("--count", type=int, default=6)
    args = ap.parse_args()

    rows = read_csv(Path(args.failure_bank))
    selected = select_probe_rows(rows, args.count)

    out_dir = Path(args.out_dir)
    wav_dir = out_dir / "wav"
    plot_dir = out_dir / "plots"

    out_rows = []

    for idx, r in enumerate(selected, 1):
        audio_path = Path(r.get("audio_path", ""))
        sr, before, read_error = read_wav_pcm(audio_path)

        action = choose_action(r)
        if read_error:
            after = before
        else:
            after = apply_repair(before, action)

        before_m = metrics(before)
        after_m = metrics(after)
        ok, reason = improved(before_m, after_m, action)

        probe_id = f"mr_{idx:03d}"
        stem = f"{probe_id}__{safe_slug(r.get('case_id'))}__{safe_slug(r.get('variant'))}__{safe_slug(action)}"
        after_path = wav_dir / f"{stem}__after.wav"
        plot_path = plot_dir / f"{stem}__before_after.png"

        write_error = ""
        plot_error = ""

        if not read_error and sr > 0 and after.size:
            try:
                write_wav_int16(after_path, sr, after)
            except Exception as exc:
                write_error = repr(exc)
            plot_error = plot_before_after(before, after, sr, stem, plot_path)
        else:
            write_error = "skip_due_to_read_error"

        out = {
            "probe_id": probe_id,
            "failure_id": r.get("failure_id"),
            "candidate_key": r.get("candidate_key"),
            "case_id": r.get("case_id"),
            "variant": r.get("variant"),
            "failure_category": r.get("failure_category"),
            "repair_action": action,
            "before_audio_path": str(audio_path),
            "after_audio_path": str(after_path),
            "plot_path": str(plot_path),
            "sr": sr,
            "read_error": read_error,
            "write_error": write_error,
            "plot_error": plot_error,
            "proxy_improved": "true" if ok and not read_error and not write_error else "false",
            "improve_reason": reason,
        }

        for k, v in before_m.items():
            out[f"before_{k}"] = v
        for k, v in after_m.items():
            out[f"after_{k}"] = v

        out_rows.append(out)

    fields = [
        "probe_id", "failure_id", "candidate_key", "case_id", "variant",
        "failure_category", "repair_action",
        "before_audio_path", "after_audio_path", "plot_path", "sr",
        "read_error", "write_error", "plot_error",
        "proxy_improved", "improve_reason",
        "before_peak_abs", "after_peak_abs",
        "before_clip_ratio", "after_clip_ratio",
        "before_rms_mean", "after_rms_mean",
        "before_silence_ratio", "after_silence_ratio",
        "before_onset_proxy_count", "after_onset_proxy_count",
    ]

    write_csv(Path(args.out_csv), out_rows, fields)
    Path(args.out_json).write_text(json.dumps(out_rows, indent=2, ensure_ascii=False), encoding="utf-8")

    action_counts: Dict[str, int] = {}
    category_counts: Dict[str, int] = {}
    for r in out_rows:
        action_counts[r["repair_action"]] = action_counts.get(r["repair_action"], 0) + 1
        category_counts[r["failure_category"]] = category_counts.get(r["failure_category"], 0) + 1

    summary = {
        "task": "w18_micro_repair_probe",
        "input_failure_bank": args.failure_bank,
        "probe_count": len(out_rows),
        "after_wav_count": sum(1 for r in out_rows if Path(r["after_audio_path"]).exists()),
        "plot_count": sum(1 for r in out_rows if Path(r["plot_path"]).exists()),
        "proxy_improve_count": sum(1 for r in out_rows if r["proxy_improved"] == "true"),
        "read_error_count": sum(1 for r in out_rows if r["read_error"]),
        "write_error_count": sum(1 for r in out_rows if r["write_error"]),
        "action_counts": action_counts,
        "category_counts": category_counts,
        "dod": {
            "probe_count_ge_6": len(out_rows) >= 6,
            "after_wav_count_ge_6": sum(1 for r in out_rows if Path(r["after_audio_path"]).exists()) >= 6,
            "plot_count_ge_6": sum(1 for r in out_rows if Path(r["plot_path"]).exists()) >= 6,
            "proxy_improve_count_ge_2": sum(1 for r in out_rows if r["proxy_improved"] == "true") >= 2,
            "read_error_count_eq_0": sum(1 for r in out_rows if r["read_error"]) == 0,
            "write_error_count_eq_0": sum(1 for r in out_rows if r["write_error"]) == 0,
        },
        "outputs": {
            "csv": args.out_csv,
            "json": args.out_json,
            "summary": args.out_summary,
            "out_dir": args.out_dir,
        },
        "boundary": "Non-destructive micro repair probe only; not a full repair engine and not a subjective audio-quality claim.",
    }

    Path(args.out_summary).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
