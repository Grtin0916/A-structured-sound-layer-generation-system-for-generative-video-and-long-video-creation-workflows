#!/usr/bin/env python3
import argparse
import csv
import json
import math
import wave
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


VARIANT_ORDER = {
    "naive": 0,
    "naive_rich": 1,
    "dss_global": 2,
    "dss_event_timeline": 3,
    "dss_layer_avoid": 4,
}


def load_wav_mono(path: Path):
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sr = wf.getframerate()
        sampwidth = wf.getsampwidth()
        nframes = wf.getnframes()
        raw = wf.readframes(nframes)

    if sampwidth == 2:
        audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sampwidth == 4:
        audio = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    elif sampwidth == 1:
        audio = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"unsupported sample width: {sampwidth}")

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    return audio, sr, channels, sampwidth


def frame_rms(audio: np.ndarray, frame: int, hop: int):
    if len(audio) < frame:
        padded = np.pad(audio, (0, max(0, frame - len(audio))))
        return np.array([float(np.sqrt(np.mean(padded ** 2)))], dtype=np.float32)

    vals = []
    for start in range(0, len(audio) - frame + 1, hop):
        seg = audio[start:start + frame]
        vals.append(float(np.sqrt(np.mean(seg ** 2))))
    return np.asarray(vals, dtype=np.float32)


def simple_onset_proxy(audio: np.ndarray, sr: int, frame_ms=40, hop_ms=20):
    frame = max(1, int(sr * frame_ms / 1000.0))
    hop = max(1, int(sr * hop_ms / 1000.0))
    env = frame_rms(audio, frame, hop)

    if len(env) < 3:
        return {
            "onset_count": 0,
            "onset_times_sec": [],
            "energy_times_sec": [0.0],
            "energy_envelope": env.tolist(),
        }

    diff = np.diff(env, prepend=env[0])
    diff = np.maximum(diff, 0)
    threshold = float(diff.mean() + 1.5 * diff.std())
    min_distance = max(1, int(0.12 / (hop / sr)))

    peaks = []
    last = -10**9
    for i in range(1, len(diff) - 1):
        if diff[i] > threshold and diff[i] >= diff[i - 1] and diff[i] >= diff[i + 1] and i - last >= min_distance:
            peaks.append(i)
            last = i

    times = (np.arange(len(env)) * hop / sr).astype(float)
    onset_times = [round(float(times[i]), 4) for i in peaks]

    return {
        "onset_count": len(onset_times),
        "onset_times_sec": onset_times,
        "energy_times_sec": [round(float(x), 4) for x in times],
        "energy_envelope": [round(float(x), 8) for x in env],
    }


def dbfs(x: float):
    return 20.0 * math.log10(max(float(x), 1e-12))


def stats_for_wav(path: Path):
    audio, sr, channels, sampwidth = load_wav_mono(path)
    abs_audio = np.abs(audio)

    peak = float(abs_audio.max()) if audio.size else 0.0
    rms = float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0
    duration = len(audio) / float(sr) if sr else 0.0
    clipped_ratio = float(np.mean(abs_audio >= 0.999)) if audio.size else 0.0
    active_ratio = float(np.mean(abs_audio >= 1e-4)) if audio.size else 0.0
    silence_ratio = float(np.mean(abs_audio < 1e-4)) if audio.size else 0.0

    onset = simple_onset_proxy(audio, sr)

    status = "ok"
    flags = []
    if rms <= 1e-6:
        status = "near_silent"
        flags.append("near_silent")
    if clipped_ratio >= 0.001:
        status = "clipping_review"
        flags.append("clipping_review")
    if dbfs(peak) >= -0.1:
        flags.append("near_full_scale_peak")
    if dbfs(rms) > -12:
        flags.append("very_loud_rms")
    if dbfs(rms) < -45:
        flags.append("very_quiet_rms")

    return {
        "sample_rate": sr,
        "channels": channels,
        "sample_width_bytes": sampwidth,
        "duration_sec": round(duration, 4),
        "rms_dbfs": round(dbfs(rms), 4),
        "peak_dbfs": round(dbfs(peak), 4),
        "clipped_ratio": round(clipped_ratio, 8),
        "active_ratio": round(active_ratio, 8),
        "silence_ratio": round(silence_ratio, 8),
        "onset_count_proxy": onset["onset_count"],
        "onset_times_sec_proxy": onset["onset_times_sec"],
        "status": status,
        "flags": flags,
        "_audio": audio,
        "_sr": sr,
        "_energy_times": onset["energy_times_sec"],
        "_energy_env": onset["energy_envelope"],
        "_onsets": onset["onset_times_sec"],
    }


def plot_case(case_id: str, rows, out_dir: Path):
    rows = sorted(rows, key=lambda r: VARIANT_ORDER.get(r["variant"], 99))
    fig, axes = plt.subplots(len(rows), 1, figsize=(14, 2.2 * len(rows)), sharex=False)
    if len(rows) == 1:
        axes = [axes]

    for ax, r in zip(axes, rows):
        audio = r["_audio"]
        sr = r["_sr"]
        t = np.arange(len(audio)) / float(sr)
        ax.plot(t, audio, linewidth=0.45, alpha=0.7)

        et = np.asarray(r["_energy_times"], dtype=float)
        ee = np.asarray(r["_energy_env"], dtype=float)
        if len(et) and len(ee) and float(ee.max(initial=0)) > 0:
            ee_scaled = (ee / max(float(ee.max()), 1e-12)) * 0.8
            ax.plot(et, ee_scaled, linewidth=0.9, alpha=0.8)

        for onset in r["_onsets"]:
            ax.axvline(float(onset), linestyle="--", linewidth=0.7, alpha=0.5)

        ax.set_ylim(-1.05, 1.05)
        ax.set_title(
            f"{r['variant']} | rms={r['rms_dbfs']} dBFS | peak={r['peak_dbfs']} dBFS | "
            f"clip={r['clipped_ratio']} | onsets={r['onset_count_proxy']} | {r['status']}",
            fontsize=9,
        )

    axes[-1].set_xlabel("time (sec)")
    fig.suptitle(case_id, fontsize=12)
    fig.tight_layout()
    out = out_dir / f"{case_id}_waveform_energy_onset_proxy.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return str(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="reports/w18_full_30job_generation_matrix_20260706.csv")
    ap.add_argument("--listening-pack", default="reports/w18_repair_aware_listening_pack_20260706.csv")
    ap.add_argument("--out-json", default="reports/w18_audio_metrics_eval_20260707.json")
    ap.add_argument("--out-csv", default="reports/w18_audio_metrics_eval_20260707.csv")
    ap.add_argument("--plot-dir", default="artifacts/eval/w18_audio_metrics_20260707")
    args = ap.parse_args()

    matrix = list(csv.DictReader(Path(args.matrix).open(encoding="utf-8")))
    selected_by_job = {}

    lp_path = Path(args.listening_pack)
    if lp_path.exists():
        for row in csv.DictReader(lp_path.open(encoding="utf-8")):
            selected_by_job[row["job_id"]] = row.get("selected_wav_path") or row.get("original_wav_path")

    out_rows = []
    rows_by_case = {}
    read_errors = []

    for row in matrix:
        job_id = row["job_id"]
        selected_path = selected_by_job.get(job_id) or row.get("wav_path")
        wav_path = Path(selected_path)

        if not wav_path.exists():
            read_errors.append({"job_id": job_id, "path": str(wav_path), "error": "missing_wav"})
            continue

        try:
            st = stats_for_wav(wav_path)
        except Exception as exc:
            read_errors.append({"job_id": job_id, "path": str(wav_path), "error": f"{type(exc).__name__}: {exc}"})
            continue

        out = {
            "case_id": row["case_id"],
            "variant": row["variant"],
            "job_id": job_id,
            "selected_wav_path": str(wav_path),
            "is_repaired_selected": "repaired" in wav_path.name,
            "prompt_chars": row.get("prompt_chars"),
            **{k: v for k, v in st.items() if not k.startswith("_")},
        }
        out_rows.append(out)

        private = dict(out)
        private.update({k: st[k] for k in ["_audio", "_sr", "_energy_times", "_energy_env", "_onsets"]})
        rows_by_case.setdefault(row["case_id"], []).append(private)

    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    plots = []
    for case_id, case_rows in sorted(rows_by_case.items()):
        plots.append(plot_case(case_id, case_rows, plot_dir))

    summary = {
        "date": "2026-07-07",
        "scope": "w18_audio_metrics_eval",
        "status": "success" if len(out_rows) >= 30 and len(plots) >= 6 and not read_errors else "review_required",
        "candidate_count": len(matrix),
        "readable_count": len(out_rows),
        "plot_count": len(plots),
        "case_count": len(rows_by_case),
        "read_error_count": len(read_errors),
        "read_errors": read_errors,
        "plots": plots,
        "metric_boundary": [
            "Onset count is a low-cost energy-onset proxy, not final audio-video synchrony.",
            "RMS/peak/clip/silence are acoustic sanity metrics, not human preference.",
            "Selected repaired WAV is used when listening pack marks a repaired candidate.",
        ],
        "outputs": {
            "json": args.out_json,
            "csv": args.out_csv,
            "plot_dir": args.plot_dir,
        },
        "rows": out_rows,
    }

    Path(args.out_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if out_rows:
        with Path(args.out_csv).open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            writer.writeheader()
            writer.writerows(out_rows)

    print(json.dumps({k: summary[k] for k in [
        "status", "candidate_count", "readable_count", "plot_count", "case_count", "read_error_count", "outputs"
    ]}, ensure_ascii=False, indent=2))

    return 0 if summary["status"] == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
