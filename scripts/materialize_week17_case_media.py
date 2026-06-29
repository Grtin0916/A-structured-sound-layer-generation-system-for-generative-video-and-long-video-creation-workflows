#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Materialize Week17 case media:
1) Convert input_video_stub.json cases into synthetic silent MP4 when ffmpeg is available.
2) Generate a deterministic control-rule Foley WAV for every DSS case.
3) Compute basic audio metrics.
4) Refresh a model input manifest with honest evidence levels.

No third-party Python dependency is required.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import struct
import subprocess
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SAMPLE_RATE = 48000
MAX_INT16 = 32767


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def run_cmd(cmd: list[str]) -> tuple[bool, str]:
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        return p.returncode == 0, (p.stdout + "\n" + p.stderr)[-4000:]
    except Exception as e:
        return False, str(e)


def escape_drawtext(text: str) -> str:
    return text.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def make_synthetic_video(case_dir: Path, dss: dict[str, Any], ffmpeg_ok: bool) -> dict[str, Any]:
    out = case_dir / "input_video.mp4"
    stub = case_dir / "input_video_stub.json"

    if out.exists():
        return {
            "case_id": case_dir.name,
            "video_status": "exists",
            "video_path": str(out),
            "video_kind": dss.get("video", {}).get("input_kind", "local_video"),
            "ffmpeg_message": "",
        }

    if not stub.exists():
        return {
            "case_id": case_dir.name,
            "video_status": "missing_no_stub",
            "video_path": "",
            "video_kind": "missing",
            "ffmpeg_message": "No input_video.mp4 and no input_video_stub.json.",
        }

    if not ffmpeg_ok:
        return {
            "case_id": case_dir.name,
            "video_status": "blocked",
            "video_path": "",
            "video_kind": "stub",
            "ffmpeg_message": "ffmpeg is not available.",
        }

    duration = float(dss.get("video", {}).get("duration_s", 8.0))
    case_id = dss.get("case_id", case_dir.name)
    scene = dss.get("scene", "")

    # Try drawtext first. If the local ffmpeg lacks drawtext/freetype, fall back to plain color + silent audio.
    text = escape_drawtext(f"{case_id} | synthetic placeholder | {scene[:80]}")
    cmd_draw = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=c=black:s=1280x720:r=24:d={duration}",
        "-f", "lavfi", "-i", f"anullsrc=channel_layout=stereo:sample_rate={SAMPLE_RATE}",
        "-shortest",
        "-vf", f"drawtext=text='{text}':fontcolor=white:fontsize=28:x=40:y=60",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        str(out),
    ]
    ok, msg = run_cmd(cmd_draw)

    if not ok:
        cmd_plain = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"testsrc2=size=1280x720:rate=24:duration={duration}",
            "-f", "lavfi", "-i", f"anullsrc=channel_layout=stereo:sample_rate={SAMPLE_RATE}",
            "-shortest",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            str(out),
        ]
        ok, msg2 = run_cmd(cmd_plain)
        msg = msg + "\n--- fallback ---\n" + msg2

    if ok and out.exists():
        dss.setdefault("video", {})
        dss["video"]["input_kind"] = "synthetic_placeholder_video"
        dss["video"]["path"] = str(out)
        dss["video"]["semantic_evidence_level"] = "pipeline_only_not_real_scene"
        write_json(case_dir / "director_sound_script.yaml", dss)
        return {
            "case_id": case_id,
            "video_status": "created",
            "video_path": str(out),
            "video_kind": "synthetic_placeholder_video",
            "ffmpeg_message": msg[-800:],
        }

    return {
        "case_id": case_id,
        "video_status": "failed",
        "video_path": "",
        "video_kind": "stub",
        "ffmpeg_message": msg[-1200:],
    }


def add_sine(buf: list[float], start_s: float, dur_s: float, freq: float, amp: float, sr: int = SAMPLE_RATE) -> None:
    start = max(0, int(start_s * sr))
    n = max(1, int(dur_s * sr))
    end = min(len(buf), start + n)
    for i in range(start, end):
        t = (i - start) / sr
        env = math.sin(math.pi * min(1.0, max(0.0, t / max(dur_s, 1e-6))))
        env *= math.sin(math.pi * min(1.0, max(0.0, (dur_s - t) / max(dur_s, 1e-6))))
        buf[i] += amp * env * math.sin(2 * math.pi * freq * t)


def add_noise(buf: list[float], start_s: float, dur_s: float, amp: float, seed: int, sr: int = SAMPLE_RATE) -> None:
    rng = random.Random(seed)
    start = max(0, int(start_s * sr))
    n = max(1, int(dur_s * sr))
    end = min(len(buf), start + n)
    for i in range(start, end):
        t = (i - start) / sr
        attack = min(1.0, t / 0.03)
        release = min(1.0, (dur_s - t) / 0.08)
        env = max(0.0, min(attack, release))
        buf[i] += amp * env * rng.uniform(-1.0, 1.0)


def add_pulse_train(buf: list[float], start_s: float, dur_s: float, freq: float, amp: float, pulses: int) -> None:
    if pulses <= 1:
        add_noise(buf, start_s, min(dur_s, 0.12), amp, int(start_s * 1000))
        return
    step = dur_s / pulses
    for j in range(pulses):
        add_noise(buf, start_s + j * step, min(0.08, step * 0.5), amp, int(start_s * 1000) + j)
        add_sine(buf, start_s + j * step, min(0.08, step * 0.5), freq, amp * 0.4)


def add_event(buf: list[float], ev: dict[str, Any], case_seed: int) -> None:
    eid = str(ev.get("event_id", "")).lower()
    intent = str(ev.get("sound_intent", "")).lower()
    start = float(ev.get("time_s", 0.0))
    dur = max(0.1, float(ev.get("duration_s", 0.3)))
    priority = int(ev.get("priority", 3))
    amp = min(0.55, 0.08 + priority * 0.06)
    key = eid + " " + intent

    if any(x in key for x in ["ambience", "tone", "hum", "wind", "rain", "platform", "forest"]):
        add_noise(buf, start, dur, amp * 0.22, case_seed + int(start * 1000))
        if "hum" in key or "platform" in key:
            add_sine(buf, start, dur, 90.0, amp * 0.12)
        return

    if any(x in key for x in ["footstep", "chop", "knife"]):
        add_pulse_train(buf, start, dur, 240.0, amp, max(2, int(dur * 4)))
        return

    if any(x in key for x in ["car", "train", "rumble", "subway"]):
        add_sine(buf, start, dur, 85.0, amp * 0.55)
        add_noise(buf, start, dur, amp * 0.25, case_seed + 13)
        return

    if any(x in key for x in ["splash", "sizzle", "shatter", "glass", "rustle"]):
        add_noise(buf, start, dur, amp, case_seed + int(start * 700))
        return

    if any(x in key for x in ["beep", "chime", "bird"]):
        add_sine(buf, start, min(dur, 0.35), 880.0 if "beep" in key else 1320.0, amp)
        if "bird" in key:
            add_sine(buf, start + 0.28, min(dur, 0.30), 1540.0, amp * 0.8)
        return

    if any(x in key for x in ["brake", "servo"]):
        add_sine(buf, start, dur, 520.0 if "brake" in key else 360.0, amp * 0.7)
        add_noise(buf, start, dur, amp * 0.25, case_seed + 29)
        return

    add_noise(buf, start, dur, amp, case_seed + int(start * 1000))


def write_wav(path: Path, samples: list[float], sr: int = SAMPLE_RATE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    peak = max(max(samples), abs(min(samples)), 1e-9)
    norm = 0.95 / peak if peak > 0.95 else 1.0
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        frames = bytearray()
        for x in samples:
            y = max(-1.0, min(1.0, x * norm))
            frames.extend(struct.pack("<h", int(y * MAX_INT16)))
        wf.writeframes(frames)


def audio_metrics(samples: list[float], sr: int = SAMPLE_RATE) -> dict[str, Any]:
    n = len(samples)
    if n == 0:
        return {"duration_s": 0, "rms": 0, "peak": 0, "clip_rate": 0, "silence_ratio": 1}
    peak = max(abs(x) for x in samples)
    rms = math.sqrt(sum(x * x for x in samples) / n)
    clip_rate = sum(1 for x in samples if abs(x) >= 0.98) / n
    silence_ratio = sum(1 for x in samples if abs(x) < 0.003) / n
    return {
        "duration_s": round(n / sr, 3),
        "rms": round(rms, 6),
        "peak": round(peak, 6),
        "clip_rate": round(clip_rate, 8),
        "silence_ratio": round(silence_ratio, 6),
    }


def make_control_wav(case_dir: Path, dss: dict[str, Any], out_root: Path) -> dict[str, Any]:
    case_id = dss.get("case_id", case_dir.name)
    duration = float(dss.get("video", {}).get("duration_s", 8.0))
    total_samples = max(1, int(duration * SAMPLE_RATE))
    buf = [0.0] * total_samples
    case_seed = abs(hash(case_id)) % 100000

    for ev in dss.get("events", []):
        add_event(buf, ev, case_seed)

    out = out_root / case_id / f"{case_id}__control_rule_foley_v0.wav"
    write_wav(out, buf)
    m = audio_metrics(buf)
    m.update(
        {
            "case_id": case_id,
            "candidate_id": f"{case_id}__control_rule_foley_v0",
            "candidate_path": str(out),
            "event_count": len(dss.get("events", [])),
            "model_family": "control_rule_foley_v0",
        }
    )
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases-root", default="cases")
    ap.add_argument("--reports-root", default="reports")
    ap.add_argument("--control-root", default="artifacts/model_runs/week17_control_baseline")
    args = ap.parse_args()

    repo = Path.cwd()
    cases_root = repo / args.cases_root
    reports_root = repo / args.reports_root
    control_root = repo / args.control_root
    report_log = repo / "artifacts" / "logs"
    report_log.mkdir(parents=True, exist_ok=True)

    ffmpeg_ok = ffmpeg_available()
    case_dirs = sorted([p for p in cases_root.iterdir() if p.is_dir()]) if cases_root.exists() else []

    materialized_rows: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []

    for case_dir in case_dirs:
        dss_path = case_dir / "director_sound_script.yaml"
        if not dss_path.exists():
            materialized_rows.append({"case_id": case_dir.name, "video_status": "missing_dss", "video_path": "", "video_kind": "missing_dss"})
            continue

        dss = read_json(dss_path)
        video_row = make_synthetic_video(case_dir, dss, ffmpeg_ok)
        materialized_rows.append(video_row)

        # Reload DSS because synthetic video generation may have updated it.
        dss = read_json(dss_path)
        wav_metrics = make_control_wav(case_dir, dss, control_root)
        metrics_rows.append(wav_metrics)

        input_video = case_dir / "input_video.mp4"
        video_ready = input_video.exists()
        semantic_level = dss.get("video", {}).get("semantic_evidence_level", "real_or_existing_video")

        manifest_rows.append(
            {
                "candidate_id": f"{case_dir.name}__mmaudio_text_video_sync_v1",
                "case_id": case_dir.name,
                "model_family": "mmaudio_text_video_sync_v1",
                "input_path": str(input_video) if video_ready else str(case_dir / "input_video_stub.json"),
                "dss_path": str(dss_path),
                "expected_events_path": str(case_dir / "expected_events.csv"),
                "baseline_prompt_path": str(case_dir / "baseline_prompt.txt"),
                "runtime_precondition": "ready" if video_ready else "blocked",
                "blocked_reason": "" if video_ready else "missing_input_video_mp4",
                "semantic_evidence_level": semantic_level,
                "output_dir": f"artifacts/model_runs/week17_mmaudio/{case_dir.name}",
            }
        )

        manifest_rows.append(
            {
                "candidate_id": f"{case_dir.name}__control_rule_foley_v0",
                "case_id": case_dir.name,
                "model_family": "control_rule_foley_v0",
                "input_path": str(dss_path),
                "dss_path": str(dss_path),
                "expected_events_path": str(case_dir / "expected_events.csv"),
                "baseline_prompt_path": str(case_dir / "baseline_prompt.txt"),
                "runtime_precondition": "ready",
                "blocked_reason": "",
                "semantic_evidence_level": "dss_rule_baseline",
                "output_dir": str(control_root / case_dir.name),
            }
        )

    created_video_count = sum(1 for r in materialized_rows if r.get("video_status") == "created")
    existing_video_count = sum(1 for r in materialized_rows if r.get("video_status") == "exists")
    failed_video_count = sum(1 for r in materialized_rows if r.get("video_status") in {"failed", "blocked", "missing_no_stub", "missing_dss"})
    control_wav_count = len(metrics_rows)
    ready_slots = sum(1 for r in manifest_rows if r.get("runtime_precondition") == "ready")
    blocked_slots = sum(1 for r in manifest_rows if r.get("runtime_precondition") == "blocked")
    synthetic_v2a_slots = sum(1 for r in manifest_rows if r.get("model_family") == "mmaudio_text_video_sync_v1" and r.get("semantic_evidence_level") == "pipeline_only_not_real_scene")

    decision = "PASS"
    if control_wav_count < len(case_dirs):
        decision = "FAIL_CONTROL_WAV_INCOMPLETE"
    elif blocked_slots > 0:
        decision = "PASS_WITH_BLOCKED_VIDEO_SLOTS"
    elif synthetic_v2a_slots > 0:
        decision = "PASS_PIPELINE_READY_WITH_SYNTHETIC_VIDEO_LIMITATION"

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "ffmpeg_available": ffmpeg_ok,
        "case_count": len(case_dirs),
        "created_video_count": created_video_count,
        "existing_video_count": existing_video_count,
        "failed_video_count": failed_video_count,
        "control_wav_count": control_wav_count,
        "ready_slots": ready_slots,
        "blocked_slots": blocked_slots,
        "synthetic_v2a_slots": synthetic_v2a_slots,
        "limitation": "Synthetic placeholder videos can validate pipeline mechanics but cannot prove real V2A semantic quality.",
        "materialized_videos": materialized_rows,
    }

    write_json(reports_root / "week17_case_media_materialization_report.json", report)
    write_csv(
        reports_root / "week17_case_media_materialization_report.csv",
        materialized_rows,
        ["case_id", "video_status", "video_path", "video_kind", "ffmpeg_message"],
    )

    write_json(
        reports_root / "week17_control_candidate_metrics.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "candidate_count": len(metrics_rows),
            "metrics": metrics_rows,
        },
    )
    write_csv(
        reports_root / "week17_control_candidate_metrics.csv",
        metrics_rows,
        ["case_id", "candidate_id", "candidate_path", "model_family", "event_count", "duration_s", "rms", "peak", "clip_rate", "silence_ratio"],
    )

    write_json(
        reports_root / "week17_mmaudio_input_manifest.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "decision": decision,
            "case_count": len(case_dirs),
            "candidate_slot_count": len(manifest_rows),
            "ready_slots": ready_slots,
            "blocked_slots": blocked_slots,
            "synthetic_v2a_slots": synthetic_v2a_slots,
            "manifest_rows": manifest_rows,
        },
    )
    write_csv(
        reports_root / "week17_mmaudio_input_manifest.csv",
        manifest_rows,
        [
            "candidate_id",
            "case_id",
            "model_family",
            "input_path",
            "dss_path",
            "expected_events_path",
            "baseline_prompt_path",
            "runtime_precondition",
            "blocked_reason",
            "semantic_evidence_level",
            "output_dir",
        ],
    )

    log_path = report_log / f"week17_case_media_materialization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "decision": decision,
        "ffmpeg_available": ffmpeg_ok,
        "case_count": len(case_dirs),
        "created_video_count": created_video_count,
        "existing_video_count": existing_video_count,
        "failed_video_count": failed_video_count,
        "control_wav_count": control_wav_count,
        "ready_slots": ready_slots,
        "blocked_slots": blocked_slots,
        "synthetic_v2a_slots": synthetic_v2a_slots,
        "log_path": str(log_path),
    }, ensure_ascii=False, indent=2))

    return 0 if decision.startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())