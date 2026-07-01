#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import time
import wave
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
EXP = ROOT / "experiments" / "mmaudio_true_replacement_2026_06_30"
CANDIDATES = EXP / "candidates"

TARGET_CASE = os.environ.get("MMAUDIO_TARGET_CASE", "glass_drop_room_001")
MMAUDIO_REPO = Path(os.environ.get("MMAUDIO_REPO", str(Path.home() / "work" / "_external" / "MMAudio")))


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_prompt(candidate_id: str) -> str:
    p = REPORTS / "mmaudio_prompt_manifest.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    for item in data.get("prompts", []):
        if item.get("candidate_id") == candidate_id:
            return item.get("prompt", "")
    raise RuntimeError(f"PROMPT_NOT_FOUND: {candidate_id}")


def latest_audio_file(search_dir: Path, start_ts: float) -> Path | None:
    if not search_dir.exists():
        return None
    candidates: List[Path] = []
    for ext in ("*.wav", "*.flac", "*.mp3", "*.m4a"):
        candidates.extend(search_dir.rglob(ext))
    candidates = [p for p in candidates if p.stat().st_mtime >= start_ts - 2]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def ffmpeg_to_wav(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-i", str(src), "-ac", "1", "-ar", "16000", str(dst)]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-1200:])


def wav_metrics(path: Path) -> Dict[str, Any]:
    import numpy as np

    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        n = wf.getnframes()
        data = wf.readframes(n)

    pcm = np.frombuffer(data, dtype="<i2").astype("float32") / 32768.0
    if ch > 1:
        pcm = pcm.reshape(-1, ch).mean(axis=1)

    duration = len(pcm) / sr if sr else 0.0
    rms = float(np.sqrt((pcm * pcm).mean())) if len(pcm) else 0.0
    peak = float(abs(pcm).max()) if len(pcm) else 0.0
    clip_rate = float((abs(pcm) >= 0.98).mean()) if len(pcm) else 1.0
    silence_ratio = float((abs(pcm) < 1e-4).mean()) if len(pcm) else 1.0

    return {
        "readable": True,
        "duration_sec": round(duration, 4),
        "sample_rate": sr,
        "channels": 1,
        "rms": rms,
        "peak": peak,
        "clip_rate": clip_rate,
        "silence_ratio": silence_ratio,
    }


def main() -> int:
    CANDIDATES.mkdir(parents=True, exist_ok=True)

    candidate_id = f"{TARGET_CASE}__mmaudio__true_replacement_v0"
    prompt_candidate_id = f"{TARGET_CASE}__mmaudio__dss_avoid_priority"
    video = ROOT / "cases" / TARGET_CASE / "input_video.mp4"
    prompt = read_prompt(prompt_candidate_id)

    out_wav = CANDIDATES / f"{candidate_id}.wav"
    summary_path = REPORTS / "mmaudio_true_one_attempt_summary.json"
    metrics_csv = REPORTS / "mmaudio_true_one_attempt_metrics.csv"
    failures_json = REPORTS / "mmaudio_true_one_attempt_failures.json"

    failure = None
    status = "not_started"
    start = time.time()

    if not video.exists():
        failure = f"VIDEO_NOT_FOUND: {video}"
        status = "blocked"
    elif not (MMAUDIO_REPO / "demo.py").exists():
        failure = f"MMAUDIO_DEMO_NOT_FOUND: {MMAUDIO_REPO / 'demo.py'}"
        status = "blocked"
    elif not shutil.which("ffmpeg"):
        failure = "FFMPEG_NOT_FOUND"
        status = "blocked"
    else:
        cmd = [
            "python",
            "demo.py",
            "--duration=8",
            f"--video={video}",
            "--prompt",
            prompt,
        ]
        proc = subprocess.run(
            cmd,
            cwd=str(MMAUDIO_REPO),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=1800,
            check=False,
        )
        if proc.returncode != 0:
            failure = {
                "reason": "MMAUDIO_DEMO_NONZERO",
                "returncode": proc.returncode,
                "stderr_tail": proc.stderr[-2000:],
                "stdout_tail": proc.stdout[-1000:],
            }
            status = "blocked"
        else:
            produced = latest_audio_file(MMAUDIO_REPO / "output", start)
            if produced is None:
                failure = "MMAUDIO_FINISHED_BUT_NO_AUDIO_FOUND_UNDER_OUTPUT"
                status = "blocked"
            else:
                ffmpeg_to_wav(produced, out_wav)
                status = "true_mmaudio_generated"

    elapsed = round(time.time() - start, 3)

    if out_wav.exists():
        metrics = wav_metrics(out_wav)
    else:
        metrics = {
            "readable": False,
            "duration_sec": 0,
            "sample_rate": 0,
            "channels": 0,
            "rms": 0,
            "peak": 0,
            "clip_rate": 1,
            "silence_ratio": 1,
        }

    row = {
        "candidate_id": candidate_id,
        "case_id": TARGET_CASE,
        "model": "MMAudio",
        "status": status,
        "video_conditioned": status == "true_mmaudio_generated",
        "fallback_used": False,
        "blocked_reason": failure,
        "output_wav": rel(out_wav),
        "runtime_sec": elapsed,
        **metrics,
    }

    with metrics_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(row.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)

    failures = [] if status == "true_mmaudio_generated" else [row]
    failures_json.write_text(json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {
        "decision": "GREEN_TRUE_MMAUDIO_ONE_REPLACEMENT_READY" if status == "true_mmaudio_generated" else "YELLOW_TRUE_MMAUDIO_ONE_REPLACEMENT_BLOCKED",
        "target_case": TARGET_CASE,
        "candidate_id": candidate_id,
        "status": status,
        "mmaudio_repo": str(MMAUDIO_REPO),
        "demo_py_exists": (MMAUDIO_REPO / "demo.py").exists(),
        "video": rel(video),
        "output_wav": rel(out_wav),
        "runtime_sec": elapsed,
        "readable": metrics["readable"],
        "blocked_reason": failure,
        "claim_boundary": {
            "can_claim_true_mmaudio_v2a_success_for_one_case": status == "true_mmaudio_generated",
            "can_claim_batch_true_mmaudio_success": False,
            "can_replace_all_fallback_winners": False,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
