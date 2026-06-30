from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

from src.soundlayer.models.base_audio_runner import (
    read_wav_metrics,
    synthesize_control_audio,
    write_wav_mono,
)


def parse_event_times_from_prompt(prompt: str, duration_sec: float) -> List[float]:
    # First try t=1.20 style.
    vals = []
    for m in re.finditer(r"t\s*=\s*([0-9]+(?:\.[0-9]+)?)", prompt):
        try:
            vals.append(float(m.group(1)))
        except ValueError:
            pass
    if vals:
        return [v for v in vals if 0 <= v <= duration_sec]

    # Fallback: spread 4 events over the clip.
    k = 4
    return [duration_sec * (i + 1) / (k + 1) for i in range(k)]


def try_local_mmaudio(row: Dict[str, str], out_wav: Path, timeout_sec: int) -> Tuple[str, str, float]:
    repo = os.environ.get("MMAUDIO_REPO", "").strip()
    if not repo:
        return "blocked", "mmaudio_repo_env_missing", 0.0

    demo = Path(repo) / "demo.py"
    if not demo.exists():
        return "blocked", f"mmaudio_demo_missing:{demo}", 0.0

    # We do not assume MMAudio output naming. Run command and then inspect output dirs.
    prompt = row["command"].split("--prompt ", 1)[-1] if "--prompt " in row["command"] else ""
    cmd = [
        "python",
        str(demo),
        f"--duration={float(row['duration_sec']):.2f}",
        "--video",
        row["input_video"],
        "--prompt",
        prompt.strip("'\""),
    ]

    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=repo,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return "blocked", "local_mmaudio_timeout", float(time.time() - start)
    except Exception as e:
        return "blocked", f"local_mmaudio_exception:{repr(e)}", float(time.time() - start)

    elapsed = float(time.time() - start)
    if proc.returncode != 0:
        reason = (proc.stderr or proc.stdout or "local_mmaudio_nonzero").strip()[-800:]
        return "blocked", reason, elapsed

    # Conservative: if we cannot locate generated audio, mark blocked instead of faking.
    candidates = []
    for root in [Path(repo) / "output", Path(repo) / "outputs", Path(repo)]:
        if root.exists():
            candidates.extend(list(root.rglob("*.wav")))
            candidates.extend(list(root.rglob("*.flac")))

    if not candidates:
        return "blocked", "local_mmaudio_finished_but_no_audio_found", elapsed

    newest = max(candidates, key=lambda p: p.stat().st_mtime)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(newest, out_wav)
    return "generated", f"copied_from:{newest}", elapsed


def run_candidate(row: Dict[str, str], timeout_sec: int = 900) -> Dict[str, object]:
    out_wav = Path(row["expected_output_wav"])
    duration = float(row["duration_sec"])
    start = time.time()

    status, reason, elapsed = try_local_mmaudio(row, out_wav, timeout_sec=timeout_sec)

    fallback_used = False
    video_conditioned = True

    if status != "generated":
        fallback_used = True
        video_conditioned = False
        event_times = parse_event_times_from_prompt(row.get("command", ""), duration)
        variant_seed = abs(hash(row["candidate_id"])) % 10000
        audio = synthesize_control_audio(duration, event_times, variant_seed=variant_seed)
        write_wav_mono(out_wav, audio, sample_rate=16000)
        status = "fallback_control_generated"

    metrics = read_wav_metrics(out_wav)
    total_elapsed = float(time.time() - start)

    return {
        "candidate_id": row["candidate_id"],
        "case_id": row["case_id"],
        "model": row["model"],
        "prompt_variant": row["prompt_variant"],
        "status": status,
        "fallback_used": fallback_used,
        "video_conditioned": video_conditioned,
        "blocked_reason": None if not fallback_used else reason,
        "output_wav": str(out_wav),
        "runtime_sec": round(elapsed if elapsed else total_elapsed, 4),
        **asdict(metrics),
    }


def run_queue_csv(queue_csv: Path, out_dir: Path, max_candidates: int = 12) -> List[Dict[str, object]]:
    rows: List[Dict[str, str]]
    with queue_csv.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    results = []
    for row in rows[:max_candidates]:
        results.append(run_candidate(row))
    return results