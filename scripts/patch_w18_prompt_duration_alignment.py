#!/usr/bin/env python3
import argparse
import csv
import json
import re
import shlex
from pathlib import Path
from typing import Any, Dict, List


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def patch_prompt_duration(prompt: str, duration: float) -> str:
    # Replace only explicit duration phrases, not event timestamps.
    duration_text = f"{duration:.1f}s"
    patched = prompt

    patched = re.sub(
        r"for a \d+(?:\.\d+)?s video",
        f"for a {duration_text} video",
        patched,
    )
    patched = re.sub(
        r"for this \d+(?:\.\d+)?s video",
        f"for this {duration_text} video",
        patched,
    )
    patched = re.sub(
        r"Generate audio for a \d+(?:\.\d+)?s video",
        f"Generate audio for a {duration_text} video",
        patched,
    )
    patched = re.sub(
        r"Generate synchronized audio for a \d+(?:\.\d+)?s video",
        f"Generate synchronized audio for a {duration_text} video",
        patched,
    )
    return patched


def build_small44k_command(job: Dict[str, Any]) -> str:
    duration = max(1, int(round(float(job["duration_sec"]))))
    mainbase = "${MAINBASE:?set MAINBASE}"
    mmaudio_root = "${MMAUDIO_ROOT:?set MMAUDIO_ROOT}"
    video = f'{mainbase}/{job["video_path"]}'
    prompt = job["prompt"]
    job_dir = f'{mainbase}/{Path(job["expected_output_flac"]).parent}'
    raw_out = f'{job_dir}/mmaudio_small_44k_raw_output'
    expected_flac = f'{mainbase}/{job["expected_output_flac"]}'
    expected_wav = f'{mainbase}/{job["expected_output_wav"]}'

    return "\n".join([
        "(",
        "  set -euo pipefail",
        f"  echo '### RUN {job['job_id']} duration={duration}s variant={job['variant']}'",
        f"  mkdir -p {shlex.quote(job_dir)} {shlex.quote(raw_out)}",
        f"  cd {shlex.quote(mmaudio_root)}",
        "  test -f weights/mmaudio_small_44k.pth",
        "  test -f ext_weights/synchformer_state_dict.pth",
        "  test -f ext_weights/v1-44.pth",
        f"  python demo.py --variant small_44k --duration={duration} --video={shlex.quote(video)} --prompt {shlex.quote(prompt)} --output={shlex.quote(raw_out)} --seed=42",
        f"  NEW_AUDIO=$(find {shlex.quote(raw_out)} -type f \\( -name '*.flac' -o -name '*.wav' \\) -printf '%T@ %p\\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2- || true)",
        '  if [ -z "$NEW_AUDIO" ]; then echo "NO_MMAUDIO_OUTPUT_FOUND"; exit 2; fi',
        '  case "$NEW_AUDIO" in',
        f"    *.flac) cp \"$NEW_AUDIO\" {shlex.quote(expected_flac)} ;;",
        f"    *.wav) cp \"$NEW_AUDIO\" {shlex.quote(expected_wav)} ;;",
        "    *) echo \"UNKNOWN_AUDIO_EXT=$NEW_AUDIO\"; exit 3 ;;",
        "  esac",
        f"  if [ -f {shlex.quote(expected_flac)} ] && command -v ffmpeg >/dev/null 2>&1; then ffmpeg -y -i {shlex.quote(expected_flac)} {shlex.quote(expected_wav)} >/dev/null 2>&1 || true; fi",
        f"  ls -lh {shlex.quote(expected_flac)} {shlex.quote(expected_wav)} 2>/dev/null || true",
        ")",
    ])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-queue-jsonl", required=True)
    ap.add_argument("--out-queue-jsonl", required=True)
    ap.add_argument("--out-summary-json", required=True)
    ap.add_argument("--out-patch-csv", required=True)
    ap.add_argument("--out-runner-sh", required=True)
    args = ap.parse_args()

    jobs = read_jsonl(Path(args.in_queue_jsonl))
    patched_jobs = []
    patch_rows = []

    for job in jobs:
        duration = float(job["duration_sec"])
        old_prompt = str(job.get("prompt", ""))
        new_prompt = patch_prompt_duration(old_prompt, duration)

        new_job = dict(job)
        new_job["prompt_before_duration_patch"] = old_prompt
        new_job["prompt"] = new_prompt
        new_job["prompt_chars"] = len(new_prompt)
        new_job["prompt_duration_aligned"] = (old_prompt == new_prompt) or (f"{duration:.1f}s video" in new_prompt)

        request_path = Path(new_job["request_json"])
        if request_path.exists():
            request_path.write_text(json.dumps(new_job, ensure_ascii=False, indent=2), encoding="utf-8")

        patched_jobs.append(new_job)
        patch_rows.append({
            "job_id": new_job["job_id"],
            "case_id": new_job["case_id"],
            "variant": new_job["variant"],
            "duration_sec": duration,
            "changed": old_prompt != new_prompt,
            "old_prompt_chars": len(old_prompt),
            "new_prompt_chars": len(new_prompt),
            "old_prompt_head": old_prompt[:160],
            "new_prompt_head": new_prompt[:160],
        })

    changed = [r for r in patch_rows if r["changed"]]
    not_aligned = [
        j for j in patched_jobs
        if str(j["variant"]).startswith("dss_")
        and f"{float(j['duration_sec']):.1f}s video" not in str(j["prompt"])
        and "video" in str(j["prompt"])
    ]

    case_count = len({j["case_id"] for j in patched_jobs})
    variant_count = len({j["variant"] for j in patched_jobs})

    summary = {
        "source_queue": args.in_queue_jsonl,
        "total_jobs": len(patched_jobs),
        "case_count": case_count,
        "variant_count": variant_count,
        "prompt_changed_jobs": len(changed),
        "duration_prompt_not_aligned_count": len(not_aligned),
        "ready_for_micro_batch": (
            len(patched_jobs) == 30
            and case_count == 6
            and variant_count == 5
            and len(not_aligned) == 0
        ),
        "policy": [
            "Generation command duration and textual prompt duration must agree.",
            "This step patches only explicit duration phrases, not event timestamps.",
            "No generated audio is claimed by this patch.",
        ],
        "changed_examples": changed[:10],
    }

    write_jsonl(Path(args.out_queue_jsonl), patched_jobs)
    write_json(Path(args.out_summary_json), summary)
    write_csv(Path(args.out_patch_csv), patch_rows)

    runner = Path(args.out_runner_sh)
    runner.parent.mkdir(parents=True, exist_ok=True)
    with runner.open("w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env bash\n")
        f.write("set -euo pipefail\n")
        f.write('MAINBASE="${MAINBASE:-$HOME/work/audio_engineering_repo_skeleton_v1}"\n')
        f.write('MMAUDIO_ROOT="${MMAUDIO_ROOT:-/home/GRT/work/_external/MMAudio}"\n')
        f.write('HF_HOME="${HF_HOME:-$MMAUDIO_ROOT/.hf_cache}"\n')
        f.write('HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"\n')
        f.write('export MAINBASE MMAUDIO_ROOT HF_HOME HUGGINGFACE_HUB_CACHE\n')
        f.write('export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1\n')
        f.write("unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy\n\n")
        for job in patched_jobs:
            f.write(build_small44k_command(job))
            f.write("\n\n")

    runner.chmod(0o755)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["ready_for_micro_batch"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
