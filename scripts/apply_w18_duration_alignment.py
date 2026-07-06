#!/usr/bin/env python3
import argparse
import csv
import json
import shlex
from pathlib import Path
from typing import Any, Dict, List


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_small44k_command(job: Dict[str, Any], duration: int) -> str:
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
    ap.add_argument("--queue-jsonl", required=True)
    ap.add_argument("--duration-probe-json", required=True)
    ap.add_argument("--out-queue-jsonl", required=True)
    ap.add_argument("--out-queue-csv", required=True)
    ap.add_argument("--out-summary-json", required=True)
    ap.add_argument("--out-runner-sh", required=True)
    args = ap.parse_args()

    queue = read_jsonl(Path(args.queue_jsonl))
    probe = read_json(Path(args.duration_probe_json))

    by_case = {
        row["case_id"]: row
        for row in probe.get("cases", [])
    }

    aligned = []
    changes = []

    for job in queue:
        case_id = job["case_id"]
        case_probe = by_case.get(case_id)
        if not case_probe:
            raise SystemExit(f"missing duration probe for case_id={case_id}")

        old_duration = float(job.get("duration_sec", 0.0))
        recommended = float(case_probe["recommended_generation_duration_sec"])

        new_job = dict(job)
        new_job["requested_duration_sec_original"] = old_duration
        new_job["actual_video_duration_sec"] = case_probe["actual_video_duration_sec"]
        new_job["duration_sec"] = recommended
        new_job["duration_alignment_policy"] = "min(requested_duration_sec_original, actual_video_duration_sec)"
        new_job["duration_alignment_status"] = case_probe["status"]
        new_job["duration_changed"] = abs(old_duration - recommended) > 1e-6

        aligned.append(new_job)

        changes.append({
            "job_id": new_job["job_id"],
            "case_id": case_id,
            "variant": new_job["variant"],
            "old_duration_sec": old_duration,
            "actual_video_duration_sec": case_probe["actual_video_duration_sec"],
            "new_duration_sec": recommended,
            "changed": new_job["duration_changed"],
        })

        request_json = Path(new_job["request_json"])
        if request_json.exists():
            request_json.write_text(json.dumps(new_job, ensure_ascii=False, indent=2), encoding="utf-8")

    changed_jobs = [x for x in changes if x["changed"]]
    case_count = len({x["case_id"] for x in aligned})
    variant_count = len({x["variant"] for x in aligned})

    summary = {
        "source_queue": args.queue_jsonl,
        "duration_probe": args.duration_probe_json,
        "total_jobs": len(aligned),
        "case_count": case_count,
        "variant_count": variant_count,
        "duration_changed_jobs": len(changed_jobs),
        "unchanged_jobs": len(aligned) - len(changed_jobs),
        "ready_for_duration_aligned_batch": (
            len(aligned) == 30
            and case_count == 6
            and variant_count == 5
            and all(float(j["duration_sec"]) > 0 for j in aligned)
        ),
        "policy": [
            "Generation duration is aligned to min(original requested duration, actual video duration).",
            "Metrics must not score event windows beyond actual media duration.",
            "This file updates request.json metadata but does not claim new generated audio.",
        ],
        "changes_head": changes[:15],
    }

    write_jsonl(Path(args.out_queue_jsonl), aligned)
    write_csv(Path(args.out_queue_csv), changes)
    write_json(Path(args.out_summary_json), summary)

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

        for job in aligned:
            duration = max(1, int(round(float(job["duration_sec"]))))
            f.write(build_small44k_command(job, duration))
            f.write("\n\n")

    runner.chmod(0o755)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["ready_for_duration_aligned_batch"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
