#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import shlex
from pathlib import Path
from typing import Any, Dict, List


VARIANT_ORDER = {
    "naive": 0,
    "naive_rich": 1,
    "dss_global": 2,
    "dss_event_timeline": 3,
    "dss_layer_avoid": 4,
}


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def read_manifest(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda r: (r.get("case_id", ""), VARIANT_ORDER.get(r.get("variant", ""), 99)))
    return rows


def sha1_short(text: str, n: int = 10) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:n]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_mmaudio_command(job: Dict[str, Any], mainbase_env: str = "${MAINBASE:?set MAINBASE}") -> str:
    duration = max(1, int(round(float(job["duration_sec"]))))
    video_abs = f'{mainbase_env}/{job["video_path"]}'
    prompt = job["prompt"]
    job_dir_abs = f'{mainbase_env}/{job["job_dir"]}'
    expected_wav_abs = f'{mainbase_env}/{job["expected_output_wav"]}'
    expected_flac_abs = f'{mainbase_env}/{job["expected_output_flac"]}'

    lines = [
        "(",
        "  set -e",
        '  : "${MMAUDIO_ROOT:?set MMAUDIO_ROOT to your local MMAudio checkout}"',
        f"  mkdir -p {shlex.quote(job_dir_abs)}",
        '  cd "$MMAUDIO_ROOT"',
        f"  BEFORE_FILE=$(find output -type f \\( -name '*.flac' -o -name '*.wav' \\) -printf '%T@ %p\\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2- || true)",
        (
            "  python demo.py "
            f"--duration={duration} "
            f"--video={shlex.quote(video_abs)} "
            f"--prompt {shlex.quote(prompt)}"
        ),
        f"  AFTER_FILE=$(find output -type f \\( -name '*.flac' -o -name '*.wav' \\) -printf '%T@ %p\\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2- || true)",
        '  if [ -z "$AFTER_FILE" ]; then echo "NO_MMAUDIO_OUTPUT_FOUND"; exit 2; fi',
        '  if [ "$AFTER_FILE" = "$BEFORE_FILE" ]; then echo "MMAUDIO_OUTPUT_NOT_CHANGED"; exit 3; fi',
        f"  cp \"$AFTER_FILE\" {shlex.quote(expected_flac_abs)}",
        f"  if command -v ffmpeg >/dev/null 2>&1; then ffmpeg -y -i {shlex.quote(expected_flac_abs)} {shlex.quote(expected_wav_abs)} >/dev/null 2>&1; fi",
        ")",
    ]
    return "\n".join(lines)


def build_t2a_boundary_command(job: Dict[str, Any]) -> str:
    return (
        "# T2A fallback is intentionally not executed by this queue script. "
        "Use a text-to-audio backend only as video_conditioned=false baseline. "
        f"job_id={job['job_id']}"
    )


def make_job(row: Dict[str, Any], output_root: Path) -> Dict[str, Any]:
    case_id = row["case_id"]
    variant = row["variant"]
    job_id = row["job_id"]
    job_dir = output_root / case_id / variant

    prompt = row.get("prompt", "")
    prompt_hash = sha1_short(prompt)

    expected_wav = Path(row.get("expected_output_wav", job_dir / f"{job_id}.wav"))
    expected_flac = job_dir / f"{job_id}.flac"
    expected_metrics = Path(row.get("expected_metrics_json", job_dir / f"{job_id}.metrics.json"))
    expected_failure = Path(row.get("expected_failure_json", job_dir / f"{job_id}.failure.json"))

    ready = (
        truthy(row.get("ready_for_generation", "False"))
        and Path(row["video_path"]).exists()
        and len(prompt.strip()) > 0
    )

    job = {
        "job_id": job_id,
        "case_id": case_id,
        "variant": variant,
        "status": "queued" if ready else "blocked",
        "block_reason": "" if ready else "not_ready_or_missing_video_or_empty_prompt",
        "video_path": row["video_path"],
        "duration_sec": float(row.get("duration_sec", 10.0)),
        "prompt": prompt,
        "prompt_hash": prompt_hash,
        "prompt_chars": int(row.get("prompt_chars", 0)),
        "event_count": int(row.get("event_count", 0)),
        "avoid_count": int(row.get("avoid_count", 0)),
        "primary_source": row.get("primary_source", "MMAudio"),
        "fallback_source": row.get("fallback_source", "StableAudioOpen_or_T2A"),
        "video_conditioned_primary": truthy(row.get("video_conditioned_primary", "True")),
        "fallback_allowed": truthy(row.get("fallback_allowed", "True")),
        "claim_boundary": row.get("claim_boundary", ""),
        "job_dir": str(job_dir),
        "request_json": str(job_dir / "request.json"),
        "expected_output_wav": str(expected_wav),
        "expected_output_flac": str(expected_flac),
        "expected_metrics_json": str(expected_metrics),
        "expected_failure_json": str(expected_failure),
        "generated_audio_claim": "not_run_yet",
        "source_boundary": {
            "MMAudio": "video_conditioned=true only after real generation from video input",
            "StableAudioOpen_or_T2A": "video_conditioned=false fallback baseline only",
        },
    }

    job["mmaudio_command_hint"] = build_mmaudio_command(job)
    job["t2a_boundary_hint"] = build_t2a_boundary_command(job)
    return job


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--output-root", default="artifacts/model_runs/w18_dss_ablation")
    parser.add_argument("--queue-jsonl", required=True)
    parser.add_argument("--queue-csv", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--commands-sh", required=True)
    args = parser.parse_args()

    manifest_path = Path(args.manifest_csv)
    output_root = Path(args.output_root)

    rows = read_manifest(manifest_path)
    jobs = [make_job(row, output_root) for row in rows]

    for job in jobs:
        job_dir = Path(job["job_dir"])
        job_dir.mkdir(parents=True, exist_ok=True)
        write_json(Path(job["request_json"]), job)

    queue_csv_rows = []
    for job in jobs:
        queue_csv_rows.append(
            {
                "job_id": job["job_id"],
                "case_id": job["case_id"],
                "variant": job["variant"],
                "status": job["status"],
                "block_reason": job["block_reason"],
                "video_path": job["video_path"],
                "prompt_chars": job["prompt_chars"],
                "event_count": job["event_count"],
                "primary_source": job["primary_source"],
                "fallback_source": job["fallback_source"],
                "video_conditioned_primary": job["video_conditioned_primary"],
                "expected_output_wav": job["expected_output_wav"],
                "expected_output_flac": job["expected_output_flac"],
                "request_json": job["request_json"],
            }
        )

    ready_jobs = [job for job in jobs if job["status"] == "queued"]
    blocked_jobs = [job for job in jobs if job["status"] != "queued"]
    case_ids = sorted({job["case_id"] for job in jobs})
    variants = sorted({job["variant"] for job in jobs})

    summary = {
        "manifest_csv": str(manifest_path),
        "output_root": str(output_root),
        "total_jobs": len(jobs),
        "queued_jobs": len(ready_jobs),
        "blocked_jobs": len(blocked_jobs),
        "case_count": len(case_ids),
        "variant_count": len(variants),
        "case_ids": case_ids,
        "variants": variants,
        "request_json_count": len(list(output_root.glob("*/*/request.json"))),
        "ready_for_generation_execution": (
            len(jobs) == 30
            and len(ready_jobs) == 30
            and len(blocked_jobs) == 0
            and len(case_ids) == 6
            and len(variants) == 5
        ),
        "claim_boundary": [
            "No generated audio is claimed at queue-build time.",
            "MMAudio claim requires real generated audio from video input.",
            "T2A fallback must remain video_conditioned=false.",
            "If ffmpeg is unavailable, FLAC may exist while WAV conversion remains pending.",
        ],
        "blocked_examples": blocked_jobs[:5],
        "outputs": {
            "queue_jsonl": args.queue_jsonl,
            "queue_csv": args.queue_csv,
            "summary_json": args.summary_json,
            "commands_sh": args.commands_sh,
        },
    }

    write_jsonl(Path(args.queue_jsonl), jobs)
    write_csv(Path(args.queue_csv), queue_csv_rows)
    write_json(Path(args.summary_json), summary)

    commands_path = Path(args.commands_sh)
    commands_path.parent.mkdir(parents=True, exist_ok=True)
    with commands_path.open("w", encoding="utf-8") as handle:
        handle.write("#!/usr/bin/env bash\n")
        handle.write("set -euo pipefail\n")
        handle.write('MAINBASE="${MAINBASE:-$HOME/work/audio_engineering_repo_skeleton_v1}"\n')
        handle.write('echo "W18 MMAudio queue runner: jobs are generated from request JSON files."\n')
        handle.write('echo "Set MMAUDIO_ROOT before running this script."\n\n')
        for job in ready_jobs:
            handle.write(f'echo "### RUN {job["job_id"]}"\n')
            handle.write(job["mmaudio_command_hint"])
            handle.write("\n\n")
    commands_path.chmod(0o755)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["ready_for_generation_execution"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
