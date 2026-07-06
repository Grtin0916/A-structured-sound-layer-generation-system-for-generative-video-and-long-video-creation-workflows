#!/usr/bin/env python3
import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any


def ffprobe_duration(path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    return float(out)


def read_queue(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue-jsonl", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    queue = read_queue(Path(args.queue_jsonl))

    by_case: Dict[str, Dict[str, Any]] = {}
    for job in queue:
        case_id = job["case_id"]
        if case_id in by_case:
            continue

        video_path = Path(job["video_path"])
        requested = float(job.get("duration_sec", 0.0))
        exists = video_path.exists()

        if exists:
            try:
                actual = ffprobe_duration(video_path)
                probe_error = ""
            except Exception as exc:
                actual = None
                probe_error = f"{type(exc).__name__}: {exc}"
        else:
            actual = None
            probe_error = "video_missing"

        if actual is None:
            recommended = requested
            duration_delta = None
            status = "blocked"
        else:
            # Match MMAudio observed behavior: use actual media duration when it is shorter.
            recommended = min(requested, actual)
            duration_delta = requested - actual
            status = "ok" if abs(duration_delta) <= 0.25 else "duration_mismatch"

        by_case[case_id] = {
            "case_id": case_id,
            "video_path": str(video_path),
            "video_exists": exists,
            "requested_duration_sec": round(requested, 4),
            "actual_video_duration_sec": round(actual, 4) if actual is not None else None,
            "recommended_generation_duration_sec": round(recommended, 4) if recommended is not None else None,
            "duration_delta_requested_minus_actual_sec": round(duration_delta, 4) if duration_delta is not None else None,
            "status": status,
            "probe_error": probe_error,
        }

    rows = list(by_case.values())

    summary = {
        "queue_jsonl": args.queue_jsonl,
        "case_count": len(rows),
        "ok_count": sum(1 for r in rows if r["status"] == "ok"),
        "duration_mismatch_count": sum(1 for r in rows if r["status"] == "duration_mismatch"),
        "blocked_count": sum(1 for r in rows if r["status"] == "blocked"),
        "ready_for_batch_duration_aligned": (
            len(rows) >= 6
            and all(r["video_exists"] for r in rows)
            and all(r["recommended_generation_duration_sec"] is not None for r in rows)
        ),
        "policy": [
            "Use actual_video_duration_sec or recommended_generation_duration_sec for generation/evaluation.",
            "Do not score event windows beyond actual media duration.",
            "Do not treat MMAudio truncation as generation failure when video is shorter than requested duration.",
        ],
        "cases": rows,
    }

    write_csv(Path(args.out_csv), rows)
    write_json(Path(args.out_json), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["ready_for_batch_duration_aligned"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
