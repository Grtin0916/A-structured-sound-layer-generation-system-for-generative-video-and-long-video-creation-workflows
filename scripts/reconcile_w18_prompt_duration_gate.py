#!/usr/bin/env python3
import json
import re
from pathlib import Path

QUEUE = Path("artifacts/model_runs/w18_dss_ablation/generation_queue_duration_prompt_aligned_20260706.jsonl")
SUMMARY = Path("reports/w18_prompt_duration_alignment_summary_20260706.json")
OUT = Path("reports/w18_prompt_duration_gate_recheck_20260706.json")

jobs = [json.loads(x) for x in QUEUE.read_text(encoding="utf-8").splitlines() if x.strip()]

mismatches = []
checked = []

for job in jobs:
    variant = str(job.get("variant", ""))
    prompt = str(job.get("prompt", ""))
    duration = float(job.get("duration_sec", 0.0))

    if not variant.startswith("dss_"):
        continue

    # Only match explicit video duration phrases, not event timestamps such as 0.00s:
    found = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)s video", prompt)]
    ok = bool(found) and all(abs(x - duration) <= 1e-6 for x in found)

    row = {
        "job_id": job["job_id"],
        "case_id": job["case_id"],
        "variant": variant,
        "duration_sec": duration,
        "explicit_video_durations": found,
        "ok": ok,
        "prompt_head": prompt[:180],
    }
    checked.append(row)
    if not ok:
        mismatches.append(row)

summary = {
    "source_queue": str(QUEUE),
    "checked_dss_jobs": len(checked),
    "mismatch_count": len(mismatches),
    "ready_for_micro_batch": len(jobs) == 30 and len(mismatches) == 0,
    "mismatches": mismatches,
    "checked_head": checked[:12],
    "policy": [
        "DSS prompts must include explicit video duration.",
        "All explicit video duration phrases must match job duration_sec.",
        "Event timestamps such as 0.00s are ignored because they are not video duration phrases."
    ],
}
OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))

# Patch the existing summary only after independent recheck.
if SUMMARY.exists():
    obj = json.loads(SUMMARY.read_text(encoding="utf-8"))
    obj["duration_prompt_not_aligned_count_before_recheck"] = obj.get("duration_prompt_not_aligned_count")
    obj["duration_prompt_not_aligned_count"] = len(mismatches)
    obj["ready_for_micro_batch"] = summary["ready_for_micro_batch"]
    obj["gate_rechecked_by"] = str(OUT)
    SUMMARY.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
