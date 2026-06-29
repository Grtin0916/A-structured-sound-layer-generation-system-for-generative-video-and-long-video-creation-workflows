#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validate Week17 DirectorSound Script case bank without external YAML dependency.
The .yaml file is written as JSON-compatible YAML by prepare_demo_cases.py.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ALLOWED_LAYER_ROLES = {"ambience", "foley", "music", "dialogue", "fx"}


def load_json_compatible_yaml(path: Path) -> dict[str, Any]:
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


def validate_case(case_dir: Path, min_events: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    case_id = case_dir.name
    required_files = [
        "director_sound_script.yaml",
        "expected_events.csv",
        "baseline_prompt.txt",
        "case_notes.md",
    ]
    for name in required_files:
        if not (case_dir / name).exists():
            errors.append({"case_id": case_id, "level": "error", "field": name, "message": "missing required file"})

    has_video = (case_dir / "input_video.mp4").exists()
    has_stub = (case_dir / "input_video_stub.json").exists()
    if not has_video and not has_stub:
        errors.append({"case_id": case_id, "level": "error", "field": "input", "message": "missing input_video.mp4 or input_video_stub.json"})
    if has_stub:
        warnings.append({"case_id": case_id, "level": "warning", "field": "input_video", "message": "stub input; V2A quality claim must remain blocked"})

    dss_path = case_dir / "director_sound_script.yaml"
    dss: dict[str, Any] = {}
    if dss_path.exists():
        try:
            dss = load_json_compatible_yaml(dss_path)
        except Exception as e:
            errors.append({"case_id": case_id, "level": "error", "field": "director_sound_script.yaml", "message": f"parse failed: {e}"})

    duration = float(dss.get("video", {}).get("duration_s", -1))
    if duration <= 0:
        errors.append({"case_id": case_id, "level": "error", "field": "video.duration_s", "message": "duration must be positive"})

    events = dss.get("events", [])
    if not isinstance(events, list) or len(events) < min_events:
        errors.append({"case_id": case_id, "level": "error", "field": "events", "message": f"event count must be >= {min_events}"})

    for i, ev in enumerate(events if isinstance(events, list) else []):
        prefix = f"events[{i}]"
        event_id = ev.get("event_id", f"event_{i}")

        time_s = ev.get("time_s")
        duration_s = ev.get("duration_s")
        if not isinstance(time_s, (int, float)):
            errors.append({"case_id": case_id, "event_id": event_id, "level": "error", "field": f"{prefix}.time_s", "message": "time_s must be numeric"})
        elif time_s < 0 or time_s > duration:
            errors.append({"case_id": case_id, "event_id": event_id, "level": "error", "field": f"{prefix}.time_s", "message": "time_s out of video duration"})

        if not isinstance(duration_s, (int, float)) or duration_s <= 0:
            errors.append({"case_id": case_id, "event_id": event_id, "level": "error", "field": f"{prefix}.duration_s", "message": "duration_s must be positive"})

        layer_role = ev.get("layer_role")
        if layer_role not in ALLOWED_LAYER_ROLES:
            errors.append({"case_id": case_id, "event_id": event_id, "level": "error", "field": f"{prefix}.layer_role", "message": f"invalid layer_role={layer_role}"})

        priority = ev.get("priority")
        if not isinstance(priority, int) or priority < 1 or priority > 5:
            errors.append({"case_id": case_id, "event_id": event_id, "level": "error", "field": f"{prefix}.priority", "message": "priority must be integer 1..5"})

        tolerance_ms = ev.get("tolerance_ms")
        if not isinstance(tolerance_ms, int) or tolerance_ms <= 0:
            errors.append({"case_id": case_id, "event_id": event_id, "level": "error", "field": f"{prefix}.tolerance_ms", "message": "tolerance_ms must be positive integer"})

        if "sound_intent" not in ev or not str(ev.get("sound_intent", "")).strip():
            errors.append({"case_id": case_id, "event_id": event_id, "level": "error", "field": f"{prefix}.sound_intent", "message": "sound_intent required"})

        if not isinstance(ev.get("avoid", []), list):
            errors.append({"case_id": case_id, "event_id": event_id, "level": "error", "field": f"{prefix}.avoid", "message": "avoid must be a list"})

    case_valid = not errors
    summary = {
        "case_id": case_id,
        "valid": case_valid,
        "event_count": len(events) if isinstance(events, list) else 0,
        "duration_s": duration,
        "has_input_video": has_video,
        "has_stub": has_stub,
        "error_count": len(errors),
        "warning_count": len(warnings),
    }
    return summary, errors + warnings


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases-root", default="cases")
    ap.add_argument("--reports-root", default="reports")
    ap.add_argument("--min-cases", type=int, default=6)
    ap.add_argument("--min-events", type=int, default=3)
    args = ap.parse_args()

    cases_root = Path(args.cases_root)
    reports_root = Path(args.reports_root)
    case_dirs = sorted([p for p in cases_root.iterdir() if p.is_dir()]) if cases_root.exists() else []

    case_summaries: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []

    for case_dir in case_dirs:
        summary, case_issues = validate_case(case_dir, args.min_events)
        case_summaries.append(summary)
        issues.extend(case_issues)

    valid_case_count = sum(1 for c in case_summaries if c["valid"])
    invalid_case_count = len(case_summaries) - valid_case_count
    error_count = sum(1 for x in issues if x.get("level") == "error")
    warning_count = sum(1 for x in issues if x.get("level") == "warning")
    event_total = sum(int(c["event_count"]) for c in case_summaries)

    decision = "PASS" if valid_case_count >= args.min_cases and invalid_case_count == 0 and error_count == 0 else "FAIL"

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "case_count": len(case_summaries),
        "validCaseCount": valid_case_count,
        "invalidCaseCount": invalid_case_count,
        "event_total": event_total,
        "error_count": error_count,
        "warning_count": warning_count,
        "case_summaries": case_summaries,
        "issues": issues,
    }

    write_json(reports_root / "dss_validation_report.json", report)
    write_csv(
        reports_root / "dss_validation_report.csv",
        case_summaries,
        ["case_id", "valid", "event_count", "duration_s", "has_input_video", "has_stub", "error_count", "warning_count"],
    )
    write_csv(
        reports_root / "dss_validation_issues.csv",
        issues,
        ["case_id", "event_id", "level", "field", "message"],
    )

    print(json.dumps({k: report[k] for k in ["decision", "case_count", "validCaseCount", "invalidCaseCount", "event_total", "error_count", "warning_count"]}, ensure_ascii=False, indent=2))
    return 0 if decision == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())