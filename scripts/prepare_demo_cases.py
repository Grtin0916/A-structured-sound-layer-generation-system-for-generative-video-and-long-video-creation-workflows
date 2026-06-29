#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare Week17 demo case bank for Director-guided Video-to-Audio workflow.

Outputs:
- cases/<case_id>/director_sound_script.yaml  (JSON-compatible YAML)
- cases/<case_id>/expected_events.csv
- cases/<case_id>/baseline_prompt.txt
- cases/<case_id>/case_notes.md
- input_video.mp4 if a local source video is found, else input_video_stub.json
- reports/demo_cases_inventory.json/csv
- reports/week17_mmaudio_input_manifest.json/csv
- reports/week17_case_source_gap_report.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CASE_SPECS = [
    {
        "case_id": "street_rain_crosswalk_001",
        "duration_s": 9.0,
        "scene": "night street crosswalk with rain, traffic movement, and pedestrian footsteps",
        "baseline_prompt": "Create synchronized rain ambience, wet footsteps, a passing car, and a short puddle splash for a rainy night crosswalk.",
        "events": [
            ("rain_ambience_start", 0.0, 9.0, "ambience", "steady light rain and wet street bed", 2, 500),
            ("footsteps_enter", 1.2, 2.5, "foley", "wet shoes stepping through shallow water", 4, 180),
            ("car_pass_left_to_right", 4.0, 1.8, "foley", "muffled car pass-by on wet road", 3, 250),
            ("puddle_splash", 6.7, 0.5, "foley", "single sharp puddle splash near camera", 5, 120),
        ],
        "avoid": ["speech", "music", "thunder", "siren"],
    },
    {
        "case_id": "kitchen_chop_sizzle_001",
        "duration_s": 10.0,
        "scene": "close-up kitchen preparation with knife chopping, pan heat, and plate movement",
        "baseline_prompt": "Generate kitchen Foley with clean knife chops, pan sizzle, light utensil handling, and a plate set-down.",
        "events": [
            ("room_tone", 0.0, 10.0, "ambience", "small indoor kitchen room tone", 1, 500),
            ("knife_chops", 1.0, 3.2, "foley", "rhythmic vegetable chopping on wooden board", 5, 100),
            ("pan_sizzle_rise", 4.8, 3.0, "foley", "oil and food sizzling in pan", 4, 250),
            ("plate_set_down", 8.5, 0.4, "foley", "ceramic plate placed on counter", 3, 120),
        ],
        "avoid": ["speech", "restaurant crowd", "music"],
    },
    {
        "case_id": "robot_warehouse_pick_001",
        "duration_s": 11.0,
        "scene": "small warehouse robot arm picking a box from a shelf and confirming the task",
        "baseline_prompt": "Create mechanical servo movement, soft warehouse ambience, cardboard handling, and a short confirmation beep.",
        "events": [
            ("warehouse_hum", 0.0, 11.0, "ambience", "low warehouse ventilation hum", 1, 600),
            ("servo_arm_move", 1.5, 2.2, "foley", "precise electric servo arm movement", 4, 180),
            ("box_lift", 4.6, 1.2, "foley", "cardboard box lifted from shelf", 5, 150),
            ("confirm_beep", 7.4, 0.25, "fx", "short clean robot confirmation beep", 3, 80),
        ],
        "avoid": ["human speech", "alarm", "music"],
    },
    {
        "case_id": "forest_bird_branch_001",
        "duration_s": 8.0,
        "scene": "quiet forest shot with wind, bird movement, and a small branch crack",
        "baseline_prompt": "Generate soft forest ambience, bird chirps, leaf rustle, and a clear small branch crack synchronized to the visible motion.",
        "events": [
            ("forest_wind", 0.0, 8.0, "ambience", "soft wind through leaves", 2, 500),
            ("bird_chirp", 1.0, 0.8, "foley", "two short bird chirps", 3, 150),
            ("leaf_rustle", 3.4, 1.4, "foley", "small animal or bird rustling leaves", 4, 180),
            ("branch_crack", 5.8, 0.35, "foley", "dry branch crack", 5, 100),
        ],
        "avoid": ["speech", "music", "waterfall"],
    },
    {
        "case_id": "subway_arrival_door_001",
        "duration_s": 12.0,
        "scene": "subway platform as train arrives, brakes, and doors open",
        "baseline_prompt": "Create subway platform ambience, train rumble, brake squeal, door opening chime, and light crowd bed without dialogue.",
        "events": [
            ("platform_bed", 0.0, 12.0, "ambience", "underground platform room tone and distant crowd", 2, 700),
            ("train_rumble_approach", 1.2, 4.5, "foley", "subway train approaching with low rumble", 4, 250),
            ("brake_squeal", 5.7, 1.2, "foley", "metal brake squeal as train stops", 5, 150),
            ("door_open_chime", 8.2, 0.5, "fx", "door open chime and sliding door start", 3, 120),
        ],
        "avoid": ["clear speech", "music", "announcement voice"],
    },
    {
        "case_id": "glass_drop_room_001",
        "duration_s": 7.0,
        "scene": "indoor tabletop shot where a glass object drops and breaks",
        "baseline_prompt": "Generate quiet room tone, a glass slip, sharp impact, glass shatter, and a short reflective tail.",
        "events": [
            ("quiet_room_tone", 0.0, 7.0, "ambience", "quiet indoor room tone", 1, 500),
            ("glass_slip", 1.5, 0.7, "foley", "glass sliding off table edge", 4, 120),
            ("impact", 2.5, 0.15, "foley", "sharp glass impact on hard floor", 5, 70),
            ("shatter_tail", 2.65, 1.1, "foley", "small glass fragments scattering", 5, 100),
        ],
        "avoid": ["speech", "music", "cartoon effect"],
    },
]


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def find_source_videos(search_roots: list[Path], cases_root: Path) -> list[Path]:
    suffixes = {".mp4", ".mov", ".mkv", ".webm"}
    videos: list[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in suffixes:
                continue
            if cases_root in p.parents:
                continue
            if ".git" in p.parts:
                continue
            videos.append(p)
    return sorted(set(videos), key=lambda x: str(x))


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


def build_dss(spec: dict[str, Any], input_kind: str, input_path: str) -> dict[str, Any]:
    events = []
    for name, time_s, duration_s, layer_role, intent, priority, tolerance_ms in spec["events"]:
        events.append(
            {
                "event_id": name,
                "time_s": float(time_s),
                "duration_s": float(duration_s),
                "layer_role": layer_role,
                "object": name.split("_")[0],
                "action": name,
                "sound_intent": intent,
                "priority": int(priority),
                "tolerance_ms": int(tolerance_ms),
                "avoid": spec["avoid"],
            }
        )

    return {
        "schema_version": "dss.v0.1",
        "case_id": spec["case_id"],
        "video": {
            "duration_s": float(spec["duration_s"]),
            "input_kind": input_kind,
            "path": input_path,
            "expected_silence": True,
        },
        "scene": spec["scene"],
        "events": events,
        "layers": {
            "ambience": {"role": "continuous bed", "priority": 1},
            "foley": {"role": "synchronized event sounds", "priority": 5},
            "music": {"role": "disabled unless explicitly requested", "priority": 0},
        },
        "constraints": {
            "avoid": spec["avoid"],
            "loudness": {"target_lufs": -18, "peak_ceiling_dbfs": -1.0},
            "style": "realistic Foley, no speech, no background music",
        },
        "evaluation_targets": {
            "onset_error_ms": 250,
            "event_coverage_min": 0.75,
            "clip_rate_max": 0.001,
            "silence_ratio_max": 0.35,
        },
        "candidate_policy": {
            "mmaudio": {"enabled": True, "needs_input_video": True},
            "text_audio": {"enabled": True, "needs_input_video": False},
            "control_foley": {"enabled": True, "needs_input_video": False},
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases-root", default="cases")
    ap.add_argument("--reports-root", default="reports")
    ap.add_argument("--min-cases", type=int, default=6)
    ap.add_argument("--min-events", type=int, default=3)
    ap.add_argument("--target-real-videos", type=int, default=2)
    args = ap.parse_args()

    repo = Path.cwd()
    cases_root = repo / args.cases_root
    reports_root = repo / args.reports_root
    cases_root.mkdir(parents=True, exist_ok=True)
    reports_root.mkdir(parents=True, exist_ok=True)

    source_roots = [repo / "data", repo / "assets", repo / "examples", repo / "artifacts", repo / "results"]
    source_videos = find_source_videos(source_roots, cases_root)

    inventory_rows: list[dict[str, Any]] = []
    event_rows_all: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    source_gaps: list[dict[str, Any]] = []

    real_video_count = 0
    event_total = 0

    for idx, spec in enumerate(CASE_SPECS[: args.min_cases]):
        case_dir = cases_root / spec["case_id"]
        case_dir.mkdir(parents=True, exist_ok=True)

        source_video = source_videos[idx] if idx < len(source_videos) else None
        if source_video:
            dst = case_dir / "input_video.mp4"
            shutil.copy2(source_video, dst)
            input_kind = "local_video"
            input_path = str(dst.relative_to(repo))
            source_video_ref = str(source_video.relative_to(repo)) if source_video.is_relative_to(repo) else str(source_video)
            real_video_count += 1
        else:
            stub = {
                "case_id": spec["case_id"],
                "reason": "No local source video was found in data/assets/examples/artifacts/results.",
                "expected_duration_s": spec["duration_s"],
                "scene": spec["scene"],
                "fallback_policy": "Use T2A/control candidate first; replace with real/synthetic silent mp4 before V2A conclusion.",
            }
            write_json(case_dir / "input_video_stub.json", stub)
            input_kind = "stub"
            input_path = str((case_dir / "input_video_stub.json").relative_to(repo))
            source_video_ref = ""
            source_gaps.append(
                {
                    "case_id": spec["case_id"],
                    "gap": "missing_input_video_mp4",
                    "impact": "MMAudio/V2A slot is blocked until input_video.mp4 exists; T2A/control slots remain usable.",
                }
            )

        dss = build_dss(spec, input_kind, input_path)
        write_json(case_dir / "director_sound_script.yaml", dss)

        expected_rows = []
        for ev in dss["events"]:
            row = {
                "case_id": spec["case_id"],
                "event_id": ev["event_id"],
                "time_s": ev["time_s"],
                "duration_s": ev["duration_s"],
                "layer_role": ev["layer_role"],
                "sound_intent": ev["sound_intent"],
                "priority": ev["priority"],
                "tolerance_ms": ev["tolerance_ms"],
            }
            expected_rows.append(row)
            event_rows_all.append(row)
        write_csv(
            case_dir / "expected_events.csv",
            expected_rows,
            ["case_id", "event_id", "time_s", "duration_s", "layer_role", "sound_intent", "priority", "tolerance_ms"],
        )
        event_total += len(expected_rows)

        (case_dir / "baseline_prompt.txt").write_text(spec["baseline_prompt"] + "\n", encoding="utf-8")
        (case_dir / "case_notes.md").write_text(
            "\n".join(
                [
                    f"# {spec['case_id']}",
                    "",
                    f"- Scene: {spec['scene']}",
                    f"- Duration: {spec['duration_s']}s",
                    f"- Input kind: {input_kind}",
                    f"- Source video: {source_video_ref or 'SOURCE_GAP'}",
                    "- Engineering note: DSS is the control contract; not a quality claim.",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        inventory_rows.append(
            {
                "case_id": spec["case_id"],
                "duration_s": spec["duration_s"],
                "event_count": len(expected_rows),
                "input_kind": input_kind,
                "input_path": input_path,
                "source_video": source_video_ref,
                "dss_path": str((case_dir / "director_sound_script.yaml").relative_to(repo)),
                "expected_events_path": str((case_dir / "expected_events.csv").relative_to(repo)),
                "baseline_prompt_path": str((case_dir / "baseline_prompt.txt").relative_to(repo)),
                "case_notes_path": str((case_dir / "case_notes.md").relative_to(repo)),
            }
        )

        for model_family in ["mmaudio_text_video_sync_v1", "control_rule_foley_v0"]:
            needs_video = model_family.startswith("mmaudio")
            blocked_reason = "missing_input_video_mp4" if needs_video and input_kind == "stub" else ""
            manifest_rows.append(
                {
                    "candidate_id": f"{spec['case_id']}__{model_family}",
                    "case_id": spec["case_id"],
                    "model_family": model_family,
                    "input_path": input_path,
                    "dss_path": str((case_dir / "director_sound_script.yaml").relative_to(repo)),
                    "expected_events_path": str((case_dir / "expected_events.csv").relative_to(repo)),
                    "baseline_prompt_path": str((case_dir / "baseline_prompt.txt").relative_to(repo)),
                    "runtime_precondition": "blocked" if blocked_reason else "ready",
                    "blocked_reason": blocked_reason,
                    "output_dir": f"artifacts/model_runs/week17_mmaudio/{spec['case_id']}",
                }
            )

    stub_video_count = len(inventory_rows) - real_video_count
    candidate_slot_count = len(manifest_rows)
    source_gap_count = len(source_gaps)

    decision = "PASS"
    if len(inventory_rows) < args.min_cases or event_total < args.min_cases * args.min_events:
        decision = "FAIL_INCOMPLETE_CASE_BANK"
    elif real_video_count < args.target_real_videos:
        decision = "PASS_WITH_SOURCE_GAP"

    inventory = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "case_count": len(inventory_rows),
        "event_total": event_total,
        "real_video_count": real_video_count,
        "stub_video_count": stub_video_count,
        "source_gap_count": source_gap_count,
        "candidate_slot_count": candidate_slot_count,
        "cases_root": args.cases_root,
        "failure_explanation": (
            "Fewer real input videos than target; V2A claims must remain blocked for stub cases."
            if real_video_count < args.target_real_videos
            else "No blocking source gap detected."
        ),
        "case_hash": sha1_text(json.dumps(inventory_rows, ensure_ascii=False, sort_keys=True)),
        "cases": inventory_rows,
    }

    write_json(reports_root / "demo_cases_inventory.json", inventory)
    write_csv(
        reports_root / "demo_cases_inventory.csv",
        inventory_rows,
        [
            "case_id",
            "duration_s",
            "event_count",
            "input_kind",
            "input_path",
            "source_video",
            "dss_path",
            "expected_events_path",
            "baseline_prompt_path",
            "case_notes_path",
        ],
    )

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "decision": "READY_WITH_BLOCKED_V2A_SLOTS" if source_gap_count else "READY",
        "candidate_slot_count": candidate_slot_count,
        "case_count": len(inventory_rows),
        "source_gap_count": source_gap_count,
        "manifest_rows": manifest_rows,
    }
    write_json(reports_root / "week17_mmaudio_input_manifest.json", manifest)
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
            "output_dir",
        ],
    )

    write_json(
        reports_root / "week17_case_source_gap_report.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "decision": "SOURCE_GAP_PRESENT" if source_gaps else "NO_SOURCE_GAP",
            "source_gap_count": source_gap_count,
            "gaps": source_gaps,
        },
    )

    print(json.dumps({k: inventory[k] for k in ["decision", "case_count", "event_total", "real_video_count", "stub_video_count", "candidate_slot_count"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())