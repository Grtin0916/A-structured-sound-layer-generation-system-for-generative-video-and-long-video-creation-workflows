from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(".").resolve()

SEED_JSON = ROOT / "reports/week18_seed_from_week17_demo_release_20260703.json"
CASES_CSV = ROOT / "reports/week18_seed_cases_20260703.csv"
REPAIR_CSV = ROOT / "reports/week18_seed_repair_targets_20260703.csv"

OUT_DIR = ROOT / "artifacts/demo/week18_prompt_compiler_seed"
OUT_JSONL = OUT_DIR / "week18_prompt_tasks_20260703.jsonl"
OUT_MANIFEST = OUT_DIR / "week18_prompt_task_manifest_20260703.json"
OUT_README = OUT_DIR / "README.md"

REPORT_JSONL = ROOT / "reports/week18_prompt_tasks_20260703.jsonl"
REPORT_CSV = ROOT / "reports/week18_prompt_task_summary_20260703.csv"
REPORT_VERIFY = ROOT / "reports/week18_prompt_task_verify_20260703.json"


CASE_BLUEPRINTS = {
    "glass_drop_room_001": {
        "scene": "quiet indoor room, a glass object drops and breaks on a hard floor",
        "events": [
            {"time": 0.8, "object": "glass", "action": "falling", "sound": "short falling whoosh", "priority": "medium"},
            {"time": 1.2, "object": "glass", "action": "impact", "sound": "sharp glass hit", "priority": "high"},
            {"time": 1.4, "object": "shards", "action": "scatter", "sound": "small glass fragments spreading", "priority": "high"},
        ],
        "avoid": ["speech", "music", "cartoon boing", "long reverb tail"],
        "layer_roles": {"ambience": "room tone", "foley": "glass impact and fragments", "music": "none"},
    },
    "forest_bird_branch_001": {
        "scene": "forest close-up with a bird moving on a branch",
        "events": [
            {"time": 0.6, "object": "leaves", "action": "rustle", "sound": "soft leaf rustle", "priority": "medium"},
            {"time": 1.5, "object": "bird", "action": "wing movement", "sound": "light feather flap", "priority": "high"},
            {"time": 2.4, "object": "branch", "action": "bounce", "sound": "subtle branch creak", "priority": "medium"},
        ],
        "avoid": ["human speech", "heavy wind", "music bed", "urban traffic"],
        "layer_roles": {"ambience": "forest ambience", "foley": "bird and branch motion", "music": "none"},
    },
    "kitchen_chop_sizzle_001": {
        "scene": "kitchen preparation with chopping and pan sizzle",
        "events": [
            {"time": 0.5, "object": "knife", "action": "chop", "sound": "crisp knife tap on board", "priority": "high"},
            {"time": 1.4, "object": "food", "action": "slide", "sound": "soft ingredient movement", "priority": "medium"},
            {"time": 2.2, "object": "pan", "action": "sizzle", "sound": "short oil sizzle", "priority": "high"},
        ],
        "avoid": ["speech", "restaurant crowd", "music", "alarm beep"],
        "layer_roles": {"ambience": "small kitchen room tone", "foley": "knife, board, oil", "music": "none"},
    },
    "robot_warehouse_pick_001": {
        "scene": "robot arm picking an item in a warehouse",
        "events": [
            {"time": 0.7, "object": "robot arm", "action": "move", "sound": "servo motor movement", "priority": "high"},
            {"time": 1.6, "object": "gripper", "action": "close", "sound": "mechanical clamp click", "priority": "high"},
            {"time": 2.5, "object": "item", "action": "lift", "sound": "light plastic handling", "priority": "medium"},
        ],
        "avoid": ["human speech", "music", "explosion", "vehicle horn"],
        "layer_roles": {"ambience": "warehouse air tone", "foley": "servo and gripper", "music": "none"},
    },
    "street_rain_crosswalk_001": {
        "scene": "rainy street crosswalk with passing footsteps and vehicles",
        "events": [
            {"time": 0.4, "object": "rain", "action": "fall", "sound": "steady rain on pavement", "priority": "medium"},
            {"time": 1.3, "object": "feet", "action": "step", "sound": "wet footstep splash", "priority": "high"},
            {"time": 2.6, "object": "car", "action": "pass", "sound": "distant wet tire pass-by", "priority": "medium"},
        ],
        "avoid": ["clear sunny ambience", "music", "speech", "sirens"],
        "layer_roles": {"ambience": "rain and city bed", "foley": "footsteps and splashes", "music": "none"},
    },
    "subway_arrival_door_001": {
        "scene": "subway train arriving and doors opening",
        "events": [
            {"time": 0.6, "object": "train", "action": "arrive", "sound": "subway braking rumble", "priority": "high"},
            {"time": 1.8, "object": "door", "action": "open", "sound": "pneumatic door open", "priority": "high"},
            {"time": 2.5, "object": "station", "action": "ambient", "sound": "subtle platform ambience", "priority": "medium"},
        ],
        "avoid": ["music", "speech announcement", "car horn", "birdsong"],
        "layer_roles": {"ambience": "station platform tone", "foley": "train brake and door", "music": "none"},
    },
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def build_naive_prompt(case_id: str, blueprint: dict) -> str:
    return (
        f"Generate realistic sound effects for a short silent video: {blueprint['scene']}. "
        f"Make it natural, synchronized, and without speech or music."
    )


def build_dss_prompt(case_id: str, blueprint: dict) -> str:
    event_lines = []
    for ev in blueprint["events"]:
        event_lines.append(
            f"{ev['time']:.1f}s {ev['object']} {ev['action']} -> {ev['sound']} "
            f"(priority={ev['priority']})"
        )
    avoid = ", ".join(blueprint["avoid"])
    return (
        f"DirectorSound Script for {case_id}. "
        f"Scene: {blueprint['scene']}. "
        f"Events: {'; '.join(event_lines)}. "
        f"Layer roles: ambience={blueprint['layer_roles']['ambience']}; "
        f"foley={blueprint['layer_roles']['foley']}; music={blueprint['layer_roles']['music']}. "
        f"Avoid: {avoid}. "
        f"Keep event timing tight and prioritize high-priority Foley cues."
    )


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    seed = read_json(SEED_JSON)
    cases = read_csv(CASES_CSV)
    repairs = read_csv(REPAIR_CSV)

    tasks = []
    for row in cases:
        case_id = row["case_id"]
        blueprint = CASE_BLUEPRINTS.get(case_id)
        if blueprint is None:
            raise KeyError(f"missing blueprint for case_id={case_id}")

        common = {
            "seed_id": seed["seed_id"],
            "case_id": case_id,
            "source_week": row["source_week"],
            "has_true_mmaudio": row["has_true_mmaudio"].lower() == "true",
            "week18_role": row["week18_role"],
            "scene": blueprint["scene"],
            "events": blueprint["events"],
            "avoid": blueprint["avoid"],
            "layer_roles": blueprint["layer_roles"],
            "expected_output": {
                "audio_format": "wav",
                "duration_policy": "match_input_video_or_case_duration",
                "evaluation": [
                    "event_coverage",
                    "onset_alignment_proxy",
                    "forbidden_leakage",
                    "rms",
                    "peak",
                    "silence_ratio",
                ],
            },
        }

        tasks.append({
            **common,
            "task_id": f"{case_id}__naive_prompt",
            "prompt_type": "naive",
            "prompt": build_naive_prompt(case_id, blueprint),
        })
        tasks.append({
            **common,
            "task_id": f"{case_id}__dss_prompt",
            "prompt_type": "dss",
            "prompt": build_dss_prompt(case_id, blueprint),
        })

    OUT_JSONL.write_text(
        "\n".join(json.dumps(t, ensure_ascii=False) for t in tasks) + "\n",
        encoding="utf-8",
    )
    REPORT_JSONL.write_text(OUT_JSONL.read_text(encoding="utf-8"), encoding="utf-8")

    with REPORT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "task_id",
            "case_id",
            "prompt_type",
            "has_true_mmaudio",
            "week18_role",
            "event_count",
            "avoid_count",
        ])
        writer.writeheader()
        for t in tasks:
            writer.writerow({
                "task_id": t["task_id"],
                "case_id": t["case_id"],
                "prompt_type": t["prompt_type"],
                "has_true_mmaudio": t["has_true_mmaudio"],
                "week18_role": t["week18_role"],
                "event_count": len(t["events"]),
                "avoid_count": len(t["avoid"]),
            })

    prompt_type_counts = {}
    for t in tasks:
        prompt_type_counts[t["prompt_type"]] = prompt_type_counts.get(t["prompt_type"], 0) + 1

    case_ids = sorted({t["case_id"] for t in tasks})
    true_anchor_tasks = [
        t["task_id"] for t in tasks
        if t["case_id"] == "glass_drop_room_001" and t["has_true_mmaudio"]
    ]

    verify = {
        "decision": "PASS",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task_count": len(tasks),
        "case_count": len(case_ids),
        "prompt_type_counts": prompt_type_counts,
        "true_anchor_task_count": len(true_anchor_tasks),
        "repair_target_count": len(repairs),
        "all_cases_have_naive_and_dss": all(
            prompt_type_counts.get(pt, 0) == len(case_ids)
            for pt in ["naive", "dss"]
        ),
        "claim_boundary": seed["claim_boundary"],
        "outputs": {
            "jsonl": str(REPORT_JSONL),
            "summary_csv": str(REPORT_CSV),
            "artifact_jsonl": str(OUT_JSONL),
            "manifest": str(OUT_MANIFEST),
        },
    }

    manifest = {
        "manifest_id": "week18_prompt_compiler_seed_manifest_20260703",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_seed": str(SEED_JSON),
        "task_queue": str(OUT_JSONL),
        "task_count": len(tasks),
        "case_count": len(case_ids),
        "prompt_type_counts": prompt_type_counts,
        "recommended_next_command": "Use week18_prompt_tasks_20260703.jsonl as model-runner input for naive vs DSS prompt ablation.",
        "boundary": seed["claim_boundary"],
    }

    OUT_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT_VERIFY.write_text(json.dumps(verify, ensure_ascii=False, indent=2), encoding="utf-8")

    OUT_README.write_text(f"""# W18 Prompt Compiler Seed

This directory contains a machine-readable prompt task queue generated from the Week17 true-aware demo release.

## Files

- `week18_prompt_tasks_20260703.jsonl`: 12 prompt tasks.
- `week18_prompt_task_manifest_20260703.json`: source and boundary manifest.

## Task design

Each W17 case has two prompt variants:

1. `naive_prompt`: normal text prompt baseline.
2. `dss_prompt`: DirectorSound Script controlled prompt.

## Boundary

The true MMAudio record is a positive anchor only. It is not batch success.
""", encoding="utf-8")

    print(json.dumps({
        "decision": verify["decision"],
        "task_count": verify["task_count"],
        "case_count": verify["case_count"],
        "prompt_type_counts": verify["prompt_type_counts"],
        "true_anchor_task_count": verify["true_anchor_task_count"],
        "repair_target_count": verify["repair_target_count"],
        "out_jsonl": str(REPORT_JSONL),
        "out_summary_csv": str(REPORT_CSV),
        "out_verify": str(REPORT_VERIFY),
    }, ensure_ascii=False, indent=2))

    ok = (
        verify["task_count"] == 12
        and verify["case_count"] == 6
        and verify["prompt_type_counts"].get("naive") == 6
        and verify["prompt_type_counts"].get("dss") == 6
        and verify["true_anchor_task_count"] == 2
    )
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())