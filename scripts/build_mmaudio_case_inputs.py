#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT = Path(__file__).resolve().parents[1]
CASES_DIR = ROOT / "cases"
REPORTS_DIR = ROOT / "reports"
EXP_DIR = ROOT / "experiments" / "mmaudio_baseline_2026_06_30"
CANDIDATE_DIR = EXP_DIR / "candidates"
COMMANDS_SH = EXP_DIR / "candidate_commands.sh"
RUN_QUEUE_CSV = EXP_DIR / "candidate_run_queue.csv"

REQUIRED_CASE_FILES = [
    "director_sound_script.yaml",
    "expected_events.csv",
    "baseline_prompt.txt",
    "case_notes.md",
    "input_video.mp4",
]


def rel(path: Path) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_text(path: Path, limit: int = 6000) -> str:
    try:
        return path.read_text(encoding="utf-8")[:limit].strip()
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="ignore")[:limit].strip()


def run_cmd(cmd: List[str], timeout: int = 20) -> Dict[str, Any]:
    try:
        p = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
        return {
            "ok": p.returncode == 0,
            "returncode": p.returncode,
            "stdout": p.stdout.strip(),
            "stderr": p.stderr.strip(),
        }
    except Exception as e:
        return {"ok": False, "error": repr(e), "stdout": "", "stderr": ""}


def ffprobe_duration(video: Path) -> Optional[float]:
    if not shutil.which("ffprobe"):
        return None
    out = run_cmd(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            str(video),
        ],
        timeout=20,
    )
    if not out.get("ok"):
        return None
    try:
        return round(float(out["stdout"]), 3)
    except Exception:
        return None


def read_expected_events(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample) if sample.strip() else csv.excel
        reader = csv.DictReader(f, dialect=dialect)
        rows = []
        for row in reader:
            clean = {}
            for k, v in row.items():
                if k is None:
                    continue
                clean[k.strip()] = (v or "").strip()
            if any(clean.values()):
                rows.append(clean)
        return rows


def get_field(row: Dict[str, str], aliases: List[str]) -> str:
    lowered = {k.lower().strip(): v for k, v in row.items()}
    for a in aliases:
        if a in lowered and lowered[a]:
            return lowered[a]
    return ""


def compact_event(row: Dict[str, str], idx: int) -> str:
    t = get_field(row, ["time", "timestamp", "start", "start_time", "start_s", "onset", "onset_s"])
    obj = get_field(row, ["object", "source", "actor", "visual_object"])
    action = get_field(row, ["action", "event", "verb"])
    intent = get_field(row, ["sound_intent", "sound", "audio", "audio_intent", "description"])
    priority = get_field(row, ["priority", "importance"])
    tolerance = get_field(row, ["tolerance_ms", "sync_tolerance_ms", "tolerance"])

    parts = []
    if t:
        parts.append(f"t={t}")
    if obj:
        parts.append(f"object={obj}")
    if action:
        parts.append(f"action={action}")
    if intent:
        parts.append(f"sound={intent}")
    if priority:
        parts.append(f"priority={priority}")
    if tolerance:
        parts.append(f"tolerance={tolerance}")

    if parts:
        return f"E{idx}: " + ", ".join(parts)

    # Fallback for unknown CSV headers.
    short_items = [f"{k}={v}" for k, v in list(row.items())[:5] if v]
    return f"E{idx}: " + ", ".join(short_items)


def extract_avoid_lines(dss_text: str) -> List[str]:
    """Extract useful avoid constraints from YAML-ish DSS text.

    The first version accidentally captured structural lines such as
    `avoid: [` instead of actual avoid items. This parser is deliberately
    conservative: it keeps only human-readable leaf items and drops YAML
    keys/brackets.
    """
    avoid: List[str] = []
    in_avoid_block = False

    for raw in dss_text.splitlines():
        line = raw.strip()
        low = line.lower()

        if not line or line.startswith("#"):
            continue

        if any(k in low for k in ["avoid:", "forbidden:", "exclude:"]):
            in_avoid_block = True
            # Inline list: avoid: [speech, music]
            if "[" in line and "]" in line:
                inside = line.split("[", 1)[1].split("]", 1)[0]
                for item in inside.split(","):
                    item = item.strip().strip("'\\\"")
                    if item:
                        avoid.append(item)
            continue

        if in_avoid_block:
            # End block when another top-level-ish key begins.
            if not raw.startswith((" ", "\t", "-")) and ":" in line:
                in_avoid_block = False

            if line.startswith("-"):
                item = line[1:].strip().strip("'\\\"")
                if item and item not in ["[", "]"]:
                    avoid.append(item)
                continue

        # Fallback: catch explicit natural-language constraints only.
        if any(k in low for k in ["no ", "without ", "avoid "]):
            item = line.strip("- ").strip("'\\\"")
            if item and ":" not in item[:12] and "[" not in item and "]" not in item:
                avoid.append(item)

    clean: List[str] = []
    for item in avoid:
        item = " ".join(item.replace('"', "").replace("'", "").split())
        if not item:
            continue
        if item.lower() in {"avoid", "forbidden", "exclude"}:
            continue
        if item not in clean:
            clean.append(item)

    return clean[:6]


def build_prompt(case_id: str, baseline: str, events: List[Dict[str, str]], dss_text: str, variant: str) -> str:
    event_lines = [compact_event(e, i + 1) for i, e in enumerate(events[:8])]
    avoid_lines = extract_avoid_lines(dss_text)

    baseline_one_line = " ".join(baseline.split())
    if len(baseline_one_line) > 700:
        baseline_one_line = baseline_one_line[:700].rstrip() + "..."

    if variant == "dss_compact":
        prompt = (
            f"Generate synchronized Foley and ambience for silent video case {case_id}. "
            f"Base scene: {baseline_one_line}. "
            f"Key timed events: {' | '.join(event_lines)}. "
            f"Prioritize accurate onset timing, plausible object sounds, natural room ambience, and no narration."
        )
    else:
        avoid_text = "; ".join(avoid_lines) if avoid_lines else "speech, unrelated music, exaggerated cinematic hits"
        prompt = (
            f"Director sound script for case {case_id}. "
            f"Create audio that follows the visual event order and priority. "
            f"Timed event contract: {' | '.join(event_lines)}. "
            f"Avoid: {avoid_text}. "
            f"Keep Foley transients aligned to visible actions, preserve background ambience, and avoid speech leakage."
        )

    return " ".join(prompt.split())


def torch_preflight() -> Dict[str, Any]:
    try:
        import torch  # type: ignore

        data: Dict[str, Any] = {
            "torch_import": True,
            "torch_version": getattr(torch, "__version__", "unknown"),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        }
        if torch.cuda.is_available():
            data["cuda_device_name_0"] = torch.cuda.get_device_name(0)
        return data
    except Exception as e:
        return {"torch_import": False, "error": repr(e)}


def build_preflight() -> Dict[str, Any]:
    mmaudio_repo = os.environ.get("MMAUDIO_REPO", "").strip()
    mmaudio_demo = Path(mmaudio_repo) / "demo.py" if mmaudio_repo else None
    nvidia = run_cmd(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], timeout=10) if shutil.which("nvidia-smi") else {"ok": False, "error": "nvidia-smi not found"}

    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "ffmpeg": shutil.which("ffmpeg"),
        "ffprobe": shutil.which("ffprobe"),
        "nvidia_smi": nvidia,
        "torch": torch_preflight(),
        "MMAUDIO_REPO": mmaudio_repo or None,
        "mmaudio_demo_py_exists": bool(mmaudio_demo and mmaudio_demo.exists()),
        "execution_lane": (
            "local_mmaudio"
            if mmaudio_demo and mmaudio_demo.exists()
            else "remote_or_fallback_required"
        ),
    }


def main() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)

    failures: List[Dict[str, Any]] = []
    cases: List[Dict[str, Any]] = []
    prompts: List[Dict[str, Any]] = []
    run_rows: List[Dict[str, Any]] = []

    case_dirs = sorted([p for p in CASES_DIR.iterdir() if p.is_dir()]) if CASES_DIR.exists() else []

    for case_dir in case_dirs:
        case_id = case_dir.name
        missing = [name for name in REQUIRED_CASE_FILES if not (case_dir / name).exists()]
        if missing:
            failures.append({"case_id": case_id, "reason": "missing_required_case_files", "missing": missing})
            continue

        video = case_dir / "input_video.mp4"
        dss = case_dir / "director_sound_script.yaml"
        expected = case_dir / "expected_events.csv"
        baseline = case_dir / "baseline_prompt.txt"

        duration = ffprobe_duration(video)
        duration_for_model = duration if duration and duration > 0 else 8.0
        if duration is None:
            failures.append({"case_id": case_id, "reason": "ffprobe_duration_unavailable_use_8s_default"})

        video_size_mb = round(video.stat().st_size / 1024 / 1024, 3)
        if video_size_mb > 90:
            failures.append({"case_id": case_id, "reason": "input_video_over_90mb_git_risk", "size_mb": video_size_mb})

        events = read_expected_events(expected)
        if len(events) < 1:
            failures.append({"case_id": case_id, "reason": "expected_events_empty"})

        dss_text = read_text(dss)
        baseline_text = read_text(baseline)

        case_entry = {
            "case_id": case_id,
            "case_dir": rel(case_dir),
            "input_video": rel(video),
            "input_video_size_mb": video_size_mb,
            "duration_sec": duration,
            "duration_for_model_sec": duration_for_model,
            "dss_path": rel(dss),
            "expected_events_path": rel(expected),
            "baseline_prompt_path": rel(baseline),
            "event_count": len(events),
            "candidate_slots": [],
        }

        for variant in ["dss_compact", "dss_avoid_priority"]:
            candidate_id = f"{case_id}__mmaudio__{variant}"
            prompt = build_prompt(case_id, baseline_text, events, dss_text, variant)
            expected_wav = CANDIDATE_DIR / f"{candidate_id}.wav"

            cmd = (
                'cd "$MMAUDIO_REPO" && '
                f"python demo.py --duration={duration_for_model:.2f} "
                f"--video {shlex.quote(str(video))} "
                f"--prompt {shlex.quote(prompt)}"
            )

            row = {
                "candidate_id": candidate_id,
                "case_id": case_id,
                "model": "MMAudio",
                "prompt_variant": variant,
                "video_conditioned": "true",
                "input_video": str(video),
                "duration_sec": f"{duration_for_model:.2f}",
                "expected_output_wav": str(expected_wav),
                "status": "queued",
                "command": cmd,
            }
            run_rows.append(row)

            prompt_entry = {
                "candidate_id": candidate_id,
                "case_id": case_id,
                "model": "MMAudio",
                "prompt_variant": variant,
                "prompt": prompt,
                "event_count": len(events),
                "video_conditioned": True,
                "expected_output_wav": rel(expected_wav),
            }
            prompts.append(prompt_entry)
            case_entry["candidate_slots"].append(prompt_entry)

        cases.append(case_entry)

    preflight = build_preflight()

    manifest = {
        "name": "week17_mmaudio_case_input_manifest",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Translate Week17 DSS demo cases into executable MMAudio candidate slots.",
        "case_count": len(cases),
        "candidate_slot_count": len(run_rows),
        "cases": cases,
        "runtime_preflight": preflight,
        "failure_count": len(failures),
        "readiness_failures": failures,
        "boundary": {
            "mmaudio_local_ready": bool(preflight.get("mmaudio_demo_py_exists")),
            "fallback_required_if_local_not_ready": not bool(preflight.get("mmaudio_demo_py_exists")),
            "do_not_claim_v2a_success_until_audio_files_exist": True,
        },
    }

    prompt_manifest = {
        "name": "week17_mmaudio_prompt_manifest",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "prompt_count": len(prompts),
        "prompts": prompts,
    }

    (REPORTS_DIR / "mmaudio_case_input_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (REPORTS_DIR / "mmaudio_prompt_manifest.json").write_text(json.dumps(prompt_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (REPORTS_DIR / "mmaudio_runtime_preflight.json").write_text(json.dumps(preflight, indent=2, ensure_ascii=False), encoding="utf-8")
    (REPORTS_DIR / "mmaudio_case_readiness_failures.json").write_text(json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8")

    with RUN_QUEUE_CSV.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "candidate_id",
            "case_id",
            "model",
            "prompt_variant",
            "video_conditioned",
            "input_video",
            "duration_sec",
            "expected_output_wav",
            "status",
            "command",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(run_rows)

    lines = [
        "#!/usr/bin/env bash",
        "set -u",
        ': "${MMAUDIO_REPO:?Set MMAUDIO_REPO=/path/to/MMAudio before running local MMAudio candidates}"',
        'echo "MMAUDIO_REPO=$MMAUDIO_REPO"',
        'test -f "$MMAUDIO_REPO/demo.py" || { echo "demo.py not found under $MMAUDIO_REPO"; exit 2; }',
        "",
    ]
    for row in run_rows:
        lines.append(f'echo "### RUN {row["candidate_id"]}"')
        lines.append(row["command"])
        lines.append('echo "### CHECK_OUTPUT_DIR $MMAUDIO_REPO/output"')
        lines.append("")
    COMMANDS_SH.write_text("\n".join(lines), encoding="utf-8")
    COMMANDS_SH.chmod(0o755)

    summary = {
        "case_count": len(cases),
        "candidate_slot_count": len(run_rows),
        "failure_count": len(failures),
        "execution_lane": preflight.get("execution_lane"),
        "mmaudio_demo_py_exists": preflight.get("mmaudio_demo_py_exists"),
        "outputs": {
            "case_input_manifest": rel(REPORTS_DIR / "mmaudio_case_input_manifest.json"),
            "prompt_manifest": rel(REPORTS_DIR / "mmaudio_prompt_manifest.json"),
            "runtime_preflight": rel(REPORTS_DIR / "mmaudio_runtime_preflight.json"),
            "readiness_failures": rel(REPORTS_DIR / "mmaudio_case_readiness_failures.json"),
            "run_queue_csv": rel(RUN_QUEUE_CSV),
            "candidate_commands": rel(COMMANDS_SH),
        },
    }

    status = "PASS_MMAUDIO_INPUTS_READY" if len(cases) >= 6 and len(run_rows) >= 12 else "PARTIAL_MMAUDIO_INPUTS_READY"
    print(status)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if len(cases) >= 6 and len(run_rows) >= 12 else 1


if __name__ == "__main__":
    raise SystemExit(main())