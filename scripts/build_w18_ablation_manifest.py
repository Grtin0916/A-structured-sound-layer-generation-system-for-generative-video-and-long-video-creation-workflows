#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as exc:
                raise RuntimeError(f"failed_to_parse_jsonl:{path}:{line_no}:{exc}") from exc
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def case_video_path(case_id: str, cases_root: Path) -> str:
    case_dir = cases_root / case_id
    for name in ["input_video.mp4", "video.mp4", "source.mp4"]:
        p = case_dir / name
        if p.exists():
            return str(p)
    return str(case_dir / "input_video.mp4")


def choose_source_route(variant: str) -> Dict[str, Any]:
    # Today this is a generation plan, not a success claim.
    # MMAudio is the preferred true V2A route when the local model path is available.
    # T2A/control are explicit fallbacks and must not be claimed as video-conditioned.
    if variant in {"naive", "naive_rich"}:
        return {
            "primary_source": "MMAudio",
            "fallback_source": "StableAudioOpen_or_T2A",
            "route": "video_to_audio_primary_with_text_fallback",
            "video_conditioned_primary": True,
            "fallback_allowed": True,
            "claim_boundary": "MMAudio output can be claimed as V2A only if WAV is actually generated from video input; T2A fallback is not video-synchronized evidence.",
        }

    return {
        "primary_source": "MMAudio",
        "fallback_source": "StableAudioOpen_or_T2A",
        "route": "dss_prompt_to_video_to_audio_primary_with_text_fallback",
        "video_conditioned_primary": True,
        "fallback_allowed": True,
        "claim_boundary": "DSS prompt controls generation input; only true video-conditioned model output can be claimed as V2A.",
    }


def prompt_quality_flags(row: Dict[str, Any]) -> List[str]:
    flags = []
    prompt = str(row.get("prompt", ""))
    variant = str(row.get("variant", ""))

    prompt_chars = int(row.get("prompt_chars", len(prompt)))
    event_count = int(row.get("event_count", 0))
    avoid_count = int(row.get("avoid_count", 0))

    if prompt_chars < 80:
        flags.append("prompt_too_short")
    if prompt_chars > 1200:
        flags.append("prompt_too_long")
    if event_count <= 0:
        flags.append("missing_events")
    if avoid_count <= 0:
        flags.append("missing_avoid_constraints")
    if variant.startswith("dss_") and "Event timeline:" not in prompt and variant == "dss_event_timeline":
        flags.append("timeline_variant_missing_timeline_text")
    if variant == "dss_layer_avoid" and "Forbidden content:" not in prompt:
        flags.append("layer_avoid_missing_forbidden_text")

    return flags


def build_manifest(rows: List[Dict[str, Any]], cases_root: Path, output_root: Path) -> List[Dict[str, Any]]:
    manifest = []

    for idx, row in enumerate(rows, start=1):
        case_id = str(row["case_id"])
        variant = str(row["variant"])
        source = choose_source_route(variant)

        job_id = f"w18_{idx:03d}_{case_id}_{variant}"
        output_dir = output_root / case_id / variant

        flags = prompt_quality_flags(row)

        item = {
            "job_id": job_id,
            "case_id": case_id,
            "variant": variant,
            "video_path": case_video_path(case_id, cases_root),
            "prompt": row.get("prompt", ""),
            "prompt_chars": int(row.get("prompt_chars", 0)),
            "event_count": int(row.get("event_count", 0)),
            "avoid_count": int(row.get("avoid_count", 0)),
            "duration_sec": float(row.get("duration_sec", 10.0)),
            "primary_source": source["primary_source"],
            "fallback_source": source["fallback_source"],
            "route": source["route"],
            "video_conditioned_primary": source["video_conditioned_primary"],
            "fallback_allowed": source["fallback_allowed"],
            "claim_boundary": source["claim_boundary"],
            "expected_output_wav": str(output_dir / f"{job_id}.wav"),
            "expected_metrics_json": str(output_dir / f"{job_id}.metrics.json"),
            "expected_failure_json": str(output_dir / f"{job_id}.failure.json"),
            "ready_for_generation": bool(row.get("ready_for_generation", False)) and len(flags) == 0,
            "qa_flags": "|".join(flags),
        }
        manifest.append(item)

    return manifest


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiled-prompts", required=True)
    parser.add_argument("--cases-root", default="cases")
    parser.add_argument("--output-root", default="artifacts/model_runs/w18_dss_ablation")
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--qa-json", required=True)
    parser.add_argument("--source-plan-json", required=True)
    args = parser.parse_args()

    compiled_path = Path(args.compiled_prompts)
    cases_root = Path(args.cases_root)
    output_root = Path(args.output_root)

    rows = read_jsonl(compiled_path)
    manifest = build_manifest(rows, cases_root, output_root)

    case_ids = sorted({row["case_id"] for row in manifest})
    variants = sorted({row["variant"] for row in manifest})
    bad_rows = [row for row in manifest if not row["ready_for_generation"]]
    missing_video = [
        row for row in manifest
        if not Path(row["video_path"]).exists()
    ]

    qa = {
        "compiled_prompts": str(compiled_path),
        "manifest_rows": len(manifest),
        "case_count": len(case_ids),
        "variant_count": len(variants),
        "case_ids": case_ids,
        "variants": variants,
        "ready_rows": len(manifest) - len(bad_rows),
        "bad_rows": len(bad_rows),
        "missing_video_count": len(missing_video),
        "paired_naive_dss": all(
            {"naive", "dss_global", "dss_event_timeline", "dss_layer_avoid"}.issubset(
                {row["variant"] for row in manifest if row["case_id"] == case_id}
            )
            for case_id in case_ids
        ),
        "ready_for_model_day": (
            len(manifest) >= 24
            and len(case_ids) >= 6
            and len(variants) >= 4
            and len(bad_rows) == 0
        ),
        "warning": (
            "missing_video_count does not block text fallback planning, but blocks true V2A claim for those cases."
            if missing_video else ""
        ),
        "bad_row_examples": bad_rows[:5],
        "missing_video_examples": missing_video[:5],
    }

    source_plan = {
        "experiment": "w18_dss_prompt_ablation",
        "model_day_target": {
            "minimum_generation_jobs": 24,
            "minimum_cases": 6,
            "required_pairing": "naive vs DSS variants per case",
        },
        "sources": [
            {
                "name": "MMAudio",
                "role": "primary V2A route",
                "video_conditioned": True,
                "claim_rule": "claim success only for real generated WAVs from video input",
            },
            {
                "name": "StableAudioOpen_or_T2A",
                "role": "fallback candidate baseline",
                "video_conditioned": False,
                "claim_rule": "do not claim video synchronization; use as text-controlled candidate baseline",
            },
            {
                "name": "manual_or_rule_control",
                "role": "control and repair-bank seed",
                "video_conditioned": False,
                "claim_rule": "use for ranking sanity check, not model capability claim",
            },
        ],
        "outputs": {
            "manifest_csv": args.out_csv,
            "manifest_json": args.out_json,
            "qa_json": args.qa_json,
            "output_root": args.output_root,
        },
    }

    write_csv(Path(args.out_csv), manifest)
    write_json(Path(args.out_json), manifest)
    write_json(Path(args.qa_json), qa)
    write_json(Path(args.source_plan_json), source_plan)

    print(json.dumps(qa, ensure_ascii=False, indent=2))
    return 0 if qa["ready_for_model_day"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
