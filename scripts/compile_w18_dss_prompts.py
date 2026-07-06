#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from soundlayer.dss.prompt_compiler import compile_prompt
from soundlayer.dss.schema import DSSCase


def read_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []

    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return []

    records: List[Dict[str, Any]] = []

    if path.suffix.lower() == ".jsonl":
        for line_no, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as exc:
                raise RuntimeError(f"failed_to_parse_jsonl:{path}:{line_no}:{exc}") from exc
            if isinstance(obj, dict):
                records.append(obj)
        return records

    try:
        obj = json.loads(text)
    except Exception as exc:
        raise RuntimeError(f"failed_to_parse_json:{path}:{exc}") from exc

    if isinstance(obj, list):
        return [item for item in obj if isinstance(item, dict)]

    if isinstance(obj, dict):
        for key in ["cases", "case_records", "prompt_tasks", "tasks", "items", "data", "records"]:
            value = obj.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        return [obj]

    return []


def read_csv_events(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for idx, row in enumerate(reader):
            if not row:
                continue
            item = {str(k).strip(): v for k, v in row.items() if k is not None}
            if "time_sec" not in item:
                for alt in ["time", "start_sec", "timestamp_sec", "onset_sec"]:
                    if alt in item:
                        item["time_sec"] = item[alt]
                        break
            if "sound_intent" not in item:
                for alt in ["sound", "expected_sound", "sound_description", "description"]:
                    if alt in item:
                        item["sound_intent"] = item[alt]
                        break
            if "action" not in item and "event" in item:
                item["action"] = item["event"]
            if "object" not in item:
                item["object"] = item.get("actor", item.get("target", f"event_{idx}"))
            rows.append(item)
        return rows


def try_read_yaml(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None

    try:
        import yaml  # type: ignore
    except Exception:
        return None

    try:
        obj = yaml.safe_load(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None

    return obj if isinstance(obj, dict) else None


def first_text_line(path: Path, max_chars: int = 800) -> Optional[str]:
    if not path.exists():
        return None

    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return None

    for line in text.splitlines():
        line = line.strip()
        if line:
            return line[:max_chars]
    return text[:max_chars]


def case_id_from_raw(raw: Dict[str, Any], fallback: str) -> str:
    for key in ["case_id", "id", "name", "case"]:
        value = raw.get(key)
        if value not in (None, "", []):
            return str(value)
    nested = raw.get("case")
    if isinstance(nested, dict):
        for key in ["case_id", "id", "name"]:
            value = nested.get(key)
            if value not in (None, "", []):
                return str(value)
    return fallback


def merge_record(base: Dict[str, Any], raw: Dict[str, Any]) -> Dict[str, Any]:
    nested = raw.get("case")
    if isinstance(nested, dict):
        for key, value in nested.items():
            if value not in (None, "", []):
                base.setdefault(key, value)

    for key, value in raw.items():
        if key in ["prompt", "compiled_prompt"]:
            continue
        if value not in (None, "", []):
            base.setdefault(key, value)

    for event_key in ["events", "expected_events", "event_timeline", "timeline"]:
        value = raw.get(event_key)
        if isinstance(value, list) and value and "events" not in base:
            base["events"] = value

    return base


def normalize_records(seed_records: List[Dict[str, Any]], task_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_case: Dict[str, Dict[str, Any]] = {}

    combined = seed_records + task_records
    for idx, raw in enumerate(combined, start=1):
        case_id = case_id_from_raw(raw, f"case_{idx:03d}")
        base = by_case.setdefault(case_id, {"case_id": case_id})
        merge_record(base, raw)

    return list(by_case.values())


def records_from_cases_dir(cases_root: Path) -> List[Dict[str, Any]]:
    if not cases_root.exists():
        return []

    records: List[Dict[str, Any]] = []

    for case_dir in sorted([p for p in cases_root.iterdir() if p.is_dir()]):
        record: Dict[str, Any] = {
            "case_id": case_dir.name,
            "duration_sec": 10.0,
            "scene": f"short video case {case_dir.name}",
        }

        dss_yaml = try_read_yaml(case_dir / "director_sound_script.yaml")
        if isinstance(dss_yaml, dict):
            merge_record(record, dss_yaml)
            record["case_id"] = str(record.get("case_id", case_dir.name))

        baseline = first_text_line(case_dir / "baseline_prompt.txt")
        notes = first_text_line(case_dir / "case_notes.md")
        if baseline and record.get("scene", "").startswith("short video case"):
            record["scene"] = baseline
        elif notes and record.get("scene", "").startswith("short video case"):
            record["scene"] = notes

        events = read_csv_events(case_dir / "expected_events.csv")
        if events and "events" not in record:
            record["events"] = events

        records.append(record)

    return records


def enrich_from_cases_dir(records: List[Dict[str, Any]], cases_root: Path) -> List[Dict[str, Any]]:
    by_case = {str(item.get("case_id")): item for item in records if item.get("case_id")}

    for case_record in records_from_cases_dir(cases_root):
        case_id = str(case_record.get("case_id"))
        if case_id in by_case:
            merge_record(by_case[case_id], case_record)
            # Prefer concrete case files for events because prompt-task JSONL often has prompt-only rows.
            if case_record.get("events"):
                by_case[case_id]["events"] = case_record["events"]
            if case_record.get("scene") and not by_case[case_id].get("scene"):
                by_case[case_id]["scene"] = case_record["scene"]
        else:
            by_case[case_id] = case_record

    return list(by_case.values())


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_matrix_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "case_id",
        "variant",
        "duration_sec",
        "event_count",
        "avoid_count",
        "prompt_chars",
        "ready_for_generation",
        "validation_errors",
        "prompt",
    ]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {key: row.get(key) for key in fieldnames}
            if isinstance(out.get("validation_errors"), list):
                out["validation_errors"] = "|".join(out["validation_errors"])
            writer.writerow(out)


def build_run_plan(out_jsonl: Path, out_csv: Path, variants: List[str], case_ids: List[str]) -> Dict[str, Any]:
    return {
        "experiment": "w18_dss_prompt_ablation",
        "compiled_prompts_jsonl": str(out_jsonl),
        "prompt_matrix_csv": str(out_csv),
        "paired_naive_dss": True,
        "required_variants": variants,
        "case_count": len(case_ids),
        "case_ids": case_ids,
        "source_plan": [
            {
                "source": "MMAudio",
                "route": "video_to_audio",
                "video_conditioned": True,
                "fallback_allowed": False,
                "claim_boundary": "Only claim true V2A success for WAVs actually generated by MMAudio from video input.",
            },
            {
                "source": "Stable Audio Open or other T2A fallback",
                "route": "text_to_audio",
                "video_conditioned": False,
                "fallback_allowed": True,
                "claim_boundary": "Text-to-audio candidate baseline only; do not claim video-synchronized V2A.",
            },
            {
                "source": "manual_or_rule_control",
                "route": "control",
                "video_conditioned": False,
                "fallback_allowed": True,
                "claim_boundary": "Control candidate for ranking, comparison, and repair-bank seeding.",
            },
        ],
        "minimum_next_day_generation_target": {
            "paired_cases": min(6, len(case_ids)),
            "minimum_candidates": max(24, min(6, len(case_ids)) * len(variants)),
            "must_pair_naive_and_dss": True,
            "do_not_claim": [
                "batch true MMAudio success before real WAV generation",
                "production SLO without production deployment",
                "live Grafana import without actual import",
                "k6 threshold pass without actual k6 threshold run",
            ],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--week18-seed", required=True)
    parser.add_argument("--prompt-tasks", required=True)
    parser.add_argument("--cases-root", default="cases")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--failures-json", required=True)
    parser.add_argument("--run-plan-json", required=True)
    parser.add_argument("--variants", default="naive,dss_global,dss_event_timeline,dss_layer_avoid")
    args = parser.parse_args()

    seed_path = Path(args.week18_seed)
    task_path = Path(args.prompt_tasks)
    cases_root = Path(args.cases_root)
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]

    seed_records = read_json_or_jsonl(seed_path)
    task_records = read_json_or_jsonl(task_path)

    records = normalize_records(seed_records, task_records)
    records = enrich_from_cases_dir(records, cases_root)

    compiled: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for raw in records:
        case = DSSCase.from_dict(raw, source_file=str(seed_path))
        errors = case.validate()

        if errors:
            failures.append(
                {
                    "case_id": case.case_id,
                    "errors": errors,
                    "event_count": len(case.events),
                    "duration_sec": case.duration_sec,
                    "source_file": case.source_file,
                }
            )

        for variant in variants:
            try:
                item = compile_prompt(case, variant)
                item["source_file"] = case.source_file
                compiled.append(item)
            except Exception as exc:
                failures.append(
                    {
                        "case_id": case.case_id,
                        "variant": variant,
                        "errors": [f"compile_error:{type(exc).__name__}:{exc}"],
                    }
                )

    case_ids = sorted({str(item["case_id"]) for item in compiled})
    ready_rows = [item for item in compiled if item.get("ready_for_generation") is True]

    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)
    out_jsonl = Path(args.out_jsonl)
    failures_json = Path(args.failures_json)
    run_plan_json = Path(args.run_plan_json)

    write_matrix_csv(out_csv, compiled)
    write_jsonl(out_jsonl, compiled)

    summary = {
        "input_seed": str(seed_path),
        "input_prompt_tasks": str(task_path),
        "cases_root": str(cases_root),
        "seed_record_count": len(seed_records),
        "prompt_task_record_count": len(task_records),
        "case_count": len(case_ids),
        "case_ids": case_ids,
        "variants": variants,
        "variant_count": len(variants),
        "prompt_task_count": len(compiled),
        "ready_prompt_task_count": len(ready_rows),
        "failure_count": len(failures),
        "ready_for_ablation": (
            len(case_ids) >= 6
            and len(variants) >= 4
            and len(compiled) >= 24
            and len(failures) == 0
        ),
        "outputs": {
            "summary_json": str(out_json),
            "matrix_csv": str(out_csv),
            "compiled_prompts_jsonl": str(out_jsonl),
            "failures_json": str(failures_json),
            "run_plan_json": str(run_plan_json),
        },
        "claim_boundary": [
            "This step compiles prompt ablation inputs only.",
            "It does not claim model generation success.",
            "T2A fallbacks must be marked video_conditioned=false.",
        ],
    }

    run_plan = build_run_plan(out_jsonl, out_csv, variants, case_ids)

    write_json(out_json, summary)
    write_json(failures_json, failures)
    write_json(run_plan_json, run_plan)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
