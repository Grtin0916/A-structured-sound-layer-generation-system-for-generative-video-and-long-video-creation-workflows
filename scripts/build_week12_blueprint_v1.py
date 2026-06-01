#!/usr/bin/env python3
"""
Build Week12 SoundLayer Blueprint V1 artifacts from the existing seed manifest.

Scope:
- Generate a machine-readable Blueprint V1 JSON Schema.
- Convert seed cases into Blueprint V1 cases with scene/layer/event/timing fields.
- Export event timeline as JSONL and CSV.
- Generate a contact-sheet style timeline visualization.
- Produce a validation report.

Boundary:
- This script does not generate audio.
- It does not claim semantic/audio quality.
- It only hardens the intermediate representation contract.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


SCHEMA_VERSION = "soundlayer_blueprint_v1"
DEFAULT_DURATION_SECONDS = 8.0


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def stable_id(prefix: str, text: str, width: int = 10) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:width]
    return f"{prefix}_{digest}"


def safe_float(value: Any, default: float) -> float:
    try:
        if value is None:
            return default
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return default
        return out
    except Exception:
        return default


def normalize_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def deep_find_first(obj: Any, key_patterns: Iterable[str]) -> Optional[Any]:
    patterns = [re.compile(p, re.I) for p in key_patterns]

    def visit(x: Any) -> Optional[Any]:
        if isinstance(x, dict):
            for k, v in x.items():
                if any(p.search(str(k)) for p in patterns):
                    if isinstance(v, (str, int, float)) and str(v).strip():
                        return v
            for v in x.values():
                got = visit(v)
                if got is not None:
                    return got
        elif isinstance(x, list):
            for item in x:
                got = visit(item)
                if got is not None:
                    return got
        return None

    return visit(obj)


def deep_collect_strings(obj: Any, max_items: int = 12) -> List[str]:
    out: List[str] = []

    def visit(x: Any) -> None:
        if len(out) >= max_items:
            return
        if isinstance(x, str):
            s = x.strip()
            if s and len(s) <= 300 and not s.startswith("{"):
                out.append(s)
        elif isinstance(x, dict):
            for v in x.values():
                visit(v)
        elif isinstance(x, list):
            for item in x:
                visit(item)

    visit(obj)
    dedup: List[str] = []
    seen = set()
    for item in out:
        if item not in seen:
            seen.add(item)
            dedup.append(item)
    return dedup


def extract_seed_records(raw: Any) -> List[Dict[str, Any]]:
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]

    if not isinstance(raw, dict):
        return []

    preferred_keys = [
        "seeds",
        "seed_cases",
        "cases",
        "items",
        "records",
        "blueprints",
        "manifest",
        "data",
    ]
    for key in preferred_keys:
        value = raw.get(key)
        if isinstance(value, list) and all(isinstance(x, dict) for x in value):
            return list(value)

    candidates: List[List[Dict[str, Any]]] = []

    def visit(x: Any) -> None:
        if isinstance(x, list) and x and all(isinstance(i, dict) for i in x):
            candidates.append(list(x))
        elif isinstance(x, dict):
            for v in x.values():
                visit(v)

    visit(raw)
    if candidates:
        candidates.sort(key=len, reverse=True)
        return candidates[0]

    return []


def get_text(seed: Dict[str, Any], fallback: str) -> str:
    keys = [
        "scene_description",
        "description",
        "prompt",
        "text_prompt",
        "scene",
        "caption",
        "query",
        "title",
        "name",
    ]
    for key in keys:
        value = seed.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    collected = deep_collect_strings(seed, max_items=8)
    if collected:
        return " | ".join(collected[:3])

    return fallback


def get_seed_id(seed: Dict[str, Any], index: int, text: str) -> str:
    for key in ["seed_id", "case_id", "id", "name", "title"]:
        value = seed.get(key)
        if isinstance(value, (str, int)) and str(value).strip():
            return normalize_key(str(value))[:80]
    return f"seed_{index:04d}_{stable_id('case', text, width=6)}"


def infer_duration(seed: Dict[str, Any]) -> float:
    direct = deep_find_first(
        seed,
        [
            r"duration",
            r"clip_seconds",
            r"seconds",
            r"video_length",
        ],
    )
    duration = safe_float(direct, DEFAULT_DURATION_SECONDS)
    if duration <= 0.5:
        duration = DEFAULT_DURATION_SECONDS
    return round(min(max(duration, 3.0), 30.0), 3)


def infer_layer_requirements(text: str) -> List[Dict[str, Any]]:
    low = text.lower()
    layers: List[Dict[str, Any]] = [
        {
            "layer_id": "ambience",
            "layer_type": "ambience",
            "role": "continuous scene bed",
            "priority": "medium",
            "generation_hint": "derive ambient bed from scene context; keep unobtrusive",
        },
        {
            "layer_id": "foley",
            "layer_type": "foley",
            "role": "time-aligned foreground events",
            "priority": "high",
            "generation_hint": "align short sound events to visual actions",
        },
    ]

    if any(k in low for k in ["music", "bgm", "score", "trailer", "cinematic", "mood"]):
        layers.append(
            {
                "layer_id": "music",
                "layer_type": "music",
                "role": "optional emotional or rhythmic bed",
                "priority": "low",
                "generation_hint": "use only when it does not mask foley or speech-critical events",
            }
        )

    return layers


def infer_events(seed_id: str, text: str, duration: float, layers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    has_music = any(layer["layer_type"] == "music" for layer in layers)

    events: List[Dict[str, Any]] = [
        {
            "event_id": f"{seed_id}_evt_001",
            "label": "scene ambience bed",
            "layer_type": "ambience",
            "start_seconds": 0.0,
            "end_seconds": duration,
            "priority": "medium",
            "intensity": 0.35,
            "prompt": f"Ambient sound bed for: {text}",
            "alignment_rule": "span_full_clip",
        },
        {
            "event_id": f"{seed_id}_evt_002",
            "label": "primary foreground sound event",
            "layer_type": "foley",
            "start_seconds": round(duration * 0.18, 3),
            "end_seconds": round(duration * 0.72, 3),
            "priority": "high",
            "intensity": 0.75,
            "prompt": f"Foreground foley event synchronized with visible action: {text}",
            "alignment_rule": "align_to_main_visual_action",
        },
    ]

    if has_music:
        events.append(
            {
                "event_id": f"{seed_id}_evt_003",
                "label": "optional music bed",
                "layer_type": "music",
                "start_seconds": 0.0,
                "end_seconds": duration,
                "priority": "low",
                "intensity": 0.25,
                "prompt": f"Subtle music bed matching scene mood: {text}",
                "alignment_rule": "do_not_mask_foley",
            }
        )

    return events


def build_blueprint(seed: Dict[str, Any], index: int) -> Dict[str, Any]:
    text = get_text(seed, fallback=f"Week12 seed case {index}")
    seed_id = get_seed_id(seed, index, text)
    duration = infer_duration(seed)
    layers = infer_layer_requirements(text)
    events = infer_events(seed_id, text, duration, layers)

    source_artifact_uri = deep_find_first(seed, [r"artifact.*uri", r"artifact.*path", r"artifact"])
    eval_summary_uri = deep_find_first(seed, [r"eval.*summary.*uri", r"eval.*summary", r"eval.*path", r"score.*path"])

    blueprint_id = stable_id("blueprint_v1", f"{seed_id}|{text}|{duration}")

    return {
        "schema_version": SCHEMA_VERSION,
        "blueprint_id": blueprint_id,
        "source_seed_id": seed_id,
        "scene": {
            "description": text,
            "duration_seconds": duration,
            "source_type": "week12_seed_manifest",
        },
        "layers": layers,
        "events": events,
        "artifacts": {
            "source_artifact_uri": str(source_artifact_uri) if source_artifact_uri is not None else None,
            "eval_summary_uri": str(eval_summary_uri) if eval_summary_uri is not None else None,
            "blueprint_artifact_uri": f"artifacts/manifests/week12_blueprint_v1_manifest.json#{blueprint_id}",
            "timeline_jsonl_uri": "artifacts/manifests/week12_event_timeline.jsonl",
            "timeline_csv_uri": "artifacts/manifests/week12_event_timeline.csv",
        },
        "metadata": {
            "created_at": now_iso(),
            "builder": "scripts/build_week12_blueprint_v1.py",
            "boundary": "Blueprint V1 is a structured intermediate representation; it is not generated audio quality evidence.",
            "source_seed_raw_keys": sorted(seed.keys()),
        },
    }


def build_schema() -> Dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "schemas/soundlayer_blueprint_v1.schema.json",
        "title": "SoundLayer Blueprint V1",
        "type": "object",
        "required": [
            "schema_version",
            "blueprint_id",
            "source_seed_id",
            "scene",
            "layers",
            "events",
            "artifacts",
            "metadata",
        ],
        "properties": {
            "schema_version": {"const": SCHEMA_VERSION},
            "blueprint_id": {"type": "string", "minLength": 1},
            "source_seed_id": {"type": "string", "minLength": 1},
            "scene": {
                "type": "object",
                "required": ["description", "duration_seconds", "source_type"],
                "properties": {
                    "description": {"type": "string", "minLength": 1},
                    "duration_seconds": {"type": "number", "exclusiveMinimum": 0},
                    "source_type": {"type": "string"},
                },
                "additionalProperties": True,
            },
            "layers": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "required": ["layer_id", "layer_type", "role", "priority", "generation_hint"],
                    "properties": {
                        "layer_id": {"type": "string", "minLength": 1},
                        "layer_type": {"enum": ["ambience", "foley", "music", "speech", "effect"]},
                        "role": {"type": "string"},
                        "priority": {"enum": ["low", "medium", "high"]},
                        "generation_hint": {"type": "string"},
                    },
                    "additionalProperties": True,
                },
            },
            "events": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "required": [
                        "event_id",
                        "label",
                        "layer_type",
                        "start_seconds",
                        "end_seconds",
                        "priority",
                        "intensity",
                        "prompt",
                        "alignment_rule",
                    ],
                    "properties": {
                        "event_id": {"type": "string", "minLength": 1},
                        "label": {"type": "string", "minLength": 1},
                        "layer_type": {"enum": ["ambience", "foley", "music", "speech", "effect"]},
                        "start_seconds": {"type": "number", "minimum": 0},
                        "end_seconds": {"type": "number", "exclusiveMinimum": 0},
                        "priority": {"enum": ["low", "medium", "high"]},
                        "intensity": {"type": "number", "minimum": 0, "maximum": 1},
                        "prompt": {"type": "string", "minLength": 1},
                        "alignment_rule": {"type": "string", "minLength": 1},
                    },
                    "additionalProperties": True,
                },
            },
            "artifacts": {
                "type": "object",
                "required": [
                    "source_artifact_uri",
                    "eval_summary_uri",
                    "blueprint_artifact_uri",
                    "timeline_jsonl_uri",
                    "timeline_csv_uri",
                ],
                "additionalProperties": True,
            },
            "metadata": {"type": "object", "additionalProperties": True},
        },
        "additionalProperties": False,
    }


def validate_blueprint(bp: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    required = ["schema_version", "blueprint_id", "source_seed_id", "scene", "layers", "events", "artifacts", "metadata"]
    for key in required:
        if key not in bp:
            errors.append(f"missing required field: {key}")

    if bp.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version mismatch")

    scene = bp.get("scene", {})
    duration = safe_float(scene.get("duration_seconds"), -1.0) if isinstance(scene, dict) else -1.0
    if duration <= 0:
        errors.append("scene.duration_seconds must be positive")

    layer_types = set()
    for layer in bp.get("layers", []):
        if isinstance(layer, dict):
            layer_types.add(layer.get("layer_type"))

    if not layer_types:
        errors.append("no valid layers")

    for event in bp.get("events", []):
        if not isinstance(event, dict):
            errors.append("event is not object")
            continue
        start = safe_float(event.get("start_seconds"), -1.0)
        end = safe_float(event.get("end_seconds"), -1.0)
        if start < 0 or end <= start:
            errors.append(f"invalid event timing: {event.get('event_id')}")
        if duration > 0 and end > duration + 1e-6:
            errors.append(f"event exceeds duration: {event.get('event_id')}")
        if event.get("layer_type") not in layer_types:
            errors.append(f"event layer_type not declared in layers: {event.get('event_id')}")

    artifacts = bp.get("artifacts", {})
    if isinstance(artifacts, dict):
        if not artifacts.get("source_artifact_uri"):
            warnings.append("source_artifact_uri missing; downstream Java/Cloud binding may need seed-level fallback")
        if not artifacts.get("eval_summary_uri"):
            warnings.append("eval_summary_uri missing; downstream quality gate is weaker")
    else:
        errors.append("artifacts must be object")

    return errors, warnings


def flatten_events(blueprints: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for bp in blueprints:
        for event in bp.get("events", []):
            rows.append(
                {
                    "blueprint_id": bp["blueprint_id"],
                    "source_seed_id": bp["source_seed_id"],
                    "scene_description": bp["scene"]["description"],
                    "duration_seconds": bp["scene"]["duration_seconds"],
                    "event_id": event["event_id"],
                    "label": event["label"],
                    "layer_type": event["layer_type"],
                    "start_seconds": event["start_seconds"],
                    "end_seconds": event["end_seconds"],
                    "priority": event["priority"],
                    "intensity": event["intensity"],
                    "alignment_rule": event["alignment_rule"],
                    "prompt": event["prompt"],
                    "blueprint_artifact_uri": bp["artifacts"]["blueprint_artifact_uri"],
                }
            )
    return rows


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


def write_contact_sheet_svg(path: Path, rows: List[Dict[str, Any]], max_width: int = 1400) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["blueprint_id"], []).append(row)

    row_height = 72
    top = 70
    left = 260
    track_width = max_width - left - 80
    height = top + max(1, len(grouped)) * row_height + 40

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{max_width}" height="{height}" viewBox="0 0 {max_width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="24" y="32" font-family="Arial" font-size="22" font-weight="700">Week12 SoundLayer Blueprint V1 Event Timeline</text>',
        '<text x="24" y="56" font-family="Arial" font-size="13">Each bar is a machine-readable sound event; this is not audio quality evidence.</text>',
    ]

    y = top
    for bp_id, events in grouped.items():
        duration = max(safe_float(e.get("duration_seconds"), DEFAULT_DURATION_SECONDS) for e in events)
        parts.append(f'<text x="24" y="{y + 24}" font-family="Arial" font-size="13" font-weight="700">{html.escape(bp_id)}</text>')
        desc = str(events[0].get("scene_description", ""))[:70]
        parts.append(f'<text x="24" y="{y + 44}" font-family="Arial" font-size="11">{html.escape(desc)}</text>')
        parts.append(f'<line x1="{left}" y1="{y + 32}" x2="{left + track_width}" y2="{y + 32}" stroke="#222" stroke-width="1"/>')

        for event in events:
            start = safe_float(event.get("start_seconds"), 0.0)
            end = safe_float(event.get("end_seconds"), duration)
            x = left + (start / duration) * track_width
            w = max(4.0, ((end - start) / duration) * track_width)
            layer = html.escape(str(event.get("layer_type", "")))
            label = html.escape(str(event.get("label", ""))[:32])
            bar_y = y + 18 + {"ambience": 0, "foley": 14, "music": 28}.get(str(event.get("layer_type")), 42)
            parts.append(f'<rect x="{x:.2f}" y="{bar_y}" width="{w:.2f}" height="10" rx="2" fill="#444"/>')
            parts.append(f'<text x="{x + 4:.2f}" y="{bar_y - 2}" font-family="Arial" font-size="9">{layer}: {label}</text>')

        parts.append(f'<text x="{left}" y="{y + 62}" font-family="Arial" font-size="10">0s</text>')
        parts.append(f'<text x="{left + track_width - 36}" y="{y + 62}" font-family="Arial" font-size="10">{duration:g}s</text>')
        y += row_height

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_contact_sheet_png(path: Path, rows: List[Dict[str, Any]]) -> bool:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return False

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["blueprint_id"], []).append(row)

    fig_height = max(3.0, 1.1 * max(1, len(grouped)))
    fig, ax = plt.subplots(figsize=(14, fig_height))
    y_ticks: List[float] = []
    y_labels: List[str] = []

    y = 0
    for bp_id, events in grouped.items():
        duration = max(safe_float(e.get("duration_seconds"), DEFAULT_DURATION_SECONDS) for e in events)
        for event in events:
            start = safe_float(event.get("start_seconds"), 0.0)
            end = safe_float(event.get("end_seconds"), duration)
            lane = {"ambience": 0.0, "foley": 0.22, "music": 0.44}.get(str(event.get("layer_type")), 0.66)
            ax.broken_barh([(start, max(0.05, end - start))], (y + lane, 0.16))
        y_ticks.append(y + 0.3)
        y_labels.append(bp_id)
        y += 1

    ax.set_title("Week12 SoundLayer Blueprint V1 Event Timeline")
    ax.set_xlabel("seconds")
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.grid(True, axis="x", alpha=0.3)
    ax.set_ylim(-0.1, max(1, y))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-manifest", type=Path, default=Path("artifacts/manifests/week12_blueprint_seed_manifest.json"))
    parser.add_argument("--schema-out", type=Path, default=Path("schemas/soundlayer_blueprint_v1.schema.json"))
    parser.add_argument("--manifest-out", type=Path, default=Path("artifacts/manifests/week12_blueprint_v1_manifest.json"))
    parser.add_argument("--timeline-jsonl", type=Path, default=Path("artifacts/manifests/week12_event_timeline.jsonl"))
    parser.add_argument("--timeline-csv", type=Path, default=Path("artifacts/manifests/week12_event_timeline.csv"))
    parser.add_argument("--contact-sheet-svg", type=Path, default=Path("artifacts/visuals/week12_event_timeline_contact_sheet.svg"))
    parser.add_argument("--contact-sheet-png", type=Path, default=Path("artifacts/visuals/week12_event_timeline_contact_sheet.png"))
    parser.add_argument("--validation-out", type=Path, default=Path("artifacts/manifests/week12_blueprint_v1_validation_report.json"))
    parser.add_argument("--max-cases", type=int, default=5)
    args = parser.parse_args()

    raw = read_json(args.seed_manifest)
    seeds = extract_seed_records(raw)
    if not seeds:
        raise SystemExit(f"[FAIL] No seed records found in {args.seed_manifest}")

    selected = seeds[: args.max_cases]
    blueprints = [build_blueprint(seed, idx + 1) for idx, seed in enumerate(selected)]
    rows = flatten_events(blueprints)

    all_errors: List[str] = []
    all_warnings: List[str] = []
    per_blueprint: List[Dict[str, Any]] = []

    for bp in blueprints:
        errors, warnings = validate_blueprint(bp)
        all_errors.extend([f"{bp['blueprint_id']}: {x}" for x in errors])
        all_warnings.extend([f"{bp['blueprint_id']}: {x}" for x in warnings])
        per_blueprint.append(
            {
                "blueprint_id": bp["blueprint_id"],
                "source_seed_id": bp["source_seed_id"],
                "event_count": len(bp.get("events", [])),
                "layer_count": len(bp.get("layers", [])),
                "errors": errors,
                "warnings": warnings,
            }
        )

    schema = build_schema()
    manifest = {
        "schema_version": "week12_blueprint_v1_manifest",
        "created_at": now_iso(),
        "source_seed_manifest": str(args.seed_manifest),
        "summary": {
            "seed_count_in_source": len(seeds),
            "blueprint_count": len(blueprints),
            "event_count": len(rows),
            "status": "PASS" if not all_errors else "FAIL",
            "warning_count": len(all_warnings),
        },
        "artifacts": {
            "schema": str(args.schema_out),
            "manifest": str(args.manifest_out),
            "timeline_jsonl": str(args.timeline_jsonl),
            "timeline_csv": str(args.timeline_csv),
            "contact_sheet_svg": str(args.contact_sheet_svg),
            "contact_sheet_png": str(args.contact_sheet_png),
            "validation_report": str(args.validation_out),
        },
        "blueprints": blueprints,
        "boundary": {
            "not_audio_generation": True,
            "not_subjective_audio_quality": True,
            "purpose": "Convert seed cases into a structured scene/event/layer/timing contract for downstream Java/Cloud consumption.",
        },
    }

    write_json(args.schema_out, schema)
    write_json(args.manifest_out, manifest)
    write_jsonl(args.timeline_jsonl, rows)
    write_csv(args.timeline_csv, rows)
    write_contact_sheet_svg(args.contact_sheet_svg, rows)
    png_ok = write_contact_sheet_png(args.contact_sheet_png, rows)

    report = {
        "schema_version": "week12_blueprint_v1_validation_report",
        "created_at": now_iso(),
        "status": "PASS" if not all_errors else "FAIL",
        "blueprint_count": len(blueprints),
        "event_count": len(rows),
        "error_count": len(all_errors),
        "warning_count": len(all_warnings),
        "errors": all_errors,
        "warnings": all_warnings,
        "per_blueprint": per_blueprint,
        "contact_sheet_png_created": png_ok,
        "contact_sheet_svg_created": args.contact_sheet_svg.exists(),
    }
    write_json(args.validation_out, report)

    print("[PASS]" if not all_errors else "[FAIL]", "Week12 Blueprint V1 build")
    print(f"source_seed_manifest={args.seed_manifest}")
    print(f"blueprint_count={len(blueprints)}")
    print(f"event_count={len(rows)}")
    print(f"warning_count={len(all_warnings)}")
    print(f"schema={args.schema_out}")
    print(f"manifest={args.manifest_out}")
    print(f"timeline_jsonl={args.timeline_jsonl}")
    print(f"timeline_csv={args.timeline_csv}")
    print(f"contact_sheet_svg={args.contact_sheet_svg}")
    print(f"contact_sheet_png={args.contact_sheet_png if png_ok else 'NOT_CREATED_matplotlib_missing'}")
    print(f"validation_report={args.validation_out}")

    return 0 if not all_errors else 2


if __name__ == "__main__":
    raise SystemExit(main())