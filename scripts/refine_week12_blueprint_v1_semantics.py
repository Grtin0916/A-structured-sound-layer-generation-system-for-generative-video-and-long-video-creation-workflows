#!/usr/bin/env python3
"""
Refine Week12 SoundLayer Blueprint V1 semantics.

Purpose:
- Keep the already-passing Blueprint V1 structure.
- Replace ID-like scene descriptions with generation-useful semantic descriptions.
- Rewrite ambience/foley prompts into audio-candidate-ready controls.
- Regenerate timeline JSONL/CSV, validation report, semantic quality report, and contact sheet.

Boundary:
- No audio generation.
- No subjective quality claim.
- This is a semantic contract refinement for downstream Java/Cloud/audio-candidate consumption.
"""

from __future__ import annotations

import csv
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(".")
MANIFEST = ROOT / "artifacts/manifests/week12_blueprint_v1_manifest.json"
BACKUP = ROOT / "artifacts/manifests/week12_blueprint_v1_manifest_before_semantic_refine.json"
TIMELINE_JSONL = ROOT / "artifacts/manifests/week12_event_timeline.jsonl"
TIMELINE_CSV = ROOT / "artifacts/manifests/week12_event_timeline.csv"
VALIDATION = ROOT / "artifacts/manifests/week12_blueprint_v1_validation_report.json"
SEMANTIC_REPORT = ROOT / "artifacts/manifests/week12_blueprint_v1_semantic_report.json"
CONTACT_SHEET_SVG = ROOT / "artifacts/visuals/week12_event_timeline_contact_sheet.svg"
CONTACT_SHEET_PNG = ROOT / "artifacts/visuals/week12_event_timeline_contact_sheet.png"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def clean_phrase(raw: str) -> str:
    parts = [p.strip() for p in raw.split("|") if p.strip()]
    candidate = parts[0] if parts else raw

    candidate = candidate.lower()
    candidate = re.sub(r"soundlayer_blueprint_seed_v\d+", " ", candidate)
    candidate = re.sub(r"blueprint_seed_v\d+", " ", candidate)
    candidate = re.sub(r"^bp_", " ", candidate)
    candidate = re.sub(r"week\d+", " ", candidate)
    candidate = re.sub(r"seed_\d+", " ", candidate)
    candidate = re.sub(r"case_[a-f0-9]+", " ", candidate)
    candidate = re.sub(r"\b\d+\b", " ", candidate)
    candidate = re.sub(r"[_\-]+", " ", candidate)
    candidate = re.sub(r"\s+", " ", candidate).strip()

    stop = {"bp", "v", "case", "seed", "soundlayer", "blueprint"}
    tokens = [t for t in candidate.split() if t not in stop]
    phrase = " ".join(tokens).strip()
    return phrase or "generic scene"


def profile_from_phrase(phrase: str) -> Dict[str, Any]:
    p = phrase.lower()

    profiles = [
        {
            "keys": ["city", "walk", "street", "pedestrian"],
            "scene": "Urban walking scene with visible pedestrian motion, continuous street ambience, and time-aligned footsteps.",
            "tags": ["urban", "walking", "footsteps", "street_ambience"],
            "ambience": "Low-level city street ambience with distant traffic and outdoor environmental noise.",
            "foley": "Footsteps synchronized to the walking action, with subtle cloth movement and ground contact.",
            "event_label": "time-aligned footsteps",
        },
        {
            "keys": ["door", "close", "closing"],
            "scene": "Indoor door-closing scene with a short foreground door impact and quiet room tone.",
            "tags": ["indoor", "door_close", "impact", "room_tone"],
            "ambience": "Quiet indoor room tone before and after the door action.",
            "foley": "Door closing thud with latch detail aligned to the visible closing moment.",
            "event_label": "door close impact",
        },
        {
            "keys": ["rain", "window"],
            "scene": "Rain-at-window scene with continuous rainfall ambience and localized water impact texture.",
            "tags": ["rain", "window", "weather", "water_impact"],
            "ambience": "Continuous rainfall bed with soft outdoor weather texture.",
            "foley": "Raindrop impacts against glass aligned with visible window/rain cues.",
            "event_label": "raindrop impacts on window",
        },
        {
            "keys": ["traffic", "car", "road"],
            "scene": "Road traffic scene with continuous vehicle ambience and foreground pass-by motion cues.",
            "tags": ["traffic", "vehicle", "road", "pass_by"],
            "ambience": "Layered road ambience with distant engines and tire noise.",
            "foley": "Foreground vehicle pass-by or motion cue aligned to visible traffic movement.",
            "event_label": "vehicle pass-by cue",
        },
        {
            "keys": ["keyboard", "typing", "laptop"],
            "scene": "Desk typing scene with close foreground keystrokes and low indoor ambience.",
            "tags": ["typing", "keyboard", "desk", "indoor"],
            "ambience": "Quiet indoor desk ambience with minimal background noise.",
            "foley": "Short keyboard taps aligned to visible typing motion.",
            "event_label": "keyboard typing taps",
        },
        {
            "keys": ["dog", "pet"],
            "scene": "Pet scene with foreground animal movement or vocal cue and natural room/outdoor ambience.",
            "tags": ["pet", "animal", "movement", "vocal"],
            "ambience": "Natural background ambience matching the pet scene setting.",
            "foley": "Animal movement, paw contact, or vocal cue aligned to visible pet action.",
            "event_label": "pet foreground cue",
        },
        {
            "keys": ["water", "beach", "wave", "sea"],
            "scene": "Waterfront scene with continuous wave ambience and soft foreground water movement.",
            "tags": ["water", "waves", "coast", "ambience"],
            "ambience": "Continuous wave and shoreline ambience.",
            "foley": "Foreground water movement cue aligned to visible wave or splash action.",
            "event_label": "foreground water movement",
        },
    ]

    for prof in profiles:
        if any(k in p for k in prof["keys"]):
            return prof

    readable = phrase.replace("_", " ").strip()
    return {
        "keys": [],
        "scene": f"{readable.title()} scene with one continuous ambience layer and one time-aligned foreground foley event.",
        "tags": [t for t in re.split(r"\s+", readable) if t][:6],
        "ambience": f"Continuous ambience bed matching the {readable} scene context.",
        "foley": f"Primary foreground foley cue aligned to the main visible action in the {readable} scene.",
        "event_label": "primary foreground foley cue",
    }


def is_id_like(text: str) -> bool:
    low = text.lower()
    bad = ["soundlayer_blueprint_seed", "bp_week", "seed_", "case_"]
    return any(x in low for x in bad)


def rewrite_blueprint(bp: Dict[str, Any]) -> Dict[str, Any]:
    raw_scene = str(bp.get("scene", {}).get("description", ""))
    source_seed_id = str(bp.get("source_seed_id", ""))
    phrase = clean_phrase(raw_scene or source_seed_id)
    profile = profile_from_phrase(phrase)

    scene = bp.setdefault("scene", {})
    scene["description"] = profile["scene"]
    scene["semantic_tags"] = profile["tags"]
    scene["semantic_source"] = {
        "method": "heuristic_from_seed_identifier",
        "raw_scene_description": raw_scene,
        "normalized_phrase": phrase,
    }

    for layer in bp.get("layers", []):
        if layer.get("layer_type") == "ambience":
            layer["generation_hint"] = profile["ambience"]
            layer["role"] = "continuous contextual sound bed"
        elif layer.get("layer_type") == "foley":
            layer["generation_hint"] = profile["foley"]
            layer["role"] = "time-aligned foreground event"

    duration = float(scene.get("duration_seconds", 8.0))
    for event in bp.get("events", []):
        layer_type = event.get("layer_type")
        if layer_type == "ambience":
            event["label"] = "contextual ambience bed"
            event["start_seconds"] = 0.0
            event["end_seconds"] = duration
            event["intensity"] = 0.35
            event["prompt"] = profile["ambience"]
            event["alignment_rule"] = "span_full_clip_as_context"
        elif layer_type == "foley":
            event["label"] = profile["event_label"]
            event["start_seconds"] = round(duration * 0.18, 3)
            event["end_seconds"] = round(duration * 0.72, 3)
            event["intensity"] = 0.78
            event["prompt"] = profile["foley"]
            event["alignment_rule"] = "align_to_main_visible_action"

    metadata = bp.setdefault("metadata", {})
    metadata["semantic_refinement"] = {
        "created_at": now_iso(),
        "script": "scripts/refine_week12_blueprint_v1_semantics.py",
        "boundary": "Heuristic semantic repair for Blueprint V1 control fields; not audio generation evidence.",
    }

    return bp


def flatten_events(blueprints: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for bp in blueprints:
        for event in bp.get("events", []):
            rows.append(
                {
                    "blueprint_id": bp["blueprint_id"],
                    "source_seed_id": bp["source_seed_id"],
                    "scene_description": bp["scene"]["description"],
                    "semantic_tags": "|".join(bp["scene"].get("semantic_tags", [])),
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
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_svg(path: Path, rows: List[Dict[str, Any]]) -> None:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["blueprint_id"], []).append(row)

    width = 1500
    left = 360
    track_width = 1040
    row_h = 82
    height = 90 + len(grouped) * row_h + 40

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="24" y="32" font-family="Arial" font-size="22" font-weight="700">Week12 Blueprint V1 Semantic Event Timeline</text>',
        '<text x="24" y="56" font-family="Arial" font-size="13">Semantic-refined scene/event/layer/timing contract. Not audio quality evidence.</text>',
    ]

    y = 86
    for bp_id, evs in grouped.items():
        desc = evs[0]["scene_description"][:92]
        duration = max(float(e["duration_seconds"]) for e in evs)
        parts.append(f'<text x="24" y="{y+18}" font-family="Arial" font-size="12" font-weight="700">{bp_id}</text>')
        parts.append(f'<text x="24" y="{y+38}" font-family="Arial" font-size="11">{desc}</text>')
        parts.append(f'<line x1="{left}" y1="{y+32}" x2="{left+track_width}" y2="{y+32}" stroke="#222" stroke-width="1"/>')

        for e in evs:
            start = float(e["start_seconds"])
            end = float(e["end_seconds"])
            x = left + start / duration * track_width
            w = max(5.0, (end - start) / duration * track_width)
            lane = {"ambience": 0, "foley": 18, "music": 36}.get(e["layer_type"], 54)
            label = f'{e["layer_type"]}: {e["label"]}'[:42]
            parts.append(f'<rect x="{x:.1f}" y="{y+14+lane}" width="{w:.1f}" height="12" rx="2" fill="#444"/>')
            parts.append(f'<text x="{x+4:.1f}" y="{y+11+lane}" font-family="Arial" font-size="9">{label}</text>')

        y += row_h

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_png(path: Path, rows: List[Dict[str, Any]]) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["blueprint_id"], []).append(row)

    fig, ax = plt.subplots(figsize=(15, max(4, len(grouped) * 1.15)))
    yticks = []
    ylabels = []

    y = 0
    for bp_id, evs in grouped.items():
        duration = max(float(e["duration_seconds"]) for e in evs)
        for e in evs:
            lane = {"ambience": 0.00, "foley": 0.24, "music": 0.48}.get(e["layer_type"], 0.72)
            start = float(e["start_seconds"])
            end = float(e["end_seconds"])
            ax.broken_barh([(start, max(0.05, end - start))], (y + lane, 0.18))
        yticks.append(y + 0.32)
        ylabels.append(f"{bp_id}\n{evs[0]['scene_description'][:58]}")
        y += 1

    ax.set_title("Week12 Blueprint V1 Semantic Event Timeline")
    ax.set_xlabel("seconds")
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.grid(True, axis="x", alpha=0.3)
    ax.set_ylim(-0.1, max(1, y))
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def validate_semantics(manifest: Dict[str, Any], rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []
    per_bp = []

    for bp in manifest.get("blueprints", []):
        desc = bp.get("scene", {}).get("description", "")
        bp_id = bp.get("blueprint_id")
        if is_id_like(desc):
            errors.append(f"{bp_id}: scene description is still id-like")
        if len(desc.split()) < 8:
            warnings.append(f"{bp_id}: scene description may be too short")
        for e in bp.get("events", []):
            if is_id_like(str(e.get("prompt", ""))):
                errors.append(f"{bp_id}/{e.get('event_id')}: prompt is still id-like")
        per_bp.append(
            {
                "blueprint_id": bp_id,
                "scene_description": desc,
                "semantic_tags": bp.get("scene", {}).get("semantic_tags", []),
                "event_labels": [e.get("label") for e in bp.get("events", [])],
            }
        )

    return {
        "schema_version": "week12_blueprint_v1_semantic_report",
        "created_at": now_iso(),
        "status": "PASS" if not errors else "FAIL",
        "blueprint_count": len(manifest.get("blueprints", [])),
        "event_count": len(rows),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
        "per_blueprint": per_bp,
    }


def main() -> int:
    if not MANIFEST.exists():
        raise SystemExit(f"[FAIL] missing manifest: {MANIFEST}")

    if not BACKUP.exists():
        shutil.copy2(MANIFEST, BACKUP)

    manifest = load_json(MANIFEST)
    blueprints = manifest.get("blueprints", [])
    if not isinstance(blueprints, list) or not blueprints:
        raise SystemExit("[FAIL] manifest.blueprints is missing or empty")

    refined = [rewrite_blueprint(bp) for bp in blueprints]
    manifest["blueprints"] = refined
    manifest["summary"]["semantic_refined"] = True
    manifest["summary"]["semantic_refined_at"] = now_iso()
    manifest["boundary"]["semantic_refinement"] = (
        "Scene/event prompts are heuristically repaired from seed identifiers. "
        "This improves control fields but does not claim generated audio quality."
    )

    rows = flatten_events(refined)
    report = validate_semantics(manifest, rows)

    validation = {
        "schema_version": "week12_blueprint_v1_validation_report",
        "created_at": now_iso(),
        "status": report["status"],
        "blueprint_count": report["blueprint_count"],
        "event_count": report["event_count"],
        "error_count": report["error_count"],
        "warning_count": report["warning_count"],
        "errors": report["errors"],
        "warnings": report["warnings"],
        "per_blueprint": [
            {
                "blueprint_id": bp["blueprint_id"],
                "source_seed_id": bp["source_seed_id"],
                "event_count": len(bp.get("events", [])),
                "layer_count": len(bp.get("layers", [])),
                "semantic_tags": bp.get("scene", {}).get("semantic_tags", []),
                "errors": [],
                "warnings": [],
            }
            for bp in refined
        ],
        "contact_sheet_png_created": True,
        "contact_sheet_svg_created": True,
    }

    write_json(MANIFEST, manifest)
    write_json(SEMANTIC_REPORT, report)
    write_json(VALIDATION, validation)
    write_jsonl(TIMELINE_JSONL, rows)
    write_csv(TIMELINE_CSV, rows)
    write_svg(CONTACT_SHEET_SVG, rows)
    png_ok = write_png(CONTACT_SHEET_PNG, rows)

    validation["contact_sheet_png_created"] = png_ok
    write_json(VALIDATION, validation)

    print("[PASS]" if report["status"] == "PASS" else "[FAIL]", "Week12 Blueprint V1 semantic refinement")
    print(f"blueprint_count={report['blueprint_count']}")
    print(f"event_count={report['event_count']}")
    print(f"error_count={report['error_count']}")
    print(f"warning_count={report['warning_count']}")
    print(f"manifest={MANIFEST}")
    print(f"timeline_csv={TIMELINE_CSV}")
    print(f"timeline_jsonl={TIMELINE_JSONL}")
    print(f"semantic_report={SEMANTIC_REPORT}")
    print(f"contact_sheet_svg={CONTACT_SHEET_SVG}")
    print(f"contact_sheet_png={CONTACT_SHEET_PNG if png_ok else 'NOT_CREATED'}")

    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())