#!/usr/bin/env python3
"""
Finalize Week12 Blueprint V1 pack.

Purpose:
- Patch the remaining generic bike-pass semantics.
- Regenerate CSV/JSONL timeline with candidate-audio-ready prompts.
- Regenerate a more readable contact sheet.
- Emit a compact final summary for commit evidence.

Boundary:
- No audio generation.
- No subjective audio quality claim.
- This only finalizes the structured Blueprint V1 contract.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


MANIFEST = Path("artifacts/manifests/week12_blueprint_v1_manifest.json")
TIMELINE_CSV = Path("artifacts/manifests/week12_event_timeline.csv")
TIMELINE_JSONL = Path("artifacts/manifests/week12_event_timeline.jsonl")
VALIDATION = Path("artifacts/manifests/week12_blueprint_v1_validation_report.json")
SEMANTIC_REPORT = Path("artifacts/manifests/week12_blueprint_v1_semantic_report.json")
FINAL_SUMMARY = Path("artifacts/manifests/week12_blueprint_v1_final_summary.json")
CONTACT_SHEET_PNG = Path("artifacts/visuals/week12_event_timeline_contact_sheet.png")
CONTACT_SHEET_SVG = Path("artifacts/visuals/week12_event_timeline_contact_sheet.svg")


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


def is_bike_case(bp: Dict[str, Any]) -> bool:
    text = " ".join(
        [
            str(bp.get("source_seed_id", "")),
            str(bp.get("scene", {}).get("description", "")),
            " ".join(bp.get("scene", {}).get("semantic_tags", [])),
        ]
    ).lower()
    return "bike" in text or "bicycle" in text or "pass" in text


def patch_bike_semantics(bp: Dict[str, Any]) -> None:
    scene = bp.setdefault("scene", {})
    scene["description"] = (
        "Bicycle pass-by scene with continuous outdoor ambience and a foreground bicycle motion cue aligned to the visible pass."
    )
    scene["semantic_tags"] = ["bike", "pass_by", "motion", "outdoor_ambience", "foreground_foley"]
    scene.setdefault("semantic_source", {})
    scene["semantic_source"]["final_patch"] = {
        "script": "scripts/finalize_week12_blueprint_v1_pack.py",
        "reason": "Replace fallback bike-pass wording with candidate-audio-ready control semantics.",
        "created_at": now_iso(),
    }

    for layer in bp.get("layers", []):
        if layer.get("layer_type") == "ambience":
            layer["role"] = "continuous outdoor context bed"
            layer["generation_hint"] = "Outdoor ambience with light street or path noise, kept below the foreground bicycle cue."
        elif layer.get("layer_type") == "foley":
            layer["role"] = "time-aligned bicycle pass-by cue"
            layer["generation_hint"] = "Bicycle wheel rotation, chain/gear texture, and close pass-by motion aligned to the visible bike movement."

    duration = float(scene.get("duration_seconds", 8.0))
    for event in bp.get("events", []):
        if event.get("layer_type") == "ambience":
            event["label"] = "outdoor bike-pass ambience"
            event["start_seconds"] = 0.0
            event["end_seconds"] = duration
            event["prompt"] = "Outdoor ambience with subtle street/path noise under a bicycle pass-by foreground event."
            event["alignment_rule"] = "span_full_clip_as_context"
            event["intensity"] = 0.34
        elif event.get("layer_type") == "foley":
            event["label"] = "bicycle pass-by motion cue"
            event["start_seconds"] = round(duration * 0.15, 3)
            event["end_seconds"] = round(duration * 0.68, 3)
            event["prompt"] = "Bicycle wheel, chain, and close pass-by motion cue synchronized to the visible bike movement."
            event["alignment_rule"] = "align_to_visible_bike_pass_by"
            event["intensity"] = 0.80


def flatten_events(blueprints: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for bp in blueprints:
        scene = bp["scene"]
        for event in bp.get("events", []):
            rows.append(
                {
                    "blueprint_id": bp["blueprint_id"],
                    "source_seed_id": bp["source_seed_id"],
                    "scene_description": scene["description"],
                    "semantic_tags": "|".join(scene.get("semantic_tags", [])),
                    "duration_seconds": scene["duration_seconds"],
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


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def validate_pack(manifest: Dict[str, Any], rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []

    if manifest.get("summary", {}).get("blueprint_count") != 5:
        errors.append("expected blueprint_count=5")

    if len(rows) != 10:
        errors.append("expected event_count=10")

    for bp in manifest.get("blueprints", []):
        desc = bp.get("scene", {}).get("description", "")
        if "soundlayer_blueprint_seed" in desc or "bp_week" in desc:
            errors.append(f"{bp.get('blueprint_id')}: scene description still contains raw seed token")
        if is_bike_case(bp) and not any(x in desc.lower() for x in ["bicycle pass-by", "bike pass-by"]):
            errors.append(f"{bp.get('blueprint_id')}: bike case was not semantically finalized")
        for event in bp.get("events", []):
            prompt = event.get("prompt", "")
            if "soundlayer_blueprint_seed" in prompt or "bp_week" in prompt:
                errors.append(f"{bp.get('blueprint_id')}/{event.get('event_id')}: prompt still contains raw seed token")

    return {
        "schema_version": "week12_blueprint_v1_final_summary",
        "created_at": now_iso(),
        "status": "PASS" if not errors else "FAIL",
        "blueprint_count": len(manifest.get("blueprints", [])),
        "event_count": len(rows),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
        "deliverables": {
            "schema": "schemas/soundlayer_blueprint_v1.schema.json",
            "manifest": str(MANIFEST),
            "timeline_csv": str(TIMELINE_CSV),
            "timeline_jsonl": str(TIMELINE_JSONL),
            "semantic_report": str(SEMANTIC_REPORT),
            "validation_report": str(VALIDATION),
            "contact_sheet_png": str(CONTACT_SHEET_PNG),
            "contact_sheet_svg": str(CONTACT_SHEET_SVG),
        },
        "boundary": (
            "Finalized Blueprint V1 provides scene/event/layer/timing/prompt/artifact pointers. "
            "It does not claim audio generation, subjective audio quality, or downstream Java/Cloud integration."
        ),
    }


def write_svg(path: Path, rows: List[Dict[str, Any]]) -> None:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["blueprint_id"], []).append(row)

    width = 1680
    height = 115 + 95 * len(grouped)
    left = 500
    track_width = 1060
    row_h = 95

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="24" y="34" font-family="Arial" font-size="24" font-weight="700">Week12 Blueprint V1 Final Semantic Event Timeline</text>',
        '<text x="24" y="60" font-family="Arial" font-size="14">Two lanes per scene: ambience spans the clip; foley is aligned to the foreground event. This is a structured control artifact, not audio quality evidence.</text>',
        '<rect x="24" y="78" width="16" height="10" fill="#7a7a7a"/><text x="48" y="88" font-family="Arial" font-size="12">ambience</text>',
        '<rect x="130" y="78" width="16" height="10" fill="#1f77b4"/><text x="154" y="88" font-family="Arial" font-size="12">foley</text>',
    ]

    y = 110
    for bp_id, evs in grouped.items():
        desc = evs[0]["scene_description"]
        tags = evs[0]["semantic_tags"]
        duration = max(float(e["duration_seconds"]) for e in evs)

        parts.append(f'<text x="24" y="{y+18}" font-family="Arial" font-size="13" font-weight="700">{bp_id}</text>')
        parts.append(f'<text x="24" y="{y+39}" font-family="Arial" font-size="12">{desc[:110]}</text>')
        parts.append(f'<text x="24" y="{y+59}" font-family="Arial" font-size="11">tags: {tags}</text>')
        parts.append(f'<line x1="{left}" y1="{y+42}" x2="{left+track_width}" y2="{y+42}" stroke="#222" stroke-width="1"/>')
        parts.append(f'<text x="{left}" y="{y+74}" font-family="Arial" font-size="10">0s</text>')
        parts.append(f'<text x="{left+track_width-28}" y="{y+74}" font-family="Arial" font-size="10">{duration:g}s</text>')

        for event in evs:
            start = float(event["start_seconds"])
            end = float(event["end_seconds"])
            x = left + start / duration * track_width
            w = max(6, (end - start) / duration * track_width)
            layer = event["layer_type"]
            bar_y = y + (25 if layer == "ambience" else 47)
            color = "#7a7a7a" if layer == "ambience" else "#1f77b4"
            label = f'{layer}: {event["label"]}'
            parts.append(f'<rect x="{x:.1f}" y="{bar_y}" width="{w:.1f}" height="13" rx="2" fill="{color}"/>')
            parts.append(f'<text x="{min(x+4, left+track_width-260):.1f}" y="{bar_y-3}" font-family="Arial" font-size="10">{label[:54]}</text>')

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

    fig, ax = plt.subplots(figsize=(17, 6.8))
    yticks = []
    ylabels = []
    y = 0.0

    for bp_id, evs in grouped.items():
        duration = max(float(e["duration_seconds"]) for e in evs)
        for event in evs:
            start = float(event["start_seconds"])
            end = float(event["end_seconds"])
            lane = 0.00 if event["layer_type"] == "ambience" else 0.30
            ax.broken_barh([(start, max(0.05, end - start))], (y + lane, 0.18))
            ax.text(
                start + 0.02,
                y + lane + 0.22,
                f'{event["layer_type"]}: {event["label"]}',
                fontsize=7,
                va="bottom",
            )
        yticks.append(y + 0.25)
        ylabels.append(f"{bp_id}\n{evs[0]['scene_description'][:72]}")
        y += 1.0

    ax.set_title("Week12 Blueprint V1 Final Semantic Event Timeline")
    ax.set_xlabel("seconds")
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_xlim(-0.1, 8.4)
    ax.grid(True, axis="x", alpha=0.3)
    ax.set_ylim(-0.1, y)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def main() -> int:
    manifest = load_json(MANIFEST)
    blueprints = manifest.get("blueprints", [])
    if not blueprints:
        raise SystemExit("[FAIL] manifest.blueprints is empty")

    for bp in blueprints:
        if is_bike_case(bp):
            patch_bike_semantics(bp)

    manifest.setdefault("summary", {})
    manifest["summary"]["finalized"] = True
    manifest["summary"]["finalized_at"] = now_iso()
    manifest["summary"]["finalized_by"] = "scripts/finalize_week12_blueprint_v1_pack.py"

    rows = flatten_events(blueprints)
    summary = validate_pack(manifest, rows)

    write_json(MANIFEST, manifest)
    write_csv(TIMELINE_CSV, rows)
    write_jsonl(TIMELINE_JSONL, rows)
    write_svg(CONTACT_SHEET_SVG, rows)
    png_ok = write_png(CONTACT_SHEET_PNG, rows)

    validation = load_json(VALIDATION)
    validation["status"] = summary["status"]
    validation["error_count"] = summary["error_count"]
    validation["warning_count"] = summary["warning_count"]
    validation["errors"] = summary["errors"]
    validation["warnings"] = summary["warnings"]
    validation["contact_sheet_png_created"] = png_ok
    validation["contact_sheet_svg_created"] = CONTACT_SHEET_SVG.exists()
    validation["finalized"] = True
    validation["finalized_at"] = now_iso()
    write_json(VALIDATION, validation)

    semantic = load_json(SEMANTIC_REPORT)
    semantic["status"] = summary["status"]
    semantic["error_count"] = summary["error_count"]
    semantic["warning_count"] = summary["warning_count"]
    semantic["errors"] = summary["errors"]
    semantic["warnings"] = summary["warnings"]
    semantic["finalized"] = True
    semantic["finalized_at"] = now_iso()
    semantic["per_blueprint"] = [
        {
            "blueprint_id": bp["blueprint_id"],
            "scene_description": bp["scene"]["description"],
            "semantic_tags": bp["scene"].get("semantic_tags", []),
            "event_labels": [e.get("label") for e in bp.get("events", [])],
        }
        for bp in blueprints
    ]
    write_json(SEMANTIC_REPORT, semantic)

    summary["contact_sheet_png_created"] = png_ok
    summary["contact_sheet_svg_created"] = CONTACT_SHEET_SVG.exists()
    write_json(FINAL_SUMMARY, summary)

    print("[PASS]" if summary["status"] == "PASS" else "[FAIL]", "Week12 Blueprint V1 final pack")
    print(f"blueprint_count={summary['blueprint_count']}")
    print(f"event_count={summary['event_count']}")
    print(f"error_count={summary['error_count']}")
    print(f"warning_count={summary['warning_count']}")
    print(f"contact_sheet_png_created={png_ok}")
    print(f"final_summary={FINAL_SUMMARY}")
    return 0 if summary["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())