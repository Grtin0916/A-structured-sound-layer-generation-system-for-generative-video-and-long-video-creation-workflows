#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


ROOT = Path.cwd()

PLACEMENT_MANIFEST_PATH = ROOT / "artifacts/manifests/week13_mix_placement_manifest.json"
PLACEMENT_TABLE_CSV_PATH = ROOT / "artifacts/evals/week13_mix_global_placement_table.csv"

OUT_PREVIEW_MANIFEST_PATH = ROOT / "artifacts/audio_mix/week13_mix_preview_manifest.json"
OUT_TIMELINE_CSV_PATH = ROOT / "artifacts/evals/week13_mix_timeline_dryrun.csv"
OUT_NAIVE_REGRESSION_CSV_PATH = ROOT / "artifacts/evals/week13_mix_naive_zero_regression_table.csv"
OUT_LOG_PATH = ROOT / "artifacts/logs" / f"week13_mix_dryrun_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"


def git_short_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip()
    except Exception:
        return "UNKNOWN"


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"missing required file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def dec(value: Any) -> Decimal | None:
    if value is None or value == "":
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None


def fnum(value: Decimal | None, ndigits: int = 6) -> float | None:
    if value is None:
        return None
    return float(round(value, ndigits))


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"missing required file: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def overlap(a_start: Decimal, a_end: Decimal, b_start: Decimal, b_end: Decimal) -> Decimal:
    return max(Decimal("0"), min(a_end, b_end) - max(a_start, b_start))


def main() -> int:
    placement_manifest = load_json(PLACEMENT_MANIFEST_PATH)
    if placement_manifest.get("status") != "PASS":
        raise RuntimeError("week13 placement manifest is not PASS; do not render dry-run")

    rows = read_rows(PLACEMENT_TABLE_CSV_PATH)
    blockers: list[str] = []

    timeline_rows: list[dict[str, Any]] = []
    regression_rows: list[dict[str, Any]] = []

    case_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    max_global_end = Decimal("0")

    for idx, row in enumerate(rows, start=1):
        candidate_id = row.get("candidateId") or f"row_{idx}"
        asset_time_mode = row.get("assetTimeMode", "")
        expected_start = dec(row.get("expectedStartSec"))
        global_start = dec(row.get("globalStartSec"))
        global_end = dec(row.get("globalEndSec"))
        placement_offset = dec(row.get("placementOffsetSec"))

        if expected_start is None:
            blockers.append(f"{candidate_id}:MISSING_EXPECTED_START_SEC")
            continue
        if global_start is None:
            blockers.append(f"{candidate_id}:MISSING_GLOBAL_START_SEC")
            continue
        if global_end is None:
            blockers.append(f"{candidate_id}:MISSING_GLOBAL_END_SEC")
            continue

        max_global_end = max(max_global_end, global_end)

        case_id = row.get("caseId") or "unknown_case"
        scene_id = row.get("sceneId") or "unknown_scene"

        correct_start = global_start
        correct_end = global_end

        naive_start = Decimal("0") if asset_time_mode == "event_local" else global_start
        duration = global_end - global_start
        naive_end = naive_start + duration

        naive_start_error = abs(naive_start - expected_start)
        fixed_start_error = abs(correct_start - expected_start) if asset_time_mode == "event_local" else Decimal("0")

        naive_would_misplace = (
            asset_time_mode == "event_local"
            and expected_start != Decimal("0")
            and naive_start != expected_start
        )
        fixed_is_misplaced = (
            asset_time_mode == "event_local"
            and correct_start != expected_start
        )

        if fixed_is_misplaced:
            blockers.append(f"{candidate_id}:FIXED_PLACEMENT_MISPLACED")

        timeline_item = {
            "dryRunOrder": idx,
            "candidateId": candidate_id,
            "audioUri": row.get("audioUri"),
            "sourceType": row.get("sourceType"),
            "caseId": case_id,
            "sceneId": scene_id,
            "eventId": row.get("eventId"),
            "layer": row.get("layer"),
            "label": row.get("label"),
            "assetTimeMode": asset_time_mode,
            "placementRequired": row.get("placementRequired"),
            "expectedStartSec": fnum(expected_start),
            "expectedEndSec": row.get("expectedEndSec"),
            "correctGlobalStartSec": fnum(correct_start),
            "correctGlobalEndSec": fnum(correct_end),
            "placementOffsetSec": fnum(placement_offset),
            "dryRunAction": (
                "place_local_audio_at_expectedStartSec"
                if asset_time_mode == "event_local"
                else "place_full_clip_on_global_timeline"
            ),
            "runtimeMixerInput": {
                "audioUri": row.get("audioUri"),
                "globalStartSec": fnum(correct_start),
                "globalEndSec": fnum(correct_end),
                "layer": row.get("layer"),
                "assetTimeMode": asset_time_mode,
                "placementOffsetSec": fnum(placement_offset),
            },
        }
        timeline_rows.append(timeline_item)
        case_groups[case_id].append(timeline_item)

        regression_rows.append({
            "candidateId": candidate_id,
            "eventId": row.get("eventId"),
            "label": row.get("label"),
            "assetTimeMode": asset_time_mode,
            "expectedStartSec": fnum(expected_start),
            "fixedGlobalStartSec": fnum(correct_start),
            "naiveZeroGlobalStartSec": fnum(naive_start),
            "fixedStartErrorSec": fnum(fixed_start_error),
            "naiveZeroStartErrorSec": fnum(naive_start_error),
            "naiveZeroWouldMisplace": naive_would_misplace,
            "fixedIsMisplaced": fixed_is_misplaced,
        })

    # Case-level lane/overlap summary. 这是 dry-run，不生成 final mix。
    case_summaries: list[dict[str, Any]] = []
    for case_id, items in sorted(case_groups.items()):
        layer_counts: dict[str, int] = defaultdict(int)
        event_local_count = 0
        full_clip_count = 0
        overlaps: list[dict[str, Any]] = []

        for item in items:
            layer_counts[str(item.get("layer"))] += 1
            if item.get("assetTimeMode") == "event_local":
                event_local_count += 1
            if item.get("assetTimeMode") == "full_clip":
                full_clip_count += 1

        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a = items[i]
                b = items[j]
                a_s = dec(a["correctGlobalStartSec"])
                a_e = dec(a["correctGlobalEndSec"])
                b_s = dec(b["correctGlobalStartSec"])
                b_e = dec(b["correctGlobalEndSec"])
                if None in (a_s, a_e, b_s, b_e):
                    continue
                ov = overlap(a_s, a_e, b_s, b_e)
                if ov > 0:
                    overlaps.append({
                        "a": a["candidateId"],
                        "b": b["candidateId"],
                        "overlapSec": fnum(ov),
                        "layers": [a.get("layer"), b.get("layer")],
                    })

        case_summaries.append({
            "caseId": case_id,
            "candidateCount": len(items),
            "fullClipCount": full_clip_count,
            "eventLocalCount": event_local_count,
            "layerCounts": dict(layer_counts),
            "overlapCount": len(overlaps),
            "overlaps": overlaps[:20],
        })

    candidate_count = len(timeline_rows)
    event_local_count = sum(1 for r in timeline_rows if r["assetTimeMode"] == "event_local")
    full_clip_count = sum(1 for r in timeline_rows if r["assetTimeMode"] == "full_clip")
    fixed_misplaced_count = sum(1 for r in regression_rows if r["fixedIsMisplaced"])
    naive_zero_misplaced_count = sum(1 for r in regression_rows if r["naiveZeroWouldMisplace"])

    if candidate_count != 10:
        blockers.append(f"EXPECTED_10_CANDIDATES_GOT_{candidate_count}")
    if event_local_count != 5:
        blockers.append(f"EXPECTED_5_EVENT_LOCAL_GOT_{event_local_count}")
    if full_clip_count != 5:
        blockers.append(f"EXPECTED_5_FULL_CLIP_GOT_{full_clip_count}")
    if fixed_misplaced_count != 0:
        blockers.append(f"FIXED_PLACEMENT_MISPLACED_COUNT_{fixed_misplaced_count}")
    if naive_zero_misplaced_count != 5:
        blockers.append(f"EXPECTED_NAIVE_ZERO_MISPLACED_5_GOT_{naive_zero_misplaced_count}")

    status = "PASS" if not blockers else "FAIL"

    preview_manifest = {
        "status": status,
        "scope": "week13_mix_dryrun_preview_v0",
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "sourceRepo": {
            "path": str(ROOT),
            "head": git_short_head(),
        },
        "dryRunOnly": True,
        "inputs": {
            "placementManifest": str(PLACEMENT_MANIFEST_PATH.relative_to(ROOT)),
            "placementTableCsv": str(PLACEMENT_TABLE_CSV_PATH.relative_to(ROOT)),
        },
        "outputs": {
            "previewManifest": str(OUT_PREVIEW_MANIFEST_PATH.relative_to(ROOT)),
            "timelineDryrunCsv": str(OUT_TIMELINE_CSV_PATH.relative_to(ROOT)),
            "naiveZeroRegressionCsv": str(OUT_NAIVE_REGRESSION_CSV_PATH.relative_to(ROOT)),
            "log": str(OUT_LOG_PATH.relative_to(ROOT)),
        },
        "candidateCount": candidate_count,
        "assetTimeModeCounts": {
            "full_clip": full_clip_count,
            "event_local": event_local_count,
        },
        "timelineDurationSec": fnum(max_global_end),
        "fixedPlacementMisplacedCount": fixed_misplaced_count,
        "naiveZeroWouldMisplaceCount": naive_zero_misplaced_count,
        "regressionGuard": {
            "purpose": "Prevent event_local foley assets from being accidentally consumed as t=0 global audio.",
            "passCondition": "fixedPlacementMisplacedCount == 0 and naiveZeroWouldMisplaceCount == 5",
            "fixedPlacementRule": "event_local.globalStartSec = expectedStartSec",
            "negativeControl": "naive_zero places event_local at 0 and should misplace all 5 event_local assets in the current fixture.",
        },
        "caseSummaries": case_summaries,
        "timeline": timeline_rows,
        "boundaryStatement": (
            "This is a mixer/runtime dry-run manifest. It validates placement semantics and regression risk only. "
            "It does not generate a final audio mix, does not claim semantic audio quality, human audition, "
            "durable registry, real object storage, or production readiness."
        ),
        "blockers": blockers,
    }

    OUT_PREVIEW_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_TIMELINE_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_NAIVE_REGRESSION_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    OUT_PREVIEW_MANIFEST_PATH.write_text(
        json.dumps(preview_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    timeline_fields = [
        "dryRunOrder",
        "candidateId",
        "audioUri",
        "sourceType",
        "caseId",
        "sceneId",
        "eventId",
        "layer",
        "label",
        "assetTimeMode",
        "placementRequired",
        "expectedStartSec",
        "expectedEndSec",
        "correctGlobalStartSec",
        "correctGlobalEndSec",
        "placementOffsetSec",
        "dryRunAction",
    ]

    with OUT_TIMELINE_CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=timeline_fields)
        writer.writeheader()
        for r in timeline_rows:
            writer.writerow({k: r.get(k) for k in timeline_fields})

    regression_fields = [
        "candidateId",
        "eventId",
        "label",
        "assetTimeMode",
        "expectedStartSec",
        "fixedGlobalStartSec",
        "naiveZeroGlobalStartSec",
        "fixedStartErrorSec",
        "naiveZeroStartErrorSec",
        "naiveZeroWouldMisplace",
        "fixedIsMisplaced",
    ]

    with OUT_NAIVE_REGRESSION_CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=regression_fields)
        writer.writeheader()
        for r in regression_rows:
            writer.writerow({k: r.get(k) for k in regression_fields})

    log_text = "\n".join([
        f"status={status}",
        f"candidateCount={candidate_count}",
        f"fullClipCount={full_clip_count}",
        f"eventLocalCount={event_local_count}",
        f"timelineDurationSec={fnum(max_global_end)}",
        f"fixedPlacementMisplacedCount={fixed_misplaced_count}",
        f"naiveZeroWouldMisplaceCount={naive_zero_misplaced_count}",
        f"blockers={blockers}",
        f"previewManifest={OUT_PREVIEW_MANIFEST_PATH}",
        f"timelineDryrunCsv={OUT_TIMELINE_CSV_PATH}",
        f"naiveZeroRegressionCsv={OUT_NAIVE_REGRESSION_CSV_PATH}",
    ])
    OUT_LOG_PATH.write_text(log_text + "\n", encoding="utf-8")

    print(json.dumps({
        "status": status,
        "candidateCount": candidate_count,
        "assetTimeModeCounts": {
            "full_clip": full_clip_count,
            "event_local": event_local_count,
        },
        "timelineDurationSec": fnum(max_global_end),
        "fixedPlacementMisplacedCount": fixed_misplaced_count,
        "naiveZeroWouldMisplaceCount": naive_zero_misplaced_count,
        "outputs": preview_manifest["outputs"],
        "blockers": blockers,
    }, ensure_ascii=False, indent=2))

    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())