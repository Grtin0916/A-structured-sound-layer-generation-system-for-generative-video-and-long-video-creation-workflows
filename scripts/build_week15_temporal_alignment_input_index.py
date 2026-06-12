#!/usr/bin/env python3
"""
Build Week15 temporal alignment input index from Week13 Candidate Audio Bank V1 evidence.

This script does not score semantic quality or human audition.
It only materializes the eval input contract:
candidate -> placement -> timing fields -> audio artifact reference -> next eval readiness.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def find_key(obj: Any, names: list[str]) -> Any:
    targets = {x.lower() for x in names}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower() in targets:
                return v
        for v in obj.values():
            found = find_key(v, names)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = find_key(item, names)
            if found is not None:
                return found
    return None


def collect_dicts_with_candidate_id(obj: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if isinstance(obj, dict):
        if find_key(obj, ["candidateId", "candidate_id", "id"]) is not None:
            out.append(obj)
        for v in obj.values():
            out.extend(collect_dicts_with_candidate_id(v))
    elif isinstance(obj, list):
        for item in obj:
            out.extend(collect_dicts_with_candidate_id(item))
    return out


def norm_candidate_id(row: dict[str, Any]) -> str | None:
    v = find_key(row, ["candidateId", "candidate_id", "id"])
    if v is None:
        return None
    return str(v)


def first_present(*values: Any) -> Any:
    for v in values:
        if v is not None:
            return v
    return None


def as_float(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except Exception:
        return None


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mainbase", default=".")
    ap.add_argument(
        "--out",
        default="artifacts/evals/week15_temporal_alignment_input_index.json",
    )
    args = ap.parse_args()

    root = Path(args.mainbase).expanduser().resolve()

    stage_gate_path = root / "artifacts/manifests/week13_candidate_bank_v1_stage_gate_feedback.json"
    demo_index_path = root / "artifacts/manifests/week13_candidate_bank_demo_index.json"
    placement_manifest_path = root / "artifacts/manifests/week13_mix_placement_manifest.json"
    placement_table_csv_path = root / "artifacts/evals/week13_mix_global_placement_table.csv"
    preview_manifest_path = root / "artifacts/audio_mix/week13_mix_preview_manifest.json"

    stage_gate = read_json(stage_gate_path) or {}
    demo_index = read_json(demo_index_path) or {}
    placement_manifest = read_json(placement_manifest_path) or {}
    preview_manifest = read_json(preview_manifest_path) or {}
    placement_rows = read_csv_rows(placement_table_csv_path)

    blockers: list[str] = []

    if stage_gate.get("status") != "PASS":
        blockers.append(f"stage gate feedback not PASS: {stage_gate.get('status')}")
    if stage_gate.get("transitionDecision") != "ENTER_WEEK15_TEMPORAL_ALIGNMENT_EVAL_INPUT":
        blockers.append(f"unexpected transitionDecision: {stage_gate.get('transitionDecision')}")

    candidate_records = collect_dicts_with_candidate_id(demo_index)
    placement_records = collect_dicts_with_candidate_id(placement_manifest)
    preview_records = collect_dicts_with_candidate_id(preview_manifest)

    by_id: dict[str, dict[str, Any]] = {}

    def merge_record(source: str, row: dict[str, Any]) -> None:
        cid = norm_candidate_id(row)
        if not cid:
            return
        item = by_id.setdefault(cid, {"candidateId": cid, "sources": []})
        if source not in item["sources"]:
            item["sources"].append(source)

        item.setdefault("assetTimeMode", find_key(row, ["assetTimeMode", "timeMode"]))
        item.setdefault("expectedStartSec", find_key(row, ["expectedStartSec", "eventStartSec", "startSec"]))
        item.setdefault("globalStartSec", find_key(row, ["globalStartSec", "placementStartSec", "mixStartSec"]))
        item.setdefault("globalEndSec", find_key(row, ["globalEndSec", "placementEndSec", "mixEndSec"]))
        item.setdefault("audioUri", find_key(row, ["audioUri", "audioPath", "artifactPath", "materializedPath", "wavPath"]))
        item.setdefault("sourceType", find_key(row, ["sourceType", "generatorName"]))
        item.setdefault("placementRequired", find_key(row, ["placementRequired"]))
        item.setdefault("status", find_key(row, ["status"]))

    for row in candidate_records:
        merge_record("week13_candidate_bank_demo_index", row)
    for row in placement_records:
        merge_record("week13_mix_placement_manifest", row)
    for row in preview_records:
        merge_record("week13_mix_preview_manifest", row)

    for row in placement_rows:
        cid = first_present(row.get("candidateId"), row.get("candidate_id"), row.get("id"))
        if not cid:
            continue
        item = by_id.setdefault(str(cid), {"candidateId": str(cid), "sources": []})
        if "week13_mix_global_placement_table_csv" not in item["sources"]:
            item["sources"].append("week13_mix_global_placement_table_csv")
        item.setdefault("assetTimeMode", row.get("assetTimeMode") or row.get("timeMode"))
        item.setdefault("expectedStartSec", row.get("expectedStartSec") or row.get("eventStartSec"))
        item.setdefault("globalStartSec", row.get("globalStartSec") or row.get("placementStartSec"))
        item.setdefault("globalEndSec", row.get("globalEndSec") or row.get("placementEndSec"))
        item.setdefault("audioUri", row.get("audioUri") or row.get("audioPath") or row.get("artifactPath"))

    eval_inputs: list[dict[str, Any]] = []
    for cid in sorted(by_id):
        item = by_id[cid]
        asset_time_mode = item.get("assetTimeMode")
        expected_start = as_float(item.get("expectedStartSec"))
        global_start = as_float(item.get("globalStartSec"))

        timing_ready = global_start is not None and (
            str(asset_time_mode) == "full_clip"
            or expected_start is not None
            or bool(item.get("placementRequired")) is False
        )

        event_local_expected_match = None
        if str(asset_time_mode) == "event_local" and expected_start is not None and global_start is not None:
            event_local_expected_match = abs(global_start - expected_start) <= 1e-6

        eval_inputs.append({
            "candidateId": cid,
            "assetTimeMode": asset_time_mode,
            "expectedStartSec": expected_start,
            "globalStartSec": global_start,
            "globalEndSec": as_float(item.get("globalEndSec")),
            "audioUri": item.get("audioUri"),
            "sourceType": item.get("sourceType"),
            "placementRequired": item.get("placementRequired"),
            "timingEvalInputReady": timing_ready,
            "eventLocalExpectedStartMatch": event_local_expected_match,
            "evidenceSources": item.get("sources", []),
            "nextMetricPlaceholders": {
                "onsetPeakSec": None,
                "onsetDeltaSec": None,
                "energyWindowMean": None,
                "alignmentStatus": "PENDING_WEEK15_SCORING",
            },
        })

    candidate_count = len(eval_inputs)
    ready_count = sum(1 for x in eval_inputs if x["timingEvalInputReady"])
    event_local_count = sum(1 for x in eval_inputs if x["assetTimeMode"] == "event_local")
    event_local_match_count = sum(1 for x in eval_inputs if x["eventLocalExpectedStartMatch"] is True)

    if candidate_count != 10:
        blockers.append(f"candidateCount expected 10, got {candidate_count}")
    if ready_count != 10:
        blockers.append(f"timingEvalInputReadyCount expected 10, got {ready_count}")
    if event_local_count and event_local_match_count != event_local_count:
        blockers.append(
            f"eventLocalExpectedStartMatch expected {event_local_count}, got {event_local_match_count}"
        )

    status = "PASS" if not blockers else "FAIL"

    report = {
        "schemaVersion": "week15.temporal_alignment_input_index.v1",
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "purpose": "Materialize Week15 temporal alignment eval inputs from Week13 Candidate Audio Bank V1 evidence.",
        "inputs": {
            "stageGateFeedback": str(stage_gate_path),
            "candidateBankDemoIndex": str(demo_index_path),
            "mixPlacementManifest": str(placement_manifest_path),
            "mixGlobalPlacementTableCsv": str(placement_table_csv_path),
            "mixPreviewManifest": str(preview_manifest_path),
        },
        "summary": {
            "candidateCount": candidate_count,
            "timingEvalInputReadyCount": ready_count,
            "eventLocalCount": event_local_count,
            "eventLocalExpectedStartMatchCount": event_local_match_count,
            "blockerCount": len(blockers),
        },
        "evalInputs": eval_inputs,
        "blockers": blockers,
        "boundary": [
            "does_not_score_semantic_audio_quality",
            "does_not_claim_human_audition_passed",
            "does_not_claim_final_mix_readiness",
            "does_not_regenerate_audio",
            "prepares_inputs_for_week15_onset_energy_scoring",
        ],
    }

    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps({
        "output": str(out),
        "status": status,
        "summary": report["summary"],
        "blockers": blockers,
    }, indent=2, ensure_ascii=False))

    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())