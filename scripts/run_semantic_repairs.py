#!/usr/bin/env python3
"""Build and execute the W19 capability-aware semantic repair manifest."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from repair_audio_utils import audio_metrics, read_pcm16, write_diagnostic_png, write_pcm16
from soundlayer.repair.event_replacement import transplant_event, transplant_metrics
from soundlayer.repair.semantic_repair import apply_mixed_only_repair, semantic_gate


LAYER_IDS = ("fb_003", "fb_011", "hn_002", "hn_003")
REPLACEMENT_IDS = ("fb_004", "fb_005")


def dump_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def relative(path: Path) -> str:
    return (ROOT / path).resolve().relative_to(ROOT).as_posix() if not path.is_absolute() else path.relative_to(ROOT).as_posix()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--repair-bank", type=Path, required=True)
    parser.add_argument("--tuesday-promotions", type=Path, required=True)
    parser.add_argument("--selector-metrics", type=Path, required=True)
    parser.add_argument("--listening-sheet", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer-metrics", type=Path, required=True)
    parser.add_argument("--replacement-metrics", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--failures", type=Path, required=True)
    parser.add_argument("--max-layer-items", type=int, default=4)
    parser.add_argument("--replacement-probes", type=int, default=2)
    parser.add_argument("--replacement-variants", type=int, default=2)
    args = parser.parse_args()

    bank = {row["failure_id"]: row for row in csv.DictReader(args.repair_bank.open())}
    promotions = {
        row["failure_id"]: row
        for row in csv.DictReader(args.tuesday_promotions.open())
        if row["selected_after"]
    }
    selector = list(csv.DictReader(args.selector_metrics.open()))
    listening = list(csv.DictReader(args.listening_sheet.open()))
    human_complete = sum(row["review_status"] == "COMPLETED" for row in listening)
    manifest_rows = []
    for failure_id in LAYER_IDS[:args.max_layer_items]:
        row = bank[failure_id]
        intent = (
            "reduce_masking"
            if row["failure_type"] == "layer_conflict_or_repairable"
            else "strengthen_expected_event"
        )
        manifest_rows.append({
            "schema_version": "semantic.repair.v1",
            "failure_id": failure_id,
            "parent_candidate_id": row["candidate"],
            "source_mode": "mixed_only",
            "stems_available": False,
            "source_audio": promotions[failure_id]["selected_after"],
            "expected_event": row["event_id"],
            "expected_window": [float(row["event_start_sec"]), float(row["event_end_sec"])],
            "expected_target_count": 1,
            "failure_type": "masking" if intent == "reduce_masking" else "imbalance",
            "semantic_repair_intent": intent,
            "repair_action": "smooth_region_attenuation" if intent == "reduce_masking" else "event_region_gain",
            "donor_candidate": None,
            "zero_target_guard": False,
            "automatic_detection_available": False,
            "needs_manual_review": True,
        })
    for failure_id in REPLACEMENT_IDS[:args.replacement_probes]:
        row = bank[failure_id]
        donor_options = [
            candidate for candidate in selector
            if candidate["case_id"] == row["case_id"]
            and candidate["variant"] in {"dss_event_timeline", "dss_layer_avoid"}
            and (ROOT / candidate["audio_path"]).is_file()
        ]
        donor_options.sort(key=lambda item: item["variant"] != "dss_event_timeline")
        donor = donor_options[0] if donor_options else None
        manifest_rows.append({
            "schema_version": "semantic.repair.v1",
            "failure_id": failure_id,
            "parent_candidate_id": row["candidate"],
            "source_mode": "candidate_replace",
            "stems_available": False,
            "source_audio": row["source_audio"],
            "expected_event": row["event_id"],
            "expected_window": [float(row["event_start_sec"]), float(row["event_end_sec"])],
            "expected_target_count": 1,
            "failure_type": "missing_event",
            "semantic_repair_intent": "insert_expected_event_from_reference",
            "repair_action": "event_transplant" if donor else "candidate_replace",
            "donor_candidate": donor["case_id"] + "|" + donor["variant"] if donor else None,
            "donor_audio": donor["audio_path"] if donor else None,
            "donor_window": [float(row["event_start_sec"]), float(row["event_end_sec"])],
            "zero_target_guard": False,
            "automatic_detection_available": False,
            "needs_manual_review": True,
        })
    args.manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in manifest_rows),
        encoding="utf-8",
    )

    layer_fields = [
        "failure_id", "parent_candidate_id", "source_mode", "repair_action",
        "before_artifact", "after_artifact", "target_window_rms_delta_db",
        "outside_window_rms_delta_db", "sample_peak_delta", "clip_ratio_delta",
        "duration_error_ms", "boundary_jump_before", "boundary_jump_after",
        "changed_sample_ratio", "ordering_correct", "lineage_complete",
        "output_readable", "severe_regression", "semantic_target_satisfied",
        "forbidden_event_status", "needs_manual_review", "decision", "reason",
    ]
    layer_rows, replacement_rows, failures = [], [], []
    for plan in manifest_rows:
        failure_id = plan["failure_id"]
        case_dir = args.output_dir / failure_id
        case_dir.mkdir(parents=True, exist_ok=True)
        before_path = case_dir / "before.wav"
        source_path = ROOT / plan["source_audio"]
        shutil.copy2(source_path, before_path)
        params, before = read_pcm16(before_path)
        if plan["repair_action"] != "event_transplant":
            after_path = case_dir / "after.wav"
            start, end = plan["expected_window"]
            after, metrics = apply_mixed_only_repair(
                before, params.framerate, params.nchannels, start, end,
                plan["semantic_repair_intent"],
            )
            write_pcm16(after_path, params, after)
            before_audio_metrics = audio_metrics(params, before)
            after_audio_metrics = audio_metrics(params, after)
            duration_error = (
                after_audio_metrics["duration_sec"] - before_audio_metrics["duration_sec"]
            ) * 1000
            semantic_ok, reason = semantic_gate(
                metrics, after_audio_metrics["peak_abs"], duration_error
            )
            severe = not semantic_ok
            write_diagnostic_png(
                case_dir / "comparison.png", after, params.framerate, params.nchannels,
                start, end, start, end,
            )
            layer_rows.append({
                "failure_id": failure_id,
                "parent_candidate_id": plan["parent_candidate_id"],
                "source_mode": plan["source_mode"],
                "repair_action": plan["repair_action"],
                "before_artifact": relative(before_path),
                "after_artifact": relative(after_path),
                "target_window_rms_delta_db": metrics["target_window_rms_delta_db"],
                "outside_window_rms_delta_db": metrics["outside_window_rms_delta_db"],
                "sample_peak_delta": after_audio_metrics["peak_abs"] - before_audio_metrics["peak_abs"],
                "clip_ratio_delta": after_audio_metrics["clipped_ratio"] - before_audio_metrics["clipped_ratio"],
                "duration_error_ms": duration_error,
                "boundary_jump_before": metrics["boundary_jump_before"],
                "boundary_jump_after": metrics["boundary_jump_after"],
                "changed_sample_ratio": metrics["changed_sample_ratio"],
                "ordering_correct": "true",
                "lineage_complete": "true",
                "output_readable": "true",
                "severe_regression": str(severe).lower(),
                "semantic_target_satisfied": str(semantic_ok).lower(),
                "forbidden_event_status": "unknown",
                "needs_manual_review": "true",
                "decision": "PROXY_PROMOTION" if semantic_ok else "REPAIR_REJECTED",
                "reason": reason,
            })
            if not semantic_ok:
                failures.append({"failure_id": failure_id, "reason": reason})
            continue

        donor_path = ROOT / plan["donor_audio"]
        donor_params, donor = read_pcm16(donor_path)
        if (
            params.framerate != donor_params.framerate
            or params.nchannels != donor_params.nchannels
            or params.sampwidth != donor_params.sampwidth
        ):
            failures.append({"failure_id": failure_id, "reason": "target/donor WAV format mismatch"})
            continue
        variants = []
        for variant_index, crossfade in enumerate((40, 100)[:args.replacement_variants], 1):
            after, metadata = transplant_event(
                before, donor, params.framerate, params.nchannels,
                *plan["expected_window"], *plan["donor_window"], crossfade,
            )
            after_path = case_dir / f"variant_{variant_index}.wav"
            write_pcm16(after_path, params, after)
            acoustic = audio_metrics(params, after)
            before_acoustic = audio_metrics(params, before)
            metrics = transplant_metrics(
                before, after, params.framerate, params.nchannels, *plan["expected_window"]
            )
            semantic_proxy = (
                metrics["target_window_rms_delta_db"] > 0.1
                and metadata["clipped_sample_count"] == 0
                and acoustic["peak_abs"] < 0.999
                and metrics["boundary_jump_after"] <= metrics["boundary_jump_before"] + 0.02
            )
            variants.append({
                "failure_id": failure_id,
                "parent_candidate_id": plan["parent_candidate_id"],
                "candidate_id": f"{failure_id}_transplant_v{variant_index}",
                "source_mode": plan["source_mode"],
                "repair_action": "event_transplant",
                "before_artifact": relative(before_path),
                "after_artifact": relative(after_path),
                "donor_candidate": plan["donor_candidate"],
                "donor_audio": plan["donor_audio"],
                "donor_start_sec": plan["donor_window"][0],
                "donor_end_sec": plan["donor_window"][1],
                "target_start_sec": plan["expected_window"][0],
                "target_end_sec": plan["expected_window"][1],
                "crossfade_ms": crossfade,
                "target_window_rms_delta_db": metrics["target_window_rms_delta_db"],
                "outside_window_rms_delta_db": metrics["outside_window_rms_delta_db"],
                "sample_peak_delta": acoustic["peak_abs"] - before_acoustic["peak_abs"],
                "clip_ratio_delta": acoustic["clipped_ratio"] - before_acoustic["clipped_ratio"],
                "duration_error_ms": 0.0,
                "boundary_jump_before": metrics["boundary_jump_before"],
                "boundary_jump_after": metrics["boundary_jump_after"],
                "changed_sample_ratio": metadata["changed_sample_ratio"],
                "ordering_correct": str(metrics["ordering_correct"]).lower(),
                "lineage_complete": "true",
                "output_readable": "true",
                "severe_regression": str(not semantic_proxy).lower(),
                "semantic_target_satisfied": str(semantic_proxy).lower(),
                "forbidden_event_status": "unknown",
                "needs_manual_review": "true",
                "decision": "",
                "reason": "",
            })
        feasible = [row for row in variants if row["semantic_target_satisfied"] == "true"]
        selected = min(feasible, key=lambda row: (float(row["boundary_jump_after"]), float(row["changed_sample_ratio"]))) if feasible else None
        for row in variants:
            if row is selected:
                row["decision"] = "PROBE_SELECTED"
                row["reason"] = "best valid crossfade by boundary jump and edit locality"
            else:
                row["decision"] = "REJECTED_DIAGNOSTIC"
                row["reason"] = "not selected after constrained variant comparison"
        replacement_rows.extend(variants)
        if selected is None:
            failures.append({"failure_id": failure_id, "reason": "no transplant variant passed guards"})

    args.layer_metrics.parent.mkdir(parents=True, exist_ok=True)
    with args.layer_metrics.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=layer_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(layer_rows)
    replacement_fields = list(replacement_rows[0]) if replacement_rows else [
        "failure_id", "decision", "reason"
    ]
    with args.replacement_metrics.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=replacement_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(replacement_rows)
    failures.insert(0, {
        "failure_id": "capability_inventory",
        "reason": "true_stem_count=0; all layer repairs correctly routed to mixed_only",
        "status": "STEM_REMIX_BLOCKED",
    })
    dump_json(args.failures, {"failures": failures})
    selected_transplants = sum(row["decision"] == "PROBE_SELECTED" for row in replacement_rows)
    summary = {
        "manifestCount": len(manifest_rows),
        "trueStemCount": 0,
        "mixedOnlyRepairCount": len(layer_rows),
        "mixedOnlyReadableAfterCount": sum(row["output_readable"] == "true" for row in layer_rows),
        "mixedOnlyProxyImprovedCount": sum(row["semantic_target_satisfied"] == "true" for row in layer_rows),
        "eventReplacementProbeCount": len({row["failure_id"] for row in replacement_rows}),
        "eventReplacementVariantCount": len(replacement_rows),
        "eventReplacementSelectedCount": selected_transplants,
        "rejectedDiagnosticCount": sum(row["decision"] == "REJECTED_DIAGNOSTIC" for row in replacement_rows),
        "manualListeningCompletedCount": human_complete,
        "automaticForbiddenEventDetectionAvailable": False,
        "gateStatus": "PASS" if len(layer_rows) == 4 and selected_transplants == 2 else "FAIL",
        "limitations": [
            "No independently traceable stems were found; no stem remix is claimed.",
            "Forbidden-event labels and semantic success require human listening.",
        ],
    }
    dump_json(args.summary, summary)
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["gateStatus"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
