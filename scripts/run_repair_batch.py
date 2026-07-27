#!/usr/bin/env python3
"""Run constrained minimal-edit repair search over the W19 execution manifest."""

from __future__ import annotations

from array import array
import argparse
import csv
import json
from pathlib import Path
import random
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from repair_audio_utils import read_pcm16, write_diagnostic_png, write_pcm16
from soundlayer.repair.repair_metrics import (
    action_compatibility,
    detect_subtype,
    diagnose_audio,
    region_rms,
)
from soundlayer.repair.repair_policy import route_action
from soundlayer.repair.repair_search import (
    edit_cost,
    enumerate_variants,
    select_minimal_valid_edit,
)
from soundlayer.repair.signal_repair import (
    adaptive_headroom,
    conservative_micro_declip,
    silence_aware_trim,
    smooth_region_gain,
)


FIELDS = [
    "failure_id", "case_id", "candidate", "failure_type", "detected_subtype",
    "planned_action", "selected_action", "capability_status",
    "action_metric_compatible", "compatibility_reason", "status",
    "target_metric", "target_delta", "target_improved", "guard_status",
    "severe_regression", "edit_cost", "changed_sample_ratio", "output_readable",
    "lineage_complete", "fallback_count", "repair_kind",
    "promotion_recommendation", "manual_review_required", "before_audio",
    "selected_after", "rejected_diagnostic_audio", "reason",
]


def _json_dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _apply(
    samples: array, params, variant: dict, plan: dict
) -> tuple[array, dict]:
    action = variant["action"]
    if action == "attenuate_limit":
        return adaptive_headroom(samples, float(variant["ceiling"]))
    if action == "micro_declip":
        return conservative_micro_declip(
            samples, float(variant["threshold"]), int(variant["max_run"])
        )
    if action == "trim":
        return silence_aware_trim(
            samples, params.framerate, params.nchannels,
            float(variant["silence_threshold"]), int(variant["padding_ms"])
        )
    window = (
        plan.get("event") or {}
        if action == "event_local_gain"
        else plan.get("window") or plan.get("event") or {}
    )
    start = float(window.get("start_sec", 0))
    end = float(window.get("end_sec", len(samples) / params.framerate / params.nchannels))
    if action in {"event_local_gain", "mixed_region_attenuation"}:
        return smooth_region_gain(
            samples, params.framerate, params.nchannels, start, end,
            float(variant["gain"]), int(variant["fade_ms"])
        )
    raise ValueError(f"unsupported action: {action}")


def _target(
    subtype: str, before: dict, after: dict, plan: dict, params, before_audio: array,
    after_audio: array
) -> tuple[str, float, bool]:
    if subtype == "peak_near_ceiling":
        delta = after["sample_peak_abs"] - before["sample_peak_abs"]
        return "sample_peak_abs", delta, delta < -1.0e-6
    if subtype == "short_flat_top":
        delta = after["flat_top_max_run"] - before["flat_top_max_run"]
        return "flat_top_max_run", float(delta), delta < 0
    if subtype == "leading_trailing_silence":
        before_edge = before["leading_silence_ms"] + before["trailing_silence_ms"]
        after_edge = after["leading_silence_ms"] + after["trailing_silence_ms"]
        return "edge_silence_ms", after_edge - before_edge, after_edge < before_edge
    window = plan.get("window") or plan.get("event") or {}
    start, end = float(window.get("start_sec", 0)), float(window.get("end_sec", 0))
    if subtype == "mixed_region_only":
        old = region_rms(before_audio, params.framerate, params.nchannels, start, end)
        new = region_rms(after_audio, params.framerate, params.nchannels, start, end)
        return "region_rms", new - old, new < old
    delta = after["event_window_rms"] - before["event_window_rms"]
    return "event_window_rms", delta, delta > 1.0e-6


def _regression(subtype: str, before: dict, after: dict, changed_ratio: float) -> bool:
    if subtype == "leading_trailing_silence":
        return after["event_window_rms"] < before["event_window_rms"] * 0.95
    if subtype == "weak_event_window":
        return after["sample_peak_abs"] > 0.999 or changed_ratio > 0.40
    if subtype == "mixed_region_only":
        return after["sample_peak_abs"] > before["sample_peak_abs"] + 1.0e-6
    return (
        abs(after["duration_sec"] - before["duration_sec"]) > 1.0e-6
        or after["rms_dbfs"] < before["rms_dbfs"] - 1.0
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--search-trace", type=Path, required=True)
    parser.add_argument("--failures", type=Path, required=True)
    parser.add_argument("--promotion", type=Path, default=Path("reports/repair_promotion_candidates_20260714.csv"))
    parser.add_argument("--listening-sheet", type=Path, default=Path("reports/repair_listening_sheet_20260714.csv"))
    parser.add_argument("--max-variants-per-plan", type=int, default=5)
    parser.add_argument("--selection-policy", choices=["minimal_valid_edit"], default="minimal_valid_edit")
    args = parser.parse_args()

    plans = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    rows, traces, failures = [], [], []
    for plan in plans:
        failure_id = plan["failure_id"]
        failure_dir = args.output_dir / failure_id
        before_path, after_path = failure_dir / "before.wav", failure_dir / "after.wav"
        try:
            failure_dir.mkdir(parents=True, exist_ok=True)
            source = ROOT / plan["source_audio"]
            shutil.copy2(source, before_path)
            params, samples = read_pcm16(before_path)
            before = diagnose_audio(
                samples, params.framerate, params.nchannels, plan.get("event")
            )
            subtype = detect_subtype(plan["failure_type"], before)
            routed_action, route_reason = route_action(
                plan["failure_type"], subtype, plan["action"]
            )
            compatible, compatibility_reason = action_compatibility(
                plan["failure_type"], subtype, routed_action
            )
            diagnosis = dict(before, detected_subtype=subtype)
            variants = enumerate_variants({**plan, "action": routed_action}, diagnosis)
            variants = variants[:args.max_variants_per_plan] if compatible else []
            candidates = []
            for variant_index, variant in enumerate(variants):
                repaired, meta = _apply(samples, params, variant, plan)
                after = diagnose_audio(
                    repaired, params.framerate, params.nchannels, plan.get("event")
                )
                metric, delta, improved = _target(
                    subtype, before, after, plan, params, samples, repaired
                )
                changed_ratio, mean_delta = edit_cost(samples, repaired)
                regression = _regression(subtype, before, after, changed_ratio)
                candidate = {
                    "failure_id": failure_id,
                    "variant_index": variant_index,
                    "variant": variant,
                    "target_metric": metric,
                    "target_delta": delta,
                    "target_improved": improved,
                    "severe_regression": regression,
                    "output_readable": True,
                    "lineage_complete": bool(plan.get("source_audio") and plan.get("lineage")),
                    "action_metric_compatible": compatible,
                    "edit_cost": mean_delta + changed_ratio,
                    "changed_sample_ratio": changed_ratio,
                    "after_metrics": after,
                    "meta": meta,
                    "_samples": repaired,
                }
                candidates.append(candidate)
                traces.append({k: v for k, v in candidate.items() if k != "_samples"})
            selected = select_minimal_valid_edit(candidates)
            rejected_diagnostic = ""
            if selected:
                write_pcm16(after_path, params, selected["_samples"])
                event = plan.get("event") or {}
                window = plan.get("window") or event
                write_diagnostic_png(
                    failure_dir / "comparison.png", selected["_samples"],
                    params.framerate, params.nchannels,
                    float(event.get("start_sec", 0)), float(event.get("end_sec", before["duration_sec"])),
                    float(window.get("start_sec", 0)), float(window.get("end_sec", before["duration_sec"])),
                )
                status = "REPAIR_SELECTED"
                recommendation = "PROMOTION_RECOMMENDED"
                reason = route_reason
                capability = "SUPPORTED"
            else:
                if subtype == "long_flat_top":
                    diagnostic_audio, _ = adaptive_headroom(samples, 0.95)
                    diagnostic_path = failure_dir / "rejected_diagnostic.wav"
                    write_pcm16(diagnostic_path, params, diagnostic_audio)
                    rejected_diagnostic = diagnostic_path.as_posix()
                status = "REPAIR_BLOCKED" if not compatible else "REPAIR_REJECTED"
                recommendation = "MANUAL_REVIEW"
                reason = compatibility_reason if not compatible else "no variant passed target and guards"
                capability = "BLOCKED" if not compatible else "SUPPORTED_NO_FEASIBLE_VARIANT"
                failures.append({
                    "failure_id": failure_id, "status": status, "reason": reason,
                    "detected_subtype": subtype,
                })
            chosen = selected or {}
            rows.append({
                "failure_id": failure_id, "case_id": plan["case_id"],
                "candidate": plan["candidate"], "failure_type": plan["failure_type"],
                "detected_subtype": subtype, "planned_action": plan["action"],
                "selected_action": (chosen.get("variant") or {}).get("action", routed_action),
                "capability_status": capability,
                "action_metric_compatible": str(compatible).lower(),
                "compatibility_reason": compatibility_reason, "status": status,
                "target_metric": chosen.get("target_metric", plan["target_metric"]),
                "target_delta": chosen.get("target_delta", ""),
                "target_improved": str(bool(chosen.get("target_improved"))).lower(),
                "guard_status": "PASS" if selected else "NOT_PASSED",
                "severe_regression": str(bool(chosen.get("severe_regression"))).lower(),
                "edit_cost": chosen.get("edit_cost", ""),
                "changed_sample_ratio": chosen.get("changed_sample_ratio", ""),
                "output_readable": str(after_path.is_file()).lower(),
                "lineage_complete": str(bool(plan.get("lineage"))).lower(),
                "fallback_count": 0,
                "repair_kind": (chosen.get("meta") or {}).get("repair_kind", ""),
                "promotion_recommendation": recommendation,
                "manual_review_required": "true",
                "before_audio": before_path.as_posix(),
                "selected_after": after_path.as_posix() if selected else "",
                "rejected_diagnostic_audio": rejected_diagnostic,
                "reason": reason,
            })
        except Exception as exc:
            failures.append({"failure_id": failure_id, "status": "EXECUTION_FAILED", "reason": f"{type(exc).__name__}: {exc}"})
            rows.append({
                **{field: "" for field in FIELDS}, "failure_id": failure_id,
                "case_id": plan.get("case_id", ""), "candidate": plan.get("candidate", ""),
                "failure_type": plan.get("failure_type", ""), "status": "EXECUTION_FAILED",
                "severe_regression": "false", "output_readable": "false",
                "lineage_complete": "false", "reason": failures[-1]["reason"],
            })

    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    with args.metrics.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    args.search_trace.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in traces), encoding="utf-8"
    )
    _json_dump(args.failures, {"failures": failures})
    promoted = [row for row in rows if row["promotion_recommendation"] == "PROMOTION_RECOMMENDED"]
    with args.promotion.open("w", newline="", encoding="utf-8") as handle:
        fields = ["failure_id", "candidate", "selected_after", "repair_kind", "target_delta", "guard_status", "edit_cost", "promotion_recommendation", "manual_review_required", "reason"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    randomizer = random.Random(20260714)
    review_pool = promoted[:3] + [row for row in rows if row["status"] != "REPAIR_SELECTED"][:3]
    if len(review_pool) < 6:
        review_pool += promoted[3:3 + 6 - len(review_pool)]
    with args.listening_sheet.open("w", newline="", encoding="utf-8") as handle:
        fields = ["pair_id", "failure_id", "audio_a", "audio_b", "preference", "reason", "confidence", "audible_artifact", "review_status", "mapping_hidden"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        listening_key = []
        for index, row in enumerate(review_pool[:6], 1):
            sources = [
                ("before", row["before_audio"]),
                ("after_or_rejected", row["selected_after"] or row["rejected_diagnostic_audio"]),
            ]
            randomizer.shuffle(sources)
            pair_dir = ROOT / "artifacts" / "repair" / "listening_20260714" / f"pair_{index:02d}"
            pair_dir.mkdir(parents=True, exist_ok=True)
            audio_a, audio_b = pair_dir / "a.wav", pair_dir / "b.wav"
            shutil.copy2(sources[0][1], audio_a)
            shutil.copy2(sources[1][1], audio_b)
            writer.writerow({
                "pair_id": f"ab_{index:02d}", "failure_id": row["failure_id"],
                "audio_a": audio_a.relative_to(ROOT).as_posix(),
                "audio_b": audio_b.relative_to(ROOT).as_posix(), "preference": "",
                "reason": "", "confidence": "", "audible_artifact": "",
                "review_status": "PENDING_HUMAN_LISTENING",
                "mapping_hidden": "true",
            })
            listening_key.append({
                "pair_id": f"ab_{index:02d}",
                "a_role": sources[0][0],
                "b_role": sources[1][0],
                "failure_id": row["failure_id"],
            })
    _json_dump(
        ROOT / "reports" / "repair_listening_key_20260714.json",
        {"doNotOpenBeforeReview": True, "pairs": listening_key},
    )
    selected_count = sum(row["status"] == "REPAIR_SELECTED" for row in rows)
    severe_count = sum(row["severe_regression"] == "true" for row in rows)
    summary = {
        "manifestCount": len(plans), "statusCount": len(rows),
        "readableBeforeCount": sum(Path(row["before_audio"]).is_file() for row in rows if row["before_audio"]),
        "readableAfterCount": sum(Path(row["selected_after"]).is_file() for row in rows if row["selected_after"]),
        "targetMetricImprovedCount": sum(row["target_improved"] == "true" for row in rows),
        "selectedCount": selected_count, "promotionCandidateCount": len(promoted),
        "blockedOrRejectedCount": sum(row["status"] != "REPAIR_SELECTED" for row in rows),
        "severeRegressionCount": severe_count, "searchEvaluationCount": len(traces),
        "listeningPairCount": min(len(review_pool), 6),
        "manualListeningCompletedCount": 0,
        "gateStatus": "PASS" if len(rows) == 13 and selected_count >= 6 and severe_count == 0 else "FAIL",
        "limitations": [
            "Human A/B preferences remain pending and are not inferred from proxy metrics.",
            "Micro-declip uses a standard-library linear probe because SciPy is unavailable; it does not claim waveform recovery.",
        ],
    }
    _json_dump(args.summary, summary)
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["gateStatus"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
