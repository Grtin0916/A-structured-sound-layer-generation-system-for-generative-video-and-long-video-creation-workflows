#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        return list(csv.DictReader(f))


def exists_info(path_str: str) -> Dict[str, Any]:
    p = Path(path_str)
    return {
        "path": path_str,
        "exists": p.exists(),
        "size": p.stat().st_size if p.exists() else 0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selector-v2", required=True)
    ap.add_argument("--failure-bank", required=True)
    ap.add_argument("--repair-probe", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-summary", required=True)
    args = ap.parse_args()

    selector_rows = read_csv(Path(args.selector_v2))
    failure_rows = read_csv(Path(args.failure_bank))
    repair_rows = read_csv(Path(args.repair_probe))

    winners = [
        {
            "case_id": r.get("case_id"),
            "candidate_key": r.get("candidate_key"),
            "variant": r.get("variant"),
            "selector_v2_score": r.get("selector_v2_score"),
            "metric_confidence": r.get("metric_confidence"),
            "repairability": r.get("repairability"),
            "audio_path": r.get("audio_path"),
        }
        for r in selector_rows
        if r.get("selector_v2_decision") == "winner"
    ]

    repaired = []
    for r in repair_rows:
        repaired.append({
            "probe_id": r.get("probe_id"),
            "failure_id": r.get("failure_id"),
            "case_id": r.get("case_id"),
            "variant": r.get("variant"),
            "failure_category": r.get("failure_category"),
            "repair_action": r.get("repair_action"),
            "proxy_improved": r.get("proxy_improved"),
            "improve_reason": r.get("improve_reason"),
            "before_audio": exists_info(r.get("before_audio_path", "")),
            "after_audio": exists_info(r.get("after_audio_path", "")),
            "before_after_plot": exists_info(r.get("plot_path", "")),
            "before_peak_abs": r.get("before_peak_abs"),
            "after_peak_abs": r.get("after_peak_abs"),
            "before_clip_ratio": r.get("before_clip_ratio"),
            "after_clip_ratio": r.get("after_clip_ratio"),
            "before_rms_mean": r.get("before_rms_mean"),
            "after_rms_mean": r.get("after_rms_mean"),
            "before_silence_ratio": r.get("before_silence_ratio"),
            "after_silence_ratio": r.get("after_silence_ratio"),
        })

    failure_category_counts: Dict[str, int] = {}
    for r in failure_rows:
        cat = r.get("failure_category", "unknown")
        failure_category_counts[cat] = failure_category_counts.get(cat, 0) + 1

    action_counts: Dict[str, int] = {}
    for r in repair_rows:
        act = r.get("repair_action", "unknown")
        action_counts[act] = action_counts.get(act, 0) + 1

    missing_assets = []
    for item in repaired:
        for key in ["before_audio", "after_audio", "before_after_plot"]:
            asset = item[key]
            if not asset["exists"] or asset["size"] <= 0:
                missing_assets.append({
                    "probe_id": item["probe_id"],
                    "asset_type": key,
                    "path": asset["path"],
                    "exists": asset["exists"],
                    "size": asset["size"],
                })

    contract = {
        "contract_name": "week18_selector_repair_handoff",
        "contract_version": "2026-07-08.v1",
        "producer": "mainbase",
        "intended_consumers": ["java", "cloud"],
        "boundary": (
            "Selector v2 and micro repair probe are proxy-backed engineering signals. "
            "They are not subjective listening labels, not production SLO, and not a full repair engine."
        ),
        "inputs": {
            "selector_v2": args.selector_v2,
            "failure_bank": args.failure_bank,
            "repair_probe": args.repair_probe,
        },
        "summary": {
            "winner_count": len(winners),
            "failure_count": len(failure_rows),
            "repair_probe_count": len(repaired),
            "proxy_improved_count": sum(1 for r in repaired if r.get("proxy_improved") == "true"),
            "failure_category_counts": failure_category_counts,
            "repair_action_counts": action_counts,
            "missing_asset_count": len(missing_assets),
        },
        "winners": winners,
        "repaired_candidates": repaired,
        "missing_assets": missing_assets,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(contract, indent=2, ensure_ascii=False), encoding="utf-8")

    dod = {
        "winner_count_eq_6": len(winners) == 6,
        "repair_probe_count_eq_6": len(repaired) == 6,
        "proxy_improved_count_ge_2": contract["summary"]["proxy_improved_count"] >= 2,
        "missing_asset_count_eq_0": len(missing_assets) == 0,
        "has_boundary": bool(contract["boundary"]),
    }

    summary = {
        "task": "export_w18_repair_handoff_contract",
        "out_json": args.out_json,
        "winner_count": len(winners),
        "failure_count": len(failure_rows),
        "repair_probe_count": len(repaired),
        "proxy_improved_count": contract["summary"]["proxy_improved_count"],
        "missing_asset_count": len(missing_assets),
        "failure_category_counts": failure_category_counts,
        "repair_action_counts": action_counts,
        "dod": dod,
    }

    Path(args.out_summary).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
