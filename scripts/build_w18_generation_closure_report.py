#!/usr/bin/env python3
import json
from pathlib import Path

FULL = Path("reports/w18_full_30job_generation_summary_20260706.json")
ANALYSIS = Path("reports/w18_full_30job_ablation_analysis_20260706.json")
REPAIR = Path("reports/w18_clipping_repair_report_20260706.json")
LISTEN = Path("reports/w18_repair_aware_listening_pack_20260706.json")
OUT = Path("reports/w18_generation_closure_report_20260706.json")

full = json.loads(FULL.read_text(encoding="utf-8"))
analysis = json.loads(ANALYSIS.read_text(encoding="utf-8"))
repair = json.loads(REPAIR.read_text(encoding="utf-8"))
listen = json.loads(LISTEN.read_text(encoding="utf-8"))

closure = {
    "date": "2026-07-06",
    "scope": "w18_generation_closure",
    "status": "closed_for_generation_phase" if (
        full.get("generated_count") == 30
        and analysis.get("generated_count") == 30
        and repair.get("target_count") == repair.get("repaired_count")
        and listen.get("playlist_item_count") == 30
    ) else "review_required",
    "completed": {
        "compiled_prompt_jobs": 30,
        "generated_audio_jobs": full.get("generated_count"),
        "case_count": full.get("case_count"),
        "variant_counts": full.get("variant_counts"),
        "analysis_status": analysis.get("status"),
        "repair_status": repair.get("status"),
        "listening_pack_status": listen.get("status"),
        "playlist_item_count": listen.get("playlist_item_count"),
    },
    "key_findings": [
        "MMAudio small_44k offline runtime is usable through conda env mmaudio-mini.",
        "All 6 cases and 5 prompt variants generated local audio artifacts.",
        "Subway dss_event_timeline and dss_layer_avoid required peak repair due to clipping review.",
        "Repair was non-destructive and generated peak-normalized local candidates.",
        "Current metrics are acoustic sanity metrics only, not semantic/synchrony superiority metrics.",
    ],
    "do_not_claim": [
        "DSS variants outperform naive baselines.",
        "Final production mix is ready.",
        "Human preference or semantic correctness is proven.",
        "Large-v2 MMAudio generation is complete.",
    ],
    "next_phase": [
        "Human listening review using the repair-aware HTML/M3U pack.",
        "Event-level timing metrics for high-transient cases such as glass and kitchen.",
        "Reranking policy using listening score + clipping/sanity gates.",
        "Java/Cloud handoff of candidate manifest after listening gate is defined.",
    ],
    "source_reports": {
        "full_generation": str(FULL),
        "analysis": str(ANALYSIS),
        "repair": str(REPAIR),
        "listening_pack": str(LISTEN),
    },
}

OUT.write_text(json.dumps(closure, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(closure, ensure_ascii=False, indent=2))
raise SystemExit(0 if closure["status"] == "closed_for_generation_phase" else 2)
