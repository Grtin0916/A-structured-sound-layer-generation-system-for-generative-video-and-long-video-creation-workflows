#!/usr/bin/env python3
import csv
import json
from pathlib import Path

FULL = Path("reports/w18_full_30job_generation_summary_20260706.json")
ANALYSIS = Path("reports/w18_full_30job_ablation_analysis_20260706.json")
REPAIR = Path("reports/w18_clipping_repair_report_20260706.json")
LISTEN = Path("reports/w18_repair_aware_listening_pack_20260706.json")
CLOSURE = Path("reports/w18_generation_closure_report_20260706.json")
MATRIX = Path("reports/w18_full_30job_generation_matrix_20260706.csv")

OUT_JSON = Path("reports/w18_java_cloud_handoff_manifest_20260706.json")
OUT_SCHEMA = Path("reports/w18_java_cloud_handoff_schema_20260706.json")
OUT_CSV = Path("reports/w18_java_cloud_handoff_cases_20260706.csv")

full = json.loads(FULL.read_text(encoding="utf-8"))
analysis = json.loads(ANALYSIS.read_text(encoding="utf-8"))
repair = json.loads(REPAIR.read_text(encoding="utf-8"))
listen = json.loads(LISTEN.read_text(encoding="utf-8"))
closure = json.loads(CLOSURE.read_text(encoding="utf-8"))

matrix_rows = list(csv.DictReader(MATRIX.open(encoding="utf-8")))

repair_by_job = {
    row["job_id"]: row
    for row in repair.get("rows", [])
    if row.get("status") == "success"
}

matrix_by_case = {}
for row in matrix_rows:
    matrix_by_case.setdefault(row["case_id"], []).append(row)

case_candidates = {
    row["case_id"]: row
    for row in analysis.get("case_summary", [])
}

handoff_cases = []
flat_rows = []

for case in listen.get("cases", []):
    case_id = case["case_id"]
    analysis_case = case_candidates.get(case_id, {})

    variants = []
    for variant in case.get("variants", []):
        job_id = variant["job_id"]
        repair_row = repair_by_job.get(job_id)

        selected_audio = variant["selected_wav_path"]
        artifact_status = "repaired_candidate" if repair_row else "generated_candidate"

        item = {
            "job_id": job_id,
            "variant": variant["variant"],
            "selected_audio_path": selected_audio,
            "original_audio_path": variant["original_wav_path"],
            "artifact_status": artifact_status,
            "is_repaired": bool(repair_row),
            "review_reason": variant.get("review_reason", ""),
            "basic_metrics": {
                "rms_dbfs": variant.get("rms_dbfs"),
                "peak_dbfs": variant.get("peak_dbfs"),
                "clipped_ratio": variant.get("clipped_ratio"),
                "status": variant.get("status"),
            },
        }
        variants.append(item)

    default_variant = analysis_case.get("candidate_variant_for_listening")
    default_item = next((v for v in variants if v["variant"] == default_variant), None)

    handoff_case = {
        "case_id": case_id,
        "default_candidate_variant": default_variant,
        "default_candidate_audio_path": default_item["selected_audio_path"] if default_item else analysis_case.get("candidate_audio_path"),
        "review_variants": analysis_case.get("review_variants", []),
        "variant_count": len(variants),
        "generated_count": len([v for v in variants if v["selected_audio_path"]]),
        "variants": variants,
    }
    handoff_cases.append(handoff_case)

    flat_rows.append({
        "case_id": case_id,
        "default_candidate_variant": handoff_case["default_candidate_variant"],
        "default_candidate_audio_path": handoff_case["default_candidate_audio_path"],
        "review_variants": "|".join(handoff_case["review_variants"]),
        "variant_count": handoff_case["variant_count"],
        "generated_count": handoff_case["generated_count"],
    })

manifest = {
    "date": "2026-07-06",
    "scope": "w18_java_cloud_handoff",
    "status": "ready_for_java_contract" if (
        full.get("generated_count") == 30
        and listen.get("playlist_item_count") == 30
        and closure.get("status") == "closed_for_generation_phase"
        and len(handoff_cases) == 6
    ) else "review_required",
    "source_commits": {
        "mainbase_latest_expected": "fd1ded9",
    },
    "summary": {
        "case_count": len(handoff_cases),
        "job_count": full.get("job_count"),
        "generated_count": full.get("generated_count"),
        "playlist_item_count": listen.get("playlist_item_count"),
        "repair_applied_count": listen.get("repair_applied_count"),
        "variant_counts": full.get("variant_counts"),
    },
    "contract_version": "w18-generation-handoff-v1",
    "consumer_expectation": {
        "java": [
            "Read this manifest as the W18 result-card source.",
            "Expose case list, default candidate, variants, review flags, and artifact status.",
            "Do not serve audio bytes from Git; audio paths are local artifact references unless mounted.",
        ],
        "cloud": [
            "Use Java result-card API as the deployment object.",
            "SLO/dashboard should track manifest availability, case_count, generated_count, and repair_applied_count.",
        ],
    },
    "claim_boundary": [
        "Generation coverage is complete for 6 cases x 5 variants.",
        "This handoff does not claim DSS superiority.",
        "This handoff does not claim final production mix readiness.",
        "Human listening and reranking remain next-phase tasks.",
    ],
    "cases": handoff_cases,
    "source_reports": {
        "full_generation": str(FULL),
        "analysis": str(ANALYSIS),
        "repair": str(REPAIR),
        "listening_pack": str(LISTEN),
        "closure": str(CLOSURE),
    },
}

schema = {
    "contract_version": "w18-generation-handoff-v1",
    "required_top_level_keys": [
        "date",
        "scope",
        "status",
        "summary",
        "contract_version",
        "consumer_expectation",
        "claim_boundary",
        "cases",
        "source_reports",
    ],
    "case_required_keys": [
        "case_id",
        "default_candidate_variant",
        "default_candidate_audio_path",
        "review_variants",
        "variant_count",
        "generated_count",
        "variants",
    ],
    "variant_required_keys": [
        "job_id",
        "variant",
        "selected_audio_path",
        "original_audio_path",
        "artifact_status",
        "is_repaired",
        "review_reason",
        "basic_metrics",
    ],
    "valid_status": [
        "ready_for_java_contract",
        "review_required",
    ],
    "valid_artifact_status": [
        "generated_candidate",
        "repaired_candidate",
    ],
}

OUT_JSON.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
OUT_SCHEMA.write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")

with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()))
    writer.writeheader()
    writer.writerows(flat_rows)

print(json.dumps({
    "status": manifest["status"],
    "case_count": manifest["summary"]["case_count"],
    "generated_count": manifest["summary"]["generated_count"],
    "playlist_item_count": manifest["summary"]["playlist_item_count"],
    "repair_applied_count": manifest["summary"]["repair_applied_count"],
    "contract_version": manifest["contract_version"],
    "outputs": [str(OUT_JSON), str(OUT_SCHEMA), str(OUT_CSV)],
}, ensure_ascii=False, indent=2))

raise SystemExit(0 if manifest["status"] == "ready_for_java_contract" else 2)
