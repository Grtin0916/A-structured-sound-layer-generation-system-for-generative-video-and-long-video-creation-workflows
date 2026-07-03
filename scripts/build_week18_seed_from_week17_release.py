from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(".").resolve()

RELEASE_VERIFY = ROOT / "reports/week17_demo_release_verify_20260703.json"
RELEASE_MANIFEST = ROOT / "reports/week17_demo_release_manifest_20260703.json"
CLAIM_CARD = ROOT / "reports/week17_demo_claim_boundary_card_20260703.json"
CANDIDATE_REGISTRY = ROOT / "reports/week17_true_aware_candidate_registry_20260702.csv"
TRUE_AWARE_WINNERS = ROOT / "reports/week17_true_aware_winners_20260701.json"
FALLBACK_RANKING = ROOT / "reports/week17_fallback_aware_ranking_20260701.csv"
RELEASE_ZIP = ROOT / "artifacts/demo/week17_true_aware_demo_release_20260703.zip"

OUT_SEED_JSON = ROOT / "reports/week18_seed_from_week17_demo_release_20260703.json"
OUT_CASES_CSV = ROOT / "reports/week18_seed_cases_20260703.csv"
OUT_REPAIR_CSV = ROOT / "reports/week18_seed_repair_targets_20260703.csv"
OUT_PROMPT_MD = ROOT / "reports/week18_prompt_compiler_seed_20260703.md"
OUT_INTERVIEW_MD = ROOT / "docs/demo/week17_demo_release_interview_compact_20260703.md"


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def guess_case_id(row: dict) -> str:
    for k in ["case_id", "caseId", "demo_case_id"]:
        if row.get(k):
            return row[k]
    for value in row.values():
        if isinstance(value, str) and "_001" in value:
            parts = value.split("/")
            for p in parts:
                if p.endswith("_001"):
                    return p
    return "unknown_case"


def main() -> int:
    verify = read_json(RELEASE_VERIFY)
    manifest = read_json(RELEASE_MANIFEST)
    claim = read_json(CLAIM_CARD)
    registry_rows = read_csv(CANDIDATE_REGISTRY)
    fallback_rows = read_csv(FALLBACK_RANKING)
    winners = read_json(TRUE_AWARE_WINNERS)

    checks = verify.get("checks", {})

    true_case = "glass_drop_room_001"
    true_wav = "artifacts/demo/week17_true_aware_demo_release/audio/glass_drop_room_001__mmaudio__true_replacement_v0.wav"

    # Try to recover case ids from prior ranking/registry. Fall back to known W17 case bank.
    case_ids = []
    for row in registry_rows + fallback_rows:
        cid = guess_case_id(row)
        if cid != "unknown_case" and cid not in case_ids:
            case_ids.append(cid)

    fallback_known = [
        "glass_drop_room_001",
        "street_rain_crosswalk_001",
        "subway_arrival_door_001",
        "kitchen_chop_sizzle_001",
        "forest_bird_branch_001",
        "robot_warehouse_pick_001",
    ]
    for cid in fallback_known:
        if cid not in case_ids:
            case_ids.append(cid)

    case_records = []
    for cid in case_ids[:12]:
        case_records.append({
            "case_id": cid,
            "week18_role": "true_anchor" if cid == true_case else "dss_prompt_compiler_seed",
            "source_week": "W17",
            "has_true_mmaudio": cid == true_case,
            "recommended_next_action": (
                "preserve as positive anchor; compare DSS prompt vs naive prompt"
                if cid == true_case
                else "generate DSS prompt variants and evaluate event coverage / forbidden leakage"
            ),
        })

    repair_targets = [
        {
            "case_id": true_case,
            "target_type": "positive_anchor",
            "failure_mode": "none_for_true_single; use as reference",
            "week18_action": "protect timing and claim boundary while expanding prompt compiler",
            "priority": "P0",
        }
    ]

    for cid in case_ids:
        if cid == true_case:
            continue
        repair_targets.append({
            "case_id": cid,
            "target_type": "candidate_or_fallback",
            "failure_mode": "needs DSS-aware regeneration or repair inspection",
            "week18_action": "run naive prompt vs DSS prompt; if onset/coverage fails, add to repair bank",
            "priority": "P1",
        })

    seed = {
        "seed_id": "week18_seed_from_week17_demo_release_20260703",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "READY_FOR_W18_DSS_PROMPT_COMPILER",
        "week17_release": {
            "verify_decision": verify.get("decision"),
            "release_zip": str(RELEASE_ZIP),
            "release_zip_exists": RELEASE_ZIP.exists(),
            "release_zip_size_bytes": RELEASE_ZIP.stat().st_size if RELEASE_ZIP.exists() else 0,
            "zip_valid": checks.get("zip_valid"),
            "zip_contains_index": checks.get("zip_contains_index"),
            "zip_contains_wav": checks.get("zip_contains_wav"),
            "safe_true_mmaudio_record_count": checks.get("safe_true_mmaudio_record_count"),
            "manifest_release_id": manifest.get("release_id"),
        },
        "claim_boundary": {
            "safeTrueMmaudioRecordCount": claim.get("safeTrueMmaudioRecordCount"),
            "trueMmaudioBatchSuccess": claim.get("trueMmaudioBatchSuccess"),
            "fullCandidateRankingAvailable": claim.get("fullCandidateRankingAvailable"),
            "productionSloVerified": claim.get("productionSloVerified"),
            "k6ThresholdPassVerified": claim.get("k6ThresholdPassVerified"),
            "liveGrafanaImportVerified": claim.get("liveGrafanaImportVerified"),
        },
        "w18_objectives": [
            "Build DSS v1 schema and prompt compiler.",
            "Compare naive prompt vs DSS prompt on W17 case bank.",
            "Use the single true MMAudio result as a positive anchor, not as batch-success evidence.",
            "Convert failed/fallback cases into repair targets instead of hiding them.",
            "Keep Java/Cloud fields ready for downstream task lifecycle and dashboard aggregation.",
        ],
        "case_records": case_records,
        "repair_targets": repair_targets,
        "source_artifacts": {
            "release_verify": str(RELEASE_VERIFY),
            "release_manifest": str(RELEASE_MANIFEST),
            "claim_card": str(CLAIM_CARD),
            "candidate_registry": str(CANDIDATE_REGISTRY),
            "true_aware_winners": str(TRUE_AWARE_WINNERS),
            "fallback_ranking": str(FALLBACK_RANKING),
        },
    }

    OUT_SEED_JSON.write_text(json.dumps(seed, ensure_ascii=False, indent=2), encoding="utf-8")

    with OUT_CASES_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "case_id",
            "week18_role",
            "source_week",
            "has_true_mmaudio",
            "recommended_next_action",
        ])
        writer.writeheader()
        writer.writerows(case_records)

    with OUT_REPAIR_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "case_id",
            "target_type",
            "failure_mode",
            "week18_action",
            "priority",
        ])
        writer.writeheader()
        writer.writerows(repair_targets)

    OUT_PROMPT_MD.write_text(f"""# W18 Prompt Compiler Seed

## Seed

`{OUT_SEED_JSON}`

## Core decision

W17 produced a claim-safe demo release candidate. W18 should not restart from generic prompts. It should use W17 case records as controlled inputs.

## Positive anchor

- Case: `{true_case}`
- Artifact: `{true_wav}`
- Meaning: one true MMAudio replacement is traceable.
- Boundary: this is not batch success.

## W18 experiment design

1. Build DSS v1 fields: scene, events, layer roles, avoid list, timing tolerance.
2. Generate naive prompt and DSS prompt for each case.
3. Evaluate event coverage, onset alignment proxy, forbidden leakage, loudness, silence ratio.
4. Promote successes into selector examples.
5. Convert failures into repair targets.

## Hard boundary

Do not claim:

- true MMAudio batch success
- full candidate ranking
- production SLO
- k6 threshold pass
- live Grafana import
""", encoding="utf-8")

    OUT_INTERVIEW_MD.write_text(f"""# Week17 Demo Release Interview Compact

## One-line story

I built a claim-safe demo release path for a Director-guided Video-to-Audio SoundLayer System: Mainbase packages the true-aware audio demo, Java exposes it as a handoff API, and Cloud turns it into a release gate with observability artifacts.

## What is real

- Mainbase release verify: `{verify.get("decision")}`
- Release ZIP valid: `{checks.get("zip_valid")}`
- WAV fallback present: `{checks.get("zip_contains_wav")}`
- Safe true MMAudio records: `{checks.get("safe_true_mmaudio_record_count")}`
- Java handoff endpoint: `/api/week17/demo-release-handoff`
- Cloud gate: release-ready, dashboard-ready, Prometheus-sample-ready

## What I refuse to overclaim

- true MMAudio batch success: `{claim.get("trueMmaudioBatchSuccess")}`
- full candidate ranking: `{claim.get("fullCandidateRankingAvailable")}`
- production SLO: `{claim.get("productionSloVerified")}`
- k6 threshold pass: `{claim.get("k6ThresholdPassVerified")}`
- live Grafana import: `{claim.get("liveGrafanaImportVerified")}`

## Why this matters

The project is not just a model call. It has controllable inputs, traceable artifacts, platform handoff, cloud gate, and explicit failure boundaries. That makes it easier to defend in an interview than a black-box generated audio sample.

## W18 bridge

Next week should use this seed to implement DSS prompt compiler and compare naive prompt vs DSS-controlled prompt.
""", encoding="utf-8")

    print(json.dumps({
        "decision": seed["decision"],
        "case_count": len(case_records),
        "repair_target_count": len(repair_targets),
        "safeTrueMmaudioRecordCount": claim.get("safeTrueMmaudioRecordCount"),
        "zip_valid": checks.get("zip_valid"),
        "out_seed": str(OUT_SEED_JSON),
        "out_cases": str(OUT_CASES_CSV),
        "out_repair": str(OUT_REPAIR_CSV),
        "out_prompt_seed": str(OUT_PROMPT_MD),
        "out_interview_compact": str(OUT_INTERVIEW_MD),
    }, ensure_ascii=False, indent=2))

    return 0 if verify.get("decision") == "PASS" and checks.get("safe_true_mmaudio_record_count", 0) >= 1 else 2


if __name__ == "__main__":
    raise SystemExit(main())