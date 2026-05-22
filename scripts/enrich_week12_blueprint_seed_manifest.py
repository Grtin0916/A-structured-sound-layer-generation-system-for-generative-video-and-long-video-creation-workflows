#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path.cwd()

IN_JSON = ROOT / "artifacts/manifests/week12_blueprint_seed_manifest.json"
OUT_JSON = ROOT / "artifacts/manifests/week12_blueprint_seed_manifest.json"
OUT_CSV = ROOT / "artifacts/manifests/week12_blueprint_seed_manifest.csv"
LOG_DIR = ROOT / "artifacts/logs"

MAINBASE_EVAL_ARTIFACT = "artifacts/evals/week11_eval_v0.json"
MAINBASE_EVAL_METRICS = "artifacts/evals/week11_eval_v0_metrics.json"
MAINBASE_E2E_INDEX = "artifacts/manifests/week11_e2e_demo_index.json"
MAINBASE_HANDOFF = "artifacts/manifests/week11_to_w12_handoff_manifest.json"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def bounded_float(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
    except Exception:
        return default
    return max(0.0, min(1.0, x))


def compute_proxy_quality(seed: dict[str, Any]) -> float:
    scores = seed.setdefault("scores_proxy", {})
    semantic = bounded_float(scores.get("semantic"), 0.0)
    temporal = bounded_float(scores.get("temporal"), 0.0)
    current_quality = bounded_float(scores.get("quality"), 0.0)

    # 如果已有非零质量分，保留；否则用 W11 proxy evidence 构造可排序的 V0 质量代理。
    if current_quality > 0:
        return round(current_quality, 4)

    if semantic > 0 or temporal > 0:
        return round(0.6 * semantic + 0.4 * temporal, 4)

    return 0.0


def enrich_seed(seed: dict[str, Any]) -> dict[str, Any]:
    source = seed.setdefault("source", {})
    scores = seed.setdefault("scores_proxy", {})

    if not source.get("artifactUri"):
        source["artifactUri"] = MAINBASE_EVAL_ARTIFACT

    # 明确拆开 Mainbase eval summary 和 Cloud gate summary，避免把 Cloud gate 当成 eval summary。
    old_eval_summary = str(source.get("evalSummaryUri") or "")
    source["mainbaseEvalSummaryUri"] = MAINBASE_EVAL_METRICS
    source["mainbaseE2EIndexUri"] = MAINBASE_E2E_INDEX
    source["handoffManifestUri"] = MAINBASE_HANDOFF

    if old_eval_summary:
        source["cloudGateSummaryUri"] = old_eval_summary
    else:
        source["cloudGateSummaryUri"] = ""

    source["evalSummaryUri"] = MAINBASE_EVAL_METRICS

    quality = compute_proxy_quality(seed)
    scores["quality"] = quality
    scores["quality_formula"] = "if missing: 0.6 * semantic + 0.4 * temporal"
    scores["quality_boundary"] = "V0 proxy quality for W12 sorting only; not human perceptual audio quality."

    seed["w12_consumption_contract"] = {
        "artifact_pointer_ready": bool(source.get("artifactUri")),
        "eval_summary_ready": bool(source.get("mainbaseEvalSummaryUri")),
        "cloud_gate_summary_ready": bool(source.get("cloudGateSummaryUri")),
        "quality_proxy_ready": quality > 0,
        "expected_consumer": [
            "Mainbase SoundLayer Blueprint schema v0",
            "Java task state machine / evidence-link API",
            "Cloud local dashboard or gate summary",
        ],
    }

    return seed


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if not IN_JSON.exists():
        raise SystemExit(f"[FAIL] missing {IN_JSON.relative_to(ROOT)}")

    data = load_json(IN_JSON)
    seeds = data.get("blueprint_seeds", [])
    if not isinstance(seeds, list) or not seeds:
        raise SystemExit("[FAIL] no blueprint_seeds found")

    enriched = [enrich_seed(seed) for seed in seeds]
    data["blueprint_seeds"] = enriched

    data.setdefault("summary", {})
    data["summary"]["seed_count"] = len(enriched)
    data["summary"]["has_artifact_links"] = all(
        bool(seed.get("source", {}).get("artifactUri")) for seed in enriched
    )
    data["summary"]["has_mainbase_eval_summary_links"] = all(
        bool(seed.get("source", {}).get("mainbaseEvalSummaryUri")) for seed in enriched
    )
    data["summary"]["has_cloud_gate_summary_links"] = any(
        bool(seed.get("source", {}).get("cloudGateSummaryUri")) for seed in enriched
    )
    data["summary"]["quality_proxy_min"] = min(
        bounded_float(seed.get("scores_proxy", {}).get("quality")) for seed in enriched
    )
    data["summary"]["quality_proxy_max"] = max(
        bounded_float(seed.get("scores_proxy", {}).get("quality")) for seed in enriched
    )

    data["enrichment"] = {
        "schema_version": "week12_blueprint_seed_enrichment_v1",
        "generated_at_local": datetime.now().isoformat(timespec="seconds"),
        "purpose": "Fill W12-consumable artifact/eval pointers and nonzero proxy quality from Week11 evidence.",
        "boundary": [
            "artifactUri points to local Mainbase eval artifact, not durable object storage.",
            "quality is a proxy score for W12 ordering, not human perceptual audio quality.",
            "cloudGateSummaryUri is kept separate from mainbaseEvalSummaryUri.",
        ],
    }

    warnings: list[str] = []
    if not data["summary"]["has_artifact_links"]:
        warnings.append("some seeds still have empty artifactUri")
    if data["summary"]["quality_proxy_min"] <= 0:
        warnings.append("some seeds still have non-positive proxy quality")

    data["warnings"] = warnings

    OUT_JSON.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "blueprint_id",
            "source_case_id",
            "semantic",
            "temporal",
            "quality",
            "layers_required",
            "qualityGateStatus",
            "artifactUri",
            "mainbaseEvalSummaryUri",
            "cloudGateSummaryUri",
            "failure_reason",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for seed in enriched:
            source = seed.get("source", {})
            scores = seed.get("scores_proxy", {})
            writer.writerow(
                {
                    "blueprint_id": seed.get("blueprint_id", ""),
                    "source_case_id": seed.get("source_case_id", ""),
                    "semantic": scores.get("semantic", ""),
                    "temporal": scores.get("temporal", ""),
                    "quality": scores.get("quality", ""),
                    "layers_required": "|".join(seed.get("layers_required", [])),
                    "qualityGateStatus": source.get("qualityGateStatus", ""),
                    "artifactUri": source.get("artifactUri", ""),
                    "mainbaseEvalSummaryUri": source.get("mainbaseEvalSummaryUri", ""),
                    "cloudGateSummaryUri": source.get("cloudGateSummaryUri", ""),
                    "failure_reason": seed.get("failure_reason", ""),
                }
            )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"week12_blueprint_seed_enrichment_{stamp}.log"
    status = "PASS" if not warnings else "PASS_WITH_WARNINGS"
    log_path.write_text(
        "\n".join(
            [
                "===== Week12 Blueprint Seed Enrichment =====",
                f"status={status}",
                f"output_json={OUT_JSON.relative_to(ROOT)}",
                f"output_csv={OUT_CSV.relative_to(ROOT)}",
                f"seed_count={len(enriched)}",
                f"has_artifact_links={data['summary']['has_artifact_links']}",
                f"has_mainbase_eval_summary_links={data['summary']['has_mainbase_eval_summary_links']}",
                f"has_cloud_gate_summary_links={data['summary']['has_cloud_gate_summary_links']}",
                f"quality_proxy_min={data['summary']['quality_proxy_min']}",
                f"quality_proxy_max={data['summary']['quality_proxy_max']}",
                f"warnings={warnings}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"[{status}] Week12 blueprint seed enrichment")
    print(f"output_json={OUT_JSON.relative_to(ROOT)}")
    print(f"output_csv={OUT_CSV.relative_to(ROOT)}")
    print(f"log={log_path.relative_to(ROOT)}")
    print(f"seed_count={len(enriched)}")
    print(f"has_artifact_links={data['summary']['has_artifact_links']}")
    print(f"has_mainbase_eval_summary_links={data['summary']['has_mainbase_eval_summary_links']}")
    print(f"has_cloud_gate_summary_links={data['summary']['has_cloud_gate_summary_links']}")
    print(f"quality_proxy_min={data['summary']['quality_proxy_min']}")
    print(f"quality_proxy_max={data['summary']['quality_proxy_max']}")
    if warnings:
        for item in warnings:
            print(f"WARNING: {item}")


if __name__ == "__main__":
    main()