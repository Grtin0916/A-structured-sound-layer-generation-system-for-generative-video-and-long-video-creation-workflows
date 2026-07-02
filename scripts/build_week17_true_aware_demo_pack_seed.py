from __future__ import annotations

import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MAINBASE = Path(".").resolve()
JAVA = Path.home() / "work/media-task-platform-java"
CLOUD = Path.home() / "work/ai-job-platform-cloud"

OUT_DIR = MAINBASE / "artifacts/demo/week17_true_aware_demo_pack_seed"
INPUTS_DIR = OUT_DIR / "inputs"
AUDIO_DIR = OUT_DIR / "audio"
REPORTS_DIR = OUT_DIR / "reports"

MAINBASE_BRIDGE = MAINBASE / "reports/week17_true_aware_platform_bridge_payload_20260702.json"
MAINBASE_RESULT_CARD = MAINBASE / "artifacts/model_race/week17_true_aware_reranker/true_aware_result_card_20260702.json"
TRUE_AUDIO = MAINBASE / "experiments/mmaudio_true_replacement_2026_06_30/candidates/glass_drop_room_001__mmaudio__true_replacement_v0.flac"

JAVA_API_REPORT = JAVA / "artifacts/manifests/week17_true_aware_result_card/week17_true_aware_result_card_api_report.json"
CLOUD_GATE = CLOUD / "artifacts/demo/week17_true_aware_result_card_cloud_gate/week17_true_aware_result_card_cloud_gate.json"
CLOUD_METRICS = CLOUD / "loadtest/reports/week17_true_aware_result_card_metrics.prom"
CLOUD_DASHBOARD = CLOUD / "observability/grafana/dashboards/week17_true_aware_result_card_dashboard.json"

OUT_INDEX = OUT_DIR / "demo_pack_index.json"
OUT_CASE_CARD = OUT_DIR / "case_card_glass_drop_room_001.json"
OUT_WALKTHROUGH = OUT_DIR / "walkthrough.md"
OUT_HTML = OUT_DIR / "index.html"
OUT_MANIFEST_CSV = OUT_DIR / "artifact_manifest.csv"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return data


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_input(src: Path, dst_dir: Path) -> dict[str, Any]:
    if not src.exists():
        raise FileNotFoundError(src)
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    shutil.copy2(src, dst)
    return {
        "source": str(src),
        "pack_path": str(dst.relative_to(OUT_DIR)),
        "size_bytes": dst.stat().st_size,
        "sha256": sha256_file(dst),
    }


def main() -> None:
    bridge = load_json(MAINBASE_BRIDGE)
    result_card = load_json(MAINBASE_RESULT_CARD)
    java_report = load_json(JAVA_API_REPORT)
    cloud_gate = load_json(CLOUD_GATE)
    dashboard = load_json(CLOUD_DASHBOARD)

    decision = cloud_gate.get("decision", {})
    runtime_boundary = cloud_gate.get("runtime_boundary", {})
    strict = bridge.get("strict_boundary", {})
    platform_card = bridge.get("platform_result_card", {})

    if decision.get("readyForFridayDemoPack") is not True:
        raise RuntimeError("Cloud gate is not ready for Friday demo pack")
    if strict.get("true_mmaudio_audio_artifact_count") != 1:
        raise RuntimeError("Expected exactly one claim-safe true MMAudio audio artifact")
    if platform_card.get("safe_true_mmaudio_record_count") != 1:
        raise RuntimeError("safe_true_mmaudio_record_count must be 1")
    if not TRUE_AUDIO.exists():
        raise FileNotFoundError(TRUE_AUDIO)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    artifacts = [
        {"role": "mainbase_bridge", **copy_input(MAINBASE_BRIDGE, INPUTS_DIR)},
        {"role": "mainbase_result_card", **copy_input(MAINBASE_RESULT_CARD, INPUTS_DIR)},
        {"role": "java_api_report", **copy_input(JAVA_API_REPORT, INPUTS_DIR)},
        {"role": "cloud_gate", **copy_input(CLOUD_GATE, INPUTS_DIR)},
        {"role": "cloud_metrics", **copy_input(CLOUD_METRICS, REPORTS_DIR)},
        {"role": "cloud_dashboard", **copy_input(CLOUD_DASHBOARD, REPORTS_DIR)},
        {"role": "true_mmaudio_audio", **copy_input(TRUE_AUDIO, AUDIO_DIR)},
    ]

    case_card = {
        "case_id": "glass_drop_room_001",
        "demo_status": "single_true_v2a_candidate_available",
        "primary_model": "MMAudio",
        "primary_audio_pack_path": next(a["pack_path"] for a in artifacts if a["role"] == "true_mmaudio_audio"),
        "primary_audio_sha256": next(a["sha256"] for a in artifacts if a["role"] == "true_mmaudio_audio"),
        "safe_true_mmaudio_record_count": platform_card.get("safe_true_mmaudio_record_count"),
        "raw_candidate_record_count": platform_card.get("raw_candidate_record_count"),
        "raw_winner_record_count": platform_card.get("raw_winner_record_count"),
        "java_endpoints": java_report.get("endpoints", []),
        "cloud_decision": decision,
        "runtime_boundary": runtime_boundary,
        "claim_boundary": {
            "allowed": [
                "One true MMAudio video-conditioned candidate is available.",
                "Java exposes the result through an artifact-backed result-card API.",
                "Cloud can use the result as a Friday demo gate seed.",
            ],
            "forbidden": [
                "No batch true MMAudio success.",
                "No full candidate ranking claim.",
                "No production SLO claim.",
                "No k6 threshold pass claim.",
            ],
        },
    }

    index = {
        "schema_version": "week17.true_aware.demo_pack_seed.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "demo_pack_status": "ready_for_friday_packaging",
        "system_path": [
            "Mainbase true MMAudio single artifact",
            "Mainbase claim-safe result card",
            "Java artifact-backed result-card API",
            "Cloud demo gate seed",
            "Prometheus metrics sample",
            "Grafana dashboard seed",
        ],
        "case_cards": [str(OUT_CASE_CARD.relative_to(OUT_DIR))],
        "artifact_manifest": str(OUT_MANIFEST_CSV.relative_to(OUT_DIR)),
        "walkthrough": str(OUT_WALKTHROUGH.relative_to(OUT_DIR)),
        "html_preview": str(OUT_HTML.relative_to(OUT_DIR)),
        "artifacts": artifacts,
        "decision_summary": decision,
        "runtime_boundary": runtime_boundary,
        "dashboard_seed_title": dashboard.get("title"),
    }

    OUT_CASE_CARD.write_text(json.dumps(case_card, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_INDEX.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")

    with OUT_MANIFEST_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["role", "source", "pack_path", "size_bytes", "sha256"],
        )
        writer.writeheader()
        writer.writerows(artifacts)

    OUT_WALKTHROUGH.write_text(
        f"""# Week17 True-aware Demo Pack Seed

## What this demo shows

This seed demonstrates one claim-safe true MMAudio video-conditioned candidate flowing through the three-repo system.

## System path

1. Mainbase produced one true MMAudio candidate for `glass_drop_room_001`.
2. Mainbase wrapped it in a claim-safe result card.
3. Java exposed the result through an artifact-backed API.
4. Cloud consumed the Java report and generated a demo gate seed.
5. Cloud emitted Prometheus metrics and a Grafana dashboard seed.

## Claim boundary

Allowed:

- One true MMAudio video-conditioned candidate exists.
- Java can expose this result as a result-card API.
- Cloud can treat it as ready for Friday demo packaging.

Forbidden:

- Do not claim true MMAudio batch success.
- Do not claim full 28-candidate ranking.
- Do not claim production SLO.
- Do not claim k6 threshold pass.

## Primary audio

`{case_card["primary_audio_pack_path"]}`

## Key numbers

- safe true MMAudio count: `{case_card["safe_true_mmaudio_record_count"]}`
- raw candidate context count: `{case_card["raw_candidate_record_count"]}`
- ready for Friday demo pack: `{decision.get("readyForFridayDemoPack")}`
""",
        encoding="utf-8",
    )

    OUT_HTML.write_text(
        f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Week17 True-aware Demo Pack Seed</title>
</head>
<body>
  <h1>Week17 True-aware Demo Pack Seed</h1>
  <p><strong>Status:</strong> ready_for_friday_packaging</p>
  <p><strong>Case:</strong> glass_drop_room_001</p>
  <p><strong>Model:</strong> MMAudio</p>
  <p><strong>Safe true MMAudio count:</strong> {case_card["safe_true_mmaudio_record_count"]}</p>
  <p><strong>Raw candidate context count:</strong> {case_card["raw_candidate_record_count"]}</p>
  <p><strong>Ready for Friday demo pack:</strong> {str(decision.get("readyForFridayDemoPack")).lower()}</p>

  <h2>Primary audio</h2>
  <audio controls src="{case_card["primary_audio_pack_path"]}"></audio>
  <p>{case_card["primary_audio_pack_path"]}</p>

  <h2>Allowed claim</h2>
  <p>One true MMAudio video-conditioned candidate is available and can be consumed by Java and Cloud.</p>

  <h2>Forbidden claims</h2>
  <ul>
    <li>No true MMAudio batch success.</li>
    <li>No full candidate ranking claim.</li>
    <li>No production SLO claim.</li>
    <li>No k6 threshold pass claim.</li>
  </ul>
</body>
</html>
""",
        encoding="utf-8",
    )

    print("WROTE", OUT_INDEX)
    print("WROTE", OUT_CASE_CARD)
    print("WROTE", OUT_WALKTHROUGH)
    print("WROTE", OUT_HTML)
    print("WROTE", OUT_MANIFEST_CSV)
    print("DEMO_PACK_STATUS=ready_for_friday_packaging")
    print("SAFE_TRUE_MMAUDIO_RECORD_COUNT=", case_card["safe_true_mmaudio_record_count"])
    print("RAW_CANDIDATE_RECORD_COUNT=", case_card["raw_candidate_record_count"])
    print("PRIMARY_AUDIO_PACK_PATH=", case_card["primary_audio_pack_path"])


if __name__ == "__main__":
    main()