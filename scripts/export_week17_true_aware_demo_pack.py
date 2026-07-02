from __future__ import annotations

import csv
import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PACK_DIR = Path("artifacts/demo/week17_true_aware_demo_pack_seed")
EXPORT_DIR = Path("artifacts/demo/week17_true_aware_demo_pack_export_20260702")
EXPORT_PACK_DIR = EXPORT_DIR / "week17_true_aware_demo_pack_seed"
ZIP_PATH = Path("artifacts/demo/week17_true_aware_demo_pack_export_20260702.zip")
REPORT_PATH = Path("reports/week17_true_aware_demo_handoff_export_20260702.json")
GITIGNORE = Path(".gitignore")

INDEX_JSON = PACK_DIR / "demo_pack_index.json"
CASE_CARD = PACK_DIR / "case_card_glass_drop_room_001.json"
INDEX_HTML = PACK_DIR / "index.html"
WAV_AUDIO = PACK_DIR / "audio/glass_drop_room_001__mmaudio__true_replacement_v0.wav"
FLAC_AUDIO = PACK_DIR / "audio/glass_drop_room_001__mmaudio__true_replacement_v0.flac"


def load_json(path: Path) -> dict[str, Any]:
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


def ensure_required_files() -> None:
    required = [
        INDEX_JSON,
        CASE_CARD,
        INDEX_HTML,
        WAV_AUDIO,
        FLAC_AUDIO,
        PACK_DIR / "walkthrough.md",
        PACK_DIR / "artifact_manifest.csv",
        PACK_DIR / "inputs/week17_true_aware_result_card_api_report.json",
        PACK_DIR / "inputs/week17_true_aware_result_card_cloud_gate.json",
        PACK_DIR / "reports/week17_true_aware_result_card_metrics.prom",
        PACK_DIR / "reports/week17_true_aware_result_card_dashboard.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required demo pack files: " + json.dumps(missing, indent=2)
        )


def validate_claim_boundary() -> tuple[dict[str, Any], dict[str, Any]]:
    idx = load_json(INDEX_JSON)
    card = load_json(CASE_CARD)

    if idx.get("demo_pack_status") != "ready_for_friday_packaging":
        raise RuntimeError("demo_pack_status is not ready_for_friday_packaging")
    if card.get("safe_true_mmaudio_record_count") != 1:
        raise RuntimeError("safe_true_mmaudio_record_count must be 1")
    if card.get("raw_candidate_record_count") != 9:
        raise RuntimeError("raw_candidate_record_count must be 9")
    if card.get("cloud_decision", {}).get("readyForFridayDemoPack") is not True:
        raise RuntimeError("Cloud decision is not readyForFridayDemoPack=true")

    return idx, card


def write_quickstart(card: dict[str, Any]) -> None:
    quickstart = EXPORT_DIR / "README_QUICKSTART.md"
    quickstart.write_text(
        f"""# Week17 True-aware Demo Pack Export

## Open locally

Run from the exported directory:

    cd week17_true_aware_demo_pack_seed
    python -m http.server 8787

Then open this URL in a browser:

    http://127.0.0.1:8787/index.html

## What this demonstrates

- One claim-safe true MMAudio video-conditioned candidate.
- Mainbase result-card bridge.
- Java artifact-backed result-card API evidence.
- Cloud demo gate seed.
- Prometheus metrics sample and Grafana dashboard seed.

## Case

- case_id: {card.get("case_id")}
- primary model: {card.get("primary_model")}
- safe true MMAudio count: {card.get("safe_true_mmaudio_record_count")}
- raw candidate context count: {card.get("raw_candidate_record_count")}
- primary audio: {card.get("primary_audio_pack_path")}

## Claim boundary

Allowed:

- One true MMAudio video-conditioned candidate is available.
- Java can expose this result as an artifact-backed result-card API.
- Cloud can use this as a Friday demo gate seed.

Forbidden:

- No true MMAudio batch success.
- No full candidate ranking claim.
- No production SLO claim.
- No k6 threshold pass claim.
""",
        encoding="utf-8",
    )


def build_manifest() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(EXPORT_DIR.rglob("*")):
        if path.is_file():
            rows.append(
                {
                    "path": str(path.relative_to(EXPORT_DIR)),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )

    manifest_csv = EXPORT_DIR / "handoff_manifest.csv"
    with manifest_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "size_bytes", "sha256"])
        writer.writeheader()
        writer.writerows(rows)

    manifest_json = EXPORT_DIR / "handoff_manifest.json"
    manifest_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows


def build_zip() -> None:
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()

    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(EXPORT_DIR.rglob("*")):
            if path.is_file():
                arcname = Path(EXPORT_DIR.name) / path.relative_to(EXPORT_DIR)
                zf.write(path, arcname.as_posix())


def append_gitignore() -> bool:
    marker = "# week17 true-aware local runtime leftovers"
    block = f"""
{marker}
artifacts/logs/download_bigvgan_v2_44k_20260701.log
artifacts/logs/download_bigvgan_v2_44k_curl_20260701.log
artifacts/logs/download_mmaudio_small_44k_20260701.log
artifacts/logs/download_mmaudio_synchformer_20260701.log
artifacts/logs/download_mmaudio_v1_44_20260701.log
artifacts/logs/download_openclip_dfn5b_20260701.log
artifacts/logs/download_openclip_dfn5b_curl_20260701.log
artifacts/logs/mmaudio_manual_small44_probe_20260701.log
artifacts/logs/mmaudio_true_one_mmaudio_mini_20260701.log
artifacts/logs/mmaudio_true_one_mmaudio_mini_proxy_20260701.log
artifacts/logs/week17_true_aware_demo_preview_server_20260702.log
experiments/mmaudio_true_replacement_2026_06_30/manual_small44_probe/
"""
    old = GITIGNORE.read_text(encoding="utf-8") if GITIGNORE.exists() else ""
    if marker in old:
        return False

    GITIGNORE.write_text(old.rstrip() + "\n\n" + block.lstrip(), encoding="utf-8")
    return True


def main() -> None:
    ensure_required_files()
    idx, card = validate_claim_boundary()

    if EXPORT_DIR.exists():
        shutil.rmtree(EXPORT_DIR)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    shutil.copytree(PACK_DIR, EXPORT_PACK_DIR)
    write_quickstart(card)
    rows = build_manifest()
    build_zip()
    gitignore_updated = append_gitignore()

    report = {
        "schema_version": "week17.true_aware.demo_handoff_export.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "export_status": "ready",
        "zip_path": str(ZIP_PATH),
        "zip_exists": ZIP_PATH.exists(),
        "zip_size_bytes": ZIP_PATH.stat().st_size if ZIP_PATH.exists() else 0,
        "zip_sha256": sha256_file(ZIP_PATH) if ZIP_PATH.exists() else None,
        "export_dir": str(EXPORT_DIR),
        "export_file_count": len(rows),
        "demo_pack_status": idx.get("demo_pack_status"),
        "case_id": card.get("case_id"),
        "safe_true_mmaudio_record_count": card.get("safe_true_mmaudio_record_count"),
        "raw_candidate_record_count": card.get("raw_candidate_record_count"),
        "cloud_ready": card.get("cloud_decision", {}).get("readyForFridayDemoPack"),
        "wav_fallback_included": (
            EXPORT_PACK_DIR / "audio/glass_drop_room_001__mmaudio__true_replacement_v0.wav"
        ).exists(),
        "flac_original_included": (
            EXPORT_PACK_DIR / "audio/glass_drop_room_001__mmaudio__true_replacement_v0.flac"
        ).exists(),
        "gitignore_updated": gitignore_updated,
        "claim_boundary": card.get("claim_boundary", {}),
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("WROTE", REPORT_PATH)
    print("WROTE", ZIP_PATH)
    print("EXPORT_STATUS=", report["export_status"])
    print("ZIP_SIZE_BYTES=", report["zip_size_bytes"])
    print("EXPORT_FILE_COUNT=", report["export_file_count"])
    print("SAFE_TRUE_MMAUDIO_RECORD_COUNT=", report["safe_true_mmaudio_record_count"])
    print("RAW_CANDIDATE_RECORD_COUNT=", report["raw_candidate_record_count"])
    print("GITIGNORE_UPDATED=", gitignore_updated)


if __name__ == "__main__":
    main()