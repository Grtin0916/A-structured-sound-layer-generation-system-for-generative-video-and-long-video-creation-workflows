from __future__ import annotations

import csv
import hashlib
import html
import json
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(".").resolve()

SOURCE_PACK = ROOT / "artifacts/demo/week17_true_aware_demo_pack_seed"
SOURCE_ZIP = ROOT / "artifacts/demo/week17_true_aware_demo_pack_export_20260702.zip"

RELEASE_DIR = ROOT / "artifacts/demo/week17_true_aware_demo_release"
RELEASE_ZIP = ROOT / "artifacts/demo/week17_true_aware_demo_release_20260703.zip"

MANIFEST = ROOT / "reports/week17_demo_release_manifest_20260703.json"
CLAIM_CARD = ROOT / "reports/week17_demo_claim_boundary_card_20260703.json"
RELEASE_NOTES = ROOT / "reports/week17_demo_release_notes_20260703.md"
VERIFY_REPORT = ROOT / "reports/week17_demo_release_verify_20260703.json"

WALKTHROUGH = ROOT / "docs/demo/week17_true_aware_demo_walkthrough.md"
INTERVIEW_SCRIPT = ROOT / "docs/demo/week17_true_aware_interview_script.md"

PRIOR_PORTABILITY = ROOT / "reports/week17_true_aware_demo_export_portability_smoke_20260702.json"
PRIOR_CLAIM_GUARD = ROOT / "reports/week17_true_aware_claim_guard_20260702.json"
PRIOR_REGISTRY = ROOT / "reports/week17_true_aware_candidate_registry_20260702.csv"
PRIOR_RESULT_CARD = ROOT / "reports/week17_true_aware_result_card_payload_20260702.json"
PRIOR_PLATFORM_BRIDGE = ROOT / "reports/week17_true_aware_platform_bridge_payload_20260702.json"


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_read_error": str(exc), "_path": str(path)}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def copy_source_pack() -> None:
    if RELEASE_DIR.exists():
        shutil.rmtree(RELEASE_DIR)
    RELEASE_DIR.mkdir(parents=True, exist_ok=True)

    if SOURCE_PACK.exists():
        shutil.copytree(SOURCE_PACK, RELEASE_DIR, dirs_exist_ok=True)
    elif SOURCE_ZIP.exists():
        with zipfile.ZipFile(SOURCE_ZIP, "r") as zf:
            zf.extractall(RELEASE_DIR)
    else:
        raise FileNotFoundError("No source demo pack found")


def read_registry_rows() -> list[dict]:
    if not PRIOR_REGISTRY.exists():
        return []
    with PRIOR_REGISTRY.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def build_claim_card(wavs: list[Path], registry_rows: list[dict]) -> dict:
    true_wavs = [
        p for p in wavs
        if "mmaudio" in p.name.lower() and "true" in p.name.lower()
    ]

    card = {
        "release_id": "week17_true_aware_demo_release_20260703",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "CLAIM_SAFE_RELEASE_CANDIDATE",
        "safeTrueMmaudioRecordCount": len(true_wavs),
        "trueMmaudioBatchSuccess": False,
        "fullCandidateRankingAvailable": False,
        "productionSloVerified": False,
        "k6ThresholdPassVerified": False,
        "liveGrafanaImportVerified": False,
        "allowed_claims": [
            "A single true MMAudio replacement candidate is present and traceable.",
            "A portable demo ZIP is available for local preview.",
            "The demo keeps WAV fallback playback for browser compatibility.",
            "Java and Cloud artifacts are trace-linked as offline/demo contracts."
        ],
        "blocked_claims": [
            "Do not claim true MMAudio batch success.",
            "Do not claim full 28-candidate ranking availability.",
            "Do not claim production SLO verification.",
            "Do not claim k6 threshold pass unless k6 is actually executed.",
            "Do not claim live Grafana import unless Grafana import is actually verified."
        ],
        "evidence_inputs": {
            "source_pack_exists": SOURCE_PACK.exists(),
            "source_zip_exists": SOURCE_ZIP.exists(),
            "prior_portability_smoke_exists": PRIOR_PORTABILITY.exists(),
            "prior_claim_guard_exists": PRIOR_CLAIM_GUARD.exists(),
            "prior_registry_exists": PRIOR_REGISTRY.exists(),
            "prior_result_card_exists": PRIOR_RESULT_CARD.exists(),
            "prior_platform_bridge_exists": PRIOR_PLATFORM_BRIDGE.exists(),
            "registry_row_count": len(registry_rows),
            "wav_count": len(wavs),
            "true_wav_paths": [rel(p) for p in true_wavs],
        }
    }
    return card


def write_index_html(wavs: list[Path], claim_card: dict, manifest_name: str) -> None:
    audio_items = []
    for wav in wavs:
        rel_to_release = wav.relative_to(RELEASE_DIR).as_posix()
        audio_items.append(f"""
        <section class="card">
          <h3>{html.escape(wav.name)}</h3>
          <audio controls preload="metadata">
            <source src="{html.escape(rel_to_release)}" type="audio/wav">
            <a href="{html.escape(rel_to_release)}">Download WAV</a>
          </audio>
          <p class="path">{html.escape(rel_to_release)}</p>
        </section>
        """)

    blocked = "".join(f"<li>{html.escape(x)}</li>" for x in claim_card["blocked_claims"])
    allowed = "".join(f"<li>{html.escape(x)}</li>" for x in claim_card["allowed_claims"])

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Week17 True-aware Demo Release Candidate</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, sans-serif; margin: 40px; line-height: 1.55; }}
    .card {{ border: 1px solid #ddd; border-radius: 10px; padding: 16px; margin: 16px 0; }}
    .claim {{ background: #f8f8f8; padding: 16px; border-radius: 10px; }}
    .path {{ font-family: monospace; color: #555; }}
    code {{ background: #f2f2f2; padding: 2px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Week17 True-aware Demo Release Candidate</h1>
  <p>This is a local, claim-safe demo package for the Director-guided Video-to-Audio SoundLayer System.</p>

  <div class="claim">
    <h2>Claim boundary</h2>
    <p><b>Decision:</b> {html.escape(claim_card["decision"])}</p>
    <p><b>safeTrueMmaudioRecordCount:</b> {claim_card["safeTrueMmaudioRecordCount"]}</p>
    <p><b>trueMmaudioBatchSuccess:</b> {str(claim_card["trueMmaudioBatchSuccess"]).lower()}</p>
    <p><b>fullCandidateRankingAvailable:</b> {str(claim_card["fullCandidateRankingAvailable"]).lower()}</p>
    <p><b>productionSloVerified:</b> {str(claim_card["productionSloVerified"]).lower()}</p>
    <p><b>k6ThresholdPassVerified:</b> {str(claim_card["k6ThresholdPassVerified"]).lower()}</p>
    <p><b>liveGrafanaImportVerified:</b> {str(claim_card["liveGrafanaImportVerified"]).lower()}</p>

    <h3>Allowed claims</h3>
    <ul>{allowed}</ul>

    <h3>Blocked claims</h3>
    <ul>{blocked}</ul>
  </div>

  <h2>Audio preview</h2>
  {''.join(audio_items)}

  <h2>Trace links</h2>
  <ul>
    <li><a href="{html.escape(manifest_name)}">release manifest</a></li>
    <li><a href="walkthrough.md">walkthrough</a></li>
  </ul>
</body>
</html>
"""
    (RELEASE_DIR / "index.html").write_text(html_text, encoding="utf-8")


def write_docs(claim_card: dict) -> None:
    RELEASE_NOTES.write_text(f"""# Week17 True-aware Demo Release Candidate

## What this release proves

- A single true MMAudio replacement candidate is present.
- The demo can be packaged into a portable ZIP.
- The release keeps browser-friendly WAV playback.
- Java result-card and Cloud gate artifacts are trace-linked.

## What this release does not prove

- true MMAudio batch success: `{claim_card["trueMmaudioBatchSuccess"]}`
- full candidate ranking availability: `{claim_card["fullCandidateRankingAvailable"]}`
- production SLO verification: `{claim_card["productionSloVerified"]}`
- k6 threshold pass: `{claim_card["k6ThresholdPassVerified"]}`
- live Grafana import: `{claim_card["liveGrafanaImportVerified"]}`

## Engineering interpretation

This is a claim-safe release candidate, not a production system. Its value is that it turns partial model success and runtime boundary into a portable demo path.
""", encoding="utf-8")

    WALKTHROUGH.write_text("""# Week17 True-aware Demo Walkthrough

1. Unzip `week17_true_aware_demo_release_20260703.zip`.
2. Open `index.html` directly or serve the folder with `python -m http.server`.
3. Play the WAV candidate in the browser.
4. Read the claim boundary before presenting the result.
5. Trace the release manifest back to Mainbase, Java result-card, and Cloud gate artifacts.

Key story: the project does not overclaim batch model success. It presents one true MMAudio result, fallback-aware engineering, Java/Cloud traceability, and explicit blocked claims.
""", encoding="utf-8")

    INTERVIEW_SCRIPT.write_text("""# Week17 Interview Script

I built a Director-guided video-to-audio sound-layer workflow. The important part is not just calling one generator, but making generation controllable and auditable.

In Week17, I moved from placeholder/fallback audio toward a true-aware demo path. One MMAudio replacement candidate is preserved as a true model-backed artifact. Around it, I built claim boundaries, browser WAV preview, Java result-card handoff, and Cloud demo gate artifacts.

The honest boundary is important: I cannot claim true batch success, full candidate ranking, production SLO, k6 threshold pass, or live Grafana import. The system is a release candidate that demonstrates how to present partial model success without inflating it.
""", encoding="utf-8")


def build_manifest(wavs: list[Path], claim_card: dict) -> dict:
    tracked_files = []
    for path in sorted(RELEASE_DIR.rglob("*")):
        if path.is_file():
            tracked_files.append({
                "path": rel(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })

    source_files = [
        SOURCE_ZIP,
        PRIOR_PORTABILITY,
        PRIOR_CLAIM_GUARD,
        PRIOR_REGISTRY,
        PRIOR_RESULT_CARD,
        PRIOR_PLATFORM_BRIDGE,
    ]

    manifest = {
        "release_id": "week17_true_aware_demo_release_20260703",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "release_dir": rel(RELEASE_DIR),
        "release_zip": rel(RELEASE_ZIP),
        "index_html": rel(RELEASE_DIR / "index.html"),
        "wav_count": len(wavs),
        "true_mmaudio_wav_count": claim_card["safeTrueMmaudioRecordCount"],
        "claim_boundary_card": rel(CLAIM_CARD),
        "release_notes": rel(RELEASE_NOTES),
        "walkthrough": rel(WALKTHROUGH),
        "interview_script": rel(INTERVIEW_SCRIPT),
        "source_artifacts": [
            {"path": rel(p), "exists": p.exists(), "size_bytes": p.stat().st_size if p.exists() else 0}
            for p in source_files
        ],
        "release_files": tracked_files,
        "claim_boundary": {
            "trueMmaudioBatchSuccess": claim_card["trueMmaudioBatchSuccess"],
            "fullCandidateRankingAvailable": claim_card["fullCandidateRankingAvailable"],
            "productionSloVerified": claim_card["productionSloVerified"],
            "k6ThresholdPassVerified": claim_card["k6ThresholdPassVerified"],
            "liveGrafanaImportVerified": claim_card["liveGrafanaImportVerified"],
        }
    }
    return manifest


def zip_release() -> None:
    if RELEASE_ZIP.exists():
        RELEASE_ZIP.unlink()
    with zipfile.ZipFile(RELEASE_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(RELEASE_DIR.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(RELEASE_DIR.parent))


def verify_release(claim_card: dict, manifest: dict) -> dict:
    checks = {
        "release_dir_exists": RELEASE_DIR.exists(),
        "release_zip_exists": RELEASE_ZIP.exists(),
        "index_html_exists": (RELEASE_DIR / "index.html").exists(),
        "wav_fallback_present": any(RELEASE_DIR.rglob("*.wav")),
        "manifest_exists": MANIFEST.exists(),
        "claim_card_exists": CLAIM_CARD.exists(),
        "claim_boundary_preserved": all([
            claim_card["trueMmaudioBatchSuccess"] is False,
            claim_card["fullCandidateRankingAvailable"] is False,
            claim_card["productionSloVerified"] is False,
            claim_card["k6ThresholdPassVerified"] is False,
            claim_card["liveGrafanaImportVerified"] is False,
        ]),
        "safe_true_mmaudio_record_count": claim_card["safeTrueMmaudioRecordCount"],
        "zip_valid": False,
        "zip_contains_index": False,
        "zip_contains_wav": False,
    }

    if RELEASE_ZIP.exists():
        try:
            with zipfile.ZipFile(RELEASE_ZIP, "r") as zf:
                bad = zf.testzip()
                names = zf.namelist()
                checks["zip_valid"] = bad is None
                checks["zip_contains_index"] = any(name.endswith("index.html") for name in names)
                checks["zip_contains_wav"] = any(name.lower().endswith(".wav") for name in names)
        except Exception as exc:
            checks["zip_error"] = str(exc)

    decision = "PASS" if all([
        checks["release_dir_exists"],
        checks["release_zip_exists"],
        checks["index_html_exists"],
        checks["wav_fallback_present"],
        checks["manifest_exists"],
        checks["claim_card_exists"],
        checks["claim_boundary_preserved"],
        checks["zip_valid"],
        checks["zip_contains_index"],
        checks["zip_contains_wav"],
        checks["safe_true_mmaudio_record_count"] >= 1,
    ]) else "FAIL"

    return {
        "decision": decision,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "release_zip": manifest["release_zip"],
        "manifest": rel(MANIFEST),
        "claim_card": rel(CLAIM_CARD),
    }


def main() -> int:
    for p in [MANIFEST.parent, CLAIM_CARD.parent, RELEASE_NOTES.parent, WALKTHROUGH.parent, INTERVIEW_SCRIPT.parent]:
        p.mkdir(parents=True, exist_ok=True)

    copy_source_pack()

    wavs = sorted(RELEASE_DIR.rglob("*.wav"))
    registry_rows = read_registry_rows()
    claim_card = build_claim_card(wavs, registry_rows)

    (RELEASE_DIR / "claim_boundary_card.json").write_text(
        json.dumps(claim_card, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    write_index_html(wavs, claim_card, "release_manifest.json")
    write_docs(claim_card)

    manifest = build_manifest(wavs, claim_card)

    (RELEASE_DIR / "release_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    CLAIM_CARD.write_text(json.dumps(claim_card, ensure_ascii=False, indent=2), encoding="utf-8")
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    zip_release()

    verify = verify_release(claim_card, manifest)
    VERIFY_REPORT.write_text(json.dumps(verify, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "decision": verify["decision"],
        "release_dir": rel(RELEASE_DIR),
        "release_zip": rel(RELEASE_ZIP),
        "manifest": rel(MANIFEST),
        "claim_card": rel(CLAIM_CARD),
        "verify_report": rel(VERIFY_REPORT),
        "wav_count": manifest["wav_count"],
        "safeTrueMmaudioRecordCount": claim_card["safeTrueMmaudioRecordCount"],
    }, ensure_ascii=False, indent=2))

    return 0 if verify["decision"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
