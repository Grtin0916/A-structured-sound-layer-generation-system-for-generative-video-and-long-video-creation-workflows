from __future__ import annotations

import argparse
import json
import mimetypes
import os
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


PACK_DIR = Path("artifacts/demo/week17_true_aware_demo_pack_seed")
REPORT_PATH = Path("reports/week17_true_aware_demo_preview_smoke_20260702.json")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return data


def build_report(port: int) -> dict[str, Any]:
    index_path = PACK_DIR / "demo_pack_index.json"
    case_path = PACK_DIR / "case_card_glass_drop_room_001.json"
    html_path = PACK_DIR / "index.html"

    index = load_json(index_path)
    case = load_json(case_path)

    audio_rel = case.get("primary_audio_pack_path")
    if not isinstance(audio_rel, str) or not audio_rel:
        raise ValueError("case_card.primary_audio_pack_path is missing")

    audio_path = PACK_DIR / audio_rel
    required = {
        "pack_dir": PACK_DIR,
        "index_json": index_path,
        "case_card": case_path,
        "html": html_path,
        "audio": audio_path,
        "walkthrough": PACK_DIR / "walkthrough.md",
        "artifact_manifest": PACK_DIR / "artifact_manifest.csv",
    }

    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing demo pack files: {missing}")

    mime_type = mimetypes.guess_type(str(audio_path))[0] or "audio/flac"

    report = {
        "schema_version": "week17.true_aware.demo_preview_smoke.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "preview_status": "ready",
        "pack_dir": str(PACK_DIR),
        "entry_file": str(html_path),
        "local_url": f"http://127.0.0.1:{port}/index.html",
        "index_json_url": f"http://127.0.0.1:{port}/demo_pack_index.json",
        "case_id": case.get("case_id"),
        "demo_pack_status": index.get("demo_pack_status"),
        "safe_true_mmaudio_record_count": case.get("safe_true_mmaudio_record_count"),
        "raw_candidate_record_count": case.get("raw_candidate_record_count"),
        "cloud_ready": case.get("cloud_decision", {}).get("readyForFridayDemoPack"),
        "primary_audio_pack_path": audio_rel,
        "primary_audio_exists": audio_path.exists(),
        "primary_audio_size_bytes": audio_path.stat().st_size,
        "primary_audio_mime_guess": mime_type,
        "html_audio_element_expected": True,
        "browser_boundary": (
            "The page embeds the audio artifact through an HTML audio element. "
            "Actual FLAC playback depends on browser codec support."
        ),
        "forbidden_claims": case.get("claim_boundary", {}).get("forbidden", []),
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def serve(port: int) -> None:
    if not PACK_DIR.exists():
        raise FileNotFoundError(PACK_DIR)
    os.chdir(PACK_DIR)
    server = ThreadingHTTPServer(("127.0.0.1", port), SimpleHTTPRequestHandler)
    print(f"Serving {PACK_DIR} at http://127.0.0.1:{port}/index.html")
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--serve", action="store_true")
    args = parser.parse_args()

    report = build_report(args.port)
    print("WROTE", REPORT_PATH)
    print("PREVIEW_STATUS=", report["preview_status"])
    print("LOCAL_URL=", report["local_url"])
    print("PRIMARY_AUDIO_EXISTS=", report["primary_audio_exists"])
    print("SAFE_TRUE_MMAUDIO_RECORD_COUNT=", report["safe_true_mmaudio_record_count"])
    print("RAW_CANDIDATE_RECORD_COUNT=", report["raw_candidate_record_count"])

    if args.serve:
        serve(args.port)


if __name__ == "__main__":
    main()