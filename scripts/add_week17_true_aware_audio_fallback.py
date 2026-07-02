from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PACK_DIR = Path("artifacts/demo/week17_true_aware_demo_pack_seed")
CASE_CARD = PACK_DIR / "case_card_glass_drop_room_001.json"
INDEX_HTML = PACK_DIR / "index.html"
FLAC_AUDIO = PACK_DIR / "audio/glass_drop_room_001__mmaudio__true_replacement_v0.flac"
WAV_AUDIO = PACK_DIR / "audio/glass_drop_room_001__mmaudio__true_replacement_v0.wav"

REPORT = Path("reports/week17_true_aware_audio_fallback_20260702.json")


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return data


def run_ffmpeg() -> tuple[bool, str]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False, "ffmpeg_not_found"

    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(FLAC_AUDIO),
        "-ac",
        "1",
        "-ar",
        "44100",
        str(WAV_AUDIO),
    ]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        return False, proc.stderr[-2000:]
    return True, "ffmpeg_conversion_ok"


def write_html(case: dict[str, Any], wav_exists: bool) -> None:
    flac_rel = "audio/glass_drop_room_001__mmaudio__true_replacement_v0.flac"
    wav_rel = "audio/glass_drop_room_001__mmaudio__true_replacement_v0.wav"

    source_lines = []
    if wav_exists:
        source_lines.append(f'    <source src="{wav_rel}" type="audio/wav">')
    source_lines.append(f'    <source src="{flac_rel}" type="audio/flac">')

    source_block = "\n".join(source_lines)

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Week17 True-aware Demo Pack Seed</title>
</head>
<body>
  <h1>Week17 True-aware Demo Pack Seed</h1>
  <p><strong>Status:</strong> ready_for_friday_packaging</p>
  <p><strong>Case:</strong> {case.get("case_id")}</p>
  <p><strong>Model:</strong> {case.get("primary_model")}</p>
  <p><strong>Safe true MMAudio count:</strong> {case.get("safe_true_mmaudio_record_count")}</p>
  <p><strong>Raw candidate context count:</strong> {case.get("raw_candidate_record_count")}</p>
  <p><strong>Ready for Friday demo pack:</strong> {str(case.get("cloud_decision", {}).get("readyForFridayDemoPack")).lower()}</p>

  <h2>Primary audio</h2>
  <audio controls>
{source_block}
    Your browser does not support the audio element.
  </audio>

  <p><strong>Preferred preview:</strong> {"WAV fallback" if wav_exists else "FLAC source only"}</p>
  <p><strong>Original model artifact:</strong> {flac_rel}</p>
  <p><strong>Browser boundary:</strong> actual playback still depends on browser codec support.</p>

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
"""
    INDEX_HTML.write_text(html, encoding="utf-8")


def main() -> None:
    if not FLAC_AUDIO.exists():
        raise FileNotFoundError(FLAC_AUDIO)
    if not CASE_CARD.exists():
        raise FileNotFoundError(CASE_CARD)

    case = load_json(CASE_CARD)

    converted, conversion_note = run_ffmpeg()
    wav_exists = WAV_AUDIO.exists()

    write_html(case, wav_exists)

    report = {
        "schema_version": "week17.true_aware.audio_fallback.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "case_id": case.get("case_id"),
        "fallback_status": "wav_fallback_ready" if wav_exists else "flac_only",
        "ffmpeg_conversion_success": converted,
        "conversion_note": conversion_note,
        "flac_audio": str(FLAC_AUDIO),
        "flac_exists": FLAC_AUDIO.exists(),
        "flac_size_bytes": FLAC_AUDIO.stat().st_size,
        "wav_audio": str(WAV_AUDIO),
        "wav_exists": wav_exists,
        "wav_size_bytes": WAV_AUDIO.stat().st_size if wav_exists else 0,
        "html_updated": True,
        "html_audio_sources": ["audio/wav", "audio/flac"] if wav_exists else ["audio/flac"],
        "browser_boundary": "HTML now offers multiple audio sources when WAV fallback is available; actual playback still depends on browser support.",
        "claim_boundary": case.get("claim_boundary", {}),
    }

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("WROTE", REPORT)
    print("FALLBACK_STATUS=", report["fallback_status"])
    print("FFMPEG_CONVERSION_SUCCESS=", converted)
    print("WAV_EXISTS=", wav_exists)
    print("WAV_SIZE_BYTES=", report["wav_size_bytes"])
    print("HTML_UPDATED=1")


if __name__ == "__main__":
    main()