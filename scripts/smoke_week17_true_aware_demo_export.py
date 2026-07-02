from __future__ import annotations

import json
import shutil
import subprocess
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ZIP_PATH = Path("artifacts/demo/week17_true_aware_demo_pack_export_20260702.zip")
SMOKE_ROOT = Path("/tmp/week17_true_aware_demo_export_smoke")
REPORT_PATH = Path("reports/week17_true_aware_demo_export_portability_smoke_20260702.json")

PORT = 8789
EXPORT_DIR_NAME = "week17_true_aware_demo_pack_export_20260702"
PACK_DIR = SMOKE_ROOT / EXPORT_DIR_NAME / "week17_true_aware_demo_pack_seed"


def run_cmd(cmd: list[str], cwd: Path | None = None, timeout: int = 10) -> tuple[int, str, str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    return proc.returncode, proc.stdout, proc.stderr


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return data


def main() -> None:
    if not ZIP_PATH.exists():
        raise FileNotFoundError(ZIP_PATH)

    if SMOKE_ROOT.exists():
        shutil.rmtree(SMOKE_ROOT)
    SMOKE_ROOT.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(SMOKE_ROOT)

    required = {
        "index_html": PACK_DIR / "index.html",
        "case_card": PACK_DIR / "case_card_glass_drop_room_001.json",
        "demo_index": PACK_DIR / "demo_pack_index.json",
        "wav_audio": PACK_DIR / "audio/glass_drop_room_001__mmaudio__true_replacement_v0.wav",
        "flac_audio": PACK_DIR / "audio/glass_drop_room_001__mmaudio__true_replacement_v0.flac",
        "java_report": PACK_DIR / "inputs/week17_true_aware_result_card_api_report.json",
        "cloud_gate": PACK_DIR / "inputs/week17_true_aware_result_card_cloud_gate.json",
    }

    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing files after isolated extract: {missing}")

    case = load_json(required["case_card"])
    gate = load_json(required["cloud_gate"])

    server_log = SMOKE_ROOT / "portable_http_server.log"
    server = subprocess.Popen(
        ["python", "-m", "http.server", str(PORT), "--bind", "127.0.0.1"],
        cwd=str(PACK_DIR),
        stdout=server_log.open("w", encoding="utf-8"),
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        time.sleep(2)
        index_rc, index_head, index_err = run_cmd(
            ["curl", "-I", "--max-time", "10", f"http://127.0.0.1:{PORT}/index.html"],
            timeout=12,
        )
        wav_rc, wav_head, wav_err = run_cmd(
            [
                "curl",
                "-I",
                "--max-time",
                "10",
                f"http://127.0.0.1:{PORT}/audio/glass_drop_room_001__mmaudio__true_replacement_v0.wav",
            ],
            timeout=12,
        )
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()

    index_ok = index_rc == 0 and "200 OK" in index_head
    wav_ok = wav_rc == 0 and "200 OK" in wav_head

    report = {
        "schema_version": "week17.true_aware.demo_export_portability_smoke.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "smoke_status": "pass" if index_ok and wav_ok else "fail",
        "zip_path": str(ZIP_PATH),
        "isolated_extract_dir": str(SMOKE_ROOT),
        "pack_dir": str(PACK_DIR),
        "required_files_present": len(missing) == 0,
        "missing_files": missing,
        "http_index_ok": index_ok,
        "http_wav_ok": wav_ok,
        "http_port": PORT,
        "index_head": index_head.strip().splitlines()[:8],
        "wav_head": wav_head.strip().splitlines()[:8],
        "case_id": case.get("case_id"),
        "safe_true_mmaudio_record_count": case.get("safe_true_mmaudio_record_count"),
        "raw_candidate_record_count": case.get("raw_candidate_record_count"),
        "cloud_ready": gate.get("decision", {}).get("readyForFridayDemoPack"),
        "claim_boundary_preserved": (
            case.get("safe_true_mmaudio_record_count") == 1
            and gate.get("decision", {}).get("trueMmaudioBatchSuccess") is False
            and gate.get("decision", {}).get("fullCandidateRankingAvailable") is False
            and gate.get("decision", {}).get("productionSloVerified") is False
            and gate.get("decision", {}).get("k6ThresholdPassVerified") is False
        ),
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("WROTE", REPORT_PATH)
    print("SMOKE_STATUS=", report["smoke_status"])
    print("HTTP_INDEX_OK=", index_ok)
    print("HTTP_WAV_OK=", wav_ok)
    print("SAFE_TRUE_MMAUDIO_RECORD_COUNT=", report["safe_true_mmaudio_record_count"])
    print("RAW_CANDIDATE_RECORD_COUNT=", report["raw_candidate_record_count"])
    print("CLOUD_READY=", report["cloud_ready"])
    print("CLAIM_BOUNDARY_PRESERVED=", report["claim_boundary_preserved"])

    if report["smoke_status"] != "pass":
        raise RuntimeError("Portable export smoke failed")


if __name__ == "__main__":
    main()