#!/usr/bin/env python3
"""Verify the W18 experiment pack without regenerating source evidence."""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path


EXPECTED = {
    "caseCount": 6,
    "candidateCount": 30,
    "winnerCount": 6,
    "failureCount": 12,
    "repairBeforeAfterCount": 6,
    "javaTaskCount": 12,
    "javaGaugeCount": 6,
    "alertRuleCount": 4,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack-dir", type=Path, required=True)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    index = args.pack_dir / "index.html"
    html_text = index.read_text(encoding="utf-8") if index.is_file() else ""
    audio_links = re.findall(r'<audio[^>]+src="([^"]+)"', html_text)
    audio_links_valid = bool(audio_links) and all((args.pack_dir / item).is_file() for item in audio_links)

    zip_valid = False
    zip_members: list[str] = []
    if args.zip.is_file():
        with zipfile.ZipFile(args.zip) as archive:
            zip_members = archive.namelist()
            zip_valid = archive.testzip() is None

    counts = manifest.get("counts", {})
    checks = {f"{key}Expected": counts.get(key) == value for key, value in EXPECTED.items()}
    checks.update(
        {
            "dashboardPanelCountGe9": counts.get("dashboardPanelCount", 0) >= 9,
            "zipValid": zip_valid,
            "indexHtml": index.is_file(),
            "audioLinks": audio_links_valid,
            "claimBoundary": bool(manifest.get("claimBoundary")),
            "zipContainsIndex": any(item.endswith("/index.html") for item in zip_members),
        }
    )
    passed = all(checks.values())
    result = {
        "gateStatus": "PASS" if passed else "FAIL",
        "checks": checks,
        "counts": counts,
        "audioLinkCount": len(audio_links),
        "zipMemberCount": len(zip_members),
        "zipValid": zip_valid,
        "indexHtml": index.is_file(),
        "audioLinks": audio_links_valid,
        "claimBoundary": bool(manifest.get("claimBoundary")),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
