#!/usr/bin/env python3
from __future__ import annotations

import html
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"missing required json: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"json root must be object: {path}")
    return data


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def rel_from_review(path: Path) -> str:
    review_dir = Path("artifacts/reviews")
    return os.path.relpath(path, start=review_dir).replace(os.sep, "/")


def main() -> int:
    mainbase_root = Path.cwd()
    cloud_root = Path(
        os.environ.get(
            "CLOUD_REPO",
            str(Path.home() / "work" / "grt_work" / "ai-job-platform-cloud"),
        )
    )

    waveform_index_path = mainbase_root / "artifacts/evals/week15_temporal_alignment_waveform_rms_index.json"
    cloud_gate_path = cloud_root / "loadtest/reports/week15_temporal_alignment_evidence_gate.json"

    waveform_index = load_json(waveform_index_path)
    cloud_gate = load_json(cloud_gate_path)

    require(waveform_index.get("status") == "PASS", "waveform/RMS index must PASS")
    require(cloud_gate.get("status") == "PASS", "Cloud evidence gate must PASS")

    candidates = waveform_index.get("candidates")
    require(isinstance(candidates, list), "candidates must be array")
    require(len(candidates) == 2, f"expected 2 candidates, got {len(candidates)}")

    allowed_claim = cloud_gate.get("allowedClaim")
    blocked_claims = cloud_gate.get("blockedClaims")
    require(isinstance(allowed_claim, str) and allowed_claim, "allowedClaim missing")
    require(isinstance(blocked_claims, list) and blocked_claims, "blockedClaims missing")

    packet_records: list[dict[str, Any]] = []

    for item in candidates:
        cid = item["candidateId"]
        figure = Path(item["figure"])
        original_audio = Path(item["originalAudio"])
        remediated_audio = Path(item["remediatedAudio"])

        require(figure.exists(), f"missing figure for {cid}: {figure}")
        require(figure.stat().st_size > 0, f"empty figure for {cid}: {figure}")
        require(original_audio.exists(), f"missing original audio for {cid}: {original_audio}")
        require(remediated_audio.exists(), f"missing remediated audio for {cid}: {remediated_audio}")

        duration_trim_sec = round(float(item["originalDurationSec"]) - float(item["remediatedDurationSec"]), 6)
        require(duration_trim_sec > 0.0, f"{cid} duration trim must be positive")
        require(float(item["onsetProxyDeltaSec"]) == 0.0, f"{cid} onset proxy delta must be 0.0")

        packet_records.append(
            {
                "candidateId": cid,
                "figure": str(figure),
                "figureRelFromReviewHtml": rel_from_review(figure),
                "originalAudio": str(original_audio),
                "originalAudioRelFromReviewHtml": rel_from_review(original_audio),
                "remediatedAudio": str(remediated_audio),
                "remediatedAudioRelFromReviewHtml": rel_from_review(remediated_audio),
                "originalDurationSec": item["originalDurationSec"],
                "remediatedDurationSec": item["remediatedDurationSec"],
                "durationTrimSec": duration_trim_sec,
                "originalOnsetProxySec": item["originalOnsetProxySec"],
                "remediatedOnsetProxySec": item["remediatedOnsetProxySec"],
                "onsetProxyDeltaSec": item["onsetProxyDeltaSec"],
                "reviewChecklist": [
                    "Inspect whether the remediated waveform removes redundant leading/trailing low-value duration without cutting the salient event.",
                    "Listen to original and remediated audio before any human-audition claim.",
                    "Do not treat unchanged RMS onset proxy as onset-shift evidence.",
                    "Do not claim semantic quality or final mix readiness from this packet alone.",
                ],
            }
        )

    packet = {
        "schemaVersion": "week15.temporal_alignment_manual_review_packet.v1",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "status": "REVIEW_READY",
        "purpose": "Provide a human-facing packet for visual and audio inspection of Week15 temporal alignment remediation evidence.",
        "inputs": {
            "mainbaseWaveformRmsIndex": str(waveform_index_path),
            "cloudEvidenceGate": str(cloud_gate_path),
        },
        "sourceHeads": {
            "mainbaseWaveformRmsCommitHead": "3a5b12a",
            "cloudEvidenceGateCommitHead": "e7fd36b",
        },
        "allowedClaim": allowed_claim,
        "blockedClaims": blocked_claims,
        "records": packet_records,
        "reviewDecision": {
            "humanVisualInspection": "NOT_PERFORMED",
            "humanAudition": "NOT_PERFORMED",
            "semanticQualityReview": "NOT_PERFORMED",
            "finalMixReadiness": "NOT_CLAIMED",
        },
        "boundary": [
            "manual_review_packet_only",
            "does_not_claim_human_audition_passed",
            "does_not_claim_semantic_audio_quality",
            "does_not_claim_final_mix_readiness",
        ],
    }

    out_json = Path("artifacts/reviews/week15_temporal_alignment_manual_review_packet.json")
    out_html = Path("artifacts/reviews/week15_temporal_alignment_manual_review_packet.html")

    out_json.write_text(json.dumps(packet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    rows = []
    for record in packet_records:
        rows.append(
            f"""
            <section class="candidate">
              <h2>{html.escape(record["candidateId"])}</h2>
              <p><b>Duration trim:</b> {record["durationTrimSec"]} s |
                 <b>Onset proxy delta:</b> {record["onsetProxyDeltaSec"]} s</p>
              <img src="{html.escape(record["figureRelFromReviewHtml"])}" alt="{html.escape(record["candidateId"])} waveform RMS original vs remediated" />
              <div class="audio-row">
                <div>
                  <h3>Original audio</h3>
                  <p>{html.escape(record["originalAudio"])}</p>
                  <audio controls src="{html.escape(record["originalAudioRelFromReviewHtml"])}"></audio>
                </div>
                <div>
                  <h3>Remediated audio</h3>
                  <p>{html.escape(record["remediatedAudio"])}</p>
                  <audio controls src="{html.escape(record["remediatedAudioRelFromReviewHtml"])}"></audio>
                </div>
              </div>
              <ul>
                {''.join(f"<li>{html.escape(x)}</li>" for x in record["reviewChecklist"])}
              </ul>
            </section>
            """
        )

    blocked_html = "".join(f"<li>{html.escape(str(x))}</li>" for x in blocked_claims)

    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Week15 Temporal Alignment Manual Review Packet</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; line-height: 1.45; }}
    .claim {{ padding: 12px; border: 1px solid #aaa; margin-bottom: 16px; }}
    .candidate {{ border-top: 2px solid #333; padding-top: 16px; margin-top: 24px; }}
    img {{ max-width: 100%; border: 1px solid #ccc; }}
    .audio-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    audio {{ width: 100%; }}
    code {{ background: #eee; padding: 2px 4px; }}
  </style>
</head>
<body>
  <h1>Week15 Temporal Alignment Manual Review Packet</h1>
  <div class="claim">
    <h2>Allowed claim</h2>
    <p>{html.escape(allowed_claim)}</p>
    <h2>Blocked claims</h2>
    <ul>{blocked_html}</ul>
    <p><b>Review status:</b> REVIEW_READY. Human visual inspection and audition are not yet performed.</p>
  </div>
  {''.join(rows)}
</body>
</html>
"""

    out_html.write_text(page, encoding="utf-8")

    print(json.dumps(packet, ensure_ascii=False, indent=2))
    print(f"WROTE_JSON={out_json}")
    print(f"WROTE_HTML={out_html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
