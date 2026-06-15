#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CANDIDATES = ["procedural_v0_0004", "procedural_v0_0010"]
VALID = {"PASS", "FAIL", "NOT_PERFORMED"}


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"missing required json: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"json root must be object: {path}")
    return data


def env_required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"Missing required env var: {name}")
    return value


def env_enum(name: str) -> str:
    value = env_required(name).upper()
    if value not in VALID:
        raise SystemExit(f"{name} must be one of {sorted(VALID)}, got {value!r}")
    return value


def main() -> int:
    packet_path = Path("artifacts/reviews/week15_temporal_alignment_manual_review_packet.json")
    packet = load_json(packet_path)

    if packet.get("status") != "REVIEW_READY":
        raise SystemExit(f"manual review packet must be REVIEW_READY, got {packet.get('status')}")

    records = packet.get("records")
    if not isinstance(records, list) or len(records) != 2:
        raise SystemExit("manual review packet must contain exactly 2 records")

    packet_by_id = {item.get("candidateId"): item for item in records if isinstance(item, dict)}
    decisions = []

    for cid in CANDIDATES:
        if cid not in packet_by_id:
            raise SystemExit(f"missing candidate in packet: {cid}")

        suffix = cid.split("_")[-1]
        visual = env_enum(f"VISUAL_{suffix}")
        audition = env_enum(f"AUDITION_{suffix}")
        note = os.environ.get(f"NOTE_{suffix}", "").strip()

        item = packet_by_id[cid]

        for p in [item["figure"], item["originalAudio"], item["remediatedAudio"]]:
            if not Path(p).exists():
                raise SystemExit(f"referenced file missing for {cid}: {p}")

        decisions.append(
            {
                "candidateId": cid,
                "visualInspection": visual,
                "audition": audition,
                "note": note,
                "figure": item["figure"],
                "originalAudio": item["originalAudio"],
                "remediatedAudio": item["remediatedAudio"],
                "durationTrimSec": item["durationTrimSec"],
                "onsetProxyDeltaSec": item["onsetProxyDeltaSec"],
            }
        )

    all_visual_pass = all(x["visualInspection"] == "PASS" for x in decisions)
    all_audition_pass = all(x["audition"] == "PASS" for x in decisions)
    any_fail = any(x["visualInspection"] == "FAIL" or x["audition"] == "FAIL" for x in decisions)
    any_not_performed = any(x["visualInspection"] == "NOT_PERFORMED" or x["audition"] == "NOT_PERFORMED" for x in decisions)

    if all_visual_pass and all_audition_pass:
        status = "HUMAN_REVIEW_PASS"
    elif any_fail:
        status = "HUMAN_REVIEW_FAIL"
    elif any_not_performed:
        status = "HUMAN_REVIEW_PARTIAL"
    else:
        status = "HUMAN_REVIEW_RECORDED"

    decision = {
        "schemaVersion": "week15.temporal_alignment_manual_review_decision.v1",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "inputPacket": str(packet_path),
        "allowedClaim": packet.get("allowedClaim"),
        "blockedClaims": packet.get("blockedClaims"),
        "decisions": decisions,
        "reviewDecision": {
            "humanVisualInspection": "PASS" if all_visual_pass else "FAIL" if any(x["visualInspection"] == "FAIL" for x in decisions) else "PARTIAL",
            "humanAudition": "PASS" if all_audition_pass else "FAIL" if any(x["audition"] == "FAIL" for x in decisions) else "PARTIAL",
            "semanticQualityReview": "NOT_PERFORMED",
            "finalMixReadiness": "NOT_CLAIMED",
        },
        "boundary": [
            "manual_review_decision_only",
            "visual_signal_review_supported_by_uploaded_audio_analysis",
            "semantic_quality_review_not_performed",
            "final_mix_readiness_not_claimed",
            "does_not_claim_live_grafana_import",
            "does_not_claim_production_slo",
        ],
    }

    out = Path("artifacts/reviews/week15_temporal_alignment_manual_review_decision.json")
    out.write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(decision, ensure_ascii=False, indent=2))
    print(f"WROTE_DECISION={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
