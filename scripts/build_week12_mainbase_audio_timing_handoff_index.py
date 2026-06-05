#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]

BINDING_REPORT = ROOT / "artifacts/manifests/week12_audio_candidate_timing_binding_report_v2.json"
TEMPORAL_V0_REPORT = ROOT / "artifacts/manifests/week12_temporal_alignment_probe_report_v0.json"
TEMPORAL_V1_REPORT = ROOT / "artifacts/manifests/week12_temporal_alignment_probe_report_v1.json"
TEMPORAL_V1_CSV = ROOT / "artifacts/evals/week12_temporal_alignment_probe_v1.csv"
OUT = ROOT / "artifacts/manifests/week12_mainbase_audio_timing_handoff_index.json"


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"missing required file: {path.relative_to(ROOT)}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"missing required file: {path.relative_to(ROOT)}")
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        return list(csv.DictReader(f))


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def main() -> int:
    binding = load_json(BINDING_REPORT)
    temporal_v0 = load_json(TEMPORAL_V0_REPORT)
    temporal_v1 = load_json(TEMPORAL_V1_REPORT)
    rows = load_csv(TEMPORAL_V1_CSV)

    if binding.get("status") != "PASS":
        raise RuntimeError(f"binding report not PASS: {binding.get('status')}")
    if temporal_v1.get("status") != "PASS":
        raise RuntimeError(f"temporal v1 report not PASS: {temporal_v1.get('status')}")

    mode_counts = Counter(r.get("assetTimeMode") for r in rows)
    layer_counts = Counter(r.get("layer") for r in rows)
    layer_decision = Counter(f"{r.get('layer')}:{r.get('alignmentDecision')}" for r in rows)

    event_local_rows = [r for r in rows if r.get("assetTimeMode") == "event_local"]
    event_local_offsets = [
        {
            "candidateId": r.get("candidateId"),
            "sceneId": r.get("sceneId"),
            "eventId": r.get("eventId"),
            "layer": r.get("layer"),
            "label": r.get("label"),
            "audioUri": r.get("audioUri"),
            "expectedStartSec": r.get("expectedStartSec"),
            "expectedEndSec": r.get("expectedEndSec"),
            "expectedWindowDurationSec": r.get("expectedWindowDurationSec"),
            "peakLocalSec": r.get("peakSec"),
            "peakGlobalSec": r.get("peakGlobalSec"),
            "placementRequired": True,
        }
        for r in event_local_rows
    ]

    index = {
        "status": "PASS",
        "scope": "week12_mainbase_audio_candidate_timing_and_temporal_alignment",
        "candidateCount": temporal_v1.get("candidateCount"),
        "audioReadableCount": temporal_v1.get("audioReadableCount"),
        "timingBoundCount": binding.get("timingBoundCount"),
        "timingBindingMethodCounts": binding.get("bindingMethodCounts"),
        "alignmentPassCount": temporal_v1.get("alignmentPassCount"),
        "alignmentFailCount": temporal_v1.get("alignmentFailCount"),
        "assetTimeModeCounts": dict(mode_counts),
        "layerCounts": dict(layer_counts),
        "layerDecisionCounts": dict(layer_decision),
        "mainFindings": [
            "Timing binding v2 resolves the event timeline adapter issue using source_seed_id, compact event_id, layer_type, and *_seconds timing fields.",
            "Temporal alignment v0 exposed a coordinate-frame false negative for foley assets.",
            "Temporal alignment v1 distinguishes full_clip ambience from event_local foley and achieves 10/10 RMS/onset-proxy timing compatibility.",
            "Event-local foley assets require expectedStartSec placement offset in the later mixer or runtime consumer."
        ],
        "warnings": temporal_v1.get("warnings", []),
        "blockers": temporal_v1.get("blockers", []),
        "eventLocalPlacementOffsets": event_local_offsets,
        "inputs": {
            "bindingReport": rel(BINDING_REPORT),
            "temporalV0Report": rel(TEMPORAL_V0_REPORT),
            "temporalV1Report": rel(TEMPORAL_V1_REPORT),
            "temporalV1Csv": rel(TEMPORAL_V1_CSV),
        },
        "handoffForNextStep": {
            "cloudShouldConsume": rel(OUT),
            "requiredRuntimeSemantics": [
                "full_clip assets are interpreted on the scene timeline",
                "event_local assets are placed at expectedStartSec before mix/runtime visualization",
                "PASS does not mean semantic quality or human audition"
            ],
        },
        "boundaryStatement": (
            "PASS means candidates are bound to expected event windows and RMS/onset-proxy timing is compatible after coordinate-frame repair. "
            "It does not mean semantic quality, human audition, final mix readiness, or production readiness."
        ),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(index, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())