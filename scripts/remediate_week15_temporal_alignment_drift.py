#!/usr/bin/env python3
"""
Remediate Week15 temporal alignment drift by trimming leading silence/low-energy prefix
for event_local candidates that failed or nearly missed onset alignment.

This script:
- reads week15_temporal_alignment_summary.json
- reads week15_temporal_alignment_input_index.json
- trims only event_local rows with FAIL_* or WARN_NEAR_MISS
- writes remediated WAV files
- writes a remediated input index for the existing scorer

It does not change original artifacts and does not claim semantic quality.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def trim_wav_prefix(src: Path, dst: Path, trim_sec: float) -> dict[str, Any]:
    if trim_sec < 0:
        raise ValueError(f"trim_sec must be non-negative, got {trim_sec}")

    with wave.open(str(src), "rb") as wf:
        params = wf.getparams()
        nch = wf.getnchannels()
        sw = wf.getsampwidth()
        sr = wf.getframerate()
        nframes = wf.getnframes()
        raw = wf.readframes(nframes)

    bytes_per_frame = nch * sw
    trim_frames = int(round(trim_sec * sr))
    trim_frames = max(0, min(trim_frames, max(0, nframes - 1)))
    trim_bytes = trim_frames * bytes_per_frame
    trimmed = raw[trim_bytes:]

    if not trimmed:
        raise ValueError(f"trim produced empty wav: {src}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(dst), "wb") as wf:
        wf.setnchannels(nch)
        wf.setsampwidth(sw)
        wf.setframerate(sr)
        wf.writeframes(trimmed)

    return {
        "srcFrames": nframes,
        "dstFrames": len(trimmed) // bytes_per_frame,
        "sampleRate": sr,
        "channels": nch,
        "sampleWidthBytes": sw,
        "trimSec": trim_frames / sr,
        "srcDurationSec": nframes / sr,
        "dstDurationSec": (len(trimmed) // bytes_per_frame) / sr,
    }


def should_remediate(row: dict[str, Any]) -> bool:
    status = str(row.get("alignmentStatus", ""))
    return row.get("assetTimeMode") == "event_local" and (
        status.startswith("FAIL") or status == "WARN_NEAR_MISS"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mainbase", default=".")
    ap.add_argument("--input-index", default="artifacts/evals/week15_temporal_alignment_input_index.json")
    ap.add_argument("--score-summary", default="artifacts/evals/week15_temporal_alignment_summary.json")
    ap.add_argument("--out-index", default="artifacts/evals/week15_temporal_alignment_remediated_input_index.json")
    ap.add_argument("--plan-out", default="artifacts/evals/week15_temporal_alignment_remediation_plan.json")
    ap.add_argument("--audio-out-dir", default="artifacts/audio_candidates/week15_temporal_alignment_remediated")
    ap.add_argument("--pre-roll-sec", type=float, default=0.02)
    args = ap.parse_args()

    root = Path(args.mainbase).expanduser().resolve()
    input_index_path = root / args.input_index
    score_summary_path = root / args.score_summary
    out_index_path = root / args.out_index
    plan_out_path = root / args.plan_out
    audio_out_dir = root / args.audio_out_dir

    input_index = read_json(input_index_path)
    score_summary = read_json(score_summary_path)

    score_rows = score_summary.get("rows", [])
    score_by_id = {str(r.get("candidateId")): r for r in score_rows if r.get("candidateId")}

    blockers: list[str] = []
    actions: list[dict[str, Any]] = []
    remediated_inputs: list[dict[str, Any]] = []

    for item in input_index.get("evalInputs", []):
        new_item = dict(item)
        cid = str(item.get("candidateId"))
        score_row = score_by_id.get(cid, {})

        if should_remediate(score_row):
            local_onset = score_row.get("localOnsetSec")
            audio_uri = item.get("audioUri")
            if local_onset is None or not math.isfinite(float(local_onset)):
                blockers.append(f"{cid}: cannot remediate without finite localOnsetSec")
            elif not audio_uri:
                blockers.append(f"{cid}: cannot remediate without audioUri")
            else:
                src = root / str(audio_uri)
                dst_rel = str(Path(args.audio_out_dir) / f"{cid}_trimmed_preroll_{int(args.pre_roll_sec * 1000):02d}ms.wav")
                dst = root / dst_rel

                try:
                    trim_sec = max(0.0, float(local_onset) - args.pre_roll_sec)
                    trim_meta = trim_wav_prefix(src, dst, trim_sec)
                    new_item["audioUri"] = dst_rel
                    new_item["remediation"] = {
                        "type": "TRIM_LEADING_LOW_ENERGY_PREFIX",
                        "sourceAudioUri": audio_uri,
                        "remediatedAudioUri": dst_rel,
                        "sourceAlignmentStatus": score_row.get("alignmentStatus"),
                        "sourceLocalOnsetSec": local_onset,
                        "preRollSec": args.pre_roll_sec,
                        "trimMeta": trim_meta,
                    }
                    actions.append({
                        "candidateId": cid,
                        "action": "trimmed",
                        "sourceAlignmentStatus": score_row.get("alignmentStatus"),
                        "sourceOnsetDeltaSec": score_row.get("onsetDeltaSec"),
                        "sourceLocalOnsetSec": local_onset,
                        "sourceAudioUri": audio_uri,
                        "remediatedAudioUri": dst_rel,
                        "trimSec": trim_meta["trimSec"],
                    })
                except Exception as exc:
                    blockers.append(f"{cid}: trim failed: {exc}")
        else:
            new_item["remediation"] = {
                "type": "UNCHANGED",
                "sourceAlignmentStatus": score_row.get("alignmentStatus"),
            }

        remediated_inputs.append(new_item)

    remediated_index = dict(input_index)
    remediated_index["schemaVersion"] = "week15.temporal_alignment_remediated_input_index.v1"
    remediated_index["generatedAtUtc"] = datetime.now(timezone.utc).isoformat()
    remediated_index["status"] = "PASS" if not blockers else "FAIL"
    remediated_index["purpose"] = (
        "Remediated temporal alignment input index with trimmed event_local candidates "
        "for onset/energy proxy rescoring."
    )
    remediated_index["sourceInputIndex"] = str(input_index_path)
    remediated_index["sourceScoreSummary"] = str(score_summary_path)
    remediated_index["summary"] = {
        "candidateCount": len(remediated_inputs),
        "remediatedCandidateCount": len(actions),
        "blockerCount": len(blockers),
        "preRollSec": args.pre_roll_sec,
    }
    remediated_index["evalInputs"] = remediated_inputs
    remediated_index["blockers"] = blockers
    remediated_index["boundary"] = [
        "candidate_asset_timeline_remediation_only",
        "does_not_modify_original_audio",
        "does_not_score_semantic_audio_quality",
        "does_not_claim_human_audition_passed",
        "does_not_claim_final_mix_readiness",
    ]

    plan = {
        "schemaVersion": "week15.temporal_alignment_remediation_plan.v1",
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if not blockers else "FAIL",
        "reason": (
            "Original placement is correct, but some event_local candidate assets have late local onset. "
            "Trim leading low-energy prefix and rescore without modifying originals."
        ),
        "sourceScoreStatus": score_summary.get("status"),
        "sourceScoreSummary": score_summary.get("summary", {}),
        "actions": actions,
        "outputs": {
            "remediatedInputIndex": str(out_index_path),
            "audioOutDir": str(audio_out_dir),
        },
        "blockers": blockers,
    }

    write_json(out_index_path, remediated_index)
    write_json(plan_out_path, plan)

    print(json.dumps({
        "status": plan["status"],
        "remediatedCandidateCount": len(actions),
        "outputs": plan["outputs"],
        "blockers": blockers,
        "actions": actions,
    }, indent=2, ensure_ascii=False))

    return 0 if not blockers else 2


if __name__ == "__main__":
    raise SystemExit(main())