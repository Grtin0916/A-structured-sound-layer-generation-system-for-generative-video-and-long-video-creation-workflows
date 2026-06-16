#!/usr/bin/env python3
from __future__ import annotations

import html
import json
import math
import os
import re
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(".")
OUT_JSON = ROOT / "artifacts/reviews/week15_temporal_alignment_semantic_quality_review_v0.json"
OUT_HTML = ROOT / "artifacts/reviews/week15_temporal_alignment_semantic_quality_review_v0.html"

INPUTS = {
    "manualReviewDecision": ROOT / "artifacts/reviews/week15_temporal_alignment_manual_review_decision.json",
    "manualReviewPacket": ROOT / "artifacts/reviews/week15_temporal_alignment_manual_review_packet.json",
    "waveformRmsIndex": ROOT / "artifacts/evals/week15_temporal_alignment_waveform_rms_index.json",
    "alignmentSummary": ROOT / "artifacts/evals/week15_temporal_alignment_summary.json",
    "remediatedSummary": ROOT / "artifacts/evals/week15_temporal_alignment_remediated_summary.json",
    "regressionGate": ROOT / "artifacts/evals/week15_temporal_alignment_regression_gate.json",
}

CANDIDATE_RE = re.compile(r"procedural_v0_\d{4}")
AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac"}


def load_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def walk_values(obj: Any):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from walk_values(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from walk_values(v)
    else:
        yield obj


def collect_candidate_ids(*objects: Any) -> list[str]:
    found: set[str] = set()
    for obj in objects:
        for v in walk_values(obj):
            if isinstance(v, str):
                found.update(CANDIDATE_RE.findall(v))
    # 0004/0010 是 Week15 已暴露的 drift/remediation 样本；若输入中未显式出现，保底纳入。
    found.update({"procedural_v0_0004", "procedural_v0_0010"})
    return sorted(found)


def candidate_numeric_id(candidate_id: str) -> str | None:
    m = re.search(r"(\d{4})$", candidate_id)
    return m.group(1) if m else None


def normalize_path(value: str) -> Path | None:
    if not isinstance(value, str):
        return None
    if not any(value.lower().endswith(s) for s in AUDIO_SUFFIXES):
        return None
    p = Path(value)
    if not p.is_absolute():
        p = ROOT / p
    return p


def collect_audio_paths(candidate_id: str, *objects: Any) -> dict[str, str]:
    paths: dict[str, str] = {}
    all_strings: list[str] = []

    # 1) Prefer paths already referenced by upstream JSON artifacts.
    for obj in objects:
        for v in walk_values(obj):
            if isinstance(v, str) and candidate_id in v:
                all_strings.append(v)

    # 2) Backfill by scanning local artifact audio roots.
    # This avoids a silent failure where review JSON exists but omits original audio paths.
    scan_roots = [
        ROOT / "artifacts/audio_candidates",
        ROOT / "artifacts/audio",
        ROOT / "artifacts/wav",
    ]
    for root in scan_roots:
        if not root.exists():
            continue
        numeric_id = candidate_numeric_id(candidate_id)
        for fp in root.rglob("*"):
            if not fp.is_file() or fp.suffix.lower() not in AUDIO_SUFFIXES:
                continue
            name = fp.name
            matched = candidate_id in name
            if numeric_id is not None:
                matched = matched or name.startswith(f"{numeric_id}_") or f"_{numeric_id}_" in name
            if matched:
                all_strings.append(str(fp))

    for value in all_strings:
        p = normalize_path(value)
        if p is None or not p.exists():
            continue

        lowered = str(p).lower()
        if "remediat" in lowered or "trimmed" in lowered:
            key = "remediated"
        elif "original" in lowered or "candidate" in lowered:
            key = "original"
        else:
            key = "audio"

        rel = str(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p)
        # Preserve first path per semantic slot; deterministic enough after sorted scan fallback.
        paths.setdefault(key, rel)

    return dict(sorted(paths.items()))


def read_wav_with_stdlib(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sr = wf.getframerate()
        width = wf.getsampwidth()
        n = wf.getnframes()
        raw = wf.readframes(n)

    if width == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        data = (data - 128.0) / 128.0
    elif width == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif width == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported wav sample width: {width}")

    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data, sr


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf  # type: ignore

        data, sr = sf.read(str(path), always_2d=True)
        data = data.astype(np.float32).mean(axis=1)
        return data, int(sr)
    except Exception:
        return read_wav_with_stdlib(path)


def frame_rms(y: np.ndarray, frame_length: int = 2048, hop_length: int = 512) -> np.ndarray:
    if y.size == 0:
        return np.array([], dtype=np.float32)
    if y.size < frame_length:
        y = np.pad(y, (0, frame_length - y.size))
    frames = []
    for start in range(0, max(1, y.size - frame_length + 1), hop_length):
        frame = y[start : start + frame_length]
        frames.append(float(np.sqrt(np.mean(frame * frame))))
    return np.asarray(frames, dtype=np.float32)


def zero_crossing_rate(y: np.ndarray) -> float:
    if y.size < 2:
        return 0.0
    signs = np.signbit(y)
    return float(np.mean(signs[1:] != signs[:-1]))


def classify_audio_risks(metrics: dict[str, Any]) -> list[str]:
    if metrics.get("status") != "OK":
        return ["METRICS_UNAVAILABLE"]

    risks: list[str] = []
    low_energy = metrics.get("lowEnergyFrameRatio")
    rms = metrics.get("globalRms")
    peak = metrics.get("peakAbs")
    clip = metrics.get("clipRatioAbsGe0_999")
    duration = metrics.get("durationSec")

    if isinstance(low_energy, (int, float)) and low_energy >= 0.60:
        risks.append("HIGH_LOW_ENERGY_RATIO_REVIEW_EVENT_PRESENCE")
    if isinstance(rms, (int, float)) and rms < 0.005:
        risks.append("VERY_LOW_GLOBAL_RMS")
    if isinstance(peak, (int, float)) and peak < 0.03:
        risks.append("LOW_PEAK_AMPLITUDE")
    if isinstance(clip, (int, float)) and clip > 0.0:
        risks.append("POSSIBLE_CLIPPING")
    if isinstance(duration, (int, float)) and duration < 1.0:
        risks.append("VERY_SHORT_DURATION")
    if not risks:
        risks.append("NO_LIGHTWEIGHT_SIGNAL_RISK_DETECTED")
    return risks


def audio_metrics(path_str: str) -> dict[str, Any]:
    p = ROOT / path_str
    try:
        y, sr = read_audio(p)
        if y.size == 0 or sr <= 0:
            return {"status": "INVALID_AUDIO_EMPTY", "path": path_str, "riskFlags": ["METRICS_UNAVAILABLE"]}

        y = np.nan_to_num(y.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        abs_y = np.abs(y)
        rms_frames = frame_rms(y)
        global_rms = float(np.sqrt(np.mean(y * y)))
        silence_threshold = max(1e-4, global_rms * 0.10)
        low_energy_ratio = float(np.mean(rms_frames < silence_threshold)) if rms_frames.size else None

        metrics = {
            "status": "OK",
            "path": path_str,
            "sampleRate": int(sr),
            "durationSec": round(float(y.size / sr), 6),
            "globalRms": round(global_rms, 8),
            "peakAbs": round(float(abs_y.max()), 8),
            "clipRatioAbsGe0_999": round(float(np.mean(abs_y >= 0.999)), 8),
            "lowEnergyFrameRatio": None if low_energy_ratio is None else round(low_energy_ratio, 8),
            "zeroCrossingRate": round(zero_crossing_rate(y), 8),
        }
        metrics["riskFlags"] = classify_audio_risks(metrics)
        return metrics
    except Exception as e:
        return {"status": "AUDIO_METRICS_FAILED", "path": path_str, "error": repr(e), "riskFlags": ["METRICS_UNAVAILABLE"]}


def extract_gate_status(regression_gate: Any) -> str:
    text = json.dumps(regression_gate, ensure_ascii=False).lower() if regression_gate is not None else ""
    if "pass" in text and "failcount" in text:
        return "PASS_OR_REPORTED_PASS"
    if regression_gate is None:
        return "MISSING"
    return "PRESENT_CHECK_MANUALLY"


def stable_generated_at_utc() -> str:
    env_value = os.environ.get("SEMANTIC_REVIEW_GENERATED_AT_UTC")
    if env_value:
        return env_value

    if OUT_JSON.exists():
        try:
            old = json.loads(OUT_JSON.read_text(encoding="utf-8"))
            old_value = old.get("generatedAtUtc")
            if isinstance(old_value, str) and old_value:
                return old_value
        except Exception:
            pass

    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    loaded = {name: load_json(path) for name, path in INPUTS.items()}
    missing_inputs = [name for name, path in INPUTS.items() if not path.exists()]

    candidate_ids = collect_candidate_ids(*loaded.values())
    candidates = []
    for cid in candidate_ids:
        audio_paths = collect_audio_paths(cid, *loaded.values())
        metrics = {kind: audio_metrics(path) for kind, path in audio_paths.items()}
        candidate_risks = sorted({
            flag
            for metric in metrics.values()
            for flag in metric.get("riskFlags", [])
        }) or ["NO_AUDIO_PATHS_RESOLVED"]

        candidates.append(
            {
                "candidateId": cid,
                "audioPaths": audio_paths,
                "lightweightAudioMetrics": metrics,
                "riskFlags": candidate_risks,
                "reviewChecklist": {
                    "auditionPerformed": False,
                    "keyEventPreservedAfterRemediation": "UNREVIEWED",
                    "semanticMatchToEvent": "UNREVIEWED",
                    "obviousNoiseOrArtifact": "UNREVIEWED",
                    "obviousClipping": "UNREVIEWED",
                    "longSilenceOrMissingEvent": "UNREVIEWED",
                    "acceptableForNextFailureTaxonomy": "UNREVIEWED",
                },
                "reviewQuestions": [
                    "修复版是否仍保留关键事件听感？",
                    "裁剪是否误删事件主体或攻击段？",
                    "声音语义是否与对应 event/layer 匹配？",
                    "是否存在明显静音、削波、爆音、噪声或尾部截断？",
                    "是否可以进入下一阶段 failure taxonomy，而不是直接声明 final mix ready？",
                ],
            }
        )

    quality_status = "SEMANTIC_REVIEW_READY"
    if "manualReviewDecision" in missing_inputs or "manualReviewPacket" in missing_inputs:
        quality_status = "BLOCKED_MISSING_MANUAL_REVIEW_INPUT"
    elif not candidates:
        quality_status = "BLOCKED_NO_CANDIDATES"

    report = {
        "schemaVersion": "week15.semantic-quality-review.v0",
        "generatedAtUtc": stable_generated_at_utc(),
        "sourceScope": "local Mainbase artifacts only",
        "qualityGateLiteStatus": quality_status,
        "humanReviewStatus": "HUMAN_REVIEW_PARTIAL",
        "auditionStatus": "NOT_PERFORMED",
        "semanticQualityReviewStatus": "NOT_PERFORMED",
        "finalMixReadiness": "NOT_CLAIMED",
        "temporalRegressionGateStatus": extract_gate_status(loaded.get("regressionGate")),
        "inputs": {
            name: {
                "path": str(path),
                "exists": path.exists(),
            }
            for name, path in INPUTS.items()
        },
        "missingInputs": missing_inputs,
        "candidates": candidates,
        "allowedClaims": [
            "Temporal alignment signal/visual evidence can be used as review input if the upstream artifacts exist.",
            "This packet is ready for human audition and semantic-quality checklist filling.",
        ],
        "blockedClaims": [
            "Do not claim HUMAN_REVIEW_PASS before real audition is performed.",
            "Do not claim semantic audio quality PASS from RMS/ZCR/visual evidence alone.",
            "Do not claim final mix readiness.",
            "Do not claim production/live dashboard/SLO from this local artifact.",
        ],
        "nextConsumerHint": {
            "mainbaseNext": "Fill audition and semantic checklist after listening to original/remediated samples.",
            "cloudNext": "Consume this JSON into semantic-quality platform index and Prometheus text metrics.",
            "javaNext": "Expose review-state contract only after Mainbase JSON is stable enough.",
        },
    }

    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    rows = []
    for c in candidates:
        metrics = c["lightweightAudioMetrics"]
        rows.append(
            "<tr>"
            f"<td>{html.escape(c['candidateId'])}</td>"
            f"<td>{html.escape(json.dumps(c['audioPaths'], ensure_ascii=False))}</td>"
            f"<td><pre>{html.escape(json.dumps(metrics, ensure_ascii=False, indent=2))}</pre></td>"
            f"<td>{html.escape(c['reviewChecklist']['semanticMatchToEvent'])}</td>"
            "</tr>"
        )

    OUT_HTML.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Week15 Temporal Alignment Semantic Quality Review V0</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; line-height: 1.45; }}
table {{ border-collapse: collapse; width: 100%; }}
td, th {{ border: 1px solid #ccc; padding: 8px; vertical-align: top; }}
pre {{ white-space: pre-wrap; margin: 0; }}
.status {{ font-weight: bold; }}
.blocked {{ color: #8a0000; }}
</style>
</head>
<body>
<h1>Week15 Temporal Alignment Semantic Quality Review V0</h1>
<p class="status">qualityGateLiteStatus: {html.escape(report["qualityGateLiteStatus"])}</p>
<p>humanReviewStatus: {html.escape(report["humanReviewStatus"])}</p>
<p>auditionStatus: {html.escape(report["auditionStatus"])}</p>
<p>semanticQualityReviewStatus: {html.escape(report["semanticQualityReviewStatus"])}</p>
<p class="blocked">finalMixReadiness: {html.escape(report["finalMixReadiness"])}</p>
<h2>Blocked claims</h2>
<ul>{"".join(f"<li>{html.escape(x)}</li>" for x in report["blockedClaims"])}</ul>
<h2>Candidates</h2>
<table>
<thead><tr><th>candidateId</th><th>audio paths</th><th>lightweight metrics</th><th>semantic match</th></tr></thead>
<tbody>{"".join(rows)}</tbody>
</table>
</body>
</html>
""",
        encoding="utf-8",
    )

    print(json.dumps(
        {
            "qualityGateLiteStatus": report["qualityGateLiteStatus"],
            "humanReviewStatus": report["humanReviewStatus"],
            "auditionStatus": report["auditionStatus"],
            "semanticQualityReviewStatus": report["semanticQualityReviewStatus"],
            "candidateCount": len(candidates),
            "missingInputs": missing_inputs,
            "json": str(OUT_JSON),
            "html": str(OUT_HTML),
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0 if quality_status == "SEMANTIC_REVIEW_READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
