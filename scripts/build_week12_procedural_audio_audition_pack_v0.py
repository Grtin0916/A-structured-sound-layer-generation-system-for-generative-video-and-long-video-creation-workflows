#!/usr/bin/env python3
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import html
import json
import math
import os
import statistics
import subprocess
import wave
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_short_head(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
        text=True,
    ).strip()


def git_remote(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "remote", "get-url", "origin"],
        text=True,
    ).strip()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        obj = json.loads(line)
        if not isinstance(obj, dict):
            raise ValueError(f"{path}:{line_no} is not a JSON object")
        obj["_sourceLine"] = line_no
        rows.append(obj)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def read_pcm16_mono(path: Path) -> tuple[dict[str, Any], list[float]]:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        frame_count = wf.getnframes()
        frames = wf.readframes(frame_count)

    if sample_width != 2:
        raise ValueError(f"{path} sample width is {sample_width}, expected 2 bytes PCM16")

    samples: list[float] = []
    step = sample_width * channels

    for i in range(0, len(frames), step):
        if i + 2 > len(frames):
            break
        # Use the first channel if a future file is accidentally stereo.
        value = int.from_bytes(frames[i:i + 2], byteorder="little", signed=True)
        samples.append(value / 32768.0)

    meta = {
        "channels": channels,
        "sampleWidthBytes": sample_width,
        "sampleRate": sample_rate,
        "frameCount": frame_count,
        "durationSec": frame_count / sample_rate if sample_rate else 0.0,
    }
    return meta, samples


def dbfs(value: float) -> float | None:
    if value <= 0:
        return None
    return 20.0 * math.log10(value)


def audio_metrics(path: Path, expected_duration: float) -> dict[str, Any]:
    meta, samples = read_pcm16_mono(path)
    abs_samples = [abs(x) for x in samples]
    peak = max(abs_samples) if abs_samples else 0.0
    rms = math.sqrt(sum(x * x for x in samples) / len(samples)) if samples else 0.0

    clipping_ratio = (
        sum(1 for x in abs_samples if x >= 0.98) / len(abs_samples)
        if abs_samples else 0.0
    )
    silence_ratio = (
        sum(1 for x in abs_samples if x <= 0.004) / len(abs_samples)
        if abs_samples else 1.0
    )

    duration_diff = abs(float(meta["durationSec"]) - float(expected_duration))

    return {
        **meta,
        "expectedDurationSec": expected_duration,
        "durationDiffSec": round(duration_diff, 6),
        "durationMatchesExpected": duration_diff <= 0.02,
        "peakAbs": round(peak, 6),
        "peakDbfs": round(dbfs(peak), 3) if dbfs(peak) is not None else None,
        "rms": round(rms, 6),
        "rmsDbfs": round(dbfs(rms), 3) if dbfs(rms) is not None else None,
        "silenceFrameRatio": round(silence_ratio, 6),
        "clippingFrameRatio": round(clipping_ratio, 8),
        "formatOk": (
            meta["channels"] == 1
            and meta["sampleWidthBytes"] == 2
            and meta["sampleRate"] == 16000
            and duration_diff <= 0.02
        ),
    }


def waveform_polyline(samples: list[float], x0: int, y0: int, width: int, height: int, points: int = 420) -> str:
    if not samples:
        return ""

    n = len(samples)
    bucket = max(1, n // points)
    coords: list[str] = []
    mid = y0 + height / 2.0
    amp = height * 0.46

    for idx in range(0, n, bucket):
        chunk = samples[idx:idx + bucket]
        if not chunk:
            continue
        value = max(chunk, key=lambda v: abs(v))
        x = x0 + (idx / max(1, n - 1)) * width
        y = mid - value * amp
        coords.append(f"{x:.2f},{y:.2f}")

    return " ".join(coords)


def build_contact_sheet_svg(rows: list[dict[str, Any]], mainbase: Path, out_svg: Path) -> None:
    row_h = 112
    width = 1280
    height = 80 + row_h * len(rows)
    left = 300
    wave_w = 860
    wave_h = 62

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="24" y="34" font-family="monospace" font-size="20" font-weight="bold">Week12 Procedural Audio Candidates - Waveform Contact Sheet</text>',
        '<text x="24" y="58" font-family="monospace" font-size="12">Visual QA only: waveform amplitude overview, not semantic quality approval.</text>',
    ]

    for i, row in enumerate(rows):
        y = 82 + i * row_h
        wav_path = mainbase / row["candidateUri"]
        _, samples = read_pcm16_mono(wav_path)
        points = waveform_polyline(samples, left, y + 24, wave_w, wave_h)

        label = f'{row["candidateId"]} | {row["layer"]} | {row["eventLabel"]} | {row["format"]["durationSec"]}s'
        parts.append(f'<rect x="18" y="{y - 12}" width="{width - 36}" height="{row_h - 10}" fill="#f8f8f8" stroke="#ddd"/>')
        parts.append(f'<text x="28" y="{y + 8}" font-family="monospace" font-size="12">{html.escape(label)}</text>')
        parts.append(f'<text x="28" y="{y + 28}" font-family="monospace" font-size="11">slot={html.escape(row["sourceSlotId"])}</text>')
        parts.append(f'<text x="28" y="{y + 46}" font-family="monospace" font-size="11">event={html.escape(row["sourceEventId"])}</text>')
        parts.append(f'<line x1="{left}" y1="{y + 55}" x2="{left + wave_w}" y2="{y + 55}" stroke="#bbb" stroke-width="1"/>')
        parts.append(f'<polyline points="{points}" fill="none" stroke="#111" stroke-width="1.1"/>')

    parts.append("</svg>")
    out_svg.write_text("\n".join(parts), encoding="utf-8")


def build_html_index(rows: list[dict[str, Any]], qa_rows: list[dict[str, Any]], out_html: Path, mainbase: Path, contact_sheet: Path) -> None:
    html_dir = out_html.parent
    qa_by_id = {r["candidateId"]: r for r in qa_rows}
    contact_rel = os.path.relpath(contact_sheet, html_dir)

    parts = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8"/>',
        "<title>Week12 Procedural Audio Audition Pack</title>",
        "<style>",
        "body{font-family:system-ui,Arial,sans-serif;margin:24px;line-height:1.35}",
        "table{border-collapse:collapse;width:100%;font-size:13px}",
        "th,td{border:1px solid #ddd;padding:6px;vertical-align:top}",
        "th{background:#f4f4f4;text-align:left}",
        "code{font-size:12px}",
        ".warn{color:#9a5b00}.ok{color:#0a6b22}",
        "audio{width:260px}",
        "</style>",
        "</head>",
        "<body>",
        "<h1>Week12 Procedural Audio Audition Pack v0</h1>",
        "<p>Purpose: browser-based audition entry for procedural baseline WAV candidates. This is not semantic quality approval or final mix readiness.</p>",
        f'<p>Waveform contact sheet: <a href="{html.escape(contact_rel)}">{html.escape(contact_rel)}</a></p>',
        "<table>",
        "<thead><tr><th>ID</th><th>Audio</th><th>Layer / Event</th><th>Timing</th><th>QA</th><th>Path</th></tr></thead>",
        "<tbody>",
    ]

    for row in rows:
        qa = qa_by_id[row["candidateId"]]
        wav_abs = mainbase / row["candidateUri"]
        wav_rel = os.path.relpath(wav_abs, html_dir)
        timing = row.get("timing", {})

        qa_status = "OK" if qa["formatOk"] and qa["durationMatchesExpected"] else "CHECK"
        qa_class = "ok" if qa_status == "OK" else "warn"

        parts.extend([
            "<tr>",
            f'<td><code>{html.escape(row["candidateId"])}</code></td>',
            f'<td><audio controls preload="none" src="{html.escape(wav_rel)}"></audio></td>',
            f'<td>{html.escape(str(row["layer"]))}<br/><strong>{html.escape(str(row["eventLabel"]))}</strong><br/><code>{html.escape(str(row["sourceEventId"]))}</code></td>',
            f'<td>{timing.get("startSec")} → {timing.get("endSec")} s<br/>duration={timing.get("durationSec")} s</td>',
            f'<td class="{qa_class}">{qa_status}<br/>rms={qa["metrics"]["rmsDbfs"]} dBFS<br/>peak={qa["metrics"]["peakDbfs"]} dBFS<br/>silence={qa["metrics"]["silenceFrameRatio"]}</td>',
            f'<td><code>{html.escape(row["candidateUri"])}</code></td>',
            "</tr>",
        ])

    parts.extend([
        "</tbody>",
        "</table>",
        "</body>",
        "</html>",
    ])

    out_html.write_text("\n".join(parts), encoding="utf-8")


def main() -> int:
    mainbase = Path.home() / "work" / "grt_work" / "audio_engineering_repo_skeleton_v1"

    candidate_manifest_path = mainbase / "artifacts/manifests/week12_procedural_audio_candidates_manifest_v0.json"
    candidate_jsonl_path = mainbase / "artifacts/manifests/week12_procedural_audio_candidates_v0.jsonl"

    audition_dir = mainbase / "artifacts/audition/week12_procedural_baseline_v0"
    audition_dir.mkdir(parents=True, exist_ok=True)

    out_manifest = mainbase / "artifacts/manifests/week12_procedural_audio_audition_manifest_v0.json"
    out_jsonl = mainbase / "artifacts/manifests/week12_procedural_audio_audition_metrics_v0.jsonl"
    out_csv = mainbase / "artifacts/manifests/week12_procedural_audio_audition_metrics_v0.csv"
    out_svg = audition_dir / "waveform_contact_sheet.svg"
    out_html = audition_dir / "index.html"

    for path in [candidate_manifest_path, candidate_jsonl_path]:
        if not path.exists():
            raise SystemExit(f"MISSING_REQUIRED_FILE={path}")

    candidate_manifest = load_json(candidate_manifest_path)
    rows = load_jsonl(candidate_jsonl_path)

    if candidate_manifest.get("status") != "PASS":
        raise SystemExit(f"CANDIDATE_MANIFEST_NOT_PASS={candidate_manifest.get('status')}")
    if not rows:
        raise SystemExit("NO_PROCEDURAL_CANDIDATES")

    qa_rows: list[dict[str, Any]] = []

    for row in rows:
        wav_path = mainbase / row["candidateUri"]
        if not wav_path.exists():
            raise SystemExit(f"CANDIDATE_FILE_MISSING={wav_path}")

        expected_duration = float(row["format"]["durationSec"])
        metrics = audio_metrics(wav_path, expected_duration)

        qa_rows.append(
            {
                "candidateId": row["candidateId"],
                "candidateUri": row["candidateUri"],
                "sourceSlotId": row["sourceSlotId"],
                "requestId": row["requestId"],
                "sourceEventId": row["sourceEventId"],
                "sceneId": row["sceneId"],
                "blueprintId": row["blueprintId"],
                "layer": row["layer"],
                "eventLabel": row["eventLabel"],
                "timing": row["timing"],
                "format": row["format"],
                "metrics": metrics,
                "formatOk": metrics["formatOk"],
                "durationMatchesExpected": metrics["durationMatchesExpected"],
                "requiresHumanAudition": row["qualityBoundary"]["requiresHumanAudition"],
                "semanticFidelityClaimed": row["qualityBoundary"]["semanticFidelityClaimed"],
                "mixReadyClaimed": row["qualityBoundary"]["mixReadyClaimed"],
            }
        )

    write_jsonl(out_jsonl, qa_rows)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "candidateId",
                "layer",
                "eventLabel",
                "durationSec",
                "sampleRate",
                "channels",
                "rmsDbfs",
                "peakDbfs",
                "silenceFrameRatio",
                "clippingFrameRatio",
                "formatOk",
                "durationMatchesExpected",
                "candidateUri",
            ],
        )
        writer.writeheader()
        for row in qa_rows:
            m = row["metrics"]
            writer.writerow(
                {
                    "candidateId": row["candidateId"],
                    "layer": row["layer"],
                    "eventLabel": row["eventLabel"],
                    "durationSec": m["durationSec"],
                    "sampleRate": m["sampleRate"],
                    "channels": m["channels"],
                    "rmsDbfs": m["rmsDbfs"],
                    "peakDbfs": m["peakDbfs"],
                    "silenceFrameRatio": m["silenceFrameRatio"],
                    "clippingFrameRatio": m["clippingFrameRatio"],
                    "formatOk": row["formatOk"],
                    "durationMatchesExpected": row["durationMatchesExpected"],
                    "candidateUri": row["candidateUri"],
                }
            )

    build_contact_sheet_svg(rows, mainbase, out_svg)
    build_html_index(rows, qa_rows, out_html, mainbase, out_svg)

    all_format_ok = all(r["formatOk"] for r in qa_rows)
    all_duration_ok = all(r["durationMatchesExpected"] for r in qa_rows)
    all_require_audition = all(r["requiresHumanAudition"] for r in qa_rows)
    any_semantic_claim = any(r["semanticFidelityClaimed"] for r in qa_rows)
    any_mix_claim = any(r["mixReadyClaimed"] for r in qa_rows)

    rms_values = [r["metrics"]["rmsDbfs"] for r in qa_rows if r["metrics"]["rmsDbfs"] is not None]
    silence_values = [r["metrics"]["silenceFrameRatio"] for r in qa_rows]

    status = "PASS" if all_format_ok and all_duration_ok and all_require_audition and not any_semantic_claim and not any_mix_claim else "WARN"

    manifest = {
        "schemaVersion": "week12.procedural-audio-audition-pack-manifest.v0",
        "generatedAt": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": status,
        "candidateCount": len(rows),
        "qaRecordCount": len(qa_rows),
        "allFormatOk": all_format_ok,
        "allDurationMatchesExpected": all_duration_ok,
        "allRequireHumanAudition": all_require_audition,
        "semanticFidelityClaimedAny": any_semantic_claim,
        "mixReadyClaimedAny": any_mix_claim,
        "aggregateMetrics": {
            "rmsDbfsMin": round(min(rms_values), 3) if rms_values else None,
            "rmsDbfsMax": round(max(rms_values), 3) if rms_values else None,
            "rmsDbfsMedian": round(statistics.median(rms_values), 3) if rms_values else None,
            "silenceFrameRatioMax": round(max(silence_values), 6) if silence_values else None,
        },
        "mainbase": {
            "repo": git_remote(mainbase),
            "commit": git_short_head(mainbase),
            "candidateManifestPath": "artifacts/manifests/week12_procedural_audio_candidates_manifest_v0.json",
            "candidateManifestSha256": sha256_file(candidate_manifest_path),
            "candidateJsonlPath": "artifacts/manifests/week12_procedural_audio_candidates_v0.jsonl",
            "candidateJsonlSha256": sha256_file(candidate_jsonl_path),
        },
        "outputs": {
            "auditionHtml": "artifacts/audition/week12_procedural_baseline_v0/index.html",
            "auditionHtmlSha256": sha256_file(out_html),
            "waveformContactSheetSvg": "artifacts/audition/week12_procedural_baseline_v0/waveform_contact_sheet.svg",
            "waveformContactSheetSvgSha256": sha256_file(out_svg),
            "metricsJsonl": "artifacts/manifests/week12_procedural_audio_audition_metrics_v0.jsonl",
            "metricsJsonlSha256": sha256_file(out_jsonl),
            "metricsCsv": "artifacts/manifests/week12_procedural_audio_audition_metrics_v0.csv",
            "metricsCsvSha256": sha256_file(out_csv),
        },
        "doesNotClaim": [
            "semantic audio quality",
            "human audition has passed",
            "final mix readiness",
            "text-to-audio model inference",
            "production asset storage"
        ],
    }

    out_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0 if status == "PASS" else 4


if __name__ == "__main__":
    raise SystemExit(main())