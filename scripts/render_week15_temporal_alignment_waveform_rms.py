#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    import soundfile as sf
except Exception as exc:
    raise SystemExit("Missing dependency: soundfile") from exc

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:
    raise SystemExit("Missing dependency: matplotlib") from exc


CANDIDATES = ["procedural_v0_0004", "procedural_v0_0010"]
OUT_DIR = Path("artifacts/figures/week15_temporal_alignment")
INDEX_PATH = Path("artifacts/evals/week15_temporal_alignment_waveform_rms_index.json")


def path_exists_from_string(value: str) -> Path | None:
    if not isinstance(value, str):
        return None
    if ".wav" not in value.lower():
        return None

    candidates = []
    raw = Path(value)
    candidates.append(raw)
    candidates.append(Path.cwd() / raw)

    # 如果 JSON 里只存 basename，则全仓按 basename 找一次。
    candidates.extend(Path(".").rglob(raw.name))

    for p in candidates:
        if p.exists() and p.is_file():
            return p
    return None


def walk_json(obj: Any, candidate_id: str, context: str = "") -> list[tuple[str, Path]]:
    hits: list[tuple[str, Path]] = []

    if isinstance(obj, dict):
        joined = json.dumps(obj, ensure_ascii=False)
        record_related = candidate_id in joined
        for k, v in obj.items():
            key_context = f"{context}.{k}" if context else str(k)
            if isinstance(v, str):
                p = path_exists_from_string(v)
                if p is not None and (record_related or candidate_id in v or candidate_id in key_context):
                    hits.append((key_context, p))
            else:
                hits.extend(walk_json(v, candidate_id, key_context))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            hits.extend(walk_json(v, candidate_id, f"{context}[{i}]"))

    return hits


def collect_from_json_files(candidate_id: str) -> list[tuple[str, Path]]:
    hits: list[tuple[str, Path]] = []
    for jf in sorted(Path("artifacts").rglob("*.json")):
        if "waveform_rms_index" in jf.name:
            continue
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue
        for ctx, p in walk_json(data, candidate_id):
            hits.append((f"{jf}:{ctx}", p))
    return hits


def collect_from_csv_files(candidate_id: str) -> list[tuple[str, Path]]:
    hits: list[tuple[str, Path]] = []
    for cf in sorted(Path("artifacts").rglob("*.csv")):
        try:
            with cf.open("r", encoding="utf-8", errors="replace", newline="") as f:
                reader = csv.DictReader(f)
                for row_idx, row in enumerate(reader):
                    row_blob = json.dumps(row, ensure_ascii=False)
                    if candidate_id not in row_blob:
                        continue
                    for k, v in row.items():
                        if isinstance(v, str):
                            p = path_exists_from_string(v)
                            if p is not None:
                                hits.append((f"{cf}:row{row_idx}:{k}", p))
        except Exception:
            continue
    return hits


def collect_from_filesystem(candidate_id: str) -> list[tuple[str, Path]]:
    hits: list[tuple[str, Path]] = []
    for p in sorted(Path("artifacts").rglob("*.wav")):
        if candidate_id in str(p):
            hits.append(("filesystem", p))
    return hits


def classify_paths(hits: list[tuple[str, Path]]) -> dict[str, Path]:
    unique: list[tuple[str, Path]] = []
    seen = set()
    for ctx, p in hits:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        unique.append((ctx, p))

    original_candidates = []
    remediated_candidates = []

    for ctx, p in unique:
        s = f"{ctx} {p}".lower()
        if "remediated" in s or "trimmed" in s or "preroll" in s:
            remediated_candidates.append((ctx, p))
        else:
            original_candidates.append((ctx, p))

    if not remediated_candidates:
        # 兜底：如果只有一个包含 trimmed 的文件没有被分类到，按路径再判断一次。
        for ctx, p in unique:
            if "trim" in p.name.lower():
                remediated_candidates.append((ctx, p))

    if not original_candidates or not remediated_candidates:
        detail = {
            "allHits": [(ctx, str(p)) for ctx, p in unique],
            "originalCandidates": [(ctx, str(p)) for ctx, p in original_candidates],
            "remediatedCandidates": [(ctx, str(p)) for ctx, p in remediated_candidates],
        }
        raise FileNotFoundError(json.dumps(detail, ensure_ascii=False, indent=2))

    return {
        "original": original_candidates[0][1],
        "remediated": remediated_candidates[0][1],
    }


def find_candidate_wavs(candidate_id: str) -> dict[str, Path]:
    hits = []
    hits.extend(collect_from_json_files(candidate_id))
    hits.extend(collect_from_csv_files(candidate_id))
    hits.extend(collect_from_filesystem(candidate_id))
    return classify_paths(hits)


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    y, sr = sf.read(path, always_2d=True)
    if y.size == 0:
        raise ValueError(f"empty audio: {path}")
    y = y.astype(np.float32)
    mono = y.mean(axis=1)
    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    if peak > 1.0:
        mono = mono / peak
    return mono, int(sr)


def frame_rms(y: np.ndarray, sr: int, frame_ms: float = 25.0, hop_ms: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    frame = max(1, int(sr * frame_ms / 1000.0))
    hop = max(1, int(sr * hop_ms / 1000.0))
    if len(y) < frame:
        padded = np.zeros(frame, dtype=np.float32)
        padded[: len(y)] = y
        y = padded

    n_frames = 1 + max(0, (len(y) - frame) // hop)
    rms = np.empty(n_frames, dtype=np.float32)
    times = np.empty(n_frames, dtype=np.float32)

    for i in range(n_frames):
        start = i * hop
        chunk = y[start : start + frame]
        rms[i] = math.sqrt(float(np.mean(chunk * chunk)) + 1e-12)
        times[i] = (start + frame / 2.0) / sr

    return times, rms


def onset_proxy_sec(times: np.ndarray, rms: np.ndarray) -> float | None:
    if rms.size == 0:
        return None
    floor = float(np.percentile(rms, 10))
    high = float(np.percentile(rms, 95))
    threshold = floor + 0.20 * max(1e-8, high - floor)
    active = np.where(rms >= threshold)[0]
    if active.size == 0:
        return None
    return float(times[int(active[0])])


def plot_pair(candidate_id: str, original_path: Path, remediated_path: Path) -> dict[str, Any]:
    y0, sr0 = read_audio(original_path)
    y1, sr1 = read_audio(remediated_path)

    t0 = np.arange(len(y0), dtype=np.float32) / sr0
    t1 = np.arange(len(y1), dtype=np.float32) / sr1

    rms_t0, rms0 = frame_rms(y0, sr0)
    rms_t1, rms1 = frame_rms(y1, sr1)

    onset0 = onset_proxy_sec(rms_t0, rms0)
    onset1 = onset_proxy_sec(rms_t1, rms1)

    out_png = OUT_DIR / f"{candidate_id}_waveform_rms_original_vs_remediated.png"

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=False)

    axes[0].plot(t0, y0, linewidth=0.8, label="original waveform")
    axes[0].plot(rms_t0, rms0, linewidth=1.2, label="original RMS")
    if onset0 is not None:
        axes[0].axvline(onset0, linestyle="--", linewidth=1.0, label=f"onset proxy={onset0:.3f}s")
    axes[0].set_title(f"{candidate_id} original | sr={sr0}, duration={len(y0)/sr0:.3f}s")
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("amplitude / RMS")
    axes[0].legend(loc="upper right")

    axes[1].plot(t1, y1, linewidth=0.8, label="remediated waveform")
    axes[1].plot(rms_t1, rms1, linewidth=1.2, label="remediated RMS")
    if onset1 is not None:
        axes[1].axvline(onset1, linestyle="--", linewidth=1.0, label=f"onset proxy={onset1:.3f}s")
    axes[1].set_title(f"{candidate_id} remediated | sr={sr1}, duration={len(y1)/sr1:.3f}s")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("amplitude / RMS")
    axes[1].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

    return {
        "candidateId": candidate_id,
        "originalAudio": str(original_path),
        "remediatedAudio": str(remediated_path),
        "figure": str(out_png),
        "originalSampleRate": sr0,
        "remediatedSampleRate": sr1,
        "originalDurationSec": round(len(y0) / sr0, 6),
        "remediatedDurationSec": round(len(y1) / sr1, 6),
        "durationDeltaSec": round(len(y1) / sr1 - len(y0) / sr0, 6),
        "originalOnsetProxySec": None if onset0 is None else round(onset0, 6),
        "remediatedOnsetProxySec": None if onset1 is None else round(onset1, 6),
        "onsetProxyDeltaSec": None if onset0 is None or onset1 is None else round(onset1 - onset0, 6),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    errors = []

    for candidate_id in CANDIDATES:
        try:
            paths = find_candidate_wavs(candidate_id)
            records.append(plot_pair(candidate_id, paths["original"], paths["remediated"]))
        except Exception as exc:
            errors.append({"candidateId": candidate_id, "error": str(exc)})

    index = {
        "schemaVersion": "week15.temporal_alignment_waveform_rms_index.v2",
        "status": "PASS" if len(records) == len(CANDIDATES) and not errors else "FAIL",
        "purpose": "Explain original FAIL_DRIFT and remediated PASS using waveform and frame-level RMS evidence.",
        "candidates": records,
        "errors": errors,
        "boundary": [
            "visual_explainability_only",
            "does_not_claim_semantic_audio_quality",
            "does_not_claim_human_audition_passed",
            "does_not_claim_final_mix_readiness",
        ],
    }

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(index, ensure_ascii=False, indent=2))
    print(f"WROTE_INDEX={INDEX_PATH}")
    for r in records:
        print(f"WROTE_FIGURE={r['figure']}")

    return 0 if index["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
