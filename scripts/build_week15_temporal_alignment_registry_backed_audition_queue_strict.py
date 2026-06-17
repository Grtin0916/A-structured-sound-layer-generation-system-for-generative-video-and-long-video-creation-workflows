#!/usr/bin/env python3
import csv
import html
import json
from datetime import datetime, timezone
from pathlib import Path

CANDIDATES = {
    "procedural_v0_0002": {
        "baselineAudio": "artifacts/audio_candidates/week12_procedural_baseline_v0/0002_slot_v0_0002_generation_fallback.wav",
        "remediatedAudio": None,
        "visual": None,
        "reason": "Registry-backed semantic/timing risk candidate; strict baseline-only audition target.",
    },
    "procedural_v0_0003": {
        "baselineAudio": "artifacts/audio_candidates/week12_procedural_baseline_v0/0003_slot_v0_0003_generation_fallback.wav",
        "remediatedAudio": None,
        "visual": None,
        "reason": "Registry-backed semantic/timing risk candidate; strict baseline-only audition target.",
    },
    "procedural_v0_0004": {
        "baselineAudio": "artifacts/audio_candidates/week12_procedural_baseline_v0/0004_slot_v0_0004_generation_fallback.wav",
        "remediatedAudio": "artifacts/audio_candidates/week15_temporal_alignment_remediated/procedural_v0_0004_trimmed_preroll_20ms.wav",
        "visual": "artifacts/figures/week15_temporal_alignment/procedural_v0_0004_waveform_rms_original_vs_remediated.png",
        "reason": "Known remediation case; compare original baseline vs trimmed 20ms pre-roll remediated audio.",
    },
    "procedural_v0_0007": {
        "baselineAudio": "artifacts/audio_candidates/week12_procedural_baseline_v0/0007_slot_v0_0007_generation_fallback.wav",
        "remediatedAudio": None,
        "visual": None,
        "reason": "Registry-backed semantic/timing risk candidate; strict baseline-only audition target.",
    },
}

def exists(path):
    return bool(path) and Path(path).exists()

def rel_for_html(path):
    if not path:
        return ""
    p = Path(path)
    try:
        return "../" + p.relative_to("artifacts").as_posix()
    except ValueError:
        return p.as_posix()

items = []
for cid, spec in CANDIDATES.items():
    baseline = spec["baselineAudio"]
    remediated = spec["remediatedAudio"]
    visual = spec["visual"]

    audio_targets = []
    audio_targets.append({
        "role": "baseline",
        "path": baseline,
        "exists": exists(baseline),
    })
    if remediated:
        audio_targets.append({
            "role": "remediated",
            "path": remediated,
            "exists": exists(remediated),
        })

    image_targets = []
    if visual:
        image_targets.append({
            "role": "waveform_rms_original_vs_remediated",
            "path": visual,
            "exists": exists(visual),
        })

    items.append({
        "candidateId": cid,
        "strictMapping": True,
        "audioTargets": audio_targets,
        "imageTargets": image_targets,
        "allRequiredAudioExists": all(t["exists"] for t in audio_targets),
        "allRequiredImagesExist": all(t["exists"] for t in image_targets) if image_targets else True,
        "auditionStatus": "STRICT_AUDIO_TARGET_READY_PENDING_HUMAN_AUDITION",
        "semanticJudgement": "NOT_REVIEWED",
        "timingJudgement": "NOT_REVIEWED",
        "humanReviewStatus": "NOT_PERFORMED",
        "reason": spec["reason"],
        "notes": "",
    })

payload = {
    "schemaVersion": "week15.mainbase.temporal-alignment.registry-backed-audition-queue.strict.v1",
    "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
    "source": "Cloud/Java registry-backed riskCandidateIds with deterministic candidate-to-audio mapping",
    "claimBoundary": "Strict audition queue only. HUMAN_REVIEW_PASS is not claimed until user actually listens to the audio.",
    "riskCandidateIds": list(CANDIDATES.keys()),
    "mappingPolicy": {
        "candidateSpecificOnly": True,
        "contentWideWavFallback": False,
        "htmlAudioPreview": True,
        "visualPreviewFor0004": True,
    },
    "items": items,
}

out_json = Path("artifacts/reviews/week15_temporal_alignment_registry_backed_audition_queue_strict.json")
out_csv = Path("artifacts/reviews/week15_temporal_alignment_registry_backed_audition_queue_strict.csv")
out_html = Path("artifacts/reviews/week15_temporal_alignment_registry_backed_audition_queue_strict.html")

out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

with out_csv.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "candidateId",
        "audioRoles",
        "audioPaths",
        "allRequiredAudioExists",
        "imagePaths",
        "allRequiredImagesExist",
        "auditionStatus",
        "humanReviewStatus",
        "reason",
    ])
    writer.writeheader()
    for item in items:
        writer.writerow({
            "candidateId": item["candidateId"],
            "audioRoles": " | ".join(t["role"] for t in item["audioTargets"]),
            "audioPaths": " | ".join(t["path"] for t in item["audioTargets"]),
            "allRequiredAudioExists": item["allRequiredAudioExists"],
            "imagePaths": " | ".join(t["path"] for t in item["imageTargets"]),
            "allRequiredImagesExist": item["allRequiredImagesExist"],
            "auditionStatus": item["auditionStatus"],
            "humanReviewStatus": item["humanReviewStatus"],
            "reason": item["reason"],
        })

body = []
body.append("<html><head><meta charset='utf-8'><title>Week15 Strict Audition Queue</title>")
body.append("<style>body{font-family:Arial,sans-serif;line-height:1.45} table{border-collapse:collapse;width:100%} th,td{border:1px solid #bbb;padding:8px;vertical-align:top} code{white-space:pre-wrap} img{max-width:720px}</style>")
body.append("</head><body>")
body.append("<h1>Week15 Strict Registry-backed Audition Queue</h1>")
body.append("<p><b>Boundary:</b> Strict queue only. HUMAN_REVIEW_PASS is not claimed until actual human audition.</p>")
body.append("<table><tr><th>Candidate</th><th>Audio targets</th><th>Visual</th><th>Status</th></tr>")

for item in items:
    body.append("<tr>")
    body.append(f"<td><b>{html.escape(item['candidateId'])}</b><br><code>{html.escape(item['reason'])}</code></td>")

    audio_html = []
    for target in item["audioTargets"]:
        audio_html.append(f"<p><b>{html.escape(target['role'])}</b>: <code>{html.escape(target['path'])}</code><br>exists={target['exists']}</p>")
        if target["exists"]:
            audio_html.append(f"<audio controls src='{html.escape(rel_for_html(target['path']))}'></audio>")
    body.append("<td>" + "".join(audio_html) + "</td>")

    visual_html = []
    for target in item["imageTargets"]:
        visual_html.append(f"<p><b>{html.escape(target['role'])}</b>: <code>{html.escape(target['path'])}</code><br>exists={target['exists']}</p>")
        if target["exists"]:
            visual_html.append(f"<img src='{html.escape(rel_for_html(target['path']))}'>")
    body.append("<td>" + ("".join(visual_html) if visual_html else "NO_VISUAL_TARGET") + "</td>")

    body.append(f"<td>{html.escape(item['auditionStatus'])}<br>humanReviewStatus={html.escape(item['humanReviewStatus'])}</td>")
    body.append("</tr>")

body.append("</table></body></html>")
out_html.write_text("\n".join(body), encoding="utf-8")

summary = {
    "json": str(out_json),
    "csv": str(out_csv),
    "html": str(out_html),
    "items": [
        {
            "candidateId": item["candidateId"],
            "audioTargets": item["audioTargets"],
            "imageTargets": item["imageTargets"],
            "allRequiredAudioExists": item["allRequiredAudioExists"],
            "allRequiredImagesExist": item["allRequiredImagesExist"],
            "auditionStatus": item["auditionStatus"],
        }
        for item in items
    ],
}

print(json.dumps(summary, ensure_ascii=False, indent=2))

if len(items) != 4:
    raise SystemExit("Expected 4 candidates.")
if not all(item["allRequiredAudioExists"] for item in items):
    raise SystemExit("At least one required audio target is missing.")
if not all(item["allRequiredImagesExist"] for item in items):
    raise SystemExit("At least one required image target is missing.")
