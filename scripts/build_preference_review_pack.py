#!/usr/bin/env python3
"""Build a self-contained, opaque A/B listening page from the private key."""

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

INDEX_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>声音偏好盲测</title>
  <link rel="stylesheet" href="styles.css">
</head>
<body>
<main>
  <header><h1>声音偏好盲测</h1><p id="progress">正在载入…</p></header>
  <section class="media">
    <video id="video" muted playsinline controls></video>
    <div class="players">
      <button id="playA">播放声音 A</button>
      <button id="playB">播放声音 B</button>
    </div>
    <audio id="audioA"></audio><audio id="audioB"></audio>
  </section>
  <form id="form">
    <label>总体偏好<select name="overall_preference" required></select></label>
    <label>时序<select name="timing_preference"></select></label>
    <label>事件覆盖<select name="event_coverage_preference"></select></label>
    <label>音质<select name="audio_quality_preference"></select></label>
    <label>多余声音<select name="unwanted_event_preference"></select></label>
    <label>置信度（1-5）<input name="confidence" type="number" min="1" max="5" required></label>
    <label>原因代码<select name="reason_codes"></select></label>
    <label class="wide">简短原因<textarea name="free_text_reason" rows="2"></textarea></label>
    <label>评审人代号<input name="reviewer_id" autocomplete="off"></label>
  </form>
  <nav>
    <button id="prev">上一组</button>
    <button id="save">保存并下一组</button>
    <button id="exportJson">导出 JSON</button>
    <button id="exportCsv">导出 CSV</button>
  </nav>
  <p class="note">页面仅保存本机浏览器进度。TIE/UNJUDGEABLE 可用于确实无法判断的情况。</p>
</main>
<script src="app.js"></script>
</body>
</html>
"""

STYLES = """body{font-family:system-ui,sans-serif;background:#f5f3ee;color:#202020;margin:0}
main{max-width:900px;margin:auto;padding:24px}header,.media,form,nav{background:white;padding:18px;margin:12px 0;border-radius:10px}
video{display:block;width:100%;max-height:360px;background:#111}.players{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:14px}
button{padding:12px;border:1px solid #333;border-radius:7px;background:#f8f8f8;cursor:pointer}
form{display:grid;grid-template-columns:1fr 1fr;gap:14px}label{display:flex;flex-direction:column;gap:5px}.wide{grid-column:1/-1}
select,input,textarea{font:inherit;padding:8px}nav{display:flex;gap:10px;flex-wrap:wrap}.note{color:#555}@media(max-width:650px){form{grid-template-columns:1fr}.wide{grid-column:auto}}
"""

APP_JS = r"""const VALUES=["","LEFT","RIGHT","TIE","UNJUDGEABLE"];
const REASONS=["","BETTER_SYNC","BETTER_EVENT_COVERAGE","CLEANER_AUDIO","FEWER_UNWANTED_EVENTS","BETTER_BALANCE","NO_MEANINGFUL_DIFFERENCE","PLAYBACK_OR_CONTENT_ISSUE"];
const FIELDS=["overall_preference","timing_preference","event_coverage_preference","audio_quality_preference","unwanted_event_preference","confidence","reason_codes","free_text_reason","reviewer_id"];
let pairs=[],index=0,answers=JSON.parse(localStorage.getItem("blindPreferenceAnswers")||"{}");
const form=document.querySelector("#form"),video=document.querySelector("#video"),audioA=document.querySelector("#audioA"),audioB=document.querySelector("#audioB");
for(const select of form.querySelectorAll("select")){const values=select.name==="reason_codes"?REASONS:VALUES;for(const value of values){const option=document.createElement("option");option.value=value;option.textContent=value||"请选择";select.appendChild(option)}}
function stopAll(){for(const media of [video,audioA,audioB]){media.pause();media.currentTime=0}}
async function play(audio){stopAll();video.currentTime=0;audio.currentTime=0;await Promise.all([video.play(),audio.play()])}
function load(){const pair=pairs[index];stopAll();video.src=pair.video_media;audioA.src=pair.left_media;audioB.src=pair.right_media;document.querySelector("#progress").textContent=`${index+1} / ${pairs.length} · ${pair.block_id}`;const answer=answers[pair.opaque_pair_id]||{};for(const field of FIELDS){form.elements[field].value=answer[field]||""}}
function current(){const pair=pairs[index],row={pair_id:pair.opaque_pair_id};for(const field of FIELDS)row[field]=form.elements[field].value;row.reviewed_at=new Date().toISOString();row.submitted=Boolean(row.overall_preference&&row.confidence);return row}
function save(){answers[pairs[index].opaque_pair_id]=current();localStorage.setItem("blindPreferenceAnswers",JSON.stringify(answers))}
function rows(){return pairs.map(pair=>answers[pair.opaque_pair_id]||{pair_id:pair.opaque_pair_id,submitted:false})}
function download(name,text,type){const link=document.createElement("a");link.href=URL.createObjectURL(new Blob([text],{type}));link.download=name;link.click();URL.revokeObjectURL(link.href)}
document.querySelector("#playA").onclick=()=>play(audioA);document.querySelector("#playB").onclick=()=>play(audioB);
document.querySelector("#save").onclick=event=>{event.preventDefault();save();if(index<pairs.length-1)index++;load()};
document.querySelector("#prev").onclick=event=>{event.preventDefault();save();if(index>0)index--;load()};
document.querySelector("#exportJson").onclick=event=>{event.preventDefault();save();download("preference_labels.json",JSON.stringify(rows(),null,2),"application/json")};
document.querySelector("#exportCsv").onclick=event=>{event.preventDefault();save();const fields=["pair_id",...FIELDS,"reviewed_at","submitted"];const quote=v=>`"${String(v??"").replaceAll('"','""')}"`;download("preference_labels.csv",[fields.map(quote).join(","),...rows().map(row=>fields.map(key=>quote(row[key])).join(","))].join("\n"),"text/csv")};
fetch("data/review_pairs.json").then(response=>response.json()).then(data=>{pairs=data.pairs.sort((a,b)=>a.display_index-b.display_index);load()});
"""


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-pairs", required=True)
    parser.add_argument("--private-key", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    private = json.loads(resolve(args.private_key).read_text())
    public_path = resolve(args.public_pairs)
    public = json.loads(public_path.read_text())
    output = resolve(args.output_dir)
    media_dir = output / "media"
    data_dir = output / "data"
    if media_dir.is_dir():
        for old_asset in media_dir.iterdir():
            if old_asset.is_file():
                old_asset.unlink()
    media_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    private_by_id = {row["pair_id"]: row for row in private["pairs"]}
    copied = {}
    missing = []
    for row in public["pairs"]:
        truth = private_by_id[row["opaque_pair_id"]]
        for source_key, target_key in (
            ("left_artifact", "left_media"),
            ("right_artifact", "right_media"),
            ("video_path", "video_media"),
        ):
            source = ROOT / truth[source_key]
            target = output / row[target_key]
            if not source.is_file():
                missing.append(str(source.relative_to(ROOT)))
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if target not in copied:
                shutil.copy2(source, target)
                copied[target] = source
    if missing:
        raise FileNotFoundError("missing review media: " + ", ".join(sorted(set(missing))))
    (data_dir / "review_pairs.json").write_text(
        json.dumps(public, indent=2, ensure_ascii=False) + "\n"
    )
    (output / "index.html").write_text(INDEX_HTML)
    (output / "styles.css").write_text(STYLES)
    (output / "app.js").write_text(APP_JS)
    report = {
        "schemaVersion": "blind-review-pack/v1",
        "status": (
            "READY_FOR_HUMAN_REVIEW"
            if private["summary"]["judgmentCount"] == 48
            and private["summary"]["uniquePairShortfall"] == 0
            else "PILOT_READY_TRAINING_BLOCKED"
        ),
        "pairCount": len(public["pairs"]),
        "mediaFileCount": len(copied),
        "missingArtifactCount": 0,
        "progressPersistence": "localStorage",
        "exports": ["CSV", "JSON"],
        "finalSelectedMutationCount": 0,
    }
    (output.parent / "review_pack_build_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
