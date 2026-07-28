const VALUES=["","LEFT","RIGHT","TIE","UNJUDGEABLE"];
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
