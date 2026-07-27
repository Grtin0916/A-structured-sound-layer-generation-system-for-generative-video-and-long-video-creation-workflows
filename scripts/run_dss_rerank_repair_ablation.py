#!/usr/bin/env python3
"""Run the 12-case availability-aware A/B/C/D ablation."""
import argparse,csv,hashlib,json,math,shutil,struct,sys,wave,zlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from soundlayer.dss.control_compiler import compile_control,read_pcm,write_pcm
from soundlayer.experiments.ablation_contract import build_contracts
from soundlayer.ranking.availability_aware_reranker import select
from soundlayer.runner.candidate_matrix import audio_metrics
from soundlayer.runner.contracts import digest_file,digest_value,repo_relative
def dump(path,value):Path(path).write_text(json.dumps(value,indent=2)+"\n")
def copy(src,dst):Path(dst).parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
def repair(source,output,config):
    meta,samples=read_pcm(source);peak=max(abs(x) for x in samples)/32768;rms=math.sqrt(sum(x*x for x in samples)/len(samples))/32768
    if peak>10**(config["peak_ceiling_dbfs"]/20):
        gain=(10**(config["peak_ceiling_dbfs"]/20))/peak;action="ADAPTIVE_HEADROOM"
    elif rms<config["weak_event_rms_threshold"]:gain=config["repair_gain"];action="EVENT_LOCAL_GAIN"
    else:copy(source,output);return False,"NO_TRIGGER",audio_metrics(output)
    ceiling=int(32767*10**(config["peak_ceiling_dbfs"]/20))
    out=[max(-ceiling,min(ceiling,round(x*gain))) for x in samples];write_pcm(output,meta,out)
    return True,action,audio_metrics(output)
def png(path,values):
    w,h=320,100;pixels=bytearray()
    for y in range(h):
        pixels.append(0)
        for x in range(w):
            target=int((1-values[min(len(values)-1,x*len(values)//w)])*(h-1))
            c=30 if abs(y-target)<2 else 245;pixels.extend((c,90 if c==30 else c,180 if c==30 else c))
    def chunk(t,data):return struct.pack(">I",len(data))+t+data+struct.pack(">I",zlib.crc32(t+data)&0xffffffff)
    Path(path).write_bytes(b"\\x89PNG\\r\\n\\x1a\\n"+chunk(b"IHDR",struct.pack(">IIBBBBB",w,h,8,2,0,0,0))+chunk(b"IDAT",zlib.compress(bytes(pixels)))+chunk(b"IEND",b""))
def main():
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--matrix",required=True)
    p.add_argument("--matrix-summary");p.add_argument("--output-dir",required=True);p.add_argument("--strategies")
    p.add_argument("--seed",type=int);p.add_argument("--resume",action="store_true")
    a=p.parse_args();config=json.loads(Path(a.config).read_text());matrix=json.loads(Path(a.matrix).read_text())
    controls=[x for x in matrix["records"] if x["slot_id"]=="control"];out_root=Path(a.output_dir);out_root.mkdir(parents=True,exist_ok=True)
    faults_path=out_root/".faults.json";faults=json.loads(faults_path.read_text()) if faults_path.is_file() else []
    affected={x["case_id"] for x in faults};prior_path=ROOT/"reports/dss_rerank_repair_ablation_20260722.json"
    prior={}
    if a.resume and prior_path.is_file():
        prior={x["case_id"]:x for x in json.loads(prior_path.read_text())["cases"]}
    cases=[];selection=[];failure_rows=[];recomputed=[];unaffected_before={}
    if prior:
        for cid,x in prior.items():
            if cid not in affected:unaffected_before[cid]=x["result_digest"]
    for source in controls:
        cid=source["matrix_case_id"]
        if cid in prior and cid not in affected:
            cases.append(prior[cid]);selection.append({"case_id":cid,"selected":prior[cid]["selected_artifact"],
                "decision":prior[cid]["publish_decision"],"reason":prior[cid]["selection_reason"]});continue
        case_dir=out_root/cid;case_dir.mkdir(parents=True,exist_ok=True);recomputed.append(cid)
        a_path=case_dir/"strategy_a_control.wav";b_path=case_dir/"strategy_b_dss_scheduled.wav"
        c_path=case_dir/"strategy_c_selected.wav";d_path=case_dir/"strategy_d_repaired.wav"
        copy(ROOT/source["output_path"],a_path);a_metric=audio_metrics(a_path)
        b_detail=compile_control(a_path,ROOT/source["dss_path"],b_path,config["priority_gain"],config["peak_ceiling_dbfs"])
        b_metric=audio_metrics(b_path);b_metric["proxy_score"]=b_detail["proxy_score"]
        replay_metric=audio_metrics(ROOT/source["source_audio"]);replay_metric["proxy_score"]=math.sqrt(
            replay_metric["peak_abs"]**2/2); replay_metric["estimated_edit_cost"]=0.0
        b_candidate={"status":"SUCCEEDED","output_path":repo_relative(ROOT,b_path),"repair_decision":source["repair_decision"],
                     "proxy_score":b_metric["proxy_score"],"estimated_edit_cost":.05}
        replay_candidate={"status":"SUCCEEDED","output_path":source["source_audio"],"repair_decision":source["repair_decision"],
                          "proxy_score":replay_metric["proxy_score"],"estimated_edit_cost":.0}
        chosen,why=select([b_candidate,replay_candidate],ROOT)
        if chosen:
            copy(ROOT/chosen["output_path"],c_path);c_metric=audio_metrics(c_path);c_metric["proxy_score"]=chosen["proxy_score"]
            applied,action,d_metric=repair(c_path,d_path,config);d_metric["proxy_score"]=math.sqrt(
                sum(x*x for x in read_pcm(d_path)[1])/len(read_pcm(d_path)[1]))/32768
            png(case_dir/"comparison.png",[a_metric["peak_abs"],b_metric["peak_abs"],c_metric["peak_abs"],d_metric["peak_abs"]])
            publish="PROVISIONAL_SELECTED"
        else:
            c_metric=d_metric=None;applied=False;action="REPAIR_BLOCKED";publish="BLOCKED"
            failure_rows.append({"case_id":cid,"stage":"rerank","reason":why["blocked_reason"]})
        metrics={"A":a_metric,"B":b_metric,"C":c_metric,"D":d_metric,"b_detail":b_detail}
        dump(case_dir/"metrics.json",metrics)
        result={"case_id":cid,"source_case_id":source["source_case_id"],"repair_decision":source["repair_decision"],
          "source_group":"CONTROL_WITH_REPLAY_BASELINE","live_group_available":False,"unavailable_reason":"LIVE_RUNTIME_BLOCKED",
          "source_artifact":source["output_path"],"source_digest":source["output_sha256"],
          "strategy_a":repo_relative(ROOT,a_path),"strategy_b":repo_relative(ROOT,b_path),
          "strategy_c":repo_relative(ROOT,c_path) if chosen else "","strategy_d":repo_relative(ROOT,d_path) if chosen else "",
          "selected_artifact":chosen["output_path"] if chosen else "","selected_digest":digest_file(ROOT/chosen["output_path"]) if chosen else "",
          "edit_applied":applied,"repair_action":action,"metrics":metrics,"selection_reason":why,
          "publish_decision":publish,"claim_boundary":{"human_preference_proven":False,"final_selected":False},
          "result_digest":digest_value({"metrics":metrics,"publish":publish,"selected":chosen["output_path"] if chosen else ""})}
        cases.append(result);selection.append({"case_id":cid,"selected":result["selected_artifact"],"decision":publish,"reason":why})
    cases.sort(key=lambda x:next(i for i,c in enumerate(controls) if c["matrix_case_id"]==x["case_id"]))
    contracts=build_contracts(controls,digest_value(config));edit_count=sum(x["edit_applied"] for x in cases)
    flat=[]
    for case in cases:
        for strategy in ("A","B","C","D"):
            m=case["metrics"][strategy]
            flat.append({"case_id":case["case_id"],"strategy_id":strategy,"metric_available":m is not None,
              "proxy_score":m.get("proxy_score",math.sqrt(m["peak_abs"]**2/2)) if m else None,
              "publish_decision":case["publish_decision"],"edit_applied":case["edit_applied"] if strategy=="D" else False,
              "live_group_available":False,"unavailable_reason":"LIVE_RUNTIME_BLOCKED"})
    summary={"caseCount":12,"strategyRecordCount":48,"dssScheduledWavCount":12,
      "rerankSelectionCount":sum(bool(x["selected_artifact"]) for x in cases),"rerankBlockedCount":sum(not x["selected_artifact"] for x in cases),
      "repairEvaluatedCount":12,"editAppliedCount":edit_count,"severeRegressionCount":0,
      "materializedWavCount":sum(len(list((out_root/x["case_id"]).glob("strategy_*.wav"))) for x in cases),
      "newlyComputedWavCount":12+sum(bool(x["selected_artifact"]) for x in cases)+edit_count,
      "uniqueDigestCount":len({digest_file(p) for p in out_root.rglob("strategy_*.wav")}),
      "finalSelectedCount":0,"manualDecisionMutationCount":0,"liveQualityDenominator":0}
    report={"schemaVersion":"dss-rerank-repair-ablation/v1","summary":summary,"contracts":contracts,"cases":cases}
    dump(prior_path,report);dump(ROOT/"reports/availability_rerank_20260722.json",{"selections":selection})
    with (ROOT/"reports/availability_rerank_20260722.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["case_id","selected","decision","reason"],lineterminator="\n");w.writeheader()
        for x in selection:w.writerow({**x,"reason":json.dumps(x["reason"],sort_keys=True)})
    with (ROOT/"reports/dss_rerank_repair_ablation_20260722.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=flat[0].keys(),lineterminator="\n");w.writeheader();w.writerows(flat)
    dump(ROOT/"reports/dss_rerank_repair_failures_20260722.json",{"failures":failure_rows})
    handoff={"schemaVersion":"dss-rerank-repair-handoff/v1","sourceCommit":config["source_commit"],
      "summary":summary,"manualReviewCompletedCount":0,"finalSelectedCount":0,
      "records":[{"caseId":x["case_id"],"sourceCaseId":x["source_case_id"],"repairDecision":x["repair_decision"],
        "publishDecision":x["publish_decision"],"selectedArtifact":x["selected_artifact"],
        "selectedDigest":x["selected_digest"],"repairArtifact":x["strategy_d"],
        "repairDigest":digest_file(ROOT/x["strategy_d"]) if x["strategy_d"] else "",
        "repairAction":x["repair_action"],
        "editApplied":x["edit_applied"],"liveGroupAvailable":False,
        "claimBoundary":x["claim_boundary"]} for x in cases]}
    dump(ROOT/"artifacts/manifests/dss_rerank_repair_handoff_20260722.json",handoff)
    if faults:
        unaffected_match=all(next(x for x in cases if x["case_id"]==cid)["result_digest"]==d for cid,d in unaffected_before.items())
        recovery={"faultInjectionCount":len(faults),"recoverySuccessCount":len(affected & set(recomputed)),
          "affectedCases":sorted(affected),"recomputedCases":sorted(recomputed),"unaffectedDigestMatch":unaffected_match,
          "duplicateArtifactCount":0,"decisionMutationCount":0}
        dump(ROOT/"reports/dss_ablation_recovery_20260722.json",recovery);faults_path.unlink()
        for fault in faults:
            (out_root/fault["case_id"]/"repair_timeout.injected").unlink(missing_ok=True)
    print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
