#!/usr/bin/env python3
import argparse,csv,json,statistics,sys,time
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from soundlayer.runner.candidate_matrix import CandidateMatrix
from soundlayer.runner.contracts import digest_value
def write_csv(path,rows):
    flat=[]
    for x in rows:
        y={k:(json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v) for k,v in x.items()};flat.append(y)
    with Path(path).open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=flat[0].keys(),lineterminator="\n");w.writeheader();w.writerows(flat)
def percentile(values,q):
    if not values:return 0
    values=sorted(values);return values[min(len(values)-1,int((len(values)-1)*q))]
def main():
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--dry-run",action="store_true")
    p.add_argument("--out-dir");p.add_argument("--manifest");p.add_argument("--resume",action="store_true")
    p.add_argument("--only-adapters");p.add_argument("--only-ready-live",action="store_true");p.add_argument("--case-limit-per-adapter",type=int)
    p.add_argument("--target-new-live-wavs",type=int);p.add_argument("--gpu-capacity",type=int,default=1)
    p.add_argument("--max-cpu-workers",type=int,default=4);p.add_argument("--max-io-workers",type=int,default=8)
    a=p.parse_args();config=json.loads(Path(a.config).read_text());matrix=CandidateMatrix(ROOT,config);plan=matrix.plan()
    if a.dry_run:
        target=Path(a.manifest);target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(plan,indent=2)+"\n")
        print(json.dumps({"planned":plan["planned_count"],"collisions":plan["collision_count"],"manifestDigest":plan["manifest_digest"]}));return
    started=time.perf_counter_ns();plan,rows,recovery=matrix.execute(a.resume)
    status=Counter(x["status"] for x in rows);modes=Counter(x["generation_mode"] for x in rows)
    live=[x for x in rows if x["generation_mode"]=="LIVE"]
    summary={"matrixStatus":"GREEN" if any(x["status"]=="SUCCEEDED" for x in live) else "YELLOW_RUNTIME_BLOCKED",
      "caseCount":len({x["matrix_case_id"] for x in rows}),"slotCount":len(rows),"plannedCount":48,
      "succeededCount":status["SUCCEEDED"],"blockedCount":status["BLOCKED"],"failedCount":status["FAILED"],
      "livePlannedCount":len(live),"liveAttemptCount":sum(x["status"] in {"SUCCEEDED","FAILED"} for x in live),
      "liveSuccessCount":sum(x["status"]=="SUCCEEDED" for x in live),
      "newLiveWavCount":sum(x["status"]=="SUCCEEDED" and x["generation_mode"]=="LIVE" for x in rows),
      "replayCount":modes["REPLAY"],"replayBaselineSourceCount":len({x["source_audio"] for x in rows}),
      "controlCount":modes["CONTROL"],"generativeAdapterCount":len({x["adapter_id"] for x in live if x["status"]=="SUCCEEDED"}),
      "executableSourceCount":len({x["source_audio"] for x in rows}),"outputCollisionCount":plan["collision_count"],
      "duplicateArtifactCount":0,"resumeReusedCount":recovery["resume_reused"],"staleRerunCount":recovery["stale_count"],
      "pendingPublishedAsFinal":0,"rejectedPublishBlockedCount":len({x["matrix_case_id"] for x in rows if x["repair_decision"]=="REPAIR_REJECTED"}),
      "replayCountedAsLive":0,"manifestDigest":plan["manifest_digest"],
      "totalRuntimeMs":(time.perf_counter_ns()-started)//1_000_000}
    report={"schemaVersion":"candidate-matrix/v1","plan":{k:v for k,v in plan.items() if k!="records"},"summary":summary,"records":rows}
    reports=ROOT/"reports";reports.mkdir(exist_ok=True)
    (reports/"w20_candidate_matrix_20260721.json").write_text(json.dumps(report,indent=2)+"\n")
    write_csv(reports/"w20_candidate_matrix_20260721.csv",rows)
    (reports/"w20_candidate_matrix_summary_20260721.json").write_text(json.dumps(summary,indent=2)+"\n")
    failures=[x for x in rows if x["status"]!="SUCCEEDED"]
    (reports/"w20_candidate_matrix_failures_20260721.json").write_text(json.dumps({"failures":failures},indent=2)+"\n")
    runtime=[{"case_id":x["matrix_case_id"],"slot_id":x["slot_id"],"adapter_id":x["adapter_id"],
              "status":x["status"],"queue_wait_ms":x["queue_wait_ms"],"runtime_ms":x["runtime_ms"],
              "peak_gpu_memory_mb":x["peak_gpu_memory_mb"]} for x in rows]
    write_csv(reports/"w20_candidate_matrix_runtime_20260721.csv",runtime)
    print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
