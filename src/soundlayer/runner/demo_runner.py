"""Manifest-driven replay runner."""
import csv, html, json, shutil, subprocess, time, wave
from pathlib import Path
from .cache import StageCache
from .contracts import cache_key,digest_file,digest_value,implementation_digest,repo_relative
from .run_context import RunContext
from .stage_graph import StageGraph
from .stages import STAGES,CACHEABLE

class InjectedFailure(RuntimeError): pass
class DemoRunner:
    def __init__(self,root,config):
        self.root=Path(root).resolve(); self.config=config; self.graph=StageGraph(STAGES)
        self.config_digest=digest_value(config)
        self.impl=implementation_digest(Path(__file__))
        self.cache=StageCache(self.root/"artifacts/runs/.cache")
    def inventory(self):
        rerank=self.root/self.config["rerank_csv"]
        with rerank.open() as source: rows=list(csv.DictReader(source))
        result=[]
        for r in rows:
            case=r["parent_candidate_id"].split("|")[0]; case_dir=self.root/"cases"/case
            paths={"input_video":case_dir/"input_video.mp4","dss":case_dir/"director_sound_script.yaml",
                   "before":self.root/r["before_artifact"],"after":self.root/r["after_artifact"]}
            checks={k:p.is_file() for k,p in paths.items()}
            score=sum(checks.values())+(r["decision"]=="MANUAL_REVIEW")*2+bool(r["target_delta"])
            result.append({"case_id":case,"repair_id":r["candidate_id"],"decision":r["decision"],
                           "score":score,"checks":checks,"paths":{k:repo_relative(self.root,p) for k,p in paths.items()},
                           "repair_action":r["repair_action"],"metrics_available":bool(r["target_delta"])})
        result.sort(key=lambda x:(-x["score"],x["case_id"],x["repair_id"]))
        return result
    def choose(self,case_id=None):
        rows=self.inventory()
        valid=[x for x in rows if all(x["checks"].values()) and x["decision"]=="MANUAL_REVIEW"]
        if case_id:
            valid=[x for x in valid if x["case_id"]==case_id]
        if not valid:raise ValueError("no complete MANUAL_REVIEW replay case")
        return valid[0]
    def identity(self,case):
        inputs={k:digest_file(self.root/v) for k,v in case["paths"].items()}
        key=digest_value({"case_id":case["case_id"],"mode":"replay","config_digest":self.config_digest,
                          "input_digests":inputs,"pipeline_version":self.config["pipeline_version"]})
        return key,inputs
    def plan(self,case_id=None):
        case=self.choose(case_id); run_key,inputs=self.identity(case); stages=[]
        for sid in self.graph.order:
            mode="VERIFY" if sid in {"inventory","preflight","evaluate","rerank","repair"} else "EXECUTE"
            key=cache_key(sid,self.impl,self.config_digest,inputs,mode)
            hit,reason,_=self.cache.lookup(key,self.root)
            stages.append({"stage_id":sid,"execution_mode":"REUSE" if hit and sid in CACHEABLE else mode,
                           "cache_hit":hit,"stale_reason":reason,"input_paths":list(case["paths"].values()),
                           "expected_outputs":["outputs/provisional_mix.wav"] if sid=="mix" else [],
                           "blocked_reason":None})
        return {"case":case,"run_key":run_key,"stage_order":self.graph.order,"stages":stages,
                "publish_decision":"provisional"}
    def run(self,case_id=None,fail_after=None,resume_run_id=None):
        if resume_run_id:
            ctx=RunContext(self.root,"resume","sha256:resume",resume_run_id)
            state=json.loads(ctx.state_path.read_text()); case=state["case"]; run_key=state["run_key"]
            inputs=state["input_digests"]; attempt=len(state["attempts"])+1
            state["stages"]=[x for x in state["stages"] if x["status"]=="SUCCEEDED"]
        else:
            case=self.choose(case_id); run_key,inputs=self.identity(case); ctx=RunContext(self.root,case["case_id"],run_key)
            state={"schema_version":"demo-run/v1","run_id":ctx.run_id,"run_key":run_key,"case":case,
                   "input_digests":inputs,"generation_mode":"REPLAY","source_commit":subprocess.check_output(
                   ["git","-C",str(self.root),"rev-parse","HEAD"],text=True).strip(),
                   "status":"RUNNING","attempts":[],"stages":[]}; attempt=1
        state["attempts"].append({"attempt_id":f"attempt-{attempt:03d}","started_ns":time.time_ns()})
        completed={x["stage_id"]:x for x in state["stages"] if x["status"]=="SUCCEEDED"}
        reused=executed=0
        try:
            for sid in self.graph.order:
                if sid in completed: continue
                mode="VERIFY" if sid in {"inventory","preflight","evaluate","rerank","repair"} else "EXECUTE"
                key=cache_key(sid,self.impl,self.config_digest,inputs,mode)
                hit,stale,cached=self.cache.lookup(key,self.root)
                started=time.perf_counter_ns(); output_digests={}
                if hit and sid in CACHEABLE: actual_mode="REUSE"; reused+=1; output_digests=cached["output_digests"]
                else:
                    actual_mode=mode; executed+=1
                    if sid=="mix":
                        if case["decision"]!="MANUAL_REVIEW": raise ValueError("publish blocked by decision")
                        dst=ctx.root/"outputs/provisional_mix.wav"; shutil.copy2(self.root/case["paths"]["after"],dst)
                        with wave.open(str(dst),"rb") as w: assert w.getnframes()>0
                        output_digests[repo_relative(self.root,dst)]=digest_file(dst)
                    elif sid in CACHEABLE:
                        rel=case["paths"]["after"] if sid in {"evaluate","rerank","repair"} else case["paths"]["dss"]
                        output_digests[rel]=digest_file(self.root/rel)
                    if sid=="report": self._reports(ctx,state,case)
                    if sid=="publish": output_digests[repo_relative(self.root,ctx.root/"outputs/provisional_mix.wav")]=digest_file(ctx.root/"outputs/provisional_mix.wav")
                    if sid in CACHEABLE:self.cache.store(key,{"output_digests":output_digests})
                rec={"stage_id":sid,"implementation_version":sid+"-v1","implementation_digest":self.impl,
                     "status":"SUCCEEDED","execution_mode":actual_mode,"inputs":list(case["paths"].values()),
                     "input_digests":inputs,"outputs":list(output_digests),"output_digests":output_digests,
                     "config_digest":self.config_digest,"cache_key":key,"source_commit":state["source_commit"],
                     "duration_ms":(time.perf_counter_ns()-started)//1_000_000,"return_code":0,
                     "failure":None,"reused":actual_mode=="REUSE","stale_reason":stale}
                state["stages"].append(rec); ctx.event({"stage_id":sid,"status":"SUCCEEDED","mode":actual_mode}); ctx.save(state)
                if fail_after==sid: raise InjectedFailure(f"injected after {sid}")
            state["status"]="SUCCEEDED"
        except Exception as e:
            state["status"]="FAILED"; state["failure_injected"]=isinstance(e,InjectedFailure)
            next_stage=next((x for x in self.graph.order if x not in {r["stage_id"] for r in state["stages"]}),None)
            if next_stage:state["stages"].append({"stage_id":next_stage,"status":"FAILED","failure":str(e)})
        state["attempts"][-1].update({"ended_ns":time.time_ns(),"executed":executed,"reused":reused})
        state["reusedStageCount"]=sum(x.get("reused",False) for x in state["stages"])
        state["duplicateArtifactCount"]=0; ctx.save(state)
        (ctx.root/"run_manifest.json").write_text(json.dumps(state,indent=2)+"\n")
        return state
    def _reports(self,ctx,state,case):
        base=self.root/"reports"/f"demo_run_{ctx.run_id}"
        data={"run_id":ctx.run_id,"run_key":state["run_key"],"case_id":case["case_id"],
              "decision":case["decision"],"generation_mode":"REPLAY","publish":"provisional_mix.wav"}
        base.with_suffix(".json").write_text(json.dumps(data,indent=2)+"\n")
        with base.with_suffix(".csv").open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=data.keys(),lineterminator="\n");w.writeheader();w.writerow(data)
        base.with_suffix(".html").write_text(f"<html><body><h1>{html.escape(case['case_id'])}</h1><p>REPLAY · MANUAL_REVIEW · provisional only</p><audio controls src='../artifacts/runs/{ctx.run_id}/outputs/provisional_mix.wav'></audio></body></html>\n")
