"""Deterministic 12-case x 4-slot plan and honest local execution."""
import csv,hashlib,json,math,shutil,struct,time,wave
from pathlib import Path
from .contracts import digest_file,digest_value,repo_relative
from .model_adapter import CandidateResult
from .model_registry import probe_capabilities

SLOT_ORDER=("v2a_primary","v2a_temporal","t2a_dss","control")
def audio_metrics(path):
    with wave.open(str(path),"rb") as w:
        channels,rate,frames,width=w.getnchannels(),w.getframerate(),w.getnframes(),w.getsampwidth()
        raw=w.readframes(frames)
    values=struct.unpack("<"+"h"*(len(raw)//2),raw) if width==2 else ()
    peak=max((abs(x) for x in values),default=0)/32768
    return {"readable":True,"sample_rate":rate,"channels":channels,"duration_sec":frames/rate,
            "peak_abs":peak,"clip_ratio":sum(abs(x)>=32767 for x in values)/max(1,len(values)),
            "silence_ratio":sum(abs(x)<32 for x in values)/max(1,len(values))}
def make_control(path,seed,duration=1.0):
    path.parent.mkdir(parents=True,exist_ok=True); rate=16000; count=int(rate*duration)
    freq=220+(seed%8)*35
    samples=[int(2600*math.sin(2*math.pi*freq*i/rate)) for i in range(count)]
    with wave.open(str(path),"wb") as w:
        w.setnchannels(1);w.setsampwidth(2);w.setframerate(rate)
        w.writeframes(struct.pack("<"+"h"*count,*samples))
class CandidateMatrix:
    def __init__(self,root,config):
        self.root=Path(root).resolve();self.config=config;self.capabilities={x.adapter_id:x for x in probe_capabilities(config)}
        self.checkpoint_digests={aid:cap.checkpoint_digest
                                 for aid,cap in self.capabilities.items()}
    def plan(self):
        inventory=json.loads((self.root/self.config["inventory"]).read_text())["cases"]
        rejected=[x for x in inventory if x["decision"]=="REPAIR_REJECTED"]
        non_rejected=[x for x in inventory if x["decision"]!="REPAIR_REJECTED"]
        rejected_slots=min(2,len(rejected),self.config["case_count"])
        chosen=non_rejected[:self.config["case_count"]-rejected_slots]+rejected[:rejected_slots]
        rows=[]
        for case_index,case in enumerate(chosen):
            matrix_case=case["repair_id"]
            for slot_index,slot in enumerate(self.config["slots"]):
                adapter=slot["adapter_id"];seed=self.config["seed_base"]+case_index
                output=f"{self.config['output_dir']}/{matrix_case}/{slot['slot_id']}.wav"
                request={"matrix_case_id":matrix_case,"source_case_id":case["case_id"],"repair_id":case["repair_id"],
                    "slot_id":slot["slot_id"],"adapter_id":adapter,"requested_mode":slot["requested_mode"],
                    "resource_class":slot["resource_class"],"seed":seed,"timeout_sec":slot["timeout_sec"],
                    "video_path":case["paths"]["input_video"],"dss_path":case["paths"]["dss"],
                    "source_audio":case["paths"]["after"],"repair_decision":case["decision"],
                    "allowed_publish_type":"PROVISIONAL" if case["decision"]=="MANUAL_REVIEW" else "BLOCKED",
                    "expected_output":output,"case_order":case_index,"slot_order":slot_index}
                request["slot_key"]=digest_value({k:request[k] for k in ("matrix_case_id","slot_id","adapter_id","seed","requested_mode")})
                rows.append(request)
        outputs=[x["expected_output"] for x in rows]
        body={"matrix_id":self.config["matrix_id"],"source_commit":self.config["source_commit"],
              "config_digest":digest_value(self.config),"case_order":[x["repair_id"] for x in chosen],
              "slot_order":list(SLOT_ORDER),"planned_count":len(rows),"collision_count":len(outputs)-len(set(outputs)),
              "records":rows}
        body["manifest_digest"]=digest_value(body)
        return body
    def execute(self,resume=False):
        plan=self.plan(); prior={}
        report=self.root/"reports/w20_candidate_matrix_20260721.json"
        if resume and report.is_file():
            for r in json.loads(report.read_text())["records"]:prior[(r["matrix_case_id"],r["slot_id"])]=r
        rows=[]; stale=0; reused=0
        for req in plan["records"]:
            key=(req["matrix_case_id"],req["slot_id"]); old=prior.get(key)
            if old and old["status"]=="SUCCEEDED" and old.get("output_path"):
                p=self.root/old["output_path"]
                if p.is_file() and digest_file(p)==old["output_sha256"]:
                    old["cache_hit"]=True;old["resumed"]=True;rows.append(old);reused+=1;continue
                stale+=1
            cap=self.capabilities[req["adapter_id"]]; started=time.perf_counter_ns()
            common={**req,"model_revision":cap.model_revision,"checkpoint_digest":self.checkpoint_digests[req["adapter_id"]],
                    "queue_wait_ms":0,"peak_gpu_memory_mb":0.0,"cache_hit":False,"resumed":bool(old)}
            if cap.status not in {"READY","READY_WITH_LIMITS"}:
                rows.append({**common,"status":"BLOCKED","generation_mode":"LIVE","output_path":"","output_sha256":"",
                    "runtime_ms":0,"failure_code":cap.failure_code or cap.status,"failure_reason":cap.reason,"audio":None});continue
            if req["adapter_id"]=="replay":
                out=self.root/req["source_audio"];mode="REPLAY"
            elif req["adapter_id"]=="control":
                out=self.root/req["expected_output"];make_control(out,req["seed"]);mode="CONTROL"
            else:
                rows.append({**common,"status":"FAILED","generation_mode":"LIVE","output_path":"","output_sha256":"",
                    "runtime_ms":0,"failure_code":"ENTRYPOINT_NOT_IMPLEMENTED","failure_reason":"live adapter execution unavailable","audio":None});continue
            rows.append({**common,"status":"SUCCEEDED","generation_mode":mode,"output_path":repo_relative(self.root,out),
                "output_sha256":digest_file(out),"runtime_ms":(time.perf_counter_ns()-started)//1_000_000,
                "failure_code":"","failure_reason":"","audio":audio_metrics(out)})
        rows.sort(key=lambda x:(x["case_order"],x["slot_order"]))
        return plan,rows,{"stale_count":stale,"resume_reused":reused}
