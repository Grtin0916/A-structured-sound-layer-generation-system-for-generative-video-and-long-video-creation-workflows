#!/usr/bin/env python3
"""Verify one deleted successful output becomes stale and is the only rerun."""
import json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];CLI=ROOT/"scripts/run_candidate_matrix.py";CFG=ROOT/"configs/demo/candidate_matrix_12cases.yaml"
sys.path.insert(0,str(ROOT/"src"))
from soundlayer.runner.model_adapter import CandidateResult
from soundlayer.runner.resource_scheduler import ResourceScheduler
def run(resume=False):
    cmd=["python3",str(CLI),"--config",str(CFG)]
    if resume:cmd.append("--resume")
    started=time.perf_counter_ns();subprocess.run(cmd,cwd=ROOT,check=True,capture_output=True,text=True)
    return (time.perf_counter_ns()-started)//1_000_000
def main():
    initial=run();report=json.loads((ROOT/"reports/w20_candidate_matrix_20260721.json").read_text())
    controls=[x for x in report["records"] if x["status"]=="SUCCEEDED" and x["generation_mode"]=="CONTROL"]
    victim=controls[0];before={x["slot_key"]:x.get("output_sha256","") for x in controls[1:]}
    (ROOT/victim["output_path"]).unlink()
    resumed=run(True);after=json.loads((ROOT/"reports/w20_candidate_matrix_20260721.json").read_text())
    summary=after["summary"];unchanged=all(next(x for x in after["records"] if x["slot_key"]==k)["output_sha256"]==v for k,v in before.items())
    def timeout():raise TimeoutError("injected slot timeout")
    isolated=ResourceScheduler(1,1,1).run([
      {"resource_class":"gpu","call":timeout},
      {"resource_class":"cpu","call":lambda:CandidateResult("isolation","control","control","SUCCEEDED","CONTROL")}])
    result={"initialRunMs":initial,"resumeRunMs":resumed,"deletedSlotKey":victim["slot_key"],
            "staleRerunCount":summary["staleRerunCount"],"resumeReusedCount":summary["resumeReusedCount"],
            "unaffectedDigestMatch":unchanged,"duplicateArtifactCount":summary["duplicateArtifactCount"],
            "gpuCapacity":1,"cpuCapacity":4,"ioCapacity":8,
            "timeoutFailureCode":isolated[0]["failure_code"],"timeoutIsolationVerified":isolated[1].status=="SUCCEEDED",
            "status":"PASS"}
    assert result["staleRerunCount"]==1 and unchanged and result["duplicateArtifactCount"]==0
    assert result["timeoutFailureCode"]=="TIMEOUT" and result["timeoutIsolationVerified"]
    (ROOT/"reports/w20_scheduler_benchmark_20260721.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
if __name__=="__main__":main()
