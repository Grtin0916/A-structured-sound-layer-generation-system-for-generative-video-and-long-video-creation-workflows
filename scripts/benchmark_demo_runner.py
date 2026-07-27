#!/usr/bin/env python3
"""Exercise failure, cross-process resume, and idempotent replay."""
import csv,hashlib,json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CLI=ROOT/"scripts/run_demo.py"; CONFIG=ROOT/"configs/demo/runner.yaml"
def call(args):
    started=time.perf_counter_ns()
    result=subprocess.run(["python3",str(CLI),*args],cwd=ROOT,text=True,capture_output=True,check=True)
    return json.loads(result.stdout),(time.perf_counter_ns()-started)//1_000_000
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    failed,failed_ms=call(["run","--config",str(CONFIG),"--mode","replay","--fail-after","evaluate"])
    assert failed["status"]=="FAILED" and failed["failure_injected"]
    upstream={x["stage_id"]:x["output_digests"] for x in failed["stages"] if x["status"]=="SUCCEEDED"}
    resumed,resume_ms=call(["resume","--config",str(CONFIG),"--run-id",failed["run_id"]])
    assert resumed["status"]=="SUCCEEDED"
    assert all(next(x for x in resumed["stages"] if x["stage_id"]==sid)["output_digests"]==d for sid,d in upstream.items())
    rerun,rerun_ms=call(["run","--config",str(CONFIG),"--mode","replay"])
    assert rerun["status"]=="SUCCEEDED" and rerun["run_key"]==resumed["run_key"]
    first=ROOT/"artifacts/runs"/resumed["run_id"]/"outputs/provisional_mix.wav"
    second=ROOT/"artifacts/runs"/rerun["run_id"]/"outputs/provisional_mix.wav"
    result={
      "schemaVersion":"demo-runner-recovery/v1","caseId":resumed["case"]["case_id"],
      "failedRunId":failed["run_id"],"resumedRunId":resumed["run_id"],"idempotentRunId":rerun["run_id"],
      "coldRunMs":failed_ms+resume_ms,"failedRunMs":failed_ms,"resumeRunMs":resume_ms,
      "idempotentRerunMs":rerun_ms,"stagesExecuted":rerun["attempts"][-1]["executed"],
      "stagesReused":rerun["attempts"][-1]["reused"],"cacheHits":rerun["attempts"][-1]["reused"],
      "cacheMisses":rerun["attempts"][-1]["executed"],"duplicateArtifactCount":rerun["duplicateArtifactCount"],
      "sameRunKey":True,"resultDigestMatch":sha(first)==sha(second),
      "provisionalOutputCount":2,"finalOutputCount":0,"blockedOutputCount":0,
      "generationMode":"REPLAY","decision":"MANUAL_REVIEW"
    }
    base=ROOT/"reports/demo_runner_recovery_20260720"
    base.with_suffix(".json").write_text(json.dumps(result,indent=2)+"\n")
    with base.with_suffix(".csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=result.keys(),lineterminator="\n");w.writeheader();w.writerow(result)
    print(json.dumps(result,indent=2))
if __name__=="__main__":main()
