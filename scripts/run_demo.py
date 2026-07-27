#!/usr/bin/env python3
import argparse,csv,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from soundlayer.runner import DemoRunner
def load(p): return json.loads(Path(p).read_text())
def dump_inventory(runner):
    rows=runner.inventory(); j=ROOT/"reports/demo_case_inventory_20260720.json"; c=j.with_suffix(".csv")
    j.write_text(json.dumps({"cases":rows,"selected":runner.choose()},indent=2)+"\n")
    with c.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["case_id","repair_id","decision","score","metrics_available"],lineterminator="\n")
        w.writeheader();w.writerows({k:x[k] for k in w.fieldnames} for x in rows)
    return {"json":str(j),"selected":runner.choose()}
def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest="command",required=True)
    for name in ("inventory","plan","run","resume","status","verify"):
        q=sub.add_parser(name);q.add_argument("--config",required=True)
        q.add_argument("--case-id");q.add_argument("--mode",default="replay")
        q.add_argument("--run-id");q.add_argument("--fail-after")
    a=p.parse_args(); runner=DemoRunner(ROOT,load(a.config))
    if a.command=="inventory": out=dump_inventory(runner)
    elif a.command=="plan": out=runner.plan(a.case_id)
    elif a.command=="run": out=runner.run(a.case_id,a.fail_after)
    elif a.command=="resume": out=runner.run(resume_run_id=a.run_id)
    elif a.command=="status": out=load(ROOT/"artifacts/runs"/a.run_id/"stage_state.json")
    else:
        out=load(ROOT/"artifacts/runs"/a.run_id/"run_manifest.json")
        from soundlayer.runner.contracts import digest_file
        for rel,d in out["input_digests"].items():
            assert digest_file(ROOT/out["case"]["paths"][rel])==d
        for s in out["stages"]:
            for rel,d in s.get("output_digests",{}).items():
                assert digest_file(ROOT/rel)==d
        out={"verified":True,"run_id":a.run_id}
    print(json.dumps(out,indent=2))
if __name__=="__main__":main()
