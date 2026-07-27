#!/usr/bin/env python3
"""Build a playable, claim-aware W19 repair demo pack from Java blobs."""
import argparse, hashlib, html, json, shutil, subprocess, zipfile
from pathlib import Path

def load(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def dump(p,v):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def commit(root): return subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip()

def main():
    p=argparse.ArgumentParser()
    for x in ("mainbase-root","java-root","cloud-root","out-dir","out-zip","manifest","claim-boundary","walkthrough","weekly-summary"):
        p.add_argument("--"+x,required=True)
    a=p.parse_args(); mb,java,cloud=map(lambda x:Path(x).resolve(),(a.mainbase_root,a.java_root,a.cloud_root))
    report_path=java/"artifacts/manifests/repair_workflow_report_20260716.json"
    records=load(report_path)["records"]
    rejected=[x for x in records if x["repairDecision"]=="REPAIR_REJECTED"]
    transplant=[x for x in records if x["repairAction"]=="event_transplant" and x["repairDecision"]=="MANUAL_REVIEW"]
    manual=[x for x in records if x["repairDecision"]=="MANUAL_REVIEW" and x["repairAction"]!="event_transplant"]
    chosen=(rejected+transplant+manual[:8])[:12]
    out=Path(a.out_dir); shutil.rmtree(out,ignore_errors=True)
    for d in ("audio","plots","data"): (out/d).mkdir(parents=True,exist_ok=True)
    cards=[]; packed=[]
    for r in chosen:
        rid=r["repairId"]; item={"repairId":rid,"failureId":r["failureId"],"repairAction":r["repairAction"],
            "sourceMode":r["sourceMode"],"decision":r["repairDecision"],"workflowState":r["workflowState"],
            "statusLabel":"REPAIR_REJECTED" if r["repairDecision"]=="REPAIR_REJECTED" else "PROXY_IMPROVED · REVIEW_PENDING",
            "metrics":r["metrics"],"reason":r["reason"],"artifacts":{}}
        for side in ("before","after"):
            src=java/r[side]["materializedPath"]; dst=out/"audio"/f"{rid}__{side}.wav"
            shutil.copy2(src,dst)
            item["artifacts"][side]={"path":dst.relative_to(out).as_posix(),"sha256":sha(dst),
                                      "javaMaterializedPath":r[side]["materializedPath"]}
        possible=list((mb/"artifacts/repair").glob(f"**/{r['failureId']}/comparison.png"))
        if possible:
            dst=out/"plots"/f"{rid}.png"; shutil.copy2(possible[-1],dst)
            item["comparisonPlot"]={"available":True,"path":dst.relative_to(out).as_posix(),"sha256":sha(dst)}
        else: item["comparisonPlot"]={"available":False,"status":"METRIC_UNAVAILABLE"}
        packed.append(item)
        plot=(f'<img src="{item["comparisonPlot"]["path"]}" alt="comparison plot">' if item["comparisonPlot"]["available"]
              else "<p>Comparison plot: METRIC_UNAVAILABLE</p>")
        cards.append(f'''<article><h2>{html.escape(rid)}</h2><p class="status">{item["statusLabel"]}</p>
<p>{html.escape(r["repairAction"])} · {html.escape(r["sourceMode"])}</p>
<label>Before<audio controls src="{item["artifacts"]["before"]["path"]}"></audio></label>
<label>After<audio controls src="{item["artifacts"]["after"]["path"]}"></audio></label>{plot}
<details><summary>Decision and integrity evidence</summary><pre>{html.escape(json.dumps(item,indent=2))}</pre></details></article>''')
    dump(out/"data/records.json",packed)
    provenance={"schemaVersion":"repair-demo-provenance/v1","commits":{"mainbase":commit(mb),"java":commit(java),"cloud":commit(cloud)},
                "inputs":[str(report_path),str(java/"artifacts/manifests/repair_artifact_index_20260716.json"),
                          str(cloud/"loadtest/reports/repair_observability_20260717.json")],
                "builder":"scripts/build_repair_demo_pack.py","slsaCompliant":False,"signedAttestation":False}
    dump(out/"provenance.json",provenance)
    page='''<!doctype html><html><head><meta charset="utf-8"><title>W19 Repair Demo</title>
<style>body{font-family:sans-serif;max-width:1100px;margin:auto}article{border:1px solid #bbb;padding:1rem;margin:1rem 0}audio,img{display:block;max-width:100%;margin:.5rem 0}.status{font-weight:bold}</style>
</head><body><h1>Integrity-aware repair demo</h1><p>Proxy improvement is not human preference. Manual listening completed: 0. Final selected: 0.</p>'''+''.join(cards)+"</body></html>\n"
    (out/"index.html").write_text(page,encoding="utf-8")
    files=sorted(x for x in out.rglob("*") if x.is_file())
    manifest={"schemaVersion":"repair-demo-manifest/v1","recordCount":len(packed),
              "rejectedCount":sum(x["decision"]=="REPAIR_REJECTED" for x in packed),
              "records":[x["repairId"] for x in packed],
              "files":[{"path":x.relative_to(out).as_posix(),"sha256":sha(x),"sizeBytes":x.stat().st_size} for x in files]}
    dump(out/"manifest.json",manifest)
    files=sorted(x for x in out.rglob("*") if x.is_file())
    (out/"checksums.sha256").write_text("".join(f"{sha(x)}  {x.relative_to(out).as_posix()}\n" for x in files),encoding="utf-8")
    dump(a.manifest,manifest)
    boundary={"proxyImprovementIsHumanPreference":False,"manualListeningCompletedCount":0,"finalSelectedCount":0,
              "automaticForbiddenEventDetectionAvailable":False,"onsetMetricAvailable":False,
              "productionPrometheusVerified":False,"liveGrafanaImportVerified":False,
              "alertmanagerConfigured":False,"productionAlertingVerified":False,
              "notClaimFullRepairEngine":True,"slsaCompliant":False,"signedAttestation":False}
    dump(a.claim_boundary,boundary)
    Path(a.walkthrough).parent.mkdir(parents=True,exist_ok=True)
    Path(a.walkthrough).write_text("# Repair engine demo walkthrough\n\nServe the pack directory with `python3 -m http.server 8000`, then open `index.html`. Compare before/after evidence, note that proxy improvement remains pending human review, inspect both rejected transplant diagnostics, and finish with hashes and provenance. No final selection or production deployment is claimed.\n",encoding="utf-8")
    Path(a.weekly_summary).parent.mkdir(parents=True,exist_ok=True)
    Path(a.weekly_summary).write_text("# W19 repair summary\n\nThe chain now covers failure contracts, minimal repair probes, constraint-first reranking, Java content-addressed review orchestration, Cloud decision-funnel observability, and a verified playable pack. Current truth remains 18 manual-review decisions, 2 rejected decisions, 0 completed listening reviews, and 0 final selections.\n",encoding="utf-8")
    z=Path(a.out_zip); z.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(z,"w",zipfile.ZIP_DEFLATED) as f:
        for x in sorted(y for y in out.rglob("*") if y.is_file()): f.write(x,x.relative_to(out).as_posix())
    print(json.dumps({"records":len(packed),"rejected":manifest["rejectedCount"],"zipSha256":sha(z)}))
if __name__=="__main__": main()
