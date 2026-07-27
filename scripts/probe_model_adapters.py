#!/usr/bin/env python3
import argparse,csv,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from soundlayer.runner.model_registry import probe_capabilities
def main():
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--out-json",required=True);p.add_argument("--out-csv",required=True)
    a=p.parse_args();config=json.loads(Path(a.config).read_text());rows=[x.dict() for x in probe_capabilities(config)]
    Path(a.out_json).parent.mkdir(parents=True,exist_ok=True);Path(a.out_json).write_text(json.dumps({"adapters":rows},indent=2)+"\n")
    flat=[{**x,"imports":",".join(x["imports"])} for x in rows]
    with Path(a.out_csv).open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=flat[0].keys(),lineterminator="\n");w.writeheader();w.writerows(flat)
    print(json.dumps({"counts":{s:sum(x["status"]==s for x in rows) for s in sorted({x["status"] for x in rows})}}))
if __name__=="__main__":main()
