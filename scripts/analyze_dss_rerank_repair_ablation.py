#!/usr/bin/env python3
import argparse,csv,json,struct,sys,zlib
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from soundlayer.experiments.paired_analysis import holm,summarize
def figure(path,values):
    w,h=360,180;pix=bytearray()
    for y in range(h):
        pix.append(0)
        for x in range(w):
            bar=min(2,x//120);height=min(150,int(abs(values[bar])*1000));inside=y>=h-height
            pix.extend((45,130,210) if inside else (245,245,245))
    def c(t,d):return struct.pack(">I",len(d))+t+d+struct.pack(">I",zlib.crc32(t+d)&0xffffffff)
    Path(path).write_bytes(b"\\x89PNG\\r\\n\\x1a\\n"+c(b"IHDR",struct.pack(">IIBBBBB",w,h,8,2,0,0,0))+c(b"IDAT",zlib.compress(bytes(pix)))+c(b"IEND",b""))
def main():
    p=argparse.ArgumentParser();p.add_argument("--input",required=True);p.add_argument("--out-json",required=True);p.add_argument("--out-csv",required=True)
    p.add_argument("--out-figure",required=True);p.add_argument("--bootstrap-resamples",type=int,default=10000);p.add_argument("--seed",type=int,default=0)
    a=p.parse_args();by=defaultdict(dict)
    with open(a.input,newline="") as f:
        for x in csv.DictReader(f):
            by[x["case_id"]][x["strategy_id"]]=None if x["proxy_score"] in ("","None") else float(x["proxy_score"])
    comparisons=[]
    for i,(name,left,right) in enumerate((("B-A","A","B"),("C-B","B","C"),("D-C","C","D"))):
        vals=[v[right]-v[left] for v in by.values() if v.get(left) is not None and v.get(right) is not None]
        comparisons.append({"comparison":name,**summarize(vals,a.bootstrap_resamples,a.seed+i),"case_deltas":vals})
    adjusted=holm([x["permutation_p"] for x in comparisons])
    for x,padj in zip(comparisons,adjusted):x["holm_adjusted_p"]=padj
    payload={"comparisons":comparisons,"interpretation":"proxy evidence on available Control/Replay cases only",
      "counterexample_analysis":"RMS-style proxy is not a universal quality score: headroom repair intentionally reduces it while improving clipping safety; B-A is also mixed across cases. No human preference is inferred."}
    Path(a.out_json).write_text(json.dumps(payload,indent=2)+"\n")
    flat=[{k:v for k,v in x.items() if k!="case_deltas"} for x in comparisons]
    with open(a.out_csv,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=flat[0].keys(),lineterminator="\n");w.writeheader()
        for x in flat:w.writerow({k:json.dumps(v) if isinstance(v,list) else v for k,v in x.items()})
    figure(a.out_figure,[x["mean_delta"] for x in comparisons]);print(json.dumps(flat,indent=2))
if __name__=="__main__":main()
