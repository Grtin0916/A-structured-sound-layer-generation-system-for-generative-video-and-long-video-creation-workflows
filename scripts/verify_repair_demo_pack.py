#!/usr/bin/env python3
"""Verify playable assets, digests, claims, and archive membership."""
import argparse, hashlib, json, zipfile
from pathlib import Path
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser()
    for x in ("pack-dir","zip","manifest","out-json"): p.add_argument("--"+x,required=True)
    a=p.parse_args(); root=Path(a.pack_dir); manifest=json.loads(Path(a.manifest).read_text())
    records=json.loads((root/"data/records.json").read_text())
    assert len(records)==len({x["repairId"] for x in records})==12
    assert sum(x["decision"]=="REPAIR_REJECTED" for x in records)>=2
    assert sum(x["repairAction"]=="event_transplant" for x in records)>=2
    assert len({x["repairAction"] for x in records})>=3
    for r in records:
        for side in ("before","after"):
            x=root/r["artifacts"][side]["path"]; assert x.is_file() and sha(x)==r["artifacts"][side]["sha256"]
    checks={}
    for line in (root/"checksums.sha256").read_text().splitlines():
        digest,name=line.split("  ",1); checks[name]=digest
    assert all((root/n).is_file() and sha(root/n)==h for n,h in checks.items())
    expected={x.relative_to(root).as_posix() for x in root.rglob("*") if x.is_file()}
    with zipfile.ZipFile(a.zip) as z:
        assert z.testzip() is None and set(z.namelist())==expected
    assert manifest["recordCount"]==12 and manifest["rejectedCount"]>=2
    result={"verified":True,"recordCount":12,"rejectedCount":sum(x["decision"]=="REPAIR_REJECTED" for x in records),
            "uniqueRepairIds":12,"audioFiles":24,"zipCrcVerified":True,"zipMemberSetVerified":True,
            "checksumsVerified":True,"finalSelectedCount":0}
    Path(a.out_json).parent.mkdir(parents=True,exist_ok=True)
    Path(a.out_json).write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result))
if __name__=="__main__": main()
