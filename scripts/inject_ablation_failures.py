#!/usr/bin/env python3
import argparse,json
from pathlib import Path
def main():
    p=argparse.ArgumentParser();p.add_argument("--experiment-dir",required=True);p.add_argument("--scenario",required=True);p.add_argument("--case-id",required=True)
    a=p.parse_args();root=Path(a.experiment_dir);case=root/a.case_id
    faults_path=root/".faults.json";faults=json.loads(faults_path.read_text()) if faults_path.is_file() else []
    if a.scenario=="corrupt_metrics":(case/"metrics.json").write_text('{"invalid": NaN')
    elif a.scenario=="delete_selected_audio":(case/"strategy_c_selected.wav").unlink(missing_ok=True)
    elif a.scenario=="repair_timeout":(case/"repair_timeout.injected").write_text("TIMEOUT\n")
    else:raise ValueError("unknown scenario")
    faults.append({"scenario":a.scenario,"case_id":a.case_id});faults_path.write_text(json.dumps(faults,indent=2)+"\n")
    print(json.dumps(faults[-1]))
if __name__=="__main__":main()
