"""Persistent run identity and event storage."""
import datetime, json, uuid
from pathlib import Path
class RunContext:
    def __init__(self,root,case_id,run_key,run_id=None):
        stamp=datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_id=run_id or f"{case_id}-{stamp}-{run_key[7:15]}-{uuid.uuid4().hex[:6]}"
        self.root=Path(root)/"artifacts/runs"/self.run_id
        for d in ("inputs","outputs","logs"): (self.root/d).mkdir(parents=True,exist_ok=True)
        self.state_path=self.root/"stage_state.json"; self.events_path=self.root/"events.jsonl"
    def save(self,value): self.state_path.write_text(json.dumps(value,indent=2)+"\n")
    def event(self,value):
        with self.events_path.open("a") as f:f.write(json.dumps(value,sort_keys=True)+"\n")
