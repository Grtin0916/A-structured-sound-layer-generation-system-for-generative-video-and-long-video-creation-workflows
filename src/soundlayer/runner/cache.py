"""Content-addressed stage cache with digest validation."""
import json
from pathlib import Path
from .contracts import digest_file
class StageCache:
    def __init__(self,root): self.root=Path(root); self.root.mkdir(parents=True,exist_ok=True)
    def lookup(self,key,repo_root):
        p=self.root/f"{key.removeprefix('sha256:')}.json"
        if not p.is_file(): return False,"MISS",None
        value=json.loads(p.read_text())
        for rel,digest in value["output_digests"].items():
            target=Path(repo_root)/rel
            if not target.is_file(): return False,"OUTPUT_MISSING",value
            if digest_file(target)!=digest:return False,"OUTPUT_DIGEST_MISMATCH",value
        return True,None,value
    def store(self,key,value):
        (self.root/f"{key.removeprefix('sha256:')}.json").write_text(json.dumps(value,indent=2)+"\n")
