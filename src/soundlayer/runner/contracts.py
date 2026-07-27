"""Stable hashing and stage contract helpers."""
import hashlib, json
from pathlib import Path

STATUSES={"PENDING","RUNNING","SUCCEEDED","FAILED","BLOCKED","SKIPPED","STALE"}
MODES={"EXECUTE","VERIFY","REUSE","BLOCKED","SKIP"}
def canonical(value): return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False)
def digest_bytes(data): return "sha256:"+hashlib.sha256(data).hexdigest()
def digest_file(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return "sha256:"+h.hexdigest()
def digest_value(value): return digest_bytes(canonical(value).encode())
def repo_relative(root,path):
    root=Path(root).resolve(); path=Path(path).resolve()
    try:return path.relative_to(root).as_posix()
    except ValueError: raise ValueError(f"path outside repository: {path}")
def implementation_digest(path): return digest_file(path)
def cache_key(stage_id,implementation_digest_value,config_digest,input_digests,execution_mode):
    return digest_value({"stage_id":stage_id,"implementation_digest":implementation_digest_value,
                         "config_digest":config_digest,"input_digests":dict(sorted(input_digests.items())),
                         "execution_mode":execution_mode})
