"""Adapter registry and environment capability probes."""
import hashlib,importlib.util,os
from pathlib import Path
from .model_adapter import AdapterCapability

class ModelRegistry:
    def __init__(self): self._items={}
    def register(self,adapter):
        if adapter.adapter_id in self._items: raise ValueError(f"duplicate adapter_id: {adapter.adapter_id}")
        self._items[adapter.adapter_id]=adapter
    def get(self,adapter_id):
        if adapter_id not in self._items: raise KeyError(f"unknown adapter: {adapter_id}")
        return self._items[adapter_id]
    def ids(self): return sorted(self._items)

def _checkpoint(repo,names):
    for name in names:
        p=repo/name
        if p.is_file(): return p
    return None
def _sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return "sha256:"+h.hexdigest()
def probe_capabilities(config):
    results=[]
    for spec in config["adapters"]:
        aid=spec["adapter_id"]; repo=Path(os.path.expanduser(spec.get("repository",""))).resolve()
        base=dict(adapter_id=aid,task_type=spec["task_type"],resource_class=spec["resource_class"],
                  repository=str(repo),imports=tuple(spec.get("required_imports",[])),
                  entrypoint=spec.get("entrypoint",""),model_revision=spec.get("model_revision",""),
                  license_url=spec.get("license_url",""),license_accepted=spec.get("license_accepted",False),
                  estimated_vram_mb=spec.get("estimated_vram_mb",0))
        if aid in {"replay","control"}:
            results.append(AdapterCapability(**base,status="READY",device="cpu",reason="stdlib execution path"));continue
        if not repo.is_dir():
            results.append(AdapterCapability(**base,status="REPOSITORY_MISSING",failure_code="REPOSITORY_MISSING",reason=f"repository not found: {repo}"));continue
        checkpoint=_checkpoint(repo,spec.get("checkpoint_candidates",[]))
        if not checkpoint:
            results.append(AdapterCapability(**base,status="CHECKPOINT_MISSING",failure_code="CHECKPOINT_MISSING",reason="no configured checkpoint exists"));continue
        missing=[x for x in spec.get("required_imports",[]) if importlib.util.find_spec(x) is None]
        if missing:
            results.append(AdapterCapability(**base,status="DEPENDENCY_MISSING",checkpoint=str(checkpoint),
                checkpoint_digest=_sha(checkpoint),checkpoint_size_bytes=checkpoint.stat().st_size,
                failure_code="DEPENDENCY_MISSING",reason="missing imports: "+",".join(missing)));continue
        try:
            import torch
            cuda=torch.cuda.is_available()
        except Exception: cuda=False
        if not cuda:
            results.append(AdapterCapability(**base,status="GPU_UNAVAILABLE",checkpoint=str(checkpoint),
                checkpoint_digest=_sha(checkpoint),checkpoint_size_bytes=checkpoint.stat().st_size,
                failure_code="GPU_UNAVAILABLE",reason="CUDA runtime/device unavailable"));continue
        if not spec.get("license_accepted",False):
            results.append(AdapterCapability(**base,status="LICENSE_NOT_ACCEPTED",checkpoint=str(checkpoint),
                checkpoint_digest=_sha(checkpoint),checkpoint_size_bytes=checkpoint.stat().st_size,
                failure_code="LICENSE_NOT_ACCEPTED",reason="model license not accepted"));continue
        results.append(AdapterCapability(**base,status="READY",checkpoint=str(checkpoint),
            checkpoint_digest=_sha(checkpoint),checkpoint_size_bytes=checkpoint.stat().st_size,device="cuda",dtype="float16"))
    return results
