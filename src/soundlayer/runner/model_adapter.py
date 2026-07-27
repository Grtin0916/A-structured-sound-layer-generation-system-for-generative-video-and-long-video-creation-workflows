"""Unified contracts for replay, control, and generative adapters."""
from dataclasses import asdict,dataclass
from pathlib import Path
from typing import Protocol

@dataclass(frozen=True)
class AdapterCapability:
    adapter_id:str; task_type:str; resource_class:str; status:str
    repository:str=""; imports:tuple=(); checkpoint:str=""; checkpoint_digest:str=""; checkpoint_size_bytes:int=0
    device:str="cpu"; dtype:str="float32"
    entrypoint:str=""; model_revision:str=""; license_url:str=""; license_accepted:bool=False
    estimated_vram_mb:int=0; failure_code:str=""; reason:str=""
    def dict(self): return asdict(self)
@dataclass(frozen=True)
class CandidateRequest:
    case_id:str; slot_id:str; video_path:str; dss_path:str; source_audio:str
    repair_decision:str; requested_mode:str; seed:int; timeout_sec:int; expected_output:str
@dataclass(frozen=True)
class CandidatePlan:
    adapter_id:str; resource_class:str; timeout_sec:int; expected_output:str
    command:tuple=(); checkpoint_digest:str=""; config_digest:str=""; input_digest:str=""
@dataclass
class CandidateResult:
    case_id:str; slot_id:str; adapter_id:str; status:str; generation_mode:str
    output_path:str=""; output_sha256:str=""; model_revision:str=""; checkpoint_digest:str=""
    runtime_ms:int=0; queue_wait_ms:int=0; peak_gpu_memory_mb:float=0.0
    failure_code:str=""; failure_reason:str=""; resumed:bool=False; cache_hit:bool=False
class ModelAdapter(Protocol):
    adapter_id:str; task_type:str; resource_class:str
    def probe(self)->AdapterCapability:...
    def plan(self,request:CandidateRequest)->CandidatePlan:...
    def execute(self,request:CandidateRequest,plan:CandidatePlan)->CandidateResult:...
