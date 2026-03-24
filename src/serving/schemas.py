from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class HealthzResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"] = "ok"
    service_name: str = Field(..., description="Service identifier")
    api_version: str = Field(..., description="API version string")


class MetadataResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    service_name: str = Field(..., description="Service identifier")
    api_version: str = Field(..., description="API version string")
    model_name: str = Field(..., description="Logical model name")
    model_version: str = Field(..., description="Model version tag")
    backend: str = Field(..., description="Inference backend name")
    artifact_path: str = Field(..., description="Path to model artifact")
    provider_requested: str = Field(..., description="Configured execution provider")
    provider_actual: str = Field(..., description="Actual execution provider at runtime")
    input_name: str = Field(..., description="Model input tensor name")
    output_name: str = Field(..., description="Model output tensor name")
    input_shape: List[int] = Field(..., description="Declared input tensor shape")
    output_shape: List[int] = Field(..., description="Declared output tensor shape")
    contract_type: Literal["fixed_shape"] = Field(
        "fixed_shape",
        description="Current contract type",
    )


class ErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable error message")
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional structured error details",
    )


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instance_id: str = Field(..., description="Request instance identifier")
    input_ref: str = Field(..., description="Input reference, currently local path string")
    input_type: Literal["mel_npy", "tensor_ref"] = Field(
        ...,
        description="Minimal supported input type",
    )
    expected_input_shape: List[int] = Field(
        ...,
        description="Client-declared expected input shape",
        min_length=4,
        max_length=4,
    )

    media_ref: Optional[str] = Field(
        default=None,
        description="Reserved: upstream media reference",
    )
    segment_id: Optional[str] = Field(
        default=None,
        description="Reserved: segment identifier",
    )
    blueprint_id: Optional[str] = Field(
        default=None,
        description="Reserved: audio blueprint identifier",
    )
    output_artifact_dir: Optional[str] = Field(
        default=None,
        description="Reserved: preferred artifact output directory",
    )
    generator_name: Optional[str] = Field(
        default=None,
        description="Reserved: future generator backend name",
    )
    request_tags: Optional[List[str]] = Field(
        default=None,
        description="Reserved: request tags for later tracing/routing",
    )


class PredictResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instance_id: str = Field(..., description="Request instance identifier")
    status: Literal["accepted", "ok", "error"] = Field(
        ...,
        description="High-level request status",
    )
    model_name: str = Field(..., description="Logical model name")
    backend: str = Field(..., description="Inference backend name")
    provider_actual: str = Field(..., description="Actual execution provider at runtime")
    output_ref: Optional[str] = Field(
        default=None,
        description="Output artifact reference or placeholder",
    )
    output_shape: Optional[List[int]] = Field(
        default=None,
        description="Produced output shape if available",
    )
    error: Optional[ErrorResponse] = Field(
        default=None,
        description="Structured error object when request fails",
    )