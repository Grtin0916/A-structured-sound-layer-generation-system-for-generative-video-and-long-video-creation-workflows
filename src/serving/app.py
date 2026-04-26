from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import onnxruntime as ort
import yaml
from fastapi import FastAPI, Response, status
from prometheus_client import make_asgi_app

from src.serving.schemas import (
    ErrorResponse,
    HealthzResponse,
    MetadataResponse,
    PredictRequest,
    PredictResponse,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs/serving/service_api.yaml"
PREDICT_ARTIFACT_DIR = REPO_ROOT / "artifacts/predict"
PREDICT_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def load_config() -> Dict[str, Any]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


CFG = load_config()

SERVICE_NAME = CFG["service"]["name"]
API_VERSION = CFG["service"]["api_version"]

BACKEND = CFG["runtime"]["backend"]
PROVIDER_REQUESTED = CFG["runtime"]["provider_requested"]

MODEL_NAME = CFG["model"]["name"]
MODEL_VERSION = CFG["model"]["version"]
ARTIFACT_PATH = CFG["model"]["artifact_path"]
INPUT_NAME = CFG["model"]["input_name"]
OUTPUT_NAME = CFG["model"]["output_name"]
FIXED_INPUT_SHAPE = CFG["model"]["input_shape"]
FIXED_OUTPUT_SHAPE = CFG["model"]["output_shape"]
CONTRACT_TYPE = CFG["model"]["contract_type"]

MODEL_PATH = REPO_ROOT / ARTIFACT_PATH


@lru_cache(maxsize=1)
def get_session() -> ort.InferenceSession:
    return ort.InferenceSession(
        str(MODEL_PATH),
        providers=[PROVIDER_REQUESTED],
    )


def get_provider_actual() -> str:
    try:
        providers = get_session().get_providers()
        if providers:
            return providers[0]
    except Exception:
        pass
    return PROVIDER_REQUESTED


def error_predict_response(
    *,
    instance_id: str,
    code: str,
    message: str,
    details: Dict[str, Any],
) -> PredictResponse:
    return PredictResponse(
        instance_id=instance_id,
        status="error",
        model_name=MODEL_NAME,
        backend=BACKEND,
        provider_actual=get_provider_actual(),
        output_ref=None,
        output_shape=None,
        error=ErrorResponse(
            code=code,
            message=message,
            details=details,
        ),
    )


def resolve_input_path(input_ref: str) -> Path:
    input_path = Path(input_ref)
    if not input_path.is_absolute():
        input_path = REPO_ROOT / input_ref
    return input_path


def load_input_array(input_path: Path) -> np.ndarray:
    array = np.load(input_path, allow_pickle=False)
    array = np.asarray(array, dtype=np.float32)
    return array


app = FastAPI(
    title=SERVICE_NAME,
    version=API_VERSION,
    description=(
        "Temporary Python service adapter for baseline_v1. "
        "Current scope: /healthz, /metadata, /predict, /metrics with ORT."
    ),
)

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.get(
    "/healthz",
    response_model=HealthzResponse,
    tags=["system"],
    summary="Health check",
)
async def healthz() -> HealthzResponse:
    return HealthzResponse(
        status="ok",
        service_name=SERVICE_NAME,
        api_version=API_VERSION,
    )


@app.get(
    "/metadata",
    response_model=MetadataResponse,
    tags=["system"],
    summary="Model and runtime metadata",
)
async def metadata() -> MetadataResponse:
    return MetadataResponse(
        service_name=SERVICE_NAME,
        api_version=API_VERSION,
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        backend=BACKEND,
        artifact_path=ARTIFACT_PATH,
        provider_requested=PROVIDER_REQUESTED,
        provider_actual=get_provider_actual(),
        input_name=INPUT_NAME,
        output_name=OUTPUT_NAME,
        input_shape=FIXED_INPUT_SHAPE,
        output_shape=FIXED_OUTPUT_SHAPE,
        contract_type=CONTRACT_TYPE,
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["inference"],
    summary="Run ORT inference on a local .npy mel tensor",
)
async def predict(payload: PredictRequest, response: Response) -> PredictResponse:
    if payload.expected_input_shape != FIXED_INPUT_SHAPE:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return error_predict_response(
            instance_id=payload.instance_id,
            code="INVALID_INPUT_SHAPE",
            message="expected_input_shape does not match current fixed-shape contract",
            details={
                "expected_input_shape": FIXED_INPUT_SHAPE,
                "received_input_shape": payload.expected_input_shape,
                "contract_type": CONTRACT_TYPE,
            },
        )

    input_path = resolve_input_path(payload.input_ref)
    if not input_path.exists():
        response.status_code = status.HTTP_400_BAD_REQUEST
        return error_predict_response(
            instance_id=payload.instance_id,
            code="INVALID_INPUT_REF",
            message="input_ref does not exist",
            details={"input_ref": payload.input_ref, "resolved_path": str(input_path)},
        )

    try:
        input_array = load_input_array(input_path)
    except Exception as exc:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return error_predict_response(
            instance_id=payload.instance_id,
            code="INVALID_INPUT_REF",
            message="failed to load input_ref as .npy tensor",
            details={
                "input_ref": payload.input_ref,
                "resolved_path": str(input_path),
                "exception_type": type(exc).__name__,
                "exception": str(exc),
            },
        )

    loaded_shape: List[int] = list(input_array.shape)
    if loaded_shape != FIXED_INPUT_SHAPE:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return error_predict_response(
            instance_id=payload.instance_id,
            code="INVALID_INPUT_SHAPE",
            message="loaded tensor shape does not match current fixed-shape contract",
            details={
                "expected_input_shape": FIXED_INPUT_SHAPE,
                "loaded_input_shape": loaded_shape,
                "resolved_path": str(input_path),
            },
        )

    try:
        session = get_session()
        output_array = session.run([OUTPUT_NAME], {INPUT_NAME: input_array})[0]
        output_array = np.asarray(output_array, dtype=np.float32)
    except Exception as exc:
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return error_predict_response(
            instance_id=payload.instance_id,
            code="INFERENCE_FAILED",
            message="ORT inference failed",
            details={
                "artifact_path": ARTIFACT_PATH,
                "provider_requested": PROVIDER_REQUESTED,
                "provider_actual": get_provider_actual(),
                "exception_type": type(exc).__name__,
                "exception": str(exc),
            },
        )

    output_path = PREDICT_ARTIFACT_DIR / f"{payload.instance_id}_recon_mel.npy"
    np.save(output_path, output_array)

    return PredictResponse(
        instance_id=payload.instance_id,
        status="ok",
        model_name=MODEL_NAME,
        backend=BACKEND,
        provider_actual=get_provider_actual(),
        output_ref=str(output_path.relative_to(REPO_ROOT)),
        output_shape=list(output_array.shape),
        error=None,
    )
