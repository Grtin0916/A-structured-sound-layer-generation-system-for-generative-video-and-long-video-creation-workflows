from fastapi import FastAPI, Response, status

from src.serving.schemas import (
    ErrorResponse,
    HealthzResponse,
    MetadataResponse,
    PredictRequest,
    PredictResponse,
)

SERVICE_NAME = "baseline-v1-api"
API_VERSION = "v0.1"
MODEL_NAME = "baseline_v1"
MODEL_VERSION = "baseline_v1"
BACKEND = "onnxruntime"
ARTIFACT_PATH = "artifacts/onnx/baseline_v1.onnx"
PROVIDER_REQUESTED = "CPUExecutionProvider"
PROVIDER_ACTUAL = "CPUExecutionProvider"
INPUT_NAME = "mel"
OUTPUT_NAME = "recon_mel"
FIXED_INPUT_SHAPE = [1, 1, 64, 313]
FIXED_OUTPUT_SHAPE = [1, 1, 64, 313]
CONTRACT_TYPE = "fixed_shape"

app = FastAPI(
    title=SERVICE_NAME,
    version=API_VERSION,
    description=(
        "Temporary Python service adapter for baseline_v1. "
        "Current scope: /healthz, /metadata, /predict placeholder."
    ),
)


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
        provider_actual=PROVIDER_ACTUAL,
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
    summary="Predict placeholder",
)
async def predict(payload: PredictRequest, response: Response) -> PredictResponse:
    if payload.expected_input_shape != FIXED_INPUT_SHAPE:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return PredictResponse(
            instance_id=payload.instance_id,
            status="error",
            model_name=MODEL_NAME,
            backend=BACKEND,
            provider_actual=PROVIDER_ACTUAL,
            output_ref=None,
            output_shape=None,
            error=ErrorResponse(
                code="INVALID_INPUT_SHAPE",
                message=(
                    "expected_input_shape does not match current fixed-shape contract"
                ),
                details={
                    "expected_input_shape": FIXED_INPUT_SHAPE,
                    "received_input_shape": payload.expected_input_shape,
                    "contract_type": CONTRACT_TYPE,
                },
            ),
        )

    response.status_code = status.HTTP_501_NOT_IMPLEMENTED
    return PredictResponse(
        instance_id=payload.instance_id,
        status="error",
        model_name=MODEL_NAME,
        backend=BACKEND,
        provider_actual=PROVIDER_ACTUAL,
        output_ref=None,
        output_shape=None,
        error=ErrorResponse(
            code="MODEL_NOT_READY",
            message="predict placeholder is wired, but ORT inference is not attached yet",
            details={
                "artifact_path": ARTIFACT_PATH,
                "provider_requested": PROVIDER_REQUESTED,
                "provider_actual": PROVIDER_ACTUAL,
            },
        ),
    )
