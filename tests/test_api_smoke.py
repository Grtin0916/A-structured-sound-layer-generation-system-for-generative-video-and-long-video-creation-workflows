from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from src.serving.app import app, FIXED_INPUT_SHAPE, REPO_ROOT

client = TestClient(app)


def test_healthz():
    resp = client.get("/healthz")
    assert resp.status_code == 200

    data = resp.json()
    assert data["status"] == "ok"
    assert data["service_name"] == "baseline-v1-api"
    assert data["api_version"] == "v0.1"


def test_metadata():
    resp = client.get("/metadata")
    assert resp.status_code == 200

    data = resp.json()
    assert data["model_name"] == "baseline_v1"
    assert data["backend"] == "onnxruntime"
    assert data["input_shape"] == [1, 1, 64, 313]
    assert data["output_shape"] == [1, 1, 64, 313]
    assert data["contract_type"] == "fixed_shape"


def test_metrics_endpoint_exposes_prometheus_text():
    resp = client.get("/metrics")
    assert resp.status_code in (200, 307)

    if resp.status_code == 307:
        resp = client.get("/metrics/")
        assert resp.status_code == 200

    assert "text/plain" in resp.headers["content-type"]
    body = resp.text
    assert "# HELP" in body
    assert "# TYPE" in body


def test_predict_happy_path(tmp_path):
    input_path = tmp_path / "sample.npy"
    x = np.random.default_rng(42).standard_normal(FIXED_INPUT_SHAPE).astype(np.float32)
    np.save(input_path, x)

    resp = client.post(
        "/predict",
        json={
            "instance_id": "pytest-week03-tue-001",
            "input_ref": str(input_path),
            "input_type": "mel_npy",
            "expected_input_shape": [1, 1, 64, 313],
        },
    )
    assert resp.status_code == 200

    data = resp.json()
    assert data["status"] == "ok"
    assert data["model_name"] == "baseline_v1"
    assert data["backend"] == "onnxruntime"
    assert data["output_shape"] == [1, 1, 64, 313]
    assert data["error"] is None

    output_ref = data["output_ref"]
    assert output_ref is not None

    output_path = REPO_ROOT / output_ref
    assert output_path.exists()


def test_predict_invalid_shape(tmp_path):
    input_path = tmp_path / "sample_bad.npy"
    x = np.random.default_rng(7).standard_normal(FIXED_INPUT_SHAPE).astype(np.float32)
    np.save(input_path, x)

    resp = client.post(
        "/predict",
        json={
            "instance_id": "pytest-week03-tue-bad-shape",
            "input_ref": str(input_path),
            "input_type": "mel_npy",
            "expected_input_shape": [1, 1, 64, 312],
        },
    )
    assert resp.status_code == 400

    data = resp.json()
    assert data["status"] == "error"
    assert data["error"]["code"] == "INVALID_INPUT_SHAPE"


def test_predict_invalid_input_ref():
    resp = client.post(
        "/predict",
        json={
            "instance_id": "pytest-week04-mon-missing-ref",
            "input_ref": "tmp/definitely_not_exists/sample.npy",
            "input_type": "mel_npy",
            "expected_input_shape": [1, 1, 64, 313],
        },
    )
    assert resp.status_code == 400

    data = resp.json()
    assert data["status"] == "error"
    assert data["error"]["code"] == "INVALID_INPUT_REF"
    assert data["error"]["message"] == "input_ref does not exist"
    assert data["error"]["details"]["input_ref"] == "tmp/definitely_not_exists/sample.npy"
    assert "resolved_path" in data["error"]["details"]


def test_predict_invalid_input_ref_bad_npy(tmp_path):
    input_path = tmp_path / "broken.npy"
    input_path.write_bytes(b"this is not a valid npy file")

    resp = client.post(
        "/predict",
        json={
            "instance_id": "pytest-week04-tue-bad-npy",
            "input_ref": str(input_path),
            "input_type": "mel_npy",
            "expected_input_shape": [1, 1, 64, 313],
        },
    )
    assert resp.status_code == 400

    data = resp.json()
    assert data["status"] == "error"
    assert data["error"]["code"] == "INVALID_INPUT_REF"
    assert data["error"]["message"] == "failed to load input_ref as .npy tensor"
    assert data["error"]["details"]["input_ref"] == str(input_path)
    assert "resolved_path" in data["error"]["details"]
    assert "exception_type" in data["error"]["details"]


def test_predict_inference_failed(tmp_path, monkeypatch):
    import src.serving.app as serving_app

    class FakeSession:
        def run(self, *args, **kwargs):
            raise RuntimeError("synthetic ort failure")

    monkeypatch.setattr(serving_app, "get_session", lambda: FakeSession())

    input_path = tmp_path / "valid.npy"
    x = np.random.default_rng(123).standard_normal(FIXED_INPUT_SHAPE).astype(np.float32)
    np.save(input_path, x)

    resp = client.post(
        "/predict",
        json={
            "instance_id": "pytest-week04-tue-infer-fail",
            "input_ref": str(input_path),
            "input_type": "mel_npy",
            "expected_input_shape": [1, 1, 64, 313],
        },
    )
    assert resp.status_code == 500

    data = resp.json()
    assert data["status"] == "error"
    assert data["error"]["code"] == "INFERENCE_FAILED"
    assert data["error"]["message"] == "ORT inference failed"
    assert data["output_ref"] is None
    assert data["output_shape"] is None
    assert data["error"]["details"]["exception_type"] == "RuntimeError"
