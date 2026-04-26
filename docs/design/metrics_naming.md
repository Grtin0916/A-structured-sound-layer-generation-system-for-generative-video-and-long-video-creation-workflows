# Week08 Metrics Naming Draft

Date: 2026-04-27  
Repo: A-structured-sound-layer-generation-system-for-generative-video-and-long-video-creation-workflows  
Stage: Week08 / S1 pre-acceptance  
Status: Draft

## 1. Purpose

This document freezes the first naming convention for Week08 observability work.

The goal is not to create a complete monitoring system today. The goal is to make `/metrics`, Prometheus scrape config, and the first Grafana dashboard use the same language from the beginning.

Week08 metrics must support three engineering questions:

- Is the serving contract healthy?
- Is the runtime path observable?
- Are artifacts and evaluation outputs visible as engineering evidence?

## 2. Naming Rules

All custom metrics in this repo should follow these rules:

- Use the `mainbase_` prefix.
- Use snake_case.
- Use base units such as `seconds` and `bytes`.
- Use `_total` for monotonically increasing counters.
- Avoid high-cardinality labels such as raw file path, request id, user id, prompt text, or exception message.
- Prefer bounded labels such as `endpoint`, `method`, `status`, `model_name`, `artifact_type`, `eval_category`, and `result`.
- Do not encode units in labels when the unit should be part of the metric name.

## 3. Metric Groups

### 3.1 Contract Metrics

Contract metrics describe API boundary health, request validity, schema compatibility, and contract-level failures.

| Metric name | Type | Labels | Meaning |
| --- | --- | --- | --- |
| `mainbase_contract_requests_total` | Counter | `endpoint`, `method`, `status` | Total number of API requests observed at the serving boundary. |
| `mainbase_contract_request_duration_seconds` | Histogram | `endpoint`, `method` | Request latency at the API contract boundary. |
| `mainbase_contract_validation_errors_total` | Counter | `endpoint`, `schema`, `result` | Total number of schema or request validation failures. |
| `mainbase_contract_predict_failures_total` | Counter | `failure_type`, `endpoint` | Total number of `/predict` contract-level failures. |
| `mainbase_contract_response_bytes` | Histogram | `endpoint` | Response payload size in bytes. |

### 3.2 Runtime Metrics

Runtime metrics describe model loading, inference execution, runtime errors, and resource-sensitive behavior.

| Metric name | Type | Labels | Meaning |
| --- | --- | --- | --- |
| `mainbase_runtime_inference_requests_total` | Counter | `model_name`, `provider`, `result` | Total number of inference requests sent to the runtime layer. |
| `mainbase_runtime_inference_duration_seconds` | Histogram | `model_name`, `provider` | End-to-end inference duration in seconds. |
| `mainbase_runtime_model_load_seconds` | Histogram | `model_name`, `provider` | Model loading duration in seconds. |
| `mainbase_runtime_model_loaded` | Gauge | `model_name`, `provider` | Whether the model is currently loaded. |
| `mainbase_runtime_exceptions_total` | Counter | `component`, `exception_type` | Runtime exceptions grouped by bounded component and exception type. |

### 3.3 Artifact Metrics

Artifact metrics describe generated files, exported models, scorecard rows, manifests, and evidence completeness.

| Metric name | Type | Labels | Meaning |
| --- | --- | --- | --- |
| `mainbase_artifact_exports_total` | Counter | `artifact_type`, `result` | Total number of artifact export attempts. |
| `mainbase_artifact_export_duration_seconds` | Histogram | `artifact_type` | Artifact export duration in seconds. |
| `mainbase_artifact_size_bytes` | Gauge | `artifact_type`, `name` | Size of selected bounded artifacts in bytes. |
| `mainbase_artifact_scorecard_rows_total` | Gauge | `eval_category` | Number of scorecard rows by evaluation category. |
| `mainbase_artifact_manifest_entries_total` | Gauge | `manifest_name` | Number of entries in selected manifest files. |

## 4. Labels Allowed in Week08

Allowed low-cardinality labels:

| Label | Allowed examples | Notes |
| --- | --- | --- |
| `endpoint` | `/healthz`, `/metadata`, `/predict`, `/metrics` | Do not include query strings. |
| `method` | `GET`, `POST` | HTTP method only. |
| `status` | `2xx`, `4xx`, `5xx` | Prefer status family over raw status code for early dashboard. |
| `model_name` | `baseline_v1` | Bounded model alias only. |
| `provider` | `cpu`, `cuda`, `onnxruntime_cpu` | Runtime provider family. |
| `artifact_type` | `onnx`, `manifest`, `scorecard`, `wav`, `log` | Bounded artifact class. |
| `eval_category` | `functional`, `performance`, `failure_regression` | Must match scorecard categories. |
| `result` | `success`, `failure`, `skipped` | Bounded result value. |
| `component` | `api`, `runtime`, `artifact`, `eval` | Bounded component name. |
| `exception_type` | `ValidationError`, `RuntimeError`, `ImportError` | Exception class name only. |

Forbidden or discouraged labels:

| Label pattern | Reason |
| --- | --- |
| `request_id` | High cardinality. |
| `file_path` | High cardinality and unstable across machines. |
| `prompt` | High cardinality and privacy risk. |
| `raw_error_message` | High cardinality and unstable. |
| `user_id` | Not needed in this local engineering phase. |

## 5. Week08 Minimum `/metrics` Scope

The first `/metrics` endpoint should not pretend to cover the whole system.

Minimum acceptable scope:

- expose Prometheus text format at `/metrics`;
- include process or default Python client metrics if available;
- include at least one contract metric;
- include at least one runtime metric;
- include at least one artifact or scorecard-related metric if implementation cost remains low;
- keep all custom metric names aligned with this document.

## 6. Week08 Dashboard Boundary

The first Grafana dashboard should only show:

- request count;
- request latency;
- inference count;
- inference latency;
- error count;
- selected artifact or scorecard evidence count.

Do not include GPU, distributed tracing, MusicGen large-model benchmark, or production SLO panels in Week08 unless they already have reliable evidence.

## 7. Current Non-Goals

The following items are intentionally excluded from this draft:

- full alerting policy;
- SLO / SLA definition;
- production-grade cardinality budget;
- distributed tracing;
- multi-node deployment metrics;
- real cloud metrics;
- large-model comparative benchmark metrics.

These belong to later observability and performance milestones.

## 8. Next Implementation Steps

1. Add a minimal `/metrics` route or ASGI mount in the FastAPI serving entry.
2. Add `monitoring/prometheus/prometheus.yml`.
3. Add a minimal `mainbase-overview.json` dashboard.
4. Add smoke tests or curl evidence for `/metrics`.
5. Update README `Verified Scope`, `Not Yet Verified`, and `Next Hard Milestone` only after local evidence exists.
