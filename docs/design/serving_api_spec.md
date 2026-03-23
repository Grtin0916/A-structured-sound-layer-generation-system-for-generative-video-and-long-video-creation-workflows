# serving_api_spec

## 1. Purpose

本文档定义 Week 03 当前阶段的最小服务接口合同。

目标不是立即做完整推理服务，而是先把当前 `baseline_v1` 的 HTTP 调用边界、字段语义、返回结构、错误格式与未来扩展位冻结下来，作为后续：

- 临时 Python API 适配层
- Java REST / gRPC 迁移
- Compose 本地栈接入
- 日志与 artifact 跟踪

的统一接口基础。

当前阶段只覆盖最小端点：

- `/healthz`
- `/metadata`
- `/predict`

---

## 2. Scope

### 2.1 In scope

当前文档只定义：

- 路径与 HTTP 方法
- 请求体 / 响应体的最小字段
- 当前模型与推理后端的元信息返回口径
- 当前错误响应最小格式
- 当前日志与 artifact 记录原则
- 未来音频蓝图桥接的保留字段

### 2.2 Out of scope

当前阶段明确不包括：

- 鉴权
- 流式响应
- 批量预测任务调度
- 异步任务队列
- 数据库存储
- Java / gRPC 正式服务
- Kubernetes / service mesh / tracing 正式接入
- AudioCraft / AudioLDM / Spleeter 的真实服务集成

---

## 3. Service identity

- service_name: `baseline-v1-api`
- api_version: `v0.1`
- model_name: `baseline_v1`
- primary_backend: `onnxruntime`
- current_runtime_role: `temporary_python_service_adapter`

说明：

当前服务不是最终生产服务，而是用于把已有 `baseline_v1` 闭环推进到“可调用、可观察、可迁移”的中间适配层。

---

## 4. Current model contract

当前服务绑定的默认模型产物：

- artifact_path: `artifacts/onnx/baseline_v1.onnx`

当前输入输出合同按现有导出与 shape 审计口径记录：

- input_name: `mel`
- output_name: `recon_mel`
- input_shape: `[1, 1, 64, 313]`
- output_shape: `[1, 1, 64, 313]`

当前必须明确：

- 这是 fixed-shape contract
- 当前不声明 dynamic shape support
- 当前 batch、channel、height、width 均不允许任意变化

---

## 5. Endpoint summary

| path | method | role |
| --- | --- | --- |
| `/healthz` | `GET` | 服务存活与最小健康检查 |
| `/metadata` | `GET` | 返回当前模型、后端、shape、provider、artifact 路径等元信息 |
| `/predict` | `POST` | 接收最小预测请求并返回预测结果摘要；Week 03 周一可先占位 |

---

## 6. GET /healthz

### 6.1 Purpose

用于最小服务健康检查。

它只回答“服务进程是否可达、最小应用是否启动成功”，不承担模型加载正确性的完整证明。

### 6.2 Response

成功时返回 JSON 对象，最小字段如下：

- `status`: 固定为 `ok`
- `service_name`
- `api_version`

### 6.3 Example response

- `status`: `ok`
- `service_name`: `baseline-v1-api`
- `api_version`: `v0.1`

---

## 7. GET /metadata

### 7.1 Purpose

返回当前服务所绑定的模型与推理后端的元信息，用于：

- curl smoke
- 本地排障
- Compose health / debug
- 后续客户端适配
- Java / gRPC 迁移时复用同一字段语义

### 7.2 Response fields

最小字段如下：

- `service_name`
- `api_version`
- `model_name`
- `model_version`
- `backend`
- `artifact_path`
- `provider_requested`
- `provider_actual`
- `input_name`
- `output_name`
- `input_shape`
- `output_shape`
- `contract_type`

### 7.3 Field semantics

- `model_version`：当前可先与 `baseline_v1` 保持一致
- `backend`：当前固定记录为 `onnxruntime`
- `provider_requested`：配置中声明的 provider
- `provider_actual`：运行时实际 provider；周一可先返回配置值或占位值
- `contract_type`：当前固定写 `fixed_shape`

### 7.4 Example response semantics

该接口必须明确表达：

- 当前模型来自 `artifacts/onnx/baseline_v1.onnx`
- 当前 I/O shape 为 `[1, 1, 64, 313]`
- 当前不是 dynamic shape 模型

---

## 8. POST /predict

### 8.1 Purpose

提供最小预测入口。

Week 03 周一的目标不是完成完整 ORT 推理服务，而是先冻结：

- 请求体字段名
- 响应体字段名
- 错误返回结构
- 未来扩展位

### 8.2 Request body (minimum version)

最小请求字段：

- `instance_id`
- `input_ref`
- `input_type`
- `expected_input_shape`

字段语义：

- `instance_id`：本次请求唯一标识，便于日志与 artifact 关联
- `input_ref`：输入引用；当前阶段可先支持本地路径字符串
- `input_type`：当前建议最小口径为 `mel_npy` 或 `tensor_ref`
- `expected_input_shape`：由客户端显式传入的 shape，用于最小合同检查

### 8.3 Reserved extension fields

为长期项目预留但周一不实现的字段：

- `media_ref`
- `segment_id`
- `blueprint_id`
- `output_artifact_dir`
- `generator_name`
- `request_tags`

这些字段当前阶段只在 schema 中保留为 optional，不进入实际推理逻辑。

### 8.4 Response body (minimum version)

最小响应字段：

- `instance_id`
- `status`
- `model_name`
- `backend`
- `provider_actual`
- `output_ref`
- `output_shape`
- `error`

### 8.5 Current implementation policy

周一允许 `/predict` 先返回占位响应，只要满足：

- 路径已经存在
- 请求/响应 schema 已冻结
- 错误响应结构已冻结
- 后续周二可在此基础上挂入真实 ORT 推理

---

## 9. Error response contract

所有端点统一最小错误格式：

- `code`
- `message`
- `details`

说明：

- `code`：机器可读错误码，例如 `INVALID_INPUT_SHAPE`
- `message`：简短人类可读说明
- `details`：可选扩展信息，可为空对象

当前阶段优先覆盖以下错误类型：

- `INVALID_INPUT_SHAPE`
- `INVALID_INPUT_REF`
- `MODEL_NOT_READY`
- `INFERENCE_FAILED`
- `INTERNAL_ERROR`

---

## 10. Logging and artifact policy

### 10.1 Logging

服务运行日志与 smoke 记录优先写入：

- `artifacts/logs/`

建议后续新增：

- `artifacts/logs/week03_api_smoke.log`

### 10.2 Artifact references

预测输出若产生文件型产物，优先落到：

- `artifacts/`

其中：
- ONNX 产物继续位于 `artifacts/onnx/`
- 音频样例位于 `artifacts/wav/`
- 运行日志与 CSV 位于 `artifacts/logs/`

---

## 11. Config mapping

本接口合同后续默认映射到：

- `configs/serving/service_api.yaml`

建议配置字段包括：

- `host`
- `port`
- `service_name`
- `api_version`
- `model_name`
- `model_version`
- `backend`
- `artifact_path`
- `provider_requested`
- `log_dir`

---

## 12. Week 03 Monday acceptance

今天结束前，本文档被视为完成，需要满足：

1. 三个端点均有明确职责
2. `/metadata` 明确写出 fixed-shape contract
3. `/predict` 的 request / response 字段已经冻结
4. 未来桥接字段已预留但未过度实现
5. 文档路径放在 `docs/design/serving_api_spec.md`
6. 后续 `src/serving/app.py` 与 `src/serving/schemas.py` 必须遵循本文档字段语义
