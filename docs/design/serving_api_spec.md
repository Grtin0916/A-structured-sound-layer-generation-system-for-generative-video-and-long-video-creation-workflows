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

---

## 13. Week 04 Tuesday reserved fields semantics

本节用于把 `/predict` 请求体中的保留字段语义进一步冻结，避免后续 blueprint / generator 接入时出现同名字段含义漂移。

### 13.1 Current rule

以下字段在 Week 04 Tuesday 的定位统一为：

- 只定义语义
- 允许出现在 request schema 中
- 当前不改变实际 ORT 推理行为
- 当前不改变 artifact 落盘目录
- 当前不触发复杂路由、调度或策略分发

### 13.2 Field-by-field semantics

#### `media_ref`
- 语义：上游媒体引用
- 当前用途：保留给后续 `source_video.path` / 输入媒体身份对齐
- 当前状态：仅透传语义，不参与 `/predict` 执行分支判断

#### `segment_id`
- 语义：分段级标识
- 当前用途：预留给后续 timeline / segment 粒度桥接
- 当前状态：当前 blueprint 最小规范中尚未形成稳定直接映射，因此先保留，不参与执行逻辑

#### `blueprint_id`
- 语义：audio blueprint 唯一标识
- 当前用途：对齐 blueprint 顶层 `blueprint_id`
- 当前状态：当前不参与推理，仅用于后续请求与 blueprint 关联

#### `output_artifact_dir`
- 语义：调用方期望的输出 artifact 目录
- 当前用途：先作为保留字段记录意图
- 当前状态：本周不启用目录重定向；预测输出仍统一写入 `artifacts/predict/`
- 设计边界：不能因为该字段存在就把输出写到任意自定义路径

#### `generator_name`
- 语义：调用方声明的生成器/后端偏好标识
- 当前用途：为后续与 `metadata.generator_stack` 或 timeline 事件 `generator` 对齐做准备
- 当前状态：本周不根据该字段切换实际推理后端；`baseline_v1` 仍固定走当前 ORT 路径
- 设计边界：它是“声明性字段”，不是“立即生效的执行开关”

#### `request_tags`
- 语义：请求标签列表
- 当前用途：为后续 tracing / routing / experiment tagging 预留
- 当前状态：本周不进入业务逻辑，不参与 artifact 路径生成，不参与 provider 选择
- 设计边界：只保留标签语义，不实现复杂标签驱动行为

### 13.3 Current artifact policy clarification

当前 `/predict` 的输出引用语义固定为：

- 服务若产生预测输出文件，统一落到 `artifacts/predict/`
- `output_ref` 返回相对于仓库根目录的相对路径
- 当前不允许由 `output_artifact_dir`、`generator_name`、`request_tags` 等保留字段改变该规则

### 13.4 Week 04 Tuesday acceptance

本节补充完成后，视为当前阶段满足：

1. 保留字段的“可写入”与“可生效”边界已经分开
2. 当前实现不会因为保留字段而改变 ORT 推理路径
3. 当前实现不会因为保留字段而改变 artifact 输出目录
4. blueprint / media / generator 相关字段已经冻结最小语义，供后续桥接使用
