# 2026-04-03 week04 contract freeze

## 1. Purpose

本文档用于冻结 Week 04 当前阶段的最小 serving contract。

本次冻结的目标不是继续扩展服务能力，而是把已经通过 schema、spec、pytest 与 artifact manifest 验证过的接口边界收束成一份可引用文档，作为后续：

- Week 04 summary 引用依据
- Week 05 blueprint seed JSON 映射
- Week 06 generator audit
- 后续 Java / gRPC / Compose / observability 桥接

的稳定输入。

---

## 2. Evidence reviewed

本次 contract freeze 直接基于以下证据：

- `src/serving/schemas.py`
- `docs/design/serving_api_spec.md`
- `docs/design/audio_blueprint_spec.md`
- `tests/test_api_smoke.py`
- `artifacts/logs/week03_artifact_manifest.json`
- `artifacts/logs/week04_pytest_api.log`
- `artifacts/logs/week04_pytest_predict_negative.log`

说明：

- `artifacts/logs/week04_pytest_api.log` 已在本次周三重新执行后刷新。
- 当前 freeze 以刷新后的 `week04_pytest_api.log` 与现有 negative-path 日志共同作为 Week 04 的测试证据。

---

## 3. Frozen scope

本次冻结只覆盖当前最小 HTTP serving surface：

- `GET /healthz`
- `GET /metadata`
- `POST /predict`

本次冻结不覆盖：

- 鉴权
- 流式输出
- 批量任务调度
- 异步队列
- 数据库存储
- Java / gRPC 正式服务
- Kubernetes / tracing / metrics 正式接入
- AudioCraft / AudioLDM / Spleeter 的真实服务集成

---

## 4. Frozen model contract

当前模型合同冻结如下：

- `model_name`: `baseline_v1`
- `backend`: `onnxruntime`
- `contract_type`: `fixed_shape`
- `input_name`: `mel`
- `output_name`: `recon_mel`
- `input_shape`: `[1, 1, 64, 313]`
- `output_shape`: `[1, 1, 64, 313]`

当前明确结论：

1. 这是 fixed-shape contract。
2. 当前不支持 dynamic shape。
3. 当前 batch / channel / height / width 均不允许任意变化。
4. `/predict` 成功路径和负路径已由 pytest 留痕验证。

---

## 5. Request contract freeze

### 5.1 Active required request fields

当前阶段，`/predict` 实际生效的最小请求字段为：

- `instance_id`
- `input_ref`
- `input_type`
- `expected_input_shape`

这些字段已经进入当前执行路径，属于本次 freeze 的 active contract。

### 5.2 Reserved but non-executing fields

以下字段当前已在 schema 中声明，但本周冻结为“保留字段，不进入执行分支”：

- `media_ref`
- `segment_id`
- `blueprint_id`
- `output_artifact_dir`
- `generator_name`
- `request_tags`

本次 freeze 的明确要求：

1. 这些字段允许出现在请求体 schema 中。
2. 它们当前不改变推理执行逻辑。
3. 它们当前不改变 artifact 产出路径规则。
4. 它们当前不参与路由、调度、追踪或生成器切换。

---

## 6. Reserved field semantics at freeze time

### 6.1 `media_ref`

冻结语义：

- 保留为上游媒体引用位。
- 当前不参与 `/predict` 执行。
- 后续可用于 media probe / segment / seed 映射桥接。

### 6.2 `blueprint_id`

冻结语义：

- 保留为音频蓝图标识。
- 当前不参与 `/predict` 执行。
- 当前仅要求其语义与 blueprint 顶层 `blueprint_id` 保持一致。
- 后续用于 Week 05 seed JSON 与 blueprint 映射。

### 6.3 `output_artifact_dir`

冻结语义：

- 保留为未来 artifact 输出目录偏好。
- 当前不允许改变现有 artifact 输出规则。
- 当前输出位置仍由现有服务逻辑决定，而不是由请求方指定。

### 6.4 `generator_name`

冻结语义：

- 保留为未来生成器后端选择位。
- 当前不允许改变现有 `/predict` 后端。
- 当前 `/predict` 仍固定绑定 `onnxruntime` 路径，不做 generator routing。

---

## 7. Response and error contract freeze

### 7.1 Stable response surfaces

当前阶段以下响应面视为稳定：

- `/healthz` 返回最小健康信息
- `/metadata` 返回模型、后端、shape、provider 与 artifact 路径元信息
- `/predict` 成功时返回 `status = ok`、`model_name`、`backend`、`output_shape`、`output_ref`

### 7.2 Negative-path coverage confirmed this week

本周已留痕覆盖的负路径包括：

- 非法输入 shape
- 不存在的 `input_ref`
- 非法 / 损坏的 `.npy` 输入

冻结结论：

1. `/predict` 不能只证明 happy path。
2. 当前最小 negative-path 已经进入 contract 证据链。
3. 后续新增错误码可以扩展，但不应破坏本次已冻结的最小错误响应结构。

---

## 8. Artifact output rule freeze

依据当前 manifest 与 smoke 证据，当前 artifact 规则冻结如下：

1. 当前输出 artifact 规则由现有服务逻辑决定。
2. 当前 contract 已确认存在可落盘的预测输出路径。
3. `output_artifact_dir`、`generator_name`、`request_tags` 等保留字段当前不允许改变该规则。
4. Week 03 manifest 中的 artifact 路径证据继续作为当前输出语义的历史基线。

---

## 9. What is frozen vs what is deferred

### 9.1 Frozen now

本周明确冻结：

- 最小 endpoint surface
- fixed-shape model contract
- 当前 metadata 返回口径
- `/predict` 最小 happy path
- `/predict` 最小 negative-path
- reserved field 的“已声明但不执行”边界

### 9.2 Deferred to next stages

明确延后到后续周次再定：

- blueprint seed JSON 正式映射
- generator audit 结论模板
- request routing / tracing / tags 生效规则
- `output_artifact_dir` 的真正写入策略
- `generator_name` 的真实后端切换逻辑
- Java / gRPC 正式服务对齐版 contract
- observability / tracing / metrics 正式接入

---

## 10. Freeze decision

本次 Week 04 contract freeze 结论如下：

1. 当前 serving surface 已具备最小可引用合同文档条件。
2. 当前 `/predict` contract 以 fixed-shape ONNX Runtime 路径为准。
3. 当前 reserved fields 只保留 schema 位置，不进入执行语义。
4. 后续周次允许扩展 contract，但不允许回改本次 freeze 已确认的最小边界，除非附带新的测试、日志与 postmortem 证据。

