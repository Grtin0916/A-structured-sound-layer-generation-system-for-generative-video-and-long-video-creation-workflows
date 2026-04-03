# 2026-04-03 Week04 Summary

## 1. 本周目标

Week04 的目标不是继续横向扩展新能力，而是把主基地当前已经跑通并留有证据的最小工程链路收束成可引用的周总结，完成 serving / artifact contract 的第一轮封板，并为 Week05 的 blueprint seed JSON、Week06 的 generator audit 留出稳定输入边界。

本周收口的核心原则是：

- 不把“规划中的能力”写成“已完成”
- 不把“目录存在”误写成“验证完成”
- 只对已经有代码、文档、日志、测试留痕的内容给出已验证结论
- 明确区分已完成、未完成和下一硬里程碑

---

## 2. 本周已完成的硬结果

本周在主基地仓库中已经形成如下可引用结果：

### 2.1 serving / artifact contract 第一轮冻结

已新增并提交：

- `docs/postmortems/2026-04-03_week04_contract_freeze.md`

该文档已经把当前最小 serving contract 收束为一份可引用输入，用于后续：

- Week04 summary 引用
- Week05 blueprint seed JSON 映射
- Week06 generator audit
- 后续 Java / gRPC / Compose / observability 桥接

### 2.2 当前最小 serving surface 已被明确冻结

本周冻结只覆盖当前最小 HTTP 接口面：

- `GET /healthz`
- `GET /metadata`
- `POST /predict`

当前阶段不把以下内容写入已完成范围：

- 鉴权
- 流式输出
- 批量任务调度
- 异步队列
- 数据库存储
- Java / gRPC 正式服务
- Kubernetes / tracing / metrics 正式接入
- AudioCraft / AudioLDM / Spleeter 的真实服务集成

### 2.3 `/predict` 的最小合同边界已明确

当前冻结的模型合同为：

- `model_name`: `baseline_v1`
- `backend`: `onnxruntime`
- `contract_type`: `fixed_shape`
- `input_name`: `mel`
- `output_name`: `recon_mel`
- `input_shape`: `[1, 1, 64, 313]`
- `output_shape`: `[1, 1, 64, 313]`

当前阶段的明确结论：

- 这是 fixed-shape contract
- 当前不支持 dynamic shape
- 当前 batch / channel / height / width 不允许任意变化
- `/predict` 的成功路径与负路径都已经留痕

### 2.4 Week04 测试证据已保留

当前已存在的 Week04 测试留痕包括：

- `artifacts/logs/week04_pytest_api.log`
- `artifacts/logs/week04_pytest_predict_negative.log`

这些证据表明，本周至少已经对 `/predict` 的 smoke 路径与负路径进行了日志级留痕，能够支持 Week04 contract 收口，而不是停留在 README 口头描述层。

### 2.5 README 顶部三段已经同步到当前阶段语义

README 当前已经按以下结构收束：

- `Verified Scope`
- `Not Yet Verified`
- `Next Hard Milestone`

这意味着主基地已经不再是“泛化路线草图”，而是进入“只写已验证范围、明确未验证边界、给出下一个硬里程碑”的工程表达阶段。

---

## 3. 本周完成度判断

### 3.1 已验证范围

截至 Week04 收口，主基地已可以明确写入“已验证”的内容包括：

- 最小数据集准备、manifest 构建、baseline 训练、checkpoint 保存这一条基础训练链路
- `baseline_v1` 的 ONNX 导出
- 固定 shape 下的 PyTorch / ONNX 对齐校验
- ONNX Runtime CPU 单次推理验证
- ONNX Runtime benchmark 结果产出
- Docker 下的最小 CPU 验证链路
- Week04 serving / artifact contract 第一轮冻结
- `/predict` 的 smoke 路径与负路径日志证据

### 3.2 明确不写满的部分

以下内容虽然已经进入路线规划，但截至当前 Week04 远端状态，还不能写成“已完成”：

- `media_probe -> blueprint seed JSON` 的正式映射
- generator audit 的正式结论
- 更完整的音频生成链可复验性结论
- 主基地 observability 的正式实现与验证
- scorecard / regression / evals 的体系化收口
- 更完整的 serving runtime、性能优化与多阶段评测闭环

### 3.3 当前周的真实定位

Week04 对主基地的价值，不是功能扩张，而是工程地基收口：

- 已经具备一条可复验的最小训练 -> 导出 -> 推理 -> benchmark -> Docker 闭环
- 已经开始把服务边界从“脚本行为”上升为“contract 文档”
- 已经开始为后续 seed / audit / eval / observability 留出稳定接口语义
- 但仍处于工程地基阶段，不应误判为生成系统主链已经完成

---

## 4. 本周输入证据索引

本周 summary 直接引用的输入如下：

- `README.md`
- `docs/postmortems/2026-04-03_week04_contract_freeze.md`
- `docs/design/serving_api_spec.md`
- `docs/design/audio_blueprint_spec.md`
- `tests/test_api_smoke.py`
- `artifacts/logs/week04_pytest_api.log`
- `artifacts/logs/week04_pytest_predict_negative.log`

---

## 5. 本周最重要的工程判断

### 5.1 本周不继续改 `/predict` 功能

原因不是 `/predict` 已经“做完了”，而是当前更值钱的动作是先把已有 contract、schema、日志和 spec 收口成稳定边界。没有收口的扩张，只会把后续 seed / audit / eval 的输入继续搅乱。

### 5.2 当前 contract 的意义高于新增功能点

本周真正推进的不是“又多了一个接口”，而是：

- 请求字段语义被固定
- 响应字段语义被固定
- error 结构开始稳定
- active fields 与 reserved fields 被明确区分
- 模型 I/O shape、backend、contract_type 被正式冻结

这是后续桥接 Java、Cloud、评测、可观测的工程前提。

### 5.3 当前生成链仍未正式收口

虽然仓库中已经有早期生成链痕迹与相关实验留痕，但截至本周，主基地还没有形成“完整生成链可复验 / 不可复验 / 失败原因”的正式审计结论。因此这一部分仍应保留到 Week06 的 generator audit 统一处理，而不是在 Week04 summary 中提前写满。

---

## 6. 风险与未完成项

### 6.1 当前最主要风险

- 把 contract freeze 错写成“服务能力完成”
- 把日志文件存在错写成“全链路验证完成”
- 把规划中的 seed / audit / observability 提前写成已完成
- 在 Week04 收口前继续横向扩张，导致 W5 输入边界再次漂移

### 6.2 本周主动不做的事

为了保证收口质量，本周主动不推进以下方向：

- 主基地 observability 正式实现
- 生成链正式审计
- 复杂 serving runtime 对比
- 更大范围的 regression / eval 扩张
- Java / Cloud 在主基地内的正式集成

---

## 7. 下一步硬里程碑

### 7.1 Week05：blueprint seed JSON

下一步首先进入：

- 把 `media_probe` 输出正式映射为 blueprint seed JSON
- 落第一条最小 seed
- 补对应 mapping / regression 文档

### 7.2 Week06：generator audit

在 seed 建立后，进入：

- 对 MusicGen / AudioCraft 相关生成链做正式审计
- 给出“可复现 / 不可复现 / 失败原因”的明确结论
- 形成可引用的审计记录与日志

### 7.3 后续增强方向

在上述两步之后，再逐步推进：

- scorecard
- failure regression
- observability
- 更稳定的 serving 语义
- 更完整的系统评测闭环

---

## 8. Week04 结论

Week04 对主基地的实际贡献，不是把系统做大，而是把已经跑通的最小闭环做实、做清楚、做成可引用证据。

截至本周末，主基地已经从“训练 / 导出 / 推理 / benchmark / Docker 的最小工程闭环”进一步推进到“serving / artifact contract 第一轮冻结 + `/predict` 正负路径日志留痕”的状态。

这意味着：

- 主基地工程地基继续加厚
- Week05 可以进入 seed JSON，而不是回头补 contract
- Week06 可以进入 generator audit，而不是继续修 Week04 边界
- 三仓协同中，主基地仍然保持了旗舰仓库应有的推进速度与证据密度

当前最合理的判断是：

主基地 Week04 已具备收口条件，但不应夸大为生成主链已完成；它已经完成的是“工程地基的进一步封板”，而不是“完整系统能力封板”。