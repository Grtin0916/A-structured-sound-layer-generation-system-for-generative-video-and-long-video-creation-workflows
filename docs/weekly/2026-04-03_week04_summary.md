# 2026-04-03 Week04 Summary

## 本周定位

Week04 的目标不是继续扩功能，而是把主基地当前已经形成的 API / artifact contract、测试日志与最小 serving 证据收口成一份可引用周总结，为后续 seed JSON、generator audit、scorecard 和 observability 继续推进打下工程地基。

本周重点围绕以下三类内容展开：

1. `/predict` 路径的 smoke 与负路径留痕
2. artifact / serving contract 初版冻结
3. week04 证据归档与语义对齐

---

## 本周新增的硬结果

### 1. contract freeze 文档已形成

本周已形成 `docs/postmortems/2026-04-03_week04_contract_freeze.md`，用于冻结当前阶段的 serving / artifact contract 语义，并为后续 README、评测、回归测试和 blueprint seed 映射提供引用依据。

### 2. API smoke 证据已留存

本周已产出以下日志文件，用于证明当前 serving 接口最小链路与负路径行为已被显式检查：

- `artifacts/logs/week04_pytest_api.log`
- `artifacts/logs/week04_pytest_predict_negative.log`

它们共同支撑了本周“不是只写了接口文档，而是已经有测试留痕”的结论。

### 3. W4 的工作重心已从“继续搭壳”切换为“口径收口”

当前主基地不是从零开始，仓库已经具备：

- baseline 训练闭环
- ONNX 导出
- ONNX Runtime 推理
- Docker 最小 CPU 验证链路
- serving 基础代码与 smoke test

因此 Week04 的核心价值，不是新增更多模块，而是把已有证据和 contract 语义对齐，避免后续文档、日志、README 和代码各说各话。

---

## 当前已验证范围（截至 Week04）

| 模块 | 当前状态 | 本周结论 |
| --- | --- | --- |
| Serving API 最小骨架 | 已存在 | 已能作为 contract freeze 的落点 |
| `/predict` smoke 路径 | 已留痕 | 有 week04 pytest 日志支撑 |
| `/predict` 负路径 | 已留痕 | 有单独 negative log 支撑 |
| artifact contract 初版 | 已冻结 | 已形成可引用 freeze 文档 |
| README 与最小工程主线 | 已存在 | 可作为 week04 summary 的背景上下文 |
| 生成链可复验性结论 | 未正式封板 | 留待 W6 generator audit 收口 |
| observability 实现 | 未展开 | W4 不在主基地继续扩散 |

---

## 本周明确没有做的事

为了控制节奏并保证证据密度，Week04 没有把主基地继续扩展到以下方向：

- 不在本周引入新的生成链主能力结论
- 不在本周推进 observability 主实现
- 不把 README 改造成大而全的路线总览文档
- 不把 week04 写成新的功能扩张周

这些内容后续会分别在 generator audit、scorecard、metrics 和 observability 周推进。

---

## 与现有设计文档的关系

本周 summary 不是独立文档，而是对以下设计/冻结文档的汇总与收口：

- `docs/design/serving_api_spec.md`
- `docs/design/audio_blueprint_spec.md`
- `docs/postmortems/2026-04-03_week04_contract_freeze.md`

如果后续发现 `serving_api_spec.md` 中某些表述与日志或实际接口行为不一致，应以当前 week04 的远端代码与日志为准，再回补设计文档，而不是反过来让日志迁就旧文档。

---

## 本周留下的证据链

### 文档证据
- `docs/postmortems/2026-04-03_week04_contract_freeze.md`
- `docs/weekly/2026-04-03_week04_summary.md`

### 测试 / 日志证据
- `artifacts/logs/week04_pytest_api.log`
- `artifacts/logs/week04_pytest_predict_negative.log`

### 代码 / 结构证据
- `src/serving/`
- `tests/test_api_smoke.py`

---

## 问题与限制

### 1. 生成链的“可复验结论”还没有正式封板
目前只能确认主基地已有最小工程地基与部分生成相关 smoke 痕迹，但还不能把 generator 层面的能力说得太满。这个结论需要等 W6 的 generator audit 再正式给出。

### 2. README、设计文档、日志三者仍可能存在局部表述差异
本周 summary 已先完成收口，但下一步仍需要继续做 README 顶部三段同步，以及必要的 `serving_api_spec.md` 小补丁。

### 3. observability 目录当前还不构成主基地已验证能力
相关目录存在不等于能力已完成；W4 不把它计入已验证范围。

---

## Week04 的最终判断

Week04 对主基地的真实贡献，不是“又做了一个新模块”，而是把已有 serving / contract / log 证据正式封板为可引用周结果。

这意味着主基地从“有若干离散工程结果”，推进到了“已有一份可被后续 README、阶段总结、评测与面试叙述复用的周级证据文档”。

---

## 下一硬里程碑

下一个硬里程碑不是继续补 Week04，而是进入 Week05 / Week06 的两步推进：

1. Week05：把 `media_probe` 输出正式映射成 blueprint seed JSON，并落第一条最小 seed
2. Week06：完成 generator audit，明确哪些生成链能力已经可复验，哪些仍不可复验或失败

Week04 到此应收口，不再继续横向扩张。