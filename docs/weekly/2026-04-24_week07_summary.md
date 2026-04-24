# 2026-04-24 Week07 Mainbase Summary

## 1. Goal

本次收口目标不是继续扩展新实验，而是把主基地 Week07 已落盘的 scorecard 资产核查到可直接引用状态。

本次核查围绕四个问题展开：

- README 顶部三段是否已经同步 Week07 scorecard 状态
- `docs/evals/scorecard_v1.md` 是否定义了当前 scorecard 字段与判定边界
- `artifacts/benchmarks/scorecard_example.csv` 是否与 scorecard 文档字段一致
- 当前主基地是否仍把 Week08 metrics / Prometheus / Grafana 误写为已完成

## 2. Current verified status

当前主基地已经完成以下 Week07 资产：

- `docs/evals/scorecard_v1.md` 已落盘
- `artifacts/benchmarks/scorecard_example.csv` 已落盘
- sample CSV 当前包含 10 条样例
- scorecard 覆盖 `functional`、`performance`、`failure_regression` 三类入口
- README Verified Scope 已同步 Week05 seed、Week06 generator audit 与 Week07 scorecard 初版
- README Not Yet Verified 明确保留 `/metrics`、Prometheus、Grafana dashboard 与统一指标命名为 Week08 后续工作

因此，当前主基地已经从 Week06 的 generator audit 证据闭环推进到 Week07 的 scorecard 可引用样例阶段。

## 3. Scorecard schema decision

当前 scorecard v1 采用以下字段：

- `eval_group`
- `subject`
- `case_id`
- `input_type`
- `status`
- `failure_tag`
- `metric_key`
- `metric_value`
- `evidence_path`
- `note`

本次不再引入新的字段命名，例如 `eval_type`、`sample_id`、`source_evidence`、`latency_ms`、`throughput` 或 `environment_note`。

原因是当前 `docs/evals/scorecard_v1.md` 与 `artifacts/benchmarks/scorecard_example.csv` 已经保持一致；在 Week07 收口阶段强行改字段会制造无意义迁移成本。后续如果进入正式 benchmark 或自动化 eval，再通过 v2 schema 统一扩展性能字段。

## 4. Evidence checked

本次核查覆盖以下文件：

- `README.md`
- `docs/evals/scorecard_v1.md`
- `artifacts/benchmarks/scorecard_example.csv`
- `docs/weekly/2026-04-22_week07_day03.md`
- `docs/weekly/2026-04-23_week07_day04.md`

当前没有发现必须立即修改 README 或 CSV 的问题。

## 5. Not yet verified

以下内容仍不应写成已完成：

- scorecard 自动化执行
- 严格统一的 generator benchmark methodology
- `facebook/musicgen-medium`
- `facebook/musicgen-melody`
- 更大模型可比较结论
- `/metrics`
- Prometheus scrape
- Grafana dashboard
- 统一 metrics naming
- serving runtime 深化与跨环境性能结论

这些内容属于 Week08 及后续工作，不纳入本次 Week07 主基地收口。

## 6. Decision

主基地 Week07 当前状态判定为：

- scorecard v1 文档：完成
- scorecard sample CSV：完成
- README 顶部三段同步：完成
- weekly 收口入口：本文件补齐
- Week08 metrics 预热：只保留为下一阶段入口，不提前实现

本次收口后，主基地今天不再扩 scope，后续优先转向 Java event-flow smoke 与 Cloud CI evidence 收口。

## 7. Next hard milestone

Week08 主基地下一硬里程碑是 metrics 化与命名统一，建议优先落以下文件：

- `monitoring/prometheus/prometheus.yml`
- `monitoring/grafana/dashboards/mainbase-overview.json`
- `docs/design/metrics_naming.md`

Week08 的重点不是继续扩大 scorecard 样例数量，而是把 contract、runtime、artifact 与 eval 相关指标命名拉齐，为 S1 阶段验收准备可观测入口。
