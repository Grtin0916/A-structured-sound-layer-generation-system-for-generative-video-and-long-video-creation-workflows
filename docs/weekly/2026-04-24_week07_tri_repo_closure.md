# 2026-04-24 Week07 Tri-Repo Closure

## 1. Goal

本文件用于对 Week07 三仓协同推进做横向封板检查。

本次封板不是继续扩展新功能，而是确认三条线是否都已经从 Week06 的点状证据推进到 Week07 的可引用工程入口：

- 主基地：scorecard v1 与 sample CSV
- Java：Redis Streams event-flow smoke
- Cloud：minimum CI skeleton 与本地验证链

## 2. Tri-repo status

| Repo | Week07 target | Current closure status | Evidence entry |
|---|---|---|---|
| Mainbase | scorecard v1 + sample CSV | Closed for Week07 | `docs/evals/scorecard_v1.md`, `artifacts/benchmarks/scorecard_example.csv`, `docs/weekly/2026-04-24_week07_summary.md` |
| Java | Redis Streams event-flow smoke | Closed for Week07 | `docs/adr/0003-eventing-choice.md`, `loadtest/event-flow-smoke.md`, `docs/weekly/2026-04-24_week07_eventing_summary.md` |
| Cloud | minimum CI skeleton + local validation chain | Closed for Week07 | `.github/workflows/ci.yml`, `scripts/ci_validate.sh`, `docs/runbooks/ci-cd-minimum.md`, `docs/weekly/2026-04-24_week07_cloud_ci_summary.md` |

## 3. Mainbase closure

Mainbase Week07 的目标是把 Week06 generator audit 证据上提为 scorecard 入口。

当前已完成：

- `docs/evals/scorecard_v1.md`
- `artifacts/benchmarks/scorecard_example.csv`
- sample CSV 已覆盖 `functional`、`performance`、`failure_regression`
- README 已同步 Week07 scorecard verified scope
- Week08 的 `/metrics`、Prometheus、Grafana dashboard 与统一指标命名仍保留为 not yet verified

当前判定：

- Week07 scorecard 入口完成
- 不继续扩展 MusicGen medium / melody / 更大模型
- 不把 sample scorecard 伪装成正式自动化评测系统
- 下一步转入 Week08 metrics 化与命名统一

## 4. Java closure

Java Week07 的目标是把认证壳之后的最小异步事件流跑通，并留下可引用证据。

当前已完成：

- Redis Streams 选型 ADR：`docs/adr/0003-eventing-choice.md`
- Producer skeleton：`src/main/java/com/ryan/media/messaging/Producer.java`
- Consumer skeleton：`src/main/java/com/ryan/media/messaging/Consumer.java`
- event-flow smoke 文档：`loadtest/event-flow-smoke.md`
- event constants 已固定：
  - `MediaTaskCreated`
  - `media-task-events`
  - `media-task-group`
  - `week07-consumer`
- 本地 smoke 已验证 append / consume / ack

当前判定：

- Week07 eventing smoke 入口完成
- 不继续扩展 Kafka、DLQ、幂等、重试、补偿
- backlog 语义与状态更新闭环仍属于 not yet verified
- 下一步转入 W8 SQL tuning 与更稳定的 eventing 语义边界

## 5. Cloud closure

Cloud Week07 的目标是把 OTel 本地可复现证据推进到最小 CI/CD 骨架。

当前已完成：

- GitHub Actions workflow：`.github/workflows/ci.yml`
- validation script：`scripts/ci_validate.sh`
- CI/CD minimum runbook：`docs/runbooks/ci-cd-minimum.md`
- 本地验证链：
  - `scripts/ci_validate.sh`
  - `docker compose -f docker-compose.observability.yml config`
  - `kubectl apply --dry-run=client -f k8s/base`
- rerun 日志已保留

当前判定：

- Week07 minimum CI skeleton 完成
- 不继续扩展真实 image build / push
- 不提前声明 Terraform validate / plan 完成
- 不提前声明真实集群部署、rollout / rollback 或 SLO 完成
- 下一步转入 W8 Terraform layout 与 dev resource 边界

## 6. Cross-repo consistency check

Week07 三仓已经形成以下协同关系：

| Layer | Week07 artifact | W8 bridge |
|---|---|---|
| Mainbase | scorecard v1 | metrics naming, `/metrics`, Prometheus |
| Java | event-flow smoke | SQL tuning, consumer semantics, observability |
| Cloud | minimum CI skeleton | Terraform layout, K8s release base, SLO draft |

当前三仓没有混写职责：

- 主基地仍负责 AI / audio / eval / runtime 主链
- Java 仍负责后端平台、认证、安全、事件流与数据库语义
- Cloud 仍负责云原生、CI/CD、可观测、交付与稳定性

## 7. Not yet verified across Week07

以下内容不属于 Week07 已完成范围：

- Mainbase formal automated evaluation
- Mainbase metrics endpoint and Grafana dashboard
- Java full task state update loop
- Java retry / idempotency / compensation
- Java Kafka migration
- Cloud real image build / push
- Cloud Terraform validate / plan
- Cloud real cluster rollout / rollback
- Cloud SLO / alert policy

这些内容应进入 W8 及后续路线，不应在 Week07 封板中提前写满。

## 8. Final decision

Week07 三仓封板判定：

- Mainbase：closed
- Java：closed
- Cloud：closed

Week07 的核心目标已经完成：

- 主基地有可引用 scorecard
- Java 有最小异步链证据
- Cloud 有最小 CI 骨架与本地验证链证据

后续不再在 Week07 横向扩 scope。下一阶段进入 Week08 / S1 预热：

- 主基地：metrics 化与指标命名统一
- Java：SQL tuning 与事件语义边界
- Cloud：Terraform layout 与 dev resource 边界

## 9. Next action

下一步不应继续写功能，而应完成当天学习收尾：

- DSA：二叉树遍历或最小深度
- CS：中断、DMA、阻塞 / 非阻塞 I/O、select / poll / epoll 复盘
- 周末：准备 Week07 全周总结与 Week08 入口计划
