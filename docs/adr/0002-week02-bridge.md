# 0002 - Use baseline_v1 + host-side ONNX/ORT path as the Week 2 engineering baseline

- Status: accepted
- Date: 2026-03-16

## Context

Week 1 已经完成 baseline_v1 的最小可运行链路，并产出了以下核心资产：

- PyTorch checkpoint:
  `artifacts/experiments/baseline_v1/checkpoints/best.pt`
- ONNX export:
  `artifacts/onnx/baseline_v1.onnx`
- PT vs ORT parity report:
  `artifacts/logs/onnx_parity.md`
- ORT benchmark report:
  `docs/benchmarks/ort_cpu_baseline.md`

在 2026-03-16 的 host 环境复跑中，最短复现入口：

bash scripts/run_demo.sh demo

已经成功完成以下步骤：

1. export onnx
2. compare pytorch and onnxruntime
3. run onnxruntime inference
4. benchmark onnxruntime

本次复跑的关键信息如下：

- parity allclose: True
- mean abs err: 0.0000051313
- max abs err: 0.0000104904
- benchmark p50: 25.9407 ms
- benchmark p99: 34.2762 ms
- throughput: 39.1907 samples/s

同时，导出阶段出现了 1 个 TracerWarning，说明当前 trace 图在其他 shape 上的泛化仍需后续审计，但在当前固定 shape 合同输入下功能已验证通过。

## Decision

Week 2 继续采用以下工程基线：

1. 保留 baseline_v1 作为当前唯一受支持的主线模型基线；
2. 保留 host-side demo 复跑链路作为最短审计入口；
3. 以 ONNX/ORT 导出、对齐、推理、benchmark 作为部署侧主验证路径；
4. 本周新增工作优先围绕“审计、稳定性、对比、文档化”展开，而不是立即切换新模型或重做训练；
5. 只有在以下任一条件出现时，才考虑替换当前基线：
   - 多样本 parity 明显失稳；
   - Docker 路径与 host 路径出现不可接受偏差；
   - shape stability 审计失败；
   - 新方案在正确性和性能上同时显著优于 baseline_v1。

## Rationale

选择这条路径的原因：

1. Week 1 的产物已经形成了从 checkpoint 到 ONNX 再到 ORT benchmark 的完整闭环；
2. 今天的 host 复跑已经证明这条链路不是一次性成功，而是可以重复执行；
3. 当前最需要补的是工程可信度，而不是继续增加系统复杂度；
4. 先把“能稳定复现、能解释、能比较”做扎实，后续再扩展 Docker、多样本、shape smoke test，会更稳。

## Consequences

### Positive

- Week 2 的所有工作都能围绕一个清晰、稳定的技术底座展开；
- 后续 Docker 对照、多样本 parity、shape audit 都有统一参照物；
- README、周报、实验记录、benchmark 文档可以持续复用同一套口径。

### Negative

- 暂时不会引入更激进的新模型或新部署路径；
- 可能会错过一些短期“看起来更快”的试验机会；
- 当前 TracerWarning 说明导出图的泛化边界仍不完全清楚。

### Risks

- 目前只验证了固定 shape 与单条 demo 主链；
- benchmark 还没有形成多次重复统计；
- host 与 Docker 的性能差异尚未完成系统比较。

## Alternatives considered

### Alternative A: 立即切到新模型/新结构
未采用。  
原因：当前最缺的是工程可审计性，不是模型多样性。

### Alternative B: 重新从训练开始走 full 流程
未采用。  
原因：今天的目标是复现审计，而不是重算一遍已有结果。

### Alternative C: 只保留 PyTorch，不继续推进 ONNX/ORT
未采用。  
原因：本项目后续明确需要服务部署与推理侧，因此 ONNX/ORT 路径必须继续保留。

## Evidence

- Repro log:
  `artifacts/logs/week02_repro_host.log`
- Version freeze:
  `artifacts/logs/week02_versions.txt`
- Repro audit:
  `docs/postmortems/week02_repro_audit.md`
- Parity report:
  `artifacts/logs/onnx_parity.md`
- Benchmark:
  `docs/benchmarks/ort_cpu_baseline.md`

## Next actions

1. 周一：完成本 ADR、完成算法题记录；
2. 周二：做多样本 parity / benchmark；
3. 周三：做 shape stability / export audit；
4. 周四：准备 Docker 对照与文档补齐；
5. 周末前：更新 README、周报和结果摘要。
