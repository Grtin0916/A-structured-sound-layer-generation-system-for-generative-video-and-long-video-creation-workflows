# Week 1 Summary (2026-03-14)

## 1. 本周目标

本周的核心目标是完成一条最小但完整的工程闭环，用于支撑后续 AudioCraft 学习、音频生成工作流建模，以及更复杂系统设计的继续推进。

当前闭环聚焦于以下主线：

- 最小数据集准备
- manifest 构建
- baseline 模型训练
- checkpoint 导出与记录
- ONNX 导出与对齐校验
- ONNX Runtime 推理与 benchmark
- Docker 下的最小 CPU 验证链路

---

## 2. 本周完成项

截至 2026-03-14，`baseline_v1` 已完成以下关键环节验证：

- ESC-50 → ESC-10 miniset 数据准备完成
- 训练集 / 验证集 JSONL manifest 已生成
- `baseline_v1` 已完成训练并保存 checkpoint
- TensorBoard 训练日志链路已验证
- `best.pt -> baseline_v1.onnx` 导出完成
- PyTorch / ONNX Runtime parity check 通过
- ONNX Runtime 单次推理完成
- ONNX Runtime CPU benchmark 完成
- `scripts/run_demo.sh demo` 已可作为当前最小复现入口

这意味着本周已经完成一条可复验的 Week 1 工程主线，而不是只停留在代码拼装或单点脚本验证。

---

## 3. 本周最小复现入口

建议优先使用下述命令复现本周已验证通过的 ONNX / ORT 主线：

```bash
bash scripts/run_demo.sh demo
```

该命令顺序执行：

1. ONNX 导出
2. PyTorch / ONNX Runtime 对齐校验
3. ONNX Runtime 单次推理
4. ONNX Runtime benchmark

若后续需要从数据准备开始重走完整链路，可再使用：

```bash
bash scripts/run_demo.sh full
```

---

## 4. 本周关键输出文件

### 4.1 模型与导出产物

- `artifacts/experiments/baseline_v1/checkpoints/best.pt`
- `artifacts/onnx/baseline_v1.onnx`

### 4.2 对齐与推理日志

- `artifacts/logs/onnx_parity.md`
- `artifacts/logs/week01_demo_stdout.log`

### 4.3 Benchmark 结果

- `artifacts/logs/bench_ort_cpu.csv`
- `docs/benchmarks/ort_cpu_baseline.md`

---

## 5. ONNX 导出与对齐结果

### 5.1 导出配置

- Config: `configs/train/baseline.yaml`
- Checkpoint: `artifacts/experiments/baseline_v1/checkpoints/best.pt`
- Output ONNX: `artifacts/onnx/baseline_v1.onnx`
- Device: `cpu`
- Opset: `17`
- Input name: `mel`
- Output name: `recon_mel`
- Input shape: `(1, 1, 64, 313)`
- Output shape: `(1, 1, 64, 313)`

### 5.2 Parity Check 结果

- Sample source: `data/processed/esc10_miniset/valid/5-177957-C-40.wav`
- Provider: `CPUExecutionProvider`
- Shape equal: `True`
- Mean abs error: `0.0000051313`
- Max abs error: `0.0000104904`
- MSE: `0.0000000000`
- Allclose: `True`
- atol / rtol: `0.0001 / 0.0001`

### 5.3 结论

在当前固定输入形状与当前样本条件下，PyTorch 与 ONNX Runtime 输出已经实现逐元素对齐，说明当前 `baseline_v1` 的导出链路基本可用，可作为后续 ORT 部署与容器化验证的基础版本。

---

## 6. ORT 推理与 Benchmark 摘要

### 6.1 单次推理验证

- ONNX path: `artifacts/onnx/baseline_v1.onnx`
- Provider request: `CPUExecutionProvider`
- Provider actual: `['CPUExecutionProvider']`
- Input source: `random(shape=(1, 1, 64, 313), seed=42)`
- Input shape: `(1, 1, 64, 313)`
- Output shape: `(1, 1, 64, 313)`
- Output min / max: `-3.984205 / 2.514611`

### 6.2 Benchmark 设置

- Warmup runs: `20`
- Timed runs: `200`
- Batch size: `1`
- Intra-op threads: `0`
- Inter-op threads: `0`

### 6.3 Benchmark 结果

- min: `16.4690 ms`
- p50: `23.3445 ms`
- p90: `28.0977 ms`
- p95: `29.0531 ms`
- p99: `30.9132 ms`
- max: `33.4758 ms`
- mean: `23.7309 ms`
- std: `3.1428 ms`
- throughput: `42.1391 samples/s`

### 6.4 结论

当前 ORT benchmark 已经形成第一版 CPU 延迟证据。现阶段统计范围仅覆盖 `session.run()` 执行耗时，不包含 session 初始化与样本读取，因此这组结果适合作为 **Week 1 基线**，但还不是完整部署时延。

---

## 7. 本周遇到的关键问题

### 7.1 ONNX 导出阶段的 TracerWarning

在导出过程中出现如下提示：

> `TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect.`

对应位置来自模型中基于 shape 的条件判断：

```python
if out.shape[-2:] != x.shape[-2:]:
```

### 当前判断

这条警告**没有阻断当前导出，也没有影响当前固定 shape 下的 parity check**，因为：

- `onnx.checker.check_model` 已通过
- PT / ONNX 对齐结果通过
- ORT 推理和 benchmark 已跑通

### 风险含义

这类写法可能导致 traced graph 对其他输入形状的泛化能力不足，因此它目前不是“当前 Week 1 闭环失败”的证据，但它是一个明确的**后续工程风险点**。

### 后续处理建议

后续进入更严格导出与部署阶段时，应重点检查：

- 是否需要改写相关 shape 判断逻辑
- 是否需要增加多 shape 输入测试
- 是否需要重新验证导出图在动态或不同长度输入下的稳定性

这部分更适合在 `docs/postmortems/` 下单独形成问题记录。

---

## 8. 与仓库结构的对应关系

为保证后续仓库继续扩展时结构不乱，本周结果建议按以下职责进行沉淀：

### 8.1 `artifacts/`

放原始实验产物与中间输出：

- checkpoint
- onnx 文件
- parity 原始日志
- benchmark CSV
- demo stdout 日志

### 8.2 `docs/benchmarks/`

放适合展示与长期追踪的 benchmark 报告：

- `docs/benchmarks/ort_cpu_baseline.md`

### 8.3 `docs/weekly/`

放每周阶段性总结：

- `docs/weekly/2026-03-14_week01_summary.md`

### 8.4 `docs/postmortems/`

放问题复盘、故障根因与修复动作：

- 后续可新增 `week01_onnx_export_warning.md`
- 后续可新增 `week01_repro_audit.md`

### 8.5 `results/`

放对外展示导向更强的阶段性结果索引或版本总结；本周如果不想重复存放完整周报，可仅在 `results/` 中保留索引型文档，指向 `docs/weekly/` 与 `docs/benchmarks/`。

---

## 9. Week 1 当前结论

本周最重要的结论不是“模型多强”，而是：

**仓库已经具备一条最小但真实可复验的音频工程闭环。**

具体来说：

- 从训练到 checkpoint 保存已完成
- 从 checkpoint 到 ONNX 导出已完成
- 从 PyTorch 到 ONNX Runtime 的对齐校验已通过
- 从 ORT 推理到 benchmark 已形成可追踪结果
- 从 README 到 `scripts/run_demo.sh demo` 的仓库入口已经打通

这说明当前项目已经从“学习资料堆积”进入“可运行、可验证、可记录”的工程阶段。

---

## 10. 下周建议的主线进入点

下周优先级建议如下：

### P0：复现性审计

- 在更干净的环境中重跑 `bash scripts/run_demo.sh demo`
- 核查 README、脚本、输出路径是否完全一致
- 补一次“他人视角”的最短复现实验

### P1：Docker 再验证

- 在容器内复验 ONNX Runtime 推理与 benchmark
- 对比主机环境与容器环境的延迟差异
- 形成 `deploy/docker/` 或 `docs/postmortems/` 里的补充记录

### P1：导出稳定性检查

- 关注当前 TracerWarning
- 增加不同输入长度/shape 的导出与推理测试
- 评估是否需要更稳健的导出实现

### P2：继续接入 AudioCraft 学习主线

- 把当前 Week 1 工程底座与后续 AudioCraft / 音频生成方向学习任务衔接起来
- 保证后续学习不是重新起盘，而是在当前闭环之上继续迭代

---

## 11. 一句话版总结

Week 1 已经完成 `baseline_v1` 的训练、导出、对齐、ORT 推理与 benchmark 闭环，并形成了可复验的 `run_demo.sh demo` 入口；当前最值得继续推进的方向不是再堆功能，而是先把复现性、容器验证和导出稳定性审计做扎实。
