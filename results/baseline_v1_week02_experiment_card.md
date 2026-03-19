# baseline_v1 Week02 Experiment Card

## 1. 实验目标

本阶段目标不是继续扩模型，而是把 `baseline_v1` 的最小可复现链打通，形成可迁移到 CI 的 CPU smoke 路径。

本周重点验证：

1. ONNX 导出是否成功；
2. PyTorch 与 ONNXRuntime 输出是否对齐；
3. ORT CPU 推理是否可跑通；
4. benchmark 是否能稳定产出结构化结果。

---

## 2. 实验身份信息

- Model version: `baseline_v1`
- Stage: `week02`
- Commit: `bc2dd98`
- Date: `2026-03-19`
- Owner: `Ryan`

---

## 3. 输入资产与依赖

- Config: `configs/train/baseline.yaml`
- Checkpoint: `artifacts/experiments/baseline_v1/checkpoints/best.pt`
- Demo entry: `bash scripts/run_demo.sh demo`
- Example audio: `data/processed/esc10_miniset/valid/5-177957-C-40.wav`

---

## 4. 环境信息

- Python: `3.9.23`
- Device: `cpu`

Core packages verified:

- `torch`
- `torchaudio`
- `yaml`
- `soundfile`
- `librosa`
- `onnx`
- `onnxruntime`
- `tqdm`

---

## 5. 本阶段执行链路

实际验证命令：

    bash scripts/run_demo.sh demo

该链路完成了以下步骤：

1. export onnx
2. compare pytorch and onnxruntime
3. run onnxruntime inference
4. benchmark onnxruntime

---

## 6. 关键输出文件

### 6.1 ONNX 导出产物

- `artifacts/onnx/baseline_v1.onnx`

### 6.2 对齐报告

- `artifacts/logs/onnx_parity.md`

### 6.3 benchmark 原始结果

- `artifacts/logs/bench_ort_cpu.csv`

### 6.4 benchmark 可读报告

- `docs/benchmarks/ort_cpu_baseline.md`

### 6.5 本次 host smoke 日志

- `artifacts/logs/week02_ci_smoke_host.log`

---

## 7. 关键结果

### 7.1 ONNX 导出

- Opset: `17`
- Input shape: `(1, 1, 64, 313)`
- Output shape: `(1, 1, 64, 313)`
- `onnx.checker.check_model`: `passed`

### 7.2 PT vs ONNXRuntime 对齐

- Mean abs err: `0.0000051313`
- Max abs err: `0.0000104904`
- MSE: `0.0000000000`
- Allclose: `True`
- atol / rtol: `0.0001 / 0.0001`
- Provider: `CPUExecutionProvider`

### 7.3 ORT CPU benchmark

- Warmup / runs: `20 / 200`
- P50 ms: `23.1905`
- P95 ms: `26.5380`
- P99 ms: `26.9362`
- Mean ms: `23.2223`
- Std ms: `2.0831`
- Throughput: `43.0621 samples/s`

---

## 8. 本周学习结论

1. `baseline_v1` 已具备最小 ONNX 导出与 ORT CPU 推理闭环；
2. PyTorch 与 ONNXRuntime 在当前固定输入形状下能够稳定对齐；
3. `bash scripts/run_demo.sh demo` 可以作为后续 GitHub Actions smoke workflow 的主入口；
4. 当前结果更适合作为“固定 shape 的最小验证链”，还不是通用部署链。

---

## 9. 已知限制

1. 当前验证基于固定输入 shape `(1, 1, 64, 313)`；
2. export 日志中出现 `TracerWarning`，说明图追踪对动态形状泛化仍需谨慎；
3. 当前只验证 `CPUExecutionProvider`；
4. 当前 benchmark 是 smoke 级验证，不代表完整生产性能画像。

---

## 10. 与其他文档的关系

- 稳定总卡：`results/baseline_v1.md`
- benchmark 报告：`docs/benchmarks/ort_cpu_baseline.md`
- 周二/周三相关复盘：
  - `docs/postmortems/2026-03-16_week02_repro_audit.md`
  - `docs/postmortems/2026-03-18_week02_shape_audit.md`
- 架构桥接决策：
  - `docs/adr/0002-week02-bridge.md`

---

## 11. 下一步

1. 将当前 host 通过的最小链迁移到 `.github/workflows/smoke.yml`；
2. 在 `results/baseline_v1.md` 中增加“相关阶段实验卡”索引；
3. 后续若继续扩展 week03 / week04，沿用相同命名规范追加阶段实验卡；
4. 后续再考虑 dynamic shape、Docker CI、一致化依赖文件等问题。