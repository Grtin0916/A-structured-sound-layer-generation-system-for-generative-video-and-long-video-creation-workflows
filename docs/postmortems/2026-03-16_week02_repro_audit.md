# Week 02 Repro Audit (Host)

## 1. Goal
在 host 环境中按仓库最短复现入口执行：
bash scripts/run_demo.sh demo
将 Week 1 的“本机成功”升级为 Week 2 的“可审计复现证据”。

## 2. Run Status
- Status: PASS
- Execution mode: host
- Command:
  bash scripts/run_demo.sh demo 2>&1 | tee artifacts/logs/week02_repro_host.log

## 3. Key Outputs
- Log: artifacts/logs/week02_repro_host.log
- Version freeze: artifacts/logs/week02_versions.txt
- ONNX: artifacts/onnx/baseline_v1.onnx
- Parity report: artifacts/logs/onnx_parity.md
- ORT benchmark CSV: artifacts/logs/bench_ort_cpu.csv
- ORT benchmark Markdown: docs/benchmarks/ort_cpu_baseline.md

## 4. Observed Results

### 4.1 ONNX export
- Export status: PASS
- Opset: 17
- Device: cpu
- Input shape: (1, 1, 64, 313)
- Output shape: (1, 1, 64, 313)

### 4.2 PT vs ONNX Runtime parity
- Mean abs err: 0.0000051313
- Max abs err: 0.0000104904
- MSE: 0.0000000000
- Allclose: True
- atol / rtol: 0.0001 / 0.0001
- Provider: CPUExecutionProvider

### 4.3 ORT inference
- Provider actual: CPUExecutionProvider
- Input source: random(shape=(1, 1, 64, 313), seed=42)
- Output min / max: -3.984205 / 2.514611

### 4.4 ORT benchmark
- Warmup / runs: 20 / 200
- P50 ms: 25.9407
- P95 ms: 31.4442
- P99 ms: 34.2762
- Mean ms: 25.5162
- Std ms: 3.9204
- Throughput: 39.1907 samples/s

## 5. Warning
- TracerWarning observed during ONNX export:
  Converting a tensor to a Python boolean might cause the trace to be incorrect
- Current judgment:
  1. Current fixed-shape path works.
  2. ONNX checker passed.
  3. PT vs ORT parity passed.
  4. This warning should be revisited in later shape-stability audit.

## 6. Conclusion
- Today host-side demo reproduction passed.
- baseline_v1 can continue to serve as the engineering baseline for Week 2.
- Next step: freeze environment versions, then start ADR draft.
