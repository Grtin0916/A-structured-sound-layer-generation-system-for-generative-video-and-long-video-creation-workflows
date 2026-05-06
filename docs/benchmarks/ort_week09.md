# Week09 ORT CPU Multi-sample Benchmark

## 1. Scope

本报告记录 Week09 主基地 ORT CPU 多样本 benchmark 首轮验证结果。

- Run date: `2026-05-06`
- Provider: `CPUExecutionProvider`
- Warmup / runs: `20 / 200`

相关输入：
- ONNX: `artifacts/onnx/baseline_v1.onnx`
- Manifest: `data/manifests/esc10_miniset_valid.jsonl`

相关输出：
- Parity CSV: `artifacts/logs/week09_parity_multi_sample.csv`
- ORT multi-sample CSV: `artifacts/logs/week09_bench_ort_multi_sample.csv`

---

## 2. Parity Summary

### 2.1 Overall result
- Samples: 10
- status=ok: 10
- allclose=True: 10
- pass rate: 100.00%

### 2.2 Error statistics
- mean(mean_abs_err): 0.0000050560
- median(mean_abs_err): 0.0000012982
- max(mean_abs_err): 0.0000334792
- max(max_abs_err): 0.0001068115

### 2.3 Observation
- 多数样本的 `mean_abs_err` 处于较小量级；
- 当前多样本数值一致性成立；
- 如存在离群样本，应继续在后续 shape / export 审计中单独复查。

---

## 3. ORT CPU Latency Summary

### 3.1 Aggregate statistics
- samples: 10
- mean(p50_ms): 14.8277
- mean(p95_ms): 15.4098
- mean(p99_ms): 16.2393
- mean(mean_ms): 14.8022
- mean(throughput_samples_per_sec): 67.8790

### 3.2 Slowest / fastest samples
- slowest sample by mean_ms: sample 2
  - p50_ms: 15.7764
  - mean_ms: 15.8258
  - throughput: 63.1878

- fastest sample by mean_ms: sample 1
  - p50_ms: 13.5552
  - mean_ms: 13.4399
  - throughput: 74.4055

---

## 4. Conclusion

截至当前：
1. 多样本 PT / ORT 对齐结果已形成 CSV 证据；
2. ORT CPU 多样本延迟与吞吐结果已形成 CSV 证据；
3. 当前脚本已将周二 ad-hoc 循环固化为可重复执行入口；
4. 下一步优先级应为：
   - 复查离群样本
   - 进入 shape stability / export audit
   - 再进行今日收口提交


---

## 5. Evidence Boundary

- 当前结果只覆盖 `CPUExecutionProvider`，没有覆盖 CUDA / TensorRT / ORT GPU provider。
- 当前输入合同为固定 shape `(1, 1, 64, 313)`，不能外推为动态 shape 已验证。
- 当前统计范围只包含 `session.run()` 相关推理路径，不代表端到端音频预处理 + 推理 + 后处理总延迟。
- 后续 Week09 可继续补 graph optimization / thread config 对照，但本轮证据已经可作为 Week09 baseline。
