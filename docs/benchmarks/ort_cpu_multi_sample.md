# ORT CPU Multi-sample Benchmark

## 1. Scope

本报告记录多样本验证结果。

相关输入：
- ONNX: `artifacts/onnx/baseline_v1.onnx`
- Manifest: `data/manifests/esc10_miniset_valid.jsonl`

相关输出：
- Parity CSV: `artifacts/logs/parity_multi_sample.csv`
- ORT multi-sample CSV: `artifacts/logs/bench_ort_multi_sample.csv`

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
- mean(p50_ms): 20.8858
- mean(p95_ms): 26.2964
- mean(p99_ms): 28.6323
- mean(mean_ms): 20.9349
- mean(throughput_samples_per_sec): 48.2247

### 3.2 Slowest / fastest samples
- slowest sample by mean_ms: sample 0
  - p50_ms: 26.7298
  - mean_ms: 26.4419
  - throughput: 37.8188

- fastest sample by mean_ms: sample 9
  - p50_ms: 17.8051
  - mean_ms: 18.3423
  - throughput: 54.5189

---

## 4. Cross-observation

- parity 离群点为 `sample 6`；
- latency 最慢样本为 `sample 0`；
- 当前没有观察到“误差更大必然更慢”的直接对应关系。

这说明：
1. 数值一致性波动与运行时性能波动未必由同一因素主导；
2. 后续 shape audit 与 Docker 对照需要分别跟踪“正确性”和“性能”两条线。

---

## 5. Conclusion

截至当前：
1. 多样本 PT / ORT 对齐结果已形成 CSV 证据；
2. ORT CPU 多样本延迟与吞吐结果已形成 CSV 证据；
3. 当前脚本已将周二 ad-hoc 循环固化为可重复执行入口；
4. 下一步优先级应为：
   - 复查离群样本
   - 进入 shape stability / export audit
   - 再进行今日收口提交
