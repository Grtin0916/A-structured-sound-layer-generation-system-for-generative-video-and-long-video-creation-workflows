# ORT CPU Multi-sample Benchmark (Docker)

## 1. Scope

本报告记录多样本验证结果。

相关输入：
- ONNX: `artifacts/onnx/baseline_v1.onnx`
- Manifest: `data/manifests/esc10_miniset_valid.jsonl`

相关输出：
- Parity CSV: `artifacts/logs/parity_multi_sample_docker.csv`
- ORT multi-sample CSV: `artifacts/logs/bench_ort_multi_sample_docker.csv`

---

## 2. Parity Summary

### 2.1 Overall result
- Samples: 10
- status=ok: 10
- allclose=True: 10
- pass rate: 100.00%

### 2.2 Error statistics
- mean(mean_abs_err): 0.0000050043
- median(mean_abs_err): 0.0000012473
- max(mean_abs_err): 0.0000336710
- max(max_abs_err): 0.0001077652

### 2.3 Observation
- 多数样本的 `mean_abs_err` 处于较小量级；
- 当前多样本数值一致性成立；
- 如存在离群样本，应继续在后续 shape / export 审计中单独复查。

---

## 3. ORT CPU Latency Summary

### 3.1 Aggregate statistics
- samples: 10
- mean(p50_ms): 15.8554
- mean(p95_ms): 24.1452
- mean(p99_ms): 34.1353
- mean(mean_ms): 16.9738
- mean(throughput_samples_per_sec): 59.1221

### 3.2 Slowest / fastest samples
- slowest sample by mean_ms: sample 0
  - p50_ms: 17.4737
  - mean_ms: 19.2909
  - throughput: 51.8378

- fastest sample by mean_ms: sample 8
  - p50_ms: 14.9078
  - mean_ms: 15.3745
  - throughput: 65.0428

---

## 4. Conclusion

截至当前：
1. Docker 环境下 PT / ORT 多样本对齐结果已形成 CSV 证据；
2. Docker 环境下 ORT CPU 多样本延迟与吞吐结果已形成 CSV 证据；
3. 当前 Docker 运行合同为：镜像 + bind mount（至少挂载仓库根目录，或挂载 data/ 与 artifacts/）；
4. 当前 ONNX 推理链验证的是固定输入 shape `(1, 1, 64, 313)`；
5. 下一步优先级应为：
   - 生成 host vs Docker 对比报告
   - 完成 shape stability / export audit
   - 收口 README 与周三提交