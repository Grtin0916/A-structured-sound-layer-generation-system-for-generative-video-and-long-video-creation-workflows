# ORT Docker vs Host CPU Compare

## 1. Scope

- Host CSV: `artifacts/logs/bench_ort_multi_sample.csv`
- Docker CSV: `artifacts/logs/bench_ort_multi_sample_docker.csv`
- Samples: host=10, docker=10
- Provider: CPUExecutionProvider
- Input shape contract: `(1, 1, 64, 313)`

## 2. Aggregate comparison

| Metric | Host mean | Docker mean | Delta (Docker-Host) | Delta % |
|---|---:|---:|---:|---:|
| p50_ms | 20.8858 | 15.8554 | -5.0304 | -24.09% |
| p95_ms | 26.2964 | 24.1452 | -2.1511 | -8.18% |
| p99_ms | 28.6323 | 34.1353 | 5.5031 | 19.22% |
| mean_ms | 20.9349 | 16.9738 | -3.9611 | -18.92% |
| throughput_samples_per_sec | 48.2247 | 59.1221 | 10.8973 | 22.60% |

## 3. Interpretation

- 在当前 10 个真实样本、CPUExecutionProvider、固定输入 shape `(1, 1, 64, 313)` 的条件下，Docker 的典型延迟优于 Host：`p50_ms` 下降 24.09%，`mean_ms` 下降 18.92%，吞吐提升 22.60%。
- 但 Docker 的尾延迟劣于 Host：`p99_ms` 上升 19.22%，说明容器路径在大多数情况下更快，但极端抖动更明显。
- 因此当前不能简单写成“Docker 优于 Host”或“Docker 劣于 Host”，更准确的结论是：**Docker 在典型时延上占优，在尾时延上存在额外波动。**
- 当前比较结论仅适用于固定 shape `(1, 1, 64, 313)`，不代表已经验证动态 shape。

## 4. Conclusion

- Host 与 Docker 的 ORT CPU 多样本 benchmark 已形成可对照证据。
- 当前 Docker CPU 复现链路已经可用，但其性能特征应表述为：**typical latency better, tail latency worse**。
- 下一步应进入 shape stability / export audit，明确当前 ONNX 导出与推理合同是否为 fixed-shape contract。