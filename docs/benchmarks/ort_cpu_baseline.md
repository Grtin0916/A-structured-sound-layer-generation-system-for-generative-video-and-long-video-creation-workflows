# ORT Benchmark Report

## Summary
- ONNX: `artifacts/onnx/baseline_v1.onnx`
- Provider request: `CPUExecutionProvider`
- Provider actual: `['CPUExecutionProvider']`
- Input mode: `random`
- Input source: `random(shape=(1, 1, 64, 313), seed=42)`
- Input shape: `(1, 1, 64, 313)`
- Output shape: `(1, 1, 64, 313)`

## Benchmark setup
- Warmup runs: `20`
- Timed runs: `200`
- Batch size: `1`
- Intra-op threads: `0`
- Inter-op threads: `0`

## Latency (ms)
- min: `14.8109`
- p50: `28.0431`
- p90: `40.6832`
- p95: `45.4107`
- p99: `52.9994`
- max: `55.7772`
- mean: `28.8135`
- std: `9.4486`

## Throughput
- samples / sec: `34.7060`

## Conclusion
当前基准测试只统计 `session.run()` 的执行耗时，不包含 session 初始化与样本读取。
这一版结果可作为周四主线的第一版 ORT 延迟证据。
