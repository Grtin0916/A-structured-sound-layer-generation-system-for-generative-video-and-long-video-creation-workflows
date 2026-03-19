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
- min: `15.3372`
- p50: `25.9407`
- p90: `29.9977`
- p95: `31.4442`
- p99: `34.2762`
- max: `35.1773`
- mean: `25.5162`
- std: `3.9204`

## Throughput
- samples / sec: `39.1907`

## Conclusion
当前基准测试只统计 `session.run()` 的执行耗时，不包含 session 初始化与样本读取。
这一版结果可作为周四主线的第一版 ORT 延迟证据。
