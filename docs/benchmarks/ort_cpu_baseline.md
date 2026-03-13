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
- min: `16.6849`
- p50: `23.6402`
- p90: `28.0465`
- p95: `29.7313`
- p99: `30.9265`
- max: `32.0433`
- mean: `23.7047`
- std: `3.1926`

## Throughput
- samples / sec: `42.1857`

## Conclusion
当前基准测试只统计 `session.run()` 的执行耗时，不包含 session 初始化与样本读取。
这一版结果可作为周四主线的第一版 ORT 延迟证据。
