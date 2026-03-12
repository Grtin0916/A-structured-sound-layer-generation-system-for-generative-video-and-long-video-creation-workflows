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
- min: `17.1483`
- p50: `23.7349`
- p90: `28.8991`
- p95: `29.7665`
- p99: `31.5992`
- max: `38.2860`
- mean: `24.2473`
- std: `3.3955`

## Throughput
- samples / sec: `41.2416`

## Conclusion
当前基准测试只统计 `session.run()` 的执行耗时，不包含 session 初始化与样本读取。
这一版结果可作为周四主线的第一版 ORT 延迟证据。
