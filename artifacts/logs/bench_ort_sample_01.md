# ORT Benchmark Report

## Summary
- ONNX: `artifacts/onnx/baseline_v1.onnx`
- Provider request: `CPUExecutionProvider`
- Provider actual: `['CPUExecutionProvider']`
- Input mode: `real`
- Input source: `/workspace/data/processed/esc10_miniset/valid/5-185579-A-41.wav`
- Input shape: `(1, 1, 64, 313)`
- Output shape: `(1, 1, 64, 313)`

## Benchmark setup
- Warmup runs: `20`
- Timed runs: `200`
- Batch size: `1`
- Intra-op threads: `0`
- Inter-op threads: `0`

## Latency (ms)
- min: `13.4575`
- p50: `15.5998`
- p90: `22.5732`
- p95: `29.4246`
- p99: `42.2399`
- max: `44.1971`
- mean: `17.5170`
- std: `5.4365`

## Throughput
- samples / sec: `57.0874`

## Conclusion
当前基准测试只统计 `session.run()` 的执行耗时，不包含 session 初始化与样本读取。
这一版结果可作为周四主线的第一版 ORT 延迟证据。
