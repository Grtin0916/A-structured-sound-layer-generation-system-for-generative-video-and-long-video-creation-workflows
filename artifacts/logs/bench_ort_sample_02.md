# ORT Benchmark Report

## Summary
- ONNX: `artifacts/onnx/baseline_v1.onnx`
- Provider request: `CPUExecutionProvider`
- Provider actual: `['CPUExecutionProvider']`
- Input mode: `real`
- Input source: `/workspace/data/processed/esc10_miniset/valid/5-185579-B-41.wav`
- Input shape: `(1, 1, 64, 313)`
- Output shape: `(1, 1, 64, 313)`

## Benchmark setup
- Warmup runs: `20`
- Timed runs: `200`
- Batch size: `1`
- Intra-op threads: `0`
- Inter-op threads: `0`

## Latency (ms)
- min: `13.2296`
- p50: `15.6909`
- p90: `20.4682`
- p95: `24.7427`
- p99: `34.7884`
- max: `41.5877`
- mean: `16.9805`
- std: `4.1411`

## Throughput
- samples / sec: `58.8912`

## Conclusion
当前基准测试只统计 `session.run()` 的执行耗时，不包含 session 初始化与样本读取。
这一版结果可作为周四主线的第一版 ORT 延迟证据。
