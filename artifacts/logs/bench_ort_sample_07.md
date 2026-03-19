# ORT Benchmark Report

## Summary
- ONNX: `artifacts/onnx/baseline_v1.onnx`
- Provider request: `CPUExecutionProvider`
- Provider actual: `['CPUExecutionProvider']`
- Input mode: `real`
- Input source: `/workspace/data/processed/esc10_miniset/valid/5-195710-A-10.wav`
- Input shape: `(1, 1, 64, 313)`
- Output shape: `(1, 1, 64, 313)`

## Benchmark setup
- Warmup runs: `20`
- Timed runs: `200`
- Batch size: `1`
- Intra-op threads: `0`
- Inter-op threads: `0`

## Latency (ms)
- min: `13.7366`
- p50: `15.6425`
- p90: `18.0997`
- p95: `21.0099`
- p99: `23.4165`
- max: `32.4155`
- mean: `16.2227`
- std: `2.1736`

## Throughput
- samples / sec: `61.6420`

## Conclusion
当前基准测试只统计 `session.run()` 的执行耗时，不包含 session 初始化与样本读取。
这一版结果可作为周四主线的第一版 ORT 延迟证据。
