# ORT Benchmark Report

## Summary
- ONNX: `artifacts/onnx/baseline_v1.onnx`
- Provider request: `CPUExecutionProvider`
- Provider actual: `['CPUExecutionProvider']`
- Input mode: `real`
- Input source: `/workspace/data/processed/esc10_miniset/valid/5-194930-A-1.wav`
- Input shape: `(1, 1, 64, 313)`
- Output shape: `(1, 1, 64, 313)`

## Benchmark setup
- Warmup runs: `20`
- Timed runs: `200`
- Batch size: `1`
- Intra-op threads: `0`
- Inter-op threads: `0`

## Latency (ms)
- min: `13.2651`
- p50: `15.3478`
- p90: `21.5433`
- p95: `24.4779`
- p99: `43.1817`
- max: `56.5874`
- mean: `17.0441`
- std: `5.5271`

## Throughput
- samples / sec: `58.6713`

## Conclusion
当前基准测试只统计 `session.run()` 的执行耗时，不包含 session 初始化与样本读取。
这一版结果可作为周四主线的第一版 ORT 延迟证据。
