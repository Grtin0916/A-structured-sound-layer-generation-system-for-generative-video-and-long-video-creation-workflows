# ORT Benchmark Report

## Summary
- ONNX: `artifacts/onnx/baseline_v1.onnx`
- Provider request: `CPUExecutionProvider`
- Provider actual: `['CPUExecutionProvider']`
- Input mode: `real`
- Input source: `/workspace/data/processed/esc10_miniset/valid/5-193339-A-10.wav`
- Input shape: `(1, 1, 64, 313)`
- Output shape: `(1, 1, 64, 313)`

## Benchmark setup
- Warmup runs: `20`
- Timed runs: `200`
- Batch size: `1`
- Intra-op threads: `0`
- Inter-op threads: `0`

## Latency (ms)
- min: `13.5326`
- p50: `16.2762`
- p90: `20.3697`
- p95: `26.5358`
- p99: `40.3342`
- max: `43.8472`
- mean: `17.6247`
- std: `4.6443`

## Throughput
- samples / sec: `56.7385`

## Conclusion
当前基准测试只统计 `session.run()` 的执行耗时，不包含 session 初始化与样本读取。
这一版结果可作为周四主线的第一版 ORT 延迟证据。
