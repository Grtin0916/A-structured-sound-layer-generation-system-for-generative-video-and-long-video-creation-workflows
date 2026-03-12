# ONNX Parity Report

## Summary
- Status: **PASS**
- Config: `configs/train/baseline.yaml`
- Checkpoint: `artifacts/experiments/baseline_v1/checkpoints/best.pt`
- ONNX: `artifacts/onnx/baseline_v1.onnx`
- Report: `artifacts/logs/onnx_parity.md`

## Input
- Sample source: `/home/GRT/work/audio_engineering_repo_skeleton_v1/data/processed/esc10_miniset/valid/5-177957-C-40.wav`
- Sample index: `0`
- Input shape: `(1, 1, 64, 313)`
- ONNX input name: `mel`
- ONNX output name: `recon_mel`

## Runtime
- PyTorch device: `cpu`
- ONNX Runtime provider request: `CPUExecutionProvider`
- ONNX Runtime provider actual: `['CPUExecutionProvider']`

## Metrics
- Shape equal: `True`
- PyTorch output shape: `(1, 1, 64, 313)`
- ONNX output shape: `(1, 1, 64, 313)`
- Mean abs error: `0.0000051313`
- Max abs error: `0.0000104904`
- MSE: `0.0000000000`
- Allclose: `True`
- atol: `0.0001`
- rtol: `0.0001`

## Value range
- PyTorch min/max: `-9.7427139282` / `3.8616173267`
- ONNX min/max: `-9.7427082062` / `3.8616178036`

## Conclusion
结论：baseline_v1 在 CPU / ONNX Runtime 下与 PyTorch 前向结果对齐通过。
当前固定输入规格为 (1, 1, 64, 313)，mean abs error=5.1313e-06，max abs error=1.04904e-05，allclose=True。
可进入周四 ORT 推理与 bench 阶段。
