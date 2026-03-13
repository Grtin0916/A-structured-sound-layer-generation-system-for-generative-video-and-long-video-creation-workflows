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
当前样本下，PyTorch 与 ONNX Runtime 输出已经完成逐元素对齐检查。
如 `Allclose=True` 且误差量级很小，可认为周三主线的 parity check 基本通过。
