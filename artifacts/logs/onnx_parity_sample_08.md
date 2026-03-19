# ONNX Parity Report

## Summary
- Status: **PASS**
- Config: `configs/train/baseline.yaml`
- Checkpoint: `artifacts/experiments/baseline_v1/checkpoints/best.pt`
- ONNX: `artifacts/onnx/baseline_v1.onnx`
- Report: `artifacts/logs/onnx_parity_sample_08.md`

## Input
- Sample source: `/workspace/data/processed/esc10_miniset/valid/5-198411-A-20.wav`
- Sample index: `8`
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
- Mean abs error: `0.0000015457`
- Max abs error: `0.0000162125`
- MSE: `0.0000000000`
- Allclose: `True`
- atol: `0.0001`
- rtol: `0.0001`

## Value range
- PyTorch min/max: `-12.0536918640` / `2.2640819550`
- ONNX min/max: `-12.0536909103` / `2.2640774250`

## Conclusion
当前样本下，PyTorch 与 ONNX Runtime 输出已经完成逐元素对齐检查。
如 `Allclose=True` 且误差量级很小，可认为周三主线的 parity check 基本通过。
