# 2026-03-18 Week02 Shape Audit

## 1. Objective

本审计用于确认当前 `artifacts/onnx/baseline_v1.onnx` 的输入合同是否为固定 shape，避免将当前导出结果误写为 dynamic-shape support。

## 2. Audit target

- ONNX: `artifacts/onnx/baseline_v1.onnx`
- Provider: `CPUExecutionProvider`

## 3. Cases

- valid_ref: `(1, 1, 64, 313)`
- valid_same_shape_alt: `(1, 1, 64, 313)`
- invalid_width_minus_1: `(1, 1, 64, 312)`
- invalid_width_plus_1: `(1, 1, 64, 314)`
- invalid_channel_2: `(1, 2, 64, 313)`
- invalid_batch_2: `(2, 1, 64, 313)`

## 4. Result summary

见：`artifacts/logs/week02_shape_audit.csv`

本次共审计 6 个 case：
- 通过：2 个
- 失败：4 个
- 预期与实际一致：6/6

典型失败报错如下：
- `invalid_width_minus_1`: `Got invalid dimensions for input: mel ... index: 3 Got: 312 Expected: 313`
- `invalid_width_plus_1`: `Got invalid dimensions for input: mel ... index: 3 Got: 314 Expected: 313`
- `invalid_channel_2`: `Got invalid dimensions for input: mel ... index: 1 Got: 2 Expected: 1`
- `invalid_batch_2`: `Got invalid dimensions for input: mel ... index: 0 Got: 2 Expected: 1`

这表明当前模型输入声明为 `mel: [1, 1, 64, 313]`，宽度、通道数和 batch 维度均为固定合同的一部分，而非可自由变化的动态维度。

## 5. Conclusion

当前 ONNX / ORT 推理合同应视为 **fixed-shape contract**。  
在本次审计范围内，只有参考 shape `(1, 1, 64, 313)` 能稳定通过；宽度、通道数或 batch 变化均不应视为已支持输入。

## 6. Next step

- 若后续需要支持动态输入，应回到导出链重新设计 `dynamic_shapes` / 导出配置；
- 当前阶段应继续按固定 shape 口径记录 benchmark 与复现实验结果。