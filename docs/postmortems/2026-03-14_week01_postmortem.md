# Week 1 Postmortem (2026-03-14)

## 1. 本周完成了什么
- baseline_v1 训练闭环已完成
- best.pt -> baseline_v1.onnx 导出成功
- PT / ONNX Runtime parity check 通过
- ORT CPU inference 与 benchmark 已完成

## 2. 这周最关键的问题
### 问题 A：ONNX 导出阶段出现 TracerWarning
- 现象：
  - `Converting a tensor to a Python boolean might cause the trace to be incorrect`
- 触发位置：
  - `src/models/model.py:132`
  - `if out.shape[-2:] != x.shape[-2:]:`
- 当前影响：
  - 在固定输入 shape `(1, 1, 64, 313)` 下未阻塞导出
  - 当前 parity 通过，Allclose=True
- 潜在风险：
  - trace 图可能绑定到当前控制流
  - 对其他时长/shape 的输入，导出图不一定稳定
- 当前结论：
  - 这一版 ONNX 可以作为 week01 基线证据
  - 但还不能宣称“对动态 shape 已充分验证”

## 问题 B：Docker 镜像中 torchaudio.load 依赖未闭合
- 现象：
  - 容器内执行 `bash scripts/run_demo.sh demo` 时，`torchaudio.load()` 报错：
    `ImportError: TorchCodec is required for load_with_torchcodec`
- 根因：
  - Dockerfile 安装了 `torch` 和 `torchaudio`，但没有显式安装 `torchcodec`
  - 同时依赖未做版本钉住，存在“latest 漂移”风险
- 修复：
  - 在 Dockerfile 中补装 `torchcodec`
  - 后续将 `torch` / `torchaudio` 版本固定为本地已验证版本
- 影响范围：
  - 影响容器内数据集读取、ONNX 导出链路
  - 不影响本地已完成的 Week 1 主线证据

## 3. 指标证据
- Input shape: `(1, 1, 64, 313)`
- Mean abs error: `0.0000051313`
- Max abs error: `0.0000104904`
- MSE: `0.0000000000`
- Allclose: `True`
- ORT p50: `23.3445 ms`
- ORT p99: `30.9132 ms`
- Throughput: `42.1391 samples/s`

## 4. 根因判断
- 当前导出链更偏向静态 shape 验证
- 模型里存在 Python 级条件分支参与 trace
- 当前验证覆盖的是“已知样本 + 已知 shape”，不是完整输入空间

## 5. 已采取动作
- 保留 parity 报告
- 保留 benchmark 报告
- 在 weekly summary 中明确写明固定 shape 口径
- 不把当前结论外推成动态 shape 全面可用

## 6. 下周动作
- 检查 `model.py:132` 这段控制流能否改成更稳定的导出形式
- 增加 2~3 组不同长度/shape 的导出与 parity 检查
- 重新做 ORT benchmark，比较不同 shape 的延迟变化
- 视情况补 dynamic axes 导出实验

## 7. 一句话结论
Week 1 的 ONNX / ORT 工程闭环已经成立，但当前成立范围主要是固定输入 shape 的最小可复现链路。