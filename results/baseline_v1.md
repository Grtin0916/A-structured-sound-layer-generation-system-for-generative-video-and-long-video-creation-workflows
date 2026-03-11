# baseline_v1

## 1. 实验目标

`baseline_v1` 的目标不是追求复杂模型或高指标，而是先完成一个**最小可复现训练闭环**，用于验证当前工程骨架能够支持：

- 数据准备与预处理
- manifest 驱动的数据读取
- 最小模型训练
- checkpoint 保存
- TensorBoard 可视化
- GitHub 交付

这一步的定位很明确：先把链路打通，再谈后续结构升级。别一上来就挑战全家桶，不然很容易把项目做成“看起来很忙，实际上没闭环”。

---

## 2. 数据处理流程

本实验使用 ESC-50 中筛选得到的 ESC-10 最小子集，构建 `esc10_miniset` 作为当前 baseline 的训练数据。

### 数据处理步骤

1. 下载原始 ESC-50 数据
2. 筛选 ESC-10 子集
3. 按类别构建平衡的 train / valid 划分
4. 对音频进行统一处理：
   - 转为 mono
   - 重采样到 16 kHz
   - 固定为 5 秒长度
5. 保存为处理后的 `.wav`
6. 为每条音频生成同名 `.json` sidecar 元数据
7. 生成 train / valid manifest

### 数据组织方式

```text
data/
├─ processed/
│  └─ esc10_miniset/
│     ├─ train/
│     └─ valid/
└─ manifests/
   ├─ esc10_miniset_train.jsonl
   └─ esc10_miniset_valid.jsonl
```

### 当前数据规模

- train: 80 条
- valid: 20 条

---

## 3. 数据读取方式

训练数据采用 **manifest 驱动** 的方式加载。

每条 manifest 记录包括：

- `audio_path`
- `meta_path`
- `sample_rate`
- `channels`
- `num_frames`
- `duration`

dataset 在读取时完成以下操作：

1. 加载波形
2. 再次保证 mono
3. 再次保证长度固定
4. 动态计算 log-mel
5. 读取 sidecar json 中的 `text` 与 `label`

最终返回字段包括：

- `waveform`
- `mel`
- `text`
- `label`
- `path`

这种设计的好处是工程上比较稳：磁盘上只保存规范化 wav 和 json，不提前把 mel 烤死，后面改特征参数也不用整批重做。

---

## 4. 模型结构

当前模型为一个**最小 mel autoencoder**，输入和输出均为 log-mel 频谱。

### 模型输入

- shape: `[B, 1, 64, T]`

### 模型结构概述

- stem 卷积块
- 两级 encoder 下采样
- bottleneck 特征压缩
- 两级 decoder 上采样
- 输出 head 重建 mel

### 结构特点

- 使用卷积块完成局部时频建模
- 使用 skip connection 保留细节
- 输出尺寸强制对齐输入尺寸
- 训练目标简单直接，适合最小 baseline 验证

当前版本的目的不是生成最终高质量音频，而是验证：

- 输入接口正确
- 前向传播稳定
- loss 可下降
- checkpoint 可保存
- 可视化链路可用

---

## 5. 训练配置

### 主要配置

- sample rate: 16000
- clip seconds: 5.0
- n_fft: 1024
- hop_length: 256
- win_length: 1024
- n_mels: 64

### 训练超参数

- batch size: 8
- optimizer: AdamW
- learning rate: 1e-3
- weight decay: 1e-4
- loss: `L1 + 0.1 * MSE`

### 训练脚本

```bash
python src/engine/train.py --config configs/train/baseline.yaml --device cuda
```

---

## 6. 输出结果

训练完成后，输出目录位于：

```text
artifacts/experiments/baseline_v1/
```

包含以下内容：

- `checkpoints/last.pt`
- `checkpoints/best.pt`
- `tensorboard/`
- `metrics.jsonl`
- `config_resolved.yaml`

### 当前结果说明

当前 baseline 已经完成以下验证：

- 训练流程可跑通
- 验证流程可跑通
- checkpoint 保存正常
- TensorBoard 记录正常
- baseline_v1 已形成可复现实验闭环

---

## 7. 当前结论

`baseline_v1` 的价值不在于性能有多强，而在于它已经把**最小工程链条**打通：

- 数据进得来
- 模型跑得动
- loss 能记录
- 结果能保存
- 可视化能查看
- 仓库能交付

这一步相当于把地基打实。后面不管你是想往 AudioCraft 风格继续靠，还是往更复杂的声音层生成任务延伸，至少现在不是“概念空转”了，而是已经有了一个能跑、能记、能复现的起点。

---

## 8. 下一步

基于 `baseline_v1`，后续可以继续推进：

- 补充更完整的实验记录与结果对照
- 增强评估指标与误差分析
- 尝试更强的时频建模结构
- 逐步从最小重建任务扩展到更贴近目标任务的声音层生成任务
