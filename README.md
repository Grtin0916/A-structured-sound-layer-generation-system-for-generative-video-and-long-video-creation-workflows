# A structured sound layer generation system for generative video and long video creation workflows

## 项目简介

本仓库用于记录我围绕“面向生成式视频与长视频创作工作流的结构化声音层生成系统”所做的学习、实验与最小可复现工程实践。

当前阶段的核心目标是先完成一个**贴合 AudioCraft 学习路线的最小训练闭环**，包括：

- 最小数据集准备与处理
- manifest 构建
- dataset 读取
- 最小模型训练
- checkpoint 保存
- TensorBoard 可视化
- GitHub 可复现交付

---

## 当前阶段完成内容

当前已完成 `baseline_v1` 的最小训练闭环：

- 完成 ESC-10 miniset 的下载、筛选与预处理
- 完成 `wav + sidecar json + manifest` 数据组织
- 完成 manifest 构建脚本
- 完成 dataset 读取逻辑
- 完成最小 mel autoencoder baseline
- 完成训练脚本，支持 train / valid
- 完成 checkpoint 保存（last / best）
- 完成 TensorBoard 记录与可视化验证

---

## 仓库使用说明

### 1. 环境准备

请先根据项目环境配置安装依赖，并确保可用：

- Python
- PyTorch / torchaudio
- tensorboard
- pyyaml
- soundfile
- librosa
- tqdm
- ffmpeg

---

### 2. 数据准备

原始数据建议放置在：

```text
data/interim/esc50_raw/
```

处理后训练数据放置在：

```text
data/processed/esc10_miniset/
```

生成的 manifest 放置在：

```text
data/manifests/
```

---

### 3. 数据处理命令

#### 3.1 生成最小训练数据集

```bash
python scripts/prepare_esc50_miniset.py   --src-root data/interim/esc50_raw   --dst-root data/processed/esc10_miniset   --train-per-class 8   --valid-per-class 2   --valid-fold 5   --target-sr 16000   --clip-seconds 5.0   --seed 42
```

#### 3.2 构建 manifest

```bash
python scripts/build_manifest.py   --input-dir data/processed/esc10_miniset/train   --output-path data/manifests/esc10_miniset_train.jsonl

python scripts/build_manifest.py   --input-dir data/processed/esc10_miniset/valid   --output-path data/manifests/esc10_miniset_valid.jsonl
```

---

### 4. 训练命令

训练配置文件路径：

```text
configs/train/baseline.yaml
```

启动训练：

```bash
python src/engine/train.py --config configs/train/baseline.yaml --device cuda
```

如需使用 CPU：

```bash
python src/engine/train.py --config configs/train/baseline.yaml --device cpu
```

---

### 5. TensorBoard

启动 TensorBoard：

```bash
tensorboard --logdir artifacts/experiments/baseline_v1/tensorboard --port 6006
```

浏览器访问：

```text
http://localhost:6006
```

---

### 6. 输出目录说明

训练输出默认保存在：

```text
artifacts/experiments/baseline_v1/
```

其中包括：

- `checkpoints/last.pt`：最后一个 checkpoint
- `checkpoints/best.pt`：当前最优 checkpoint
- `tensorboard/`：TensorBoard 日志
- `metrics.jsonl`：训练指标日志
- `config_resolved.yaml`：训练时解析后的配置文件

---

## 每日更新小结

> 这一部分按天追加，每天只记录“今天完成了什么”。

### 2026-03-11

- 完成 ESC-50 / ESC-10 最小数据集准备与处理流程
- 完成训练数据目录组织与 manifest 构建
- 完成 dataset 读取逻辑
- 完成最小 mel autoencoder baseline 模型
- 完成训练脚本，支持 checkpoint 保存与 TensorBoard 可视化
- 跑通 baseline_v1 训练流程并完成结果验证
- 完成周二主线任务的最小闭环交付

---

## 下一步计划

- 在当前最小 baseline 之上补充更稳定的评估记录
- 进一步整理训练结果与实验对照
- 逐步向更贴近 AudioCraft 的音频生成建模方式迭代
- 为后续更复杂的声音层生成任务打基础
