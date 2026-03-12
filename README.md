# A structured sound layer generation system for generative video and long video creation workflows

## 项目简介

本仓库用于记录我围绕“面向生成式视频与长视频创作工作流的结构化声音层生成系统”所做的学习、实验与最小可复现工程实践。

2026-3-9 Week1  
当前阶段的核心目标是先完成一个**贴合 AudioCraft 学习路线的最小训练与导出闭环**。

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
- onnx
- onnxruntime

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
python scripts/prepare_esc50_miniset.py \
  --src-root data/interim/esc50_raw \
  --dst-root data/processed/esc10_miniset \
  --train-per-class 8 \
  --valid-per-class 2 \
  --valid-fold 5 \
  --target-sr 16000 \
  --clip-seconds 5.0 \
  --seed 42
```

#### 3.2 构建 manifest

训练集 manifest：

```bash
python scripts/build_manifest.py \
  --input-dir data/processed/esc10_miniset/train \
  --output-path data/manifests/esc10_miniset_train.jsonl
```

验证集 manifest：

```bash
python scripts/build_manifest.py \
  --input-dir data/processed/esc10_miniset/valid \
  --output-path data/manifests/esc10_miniset_valid.jsonl
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

### 6. ONNX 导出与对齐校验

#### 6.1 导出 ONNX

导出脚本路径：

```text
src/export/export_onnx.py
```

执行导出：

```bash
python src/export/export_onnx.py \
  --config configs/train/baseline.yaml \
  --checkpoint artifacts/experiments/baseline_v1/checkpoints/best.pt \
  --output artifacts/onnx/baseline_v1.onnx \
  --device cpu \
  --verify
```

说明：

- 当前导出对象为 `baseline_v1` 的最优 checkpoint
- 当前输入规格为固定 shape：`(1, 1, 64, 313)`
- 导出后通过 `onnx.checker.check_model` 校验

#### 6.2 PyTorch / ONNX 对齐校验

对齐校验脚本路径：

```text
src/infer/compare_pt_onnx.py
```

执行校验：

```bash
python src/infer/compare_pt_onnx.py \
  --config configs/train/baseline.yaml \
  --checkpoint artifacts/experiments/baseline_v1/checkpoints/best.pt \
  --onnx artifacts/onnx/baseline_v1.onnx \
  --report artifacts/logs/onnx_parity.md \
  --device cpu \
  --provider CPUExecutionProvider
```

当前校验结果：

- Input shape: `(1, 1, 64, 313)`
- PT shape: `(1, 1, 64, 313)`
- ONNX shape: `(1, 1, 64, 313)`
- Mean abs error: `0.0000051313`
- Max abs error: `0.0000104904`
- MSE: `0.0000000000`
- Allclose: `True`

说明：

- 当前 `baseline_v1` 在固定输入规格下已完成 ONNX 导出与 parity check
- 当前结果表明 PyTorch 与 ONNX Runtime 前向输出已基本对齐
- 后续可在此基础上继续进行 ONNX Runtime 推理与 benchmark

#### 6.3 ONNX Runtime 单次推理

ORT 推理脚本路径：

```text
src/infer/infer_ort.py
```

执行单次 ORT 推理（推荐先用随机输入验证链路）：

```bash
python src/infer/infer_ort.py \
  --config configs/train/baseline.yaml \
  --onnx artifacts/onnx/baseline_v1.onnx \
  --provider CPUExecutionProvider \
  --input-mode random \
  --input-shape 1,1,64,313 \
  --seed 42
```

如需使用真实样本进行推理验证：

```bash
python src/infer/infer_ort.py \
  --config configs/train/baseline.yaml \
  --onnx artifacts/onnx/baseline_v1.onnx \
  --provider CPUExecutionProvider \
  --input-mode real \
  --sample-index 0
```

说明：

- 当前 ORT 推理默认使用 `baseline_v1.onnx`
- 当前默认 provider 为 `CPUExecutionProvider`
- 当前默认输入规格与已导出的固定 shape 保持一致：`(1, 1, 64, 313)`

#### 6.4 ONNX Runtime Benchmark

ORT benchmark 脚本路径：

```text
src/infer/bench_ort.py
```

执行 benchmark：

```bash
python src/infer/bench_ort.py \
  --config configs/train/baseline.yaml \
  --onnx artifacts/onnx/baseline_v1.onnx \
  --provider CPUExecutionProvider \
  --input-mode random \
  --input-shape 1,1,64,313 \
  --warmup 20 \
  --runs 200 \
  --csv artifacts/logs/bench_ort_cpu.csv \
  --markdown docs/benchmarks/ort_cpu_baseline.md
```

说明：

- 当前 benchmark 统计 `session.run()` 的推理耗时
- 当前记录指标包括 `P50 / P90 / P95 / P99 / mean / std`
- 当前 benchmark 结果默认输出为：
  - `artifacts/logs/bench_ort_cpu.csv`
  - `docs/benchmarks/ort_cpu_baseline.md`

#### 6.5 Docker 最小运行

Dockerfile 路径：

```text
docker/ort_cpu.Dockerfile
```

构建最小 CPU 镜像：

```bash
docker build \
  --build-arg http_proxy=http://host.docker.internal:7890 \
  --build-arg https_proxy=http://host.docker.internal:7890 \
  --build-arg HTTP_PROXY=http://host.docker.internal:7890 \
  --build-arg HTTPS_PROXY=http://host.docker.internal:7890 \
  -f docker/ort_cpu.Dockerfile \
  -t audio-ort-cpu:latest .
```

运行容器默认命令：

```bash
docker run --rm audio-ort-cpu:latest
```

如需在容器内执行 benchmark：

```bash
docker run --rm audio-ort-cpu:latest \
  python src/infer/bench_ort.py \
  --config configs/train/baseline.yaml \
  --onnx artifacts/onnx/baseline_v1.onnx \
  --provider CPUExecutionProvider \
  --input-mode random \
  --input-shape 1,1,64,313 \
  --warmup 20 \
  --runs 200 \
  --csv artifacts/logs/bench_ort_cpu.csv \
  --markdown docs/benchmarks/ort_cpu_baseline.md
```

说明：

- 当前 Docker 最小镜像用于验证 ORT 推理与 benchmark 链路
- 当前代理环境下，构建阶段需显式传入代理参数
- 当前最小镜像目标是先完成可构建、可运行、可复现的 CPU 推理闭环
---

### 7. 输出目录说明

ONNX / ORT / benchmark 相关输出保存在：

```text
artifacts/onnx/
artifacts/logs/
docs/benchmarks/
```

其中包括：

- `artifacts/onnx/baseline_v1.onnx`：导出的 ONNX 模型
- `artifacts/logs/onnx_parity.md`：PyTorch / ONNX 对齐校验报告
- `artifacts/logs/bench_ort_cpu.csv`：ORT benchmark 结构化结果
- `docs/benchmarks/ort_cpu_baseline.md`：ORT benchmark 文字化报告

---

## 每日更新小结
### 2026-03-9/3.10-周一/周二

- 完成 ESC-50 / ESC-10 最小数据集准备与处理流程
- 完成训练数据目录组织与 manifest 构建
- 完成 dataset 读取逻辑
- 完成最小 mel autoencoder baseline 模型
- 完成训练脚本，支持 checkpoint 保存与 TensorBoard 可视化
- 跑通 baseline_v1 训练流程并完成结果验证
- DSA（1768, 1071, 1431）
- Docker安装-hello world
- 完成周二主线任务的最小闭环交付

### 2026-03-11-周三

- 完成 `baseline_v1` 最优 checkpoint 的 ONNX 导出
- 完成 `src/export/export_onnx.py` 导出脚本编写与验证
- 完成 `baseline_v1.onnx` 模型导出，并通过 `onnx.checker.check_model`
- 完成 `src/infer/compare_pt_onnx.py` 对齐校验脚本编写
- 完成 PyTorch / ONNX Runtime 前向输出一致性校验
- 完成 `artifacts/logs/onnx_parity.md` 结果记录
- 跑通周三主线任务的最小导出与对齐闭环

---
### 2026-03-12-周四

- 完成 `src/infer/infer_ort.py` 编写与 ORT 单次推理链路验证
- 完成 `src/infer/bench_ort.py` 编写与 benchmark 测试流程搭建
- 完成 `artifacts/logs/bench_ort_cpu.csv` 结果导出
- 完成 `docs/benchmarks/ort_cpu_baseline.md` benchmark 报告记录
- 完成 `docker/ort_cpu.Dockerfile` 最小 CPU 镜像编写
- 完成周四主线任务的 ORT 推理、benchmark 与 Docker 最小闭环

--- 
## 当前阶段完成内容

### 当前阶段完成内容

- 完成 baseline v1 数据闭环验证
- 完成最小可训练 pipeline 搭建
- 完成 checkpoint 保存与 TensorBoard 记录
- 完成 ONNX 导出脚本
- 完成 PyTorch / ONNX 前向结果对齐校验
- 完成 `baseline_v1.onnx` 导出与 parity report 记录
- 完成 ONNX Runtime 单次推理脚本
- 完成 ORT benchmark 脚本与结果记录
- 完成 Docker 最小 CPU 推理镜像闭环