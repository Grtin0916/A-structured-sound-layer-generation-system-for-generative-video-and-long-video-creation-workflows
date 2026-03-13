# A structured sound layer generation system for generative video and long video creation workflows

## 项目概览

本仓库用于记录我围绕**“面向生成式视频与长视频创作工作流的结构化声音层生成系统”**所开展的学习、实验与最小可复现工程实践。

当前阶段不追求大而全，重点是先把一条扎实的工程主线跑通：

- 最小数据集准备
- manifest 构建
- baseline 模型训练
- checkpoint 导出与记录
- ONNX 导出与对齐校验
- ONNX Runtime 推理与 benchmark
- Docker 下的最小 CPU 验证链路

目前仓库已经完成了一个可复验的 `baseline_v1` 闭环，用来支撑后续 AudioCraft、音频生成工作流建模以及更复杂系统设计的继续推进。

---

## 当前阶段目标

当前阶段的核心目标是：

> 先完成一个贴合 AudioCraft 学习路线的最小训练、导出、推理与 benchmark 工程闭环。

这不是最终系统形态，而是第一阶段的工程地基。

---

## 快速开始

### 1. 环境

建议使用：

- Python 3.9
- PyTorch / torchaudio
- tensorboard
- pyyaml
- soundfile
- librosa
- tqdm
- ffmpeg
- onnx
- onnxruntime

进入环境后执行：

```bash
conda activate audio-mini
cd ~/work/audio_engineering_repo_skeleton_v1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
```

### 2. 2026-3-exi13 周五主线最小复现

如果你只想快速复现当前已经验证通过的 ONNX / ORT 链路，直接运行：

```bash
bash scripts/run_demo.sh demo
```

该命令会顺序执行：

- ONNX 导出
- PyTorch / ONNX Runtime 对齐校验
- ONNX Runtime 单次推理
- ONNX Runtime benchmark

### 3. 从数据准备开始的完整链路

```bash
bash scripts/run_demo.sh full
```

该命令会顺序执行：

- 数据准备
- manifest 构建
- 模型训练
- ONNX 导出
- 对齐校验
- ORT 推理
- ORT benchmark

说明：

- `full` 依赖原始 ESC-50 数据已放置在 `data/interim/esc50_raw/`
- 当前训练主验证路径为 `cuda`
- 若只想复现本周核心交付，优先使用 `demo`

---

## 当前已验证范围

| 模块 | 当前状态 | 说明 |
| --- | --- | --- |
| 数据准备 | 已完成 | ESC-50 → ESC-10 miniset 已跑通 |
| manifest 构建 | 已完成 | 训练集 / 验证集 JSONL 已生成 |
| baseline 训练 | 已完成 | `baseline_v1` 已完成训练与 checkpoint 保存 |
| TensorBoard 记录 | 已完成 | 训练日志可视化链路已验证 |
| ONNX 导出 | 已完成 | `best.pt -> baseline_v1.onnx` 已导出 |
| PT / ONNX 对齐 | 已完成 | 固定 shape 下 parity check 通过 |
| ORT 单次推理 | 已完成 | CPUExecutionProvider 已验证 |
| ORT benchmark | 已完成 | CSV 与 Markdown 结果已输出 |
| Docker 最小 CPU 闭环 | 已完成 | 可用于 ORT 推理 / benchmark 验证 |

---

## 关键输出

当前阶段的主要输出如下：

```text
artifacts/experiments/baseline_v1/checkpoints/best.pt
artifacts/onnx/baseline_v1.onnx
artifacts/logs/onnx_parity.md
artifacts/logs/bench_ort_cpu.csv
docs/benchmarks/ort_cpu_baseline.md
```

---

## 主要流程说明

### 1. 数据准备

原始数据建议放置在：

```text
data/interim/esc50_raw/
```

处理后数据输出到：

```text
data/processed/esc10_miniset/
```

manifest 输出到：

```text
data/manifests/
```

核心命令：

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

python scripts/build_manifest.py \
  --input-dir data/processed/esc10_miniset/train \
  --output-path data/manifests/esc10_miniset_train.jsonl

python scripts/build_manifest.py \
  --input-dir data/processed/esc10_miniset/valid \
  --output-path data/manifests/esc10_miniset_valid.jsonl
```

### 2. 模型训练

训练配置文件：

```text
configs/train/baseline.yaml
```

当前主验证命令：

```bash
python src/engine/train.py --config configs/train/baseline.yaml --device cuda
```

如需在无 GPU 场景下做最小调试，可尝试：

```bash
python src/engine/train.py --config configs/train/baseline.yaml --device cpu
```

### 3. TensorBoard

```bash
tensorboard --logdir artifacts/experiments/baseline_v1/tensorboard --port 6006
```

浏览器访问：

```text
http://localhost:6006
```

### 4. ONNX 导出与对齐校验

导出命令：

```bash
python src/export/export_onnx.py \
  --config configs/train/baseline.yaml \
  --checkpoint artifacts/experiments/baseline_v1/checkpoints/best.pt \
  --output artifacts/onnx/baseline_v1.onnx \
  --device cpu \
  --verify
```

对齐校验命令：

```bash
python src/infer/compare_pt_onnx.py \
  --config configs/train/baseline.yaml \
  --checkpoint artifacts/experiments/baseline_v1/checkpoints/best.pt \
  --onnx artifacts/onnx/baseline_v1.onnx \
  --report artifacts/logs/onnx_parity.md \
  --device cpu \
  --provider CPUExecutionProvider
```

当前固定输入规格下的对齐结果：

- Input shape: `(1, 1, 64, 313)`
- Mean abs error: `0.0000051313`
- Max abs error: `0.0000104904`
- MSE: `0.0000000000`
- Allclose: `True`

### 5. ONNX Runtime 推理与 benchmark

单次 ORT 推理：

```bash
python src/infer/infer_ort.py \
  --config configs/train/baseline.yaml \
  --onnx artifacts/onnx/baseline_v1.onnx \
  --provider CPUExecutionProvider \
  --input-mode random \
  --input-shape 1,1,64,313 \
  --seed 42
```

ORT benchmark：

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

### 6. Docker 最小 CPU 验证

Dockerfile：

```text
docker/ort_cpu.Dockerfile
```

构建：

```bash
docker build \
  --build-arg http_proxy=http://host.docker.internal:7890 \
  --build-arg https_proxy=http://host.docker.internal:7890 \
  --build-arg HTTP_PROXY=http://host.docker.internal:7890 \
  --build-arg HTTPS_PROXY=http://host.docker.internal:7890 \
  -f docker/ort_cpu.Dockerfile \
  -t audio-ort-cpu:latest .
```

运行：

```bash
docker run --rm audio-ort-cpu:latest
```

---

## 已知边界与当前限制

- 当前 ONNX 导出与对齐验证基于固定输入 shape：`(1, 1, 64, 313)`
- 当前导出过程中存在 tracing warning，因此动态 shape / 变长输入的泛化支持仍需后续补充验证
- 当前 ORT 路径的主验证 provider 为 `CPUExecutionProvider`
- 当前 README 记录的是第一阶段的最小可复现闭环，不代表最终系统能力上限

---

## 每日更新小结

### 2026-03-09 / 2026-03-10（周一 / 周二）

- 完成 ESC-50 / ESC-10 最小数据集准备与处理流程
- 完成训练数据目录组织与 manifest 构建
- 完成 dataset 读取逻辑
- 完成最小 mel autoencoder baseline 模型
- 完成训练脚本，支持 checkpoint 保存与 TensorBoard 可视化
- 跑通 `baseline_v1` 训练流程并完成结果验证
- 完成 DSA 题目练习：1768、1071、1431
- 完成 Docker 安装与 `hello-world` 验证
- 完成周二主线任务的最小闭环交付

### 2026-03-11（周三）

- 完成 `baseline_v1` 最优 checkpoint 的 ONNX 导出
- 完成 `src/export/export_onnx.py` 导出脚本编写与验证
- 完成 `baseline_v1.onnx` 模型导出，并通过 `onnx.checker.check_model`
- 完成 `src/infer/compare_pt_onnx.py` 对齐校验脚本编写
- 完成 PyTorch / ONNX Runtime 前向输出一致性校验
- 完成 `artifacts/logs/onnx_parity.md` 结果记录
- 跑通周三主线任务的最小导出与对齐闭环

### 2026-03-12（周四）

- 完成 `src/infer/infer_ort.py` 编写与 ORT 单次推理链路验证
- 完成 `src/infer/bench_ort.py` 编写与 benchmark 测试流程搭建
- 完成 `artifacts/logs/bench_ort_cpu.csv` 结果导出
- 完成 `docs/benchmarks/ort_cpu_baseline.md` benchmark 报告记录
- 完成 `docker/ort_cpu.Dockerfile` 最小 CPU 镜像编写
- 完成周四主线任务的 ORT 推理、benchmark 与 Docker 最小闭环

### 2026-03-13（周五）

- 完成周一至周四主线命令冻结与整理
- 完成 `scripts/run_demo.sh` 封板，提供 `demo / full` 两类主入口
- 完成 `bash scripts/run_demo.sh demo` smoke test
- 完成 ONNX 导出、parity check、ORT 推理与 benchmark 的周五复验
- 完成 `.gitignore` 收口与仓库边界清理
- 完成 README 的结构化重写与交付版收口

---

## 当前阶段总结

到目前为止，这个仓库已经完成了第一阶段最重要的几件事：

- 把数据、训练、导出、推理、benchmark 串成了一条可复验主线
- 把 `baseline_v1` 做成了一个有 checkpoint、有日志、有 ONNX、有 benchmark 的最小工程样本
- 把周内零散的实验操作收成了可执行脚本与清晰的 README 入口
- 为后续继续推进 AudioCraft、音频生成模型、结构化声音层建模和更完整系统实现打下了工程基础
