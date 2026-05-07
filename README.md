# A structured sound layer generation system for generative video and long video creation workflows

## Verified Scope

<!-- week09-ort-profile-verified-start -->

### Week09 ORT profiling / CPU / memory verified update

Verified on 2026-05-07:

- Added `scripts/bench_ort_profile.py` for a conservative ONNX Runtime CPU profiling probe.
- Generated profiling metrics at `artifacts/benchmarks/ort_week09_profile_20260507.json`.
- Generated ORT trace file at `artifacts/profiles/ort_week09_profile_20260507_2026-05-07_09-53-52.json`.
- Captured run log at `artifacts/logs/week09_ort_profile_20260507.log`.
- Fixed model/input scope: `artifacts/onnx/baseline_v1.onnx`, input shape `[1, 1, 64, 313]`, output shape `[1, 1, 64, 313]`.
- Fixed runtime scope: provider `CPUExecutionProvider`, graph optimization `disable`, threads `1`, warmup/runs `1/5`.
- Observed median latency `201.8881 ms`, p95 latency `204.119 ms`, CPU user/system `0.7862s/0.2262s`, max RSS after `660.8125 MB`.

Boundary: this update verifies Week09 profiling / CPU / memory evidence only. It does not claim CUDA provider coverage, TensorRT acceleration, production tuning, GPU benchmarking, or final runtime selection.

<!-- week09-ort-profile-verified-end -->


- 已完成 S1 阶段总结：`docs/weekly/2026-05-01_stage_s1_mainbase.md` 已收口 W4-W8 的 serving contract、blueprint seed、generator audit、scorecard_v1 与 Week08 metrics / Prometheus / dashboard baseline 证据，并明确 custom `mainbase_*` 指标、完整 Grafana 导入和生产级告警仍未验证。

- 已完成 Week05 的 seed 基线落盘，仓库内已有 `artifacts/manifests/seed_0001.json` 可作为结构化输入映射样例。
- 已完成 Week06 的 generator audit 证据收口，包括 audit 模板、审计日志、postmortem 与 README verified scope 同步。
- 已完成 Week07 的 `docs/evals/scorecard_v1.md` 与 `artifacts/benchmarks/scorecard_example.csv` 初版落盘；当前 sample CSV 已扩到 10 条样例，覆盖 `functional`、`performance`、`failure_regression` 三类入口。
- 已完成 Week08 的 metrics 命名草稿：`docs/design/metrics_naming.md`，统一了 `mainbase_contract_*`、`mainbase_runtime_*`、`mainbase_artifact_*` 三类指标命名口径。
- 已完成 Week08 的最小 `/metrics` 暴露入口：`src/serving/app.py` 已通过 `prometheus_client.make_asgi_app()` 挂载 Prometheus ASGI app，`tests/test_api_smoke.py` 已新增 metrics smoke test，`pytest tests/test_api_smoke.py -q` 当前为 `8 passed`。
- 已新增 Week08 Prometheus 最小抓取配置：`monitoring/prometheus/prometheus.yml`，当前 job 为 `mainbase-serving`，metrics path 为 `/metrics`，本地验证证据保存在 `artifacts/logs/week08_metrics_*`。
- 当前 `/metrics` 访问语义已验证：`/metrics` 返回 307 redirect，`/metrics/` 返回 Prometheus text format，并包含默认 `python_info` 与 `process_*` 指标。

一句话说，当前仓库已经从 Week07 的 scorecard 可引用样例阶段，推进到 Week08 的 metrics naming、Prometheus 暴露入口与最小 scrape config 阶段；它可以作为后续 dashboard、custom metrics 与 formal eval observability 的起点。

* * *
- 已完成 Week08 Grafana dashboard 初版：`monitoring/grafana/dashboards/mainbase-overview.json` 已落盘并通过 `python -m json.tool` 校验；当前面板仅覆盖 `python_info`、`process_resident_memory_bytes`、`process_cpu_seconds_total`、`process_start_time_seconds` 等默认 Prometheus 指标，不声明 custom `mainbase_*` 指标已经接入。

- 已完成 Week08 真实 Prometheus scrape target 验证：使用 Prometheus server 容器读取 `monitoring/prometheus/prometheus.yml`，在宿主机端口 `9091` 复验 `mainbase-serving` target 进入 `up` 状态，`lastError` 为空；证据保存在 `artifacts/logs/week08_prometheus_targets_20260429_retry9091_after_wait.json`。

## Not Yet Verified

- scorecard 当前仍是 v1 sample，不是严格统一的方法学与自动化评测收口。
- failure regression 已形成最小 taxonomy，当前已有 5 条样例，但尚未扩到更系统的错误分类与批量回归执行。
- 当前未把 `facebook/musicgen-medium`、`melody` 或更大模型纳入正式可比较结论。
- 当前 `/metrics` 已能暴露 Prometheus text format，但尚未接入自定义 `mainbase_contract_*`、`mainbase_runtime_*`、`mainbase_artifact_*` 指标采集逻辑。
- Prometheus scrape target 已通过真实 Prometheus server 复验；后续仍未验证长期运行、告警、dashboard provisioning 与生产级 scrape 拓扑。
- Grafana dashboard 已完成默认指标初版，但尚未接入更完整的业务面板、dashboard provisioning 与真实 Grafana 导入验证。
- 更完整的 serving runtime、性能优化与多阶段评测闭环仍未完成。

这些方向已经进入路线规划，但截至当前仓库状态，还不应写成“已完成”。

* * *
- custom `mainbase_*` 指标、更完整的 Grafana dashboard 面板、dashboard provisioning 与生产级告警仍未验证；当前 dashboard 只作为 Week08 默认指标入口。

## Next Hard Milestone

接下来的硬里程碑按顺序是：

1. Week08：收口 metrics 化最小证据
   * 保留 `/metrics` 与 `/metrics/` 的 curl 日志证据
   * 固定 `docs/design/metrics_naming.md` 与 `monitoring/prometheus/prometheus.yml`
   * 明确当前只完成默认 Python/process metrics 暴露、Prometheus target up 复验与 dashboard 初版，不把 custom metrics、完整 Grafana 导入与生产级告警提前写成已完成

2. Week08：进入 dashboard 初版复验与 custom metrics 预热
   * 复验 `monitoring/grafana/dashboards/mainbase-overview.json` 的 JSON 格式、默认指标查询与 README 边界描述
   * 选择 1–2 个低风险 custom metrics 接入点
   * 为后续 formal eval、runtime benchmark 与 failure regression observability 留入口

## 项目概览

本仓库用于记录我围绕**“面向生成式视频与长视频创作工作流的结构化声音层生成系统”**所开展的学习、实验与最小可复现工程实践。

当前阶段不追求大而全，重点是先把一条扎实的工程主线跑通，并确保每一步都能沉淀为可引用的代码、日志、文档与产物，而不是停留在概念设计层。

当前已跑通的工程主线包括：

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

> 先完成一个贴合 AudioCraft 学习路线的最小训练、导出、推理与 benchmark 工程闭环，并在此基础上逐步推进 contract、seed、audit、eval 与 serving 相关能力。

这不是最终系统形态，而是第一阶段的工程地基。

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

进入环境后建议：

```bash
conda activate audio-mini
cd ~/work/audio_engineering_repo_skeleton_v1
```

仅在需要运行依赖 conda 动态库的 AudioCraft / MusicGen 相关命令时，再按单条命令临时注入库路径：

```bash
LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH" python your_script.py
```

不建议长期在整个 shell 中全局 `export LD_LIBRARY_PATH`，否则可能影响 git / ssh / OpenSSL 等系统工具。

### 2. 2026-03-20 周五主线最小复现

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
docker/Dockerfile
```

构建：

```bash
docker build \
  --build-arg http_proxy=http://host.docker.internal:7890 \
  --build-arg https_proxy=http://host.docker.internal:7890 \
  --build-arg HTTP_PROXY=http://host.docker.internal:7890 \
  --build-arg HTTPS_PROXY=http://host.docker.internal:7890 \
  -f docker/Dockerfile \
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
- 完成 `docker/Dockerfile` 最小 CPU 镜像编写
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

## Week 1 Evidence Index
- Weekly summary: `docs/weekly/2026-03-14_week01_summary.md`
- Postmortem: `docs/postmortems/2026-03-14_week01_postmortem.md`
- ORT benchmark: `docs/benchmarks/ort_cpu_baseline.md`
- Baseline result note: `results/baseline_v1.md`


## Week05 seed entry (verified)

当前 Week05 已验证的最小入口：

~~~bash
python scripts/build_blueprint_seed.py --input artifacts/logs/week03_video_probe_smoke.json --output artifacts/manifests/seed_0001.json
pytest -q tests/regression/test_blueprint_seed.py 2>&1 | tee artifacts/logs/week05_seed_regression.log
~~~

当前最小产物：

~~~text
artifacts/manifests/seed_0001.json
docs/design/audio_blueprint_seed_mapping.md
tests/regression/test_blueprint_seed.py
scripts/build_blueprint_seed.py
artifacts/logs/week05_seed_regression.log
~~~

说明：

- 第一条命令用于从 media_probe 输出重建 seed_0001.json
- 第二条命令用于重跑 week05 的 seed regression
- 该入口服务于后续 generator audit / blueprint 扩展 / W6 联调
