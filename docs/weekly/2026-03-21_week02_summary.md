# Week 2 Summary (2026-03-21)

## 1. 本周目标

本周的核心目标，不是继续堆新功能，而是把 Week 1 已经跑通的 `baseline_v1` 最小闭环，升级成一个 **可复现、可审计、可桥接、可延续** 的 v0.2 工程底座。

相比 Week 1，本周新增的重点不再是“再跑一遍已有流程”，而是围绕以下四条主线补齐工程可信度与后续扩展接口：

- 多样本 parity / benchmark：把单样本成功升级成小规模稳定性证据
- Docker 重现与 host 对比：把“我这台机器能跑”升级成“标准化环境能复跑”
- GitHub Actions smoke：把“手工可跑”升级成“仓库持续可跑”
- 长期项目桥接：把 `media_probe`、音频生成栈角色边界与当前 baseline 的关系落进仓库，而不是只停留在想法层

本周结束时，希望仓库能够回答三个问题：

1. 当前最小主线到底是什么？
2. 这条主线是否已经具备基本复现性与证据链？
3. 长期项目的生成链、分离链、媒体 I/O 工具链，现在在仓库里各自处于什么位置？

---

## 2. 本周完成项

截至 2026-03-21，Week 2 已经完成以下关键资产与验证动作。

### 2.1 主线：`baseline_v1` v0.2 工程化补强

在保留 Week 1 最小闭环的前提下，本周完成了以下新增能力：

- host 环境复跑与版本冻结
- 多样本 PT / ORT parity 汇总
- 多样本 ORT CPU benchmark 汇总
- 导出稳定性 / shape 风险审计
- Docker 内 CPU 路径复跑
- host vs Docker 的 benchmark 对比文档
- GitHub Actions CPU smoke workflow
- `baseline_v1` 结果卡 / 实验卡沉淀
- README、ADR、benchmark、postmortem 等文档层继续成形

这意味着当前仓库已经不再只是“单次成功的最小 demo”，而是开始具备：

- 可重复执行的最小主线
- 可展示的结构化证据
- 可继续承接 Week 3–6 工程工作的路径骨架

### 2.2 工程侧新增资产

本周新增或明确纳入主线证据的资产包括：

- `scripts/bench_ort_dataset.py`
- `scripts/audit_export_stability.py`
- `.github/workflows/smoke.yml`
- `docs/benchmarks/ort_cpu_multi_sample.md`
- `docs/benchmarks/ort_cpu_multi_sample_docker.md`
- `docs/benchmarks/ort_docker_cpu_compare.md`
- `docs/postmortems/2026-03-16_week02_repro_audit.md`
- `docs/postmortems/2026-03-18_week02_shape_audit.md`

与 Week 1 相比，主线脚本已经从“单点 demo”升级到：

- 可批量统计
- 可审计 shape 风险
- 可在 CI 中保留最低复现约束

### 2.3 长期项目桥接资产

本周还完成了第一块真实的长期项目工具层基础设施：

- `tools/media_probe/probe_media.py`
- `tools/media_probe/README.md`
- `docs/design/audio_generation_stack_design.md`
- `docs/adr/0002-week02-bridge.md`（在已有 ADR 基础上继续承接桥接决策）

当前这部分工作的定位很明确：

- `media_probe` 是 **中立工具层**
- AudioCraft / AudioLDM / Spleeter 当前是 **延后接入的候选模块**
- 它们目前不进入 `baseline_v1` 的主 smoke runtime
- 但它们已经在仓库叙事与结构层面获得了明确落位

这说明“分层生成式视频音频系统”第一次在当前仓库里出现了真实接口边界，而不是纯概念。

---

## 3. 本周 DoD 对照

### 3.1 M2+（主线）DoD

目标：

1. 在更干净环境下复跑 demo
2. 实现至少 10 个样本的 PT / ORT parity 汇总
3. 实现至少 10 个样本的 ORT benchmark 汇总
4. 明确固定 shape 合同，或完成 2–3 组 shape smoke test
5. 输出模型卡、实验卡、周报

当前判断：

- 1）已完成
- 2）已完成
- 3）已完成
- 4）已完成
- 5）部分完成，周报与结果卡今日收口

结论：**主线 DoD 基本达成。**

### 3.2 B2+（容器 / CI）DoD

目标：

1. Dockerfile 可构建
2. 容器内能跑 demo 与 bench
3. 输出主机 vs Docker 延迟对比表
4. 补齐 `.dockerignore`、固定版本说明
5. 加入 GitHub Actions CPU smoke workflow

当前判断：

- Docker 重现链已形成
- host vs Docker 对比文档已形成
- `.github/workflows/smoke.yml` 已存在
- 版本冻结与运行记录已保留
- Docker 构建定义当前已可用，但命名仍有进一步统一空间

结论：**容器 / CI DoD 基本达成，但结构命名仍有一处待收口。**

### 3.3 C2+（长期项目桥接）DoD

目标：

1. 新增 `tools/media_probe` 或等价脚本，输出媒体属性
2. 完成 AudioCraft 与 AudioLDM 的职责落位
3. 至少对其中一个库完成 smoke run 或环境检查结论
4. 形成一份 ADR，说明当前仓库为什么是长期大项目的工程底座

当前判断：

- 1）已完成
- 2）已完成
- 3）当前未计入“已完成”，只保留为后续可选补充入口
- 4）已完成

结论：**桥接 DoD 达成 3/4。当前最缺的不是架构定义，而是一条可追溯的生成链 smoke 记录。**

### 3.4 A2（DSA）DoD

本周 DSA 不作为本总结的主资产展开；它保持“保手感、不反客为主”的定位。项目主线优先级高于刷题数量。

---

## 4. 本周最小复现入口

### 4.1 当前最短主线

当前最小、最清晰、最适合作为 Week 2 主线入口的命令仍然是：

`bash scripts/run_demo.sh demo`

这条链路当前覆盖：

1. ONNX 导出
2. PyTorch / ONNX Runtime 对齐校验
3. ONNX Runtime 单次推理
4. ORT CPU benchmark

它的意义不是“功能很多”，而是：

- 入口最短
- 口径最稳
- Host、Docker、CI 都可以围绕它做最小约束

### 4.2 本周与 Week 1 的差异

Week 1 里，这条命令更多是“闭环打通”的证据。  
到了 Week 2，它已经开始承担“主 smoke spine”的角色。

也就是说，当前仓库主线不再只是：

- 有模型
- 能导出
- 能跑一次

而是升级成：

- 有固定主入口
- 有多样本稳定性证据
- 有 postmortem
- 有 benchmark 文档
- 有 Docker 复跑
- 有 CI smoke

---

## 5. 本周关键输出文件

### 5.1 结果与日志侧

- `artifacts/logs/parity_multi_sample.csv`
- `artifacts/logs/bench_ort_multi_sample.csv`
- `artifacts/logs/bench_ort_multi_sample_docker.csv`
- `artifacts/logs/week02_repro_host.log`
- `artifacts/logs/week02_repro_docker.log`
- `artifacts/logs/week02_shape_audit.csv`
- `artifacts/logs/week02_versions.txt`

### 5.2 benchmark 报告侧

- `docs/benchmarks/ort_cpu_baseline.md`
- `docs/benchmarks/ort_cpu_multi_sample.md`
- `docs/benchmarks/ort_cpu_multi_sample_docker.md`
- `docs/benchmarks/ort_docker_cpu_compare.md`

### 5.3 审计 / 复盘侧

- `docs/postmortems/2026-03-16_week02_repro_audit.md`
- `docs/postmortems/2026-03-18_week02_shape_audit.md`

### 5.4 桥接 / 设计侧

- `docs/adr/0002-week02-bridge.md`
- `docs/design/audio_generation_stack_design.md`
- `tools/media_probe/probe_media.py`
- `tools/media_probe/README.md`

### 5.5 仓库持续可跑侧

- `.github/workflows/smoke.yml`

---

## 6. `baseline_v1` 当前状态：从最小闭环到 v0.2 底座

### 6.1 当前主线身份

本周结束后，`baseline_v1` 的身份已经发生了变化。

它不再只是一个“为了练手临时训练的小模型”，而是当前仓库中唯一明确受支持、并且已经具备最小复现脊柱的主线基线。它承担的是：

- 当前仓库主 smoke 路径的锚点
- ONNX / ORT 导出与推理链的参照物
- README、周报、结果卡、benchmark、postmortem 的统一口径中心

### 6.2 本周对 `baseline_v1` 的实际提升

本周并没有切新模型、重做训练，也没有追逐更花哨的生成能力。  
本周真正完成的是对 `baseline_v1` 的“工程可信度补强”：

- 从单样本对齐变成多样本对齐
- 从单次 benchmark 变成小样本统计
- 从 host 单机验证扩展到 Docker 对照
- 从人工复现扩展到 CI smoke
- 从零散记录升级为结构化文档体系

### 6.3 当前意义

这一步的价值，不在于“模型更强”，而在于仓库第一次拥有了一个：

- 可复跑
- 可比较
- 可讲述
- 可挂接其他模块

的工程主脊柱。

---

## 7. 长期项目桥接进展

### 7.1 `media_probe` 的正确定位

本周最重要的新能力，不是某个大模型终于跑起来了，而是仓库里第一次出现了 **真实媒体 I/O 基础设施**。

`tools/media_probe/` 当前的正确定位是：

- 为后续视频 / 音频输入做属性探测
- 输出时长、采样率、通道数、fps、分辨率、流信息等元数据
- 为未来“视频输入 → 音频抽取 → 生成 → 对齐 → 评估”整条链提供输入合同

它是 **工具层**，不是当前 baseline 主推理链的一部分。

### 7.2 AudioCraft / AudioLDM / Spleeter 当前落位

截至本周结束，三者的仓库角色应明确理解为：

- AudioCraft：未来主生成底座候选
- AudioLDM：未来环境音 / Foley / 备选生成链候选
- Spleeter：未来分离工具位候选

它们当前都 **不进入** `baseline_v1` 的主 smoke workflow。  
这样做不是保守，而是为了避免在主线尚未稳定时，把生成链、分离链、媒体探针混成一个难以调试的大锅。

### 7.3 本周桥接的真实完成度

本周桥接工作已经完成的是：

- 角色边界写清
- 工具层脚本落地
- 设计文档落地
- ADR 口径成形

本周尚未计入完成的是：

- AudioCraft / AudioLDM / Spleeter 三者之一的明确 smoke 记录

这部分属于 Week 2 可选补充，不应在周报中伪装成已经完成。

---

## 8. 本周已知问题与技术债

### 8.1 导出图的 shape 泛化风险仍未彻底消失

本周虽然已经做了 shape 审计，但当前主线仍然主要建立在固定 shape 合同基础上。  
这意味着：

- 当前链路适合做稳定基线
- 但还不应被误解成“任意长度 / 任意形状都已稳健支持”

### 8.2 当前 benchmark 仍然是 v0.2 级别证据，不是完整生产画像

目前的 benchmark 更适合作为：

- Week 2 的回归对照
- host vs Docker 的初步性能参照
- 后续 CI / smoke 的稳定性门槛参考

但它还不是完整生产部署画像，原因包括：

- 统计范围有限
- 当前 provider 主要聚焦 CPUExecutionProvider
- 指标尚未扩展到更完整的服务级性能维度

### 8.3 仓库结构仍有一处收口动作待完成

当前 Docker 构建定义与日志放置，仍有继续收口空间：

- Docker 构建定义当前仍是 `docker/Dockerfile`
- 部分 Docker 安装 / 验证日志仍位于 `docker/` 子目录下

从长期目录规范看，后续宜进一步收敛为：

- Docker 构建定义统一标准入口
- 新日志统一归 `artifacts/logs/`

### 8.4 桥接 smoke 记录仍缺一条“工程事实”

当前长期项目桥接已经完成“角色定义”，但还缺一条真正可追溯的生成链或分离链 smoke 记录。  
这不是本周失败点，但它是下周最自然的前推入口之一。

---

## 9. 与仓库结构规则的对应关系

为避免后续继续把文档、日志、实验卡、设计说明混写，本周形成如下更清晰的目录职责分工：

### 9.1 `docs/weekly/`

放一整周总结。  
本文件即为本周唯一 weekly summary 主文件。

### 9.2 `results/`

放模型卡 / 实验卡 / 结果说明。  
`baseline_v1` 的主结果说明应继续收敛在 `results/baseline_v1.md`。

### 9.3 `docs/benchmarks/`

放 benchmark 的可读 Markdown 总结。  
原始 CSV 与日志不应继续放到文档目录。

### 9.4 `docs/postmortems/`

放复盘、审计、问题记录。  
Week 2 的 repro audit 与 shape audit 已经按这个职责落位。

### 9.5 `docs/design/`

放长期设计文档与模块角色说明。  
音频生成栈角色说明归入该目录是合理的。

### 9.6 `artifacts/logs/`

放运行日志、CSV、简短报告型记录。  
后续所有新产生的日志型产物都应优先归到这里。

### 9.7 `tools/`

放与主训练 / 导出 / 推理流程解耦的工具型辅助程序。  
`media_probe` 继续保持工具层，不要误塞进 `scripts/` 主线脚本组。

---

## 10. Week 2 当前结论

Week 2 最重要的结论不是“仓库又多了几个文件”，而是：

**当前项目已经从 Week 1 的“最小工程闭环”升级为一个具备初步复现性、审计性、桥接性与延续性的 v0.2 工程底座。**

具体来说：

- `baseline_v1` 已经不再只是一次性成功样例，而是当前主 smoke 基线
- 多样本 parity / benchmark 已经补上小规模稳定性证据
- Docker 与 host 对照让环境差异第一次可被记录与比较
- `.github/workflows/smoke.yml` 让最小复现从“人工动作”开始过渡到“持续约束”
- `media_probe` 与音频生成栈桥接文档，让长期项目第一次以模块边界的形式进入仓库
- 文档层开始形成 weekly / benchmark / postmortem / ADR / results 的分工秩序

这说明当前仓库已经不只是“边学边存文件”的目录集合，而是开始拥有了真正的工程主线与可讲述结构。

---

## 11. 下周第一步建议

下周不该做的，是重新起盘。  
下周最自然、最便宜、最有延续性的进入点有两个：

### 方案 A：补一条生成链 / 分离链 smoke 记录

优先级最高。  
原因：当前桥接已经有角色定义，但还缺一个可追溯工程事实。  
最小动作可以是以下三选一：

- AudioCraft 的 import / 版本检查
- AudioLDM 的最小环境检查
- Spleeter 的 Docker help / 最小命令记录

### 方案 B：继续把主线做成更稳的“服务前底座”

如果下周仍先稳主线，可继续推进：

- 更严格的输入校验
- 更统一的 Docker 入口
- 更清晰的结果卡收口
- 更稳的 benchmark / smoke 回归口径

### 当前建议

优先做 **方案 A**。  
原因很现实：现在主线已经够稳，最值得补的不是再写一份相似文档，而是给长期项目桥接补上一条真正能查、能复述、能继续接的 smoke 证据。

---

## 12. 一句话版总结

Week 2 的关键进展，不是把 `baseline_v1` 做得更花，而是把它从“能跑通”升级成“能复现、能审计、能对比、能桥接”的 v0.2 工程底座；当前最值得继续推进的，不是重开新坑，而是补上一条生成链 / 分离链的最小 smoke 事实，把长期项目真正接到这条主线上。