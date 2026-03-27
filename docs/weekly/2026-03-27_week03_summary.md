# Week 03 Summary (2026-03-27)

## 1. 本周目标

Week 03 的核心目标，不再只是维持最小训练 / 导出 / 推理闭环，而是把仓库从“可复现实验基线”推进到“可调用服务 + 可编排运行 + 可排障验证 + 长期项目点火”的状态。

本周主线分为四段：

1. 服务前接口与最小 API 骨架
2. `/predict` 接 ORT 与 API smoke
3. 本地 Compose 栈与网络证据链
4. 长期项目阶段 1 点火：MusicGen / AudioCraft smoke 与 audio blueprint 规范

换句话说，本周不是单点脚本堆砌，而是把仓库推向“工程主线 + 长期路线并存”的结构。

---

## 2. 本周完成概览

截至 2026-03-27，Week 03 已完成以下关键动作：

- 完成服务接口合同文档 `docs/design/serving_api_spec.md`
- 完成 `src/serving/schemas.py`
- 完成 `src/serving/app.py`
- 完成 `configs/serving/service_api.yaml`
- `/healthz`、`/metadata`、`/predict` 路由已落位
- `/predict` 已从 placeholder 升级为真实 ORT 推理入口
- 已产出 API smoke 日志与 artifact manifest
- 已补充 `tests/test_api_smoke.py`
- 已完成 Pydantic warning 清理
- 已完成本地 Compose 栈 `deploy/compose/compose.local.yaml`
- 已完成 Compose healthcheck、宿主机 `/healthz` 与容器内探活验证
- 已完成网络证据链日志：
  - `ip addr`
  - `ip route`
  - `ss`
  - `lsof`
- 已完成视频样本 `media_probe` smoke
- 已完成第一条真实生成音频样例：
  - `artifacts/wav/musicgen_smoke_001.wav`
- 已完成 `audio_blueprint_spec v0.1` 设计文档
- 已持续补充 `algo/week03.md`，形成栈 / 队列 / 单调栈模板周闭环

这意味着 Week 03 已经把仓库从“模型闭环”推进到“服务、编排、排障、协议、原型”多个层次。

---

## 3. 本周分日完成情况

## 3.1 周一：服务前接口骨架落位

周一的重点是先冻结合同，而不是上来写一坨服务逻辑。

完成项：

- 审核远端 GitHub 仓库实际目录结构
- 明确 canonical 路径：
  - `src/serving/`
  - `configs/serving/`
  - `docs/design/`
- 编写 `docs/design/serving_api_spec.md`
- 编写 `src/serving/schemas.py`
- 编写 `src/serving/app.py`
- 编写 `configs/serving/service_api.yaml`
- 验证：
  - `/healthz`
  - `/metadata`
  - `/predict` placeholder
  - `/docs`

周一的关键结果不是“服务很强”，而是：

**服务边界和字段语义先被钉死了。**

---

## 3.2 周二：`/predict` 接 ORT 与 API smoke

周二的重点是把 `/predict` 从占位接口推进成真实可调用的推理入口。

完成项：

- `src/serving/app.py` 改为读取 `service_api.yaml`
- 接入 ONNX Runtime `InferenceSession`
- 支持本地 `.npy` 输入
- 支持 fixed-shape 合同校验
- 成功时输出 `.npy` 预测产物
- 失败时返回结构化错误
- 生成并保存：
  - `artifacts/logs/week03_api_smoke.log`
  - `artifacts/logs/week03_artifact_manifest.json`
- 编写 `tests/test_api_smoke.py`
- 修复 `pytest` 导入路径问题
- 补齐 `pytest.ini`
- 处理 Pydantic v2 warning

周二的关键结果是：

**服务不再只是能起，而是已经具备最小可复验推理能力。**

---

## 3.3 周三：本地 Compose 栈与 Compose 审计

周三的重点是把 API 服务放进 Compose 环境中，并完成最小本地编排闭环。

完成项：

- 创建 `deploy/compose/compose.local.yaml`
- 复用 `docker/Dockerfile`
- 修复 Compose 相对路径基准问题
- 修复镜像缺少 `uvicorn / fastapi` 依赖问题
- 完成：
  - `docker compose up --build`
  - `docker compose ps`
  - `docker compose logs`
  - 宿主机 `curl /healthz`
- 生成：
  - `artifacts/logs/week03_compose_up.log`
  - `artifacts/logs/week03_compose_ps.log`
  - `artifacts/logs/week03_compose_api.log`
  - `artifacts/logs/week03_curl_healthz.log`
- 编写：
  - `docs/postmortems/2026-03-25_week03_compose_audit.md`

周三的关键结果是：

**仓库第一次具备了单服务本地编排能力。**

---

## 3.4 周四：网络排障证据链与视频 probe smoke

周四的重点不再是改服务功能，而是把“为什么能通 / 为什么不通”固化成证据链。

完成项：

- 生成：
  - `artifacts/logs/week03_ip_addr.log`
  - `artifacts/logs/week03_ip_route.log`
  - `artifacts/logs/week03_ss.log`
  - `artifacts/logs/week03_lsof.log`
- 对照验证：
  - 宿主机 `curl /healthz`
  - `docker compose ps`
  - `docker compose logs api`
  - `docker compose exec api ...`
- 编写：
  - `docs/postmortems/2026-03-26_week03_network_debug.md`
- 对视频样本执行 `media_probe`
- 生成：
  - `artifacts/logs/week03_video_probe_smoke.json`

周四的关键结果是：

**服务访问问题不再只靠猜，而是有了固定的排障模板。**

---

## 3.5 周五：长期项目阶段 1 点火

周五的重点是把长期 video→audio 项目从“工具位”推进到“真实生成样例 + 协议文档”。

完成项：

- 核查 AudioCraft / MusicGen 环境
- 记录：
  - `artifacts/logs/week03_audiocraft_smoke.log`
- 成功生成：
  - `artifacts/wav/musicgen_smoke_001.wav`
- 用 `ffprobe` 验证生成音频：
  - 时长：8 秒
  - 采样率：32000 Hz
  - 单声道
  - PCM WAV
- 编写：
  - `docs/design/audio_blueprint_spec.md`
- 更新长期栈设计文档：
  - 明确 MusicGen / AudioCraft
  - 明确 AudioLDM
  - 明确 Spleeter 的角色边界

周五的关键结果是：

**仓库第一次拥有了真实生成音频产物，而不是只有桥接讨论。**

---

## 4. 本周关键产物

### 4.1 服务与配置

- `docs/design/serving_api_spec.md`
- `src/serving/app.py`
- `src/serving/schemas.py`
- `configs/serving/service_api.yaml`

### 4.2 API 证据

- `artifacts/logs/week03_api_smoke.log`
- `artifacts/logs/week03_artifact_manifest.json`
- `tests/test_api_smoke.py`
- `pytest.ini`

### 4.3 Compose 与网络证据

- `deploy/compose/compose.local.yaml`
- `artifacts/logs/week03_compose_up.log`
- `artifacts/logs/week03_compose_ps.log`
- `artifacts/logs/week03_compose_api.log`
- `artifacts/logs/week03_curl_healthz.log`
- `artifacts/logs/week03_ip_addr.log`
- `artifacts/logs/week03_ip_route.log`
- `artifacts/logs/week03_ss.log`
- `artifacts/logs/week03_lsof.log`

### 4.4 媒体探测与长期项目点火

- `artifacts/logs/week03_video_probe_smoke.json`
- `artifacts/logs/week03_audiocraft_smoke.log`
- `artifacts/wav/musicgen_smoke_001.wav`
- `docs/design/audio_blueprint_spec.md`

### 4.5 复盘与周记

- `docs/postmortems/2026-03-25_week03_compose_audit.md`
- `docs/postmortems/2026-03-26_week03_network_debug.md`
- `docs/weekly/2026-03-27_week03_summary.md`

---

## 5. 本周最关键的技术结论

### 5.1 服务接口必须先冻结合同再实现

本周服务线证明了一件事：

如果不先写 `serving_api_spec`，后面 `schemas.py`、`app.py`、`service_api.yaml` 很容易边写边漂移。

因此，当前仓库已经形成一个明确经验：

**先冻结字段语义，再写 API 代码。**

### 5.2 `/predict` 可跑，比“接口长得像样”更重要

周一的 `/predict` 只是占位，周二把它接成真实 ORT 推理后，整个服务线才第一次具备工程意义。

这也是本周服务主线的核心拐点。

### 5.3 Compose 的最大坑是路径和依赖，不是 YAML 本身

本周 Compose 阶段的两个关键问题分别是：

- 相对路径基准错了
- 镜像里缺 `uvicorn / fastapi`

这说明 Compose 最容易翻车的地方，不是“语法多复杂”，而是：

- 项目目录基准
- `build.context`
- `dockerfile`
- bind mount
- 镜像依赖层

### 5.4 `healthy` 不等于宿主机必然可达

周四的网络排障固定了一个很重要的认知：

**容器内探活成功，并不自动等于宿主机链路完整。**

因此必须同时保留：

- `docker compose ps`
- `docker compose logs`
- 宿主机 `curl`
- 必要时容器内探活

### 5.5 `media_probe` 已经从音频辅助工具升级为 video→audio 输入入口

本周视频样本 probe 说明：

- 输入视频不再是黑盒
- 关键元信息已经可以被提取
- 后续 blueprint 和生成链有了统一上游来源

### 5.6 MusicGen smoke 是本周最重要的阶段性点火

`musicgen_smoke_001.wav` 的意义不在于音质有多强，而在于：

- 它是仓库第一条真实生成音频
- 它把长期项目从“桥接讨论”推进到了“样例落盘”
- 它为 `audio_blueprint_spec` 提供了真实输出对象

---

## 6. 本周遇到的主要问题

### 6.1 Pydantic v2 warning

问题：

- 旧 `Config`
- `min_items / max_items`

处理：

- 改为 `ConfigDict`
- 改为 `min_length / max_length`

结果：

- `pytest -q -W error::DeprecationWarning tests/test_api_smoke.py` 通过

### 6.2 Compose 路径基准错误

问题：

- 相对路径按 Compose 文件基准解析
- `dockerfile` 被拼到错误目录

处理：

- 调整 `build.context`
- 调整 bind mount 路径

结果：

- Compose 能正确 build 和挂载仓库根目录

### 6.3 镜像依赖不完整

问题：

- `No module named uvicorn`

处理：

- 补齐 `fastapi / uvicorn` 等依赖

结果：

- API 容器可正常启动

### 6.4 `media_probe` warning

问题：

- `PySoundFile failed. Trying audioread instead.`
- `cv2 not available`

判断：

- 非阻塞 warning
- `ffprobe` 主链已成功
- 当前 smoke 有效

### 6.5 AudioCraft / MusicGen 导入链动态库问题

问题：

- `av` / `openvino` / `libstdc++` 动态库冲突

处理：

- 调整运行环境优先使用 conda 库链
- 重新执行导入与生成 smoke

结果：

- MusicGen small 成功生成 8 秒 wav

### 6.6 PyTorch `weight_norm` deprecation warning

问题：

- 上游依赖仍调用旧 `weight_norm`

判断：

- 非阻塞
- 不影响本次生成

结果：

- 记入 smoke 日志，不作为本周阻塞项

---

## 7. 本周算法部分总结

本周算法围绕栈 / 队列 / 单调栈做了系统性补齐。

已覆盖题目包括：

- 20 - Valid Parentheses
- 232 - Implement Queue using Stacks
- 155 - Min Stack
- 225 - Implement Stack using Queues
- 933 - Number of Recent Calls
- 2390 - Removing Stars From a String
- 394 - Decode String
- 739 - Daily Temperatures

到本周结束，已经形成以下模板组：

### 7.1 普通栈
- 20
- 155
- 2390
- 394

### 7.2 队列
- 232
- 225
- 933

### 7.3 单调栈
- 739

这些模板已经足够支撑下一阶段继续扩到：

- 496
- 1047
- 更完整的 next greater element 系列

---

## 8. 本周最大的收获

本周最大的收获不是某一个脚本或某一条命令，而是：

**仓库开始同时拥有“稳定工程主线”和“长期项目生长点”。**

具体来说：

- 主线：
  - API
  - ORT
  - Compose
  - 网络排障
- 生长点：
  - media_probe
  - MusicGen smoke
  - audio blueprint
  - 后续 AudioLDM / Spleeter 角色落位

这意味着仓库已经不再只是“做过几个实验”，而是开始具备“每周持续推进”的工程形态。

---

## 9. 本周仍然存在的缺口

虽然 Week 03 已经明显推进，但当前仍有以下缺口：

- `audio_blueprint` 还只是 v0.1，未落到真实 JSON 样例文件
- AudioLDM / Spleeter 还只完成角色落位，未跑最小 smoke
- 生成系统仍然是单次样例，不是完整工作流
- 还没有把 blueprint 读入 serving / workflow 层
- 还没有进入“视频内容 -> 事件表自动生成”
- 还没有建立正式评测集与指标表

这些都是合理缺口，不是本周失败，而是 Week 04 继续推进的入口。

---

## 10. Week 04 suggested next step

下周建议优先按以下顺序推进：

1. 生成第一份真实 `blueprint.json`
2. 让 `musicgen_smoke_001.wav` 与 blueprint 建立明确引用关系
3. 补 `audio_generation_stack_design.md` 的 AudioLDM / Spleeter 角色段
4. 视环境情况决定：
   - 跑一次 AudioLDM 最小 smoke
   - 或跑一次 Spleeter 最小 separation smoke
5. 设计 blueprint 校验脚本
6. 继续补充 `algo/week03.md` / `week04.md` 的模板题

---

## 11. Final verdict

本周 Week 03 可判断为：

**API serving path: pass**  
**ORT predict path: pass**  
**Compose local stack: pass**  
**Network debug evidence chain: pass**  
**Video probe smoke: pass with warnings**  
**MusicGen small smoke: pass with non-blocking warning**

整体判断：

**Week 03 封板成功。**

它完成了从“最小工程基线”向“服务化 + 编排 + 证据链 + 长期项目点火”的一次关键推进。