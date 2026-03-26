# 2026-03-26 Week03 Network Debug

## 1. Objective

本次复盘用于确认 Week 03 周四阶段中，本地 API 服务的**主机网络链路**、**Compose 容器链路**与**视频探测工具链**是否已经具备最小可复验能力，并将排障过程从“现象描述”升级为“证据链判断”。

本次复盘重点回答以下问题：

- 宿主机当前的网络地址与默认路由是什么
- 8000 端口是否真的处于监听状态
- 服务“在容器里可用”和“在主机上可用”是否一致
- Compose 的健康检查是否能真实反映 API 可达状态
- `media_probe` 对视频样本的最小探测链是否已经跑通

---

## 2. Scope

本次仅覆盖以下对象：

- 宿主机网络与监听状态
- 本地 Compose 单服务 API 栈
- `/healthz` 的主机与容器内探活
- `tools/media_probe/probe_media.py` 的视频探测 smoke

本次**不覆盖**以下内容：

- 多服务网络编排
- 容器间依赖启动顺序优化
- 生产环境端口暴露策略
- 反向代理 / 网关
- 长期生成式音频工作流的完整服务化

---

## 3. Repository baseline

截至本次排障时，本地仓库状态满足以下前提：

- 工作区干净，可直接执行新的排障与记录动作
- 服务代码位于 `src/serving/`
- Compose 文件位于 `deploy/compose/compose.local.yaml`
- Docker 镜像定义位于 `docker/Dockerfile`
- 视频探测工具位于 `tools/media_probe/`
- Week 03 既有日志已经存在于 `artifacts/logs/`

这说明本次排障不是从零开始，而是在周二 `/predict` 接 ORT、周三 Compose 栈打通的基础上继续做“证据化验证”。

---

## 4. Evidence

本次排障涉及的关键证据文件包括：

- `artifacts/logs/week03_ip_addr.log`
- `artifacts/logs/week03_ip_route.log`
- `artifacts/logs/week03_ss.log`
- `artifacts/logs/week03_lsof.log`
- `artifacts/logs/week03_curl_healthz.log`
- `artifacts/logs/week03_compose_ps_thu.log`
- `artifacts/logs/week03_compose_api_thu.log`
- `artifacts/logs/week03_compose_exec_healthz.log`
- `artifacts/logs/week03_video_probe_smoke.json`

这些日志对应的工具职责分别为：

- `ip addr`：查看网络接口与地址信息，重点确认回环地址与宿主机可见地址
- `ip route`：查看默认路由与路由表，确认主机流量的默认出口
- `ss`：查看 TCP 监听状态与套接字占用，确认 `8000` 端口是否真的监听
- `lsof`：从进程侧确认哪个程序占用了监听端口

---

## 5. What was debugged

### 5.1 Compose path base issue was already resolved

在周三阶段，最初的 Compose 失败并不是 API 本身不可用，而是 Compose 文件中的相对路径基准错误。

当 Compose 文件位于 `deploy/compose/compose.local.yaml` 时，相对路径默认不是相对当前 shell 工作目录，而是相对 Compose 文件所在的基准路径 / 项目目录解析；`build.context` 与 `dockerfile` 的相对解析也不是同一层级。

当时的典型现象是：

- `build.context` 被解析到 `deploy/compose/`
- `dockerfile: docker/Dockerfile` 因此被错误拼接
- bind mount 也只挂到了 `deploy/compose/`

修复方式是：

- 将 `build.context` 调整为回到仓库根目录的相对路径
- 将 bind mount 对应改为仓库根目录到 `/workspace`

这一步完成后，Compose 能够正确找到仓库根目录、Dockerfile 和源码目录。

### 5.2 Container dependency issue was also resolved

路径问题解决后，第二个阻塞点变成了镜像依赖不完整。

当时容器日志出现：

- `No module named uvicorn`

这说明容器已经开始执行入口命令，但镜像内 Python 环境缺少运行 API 所需的依赖。

后续通过补齐 `docker/Dockerfile` 中的服务依赖，Compose 栈进入可启动状态。

---

## 6. Host network debug logic

本次周四的重点不再是修改服务逻辑，而是把“能访问 / 不能访问”的判断路径写成证据链。

本次判断逻辑固定为：

1. 先看宿主机接口地址是否正常
2. 再看默认路由是否存在
3. 再看 `8000` 端口是否进入 `LISTEN`
4. 再用 `curl` 从宿主机命中 `127.0.0.1:8000/healthz`
5. 若异常，再对照 `docker compose ps`
6. 再看 `docker compose logs api`
7. 必要时进入容器内执行 `urllib.request.urlopen(...)`

这个顺序的核心好处是：

- 不会把“服务没起”和“服务起了但主机访问不到”混为一谈
- 不会只看 Compose `healthy` 就误判链路完整
- 能区分“容器内回环可达”和“宿主机端口映射可达”是否一致

---

## 7. Healthcheck interpretation

本次 Compose 使用了容器内的 `/healthz` 探活作为健康检查。

需要特别强调的是：

**Compose `healthy` 只能说明容器内探活成功，不自动等价于宿主机访问一定成功。**

因此，本次周四要求同时保留：

- `docker compose ps`
- `docker compose logs api`
- 宿主机 `curl /healthz`
- 必要时容器内 `exec` 探活

这四类证据，而不是只保留其中之一。

---

## 8. Video probe smoke result

本次还对视频样本执行了：

- `tools/media_probe/probe_media.py`
- 输入文件：`data/interim/test_video_probe_smoke/335040.mp4`
- 输出文件：`artifacts/logs/week03_video_probe_smoke.json`

本次 probe 的实际结果说明：

- 视频文件存在，路径有效
- `ffprobe` 主探测链路成功
- 已成功提取：
  - format 信息
  - 文件大小
  - 码率
  - 时长
  - stream 类型与数量
  - 音频 codec / 采样率 / 声道
  - 视频 codec / 分辨率 / fps

从本次输出可见，关键字段已经完整出现，因此可以判断：

**Week 03 周四的视频探测 smoke 已通过。**

同时，这次输出也包含两条 warning，需要正确理解：

### 8.1 `PySoundFile failed. Trying audioread instead.`

这不是阻塞性错误，而是 `librosa` 在读取该媒体文件时，从 `PySoundFile` 回退到了 `audioread`。

结合当前样本为 `mp4 + aac`，这一回退是可以接受的。

### 8.2 `cv2 不可用：No module named 'cv2'`

这说明 OpenCV 视频补充探测分支未启用，因此没有产出 `cv2` 侧的额外视频字段。

但这并不影响本次 smoke 的最小目标，因为：

- `ffprobe` 已经给出了分辨率和帧率
- 当前阶段只要求完成最小视频元信息探测
- 不是要求同时打通 OpenCV 全量链路

因此，本次视频探测结果应判断为：

**pass with non-blocking warnings**

---

## 9. Current conclusion

截至本次周四排障，Week 03 当前状态可总结为：

### 9.1 Service / Compose side

- 周二：`/predict` 已真实挂接 ORT
- 周三：本地 Compose 栈已能构建并拉起
- 周四：开始将宿主机网络、端口监听、容器探活做成证据链

### 9.2 Network side

当前周四的主线目标不是继续扩 API，而是固定以下判断模板：

- 地址
- 路由
- 监听
- 主机探活
- 容器探活
- 容器内外差异

这条模板是后续任何服务化排障的基础。

### 9.3 Media probe side

视频样本探测已通过最小 smoke，说明仓库不再只有音频探测路径，而是开始具备真正面向 video→audio 输入的媒体元信息检查能力。

---

## 10. Remaining gaps

当前仍存在以下未完成点：

- 还没有把宿主机 `ip/route/ss/lsof` 的全部具体结论正式写死到同一版结论表里
- 还没有补 OpenCV 视频补充探测能力
- 还没有对“容器内通但宿主机不通”的异常场景做专门对照实验
- 还没有进入周五的 MusicGen / AudioCraft 原型接线

这些都是后续几天继续推进的合理方向，但不影响今天周四的最小网络排障目标达成。

---

## 11. Final verdict

本次 Week 03 周四主线可判断为：

**Network debug evidence chain: pass**  
**Video probe smoke: pass with warnings**

当前最重要的收获不是“命令跑通了”，而是已经把以下方法固定下来：

- 先看地址与路由
- 再看监听与占用进程
- 再做主机 `curl`
- 再对照 Compose 状态与容器日志
- 最后再看容器内探活
- 对媒体输入则优先跑 `probe_media`，先拿元信息，再谈后续生成链

---
