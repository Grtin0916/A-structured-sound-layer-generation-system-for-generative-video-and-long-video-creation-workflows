# 2026-03-25 Week03 Compose Audit

## 1. Objective

本次审计用于确认 Week 03 周三的本地 Compose 栈是否已经满足最小可复验要求：

- API 服务能被 Compose 拉起
- 容器健康检查可通过
- 宿主机可通过 `curl` 命中 `/healthz`
- 路径挂载、镜像依赖与启动证据有明确记录
- 后续周四网络排障可以直接接上

---

## 2. Scope

本次仅覆盖单服务本地栈：

- Compose file: `deploy/compose/compose.local.yaml`
- Dockerfile: `docker/Dockerfile`
- API entry: `src/serving/app.py`
- Runtime target: `src.serving.app:app`

当前不覆盖：

- 多服务依赖编排
- 数据库 / 缓存
- 生产部署
- 负载均衡
- 正式压测

---

## 3. Key evidence

本次主要证据文件：

- `artifacts/logs/week03_compose_up.log`
- `artifacts/logs/week03_compose_ps.log`
- `artifacts/logs/week03_compose_api.log`
- `artifacts/logs/week03_curl_healthz.log`

---

## 4. What happened

### 4.1 First blocking issue: Compose path base was wrong

最初的失败不是 API 代码本身，而是 Compose 相对路径基准错了。

表现为：

- `build.context` 被错误解析到 `deploy/compose/`
- `dockerfile: docker/Dockerfile` 因此被拼成不存在的路径
- bind mount 也错误地只挂到了 `deploy/compose/`

修复动作：

- 将 `build.context` 改为 `../..`
- 将 bind mount 改为 `../..:/workspace`

### 4.2 Second blocking issue: image dependency was incomplete

路径修正后，容器开始能 build，但应用进程启动时报：

- `No module named uvicorn`

这说明问题已经从“路径错误”切换为“镜像依赖缺失”。

修复动作：

- 在 `docker/Dockerfile` 中补齐 FastAPI / Uvicorn 相关依赖
- 重新执行 `docker compose up --build`

### 4.3 Final state

修复后，Compose 栈进入稳定状态：

- 容器成功启动
- `docker compose ps` 显示 `healthy`
- 容器日志持续出现 `/healthz` 200
- 宿主机 `curl http://127.0.0.1:8000/healthz` 返回正常 JSON

这说明当前单服务 Compose 本地栈已经具备最小可复验能力。

---

## 5. Volume strategy decision

### Decision

本次选择 **bind mount**，而不是 named volume。

### Reason

当前阶段是本地开发与验证，不是正式持久化环境。  
今天更重要的是：

- 主机与容器同时看到源码
- 主机直接查看 `artifacts/logs/`
- 主机直接查看后续 `artifacts/wav/`
- 修改代码与配置后能快速复验

因此 bind mount 更符合当前目标。

### Current mapping

- host repo root -> `/workspace`

---

## 6. Healthcheck decision

### Decision

healthcheck 使用 Python 标准库 `urllib.request`，不额外依赖 `curl`。

### Reason

当前目标是先让 Compose 栈稳定、依赖最小、排障路径清晰。  
如果为了 healthcheck 额外引入更多系统依赖，会增加镜像复杂度，而当前并无必要。

---

## 7. Current conclusion

本次 Week 03 周三 Compose 主线已经完成以下关键动作：

- Compose 文件落位到 `deploy/compose/compose.local.yaml`
- 容器镜像构建链路可用
- API 容器可启动
- healthcheck 可通过
- 宿主机 `curl` 可命中 `/healthz`
- 关键日志证据已保留

因此，当前可以把周三主线判断为：

**Compose local stack: pass**

---

## 8. Remaining gap

当前仍然只是“单服务本地可复验栈”，还不是完整服务化终态。

仍未覆盖：

- 多服务依赖关系
- 网络排障证据（`ip / ss / lsof`）
- 容器内 / 宿主机差异定位
- 长期项目的 generator / separator 接入

这些内容应留到周四继续推进。

---

## 9. Next step for Thursday

周四第一件事不是继续猜，而是做网络排障证据链：

- `ip`
- `ss`
- `curl`
- `lsof`

目标是把“为什么通”“如果不通会卡在哪里”正式变成日志与复盘，而不是口头解释。
