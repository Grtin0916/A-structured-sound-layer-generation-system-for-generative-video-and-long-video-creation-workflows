# Scorecard v1

## 1. Purpose
本文件用于把现有 Week06 证据收束为可比较、可扩展的 scorecard 入口。
当前版本不补新实验，只整理已有 evidence。

## 2. Scope
当前仅纳入已在仓库中留痕的三类证据：
- functional eval
- performance eval
- failure regression eval

## 3. Common fields
- eval_group: functional | performance | failure_regression
- subject: model / contract / mapping object
- case_id: case identifier
- input_type: text_prompt | mel | media_probe | api_request | other
- status: pass | fail | uncertain
- failure_tag: none | env | dependency | shape | io | timeout | quality | other
- metric_key: primary metric name
- metric_value: primary metric value or NA
- evidence_path: repo-relative path
- note: short interpretation

## 4. Decision rules

### 4.1 Functional eval
- pass: 能完成预期流程，且有可引用日志/产物
- fail: 流程失败或输出不可用
- uncertain: 有输出但结论不足

### 4.2 Performance eval
- 仅记录已有 benchmark 结果
- 当前不跨环境强行对比
- 没有统一口径时写 NA，不编数字

### 4.3 Failure regression eval
- 记录已知失败类型、触发条件、是否已规避
- 先服务后续回归，不追求一次写全

## 5. Current evidence mapping
| eval_group | current source |
|---|---|
| functional | `artifacts/logs/generator_audit_001.log`, `artifacts/manifests/seed_0001.json` |
| performance | `docs/benchmarks/ort_cpu_baseline.md`, `artifacts/logs/bench_ort_cpu.csv` |
| failure_regression | `docs/postmortems/2026-04-17_week06_generator_audit.md` |

## 6. Out of scope for v1
- `musicgen-medium` / `melody` / larger models
- 正式 observability 结论
- 严格统一的 generator benchmark methodology
