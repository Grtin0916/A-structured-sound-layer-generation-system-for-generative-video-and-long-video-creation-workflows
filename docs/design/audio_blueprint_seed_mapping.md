# audio_blueprint_seed_mapping

## 1. Purpose

本文档定义 Week05 第一条 blueprint seed JSON 的最小映射规则。

目标不是一次性完成完整 audio blueprint 编排，而是先把现有
`media_probe` 输出稳定映射为一条可落盘、可回归、可继续扩展的 seed JSON。

当前 seed 的职责是：

- 固化输入视频的最小元信息
- 给后续 blueprint / generator audit / serving bridge 提供统一入口
- 为 regression test 提供稳定样本
- 先解决“字段和命名统一”，再扩展复杂 timeline

---

## 2. Current input

当前使用的输入文件：

~~~text
artifacts/logs/week03_video_probe_smoke.json
~~~

当前样例对应的视频路径：

~~~text
data/interim/test_video_probe_smoke/335040.mp4
~~~

---

## 3. Output path decision

注意：`audio_blueprint_spec.md` 当前示例路径仍使用：

~~~text
artifacts/blueprints/<blueprint_id>.json
~~~

但本周 Week05 的执行目标已经明确要求：

~~~text
artifacts/manifests/seed_0001.json
~~~

因此本周先采用以下落盘口径：

~~~text
artifacts/manifests/seed_0001.json
~~~

说明：

1. 本周先把 seed 视为“最小可回归 manifest”
2. 后续若 blueprint 与 manifest 目录需要统一，再单独做一次路径规范收口
3. 当前不为了统一命名而阻塞 Week05 首条 seed 落盘

---

## 4. Minimal seed schema for Week05

当前第一条 seed 只保留最小必需字段：

- `blueprint_version`
- `blueprint_id`
- `task_type`
- `source_video`
- `media_probe`
- `timeline`
- `output_stems`
- `metadata`

其中：

- `timeline` 当前允许为空列表
- `output_stems` 当前允许为空列表
- 当前重点是把输入视频与 probe 结果稳定挂进统一结构

---

## 5. Field mapping

| seed field | source | rule |
| --- | --- | --- |
| `blueprint_version` | fixed | `"v0.1"` |
| `blueprint_id` | fixed | `"seed_0001"` |
| `task_type` | fixed | `"video_audio_smoke"` |
| `source_video.path` | `path` | 直接拷贝 |
| `source_video.duration_sec` | `duration_sec_ffprobe` | 直接拷贝 |
| `source_video.has_audio` | `stream_types` | 是否包含 `"audio"` |
| `source_video.has_video` | `stream_types` | 是否包含 `"video"` |
| `media_probe.log_ref` | fixed | `"artifacts/logs/week03_video_probe_smoke.json"` |
| `media_probe.format_name` | `format_name` | 直接拷贝 |
| `media_probe.size_bytes` | `size_bytes` | 直接拷贝 |
| `media_probe.bit_rate` | `bit_rate` | 直接拷贝 |
| `media_probe.audio.codec` | `audio_codec` | 直接拷贝 |
| `media_probe.audio.sample_rate` | `audio_sr_ffprobe` | 直接拷贝 |
| `media_probe.audio.channels` | `audio_channels_ffprobe` | 直接拷贝 |
| `media_probe.video.codec` | `video_codec` | 直接拷贝 |
| `media_probe.video.width` | `width_ffprobe` | 直接拷贝 |
| `media_probe.video.height` | `height_ffprobe` | 直接拷贝 |
| `media_probe.video.fps` | `video_fps_ffprobe` | 直接拷贝 |
| `timeline` | fixed | `[]` |
| `output_stems` | fixed | `[]` |
| `metadata.segment_id` | fixed | `"seg_0001_full"` |
| `metadata.generator_name` | fixed | `"seed_builder_v0"` |
| `metadata.output_artifact_dir` | fixed | `"artifacts/wav/seed_0001"` |
| `metadata.source_stage` | fixed | `"week05_seed_mapping"` |

---

## 6. Non-goals for this seed

当前这条 seed 暂不实现：

- 复杂事件表自动生成
- ambience / foley / bgm 的细粒度时间轴
- 多段 segment 切分
- 多 blueprint merge / diff
- 在线 serving 直接消费完整 timeline

这些内容留到 blueprint 扩展阶段处理。

---

## 7. Acceptance criteria

本周这条 seed 达标条件：

1. `artifacts/manifests/seed_0001.json` 成功落盘
2. JSON 可被 Python 正常读取
3. 关键字段齐全
4. 输入视频路径与 probe 日志路径正确
5. `timeline` 与 `output_stems` 即使为空，也必须显式存在
6. 回归测试可检查字段存在性与基础值合法性

