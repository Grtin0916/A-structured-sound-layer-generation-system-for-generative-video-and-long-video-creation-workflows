# audio_blueprint_spec

## 1. Purpose

本文档定义当前仓库中“结构化声音层生成系统”的第一版 **audio blueprint** 规范。

它的目标不是立刻覆盖完整的生成式视频音频工作流，而是先为后续几个环节建立统一输入/输出协议：

- 视频输入探测
- 音频事件规划
- 分层生成
- stems 输出
- 元信息记录
- 评测与复现

这份规范服务于当前阶段的最小原型，也服务于后续更复杂的长视频音频生成系统。

---

## 2. Design goals

本规范当前阶段的设计目标是：

1. 让“视频输入 -> 音频规划 -> stems 输出”有统一结构
2. 让 `media_probe` 的结果能够直接接入 blueprint
3. 让 MusicGen / AudioLDM / Spleeter 等模块有明确的数据交换边界
4. 让后续评测、排障、复现有稳定字段可查
5. 让 blueprint 本身既能给程序读，也能给人看

简单说：

- 不要只有模型，没有协议
- 不要只有想法，没有字段
- 不要让后面每次都重新定义路径和命名

---

## 3. Current scope

当前版本只覆盖最小必需内容：

- 输入视频元信息
- 时间片 / 事件表
- 输出分层音轨引用
- 任务级 metadata
- 路径规范
- 最小 JSON 示例

当前版本**不覆盖**：

- 多人协同编辑协议
- 完整数据库建模
- 在线服务 API schema
- 复杂混音自动化曲线
- 多版本 blueprint merge / diff
- 最终生产级长视频编排规范

这些属于后续阶段。

---

## 4. Blueprint role in the full stack

在当前仓库中，audio blueprint 的位置如下：

- `media_probe`：负责读取输入媒体元信息
- `audio_blueprint_spec`：负责定义结构化中间协议
- MusicGen / AudioLDM：负责部分生成
- Spleeter：负责分离 / 数据准备 / 对照分析
- 后续 serving / workflow：负责调度 blueprint 并产出最终结果

也就是说：

**blueprint 不是模型本身，而是模型之间、输入输出之间的中间合同。**

---

## 5. Path conventions

当前建议统一使用以下路径约定：

### 5.1 输入视频

~~~text
data/interim/<sample_group>/<video_name>.mp4
~~~

例如：

~~~text
data/interim/test_video_probe_smoke/335040.mp4
~~~

### 5.2 blueprint 文件

~~~text
artifacts/blueprints/<blueprint_id>.json
~~~

例如：

~~~text
artifacts/blueprints/week03_smoke_001.json
~~~

### 5.3 生成音频输出

~~~text
artifacts/wav/<stem_name>.wav
~~~

例如：

~~~text
artifacts/wav/musicgen_smoke_001.wav
artifacts/wav/week03_smoke_001_bgm.wav
artifacts/wav/week03_smoke_001_foley.wav
~~~

### 5.4 日志与审计

~~~text
artifacts/logs/<log_name>.log
artifacts/logs/<log_name>.json
~~~

例如：

~~~text
artifacts/logs/week03_audiocraft_smoke.log
artifacts/logs/week03_video_probe_smoke.json
~~~

---

## 6. Core design principle

当前 blueprint 的核心思想是：

**先把“要生成什么”写清楚，再决定“用哪个模型生成”。**

因此 blueprint 主要回答的是：

- 输入视频是什么
- 视频时长是多少
- 哪些时间段要放什么声音
- 这些声音属于哪一层
- 输出结果存到哪里
- 生成任务的来源与参数是什么

而不是直接把模型内部细节塞进 schema。

---

## 7. Top-level schema

顶层建议至少包含以下字段：

- `blueprint_version`
- `blueprint_id`
- `task_type`
- `source_video`
- `media_probe`
- `timeline`
- `output_stems`
- `metadata`

这几个字段的职责如下。

### 7.1 `blueprint_version`

当前 blueprint 版本号。

建议格式：

~~~text
"v0.1"
~~~

### 7.2 `blueprint_id`

当前 blueprint 的唯一标识。

建议格式：

~~~text
"week03_smoke_001"
~~~

### 7.3 `task_type`

当前任务类型。

当前建议最小值：

- `bgm_generation`
- `foley_generation`
- `layered_audio_generation`
- `video_audio_smoke`

### 7.4 `source_video`

描述输入视频路径与基础身份信息。

### 7.5 `media_probe`

记录来自 `tools/media_probe/probe_media.py` 的关键元信息。

### 7.6 `timeline`

描述按时间组织的声音事件表。

### 7.7 `output_stems`

描述输出 stems 的路径、角色与状态。

### 7.8 `metadata`

记录任务来源、生成器、日志、备注等补充信息。

---

## 8. Source video object

`source_video` 建议至少包含：

- `path`
- `exists`
- `video_id`
- `duration_sec`
- `fps`
- `width`
- `height`

示例：

~~~json
{
  "path": "data/interim/test_video_probe_smoke/335040.mp4",
  "exists": true,
  "video_id": "335040",
  "duration_sec": 27.413333,
  "fps": 23.9760,
  "width": 3840,
  "height": 2160
}
~~~

说明：

- `video_id` 用于任务侧唯一识别
- `duration_sec` 应尽量与 probe 结果对齐
- `fps / width / height` 优先来自 probe 结果，不手填猜测

---

## 9. Media probe object

`media_probe` 用于直接承接 `probe_media.py` 的输出核心字段。

建议至少保留：

- `format_name`
- `stream_types`
- `num_streams`
- `video_codec`
- `audio_codec`
- `audio_sr`
- `audio_channels`
- `duration_sec_ffprobe`
- `video_fps_ffprobe`
- `width_ffprobe`
- `height_ffprobe`
- `warnings`

示例：

~~~json
{
  "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
  "stream_types": ["video", "audio", "data"],
  "num_streams": 3,
  "video_codec": "h264",
  "audio_codec": "aac",
  "audio_sr": 48000,
  "audio_channels": 2,
  "duration_sec_ffprobe": 27.413333,
  "video_fps_ffprobe": 23.976023976023978,
  "width_ffprobe": 3840,
  "height_ffprobe": 2160,
  "warnings": [
    "PySoundFile fallback to audioread",
    "cv2 not available"
  ]
}
~~~

说明：

- `warnings` 允许为空列表
- 不要为了“看起来干净”就丢 warning
- probe 字段优先原样保留，避免后续追查时语义漂移

---

## 10. Timeline object

`timeline` 是 blueprint 的核心，描述“什么时候发生什么声音”。

当前建议为一个数组，数组中每个元素表示一个时间事件对象。

每个事件对象建议至少包含：

- `event_id`
- `start_sec`
- `end_sec`
- `layer`
- `event_type`
- `prompt`
- `generator`
- `priority`

字段说明：

### 10.1 `event_id`
事件唯一标识。

### 10.2 `start_sec` / `end_sec`
时间范围，单位为秒。

### 10.3 `layer`
声音层级。当前建议最小值：

- `bgm`
- `foley`
- `ambience`
- `speech_reserved`

### 10.4 `event_type`
事件类型。当前建议最小值：

- `ambient_bed`
- `wind`
- `footstep`
- `rustle`
- `impact`
- `transition`

### 10.5 `prompt`
生成或描述该事件的自然语言提示。

### 10.6 `generator`
当前计划使用的生成器或来源。

建议最小值：

- `musicgen`
- `audioldm`
- `recorded`
- `spleeter_derived`
- `manual_placeholder`

### 10.7 `priority`
事件优先级，用于后续冲突处理或混音排序。

建议最小值：

- `high`
- `medium`
- `low`

---

## 11. Timeline event example

~~~json
[
  {
    "event_id": "evt_001",
    "start_sec": 0.0,
    "end_sec": 8.0,
    "layer": "bgm",
    "event_type": "ambient_bed",
    "prompt": "calm ambient background bed for a forest rescue video",
    "generator": "musicgen",
    "priority": "high"
  },
  {
    "event_id": "evt_002",
    "start_sec": 3.2,
    "end_sec": 4.5,
    "layer": "foley",
    "event_type": "rustle",
    "prompt": "light forest foliage rustle",
    "generator": "audioldm",
    "priority": "medium"
  }
]
~~~

---

## 12. Output stems object

`output_stems` 用于描述输出音频层。

当前建议为数组，每个 stem 对象至少包含：

- `stem_name`
- `layer`
- `path`
- `format`
- `sample_rate`
- `channels`
- `status`

字段说明：

### 12.1 `stem_name`
例如：

- `bgm`
- `foley`
- `ambience`
- `mix_preview`

### 12.2 `layer`
与 `timeline.layer` 对齐。

### 12.3 `path`
输出 wav 路径。

### 12.4 `format`
当前建议最小值为：

- `wav`

### 12.5 `sample_rate`
例如：

- `32000`
- `44100`
- `48000`

### 12.6 `channels`
例如：

- `1`
- `2`

### 12.7 `status`
当前建议最小值：

- `planned`
- `generated`
- `missing`
- `failed`

---

## 13. Output stems example

~~~json
[
  {
    "stem_name": "bgm",
    "layer": "bgm",
    "path": "artifacts/wav/musicgen_smoke_001.wav",
    "format": "wav",
    "sample_rate": 32000,
    "channels": 1,
    "status": "generated"
  },
  {
    "stem_name": "foley",
    "layer": "foley",
    "path": "artifacts/wav/week03_smoke_001_foley.wav",
    "format": "wav",
    "sample_rate": 32000,
    "channels": 1,
    "status": "planned"
  }
]
~~~

---

## 14. Metadata object

`metadata` 用于记录任务级补充信息。

建议至少包含：

- `created_at`
- `created_by`
- `repo_stage`
- `generator_stack`
- `log_ref`
- `notes`

示例：

~~~json
{
  "created_at": "2026-03-27T10:40:00+08:00",
  "created_by": "week03_friday_smoke",
  "repo_stage": "week03_stage1_bootstrap",
  "generator_stack": [
    "musicgen-small",
    "audioldm-planned",
    "spleeter-planned"
  ],
  "log_ref": "artifacts/logs/week03_audiocraft_smoke.log",
  "notes": "first successful MusicGen smoke output for week03"
}
~~~

---

## 15. Minimal complete example

下面给出一个完整但最小的 blueprint 示例。

~~~json
{
  "blueprint_version": "v0.1",
  "blueprint_id": "week03_smoke_001",
  "task_type": "video_audio_smoke",
  "source_video": {
    "path": "data/interim/test_video_probe_smoke/335040.mp4",
    "exists": true,
    "video_id": "335040",
    "duration_sec": 27.413333,
    "fps": 23.9760,
    "width": 3840,
    "height": 2160
  },
  "media_probe": {
    "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
    "stream_types": ["video", "audio", "data"],
    "num_streams": 3,
    "video_codec": "h264",
    "audio_codec": "aac",
    "audio_sr": 48000,
    "audio_channels": 2,
    "duration_sec_ffprobe": 27.413333,
    "video_fps_ffprobe": 23.976023976023978,
    "width_ffprobe": 3840,
    "height_ffprobe": 2160,
    "warnings": [
      "PySoundFile fallback to audioread",
      "cv2 not available"
    ]
  },
  "timeline": [
    {
      "event_id": "evt_001",
      "start_sec": 0.0,
      "end_sec": 8.0,
      "layer": "bgm",
      "event_type": "ambient_bed",
      "prompt": "calm ambient background bed for a forest rescue video",
      "generator": "musicgen",
      "priority": "high"
    }
  ],
  "output_stems": [
    {
      "stem_name": "bgm",
      "layer": "bgm",
      "path": "artifacts/wav/musicgen_smoke_001.wav",
      "format": "wav",
      "sample_rate": 32000,
      "channels": 1,
      "status": "generated"
    }
  ],
  "metadata": {
    "created_at": "2026-03-27T10:40:00+08:00",
    "created_by": "week03_friday_smoke",
    "repo_stage": "week03_stage1_bootstrap",
    "generator_stack": [
      "musicgen-small",
      "audioldm-planned",
      "spleeter-planned"
    ],
    "log_ref": "artifacts/logs/week03_audiocraft_smoke.log",
    "notes": "first successful MusicGen smoke output for week03"
  }
}
~~~

---

## 16. Validation rules

当前版本建议至少满足以下校验规则：

1. `blueprint_version` 不能为空
2. `blueprint_id` 不能为空
3. `source_video.path` 必须存在
4. `timeline` 中所有事件都必须满足：
   - `start_sec >= 0`
   - `end_sec > start_sec`
5. `timeline.layer` 必须与 `output_stems.layer` 体系一致
6. `output_stems.path` 应尽量使用仓库相对路径
7. `status = generated` 的 stem 必须真实存在
8. `metadata.log_ref` 应指向真实日志文件
9. 不允许默默丢弃 `warnings`

---

## 17. Relationship with current repository modules

当前 blueprint 与现有模块的关系如下：

### 17.1 `tools/media_probe/`
负责为 `source_video` 与 `media_probe` 提供原始元信息。

### 17.2 MusicGen / AudioCraft
当前是第一阶段主 smoke 生成入口，主要负责 `bgm` 或基础环境床音层。

### 17.3 AudioLDM
当前定义为后续 `foley / ambience` 的备选生成路线，不要求本阶段完整接入。

### 17.4 Spleeter
当前定义为分层数据准备、分离对照与评估工具位，不直接替代主生成器。

### 17.5 `src/serving/`
当前服务化层不直接生成 blueprint，但后续应能读取 blueprint 并触发任务。

---

## 18. Current non-goals

当前版本明确不做：

- 自动从视频内容中完整提取事件表
- 自动混音参数求解
- 多蓝图版本差分
- 跨视频批量调度
- 完整的 UI 编辑器
- 生产级 schema registry

这些是后续阶段的问题，不在 `v0.1` 范围内。

---

## 19. Current conclusion

`audio_blueprint_spec v0.1` 的意义不在于“字段多”，而在于：

- 它给输入视频、事件表、stems、metadata 建了统一合同
- 它把 `media_probe`、MusicGen、AudioLDM、Spleeter 的边界写清楚了
- 它为后续 prototype、服务化、评测和复盘提供了最小协议基础

因此，当前可以把它视为：

**Week 03 第一次把长期项目从“工具与想法”推进到“协议与产物”的关键文档。**

---

## 20. Next step

在本规范之后，下一步建议是：

1. 基于本 spec 生成第一份真实 blueprint JSON
2. 将 `musicgen_smoke_001.wav` 填入 `output_stems`
3. 在 `docs/weekly/2026-03-27_week03_summary.md` 中引用本规范
4. 下周继续扩展：
   - `foley` 事件槽位
   - `mix_preview`
   - 验证脚本
   - service 读取 blueprint 的最小入口