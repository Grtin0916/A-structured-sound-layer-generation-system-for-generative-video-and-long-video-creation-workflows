# media_probe

## 作用

`media_probe` 是当前仓库中的一个工具层原型，用于在训练、生成、分离或部署链路之前，对音频/视频文件做最基础的元信息探测与审计。

当前版本目标很克制：

- 读取容器与 stream 级元信息
- 补充音频采样率、声道数、时长
- 在可选依赖存在时补充视频宽高与帧率
- 以 JSON 形式输出，便于后续脚本或日志系统接入

它不是当前 `baseline_v1` 主 smoke 链的一部分，而是后续“媒体入库前检查”和“生成链路接入前审计”的工具位。

---

## 依赖关系

### 硬依赖

- `ffprobe`

没有 `ffprobe` 时，脚本直接失败，因为容器与 stream 信息都依赖它提供。

### 软依赖

- `librosa`
  - 用于补充音频采样率、声道数、音频时长
- `cv2`
  - 用于补充视频帧率、宽高、帧数

如果软依赖不存在，脚本不会直接报错退出，而是在输出 JSON 中写入 warning 字段。

---

## 最小用法

### 音频探测

```bash
python tools/media_probe/probe_media.py data/interim/esc50_raw/audio/1-155858-D-25.wav --pretty
```

### 输出到日志文件

```bash
python tools/media_probe/probe_media.py data/interim/esc50_raw/audio/1-155858-D-25.wav --pretty \
  > artifacts/logs/week02_media_probe_smoke.log 2>&1
```

---

## 当前已验证样例

已验证样本：

```text
data/interim/esc50_raw/audio/1-155858-D-25.wav
```

已验证输出日志：

```text
artifacts/logs/week02_media_probe_smoke.log
```

---

## 典型输出字段

对于音频文件，当前版本通常会输出以下字段：

- `path`
- `exists`
- `format_name`
- `format_long_name`
- `size_bytes`
- `bit_rate`
- `duration_sec_ffprobe`
- `stream_types`
- `num_streams`
- `audio_codec`
- `audio_sr_ffprobe`
- `audio_channels_ffprobe`
- `audio_sr`
- `audio_channels`
- `audio_duration_sec_librosa`

对于视频文件，如果 `cv2` 可用，还可能出现：

- `video_fps_cv2`
- `width_cv2`
- `height_cv2`
- `frame_count_cv2`

---

## 设计约束

当前版本是“最小可用工具”，不是通用媒体审计平台。

明确不做的事情包括：

- 批量目录扫描
- CSV / DataFrame 汇总
- 哈希校验
- 波形统计与频谱统计
- 多音轨精细分析
- VFR 视频的严谨时间基处理

---

## 已知限制

1. `ffprobe` 是主依赖，未安装时无法工作。
2. `librosa` 只做基础补充，不负责复杂音频解析。
3. `cv2` 仅作为可选视频补充层，不保证覆盖所有编码格式。
4. 多个音频流 / 视频流时，当前只取第一个匹配流做摘要。
5. 当前输出是单文件 JSON，不支持批处理模式。

---

## 在仓库中的角色

当前项目分三层看：

1. `baseline_v1`：主线最小训练/导出/推理闭环
2. `media_probe`：工具层辅助能力
3. AudioCraft / AudioLDM / Spleeter：后续桥接模块

所以 `media_probe` 的定位不是“新主线”，而是“给未来主线铺轨的工具”。