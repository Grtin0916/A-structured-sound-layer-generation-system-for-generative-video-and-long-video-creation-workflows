#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional


def run_ffprobe(path: str) -> Dict[str, Any]:
    """
    调用 ffprobe 获取媒体基础信息。
    返回 ffprobe 的 JSON 结果；失败时抛出 RuntimeError。
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        path,
    ]
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
    except FileNotFoundError:
        raise RuntimeError("ffprobe 未安装或不在 PATH 中。")
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.strip() if e.stderr else ""
        raise RuntimeError(f"ffprobe 执行失败：{stderr}")

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"ffprobe 输出不是合法 JSON：{e}")


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None or x == "N/A":
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def extract_stream_types(streams: List[Dict[str, Any]]) -> List[str]:
    types = []
    for s in streams:
        codec_type = s.get("codec_type")
        if codec_type and codec_type not in types:
            types.append(codec_type)
    return types


def enrich_audio_with_librosa(path: str, out: Dict[str, Any]) -> None:
    """
    用 librosa 补充音频采样率和时长。
    不强依赖 librosa，失败时只记录错误信息。
    """
    try:
        import librosa  # type: ignore
    except Exception as e:
        out["audio_probe_warning"] = f"librosa 不可用：{e}"
        return

    try:
        # sr=None: 保持原始采样率，不自动重采样到 22050
        y, sr = librosa.load(path, sr=None, mono=False)
        out["audio_sr"] = int(sr)

        if hasattr(y, "ndim"):
            if y.ndim == 1:
                out["audio_channels"] = 1
                num_samples = y.shape[0]
            else:
                out["audio_channels"] = int(y.shape[0])
                num_samples = y.shape[-1]
            out["audio_duration_sec_librosa"] = round(float(num_samples) / float(sr), 6)
    except Exception as e:
        out["audio_probe_warning"] = f"librosa 读取失败：{e}"


def parse_fraction(frac: str) -> Optional[float]:
    if not frac or frac == "0/0":
        return None
    if "/" in frac:
        a, b = frac.split("/", 1)
        try:
            a = float(a)
            b = float(b)
            if b == 0:
                return None
            return a / b
        except ValueError:
            return None
    return safe_float(frac)


def enrich_video_with_cv2(path: str, out: Dict[str, Any]) -> None:
    """
    用 OpenCV 补充视频宽高和 fps。
    不强依赖 cv2，失败时只记录错误信息。
    """
    try:
        import cv2  # type: ignore
    except Exception as e:
        out["video_probe_warning"] = f"cv2 不可用：{e}"
        return

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        out["video_probe_warning"] = "cv2 无法打开该视频文件。"
        return

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        out["video_fps_cv2"] = round(float(fps), 6) if fps else None
        out["width_cv2"] = int(width) if width else None
        out["height_cv2"] = int(height) if height else None
        out["frame_count_cv2"] = int(frame_count) if frame_count else None
    finally:
        cap.release()


def build_summary(path: str, ffprobe_data: Dict[str, Any]) -> Dict[str, Any]:
    streams = ffprobe_data.get("streams", [])
    fmt = ffprobe_data.get("format", {})

    out: Dict[str, Any] = {
        "path": path,
        "exists": os.path.exists(path),
        "format_name": fmt.get("format_name"),
        "format_long_name": fmt.get("format_long_name"),
        "size_bytes": int(fmt["size"]) if str(fmt.get("size", "")).isdigit() else None,
        "bit_rate": int(fmt["bit_rate"]) if str(fmt.get("bit_rate", "")).isdigit() else None,
        "duration_sec_ffprobe": safe_float(fmt.get("duration")),
        "stream_types": extract_stream_types(streams),
        "num_streams": len(streams),
    }

    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)

    if audio_stream is not None:
        out["audio_codec"] = audio_stream.get("codec_name")
        out["audio_sr_ffprobe"] = (
            int(audio_stream["sample_rate"])
            if str(audio_stream.get("sample_rate", "")).isdigit()
            else None
        )
        out["audio_channels_ffprobe"] = audio_stream.get("channels")

    if video_stream is not None:
        out["video_codec"] = video_stream.get("codec_name")
        out["width_ffprobe"] = video_stream.get("width")
        out["height_ffprobe"] = video_stream.get("height")
        out["video_fps_ffprobe"] = parse_fraction(video_stream.get("avg_frame_rate", ""))

    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe media metadata with ffprobe + optional librosa/cv2.")
    parser.add_argument("path", type=str, help="媒体文件路径")
    parser.add_argument("--pretty", action="store_true", help="是否美化 JSON 输出")
    args = parser.parse_args()

    path = args.path

    if not os.path.exists(path):
        print(json.dumps({
            "path": path,
            "exists": False,
            "error": "文件不存在",
        }, ensure_ascii=False, indent=2 if args.pretty else None))
        return 1

    try:
        ffprobe_data = run_ffprobe(path)
        out = build_summary(path, ffprobe_data)

        if "audio" in out.get("stream_types", []):
            enrich_audio_with_librosa(path, out)

        if "video" in out.get("stream_types", []):
            enrich_video_with_cv2(path, out)

        print(json.dumps(out, ensure_ascii=False, indent=2 if args.pretty else None))
        return 0

    except RuntimeError as e:
        print(json.dumps({
            "path": path,
            "exists": os.path.exists(path),
            "error": str(e),
        }, ensure_ascii=False, indent=2 if args.pretty else None))
        return 2


if __name__ == "__main__":
    sys.exit(main())
