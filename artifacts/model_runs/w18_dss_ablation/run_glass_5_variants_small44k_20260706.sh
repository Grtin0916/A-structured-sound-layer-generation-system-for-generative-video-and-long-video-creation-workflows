#!/usr/bin/env bash
set -euo pipefail
MAINBASE="${MAINBASE:-$HOME/work/audio_engineering_repo_skeleton_v1}"
MMAUDIO_ROOT="${MMAUDIO_ROOT:-/home/GRT/work/_external/MMAudio}"
HF_HOME="${HF_HOME:-$MMAUDIO_ROOT/.hf_cache}"
HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export MAINBASE MMAUDIO_ROOT HF_HOME HUGGINGFACE_HUB_CACHE
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
echo "CASE_ID=glass_drop_room_001"

echo '### RUN w18_006_glass_drop_room_001_naive duration=7s variant=naive'
mkdir -p "$MAINBASE/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/naive" "$MAINBASE/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/naive/mmaudio_small_44k_raw_output"
cd "$MMAUDIO_ROOT"
test -f weights/mmaudio_small_44k.pth
test -f ext_weights/synchformer_state_dict.pth
test -f ext_weights/v1-44.pth
if [ -f "$MAINBASE/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/naive/w18_006_glass_drop_room_001_naive.flac" ] || [ -f "$MAINBASE/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/naive/w18_006_glass_drop_room_001_naive.wav" ]; then echo 'SKIP existing output w18_006_glass_drop_room_001_naive'; else
  python demo.py --variant small_44k --duration=7 --video="$MAINBASE/cases/glass_drop_room_001/input_video.mp4" --prompt "Create realistic sound effects for this short video. Scene: quiet indoor room, a glass object drops and breaks on a hard floor. Use natural ambience and match visible actions. Avoid speech, music, cartoon boing, long reverb tail, lyrics, voiceover." --output="$MAINBASE/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/naive/mmaudio_small_44k_raw_output" --seed=42
  NEW_AUDIO=$(find "$MAINBASE/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/naive/mmaudio_small_44k_raw_output" -type f \( -name '*.flac' -o -name '*.wav' \) -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2- || true)
  if [ -z "$NEW_AUDIO" ]; then echo "NO_MMAUDIO_OUTPUT_FOUND"; exit 2; fi
  cp "$NEW_AUDIO" "$MAINBASE/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/naive/w18_006_glass_drop_room_001_naive.flac"
  if command -v ffmpeg >/dev/null 2>&1; then ffmpeg -y -i "$MAINBASE/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/naive/w18_006_glass_drop_room_001_naive.flac" "$MAINBASE/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/naive/w18_006_glass_drop_room_001_naive.wav" >/dev/null 2>&1 || true; fi
fi
ls -lh "$MAINBASE/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/naive/w18_006_glass_drop_room_001_naive.flac" "$MAINBASE/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/naive/w18_006_glass_drop_room_001_naive.wav" 2>/dev/null || true

echo '### RUN w18_007_glass_drop_room_001_naive_rich duration=7s variant=naive_rich'
mkdir -p "$MAINBASE/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/naive_rich" "$MAINBASE/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/naive_rich/mmaudio_small_44k_raw_output"
cd "$MMAUDIO_ROOT"
test -f weights/mmaudio_small_44k.pth
test -f ext_weights/synchformer_state_dict.pth
test -f ext_weights/v1-44.pth
if [ -f "$MAINBASE/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/naive_rich/w18_007_glass_drop_room_001_naive_rich.flac" ] || [ -f "$MAINBASE/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/naive_rich/w18_007_glass_drop_room_001_naive_rich.wav" ]; then echo 'SKIP existing output w18_007_glass_drop_room_001_naive_rich'; else
  python demo.py --variant small_44k --duration=7 --video="$MAINBASE/cases/glass_drop_room_001/input_video.mp4" --prompt "Create realistic sound effects for this short video. Scene: quiet indoor room, a glass object drops and breaks on a hard floor. The visible actions include: around 0.00s, quiet quiet_room_tone with quiet indoor room tone; around 0.00s, glass glass_slip with glass sliding off table edge; around 0.00s, impact impact with sharp glass impact on hard floor; around 0.00s, shatter shatter_tail with small glass fragments scattering. Make the sounds natural, synchronized, and cinematic. Avoid speech, music, cartoon boing, long reverb tail, lyrics, voiceover." --output="$MAINBASE/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/naive_rich/mmaudio_small_44k_raw_output" --seed=42
  NEW_AUDIO=$(find "$MAINBASE/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/naive_rich/mmaudio_small_44k_raw_output" -type f \( -name '*.flac' -o -name '*.wav' \) -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2- || true)
  if [ -z "$NEW_AUDIO" ]; then echo "NO_MMAUDIO_OUTPUT_FOUND"; exit 2; fi
  cp "$NEW_AUDIO" "$MAINBASE/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/naive_rich/w18_007_glass_drop_room_001_naive_rich.flac"
  if command -v ffmpeg >/dev/null 2>&1; then ffmpeg -y -i "$MAINBASE/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/naive_rich/w18_007_glass_drop_room_001_naive_rich.flac" "$MAINBASE/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/naive_rich/w18_007_glass_drop_room_001_naive_rich.wav" >/dev/null 2>&1 || true; fi
fi
ls -lh "$MAINBASE/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/naive_rich/w18_007_glass_drop_room_001_naive_rich.flac" "$MAINBASE/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/naive_rich/w18_007_glass_drop_room_001_naive_rich.wav" 2>/dev/null || true

echo '### RUN w18_008_glass_drop_room_001_dss_global duration=7s variant=dss_global'
mkdir -p "$MAINBASE/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/dss_global" "$MAINBASE/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/dss_global/mmaudio_small_44k_raw_output"
cd "$MMAUDIO_ROOT"
test -f weights/mmaudio_small_44k.pth
test -f ext_weights/synchformer_state_dict.pth
test -f ext_weights/v1-44.pth
if [ -f "$MAINBASE/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/dss_global/w18_008_glass_drop_room_001_dss_global.flac" ] || [ -f "$MAINBASE/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/dss_global/w18_008_glass_drop_room_001_dss_global.wav" ]; then echo 'SKIP existing output w18_008_glass_drop_room_001_dss_global'; else
  python demo.py --variant small_44k --duration=7 --video="$MAINBASE/cases/glass_drop_room_001/input_video.mp4" --prompt "Generate synchronized audio for a 10.0s video. Global scene: quiet indoor room, a glass object drops and breaks on a hard floor. Overall style: realistic Foley, no speech, no background music. Keep ambience stable, keep foley clear, avoid forbidden content: speech, music, cartoon boing, long reverb tail, lyrics, voiceover." --output="$MAINBASE/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/dss_global/mmaudio_small_44k_raw_output" --seed=42
  NEW_AUDIO=$(find "$MAINBASE/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/dss_global/mmaudio_small_44k_raw_output" -type f \( -name '*.flac' -o -name '*.wav' \) -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2- || true)
  if [ -z "$NEW_AUDIO" ]; then echo "NO_MMAUDIO_OUTPUT_FOUND"; exit 2; fi
  cp "$NEW_AUDIO" "$MAINBASE/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/dss_global/w18_008_glass_drop_room_001_dss_global.flac"
  if command -v ffmpeg >/dev/null 2>&1; then ffmpeg -y -i "$MAINBASE/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/dss_global/w18_008_glass_drop_room_001_dss_global.flac" "$MAINBASE/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/dss_global/w18_008_glass_drop_room_001_dss_global.wav" >/dev/null 2>&1 || true; fi
fi
ls -lh "$MAINBASE/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/dss_global/w18_008_glass_drop_room_001_dss_global.flac" "$MAINBASE/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/dss_global/w18_008_glass_drop_room_001_dss_global.wav" 2>/dev/null || true

echo '### RUN w18_009_glass_drop_room_001_dss_event_timeline duration=7s variant=dss_event_timeline'
mkdir -p "$MAINBASE/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/dss_event_timeline" "$MAINBASE/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/dss_event_timeline/mmaudio_small_44k_raw_output"
cd "$MMAUDIO_ROOT"
test -f weights/mmaudio_small_44k.pth
test -f ext_weights/synchformer_state_dict.pth
test -f ext_weights/v1-44.pth
if [ -f "$MAINBASE/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/dss_event_timeline/w18_009_glass_drop_room_001_dss_event_timeline.flac" ] || [ -f "$MAINBASE/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/dss_event_timeline/w18_009_glass_drop_room_001_dss_event_timeline.wav" ]; then echo 'SKIP existing output w18_009_glass_drop_room_001_dss_event_timeline'; else
  python demo.py --variant small_44k --duration=7 --video="$MAINBASE/cases/glass_drop_room_001/input_video.mp4" --prompt "Generate audio for a 10.0s video using this event timeline. Scene: quiet indoor room, a glass object drops and breaks on a hard floor. Event timeline: 0.00s: object=quiet; action=quiet_room_tone; sound=quiet indoor room tone; layer=foley; priority=1; tolerance=500ms | 0.00s: object=glass; action=glass_slip; sound=glass sliding off table edge; layer=foley; priority=4; tolerance=120ms | 0.00s: object=impact; action=impact; sound=sharp glass impact on hard floor; layer=foley; priority=5; tolerance=70ms | 0.00s: object=shatter; action=shatter_tail; sound=small glass fragments scattering; layer=foley; priority=5; tolerance=100ms. Preserve temporal order. Align transient sounds to event timestamps. Do not add: speech, music, cartoon boing, long reverb tail, lyrics, voiceover." --output="$MAINBASE/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/dss_event_timeline/mmaudio_small_44k_raw_output" --seed=42
  NEW_AUDIO=$(find "$MAINBASE/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/dss_event_timeline/mmaudio_small_44k_raw_output" -type f \( -name '*.flac' -o -name '*.wav' \) -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2- || true)
  if [ -z "$NEW_AUDIO" ]; then echo "NO_MMAUDIO_OUTPUT_FOUND"; exit 2; fi
  cp "$NEW_AUDIO" "$MAINBASE/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/dss_event_timeline/w18_009_glass_drop_room_001_dss_event_timeline.flac"
  if command -v ffmpeg >/dev/null 2>&1; then ffmpeg -y -i "$MAINBASE/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/dss_event_timeline/w18_009_glass_drop_room_001_dss_event_timeline.flac" "$MAINBASE/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/dss_event_timeline/w18_009_glass_drop_room_001_dss_event_timeline.wav" >/dev/null 2>&1 || true; fi
fi
ls -lh "$MAINBASE/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/dss_event_timeline/w18_009_glass_drop_room_001_dss_event_timeline.flac" "$MAINBASE/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/dss_event_timeline/w18_009_glass_drop_room_001_dss_event_timeline.wav" 2>/dev/null || true

echo '### RUN w18_010_glass_drop_room_001_dss_layer_avoid duration=7s variant=dss_layer_avoid'
mkdir -p "$MAINBASE/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/dss_layer_avoid" "$MAINBASE/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/dss_layer_avoid/mmaudio_small_44k_raw_output"
cd "$MMAUDIO_ROOT"
test -f weights/mmaudio_small_44k.pth
test -f ext_weights/synchformer_state_dict.pth
test -f ext_weights/v1-44.pth
if [ -f "$MAINBASE/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/dss_layer_avoid/w18_010_glass_drop_room_001_dss_layer_avoid.flac" ] || [ -f "$MAINBASE/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/dss_layer_avoid/w18_010_glass_drop_room_001_dss_layer_avoid.wav" ]; then echo 'SKIP existing output w18_010_glass_drop_room_001_dss_layer_avoid'; else
  python demo.py --variant small_44k --duration=7 --video="$MAINBASE/cases/glass_drop_room_001/input_video.mp4" --prompt "Generate layered video-synchronized audio for a 7.0s video. Scene layer: quiet indoor room, a glass object drops and breaks on a hard floor. Foley layer must emphasize these events: 0.00s: object=quiet; action=quiet_room_tone; sound=quiet indoor room tone; layer=foley; priority=1; tolerance=500ms | 0.00s: object=glass; action=glass_slip; sound=glass sliding off table edge; layer=foley; priority=4; tolerance=120ms | 0.00s: object=impact; action=impact; sound=sharp glass impact on hard floor; layer=foley; priority=5; tolerance=70ms | 0.00s: object=shatter; action=shatter_tail; sound=small glass fragments scattering; layer=foley; priority=5; tolerance=100ms. Ambience layer should remain low, continuous, and non-dominant. Music layer should be absent unless explicitly required. Forbidden content: speech, music, cartoon boing, long reverb tail, lyrics, voiceover. No speech, no lyrics, no unrelated background music." --output="$MAINBASE/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/dss_layer_avoid/mmaudio_small_44k_raw_output" --seed=42
  NEW_AUDIO=$(find "$MAINBASE/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/dss_layer_avoid/mmaudio_small_44k_raw_output" -type f \( -name '*.flac' -o -name '*.wav' \) -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2- || true)
  if [ -z "$NEW_AUDIO" ]; then echo "NO_MMAUDIO_OUTPUT_FOUND"; exit 2; fi
  cp "$NEW_AUDIO" "$MAINBASE/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/dss_layer_avoid/w18_010_glass_drop_room_001_dss_layer_avoid.flac"
  if command -v ffmpeg >/dev/null 2>&1; then ffmpeg -y -i "$MAINBASE/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/dss_layer_avoid/w18_010_glass_drop_room_001_dss_layer_avoid.flac" "$MAINBASE/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/dss_layer_avoid/w18_010_glass_drop_room_001_dss_layer_avoid.wav" >/dev/null 2>&1 || true; fi
fi
ls -lh "$MAINBASE/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/dss_layer_avoid/w18_010_glass_drop_room_001_dss_layer_avoid.flac" "$MAINBASE/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/dss_layer_avoid/w18_010_glass_drop_room_001_dss_layer_avoid.wav" 2>/dev/null || true

