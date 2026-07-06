#!/usr/bin/env bash
set -euo pipefail
MAINBASE="${MAINBASE:-$HOME/work/audio_engineering_repo_skeleton_v1}"
MMAUDIO_ROOT="${MMAUDIO_ROOT:-/home/GRT/work/_external/MMAudio}"
HF_HOME="${HF_HOME:-$MMAUDIO_ROOT/.hf_cache}"
HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export MAINBASE MMAUDIO_ROOT HF_HOME HUGGINGFACE_HUB_CACHE
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
echo "CASE_ID=robot_warehouse_pick_001"

echo '### RUN w18_016_robot_warehouse_pick_001_naive duration=10s variant=naive'
mkdir -p "$MAINBASE/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/naive" "$MAINBASE/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/naive/mmaudio_small_44k_raw_output"
cd "$MMAUDIO_ROOT"
test -f weights/mmaudio_small_44k.pth
test -f ext_weights/synchformer_state_dict.pth
test -f ext_weights/v1-44.pth
if [ -f "$MAINBASE/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/naive/w18_016_robot_warehouse_pick_001_naive.flac" ] || [ -f "$MAINBASE/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/naive/w18_016_robot_warehouse_pick_001_naive.wav" ]; then echo 'SKIP existing output w18_016_robot_warehouse_pick_001_naive'; else
  python demo.py --variant small_44k --duration=10 --video="$MAINBASE/cases/robot_warehouse_pick_001/input_video.mp4" --prompt "Create realistic sound effects for this short video. Scene: robot arm picking an item in a warehouse. Use natural ambience and match visible actions. Avoid human speech, music, explosion, vehicle horn, speech, lyrics, voiceover." --output="$MAINBASE/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/naive/mmaudio_small_44k_raw_output" --seed=42
  NEW_AUDIO=$(find "$MAINBASE/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/naive/mmaudio_small_44k_raw_output" -type f \( -name '*.flac' -o -name '*.wav' \) -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2- || true)
  if [ -z "$NEW_AUDIO" ]; then echo "NO_MMAUDIO_OUTPUT_FOUND"; exit 2; fi
  cp "$NEW_AUDIO" "$MAINBASE/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/naive/w18_016_robot_warehouse_pick_001_naive.flac"
  if command -v ffmpeg >/dev/null 2>&1; then ffmpeg -y -i "$MAINBASE/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/naive/w18_016_robot_warehouse_pick_001_naive.flac" "$MAINBASE/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/naive/w18_016_robot_warehouse_pick_001_naive.wav" >/dev/null 2>&1 || true; fi
fi
ls -lh "$MAINBASE/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/naive/w18_016_robot_warehouse_pick_001_naive.flac" "$MAINBASE/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/naive/w18_016_robot_warehouse_pick_001_naive.wav" 2>/dev/null || true

echo '### RUN w18_017_robot_warehouse_pick_001_naive_rich duration=10s variant=naive_rich'
mkdir -p "$MAINBASE/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/naive_rich" "$MAINBASE/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/naive_rich/mmaudio_small_44k_raw_output"
cd "$MMAUDIO_ROOT"
test -f weights/mmaudio_small_44k.pth
test -f ext_weights/synchformer_state_dict.pth
test -f ext_weights/v1-44.pth
if [ -f "$MAINBASE/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/naive_rich/w18_017_robot_warehouse_pick_001_naive_rich.flac" ] || [ -f "$MAINBASE/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/naive_rich/w18_017_robot_warehouse_pick_001_naive_rich.wav" ]; then echo 'SKIP existing output w18_017_robot_warehouse_pick_001_naive_rich'; else
  python demo.py --variant small_44k --duration=10 --video="$MAINBASE/cases/robot_warehouse_pick_001/input_video.mp4" --prompt "Create realistic sound effects for this short video. Scene: robot arm picking an item in a warehouse. The visible actions include: around 0.00s, warehouse warehouse_hum with low warehouse ventilation hum; around 0.00s, servo servo_arm_move with precise electric servo arm movement; around 0.00s, box box_lift with cardboard box lifted from shelf; around 0.00s, confirm confirm_beep with short clean robot confirmation beep. Make the sounds natural, synchronized, and cinematic. Avoid human speech, music, explosion, vehicle horn, speech, lyrics, voiceover." --output="$MAINBASE/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/naive_rich/mmaudio_small_44k_raw_output" --seed=42
  NEW_AUDIO=$(find "$MAINBASE/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/naive_rich/mmaudio_small_44k_raw_output" -type f \( -name '*.flac' -o -name '*.wav' \) -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2- || true)
  if [ -z "$NEW_AUDIO" ]; then echo "NO_MMAUDIO_OUTPUT_FOUND"; exit 2; fi
  cp "$NEW_AUDIO" "$MAINBASE/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/naive_rich/w18_017_robot_warehouse_pick_001_naive_rich.flac"
  if command -v ffmpeg >/dev/null 2>&1; then ffmpeg -y -i "$MAINBASE/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/naive_rich/w18_017_robot_warehouse_pick_001_naive_rich.flac" "$MAINBASE/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/naive_rich/w18_017_robot_warehouse_pick_001_naive_rich.wav" >/dev/null 2>&1 || true; fi
fi
ls -lh "$MAINBASE/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/naive_rich/w18_017_robot_warehouse_pick_001_naive_rich.flac" "$MAINBASE/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/naive_rich/w18_017_robot_warehouse_pick_001_naive_rich.wav" 2>/dev/null || true

echo '### RUN w18_018_robot_warehouse_pick_001_dss_global duration=10s variant=dss_global'
mkdir -p "$MAINBASE/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/dss_global" "$MAINBASE/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/dss_global/mmaudio_small_44k_raw_output"
cd "$MMAUDIO_ROOT"
test -f weights/mmaudio_small_44k.pth
test -f ext_weights/synchformer_state_dict.pth
test -f ext_weights/v1-44.pth
if [ -f "$MAINBASE/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/dss_global/w18_018_robot_warehouse_pick_001_dss_global.flac" ] || [ -f "$MAINBASE/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/dss_global/w18_018_robot_warehouse_pick_001_dss_global.wav" ]; then echo 'SKIP existing output w18_018_robot_warehouse_pick_001_dss_global'; else
  python demo.py --variant small_44k --duration=10 --video="$MAINBASE/cases/robot_warehouse_pick_001/input_video.mp4" --prompt "Generate synchronized audio for a 10.0s video. Global scene: robot arm picking an item in a warehouse. Overall style: realistic Foley, no speech, no background music. Keep ambience stable, keep foley clear, avoid forbidden content: human speech, music, explosion, vehicle horn, speech, lyrics, voiceover." --output="$MAINBASE/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/dss_global/mmaudio_small_44k_raw_output" --seed=42
  NEW_AUDIO=$(find "$MAINBASE/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/dss_global/mmaudio_small_44k_raw_output" -type f \( -name '*.flac' -o -name '*.wav' \) -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2- || true)
  if [ -z "$NEW_AUDIO" ]; then echo "NO_MMAUDIO_OUTPUT_FOUND"; exit 2; fi
  cp "$NEW_AUDIO" "$MAINBASE/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/dss_global/w18_018_robot_warehouse_pick_001_dss_global.flac"
  if command -v ffmpeg >/dev/null 2>&1; then ffmpeg -y -i "$MAINBASE/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/dss_global/w18_018_robot_warehouse_pick_001_dss_global.flac" "$MAINBASE/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/dss_global/w18_018_robot_warehouse_pick_001_dss_global.wav" >/dev/null 2>&1 || true; fi
fi
ls -lh "$MAINBASE/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/dss_global/w18_018_robot_warehouse_pick_001_dss_global.flac" "$MAINBASE/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/dss_global/w18_018_robot_warehouse_pick_001_dss_global.wav" 2>/dev/null || true

echo '### RUN w18_019_robot_warehouse_pick_001_dss_event_timeline duration=10s variant=dss_event_timeline'
mkdir -p "$MAINBASE/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/dss_event_timeline" "$MAINBASE/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/dss_event_timeline/mmaudio_small_44k_raw_output"
cd "$MMAUDIO_ROOT"
test -f weights/mmaudio_small_44k.pth
test -f ext_weights/synchformer_state_dict.pth
test -f ext_weights/v1-44.pth
if [ -f "$MAINBASE/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/dss_event_timeline/w18_019_robot_warehouse_pick_001_dss_event_timeline.flac" ] || [ -f "$MAINBASE/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/dss_event_timeline/w18_019_robot_warehouse_pick_001_dss_event_timeline.wav" ]; then echo 'SKIP existing output w18_019_robot_warehouse_pick_001_dss_event_timeline'; else
  python demo.py --variant small_44k --duration=10 --video="$MAINBASE/cases/robot_warehouse_pick_001/input_video.mp4" --prompt "Generate audio for a 10.0s video using this event timeline. Scene: robot arm picking an item in a warehouse. Event timeline: 0.00s: object=warehouse; action=warehouse_hum; sound=low warehouse ventilation hum; layer=foley; priority=1; tolerance=600ms | 0.00s: object=servo; action=servo_arm_move; sound=precise electric servo arm movement; layer=foley; priority=4; tolerance=180ms | 0.00s: object=box; action=box_lift; sound=cardboard box lifted from shelf; layer=foley; priority=5; tolerance=150ms | 0.00s: object=confirm; action=confirm_beep; sound=short clean robot confirmation beep; layer=foley; priority=3; tolerance=80ms. Preserve temporal order. Align transient sounds to event timestamps. Do not add: human speech, music, explosion, vehicle horn, speech, lyrics, voiceover." --output="$MAINBASE/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/dss_event_timeline/mmaudio_small_44k_raw_output" --seed=42
  NEW_AUDIO=$(find "$MAINBASE/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/dss_event_timeline/mmaudio_small_44k_raw_output" -type f \( -name '*.flac' -o -name '*.wav' \) -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2- || true)
  if [ -z "$NEW_AUDIO" ]; then echo "NO_MMAUDIO_OUTPUT_FOUND"; exit 2; fi
  cp "$NEW_AUDIO" "$MAINBASE/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/dss_event_timeline/w18_019_robot_warehouse_pick_001_dss_event_timeline.flac"
  if command -v ffmpeg >/dev/null 2>&1; then ffmpeg -y -i "$MAINBASE/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/dss_event_timeline/w18_019_robot_warehouse_pick_001_dss_event_timeline.flac" "$MAINBASE/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/dss_event_timeline/w18_019_robot_warehouse_pick_001_dss_event_timeline.wav" >/dev/null 2>&1 || true; fi
fi
ls -lh "$MAINBASE/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/dss_event_timeline/w18_019_robot_warehouse_pick_001_dss_event_timeline.flac" "$MAINBASE/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/dss_event_timeline/w18_019_robot_warehouse_pick_001_dss_event_timeline.wav" 2>/dev/null || true

echo '### RUN w18_020_robot_warehouse_pick_001_dss_layer_avoid duration=10s variant=dss_layer_avoid'
mkdir -p "$MAINBASE/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/dss_layer_avoid" "$MAINBASE/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/dss_layer_avoid/mmaudio_small_44k_raw_output"
cd "$MMAUDIO_ROOT"
test -f weights/mmaudio_small_44k.pth
test -f ext_weights/synchformer_state_dict.pth
test -f ext_weights/v1-44.pth
if [ -f "$MAINBASE/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/dss_layer_avoid/w18_020_robot_warehouse_pick_001_dss_layer_avoid.flac" ] || [ -f "$MAINBASE/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/dss_layer_avoid/w18_020_robot_warehouse_pick_001_dss_layer_avoid.wav" ]; then echo 'SKIP existing output w18_020_robot_warehouse_pick_001_dss_layer_avoid'; else
  python demo.py --variant small_44k --duration=10 --video="$MAINBASE/cases/robot_warehouse_pick_001/input_video.mp4" --prompt "Generate layered video-synchronized audio for a 10.0s video. Scene layer: robot arm picking an item in a warehouse. Foley layer must emphasize these events: 0.00s: object=warehouse; action=warehouse_hum; sound=low warehouse ventilation hum; layer=foley; priority=1; tolerance=600ms | 0.00s: object=servo; action=servo_arm_move; sound=precise electric servo arm movement; layer=foley; priority=4; tolerance=180ms | 0.00s: object=box; action=box_lift; sound=cardboard box lifted from shelf; layer=foley; priority=5; tolerance=150ms | 0.00s: object=confirm; action=confirm_beep; sound=short clean robot confirmation beep; layer=foley; priority=3; tolerance=80ms. Ambience layer should remain low, continuous, and non-dominant. Music layer should be absent unless explicitly required. Forbidden content: human speech, music, explosion, vehicle horn, speech, lyrics, voiceover. No speech, no lyrics, no unrelated background music." --output="$MAINBASE/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/dss_layer_avoid/mmaudio_small_44k_raw_output" --seed=42
  NEW_AUDIO=$(find "$MAINBASE/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/dss_layer_avoid/mmaudio_small_44k_raw_output" -type f \( -name '*.flac' -o -name '*.wav' \) -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2- || true)
  if [ -z "$NEW_AUDIO" ]; then echo "NO_MMAUDIO_OUTPUT_FOUND"; exit 2; fi
  cp "$NEW_AUDIO" "$MAINBASE/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/dss_layer_avoid/w18_020_robot_warehouse_pick_001_dss_layer_avoid.flac"
  if command -v ffmpeg >/dev/null 2>&1; then ffmpeg -y -i "$MAINBASE/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/dss_layer_avoid/w18_020_robot_warehouse_pick_001_dss_layer_avoid.flac" "$MAINBASE/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/dss_layer_avoid/w18_020_robot_warehouse_pick_001_dss_layer_avoid.wav" >/dev/null 2>&1 || true; fi
fi
ls -lh "$MAINBASE/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/dss_layer_avoid/w18_020_robot_warehouse_pick_001_dss_layer_avoid.flac" "$MAINBASE/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/dss_layer_avoid/w18_020_robot_warehouse_pick_001_dss_layer_avoid.wav" 2>/dev/null || true

