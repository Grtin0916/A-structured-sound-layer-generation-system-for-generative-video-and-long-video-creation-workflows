#!/usr/bin/env bash
set -euo pipefail
MAINBASE="${MAINBASE:-$HOME/work/audio_engineering_repo_skeleton_v1}"
MMAUDIO_ROOT="${MMAUDIO_ROOT:-/home/GRT/work/_external/MMAudio}"
HF_HOME="${HF_HOME:-$MMAUDIO_ROOT/.hf_cache}"
HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export MAINBASE MMAUDIO_ROOT HF_HOME HUGGINGFACE_HUB_CACHE
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

(
  set -euo pipefail
  echo '### RUN w18_001_forest_bird_branch_001_naive duration=8s variant=naive'
  mkdir -p '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/naive' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/naive/mmaudio_small_44k_raw_output'
  cd '${MMAUDIO_ROOT:?set MMAUDIO_ROOT}'
  test -f weights/mmaudio_small_44k.pth
  test -f ext_weights/synchformer_state_dict.pth
  test -f ext_weights/v1-44.pth
  python demo.py --variant small_44k --duration=8 --video='${MAINBASE:?set MAINBASE}/cases/forest_bird_branch_001/input_video.mp4' --prompt 'Create realistic sound effects for this short video. Scene: forest close-up with a bird moving on a branch. Use natural ambience and match visible actions. Avoid human speech, heavy wind, music bed, urban traffic, speech, lyrics, voiceover.' --output='${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/naive/mmaudio_small_44k_raw_output' --seed=42
  NEW_AUDIO=$(find '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/naive/mmaudio_small_44k_raw_output' -type f \( -name '*.flac' -o -name '*.wav' \) -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2- || true)
  if [ -z "$NEW_AUDIO" ]; then echo "NO_MMAUDIO_OUTPUT_FOUND"; exit 2; fi
  case "$NEW_AUDIO" in
    *.flac) cp "$NEW_AUDIO" '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/naive/w18_001_forest_bird_branch_001_naive.flac' ;;
    *.wav) cp "$NEW_AUDIO" '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/naive/w18_001_forest_bird_branch_001_naive.wav' ;;
    *) echo "UNKNOWN_AUDIO_EXT=$NEW_AUDIO"; exit 3 ;;
  esac
  if [ -f '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/naive/w18_001_forest_bird_branch_001_naive.flac' ] && command -v ffmpeg >/dev/null 2>&1; then ffmpeg -y -i '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/naive/w18_001_forest_bird_branch_001_naive.flac' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/naive/w18_001_forest_bird_branch_001_naive.wav' >/dev/null 2>&1 || true; fi
  ls -lh '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/naive/w18_001_forest_bird_branch_001_naive.flac' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/naive/w18_001_forest_bird_branch_001_naive.wav' 2>/dev/null || true
)

(
  set -euo pipefail
  echo '### RUN w18_002_forest_bird_branch_001_naive_rich duration=8s variant=naive_rich'
  mkdir -p '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/naive_rich' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/naive_rich/mmaudio_small_44k_raw_output'
  cd '${MMAUDIO_ROOT:?set MMAUDIO_ROOT}'
  test -f weights/mmaudio_small_44k.pth
  test -f ext_weights/synchformer_state_dict.pth
  test -f ext_weights/v1-44.pth
  python demo.py --variant small_44k --duration=8 --video='${MAINBASE:?set MAINBASE}/cases/forest_bird_branch_001/input_video.mp4' --prompt 'Create realistic sound effects for this short video. Scene: forest close-up with a bird moving on a branch. The visible actions include: around 0.00s, forest forest_wind with soft wind through leaves; around 0.00s, bird bird_chirp with two short bird chirps; around 0.00s, leaf leaf_rustle with small animal or bird rustling leaves; around 0.00s, branch branch_crack with dry branch crack. Make the sounds natural, synchronized, and cinematic. Avoid human speech, heavy wind, music bed, urban traffic, speech, lyrics, voiceover.' --output='${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/naive_rich/mmaudio_small_44k_raw_output' --seed=42
  NEW_AUDIO=$(find '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/naive_rich/mmaudio_small_44k_raw_output' -type f \( -name '*.flac' -o -name '*.wav' \) -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2- || true)
  if [ -z "$NEW_AUDIO" ]; then echo "NO_MMAUDIO_OUTPUT_FOUND"; exit 2; fi
  case "$NEW_AUDIO" in
    *.flac) cp "$NEW_AUDIO" '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/naive_rich/w18_002_forest_bird_branch_001_naive_rich.flac' ;;
    *.wav) cp "$NEW_AUDIO" '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/naive_rich/w18_002_forest_bird_branch_001_naive_rich.wav' ;;
    *) echo "UNKNOWN_AUDIO_EXT=$NEW_AUDIO"; exit 3 ;;
  esac
  if [ -f '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/naive_rich/w18_002_forest_bird_branch_001_naive_rich.flac' ] && command -v ffmpeg >/dev/null 2>&1; then ffmpeg -y -i '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/naive_rich/w18_002_forest_bird_branch_001_naive_rich.flac' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/naive_rich/w18_002_forest_bird_branch_001_naive_rich.wav' >/dev/null 2>&1 || true; fi
  ls -lh '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/naive_rich/w18_002_forest_bird_branch_001_naive_rich.flac' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/naive_rich/w18_002_forest_bird_branch_001_naive_rich.wav' 2>/dev/null || true
)

(
  set -euo pipefail
  echo '### RUN w18_003_forest_bird_branch_001_dss_global duration=8s variant=dss_global'
  mkdir -p '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/dss_global' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/dss_global/mmaudio_small_44k_raw_output'
  cd '${MMAUDIO_ROOT:?set MMAUDIO_ROOT}'
  test -f weights/mmaudio_small_44k.pth
  test -f ext_weights/synchformer_state_dict.pth
  test -f ext_weights/v1-44.pth
  python demo.py --variant small_44k --duration=8 --video='${MAINBASE:?set MAINBASE}/cases/forest_bird_branch_001/input_video.mp4' --prompt 'Generate synchronized audio for a 8.0s video. Global scene: forest close-up with a bird moving on a branch. Overall style: realistic Foley, no speech, no background music. Keep ambience stable, keep foley clear, avoid forbidden content: human speech, heavy wind, music bed, urban traffic, speech, lyrics, voiceover.' --output='${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/dss_global/mmaudio_small_44k_raw_output' --seed=42
  NEW_AUDIO=$(find '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/dss_global/mmaudio_small_44k_raw_output' -type f \( -name '*.flac' -o -name '*.wav' \) -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2- || true)
  if [ -z "$NEW_AUDIO" ]; then echo "NO_MMAUDIO_OUTPUT_FOUND"; exit 2; fi
  case "$NEW_AUDIO" in
    *.flac) cp "$NEW_AUDIO" '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/dss_global/w18_003_forest_bird_branch_001_dss_global.flac' ;;
    *.wav) cp "$NEW_AUDIO" '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/dss_global/w18_003_forest_bird_branch_001_dss_global.wav' ;;
    *) echo "UNKNOWN_AUDIO_EXT=$NEW_AUDIO"; exit 3 ;;
  esac
  if [ -f '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/dss_global/w18_003_forest_bird_branch_001_dss_global.flac' ] && command -v ffmpeg >/dev/null 2>&1; then ffmpeg -y -i '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/dss_global/w18_003_forest_bird_branch_001_dss_global.flac' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/dss_global/w18_003_forest_bird_branch_001_dss_global.wav' >/dev/null 2>&1 || true; fi
  ls -lh '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/dss_global/w18_003_forest_bird_branch_001_dss_global.flac' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/dss_global/w18_003_forest_bird_branch_001_dss_global.wav' 2>/dev/null || true
)

(
  set -euo pipefail
  echo '### RUN w18_004_forest_bird_branch_001_dss_event_timeline duration=8s variant=dss_event_timeline'
  mkdir -p '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/dss_event_timeline' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/dss_event_timeline/mmaudio_small_44k_raw_output'
  cd '${MMAUDIO_ROOT:?set MMAUDIO_ROOT}'
  test -f weights/mmaudio_small_44k.pth
  test -f ext_weights/synchformer_state_dict.pth
  test -f ext_weights/v1-44.pth
  python demo.py --variant small_44k --duration=8 --video='${MAINBASE:?set MAINBASE}/cases/forest_bird_branch_001/input_video.mp4' --prompt 'Generate audio for a 8.0s video using this event timeline. Scene: forest close-up with a bird moving on a branch. Event timeline: 0.00s: object=forest; action=forest_wind; sound=soft wind through leaves; layer=foley; priority=2; tolerance=500ms | 0.00s: object=bird; action=bird_chirp; sound=two short bird chirps; layer=foley; priority=3; tolerance=150ms | 0.00s: object=leaf; action=leaf_rustle; sound=small animal or bird rustling leaves; layer=foley; priority=4; tolerance=180ms | 0.00s: object=branch; action=branch_crack; sound=dry branch crack; layer=foley; priority=5; tolerance=100ms. Preserve temporal order. Align transient sounds to event timestamps. Do not add: human speech, heavy wind, music bed, urban traffic, speech, lyrics, voiceover.' --output='${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/dss_event_timeline/mmaudio_small_44k_raw_output' --seed=42
  NEW_AUDIO=$(find '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/dss_event_timeline/mmaudio_small_44k_raw_output' -type f \( -name '*.flac' -o -name '*.wav' \) -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2- || true)
  if [ -z "$NEW_AUDIO" ]; then echo "NO_MMAUDIO_OUTPUT_FOUND"; exit 2; fi
  case "$NEW_AUDIO" in
    *.flac) cp "$NEW_AUDIO" '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/dss_event_timeline/w18_004_forest_bird_branch_001_dss_event_timeline.flac' ;;
    *.wav) cp "$NEW_AUDIO" '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/dss_event_timeline/w18_004_forest_bird_branch_001_dss_event_timeline.wav' ;;
    *) echo "UNKNOWN_AUDIO_EXT=$NEW_AUDIO"; exit 3 ;;
  esac
  if [ -f '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/dss_event_timeline/w18_004_forest_bird_branch_001_dss_event_timeline.flac' ] && command -v ffmpeg >/dev/null 2>&1; then ffmpeg -y -i '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/dss_event_timeline/w18_004_forest_bird_branch_001_dss_event_timeline.flac' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/dss_event_timeline/w18_004_forest_bird_branch_001_dss_event_timeline.wav' >/dev/null 2>&1 || true; fi
  ls -lh '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/dss_event_timeline/w18_004_forest_bird_branch_001_dss_event_timeline.flac' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/dss_event_timeline/w18_004_forest_bird_branch_001_dss_event_timeline.wav' 2>/dev/null || true
)

(
  set -euo pipefail
  echo '### RUN w18_005_forest_bird_branch_001_dss_layer_avoid duration=8s variant=dss_layer_avoid'
  mkdir -p '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/dss_layer_avoid' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/dss_layer_avoid/mmaudio_small_44k_raw_output'
  cd '${MMAUDIO_ROOT:?set MMAUDIO_ROOT}'
  test -f weights/mmaudio_small_44k.pth
  test -f ext_weights/synchformer_state_dict.pth
  test -f ext_weights/v1-44.pth
  python demo.py --variant small_44k --duration=8 --video='${MAINBASE:?set MAINBASE}/cases/forest_bird_branch_001/input_video.mp4' --prompt 'Generate layered video-synchronized audio. Scene layer: forest close-up with a bird moving on a branch. Foley layer must emphasize these events: 0.00s: object=forest; action=forest_wind; sound=soft wind through leaves; layer=foley; priority=2; tolerance=500ms | 0.00s: object=bird; action=bird_chirp; sound=two short bird chirps; layer=foley; priority=3; tolerance=150ms | 0.00s: object=leaf; action=leaf_rustle; sound=small animal or bird rustling leaves; layer=foley; priority=4; tolerance=180ms | 0.00s: object=branch; action=branch_crack; sound=dry branch crack; layer=foley; priority=5; tolerance=100ms. Ambience layer should remain low, continuous, and non-dominant. Music layer should be absent unless explicitly required. Forbidden content: human speech, heavy wind, music bed, urban traffic, speech, lyrics, voiceover. No speech, no lyrics, no unrelated background music.' --output='${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/dss_layer_avoid/mmaudio_small_44k_raw_output' --seed=42
  NEW_AUDIO=$(find '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/dss_layer_avoid/mmaudio_small_44k_raw_output' -type f \( -name '*.flac' -o -name '*.wav' \) -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2- || true)
  if [ -z "$NEW_AUDIO" ]; then echo "NO_MMAUDIO_OUTPUT_FOUND"; exit 2; fi
  case "$NEW_AUDIO" in
    *.flac) cp "$NEW_AUDIO" '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/dss_layer_avoid/w18_005_forest_bird_branch_001_dss_layer_avoid.flac' ;;
    *.wav) cp "$NEW_AUDIO" '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/dss_layer_avoid/w18_005_forest_bird_branch_001_dss_layer_avoid.wav' ;;
    *) echo "UNKNOWN_AUDIO_EXT=$NEW_AUDIO"; exit 3 ;;
  esac
  if [ -f '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/dss_layer_avoid/w18_005_forest_bird_branch_001_dss_layer_avoid.flac' ] && command -v ffmpeg >/dev/null 2>&1; then ffmpeg -y -i '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/dss_layer_avoid/w18_005_forest_bird_branch_001_dss_layer_avoid.flac' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/dss_layer_avoid/w18_005_forest_bird_branch_001_dss_layer_avoid.wav' >/dev/null 2>&1 || true; fi
  ls -lh '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/dss_layer_avoid/w18_005_forest_bird_branch_001_dss_layer_avoid.flac' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/dss_layer_avoid/w18_005_forest_bird_branch_001_dss_layer_avoid.wav' 2>/dev/null || true
)

(
  set -euo pipefail
  echo '### RUN w18_006_glass_drop_room_001_naive duration=7s variant=naive'
  mkdir -p '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/naive' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/naive/mmaudio_small_44k_raw_output'
  cd '${MMAUDIO_ROOT:?set MMAUDIO_ROOT}'
  test -f weights/mmaudio_small_44k.pth
  test -f ext_weights/synchformer_state_dict.pth
  test -f ext_weights/v1-44.pth
  python demo.py --variant small_44k --duration=7 --video='${MAINBASE:?set MAINBASE}/cases/glass_drop_room_001/input_video.mp4' --prompt 'Create realistic sound effects for this short video. Scene: quiet indoor room, a glass object drops and breaks on a hard floor. Use natural ambience and match visible actions. Avoid speech, music, cartoon boing, long reverb tail, lyrics, voiceover.' --output='${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/naive/mmaudio_small_44k_raw_output' --seed=42
  NEW_AUDIO=$(find '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/naive/mmaudio_small_44k_raw_output' -type f \( -name '*.flac' -o -name '*.wav' \) -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2- || true)
  if [ -z "$NEW_AUDIO" ]; then echo "NO_MMAUDIO_OUTPUT_FOUND"; exit 2; fi
  case "$NEW_AUDIO" in
    *.flac) cp "$NEW_AUDIO" '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/naive/w18_006_glass_drop_room_001_naive.flac' ;;
    *.wav) cp "$NEW_AUDIO" '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/naive/w18_006_glass_drop_room_001_naive.wav' ;;
    *) echo "UNKNOWN_AUDIO_EXT=$NEW_AUDIO"; exit 3 ;;
  esac
  if [ -f '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/naive/w18_006_glass_drop_room_001_naive.flac' ] && command -v ffmpeg >/dev/null 2>&1; then ffmpeg -y -i '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/naive/w18_006_glass_drop_room_001_naive.flac' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/naive/w18_006_glass_drop_room_001_naive.wav' >/dev/null 2>&1 || true; fi
  ls -lh '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/naive/w18_006_glass_drop_room_001_naive.flac' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/naive/w18_006_glass_drop_room_001_naive.wav' 2>/dev/null || true
)

(
  set -euo pipefail
  echo '### RUN w18_007_glass_drop_room_001_naive_rich duration=7s variant=naive_rich'
  mkdir -p '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/naive_rich' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/naive_rich/mmaudio_small_44k_raw_output'
  cd '${MMAUDIO_ROOT:?set MMAUDIO_ROOT}'
  test -f weights/mmaudio_small_44k.pth
  test -f ext_weights/synchformer_state_dict.pth
  test -f ext_weights/v1-44.pth
  python demo.py --variant small_44k --duration=7 --video='${MAINBASE:?set MAINBASE}/cases/glass_drop_room_001/input_video.mp4' --prompt 'Create realistic sound effects for this short video. Scene: quiet indoor room, a glass object drops and breaks on a hard floor. The visible actions include: around 0.00s, quiet quiet_room_tone with quiet indoor room tone; around 0.00s, glass glass_slip with glass sliding off table edge; around 0.00s, impact impact with sharp glass impact on hard floor; around 0.00s, shatter shatter_tail with small glass fragments scattering. Make the sounds natural, synchronized, and cinematic. Avoid speech, music, cartoon boing, long reverb tail, lyrics, voiceover.' --output='${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/naive_rich/mmaudio_small_44k_raw_output' --seed=42
  NEW_AUDIO=$(find '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/naive_rich/mmaudio_small_44k_raw_output' -type f \( -name '*.flac' -o -name '*.wav' \) -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2- || true)
  if [ -z "$NEW_AUDIO" ]; then echo "NO_MMAUDIO_OUTPUT_FOUND"; exit 2; fi
  case "$NEW_AUDIO" in
    *.flac) cp "$NEW_AUDIO" '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/naive_rich/w18_007_glass_drop_room_001_naive_rich.flac' ;;
    *.wav) cp "$NEW_AUDIO" '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/naive_rich/w18_007_glass_drop_room_001_naive_rich.wav' ;;
    *) echo "UNKNOWN_AUDIO_EXT=$NEW_AUDIO"; exit 3 ;;
  esac
  if [ -f '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/naive_rich/w18_007_glass_drop_room_001_naive_rich.flac' ] && command -v ffmpeg >/dev/null 2>&1; then ffmpeg -y -i '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/naive_rich/w18_007_glass_drop_room_001_naive_rich.flac' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/naive_rich/w18_007_glass_drop_room_001_naive_rich.wav' >/dev/null 2>&1 || true; fi
  ls -lh '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/naive_rich/w18_007_glass_drop_room_001_naive_rich.flac' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/naive_rich/w18_007_glass_drop_room_001_naive_rich.wav' 2>/dev/null || true
)

(
  set -euo pipefail
  echo '### RUN w18_008_glass_drop_room_001_dss_global duration=7s variant=dss_global'
  mkdir -p '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/dss_global' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/dss_global/mmaudio_small_44k_raw_output'
  cd '${MMAUDIO_ROOT:?set MMAUDIO_ROOT}'
  test -f weights/mmaudio_small_44k.pth
  test -f ext_weights/synchformer_state_dict.pth
  test -f ext_weights/v1-44.pth
  python demo.py --variant small_44k --duration=7 --video='${MAINBASE:?set MAINBASE}/cases/glass_drop_room_001/input_video.mp4' --prompt 'Generate synchronized audio for a 7.0s video. Global scene: quiet indoor room, a glass object drops and breaks on a hard floor. Overall style: realistic Foley, no speech, no background music. Keep ambience stable, keep foley clear, avoid forbidden content: speech, music, cartoon boing, long reverb tail, lyrics, voiceover.' --output='${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/dss_global/mmaudio_small_44k_raw_output' --seed=42
  NEW_AUDIO=$(find '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/dss_global/mmaudio_small_44k_raw_output' -type f \( -name '*.flac' -o -name '*.wav' \) -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2- || true)
  if [ -z "$NEW_AUDIO" ]; then echo "NO_MMAUDIO_OUTPUT_FOUND"; exit 2; fi
  case "$NEW_AUDIO" in
    *.flac) cp "$NEW_AUDIO" '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/dss_global/w18_008_glass_drop_room_001_dss_global.flac' ;;
    *.wav) cp "$NEW_AUDIO" '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/dss_global/w18_008_glass_drop_room_001_dss_global.wav' ;;
    *) echo "UNKNOWN_AUDIO_EXT=$NEW_AUDIO"; exit 3 ;;
  esac
  if [ -f '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/dss_global/w18_008_glass_drop_room_001_dss_global.flac' ] && command -v ffmpeg >/dev/null 2>&1; then ffmpeg -y -i '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/dss_global/w18_008_glass_drop_room_001_dss_global.flac' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/dss_global/w18_008_glass_drop_room_001_dss_global.wav' >/dev/null 2>&1 || true; fi
  ls -lh '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/dss_global/w18_008_glass_drop_room_001_dss_global.flac' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/dss_global/w18_008_glass_drop_room_001_dss_global.wav' 2>/dev/null || true
)

(
  set -euo pipefail
  echo '### RUN w18_009_glass_drop_room_001_dss_event_timeline duration=7s variant=dss_event_timeline'
  mkdir -p '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/dss_event_timeline' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/dss_event_timeline/mmaudio_small_44k_raw_output'
  cd '${MMAUDIO_ROOT:?set MMAUDIO_ROOT}'
  test -f weights/mmaudio_small_44k.pth
  test -f ext_weights/synchformer_state_dict.pth
  test -f ext_weights/v1-44.pth
  python demo.py --variant small_44k --duration=7 --video='${MAINBASE:?set MAINBASE}/cases/glass_drop_room_001/input_video.mp4' --prompt 'Generate audio for a 7.0s video using this event timeline. Scene: quiet indoor room, a glass object drops and breaks on a hard floor. Event timeline: 0.00s: object=quiet; action=quiet_room_tone; sound=quiet indoor room tone; layer=foley; priority=1; tolerance=500ms | 0.00s: object=glass; action=glass_slip; sound=glass sliding off table edge; layer=foley; priority=4; tolerance=120ms | 0.00s: object=impact; action=impact; sound=sharp glass impact on hard floor; layer=foley; priority=5; tolerance=70ms | 0.00s: object=shatter; action=shatter_tail; sound=small glass fragments scattering; layer=foley; priority=5; tolerance=100ms. Preserve temporal order. Align transient sounds to event timestamps. Do not add: speech, music, cartoon boing, long reverb tail, lyrics, voiceover.' --output='${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/dss_event_timeline/mmaudio_small_44k_raw_output' --seed=42
  NEW_AUDIO=$(find '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/dss_event_timeline/mmaudio_small_44k_raw_output' -type f \( -name '*.flac' -o -name '*.wav' \) -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2- || true)
  if [ -z "$NEW_AUDIO" ]; then echo "NO_MMAUDIO_OUTPUT_FOUND"; exit 2; fi
  case "$NEW_AUDIO" in
    *.flac) cp "$NEW_AUDIO" '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/dss_event_timeline/w18_009_glass_drop_room_001_dss_event_timeline.flac' ;;
    *.wav) cp "$NEW_AUDIO" '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/dss_event_timeline/w18_009_glass_drop_room_001_dss_event_timeline.wav' ;;
    *) echo "UNKNOWN_AUDIO_EXT=$NEW_AUDIO"; exit 3 ;;
  esac
  if [ -f '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/dss_event_timeline/w18_009_glass_drop_room_001_dss_event_timeline.flac' ] && command -v ffmpeg >/dev/null 2>&1; then ffmpeg -y -i '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/dss_event_timeline/w18_009_glass_drop_room_001_dss_event_timeline.flac' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/dss_event_timeline/w18_009_glass_drop_room_001_dss_event_timeline.wav' >/dev/null 2>&1 || true; fi
  ls -lh '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/dss_event_timeline/w18_009_glass_drop_room_001_dss_event_timeline.flac' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/dss_event_timeline/w18_009_glass_drop_room_001_dss_event_timeline.wav' 2>/dev/null || true
)

(
  set -euo pipefail
  echo '### RUN w18_010_glass_drop_room_001_dss_layer_avoid duration=7s variant=dss_layer_avoid'
  mkdir -p '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/dss_layer_avoid' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/dss_layer_avoid/mmaudio_small_44k_raw_output'
  cd '${MMAUDIO_ROOT:?set MMAUDIO_ROOT}'
  test -f weights/mmaudio_small_44k.pth
  test -f ext_weights/synchformer_state_dict.pth
  test -f ext_weights/v1-44.pth
  python demo.py --variant small_44k --duration=7 --video='${MAINBASE:?set MAINBASE}/cases/glass_drop_room_001/input_video.mp4' --prompt 'Generate layered video-synchronized audio. Scene layer: quiet indoor room, a glass object drops and breaks on a hard floor. Foley layer must emphasize these events: 0.00s: object=quiet; action=quiet_room_tone; sound=quiet indoor room tone; layer=foley; priority=1; tolerance=500ms | 0.00s: object=glass; action=glass_slip; sound=glass sliding off table edge; layer=foley; priority=4; tolerance=120ms | 0.00s: object=impact; action=impact; sound=sharp glass impact on hard floor; layer=foley; priority=5; tolerance=70ms | 0.00s: object=shatter; action=shatter_tail; sound=small glass fragments scattering; layer=foley; priority=5; tolerance=100ms. Ambience layer should remain low, continuous, and non-dominant. Music layer should be absent unless explicitly required. Forbidden content: speech, music, cartoon boing, long reverb tail, lyrics, voiceover. No speech, no lyrics, no unrelated background music.' --output='${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/dss_layer_avoid/mmaudio_small_44k_raw_output' --seed=42
  NEW_AUDIO=$(find '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/dss_layer_avoid/mmaudio_small_44k_raw_output' -type f \( -name '*.flac' -o -name '*.wav' \) -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2- || true)
  if [ -z "$NEW_AUDIO" ]; then echo "NO_MMAUDIO_OUTPUT_FOUND"; exit 2; fi
  case "$NEW_AUDIO" in
    *.flac) cp "$NEW_AUDIO" '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/dss_layer_avoid/w18_010_glass_drop_room_001_dss_layer_avoid.flac' ;;
    *.wav) cp "$NEW_AUDIO" '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/dss_layer_avoid/w18_010_glass_drop_room_001_dss_layer_avoid.wav' ;;
    *) echo "UNKNOWN_AUDIO_EXT=$NEW_AUDIO"; exit 3 ;;
  esac
  if [ -f '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/dss_layer_avoid/w18_010_glass_drop_room_001_dss_layer_avoid.flac' ] && command -v ffmpeg >/dev/null 2>&1; then ffmpeg -y -i '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/dss_layer_avoid/w18_010_glass_drop_room_001_dss_layer_avoid.flac' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/dss_layer_avoid/w18_010_glass_drop_room_001_dss_layer_avoid.wav' >/dev/null 2>&1 || true; fi
  ls -lh '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/dss_layer_avoid/w18_010_glass_drop_room_001_dss_layer_avoid.flac' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/glass_drop_room_001/dss_layer_avoid/w18_010_glass_drop_room_001_dss_layer_avoid.wav' 2>/dev/null || true
)

(
  set -euo pipefail
  echo '### RUN w18_011_kitchen_chop_sizzle_001_naive duration=10s variant=naive'
  mkdir -p '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/kitchen_chop_sizzle_001/naive' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/kitchen_chop_sizzle_001/naive/mmaudio_small_44k_raw_output'
  cd '${MMAUDIO_ROOT:?set MMAUDIO_ROOT}'
  test -f weights/mmaudio_small_44k.pth
  test -f ext_weights/synchformer_state_dict.pth
  test -f ext_weights/v1-44.pth
  python demo.py --variant small_44k --duration=10 --video='${MAINBASE:?set MAINBASE}/cases/kitchen_chop_sizzle_001/input_video.mp4' --prompt 'Create realistic sound effects for this short video. Scene: kitchen preparation with chopping and pan sizzle. Use natural ambience and match visible actions. Avoid speech, restaurant crowd, music, alarm beep, lyrics, voiceover.' --output='${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/kitchen_chop_sizzle_001/naive/mmaudio_small_44k_raw_output' --seed=42
  NEW_AUDIO=$(find '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/kitchen_chop_sizzle_001/naive/mmaudio_small_44k_raw_output' -type f \( -name '*.flac' -o -name '*.wav' \) -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2- || true)
  if [ -z "$NEW_AUDIO" ]; then echo "NO_MMAUDIO_OUTPUT_FOUND"; exit 2; fi
  case "$NEW_AUDIO" in
    *.flac) cp "$NEW_AUDIO" '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/kitchen_chop_sizzle_001/naive/w18_011_kitchen_chop_sizzle_001_naive.flac' ;;
    *.wav) cp "$NEW_AUDIO" '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/kitchen_chop_sizzle_001/naive/w18_011_kitchen_chop_sizzle_001_naive.wav' ;;
    *) echo "UNKNOWN_AUDIO_EXT=$NEW_AUDIO"; exit 3 ;;
  esac
  if [ -f '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/kitchen_chop_sizzle_001/naive/w18_011_kitchen_chop_sizzle_001_naive.flac' ] && command -v ffmpeg >/dev/null 2>&1; then ffmpeg -y -i '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/kitchen_chop_sizzle_001/naive/w18_011_kitchen_chop_sizzle_001_naive.flac' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/kitchen_chop_sizzle_001/naive/w18_011_kitchen_chop_sizzle_001_naive.wav' >/dev/null 2>&1 || true; fi
  ls -lh '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/kitchen_chop_sizzle_001/naive/w18_011_kitchen_chop_sizzle_001_naive.flac' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/kitchen_chop_sizzle_001/naive/w18_011_kitchen_chop_sizzle_001_naive.wav' 2>/dev/null || true
)

(
  set -euo pipefail
  echo '### RUN w18_012_kitchen_chop_sizzle_001_naive_rich duration=10s variant=naive_rich'
  mkdir -p '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/kitchen_chop_sizzle_001/naive_rich' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/kitchen_chop_sizzle_001/naive_rich/mmaudio_small_44k_raw_output'
  cd '${MMAUDIO_ROOT:?set MMAUDIO_ROOT}'
  test -f weights/mmaudio_small_44k.pth
  test -f ext_weights/synchformer_state_dict.pth
  test -f ext_weights/v1-44.pth
  python demo.py --variant small_44k --duration=10 --video='${MAINBASE:?set MAINBASE}/cases/kitchen_chop_sizzle_001/input_video.mp4' --prompt 'Create realistic sound effects for this short video. Scene: kitchen preparation with chopping and pan sizzle. The visible actions include: around 0.00s, room room_tone with small indoor kitchen room tone; around 0.00s, knife knife_chops with rhythmic vegetable chopping on wooden board; around 0.00s, pan pan_sizzle_rise with oil and food sizzling in pan; around 0.00s, plate plate_set_down with ceramic plate placed on counter. Make the sounds natural, synchronized, and cinematic. Avoid speech, restaurant crowd, music, alarm beep, lyrics, voiceover.' --output='${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/kitchen_chop_sizzle_001/naive_rich/mmaudio_small_44k_raw_output' --seed=42
  NEW_AUDIO=$(find '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/kitchen_chop_sizzle_001/naive_rich/mmaudio_small_44k_raw_output' -type f \( -name '*.flac' -o -name '*.wav' \) -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2- || true)
  if [ -z "$NEW_AUDIO" ]; then echo "NO_MMAUDIO_OUTPUT_FOUND"; exit 2; fi
  case "$NEW_AUDIO" in
    *.flac) cp "$NEW_AUDIO" '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/kitchen_chop_sizzle_001/naive_rich/w18_012_kitchen_chop_sizzle_001_naive_rich.flac' ;;
    *.wav) cp "$NEW_AUDIO" '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/kitchen_chop_sizzle_001/naive_rich/w18_012_kitchen_chop_sizzle_001_naive_rich.wav' ;;
    *) echo "UNKNOWN_AUDIO_EXT=$NEW_AUDIO"; exit 3 ;;
  esac
  if [ -f '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/kitchen_chop_sizzle_001/naive_rich/w18_012_kitchen_chop_sizzle_001_naive_rich.flac' ] && command -v ffmpeg >/dev/null 2>&1; then ffmpeg -y -i '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/kitchen_chop_sizzle_001/naive_rich/w18_012_kitchen_chop_sizzle_001_naive_rich.flac' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/kitchen_chop_sizzle_001/naive_rich/w18_012_kitchen_chop_sizzle_001_naive_rich.wav' >/dev/null 2>&1 || true; fi
  ls -lh '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/kitchen_chop_sizzle_001/naive_rich/w18_012_kitchen_chop_sizzle_001_naive_rich.flac' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/kitchen_chop_sizzle_001/naive_rich/w18_012_kitchen_chop_sizzle_001_naive_rich.wav' 2>/dev/null || true
)

(
  set -euo pipefail
  echo '### RUN w18_013_kitchen_chop_sizzle_001_dss_global duration=10s variant=dss_global'
  mkdir -p '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/kitchen_chop_sizzle_001/dss_global' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/kitchen_chop_sizzle_001/dss_global/mmaudio_small_44k_raw_output'
  cd '${MMAUDIO_ROOT:?set MMAUDIO_ROOT}'
  test -f weights/mmaudio_small_44k.pth
  test -f ext_weights/synchformer_state_dict.pth
  test -f ext_weights/v1-44.pth
  python demo.py --variant small_44k --duration=10 --video='${MAINBASE:?set MAINBASE}/cases/kitchen_chop_sizzle_001/input_video.mp4' --prompt 'Generate synchronized audio for a 10.0s video. Global scene: kitchen preparation with chopping and pan sizzle. Overall style: realistic Foley, no speech, no background music. Keep ambience stable, keep foley clear, avoid forbidden content: speech, restaurant crowd, music, alarm beep, lyrics, voiceover.' --output='${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/kitchen_chop_sizzle_001/dss_global/mmaudio_small_44k_raw_output' --seed=42
  NEW_AUDIO=$(find '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/kitchen_chop_sizzle_001/dss_global/mmaudio_small_44k_raw_output' -type f \( -name '*.flac' -o -name '*.wav' \) -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2- || true)
  if [ -z "$NEW_AUDIO" ]; then echo "NO_MMAUDIO_OUTPUT_FOUND"; exit 2; fi
  case "$NEW_AUDIO" in
    *.flac) cp "$NEW_AUDIO" '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/kitchen_chop_sizzle_001/dss_global/w18_013_kitchen_chop_sizzle_001_dss_global.flac' ;;
    *.wav) cp "$NEW_AUDIO" '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/kitchen_chop_sizzle_001/dss_global/w18_013_kitchen_chop_sizzle_001_dss_global.wav' ;;
    *) echo "UNKNOWN_AUDIO_EXT=$NEW_AUDIO"; exit 3 ;;
  esac
  if [ -f '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/kitchen_chop_sizzle_001/dss_global/w18_013_kitchen_chop_sizzle_001_dss_global.flac' ] && command -v ffmpeg >/dev/null 2>&1; then ffmpeg -y -i '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/kitchen_chop_sizzle_001/dss_global/w18_013_kitchen_chop_sizzle_001_dss_global.flac' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/kitchen_chop_sizzle_001/dss_global/w18_013_kitchen_chop_sizzle_001_dss_global.wav' >/dev/null 2>&1 || true; fi
  ls -lh '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/kitchen_chop_sizzle_001/dss_global/w18_013_kitchen_chop_sizzle_001_dss_global.flac' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/kitchen_chop_sizzle_001/dss_global/w18_013_kitchen_chop_sizzle_001_dss_global.wav' 2>/dev/null || true
)

(
  set -euo pipefail
  echo '### RUN w18_014_kitchen_chop_sizzle_001_dss_event_timeline duration=10s variant=dss_event_timeline'
  mkdir -p '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/kitchen_chop_sizzle_001/dss_event_timeline' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/kitchen_chop_sizzle_001/dss_event_timeline/mmaudio_small_44k_raw_output'
  cd '${MMAUDIO_ROOT:?set MMAUDIO_ROOT}'
  test -f weights/mmaudio_small_44k.pth
  test -f ext_weights/synchformer_state_dict.pth
  test -f ext_weights/v1-44.pth
  python demo.py --variant small_44k --duration=10 --video='${MAINBASE:?set MAINBASE}/cases/kitchen_chop_sizzle_001/input_video.mp4' --prompt 'Generate audio for a 10.0s video using this event timeline. Scene: kitchen preparation with chopping and pan sizzle. Event timeline: 0.00s: object=room; action=room_tone; sound=small indoor kitchen room tone; layer=foley; priority=1; tolerance=500ms | 0.00s: object=knife; action=knife_chops; sound=rhythmic vegetable chopping on wooden board; layer=foley; priority=5; tolerance=100ms | 0.00s: object=pan; action=pan_sizzle_rise; sound=oil and food sizzling in pan; layer=foley; priority=4; tolerance=250ms | 0.00s: object=plate; action=plate_set_down; sound=ceramic plate placed on counter; layer=foley; priority=3; tolerance=120ms. Preserve temporal order. Align transient sounds to event timestamps. Do not add: speech, restaurant crowd, music, alarm beep, lyrics, voiceover.' --output='${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/kitchen_chop_sizzle_001/dss_event_timeline/mmaudio_small_44k_raw_output' --seed=42
  NEW_AUDIO=$(find '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/kitchen_chop_sizzle_001/dss_event_timeline/mmaudio_small_44k_raw_output' -type f \( -name '*.flac' -o -name '*.wav' \) -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2- || true)
  if [ -z "$NEW_AUDIO" ]; then echo "NO_MMAUDIO_OUTPUT_FOUND"; exit 2; fi
  case "$NEW_AUDIO" in
    *.flac) cp "$NEW_AUDIO" '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/kitchen_chop_sizzle_001/dss_event_timeline/w18_014_kitchen_chop_sizzle_001_dss_event_timeline.flac' ;;
    *.wav) cp "$NEW_AUDIO" '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/kitchen_chop_sizzle_001/dss_event_timeline/w18_014_kitchen_chop_sizzle_001_dss_event_timeline.wav' ;;
    *) echo "UNKNOWN_AUDIO_EXT=$NEW_AUDIO"; exit 3 ;;
  esac
  if [ -f '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/kitchen_chop_sizzle_001/dss_event_timeline/w18_014_kitchen_chop_sizzle_001_dss_event_timeline.flac' ] && command -v ffmpeg >/dev/null 2>&1; then ffmpeg -y -i '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/kitchen_chop_sizzle_001/dss_event_timeline/w18_014_kitchen_chop_sizzle_001_dss_event_timeline.flac' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/kitchen_chop_sizzle_001/dss_event_timeline/w18_014_kitchen_chop_sizzle_001_dss_event_timeline.wav' >/dev/null 2>&1 || true; fi
  ls -lh '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/kitchen_chop_sizzle_001/dss_event_timeline/w18_014_kitchen_chop_sizzle_001_dss_event_timeline.flac' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/kitchen_chop_sizzle_001/dss_event_timeline/w18_014_kitchen_chop_sizzle_001_dss_event_timeline.wav' 2>/dev/null || true
)

(
  set -euo pipefail
  echo '### RUN w18_015_kitchen_chop_sizzle_001_dss_layer_avoid duration=10s variant=dss_layer_avoid'
  mkdir -p '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/kitchen_chop_sizzle_001/dss_layer_avoid' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/kitchen_chop_sizzle_001/dss_layer_avoid/mmaudio_small_44k_raw_output'
  cd '${MMAUDIO_ROOT:?set MMAUDIO_ROOT}'
  test -f weights/mmaudio_small_44k.pth
  test -f ext_weights/synchformer_state_dict.pth
  test -f ext_weights/v1-44.pth
  python demo.py --variant small_44k --duration=10 --video='${MAINBASE:?set MAINBASE}/cases/kitchen_chop_sizzle_001/input_video.mp4' --prompt 'Generate layered video-synchronized audio. Scene layer: kitchen preparation with chopping and pan sizzle. Foley layer must emphasize these events: 0.00s: object=room; action=room_tone; sound=small indoor kitchen room tone; layer=foley; priority=1; tolerance=500ms | 0.00s: object=knife; action=knife_chops; sound=rhythmic vegetable chopping on wooden board; layer=foley; priority=5; tolerance=100ms | 0.00s: object=pan; action=pan_sizzle_rise; sound=oil and food sizzling in pan; layer=foley; priority=4; tolerance=250ms | 0.00s: object=plate; action=plate_set_down; sound=ceramic plate placed on counter; layer=foley; priority=3; tolerance=120ms. Ambience layer should remain low, continuous, and non-dominant. Music layer should be absent unless explicitly required. Forbidden content: speech, restaurant crowd, music, alarm beep, lyrics, voiceover. No speech, no lyrics, no unrelated background music.' --output='${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/kitchen_chop_sizzle_001/dss_layer_avoid/mmaudio_small_44k_raw_output' --seed=42
  NEW_AUDIO=$(find '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/kitchen_chop_sizzle_001/dss_layer_avoid/mmaudio_small_44k_raw_output' -type f \( -name '*.flac' -o -name '*.wav' \) -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2- || true)
  if [ -z "$NEW_AUDIO" ]; then echo "NO_MMAUDIO_OUTPUT_FOUND"; exit 2; fi
  case "$NEW_AUDIO" in
    *.flac) cp "$NEW_AUDIO" '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/kitchen_chop_sizzle_001/dss_layer_avoid/w18_015_kitchen_chop_sizzle_001_dss_layer_avoid.flac' ;;
    *.wav) cp "$NEW_AUDIO" '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/kitchen_chop_sizzle_001/dss_layer_avoid/w18_015_kitchen_chop_sizzle_001_dss_layer_avoid.wav' ;;
    *) echo "UNKNOWN_AUDIO_EXT=$NEW_AUDIO"; exit 3 ;;
  esac
  if [ -f '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/kitchen_chop_sizzle_001/dss_layer_avoid/w18_015_kitchen_chop_sizzle_001_dss_layer_avoid.flac' ] && command -v ffmpeg >/dev/null 2>&1; then ffmpeg -y -i '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/kitchen_chop_sizzle_001/dss_layer_avoid/w18_015_kitchen_chop_sizzle_001_dss_layer_avoid.flac' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/kitchen_chop_sizzle_001/dss_layer_avoid/w18_015_kitchen_chop_sizzle_001_dss_layer_avoid.wav' >/dev/null 2>&1 || true; fi
  ls -lh '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/kitchen_chop_sizzle_001/dss_layer_avoid/w18_015_kitchen_chop_sizzle_001_dss_layer_avoid.flac' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/kitchen_chop_sizzle_001/dss_layer_avoid/w18_015_kitchen_chop_sizzle_001_dss_layer_avoid.wav' 2>/dev/null || true
)

(
  set -euo pipefail
  echo '### RUN w18_016_robot_warehouse_pick_001_naive duration=10s variant=naive'
  mkdir -p '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/naive' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/naive/mmaudio_small_44k_raw_output'
  cd '${MMAUDIO_ROOT:?set MMAUDIO_ROOT}'
  test -f weights/mmaudio_small_44k.pth
  test -f ext_weights/synchformer_state_dict.pth
  test -f ext_weights/v1-44.pth
  python demo.py --variant small_44k --duration=10 --video='${MAINBASE:?set MAINBASE}/cases/robot_warehouse_pick_001/input_video.mp4' --prompt 'Create realistic sound effects for this short video. Scene: robot arm picking an item in a warehouse. Use natural ambience and match visible actions. Avoid human speech, music, explosion, vehicle horn, speech, lyrics, voiceover.' --output='${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/naive/mmaudio_small_44k_raw_output' --seed=42
  NEW_AUDIO=$(find '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/naive/mmaudio_small_44k_raw_output' -type f \( -name '*.flac' -o -name '*.wav' \) -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2- || true)
  if [ -z "$NEW_AUDIO" ]; then echo "NO_MMAUDIO_OUTPUT_FOUND"; exit 2; fi
  case "$NEW_AUDIO" in
    *.flac) cp "$NEW_AUDIO" '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/naive/w18_016_robot_warehouse_pick_001_naive.flac' ;;
    *.wav) cp "$NEW_AUDIO" '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/naive/w18_016_robot_warehouse_pick_001_naive.wav' ;;
    *) echo "UNKNOWN_AUDIO_EXT=$NEW_AUDIO"; exit 3 ;;
  esac
  if [ -f '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/naive/w18_016_robot_warehouse_pick_001_naive.flac' ] && command -v ffmpeg >/dev/null 2>&1; then ffmpeg -y -i '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/naive/w18_016_robot_warehouse_pick_001_naive.flac' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/naive/w18_016_robot_warehouse_pick_001_naive.wav' >/dev/null 2>&1 || true; fi
  ls -lh '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/naive/w18_016_robot_warehouse_pick_001_naive.flac' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/naive/w18_016_robot_warehouse_pick_001_naive.wav' 2>/dev/null || true
)

(
  set -euo pipefail
  echo '### RUN w18_017_robot_warehouse_pick_001_naive_rich duration=10s variant=naive_rich'
  mkdir -p '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/naive_rich' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/naive_rich/mmaudio_small_44k_raw_output'
  cd '${MMAUDIO_ROOT:?set MMAUDIO_ROOT}'
  test -f weights/mmaudio_small_44k.pth
  test -f ext_weights/synchformer_state_dict.pth
  test -f ext_weights/v1-44.pth
  python demo.py --variant small_44k --duration=10 --video='${MAINBASE:?set MAINBASE}/cases/robot_warehouse_pick_001/input_video.mp4' --prompt 'Create realistic sound effects for this short video. Scene: robot arm picking an item in a warehouse. The visible actions include: around 0.00s, warehouse warehouse_hum with low warehouse ventilation hum; around 0.00s, servo servo_arm_move with precise electric servo arm movement; around 0.00s, box box_lift with cardboard box lifted from shelf; around 0.00s, confirm confirm_beep with short clean robot confirmation beep. Make the sounds natural, synchronized, and cinematic. Avoid human speech, music, explosion, vehicle horn, speech, lyrics, voiceover.' --output='${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/naive_rich/mmaudio_small_44k_raw_output' --seed=42
  NEW_AUDIO=$(find '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/naive_rich/mmaudio_small_44k_raw_output' -type f \( -name '*.flac' -o -name '*.wav' \) -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2- || true)
  if [ -z "$NEW_AUDIO" ]; then echo "NO_MMAUDIO_OUTPUT_FOUND"; exit 2; fi
  case "$NEW_AUDIO" in
    *.flac) cp "$NEW_AUDIO" '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/naive_rich/w18_017_robot_warehouse_pick_001_naive_rich.flac' ;;
    *.wav) cp "$NEW_AUDIO" '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/naive_rich/w18_017_robot_warehouse_pick_001_naive_rich.wav' ;;
    *) echo "UNKNOWN_AUDIO_EXT=$NEW_AUDIO"; exit 3 ;;
  esac
  if [ -f '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/naive_rich/w18_017_robot_warehouse_pick_001_naive_rich.flac' ] && command -v ffmpeg >/dev/null 2>&1; then ffmpeg -y -i '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/naive_rich/w18_017_robot_warehouse_pick_001_naive_rich.flac' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/naive_rich/w18_017_robot_warehouse_pick_001_naive_rich.wav' >/dev/null 2>&1 || true; fi
  ls -lh '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/naive_rich/w18_017_robot_warehouse_pick_001_naive_rich.flac' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/naive_rich/w18_017_robot_warehouse_pick_001_naive_rich.wav' 2>/dev/null || true
)

(
  set -euo pipefail
  echo '### RUN w18_018_robot_warehouse_pick_001_dss_global duration=10s variant=dss_global'
  mkdir -p '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/dss_global' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/dss_global/mmaudio_small_44k_raw_output'
  cd '${MMAUDIO_ROOT:?set MMAUDIO_ROOT}'
  test -f weights/mmaudio_small_44k.pth
  test -f ext_weights/synchformer_state_dict.pth
  test -f ext_weights/v1-44.pth
  python demo.py --variant small_44k --duration=10 --video='${MAINBASE:?set MAINBASE}/cases/robot_warehouse_pick_001/input_video.mp4' --prompt 'Generate synchronized audio for a 10.0s video. Global scene: robot arm picking an item in a warehouse. Overall style: realistic Foley, no speech, no background music. Keep ambience stable, keep foley clear, avoid forbidden content: human speech, music, explosion, vehicle horn, speech, lyrics, voiceover.' --output='${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/dss_global/mmaudio_small_44k_raw_output' --seed=42
  NEW_AUDIO=$(find '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/dss_global/mmaudio_small_44k_raw_output' -type f \( -name '*.flac' -o -name '*.wav' \) -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2- || true)
  if [ -z "$NEW_AUDIO" ]; then echo "NO_MMAUDIO_OUTPUT_FOUND"; exit 2; fi
  case "$NEW_AUDIO" in
    *.flac) cp "$NEW_AUDIO" '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/dss_global/w18_018_robot_warehouse_pick_001_dss_global.flac' ;;
    *.wav) cp "$NEW_AUDIO" '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/dss_global/w18_018_robot_warehouse_pick_001_dss_global.wav' ;;
    *) echo "UNKNOWN_AUDIO_EXT=$NEW_AUDIO"; exit 3 ;;
  esac
  if [ -f '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/dss_global/w18_018_robot_warehouse_pick_001_dss_global.flac' ] && command -v ffmpeg >/dev/null 2>&1; then ffmpeg -y -i '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/dss_global/w18_018_robot_warehouse_pick_001_dss_global.flac' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/dss_global/w18_018_robot_warehouse_pick_001_dss_global.wav' >/dev/null 2>&1 || true; fi
  ls -lh '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/dss_global/w18_018_robot_warehouse_pick_001_dss_global.flac' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/dss_global/w18_018_robot_warehouse_pick_001_dss_global.wav' 2>/dev/null || true
)

(
  set -euo pipefail
  echo '### RUN w18_019_robot_warehouse_pick_001_dss_event_timeline duration=10s variant=dss_event_timeline'
  mkdir -p '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/dss_event_timeline' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/dss_event_timeline/mmaudio_small_44k_raw_output'
  cd '${MMAUDIO_ROOT:?set MMAUDIO_ROOT}'
  test -f weights/mmaudio_small_44k.pth
  test -f ext_weights/synchformer_state_dict.pth
  test -f ext_weights/v1-44.pth
  python demo.py --variant small_44k --duration=10 --video='${MAINBASE:?set MAINBASE}/cases/robot_warehouse_pick_001/input_video.mp4' --prompt 'Generate audio for a 10.0s video using this event timeline. Scene: robot arm picking an item in a warehouse. Event timeline: 0.00s: object=warehouse; action=warehouse_hum; sound=low warehouse ventilation hum; layer=foley; priority=1; tolerance=600ms | 0.00s: object=servo; action=servo_arm_move; sound=precise electric servo arm movement; layer=foley; priority=4; tolerance=180ms | 0.00s: object=box; action=box_lift; sound=cardboard box lifted from shelf; layer=foley; priority=5; tolerance=150ms | 0.00s: object=confirm; action=confirm_beep; sound=short clean robot confirmation beep; layer=foley; priority=3; tolerance=80ms. Preserve temporal order. Align transient sounds to event timestamps. Do not add: human speech, music, explosion, vehicle horn, speech, lyrics, voiceover.' --output='${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/dss_event_timeline/mmaudio_small_44k_raw_output' --seed=42
  NEW_AUDIO=$(find '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/dss_event_timeline/mmaudio_small_44k_raw_output' -type f \( -name '*.flac' -o -name '*.wav' \) -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2- || true)
  if [ -z "$NEW_AUDIO" ]; then echo "NO_MMAUDIO_OUTPUT_FOUND"; exit 2; fi
  case "$NEW_AUDIO" in
    *.flac) cp "$NEW_AUDIO" '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/dss_event_timeline/w18_019_robot_warehouse_pick_001_dss_event_timeline.flac' ;;
    *.wav) cp "$NEW_AUDIO" '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/dss_event_timeline/w18_019_robot_warehouse_pick_001_dss_event_timeline.wav' ;;
    *) echo "UNKNOWN_AUDIO_EXT=$NEW_AUDIO"; exit 3 ;;
  esac
  if [ -f '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/dss_event_timeline/w18_019_robot_warehouse_pick_001_dss_event_timeline.flac' ] && command -v ffmpeg >/dev/null 2>&1; then ffmpeg -y -i '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/dss_event_timeline/w18_019_robot_warehouse_pick_001_dss_event_timeline.flac' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/dss_event_timeline/w18_019_robot_warehouse_pick_001_dss_event_timeline.wav' >/dev/null 2>&1 || true; fi
  ls -lh '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/dss_event_timeline/w18_019_robot_warehouse_pick_001_dss_event_timeline.flac' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/dss_event_timeline/w18_019_robot_warehouse_pick_001_dss_event_timeline.wav' 2>/dev/null || true
)

(
  set -euo pipefail
  echo '### RUN w18_020_robot_warehouse_pick_001_dss_layer_avoid duration=10s variant=dss_layer_avoid'
  mkdir -p '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/dss_layer_avoid' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/dss_layer_avoid/mmaudio_small_44k_raw_output'
  cd '${MMAUDIO_ROOT:?set MMAUDIO_ROOT}'
  test -f weights/mmaudio_small_44k.pth
  test -f ext_weights/synchformer_state_dict.pth
  test -f ext_weights/v1-44.pth
  python demo.py --variant small_44k --duration=10 --video='${MAINBASE:?set MAINBASE}/cases/robot_warehouse_pick_001/input_video.mp4' --prompt 'Generate layered video-synchronized audio. Scene layer: robot arm picking an item in a warehouse. Foley layer must emphasize these events: 0.00s: object=warehouse; action=warehouse_hum; sound=low warehouse ventilation hum; layer=foley; priority=1; tolerance=600ms | 0.00s: object=servo; action=servo_arm_move; sound=precise electric servo arm movement; layer=foley; priority=4; tolerance=180ms | 0.00s: object=box; action=box_lift; sound=cardboard box lifted from shelf; layer=foley; priority=5; tolerance=150ms | 0.00s: object=confirm; action=confirm_beep; sound=short clean robot confirmation beep; layer=foley; priority=3; tolerance=80ms. Ambience layer should remain low, continuous, and non-dominant. Music layer should be absent unless explicitly required. Forbidden content: human speech, music, explosion, vehicle horn, speech, lyrics, voiceover. No speech, no lyrics, no unrelated background music.' --output='${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/dss_layer_avoid/mmaudio_small_44k_raw_output' --seed=42
  NEW_AUDIO=$(find '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/dss_layer_avoid/mmaudio_small_44k_raw_output' -type f \( -name '*.flac' -o -name '*.wav' \) -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2- || true)
  if [ -z "$NEW_AUDIO" ]; then echo "NO_MMAUDIO_OUTPUT_FOUND"; exit 2; fi
  case "$NEW_AUDIO" in
    *.flac) cp "$NEW_AUDIO" '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/dss_layer_avoid/w18_020_robot_warehouse_pick_001_dss_layer_avoid.flac' ;;
    *.wav) cp "$NEW_AUDIO" '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/dss_layer_avoid/w18_020_robot_warehouse_pick_001_dss_layer_avoid.wav' ;;
    *) echo "UNKNOWN_AUDIO_EXT=$NEW_AUDIO"; exit 3 ;;
  esac
  if [ -f '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/dss_layer_avoid/w18_020_robot_warehouse_pick_001_dss_layer_avoid.flac' ] && command -v ffmpeg >/dev/null 2>&1; then ffmpeg -y -i '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/dss_layer_avoid/w18_020_robot_warehouse_pick_001_dss_layer_avoid.flac' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/dss_layer_avoid/w18_020_robot_warehouse_pick_001_dss_layer_avoid.wav' >/dev/null 2>&1 || true; fi
  ls -lh '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/dss_layer_avoid/w18_020_robot_warehouse_pick_001_dss_layer_avoid.flac' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/robot_warehouse_pick_001/dss_layer_avoid/w18_020_robot_warehouse_pick_001_dss_layer_avoid.wav' 2>/dev/null || true
)

(
  set -euo pipefail
  echo '### RUN w18_021_street_rain_crosswalk_001_naive duration=9s variant=naive'
  mkdir -p '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/street_rain_crosswalk_001/naive' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/street_rain_crosswalk_001/naive/mmaudio_small_44k_raw_output'
  cd '${MMAUDIO_ROOT:?set MMAUDIO_ROOT}'
  test -f weights/mmaudio_small_44k.pth
  test -f ext_weights/synchformer_state_dict.pth
  test -f ext_weights/v1-44.pth
  python demo.py --variant small_44k --duration=9 --video='${MAINBASE:?set MAINBASE}/cases/street_rain_crosswalk_001/input_video.mp4' --prompt 'Create realistic sound effects for this short video. Scene: rainy street crosswalk with passing footsteps and vehicles. Use natural ambience and match visible actions. Avoid clear sunny ambience, music, speech, sirens, lyrics, voiceover.' --output='${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/street_rain_crosswalk_001/naive/mmaudio_small_44k_raw_output' --seed=42
  NEW_AUDIO=$(find '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/street_rain_crosswalk_001/naive/mmaudio_small_44k_raw_output' -type f \( -name '*.flac' -o -name '*.wav' \) -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2- || true)
  if [ -z "$NEW_AUDIO" ]; then echo "NO_MMAUDIO_OUTPUT_FOUND"; exit 2; fi
  case "$NEW_AUDIO" in
    *.flac) cp "$NEW_AUDIO" '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/street_rain_crosswalk_001/naive/w18_021_street_rain_crosswalk_001_naive.flac' ;;
    *.wav) cp "$NEW_AUDIO" '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/street_rain_crosswalk_001/naive/w18_021_street_rain_crosswalk_001_naive.wav' ;;
    *) echo "UNKNOWN_AUDIO_EXT=$NEW_AUDIO"; exit 3 ;;
  esac
  if [ -f '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/street_rain_crosswalk_001/naive/w18_021_street_rain_crosswalk_001_naive.flac' ] && command -v ffmpeg >/dev/null 2>&1; then ffmpeg -y -i '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/street_rain_crosswalk_001/naive/w18_021_street_rain_crosswalk_001_naive.flac' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/street_rain_crosswalk_001/naive/w18_021_street_rain_crosswalk_001_naive.wav' >/dev/null 2>&1 || true; fi
  ls -lh '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/street_rain_crosswalk_001/naive/w18_021_street_rain_crosswalk_001_naive.flac' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/street_rain_crosswalk_001/naive/w18_021_street_rain_crosswalk_001_naive.wav' 2>/dev/null || true
)

(
  set -euo pipefail
  echo '### RUN w18_022_street_rain_crosswalk_001_naive_rich duration=9s variant=naive_rich'
  mkdir -p '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/street_rain_crosswalk_001/naive_rich' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/street_rain_crosswalk_001/naive_rich/mmaudio_small_44k_raw_output'
  cd '${MMAUDIO_ROOT:?set MMAUDIO_ROOT}'
  test -f weights/mmaudio_small_44k.pth
  test -f ext_weights/synchformer_state_dict.pth
  test -f ext_weights/v1-44.pth
  python demo.py --variant small_44k --duration=9 --video='${MAINBASE:?set MAINBASE}/cases/street_rain_crosswalk_001/input_video.mp4' --prompt 'Create realistic sound effects for this short video. Scene: rainy street crosswalk with passing footsteps and vehicles. The visible actions include: around 0.00s, rain rain_ambience_start with steady light rain and wet street bed; around 0.00s, footsteps footsteps_enter with wet shoes stepping through shallow water; around 0.00s, car car_pass_left_to_right with muffled car pass-by on wet road; around 0.00s, puddle puddle_splash with single sharp puddle splash near camera. Make the sounds natural, synchronized, and cinematic. Avoid clear sunny ambience, music, speech, sirens, lyrics, voiceover.' --output='${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/street_rain_crosswalk_001/naive_rich/mmaudio_small_44k_raw_output' --seed=42
  NEW_AUDIO=$(find '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/street_rain_crosswalk_001/naive_rich/mmaudio_small_44k_raw_output' -type f \( -name '*.flac' -o -name '*.wav' \) -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2- || true)
  if [ -z "$NEW_AUDIO" ]; then echo "NO_MMAUDIO_OUTPUT_FOUND"; exit 2; fi
  case "$NEW_AUDIO" in
    *.flac) cp "$NEW_AUDIO" '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/street_rain_crosswalk_001/naive_rich/w18_022_street_rain_crosswalk_001_naive_rich.flac' ;;
    *.wav) cp "$NEW_AUDIO" '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/street_rain_crosswalk_001/naive_rich/w18_022_street_rain_crosswalk_001_naive_rich.wav' ;;
    *) echo "UNKNOWN_AUDIO_EXT=$NEW_AUDIO"; exit 3 ;;
  esac
  if [ -f '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/street_rain_crosswalk_001/naive_rich/w18_022_street_rain_crosswalk_001_naive_rich.flac' ] && command -v ffmpeg >/dev/null 2>&1; then ffmpeg -y -i '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/street_rain_crosswalk_001/naive_rich/w18_022_street_rain_crosswalk_001_naive_rich.flac' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/street_rain_crosswalk_001/naive_rich/w18_022_street_rain_crosswalk_001_naive_rich.wav' >/dev/null 2>&1 || true; fi
  ls -lh '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/street_rain_crosswalk_001/naive_rich/w18_022_street_rain_crosswalk_001_naive_rich.flac' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/street_rain_crosswalk_001/naive_rich/w18_022_street_rain_crosswalk_001_naive_rich.wav' 2>/dev/null || true
)

(
  set -euo pipefail
  echo '### RUN w18_023_street_rain_crosswalk_001_dss_global duration=9s variant=dss_global'
  mkdir -p '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/street_rain_crosswalk_001/dss_global' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/street_rain_crosswalk_001/dss_global/mmaudio_small_44k_raw_output'
  cd '${MMAUDIO_ROOT:?set MMAUDIO_ROOT}'
  test -f weights/mmaudio_small_44k.pth
  test -f ext_weights/synchformer_state_dict.pth
  test -f ext_weights/v1-44.pth
  python demo.py --variant small_44k --duration=9 --video='${MAINBASE:?set MAINBASE}/cases/street_rain_crosswalk_001/input_video.mp4' --prompt 'Generate synchronized audio for a 9.0s video. Global scene: rainy street crosswalk with passing footsteps and vehicles. Overall style: realistic Foley, no speech, no background music. Keep ambience stable, keep foley clear, avoid forbidden content: clear sunny ambience, music, speech, sirens, lyrics, voiceover.' --output='${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/street_rain_crosswalk_001/dss_global/mmaudio_small_44k_raw_output' --seed=42
  NEW_AUDIO=$(find '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/street_rain_crosswalk_001/dss_global/mmaudio_small_44k_raw_output' -type f \( -name '*.flac' -o -name '*.wav' \) -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2- || true)
  if [ -z "$NEW_AUDIO" ]; then echo "NO_MMAUDIO_OUTPUT_FOUND"; exit 2; fi
  case "$NEW_AUDIO" in
    *.flac) cp "$NEW_AUDIO" '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/street_rain_crosswalk_001/dss_global/w18_023_street_rain_crosswalk_001_dss_global.flac' ;;
    *.wav) cp "$NEW_AUDIO" '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/street_rain_crosswalk_001/dss_global/w18_023_street_rain_crosswalk_001_dss_global.wav' ;;
    *) echo "UNKNOWN_AUDIO_EXT=$NEW_AUDIO"; exit 3 ;;
  esac
  if [ -f '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/street_rain_crosswalk_001/dss_global/w18_023_street_rain_crosswalk_001_dss_global.flac' ] && command -v ffmpeg >/dev/null 2>&1; then ffmpeg -y -i '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/street_rain_crosswalk_001/dss_global/w18_023_street_rain_crosswalk_001_dss_global.flac' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/street_rain_crosswalk_001/dss_global/w18_023_street_rain_crosswalk_001_dss_global.wav' >/dev/null 2>&1 || true; fi
  ls -lh '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/street_rain_crosswalk_001/dss_global/w18_023_street_rain_crosswalk_001_dss_global.flac' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/street_rain_crosswalk_001/dss_global/w18_023_street_rain_crosswalk_001_dss_global.wav' 2>/dev/null || true
)

(
  set -euo pipefail
  echo '### RUN w18_024_street_rain_crosswalk_001_dss_event_timeline duration=9s variant=dss_event_timeline'
  mkdir -p '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/street_rain_crosswalk_001/dss_event_timeline' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/street_rain_crosswalk_001/dss_event_timeline/mmaudio_small_44k_raw_output'
  cd '${MMAUDIO_ROOT:?set MMAUDIO_ROOT}'
  test -f weights/mmaudio_small_44k.pth
  test -f ext_weights/synchformer_state_dict.pth
  test -f ext_weights/v1-44.pth
  python demo.py --variant small_44k --duration=9 --video='${MAINBASE:?set MAINBASE}/cases/street_rain_crosswalk_001/input_video.mp4' --prompt 'Generate audio for a 9.0s video using this event timeline. Scene: rainy street crosswalk with passing footsteps and vehicles. Event timeline: 0.00s: object=rain; action=rain_ambience_start; sound=steady light rain and wet street bed; layer=foley; priority=2; tolerance=500ms | 0.00s: object=footsteps; action=footsteps_enter; sound=wet shoes stepping through shallow water; layer=foley; priority=4; tolerance=180ms | 0.00s: object=car; action=car_pass_left_to_right; sound=muffled car pass-by on wet road; layer=foley; priority=3; tolerance=250ms | 0.00s: object=puddle; action=puddle_splash; sound=single sharp puddle splash near camera; layer=foley; priority=5; tolerance=120ms. Preserve temporal order. Align transient sounds to event timestamps. Do not add: clear sunny ambience, music, speech, sirens, lyrics, voiceover.' --output='${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/street_rain_crosswalk_001/dss_event_timeline/mmaudio_small_44k_raw_output' --seed=42
  NEW_AUDIO=$(find '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/street_rain_crosswalk_001/dss_event_timeline/mmaudio_small_44k_raw_output' -type f \( -name '*.flac' -o -name '*.wav' \) -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2- || true)
  if [ -z "$NEW_AUDIO" ]; then echo "NO_MMAUDIO_OUTPUT_FOUND"; exit 2; fi
  case "$NEW_AUDIO" in
    *.flac) cp "$NEW_AUDIO" '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/street_rain_crosswalk_001/dss_event_timeline/w18_024_street_rain_crosswalk_001_dss_event_timeline.flac' ;;
    *.wav) cp "$NEW_AUDIO" '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/street_rain_crosswalk_001/dss_event_timeline/w18_024_street_rain_crosswalk_001_dss_event_timeline.wav' ;;
    *) echo "UNKNOWN_AUDIO_EXT=$NEW_AUDIO"; exit 3 ;;
  esac
  if [ -f '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/street_rain_crosswalk_001/dss_event_timeline/w18_024_street_rain_crosswalk_001_dss_event_timeline.flac' ] && command -v ffmpeg >/dev/null 2>&1; then ffmpeg -y -i '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/street_rain_crosswalk_001/dss_event_timeline/w18_024_street_rain_crosswalk_001_dss_event_timeline.flac' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/street_rain_crosswalk_001/dss_event_timeline/w18_024_street_rain_crosswalk_001_dss_event_timeline.wav' >/dev/null 2>&1 || true; fi
  ls -lh '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/street_rain_crosswalk_001/dss_event_timeline/w18_024_street_rain_crosswalk_001_dss_event_timeline.flac' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/street_rain_crosswalk_001/dss_event_timeline/w18_024_street_rain_crosswalk_001_dss_event_timeline.wav' 2>/dev/null || true
)

(
  set -euo pipefail
  echo '### RUN w18_025_street_rain_crosswalk_001_dss_layer_avoid duration=9s variant=dss_layer_avoid'
  mkdir -p '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/street_rain_crosswalk_001/dss_layer_avoid' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/street_rain_crosswalk_001/dss_layer_avoid/mmaudio_small_44k_raw_output'
  cd '${MMAUDIO_ROOT:?set MMAUDIO_ROOT}'
  test -f weights/mmaudio_small_44k.pth
  test -f ext_weights/synchformer_state_dict.pth
  test -f ext_weights/v1-44.pth
  python demo.py --variant small_44k --duration=9 --video='${MAINBASE:?set MAINBASE}/cases/street_rain_crosswalk_001/input_video.mp4' --prompt 'Generate layered video-synchronized audio. Scene layer: rainy street crosswalk with passing footsteps and vehicles. Foley layer must emphasize these events: 0.00s: object=rain; action=rain_ambience_start; sound=steady light rain and wet street bed; layer=foley; priority=2; tolerance=500ms | 0.00s: object=footsteps; action=footsteps_enter; sound=wet shoes stepping through shallow water; layer=foley; priority=4; tolerance=180ms | 0.00s: object=car; action=car_pass_left_to_right; sound=muffled car pass-by on wet road; layer=foley; priority=3; tolerance=250ms | 0.00s: object=puddle; action=puddle_splash; sound=single sharp puddle splash near camera; layer=foley; priority=5; tolerance=120ms. Ambience layer should remain low, continuous, and non-dominant. Music layer should be absent unless explicitly required. Forbidden content: clear sunny ambience, music, speech, sirens, lyrics, voiceover. No speech, no lyrics, no unrelated background music.' --output='${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/street_rain_crosswalk_001/dss_layer_avoid/mmaudio_small_44k_raw_output' --seed=42
  NEW_AUDIO=$(find '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/street_rain_crosswalk_001/dss_layer_avoid/mmaudio_small_44k_raw_output' -type f \( -name '*.flac' -o -name '*.wav' \) -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2- || true)
  if [ -z "$NEW_AUDIO" ]; then echo "NO_MMAUDIO_OUTPUT_FOUND"; exit 2; fi
  case "$NEW_AUDIO" in
    *.flac) cp "$NEW_AUDIO" '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/street_rain_crosswalk_001/dss_layer_avoid/w18_025_street_rain_crosswalk_001_dss_layer_avoid.flac' ;;
    *.wav) cp "$NEW_AUDIO" '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/street_rain_crosswalk_001/dss_layer_avoid/w18_025_street_rain_crosswalk_001_dss_layer_avoid.wav' ;;
    *) echo "UNKNOWN_AUDIO_EXT=$NEW_AUDIO"; exit 3 ;;
  esac
  if [ -f '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/street_rain_crosswalk_001/dss_layer_avoid/w18_025_street_rain_crosswalk_001_dss_layer_avoid.flac' ] && command -v ffmpeg >/dev/null 2>&1; then ffmpeg -y -i '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/street_rain_crosswalk_001/dss_layer_avoid/w18_025_street_rain_crosswalk_001_dss_layer_avoid.flac' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/street_rain_crosswalk_001/dss_layer_avoid/w18_025_street_rain_crosswalk_001_dss_layer_avoid.wav' >/dev/null 2>&1 || true; fi
  ls -lh '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/street_rain_crosswalk_001/dss_layer_avoid/w18_025_street_rain_crosswalk_001_dss_layer_avoid.flac' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/street_rain_crosswalk_001/dss_layer_avoid/w18_025_street_rain_crosswalk_001_dss_layer_avoid.wav' 2>/dev/null || true
)

(
  set -euo pipefail
  echo '### RUN w18_026_subway_arrival_door_001_naive duration=10s variant=naive'
  mkdir -p '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/subway_arrival_door_001/naive' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/subway_arrival_door_001/naive/mmaudio_small_44k_raw_output'
  cd '${MMAUDIO_ROOT:?set MMAUDIO_ROOT}'
  test -f weights/mmaudio_small_44k.pth
  test -f ext_weights/synchformer_state_dict.pth
  test -f ext_weights/v1-44.pth
  python demo.py --variant small_44k --duration=10 --video='${MAINBASE:?set MAINBASE}/cases/subway_arrival_door_001/input_video.mp4' --prompt 'Create realistic sound effects for this short video. Scene: subway train arriving and doors opening. Use natural ambience and match visible actions. Avoid music, speech announcement, car horn, birdsong, speech, lyrics, voiceover.' --output='${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/subway_arrival_door_001/naive/mmaudio_small_44k_raw_output' --seed=42
  NEW_AUDIO=$(find '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/subway_arrival_door_001/naive/mmaudio_small_44k_raw_output' -type f \( -name '*.flac' -o -name '*.wav' \) -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2- || true)
  if [ -z "$NEW_AUDIO" ]; then echo "NO_MMAUDIO_OUTPUT_FOUND"; exit 2; fi
  case "$NEW_AUDIO" in
    *.flac) cp "$NEW_AUDIO" '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/subway_arrival_door_001/naive/w18_026_subway_arrival_door_001_naive.flac' ;;
    *.wav) cp "$NEW_AUDIO" '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/subway_arrival_door_001/naive/w18_026_subway_arrival_door_001_naive.wav' ;;
    *) echo "UNKNOWN_AUDIO_EXT=$NEW_AUDIO"; exit 3 ;;
  esac
  if [ -f '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/subway_arrival_door_001/naive/w18_026_subway_arrival_door_001_naive.flac' ] && command -v ffmpeg >/dev/null 2>&1; then ffmpeg -y -i '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/subway_arrival_door_001/naive/w18_026_subway_arrival_door_001_naive.flac' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/subway_arrival_door_001/naive/w18_026_subway_arrival_door_001_naive.wav' >/dev/null 2>&1 || true; fi
  ls -lh '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/subway_arrival_door_001/naive/w18_026_subway_arrival_door_001_naive.flac' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/subway_arrival_door_001/naive/w18_026_subway_arrival_door_001_naive.wav' 2>/dev/null || true
)

(
  set -euo pipefail
  echo '### RUN w18_027_subway_arrival_door_001_naive_rich duration=10s variant=naive_rich'
  mkdir -p '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/subway_arrival_door_001/naive_rich' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/subway_arrival_door_001/naive_rich/mmaudio_small_44k_raw_output'
  cd '${MMAUDIO_ROOT:?set MMAUDIO_ROOT}'
  test -f weights/mmaudio_small_44k.pth
  test -f ext_weights/synchformer_state_dict.pth
  test -f ext_weights/v1-44.pth
  python demo.py --variant small_44k --duration=10 --video='${MAINBASE:?set MAINBASE}/cases/subway_arrival_door_001/input_video.mp4' --prompt 'Create realistic sound effects for this short video. Scene: subway train arriving and doors opening. The visible actions include: around 0.00s, platform platform_bed with underground platform room tone and distant crowd; around 0.00s, train train_rumble_approach with subway train approaching with low rumble; around 0.00s, brake brake_squeal with metal brake squeal as train stops; around 0.00s, door door_open_chime with door open chime and sliding door start. Make the sounds natural, synchronized, and cinematic. Avoid music, speech announcement, car horn, birdsong, speech, lyrics, voiceover.' --output='${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/subway_arrival_door_001/naive_rich/mmaudio_small_44k_raw_output' --seed=42
  NEW_AUDIO=$(find '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/subway_arrival_door_001/naive_rich/mmaudio_small_44k_raw_output' -type f \( -name '*.flac' -o -name '*.wav' \) -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2- || true)
  if [ -z "$NEW_AUDIO" ]; then echo "NO_MMAUDIO_OUTPUT_FOUND"; exit 2; fi
  case "$NEW_AUDIO" in
    *.flac) cp "$NEW_AUDIO" '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/subway_arrival_door_001/naive_rich/w18_027_subway_arrival_door_001_naive_rich.flac' ;;
    *.wav) cp "$NEW_AUDIO" '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/subway_arrival_door_001/naive_rich/w18_027_subway_arrival_door_001_naive_rich.wav' ;;
    *) echo "UNKNOWN_AUDIO_EXT=$NEW_AUDIO"; exit 3 ;;
  esac
  if [ -f '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/subway_arrival_door_001/naive_rich/w18_027_subway_arrival_door_001_naive_rich.flac' ] && command -v ffmpeg >/dev/null 2>&1; then ffmpeg -y -i '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/subway_arrival_door_001/naive_rich/w18_027_subway_arrival_door_001_naive_rich.flac' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/subway_arrival_door_001/naive_rich/w18_027_subway_arrival_door_001_naive_rich.wav' >/dev/null 2>&1 || true; fi
  ls -lh '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/subway_arrival_door_001/naive_rich/w18_027_subway_arrival_door_001_naive_rich.flac' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/subway_arrival_door_001/naive_rich/w18_027_subway_arrival_door_001_naive_rich.wav' 2>/dev/null || true
)

(
  set -euo pipefail
  echo '### RUN w18_028_subway_arrival_door_001_dss_global duration=10s variant=dss_global'
  mkdir -p '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/subway_arrival_door_001/dss_global' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/subway_arrival_door_001/dss_global/mmaudio_small_44k_raw_output'
  cd '${MMAUDIO_ROOT:?set MMAUDIO_ROOT}'
  test -f weights/mmaudio_small_44k.pth
  test -f ext_weights/synchformer_state_dict.pth
  test -f ext_weights/v1-44.pth
  python demo.py --variant small_44k --duration=10 --video='${MAINBASE:?set MAINBASE}/cases/subway_arrival_door_001/input_video.mp4' --prompt 'Generate synchronized audio for a 10.0s video. Global scene: subway train arriving and doors opening. Overall style: realistic Foley, no speech, no background music. Keep ambience stable, keep foley clear, avoid forbidden content: music, speech announcement, car horn, birdsong, speech, lyrics, voiceover.' --output='${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/subway_arrival_door_001/dss_global/mmaudio_small_44k_raw_output' --seed=42
  NEW_AUDIO=$(find '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/subway_arrival_door_001/dss_global/mmaudio_small_44k_raw_output' -type f \( -name '*.flac' -o -name '*.wav' \) -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2- || true)
  if [ -z "$NEW_AUDIO" ]; then echo "NO_MMAUDIO_OUTPUT_FOUND"; exit 2; fi
  case "$NEW_AUDIO" in
    *.flac) cp "$NEW_AUDIO" '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/subway_arrival_door_001/dss_global/w18_028_subway_arrival_door_001_dss_global.flac' ;;
    *.wav) cp "$NEW_AUDIO" '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/subway_arrival_door_001/dss_global/w18_028_subway_arrival_door_001_dss_global.wav' ;;
    *) echo "UNKNOWN_AUDIO_EXT=$NEW_AUDIO"; exit 3 ;;
  esac
  if [ -f '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/subway_arrival_door_001/dss_global/w18_028_subway_arrival_door_001_dss_global.flac' ] && command -v ffmpeg >/dev/null 2>&1; then ffmpeg -y -i '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/subway_arrival_door_001/dss_global/w18_028_subway_arrival_door_001_dss_global.flac' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/subway_arrival_door_001/dss_global/w18_028_subway_arrival_door_001_dss_global.wav' >/dev/null 2>&1 || true; fi
  ls -lh '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/subway_arrival_door_001/dss_global/w18_028_subway_arrival_door_001_dss_global.flac' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/subway_arrival_door_001/dss_global/w18_028_subway_arrival_door_001_dss_global.wav' 2>/dev/null || true
)

(
  set -euo pipefail
  echo '### RUN w18_029_subway_arrival_door_001_dss_event_timeline duration=10s variant=dss_event_timeline'
  mkdir -p '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/subway_arrival_door_001/dss_event_timeline' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/subway_arrival_door_001/dss_event_timeline/mmaudio_small_44k_raw_output'
  cd '${MMAUDIO_ROOT:?set MMAUDIO_ROOT}'
  test -f weights/mmaudio_small_44k.pth
  test -f ext_weights/synchformer_state_dict.pth
  test -f ext_weights/v1-44.pth
  python demo.py --variant small_44k --duration=10 --video='${MAINBASE:?set MAINBASE}/cases/subway_arrival_door_001/input_video.mp4' --prompt 'Generate audio for a 10.0s video using this event timeline. Scene: subway train arriving and doors opening. Event timeline: 0.00s: object=platform; action=platform_bed; sound=underground platform room tone and distant crowd; layer=foley; priority=2; tolerance=700ms | 0.00s: object=train; action=train_rumble_approach; sound=subway train approaching with low rumble; layer=foley; priority=4; tolerance=250ms | 0.00s: object=brake; action=brake_squeal; sound=metal brake squeal as train stops; layer=foley; priority=5; tolerance=150ms | 0.00s: object=door; action=door_open_chime; sound=door open chime and sliding door start; layer=foley; priority=3; tolerance=120ms. Preserve temporal order. Align transient sounds to event timestamps. Do not add: music, speech announcement, car horn, birdsong, speech, lyrics, voiceover.' --output='${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/subway_arrival_door_001/dss_event_timeline/mmaudio_small_44k_raw_output' --seed=42
  NEW_AUDIO=$(find '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/subway_arrival_door_001/dss_event_timeline/mmaudio_small_44k_raw_output' -type f \( -name '*.flac' -o -name '*.wav' \) -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2- || true)
  if [ -z "$NEW_AUDIO" ]; then echo "NO_MMAUDIO_OUTPUT_FOUND"; exit 2; fi
  case "$NEW_AUDIO" in
    *.flac) cp "$NEW_AUDIO" '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/subway_arrival_door_001/dss_event_timeline/w18_029_subway_arrival_door_001_dss_event_timeline.flac' ;;
    *.wav) cp "$NEW_AUDIO" '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/subway_arrival_door_001/dss_event_timeline/w18_029_subway_arrival_door_001_dss_event_timeline.wav' ;;
    *) echo "UNKNOWN_AUDIO_EXT=$NEW_AUDIO"; exit 3 ;;
  esac
  if [ -f '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/subway_arrival_door_001/dss_event_timeline/w18_029_subway_arrival_door_001_dss_event_timeline.flac' ] && command -v ffmpeg >/dev/null 2>&1; then ffmpeg -y -i '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/subway_arrival_door_001/dss_event_timeline/w18_029_subway_arrival_door_001_dss_event_timeline.flac' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/subway_arrival_door_001/dss_event_timeline/w18_029_subway_arrival_door_001_dss_event_timeline.wav' >/dev/null 2>&1 || true; fi
  ls -lh '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/subway_arrival_door_001/dss_event_timeline/w18_029_subway_arrival_door_001_dss_event_timeline.flac' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/subway_arrival_door_001/dss_event_timeline/w18_029_subway_arrival_door_001_dss_event_timeline.wav' 2>/dev/null || true
)

(
  set -euo pipefail
  echo '### RUN w18_030_subway_arrival_door_001_dss_layer_avoid duration=10s variant=dss_layer_avoid'
  mkdir -p '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/subway_arrival_door_001/dss_layer_avoid' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/subway_arrival_door_001/dss_layer_avoid/mmaudio_small_44k_raw_output'
  cd '${MMAUDIO_ROOT:?set MMAUDIO_ROOT}'
  test -f weights/mmaudio_small_44k.pth
  test -f ext_weights/synchformer_state_dict.pth
  test -f ext_weights/v1-44.pth
  python demo.py --variant small_44k --duration=10 --video='${MAINBASE:?set MAINBASE}/cases/subway_arrival_door_001/input_video.mp4' --prompt 'Generate layered video-synchronized audio. Scene layer: subway train arriving and doors opening. Foley layer must emphasize these events: 0.00s: object=platform; action=platform_bed; sound=underground platform room tone and distant crowd; layer=foley; priority=2; tolerance=700ms | 0.00s: object=train; action=train_rumble_approach; sound=subway train approaching with low rumble; layer=foley; priority=4; tolerance=250ms | 0.00s: object=brake; action=brake_squeal; sound=metal brake squeal as train stops; layer=foley; priority=5; tolerance=150ms | 0.00s: object=door; action=door_open_chime; sound=door open chime and sliding door start; layer=foley; priority=3; tolerance=120ms. Ambience layer should remain low, continuous, and non-dominant. Music layer should be absent unless explicitly required. Forbidden content: music, speech announcement, car horn, birdsong, speech, lyrics, voiceover. No speech, no lyrics, no unrelated background music.' --output='${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/subway_arrival_door_001/dss_layer_avoid/mmaudio_small_44k_raw_output' --seed=42
  NEW_AUDIO=$(find '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/subway_arrival_door_001/dss_layer_avoid/mmaudio_small_44k_raw_output' -type f \( -name '*.flac' -o -name '*.wav' \) -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2- || true)
  if [ -z "$NEW_AUDIO" ]; then echo "NO_MMAUDIO_OUTPUT_FOUND"; exit 2; fi
  case "$NEW_AUDIO" in
    *.flac) cp "$NEW_AUDIO" '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/subway_arrival_door_001/dss_layer_avoid/w18_030_subway_arrival_door_001_dss_layer_avoid.flac' ;;
    *.wav) cp "$NEW_AUDIO" '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/subway_arrival_door_001/dss_layer_avoid/w18_030_subway_arrival_door_001_dss_layer_avoid.wav' ;;
    *) echo "UNKNOWN_AUDIO_EXT=$NEW_AUDIO"; exit 3 ;;
  esac
  if [ -f '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/subway_arrival_door_001/dss_layer_avoid/w18_030_subway_arrival_door_001_dss_layer_avoid.flac' ] && command -v ffmpeg >/dev/null 2>&1; then ffmpeg -y -i '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/subway_arrival_door_001/dss_layer_avoid/w18_030_subway_arrival_door_001_dss_layer_avoid.flac' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/subway_arrival_door_001/dss_layer_avoid/w18_030_subway_arrival_door_001_dss_layer_avoid.wav' >/dev/null 2>&1 || true; fi
  ls -lh '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/subway_arrival_door_001/dss_layer_avoid/w18_030_subway_arrival_door_001_dss_layer_avoid.flac' '${MAINBASE:?set MAINBASE}/artifacts/model_runs/w18_dss_ablation/subway_arrival_door_001/dss_layer_avoid/w18_030_subway_arrival_door_001_dss_layer_avoid.wav' 2>/dev/null || true
)

