#!/usr/bin/env bash
set -euo pipefail
MAINBASE="${MAINBASE:-$HOME/work/audio_engineering_repo_skeleton_v1}"
MMAUDIO_ROOT="${MMAUDIO_ROOT:-/home/GRT/work/_external/MMAudio}"
HF_HOME="${HF_HOME:-$MMAUDIO_ROOT/.hf_cache}"
HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export MAINBASE MMAUDIO_ROOT HF_HOME HUGGINGFACE_HUB_CACHE
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

echo '### RUN w18_003_forest_bird_branch_001_dss_global duration=8s variant=dss_global'
mkdir -p "$MAINBASE/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/dss_global" "$MAINBASE/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/dss_global/mmaudio_small_44k_raw_output"
cd "$MMAUDIO_ROOT"
test -f weights/mmaudio_small_44k.pth
test -f ext_weights/synchformer_state_dict.pth
test -f ext_weights/v1-44.pth
python demo.py --variant small_44k --duration=8 --video="$MAINBASE/cases/forest_bird_branch_001/input_video.mp4" --prompt "Generate synchronized audio for a 10.0s video. Global scene: forest close-up with a bird moving on a branch. Overall style: realistic Foley, no speech, no background music. Keep ambience stable, keep foley clear, avoid forbidden content: human speech, heavy wind, music bed, urban traffic, speech, lyrics, voiceover." --output="$MAINBASE/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/dss_global/mmaudio_small_44k_raw_output" --seed=42
NEW_AUDIO=$(find "$MAINBASE/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/dss_global/mmaudio_small_44k_raw_output" -type f \( -name '*.flac' -o -name '*.wav' \) -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2- || true)
if [ -z "$NEW_AUDIO" ]; then echo "NO_MMAUDIO_OUTPUT_FOUND"; exit 2; fi
cp "$NEW_AUDIO" "$MAINBASE/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/dss_global/w18_003_forest_bird_branch_001_dss_global.flac"
if command -v ffmpeg >/dev/null 2>&1; then ffmpeg -y -i "$MAINBASE/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/dss_global/w18_003_forest_bird_branch_001_dss_global.flac" "$MAINBASE/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/dss_global/w18_003_forest_bird_branch_001_dss_global.wav" >/dev/null 2>&1 || true; fi
ls -lh "$MAINBASE/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/dss_global/w18_003_forest_bird_branch_001_dss_global.flac" "$MAINBASE/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/dss_global/w18_003_forest_bird_branch_001_dss_global.wav" 2>/dev/null || true

echo '### RUN w18_004_forest_bird_branch_001_dss_event_timeline duration=8s variant=dss_event_timeline'
mkdir -p "$MAINBASE/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/dss_event_timeline" "$MAINBASE/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/dss_event_timeline/mmaudio_small_44k_raw_output"
cd "$MMAUDIO_ROOT"
test -f weights/mmaudio_small_44k.pth
test -f ext_weights/synchformer_state_dict.pth
test -f ext_weights/v1-44.pth
python demo.py --variant small_44k --duration=8 --video="$MAINBASE/cases/forest_bird_branch_001/input_video.mp4" --prompt "Generate audio for a 10.0s video using this event timeline. Scene: forest close-up with a bird moving on a branch. Event timeline: 0.00s: object=forest; action=forest_wind; sound=soft wind through leaves; layer=foley; priority=2; tolerance=500ms | 0.00s: object=bird; action=bird_chirp; sound=two short bird chirps; layer=foley; priority=3; tolerance=150ms | 0.00s: object=leaf; action=leaf_rustle; sound=small animal or bird rustling leaves; layer=foley; priority=4; tolerance=180ms | 0.00s: object=branch; action=branch_crack; sound=dry branch crack; layer=foley; priority=5; tolerance=100ms. Preserve temporal order. Align transient sounds to event timestamps. Do not add: human speech, heavy wind, music bed, urban traffic, speech, lyrics, voiceover." --output="$MAINBASE/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/dss_event_timeline/mmaudio_small_44k_raw_output" --seed=42
NEW_AUDIO=$(find "$MAINBASE/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/dss_event_timeline/mmaudio_small_44k_raw_output" -type f \( -name '*.flac' -o -name '*.wav' \) -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2- || true)
if [ -z "$NEW_AUDIO" ]; then echo "NO_MMAUDIO_OUTPUT_FOUND"; exit 2; fi
cp "$NEW_AUDIO" "$MAINBASE/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/dss_event_timeline/w18_004_forest_bird_branch_001_dss_event_timeline.flac"
if command -v ffmpeg >/dev/null 2>&1; then ffmpeg -y -i "$MAINBASE/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/dss_event_timeline/w18_004_forest_bird_branch_001_dss_event_timeline.flac" "$MAINBASE/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/dss_event_timeline/w18_004_forest_bird_branch_001_dss_event_timeline.wav" >/dev/null 2>&1 || true; fi
ls -lh "$MAINBASE/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/dss_event_timeline/w18_004_forest_bird_branch_001_dss_event_timeline.flac" "$MAINBASE/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/dss_event_timeline/w18_004_forest_bird_branch_001_dss_event_timeline.wav" 2>/dev/null || true

echo '### RUN w18_005_forest_bird_branch_001_dss_layer_avoid duration=8s variant=dss_layer_avoid'
mkdir -p "$MAINBASE/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/dss_layer_avoid" "$MAINBASE/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/dss_layer_avoid/mmaudio_small_44k_raw_output"
cd "$MMAUDIO_ROOT"
test -f weights/mmaudio_small_44k.pth
test -f ext_weights/synchformer_state_dict.pth
test -f ext_weights/v1-44.pth
python demo.py --variant small_44k --duration=8 --video="$MAINBASE/cases/forest_bird_branch_001/input_video.mp4" --prompt "Generate layered video-synchronized audio for a 8.0s video. Scene layer: forest close-up with a bird moving on a branch. Foley layer must emphasize these events: 0.00s: object=forest; action=forest_wind; sound=soft wind through leaves; layer=foley; priority=2; tolerance=500ms | 0.00s: object=bird; action=bird_chirp; sound=two short bird chirps; layer=foley; priority=3; tolerance=150ms | 0.00s: object=leaf; action=leaf_rustle; sound=small animal or bird rustling leaves; layer=foley; priority=4; tolerance=180ms | 0.00s: object=branch; action=branch_crack; sound=dry branch crack; layer=foley; priority=5; tolerance=100ms. Ambience layer should remain low, continuous, and non-dominant. Music layer should be absent unless explicitly required. Forbidden content: human speech, heavy wind, music bed, urban traffic, speech, lyrics, voiceover. No speech, no lyrics, no unrelated background music." --output="$MAINBASE/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/dss_layer_avoid/mmaudio_small_44k_raw_output" --seed=42
NEW_AUDIO=$(find "$MAINBASE/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/dss_layer_avoid/mmaudio_small_44k_raw_output" -type f \( -name '*.flac' -o -name '*.wav' \) -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2- || true)
if [ -z "$NEW_AUDIO" ]; then echo "NO_MMAUDIO_OUTPUT_FOUND"; exit 2; fi
cp "$NEW_AUDIO" "$MAINBASE/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/dss_layer_avoid/w18_005_forest_bird_branch_001_dss_layer_avoid.flac"
if command -v ffmpeg >/dev/null 2>&1; then ffmpeg -y -i "$MAINBASE/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/dss_layer_avoid/w18_005_forest_bird_branch_001_dss_layer_avoid.flac" "$MAINBASE/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/dss_layer_avoid/w18_005_forest_bird_branch_001_dss_layer_avoid.wav" >/dev/null 2>&1 || true; fi
ls -lh "$MAINBASE/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/dss_layer_avoid/w18_005_forest_bird_branch_001_dss_layer_avoid.flac" "$MAINBASE/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/dss_layer_avoid/w18_005_forest_bird_branch_001_dss_layer_avoid.wav" 2>/dev/null || true

