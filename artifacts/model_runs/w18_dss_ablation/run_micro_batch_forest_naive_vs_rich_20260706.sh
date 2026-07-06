#!/usr/bin/env bash
set -euo pipefail
MAINBASE="${MAINBASE:-$HOME/work/audio_engineering_repo_skeleton_v1}"
MMAUDIO_ROOT="${MMAUDIO_ROOT:-/home/GRT/work/_external/MMAudio}"
HF_HOME="${HF_HOME:-$MMAUDIO_ROOT/.hf_cache}"
HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export MAINBASE MMAUDIO_ROOT HF_HOME HUGGINGFACE_HUB_CACHE
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

echo '### MICRO w18_001_forest_bird_branch_001_naive duration=8s variant=naive'
mkdir -p "$MAINBASE/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/naive" "$MAINBASE/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/naive/mmaudio_small_44k_raw_output"
cd "$MMAUDIO_ROOT"
test -f weights/mmaudio_small_44k.pth
test -f ext_weights/synchformer_state_dict.pth
test -f ext_weights/v1-44.pth
python demo.py --variant small_44k --duration=8 --video="$MAINBASE/cases/forest_bird_branch_001/input_video.mp4" --prompt "Create realistic sound effects for this short video. Scene: forest close-up with a bird moving on a branch. Use natural ambience and match visible actions. Avoid human speech, heavy wind, music bed, urban traffic, speech, lyrics, voiceover." --output="$MAINBASE/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/naive/mmaudio_small_44k_raw_output" --seed=42
NEW_AUDIO=$(find "$MAINBASE/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/naive/mmaudio_small_44k_raw_output" -type f \( -name '*.flac' -o -name '*.wav' \) -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2- || true)
if [ -z "$NEW_AUDIO" ]; then echo "NO_MMAUDIO_OUTPUT_FOUND"; exit 2; fi
cp "$NEW_AUDIO" "$MAINBASE/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/naive/w18_001_forest_bird_branch_001_naive.flac"
if command -v ffmpeg >/dev/null 2>&1; then ffmpeg -y -i "$MAINBASE/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/naive/w18_001_forest_bird_branch_001_naive.flac" "$MAINBASE/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/naive/w18_001_forest_bird_branch_001_naive.wav" >/dev/null 2>&1 || true; fi
ls -lh "$MAINBASE/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/naive/w18_001_forest_bird_branch_001_naive.flac" "$MAINBASE/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/naive/w18_001_forest_bird_branch_001_naive.wav" 2>/dev/null || true

echo '### MICRO w18_002_forest_bird_branch_001_naive_rich duration=8s variant=naive_rich'
mkdir -p "$MAINBASE/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/naive_rich" "$MAINBASE/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/naive_rich/mmaudio_small_44k_raw_output"
cd "$MMAUDIO_ROOT"
test -f weights/mmaudio_small_44k.pth
test -f ext_weights/synchformer_state_dict.pth
test -f ext_weights/v1-44.pth
python demo.py --variant small_44k --duration=8 --video="$MAINBASE/cases/forest_bird_branch_001/input_video.mp4" --prompt "Create realistic sound effects for this short video. Scene: forest close-up with a bird moving on a branch. The visible actions include: around 0.00s, forest forest_wind with soft wind through leaves; around 0.00s, bird bird_chirp with two short bird chirps; around 0.00s, leaf leaf_rustle with small animal or bird rustling leaves; around 0.00s, branch branch_crack with dry branch crack. Make the sounds natural, synchronized, and cinematic. Avoid human speech, heavy wind, music bed, urban traffic, speech, lyrics, voiceover." --output="$MAINBASE/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/naive_rich/mmaudio_small_44k_raw_output" --seed=42
NEW_AUDIO=$(find "$MAINBASE/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/naive_rich/mmaudio_small_44k_raw_output" -type f \( -name '*.flac' -o -name '*.wav' \) -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2- || true)
if [ -z "$NEW_AUDIO" ]; then echo "NO_MMAUDIO_OUTPUT_FOUND"; exit 2; fi
cp "$NEW_AUDIO" "$MAINBASE/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/naive_rich/w18_002_forest_bird_branch_001_naive_rich.flac"
if command -v ffmpeg >/dev/null 2>&1; then ffmpeg -y -i "$MAINBASE/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/naive_rich/w18_002_forest_bird_branch_001_naive_rich.flac" "$MAINBASE/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/naive_rich/w18_002_forest_bird_branch_001_naive_rich.wav" >/dev/null 2>&1 || true; fi
ls -lh "$MAINBASE/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/naive_rich/w18_002_forest_bird_branch_001_naive_rich.flac" "$MAINBASE/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/naive_rich/w18_002_forest_bird_branch_001_naive_rich.wav" 2>/dev/null || true

