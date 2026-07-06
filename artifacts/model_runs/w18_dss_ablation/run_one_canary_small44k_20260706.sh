#!/usr/bin/env bash
set -euo pipefail

MAINBASE="${MAINBASE:-$HOME/work/audio_engineering_repo_skeleton_v1}"
MMAUDIO_ROOT="${MMAUDIO_ROOT:-/home/GRT/work/_external/MMAudio}"

JOB_ID="w18_001_forest_bird_branch_001_naive"
CASE_ID="forest_bird_branch_001"
VARIANT="naive"
VIDEO="$MAINBASE/cases/forest_bird_branch_001/input_video.mp4"
PROMPT="Create realistic sound effects for this short video. Scene: forest close-up with a bird moving on a branch. Use natural ambience and match visible actions. Avoid human speech, heavy wind, music bed, urban traffic, speech, lyrics, voiceover."
JOB_DIR="$MAINBASE/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/naive"
RAW_OUT="$MAINBASE/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/naive/mmaudio_small_44k_raw_output"
EXPECTED_FLAC="$MAINBASE/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/naive/w18_001_forest_bird_branch_001_naive.flac"
EXPECTED_WAV="$MAINBASE/artifacts/model_runs/w18_dss_ablation/forest_bird_branch_001/naive/w18_001_forest_bird_branch_001_naive.wav"

echo "JOB_ID=$JOB_ID"
echo "CASE_ID=$CASE_ID"
echo "PROMPT_VARIANT=$VARIANT"
echo "MMAUDIO_MODEL_VARIANT=small_44k"
echo "VIDEO=$VIDEO"
echo "RAW_OUT=$RAW_OUT"
echo "EXPECTED_FLAC=$EXPECTED_FLAC"
echo "EXPECTED_WAV=$EXPECTED_WAV"

test -f "$MMAUDIO_ROOT/demo.py"
test -f "$MMAUDIO_ROOT/weights/mmaudio_small_44k.pth"
test -f "$MMAUDIO_ROOT/ext_weights/synchformer_state_dict.pth"
test -f "$MMAUDIO_ROOT/ext_weights/v1-44.pth"
test -f "$VIDEO"

mkdir -p "$JOB_DIR" "$RAW_OUT"

cd "$MMAUDIO_ROOT"

python demo.py \
  --variant small_44k \
  --duration=10 \
  --video="$VIDEO" \
  --prompt "$PROMPT" \
  --output="$RAW_OUT" \
  --seed=42

NEW_AUDIO=$(find "$RAW_OUT" -type f \( -name '*.flac' -o -name '*.wav' \) -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2- || true)

if [ -z "$NEW_AUDIO" ]; then
  echo "NO_MMAUDIO_OUTPUT_FOUND_IN_RAW_OUT"
  find "$RAW_OUT" -maxdepth 2 -type f -printf '%p\t%s bytes\n' 2>/dev/null || true
  exit 2
fi

case "$NEW_AUDIO" in
  *.flac)
    cp "$NEW_AUDIO" "$EXPECTED_FLAC"
    ;;
  *.wav)
    cp "$NEW_AUDIO" "$EXPECTED_WAV"
    ;;
  *)
    echo "UNKNOWN_AUDIO_EXT=$NEW_AUDIO"
    exit 3
    ;;
esac

if [ -f "$EXPECTED_FLAC" ] && command -v ffmpeg >/dev/null 2>&1; then
  ffmpeg -y -i "$EXPECTED_FLAC" "$EXPECTED_WAV" >/dev/null 2>&1 || true
fi

echo "NEW_AUDIO=$NEW_AUDIO"
ls -lh "$EXPECTED_FLAC" "$EXPECTED_WAV" 2>/dev/null || true
