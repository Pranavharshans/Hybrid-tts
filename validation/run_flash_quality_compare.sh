#!/usr/bin/env bash
set -euo pipefail

source /workspace/venvs/chatterbox-flash/bin/activate
cd /workspace/Hybrid-tts
export HF_HOME=/workspace/.hf_home
export FLASHINFER_CACHEDIR=/workspace/.flashinfer
export PYTHONPATH=/workspace/Hybrid-tts/validation

ROOT=/workspace/nano-flash-artifacts/g7/flash-quality
SNAPSHOT=/workspace/.hf_home/hub/models--ResembleAI--chatterbox-flash/snapshots/4385507288b8197e6dab8b4e6b1603328d549d9d
PROMPT=/workspace/external/MOSS-TTS-Nano/assets/audio/en_6.wav
mkdir -p "$ROOT"

python validation/flash_challenge_synthesize.py --snapshot "$SNAPSHOT" --prompt-audio "$PROMPT" --suite validation/english_challenge_suite.json --backend torch --block-size 16 --output-dir "$ROOT/torch-b16"
python validation/flash_challenge_synthesize.py --snapshot "$SNAPSHOT" --prompt-audio "$PROMPT" --suite validation/english_challenge_suite.json --backend flashinfer --block-size 16 --cuda-graph --output-dir "$ROOT/flash-b16"
python validation/flash_challenge_synthesize.py --snapshot "$SNAPSHOT" --prompt-audio "$PROMPT" --suite validation/english_challenge_suite.json --backend flashinfer --block-size 32 --cuda-graph --output-dir "$ROOT/flash-b32"

python validation/asr_multi_compare.py \
  --group "torch-b16=$ROOT/torch-b16" \
  --group "flash-b16=$ROOT/flash-b16" \
  --group "flash-b32=$ROOT/flash-b32" \
  --output "$ROOT/asr-comparison.json"
