#!/usr/bin/env bash
set -euo pipefail

source /venv/main/bin/activate
cd /workspace/Hybrid-tts
export HF_HOME=/workspace/.hf_home
export PYTHONPATH="/workspace/Hybrid-tts/validation/compat:/workspace/external/MOSS-TTS-Nano${PYTHONPATH:+:${PYTHONPATH}}"

ROOT=/workspace/nano-flash-artifacts/g4/moss-intelligibility
CODEC=/workspace/.hf_home/hub/models--OpenMOSS-Team--MOSS-Audio-Tokenizer-Nano/snapshots/6aa02b01e445cc585582cf0ba480bc3ea6c8dd68
PROMPT=/workspace/external/MOSS-TTS-Nano/assets/audio/en_6.wav
mkdir -p "$ROOT"

python validation/moss_challenge_synthesize.py \
  --model /workspace/nano-flash-artifacts/g3/moss-overfit/merged-model \
  --audio-tokenizer "$CODEC" \
  --prompt-audio "$PROMPT" \
  --suite validation/english_challenge_suite.json \
  --output-dir "$ROOT/baseline"

python validation/moss_challenge_synthesize.py \
  --model /workspace/nano-flash-artifacts/g4/moss-lean-adaptation/checkpoints/checkpoint-last \
  --audio-tokenizer "$CODEC" \
  --prompt-audio "$PROMPT" \
  --suite validation/english_challenge_suite.json \
  --output-dir "$ROOT/adapted"

python validation/asr_compare.py \
  --baseline "$ROOT/baseline" \
  --adapted "$ROOT/adapted" \
  --output "$ROOT/asr-comparison.json"
