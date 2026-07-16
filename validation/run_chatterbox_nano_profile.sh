#!/usr/bin/env bash
set -euo pipefail

source /workspace/venvs/chatterbox-nano/bin/activate
cd /workspace/Hybrid-tts
export PYTHONPATH=/workspace/external/chatterbox-nano-demo
export HF_HOME=/workspace/.hf_home
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python validation/chatterbox_nano_profile.py \
  --nano-source /workspace/external/chatterbox-nano-demo \
  --snapshot /workspace/.hf_home/hub/models--ResembleAI--chatterbox-nano/snapshots/493317046f21b7e557146a9285a111c050564bb4 \
  --prompt-audio /workspace/external/MOSS-TTS-Nano/assets/audio/en_6.wav \
  --output-dir /workspace/nano-flash-artifacts/g1/chatterbox-nano-profile
