#!/usr/bin/env bash
set -euo pipefail

source /workspace/venvs/chatterbox-flash/bin/activate
cd /workspace/Hybrid-tts
export HF_HOME=/workspace/.hf_home
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CHATTERBOX_FLASH_ENGINE=torch

python validation/chatterbox_flash_profile.py \
  --flash-source /workspace/external/chatterbox-flash \
  --snapshot /workspace/.hf_home/hub/models--ResembleAI--chatterbox-flash/snapshots/4385507288b8197e6dab8b4e6b1603328d549d9d \
  --prompt-audio /workspace/external/MOSS-TTS-Nano/assets/audio/en_6.wav \
  --output-dir /workspace/nano-flash-artifacts/g1/chatterbox-flash-profile
