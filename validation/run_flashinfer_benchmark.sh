#!/usr/bin/env bash
set -euo pipefail

source /workspace/venvs/chatterbox-flash/bin/activate
cd /workspace/Hybrid-tts
export HF_HOME=/workspace/.hf_home
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export FLASHINFER_CACHEDIR=/workspace/.flashinfer

python validation/flashinfer_benchmark.py \
  --snapshot /workspace/.hf_home/hub/models--ResembleAI--chatterbox-flash/snapshots/4385507288b8197e6dab8b4e6b1603328d549d9d \
  --prompt-audio /workspace/external/MOSS-TTS-Nano/assets/audio/en_6.wav \
  --output /workspace/nano-flash-artifacts/g7/flashinfer-benchmark.json
