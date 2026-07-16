#!/usr/bin/env bash
set -euo pipefail

source /venv/main/bin/activate
cd /workspace/Hybrid-tts
export HF_HOME=/workspace/.hf_home
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="/workspace/Hybrid-tts/validation/compat${PYTHONPATH:+:${PYTHONPATH}}"

python validation/moss_profile.py \
  --moss-repo /workspace/external/MOSS-TTS-Nano \
  --output-dir /workspace/nano-flash-artifacts/g1/moss-profile \
  --model-snapshot /workspace/.hf_home/hub/models--OpenMOSS-Team--MOSS-TTS-Nano/snapshots/44502f80dbf9743528fa921cc544d662c685ebec \
  --text-tokenizer-snapshot /workspace/.hf_home/modules/transformers_modules/OpenMOSS_hyphen_Team/MOSS_hyphen_TTS_hyphen_Nano/44502f80dbf9743528fa921cc544d662c685ebec/.cache/huggingface/models--OpenMOSS-Team--MOSS-TTS-Nano/snapshots/44502f80dbf9743528fa921cc544d662c685ebec \
  --audio-tokenizer-snapshot /workspace/.hf_home/hub/models--OpenMOSS-Team--MOSS-Audio-Tokenizer-Nano/snapshots/6aa02b01e445cc585582cf0ba480bc3ea6c8dd68
