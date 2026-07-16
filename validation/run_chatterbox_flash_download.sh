#!/usr/bin/env bash
set -euo pipefail

source /workspace/venvs/chatterbox-flash/bin/activate
cd /workspace/Hybrid-tts
export HF_HOME=/workspace/.hf_home

python validation/download_chatterbox_flash.py
