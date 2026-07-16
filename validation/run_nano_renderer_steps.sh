#!/usr/bin/env bash
set -euo pipefail

source /workspace/venvs/chatterbox-nano/bin/activate
cd /workspace/Hybrid-tts
export PYTHONPATH=/workspace/external/chatterbox-nano-demo
export HF_HOME=/workspace/.hf_home
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python validation/nano_renderer_steps.py \
  --snapshot /workspace/.hf_home/hub/models--ResembleAI--chatterbox-nano/snapshots/493317046f21b7e557146a9285a111c050564bb4 \
  --cache-dir /workspace/nano-flash-artifacts/g6/nano-renderer-cache \
  --output /workspace/nano-flash-artifacts/g6/nano-renderer-steps.json
