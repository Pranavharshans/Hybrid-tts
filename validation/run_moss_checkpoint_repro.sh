#!/usr/bin/env bash
set -euo pipefail

source /venv/main/bin/activate
cd /workspace/Hybrid-tts
export HF_HOME=/workspace/.hf_home
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="/workspace/Hybrid-tts/validation/compat:/workspace/external/MOSS-TTS-Nano${PYTHONPATH:+:${PYTHONPATH}}"

python validation/moss_checkpoint_repro.py \
  --moss-repo /workspace/external/MOSS-TTS-Nano \
  --model /workspace/nano-flash-artifacts/g3/moss-overfit/checkpoints/checkpoint-last \
  --train-jsonl /workspace/nano-flash-artifacts/g3/moss-data-integrity/prepared.jsonl \
  --output-dir /workspace/nano-flash-artifacts/g3/moss-checkpoint-repro
