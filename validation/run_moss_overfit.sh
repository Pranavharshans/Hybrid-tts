#!/usr/bin/env bash
set -euo pipefail

source /venv/main/bin/activate
cd /workspace/Hybrid-tts
export HF_HOME=/workspace/.hf_home
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="/workspace/Hybrid-tts/validation/compat${PYTHONPATH:+:${PYTHONPATH}}"

ROOT=/workspace/nano-flash-artifacts/g3/moss-overfit
MODEL=/workspace/.hf_home/hub/models--OpenMOSS-Team--MOSS-TTS-Nano/snapshots/44502f80dbf9743528fa921cc544d662c685ebec
CODEC=/workspace/.hf_home/hub/models--OpenMOSS-Team--MOSS-Audio-Tokenizer-Nano/snapshots/6aa02b01e445cc585582cf0ba480bc3ea6c8dd68
TRAIN=/workspace/nano-flash-artifacts/g3/moss-data-integrity/prepared.jsonl
mkdir -p "$ROOT"

python /workspace/external/MOSS-TTS-Nano/finetuning/sft.py \
  --model-path "$MODEL" \
  --codec-path "$CODEC" \
  --train-jsonl "$TRAIN" \
  --output-dir "$ROOT/checkpoints" \
  --max-length 512 \
  --per-device-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --learning-rate 0.0001 \
  --weight-decay 0 \
  --warmup-ratio 0 \
  --num-epochs 1 \
  --max-train-steps 40 \
  --max-grad-norm 1 \
  --logging-steps 1 \
  --save-every-epochs 1 \
  --mixed-precision bf16 \
  --attn-implementation sdpa \
  --channelwise-loss-weight 1,32 \
  --seed 20260716

python validation/moss_overfit_validate.py \
  --log /workspace/nano-flash-artifacts/g3/moss-overfit-supervisor.log \
  --checkpoint "$ROOT/checkpoints/checkpoint-last" \
  --output "$ROOT/overfit.json"
