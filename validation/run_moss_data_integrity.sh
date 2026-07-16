#!/usr/bin/env bash
set -euo pipefail

source /venv/main/bin/activate
cd /workspace/Hybrid-tts
export HF_HOME=/workspace/.hf_home
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="/workspace/Hybrid-tts/validation/compat${PYTHONPATH:+:${PYTHONPATH}}"

OUTPUT_DIR=/workspace/nano-flash-artifacts/g3/moss-data-integrity
MODEL=/workspace/.hf_home/hub/models--OpenMOSS-Team--MOSS-TTS-Nano/snapshots/44502f80dbf9743528fa921cc544d662c685ebec
CODEC=/workspace/.hf_home/hub/models--OpenMOSS-Team--MOSS-Audio-Tokenizer-Nano/snapshots/6aa02b01e445cc585582cf0ba480bc3ea6c8dd68
mkdir -p "$OUTPUT_DIR"

python validation/build_moss_overfit_manifest.py \
  --audio /workspace/external/MOSS-TTS-Nano/assets/audio/en_6.wav \
  --output "$OUTPUT_DIR/raw.jsonl"

python /workspace/external/MOSS-TTS-Nano/finetuning/prepare_data.py \
  --codec-path "$CODEC" \
  --input-jsonl "$OUTPUT_DIR/raw.jsonl" \
  --output-jsonl "$OUTPUT_DIR/prepared.jsonl" \
  --batch-size 1 \
  --skip-reference-audio-codes

python validation/moss_data_integrity.py \
  --moss-repo /workspace/external/MOSS-TTS-Nano \
  --model-snapshot "$MODEL" \
  --prepared-jsonl "$OUTPUT_DIR/prepared.jsonl" \
  --output "$OUTPUT_DIR/integrity.json"
