#!/usr/bin/env bash
set -euo pipefail

source /venv/main/bin/activate
cd /workspace/Hybrid-tts

python validation/download_librispeech_subset.py \
  --output-dir /workspace/nano-flash-artifacts/g4/librispeech-subset
