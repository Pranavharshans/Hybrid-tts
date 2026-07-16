#!/usr/bin/env bash
set -euo pipefail

source /venv/main/bin/activate
cd /workspace/Hybrid-tts

python validation/moss_smoke.py \
  --moss-repo /workspace/external/MOSS-TTS-Nano \
  --output-dir /workspace/nano-flash-artifacts/g1/moss-smoke
