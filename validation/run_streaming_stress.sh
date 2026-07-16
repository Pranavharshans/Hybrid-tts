#!/usr/bin/env bash
set -euo pipefail

source /venv/main/bin/activate
cd /workspace/Hybrid-tts
export PYTHONPATH=/workspace/Hybrid-tts/validation

python validation/streaming_stress.py
