#!/usr/bin/env bash
set -euo pipefail
cd /workspace/Hybrid-tts
source /venv/main/bin/activate
export PYTHONPATH=/workspace/Hybrid-tts/validation
python validation/final_adversarial_matrix.py
