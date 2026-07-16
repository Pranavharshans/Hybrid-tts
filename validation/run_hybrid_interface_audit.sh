#!/usr/bin/env bash
set -euo pipefail

source /workspace/venvs/chatterbox-flash/bin/activate
cd /workspace/Hybrid-tts

ROOT=/workspace/nano-flash-artifacts/g8/interface-audit
mkdir -p "$ROOT"
set +e
python validation/hybrid_interface_audit.py \
  --moss-config /workspace/.hf_home/hub/models--OpenMOSS-Team--MOSS-TTS-Nano/snapshots/44502f80dbf9743528fa921cc544d662c685ebec/config.json \
  --flash-checkpoint /workspace/.hf_home/hub/models--ResembleAI--chatterbox-flash/snapshots/4385507288b8197e6dab8b4e6b1603328d549d9d/t3_flash.safetensors \
  --flash-audio /workspace/nano-flash-artifacts/g7/flash-quality/flash-b32/plain.wav \
  --moss-source /workspace/external/MOSS-TTS-Nano \
  --flash-source /workspace/external/chatterbox-flash \
  --output "$ROOT/audit.json"
code=$?
set -e
test "$code" -eq 2
