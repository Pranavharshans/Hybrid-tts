#!/usr/bin/env bash
set -euo pipefail

source /venv/main/bin/activate
cd /workspace/Hybrid-tts
export HF_HOME=/workspace/.hf_home
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="/workspace/Hybrid-tts/validation/compat:/workspace/external/MOSS-TTS-Nano${PYTHONPATH:+:${PYTHONPATH}}"

ROOT=/workspace/nano-flash-artifacts/g4/moss-lean-adaptation
DATA=/workspace/nano-flash-artifacts/g4/librispeech-subset
MODEL=/workspace/nano-flash-artifacts/g3/moss-overfit/merged-model
CODEC=/workspace/.hf_home/hub/models--OpenMOSS-Team--MOSS-Audio-Tokenizer-Nano/snapshots/6aa02b01e445cc585582cf0ba480bc3ea6c8dd68
mkdir -p "$ROOT"

python validation/moss_eval_loss.py \
  --moss-repo /workspace/external/MOSS-TTS-Nano \
  --model "$MODEL" \
  --prepared-jsonl "$DATA/valid.prepared.jsonl" \
  --output "$ROOT/baseline-heldout.json"

python /workspace/external/MOSS-TTS-Nano/finetuning/sft.py \
  --model-path "$MODEL" \
  --codec-path "$CODEC" \
  --train-jsonl "$DATA/train.prepared.jsonl" \
  --output-dir "$ROOT/checkpoints" \
  --max-length 512 \
  --per-device-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --learning-rate 0.00001 \
  --weight-decay 0.01 \
  --warmup-steps 5 \
  --num-epochs 1 \
  --max-train-steps 100 \
  --max-grad-norm 1 \
  --logging-steps 1 \
  --save-every-epochs 1 \
  --mixed-precision bf16 \
  --attn-implementation sdpa \
  --channelwise-loss-weight 1,32 \
  --seed 20260716

python validation/moss_eval_loss.py \
  --moss-repo /workspace/external/MOSS-TTS-Nano \
  --model "$ROOT/checkpoints/checkpoint-last" \
  --prepared-jsonl "$DATA/valid.prepared.jsonl" \
  --output "$ROOT/adapted-heldout.json"

python - <<'PY'
import hashlib
import json
import os
import re
from pathlib import Path

root = Path("/workspace/nano-flash-artifacts/g4/moss-lean-adaptation")
baseline = json.loads((root / "baseline-heldout.json").read_text())
adapted = json.loads((root / "adapted-heldout.json").read_text())
log = Path("/workspace/nano-flash-artifacts/g4/moss-lean-adaptation-supervisor.log").read_text(errors="replace")
matches = re.findall(r"step=(\d+)/(\d+)\s+loss=([0-9.eE+-]+)", log)
losses = [float(match[2]) for match in matches]
ratio = adapted["mean_loss"] / baseline["mean_loss"]
weight = root / "checkpoints/checkpoint-last/pytorch_model.bin"
checks = {
    "one_hundred_steps": len(losses) >= 100 and int(matches[-1][0]) == 100,
    "training_loss_finite": all(value == value and abs(value) != float("inf") for value in losses),
    "late_train_loss_below_early": sum(losses[-10:]) / 10 < sum(losses[:10]) / 10,
    "heldout_losses_valid": baseline["pass"] and adapted["pass"],
    "heldout_loss_improved": ratio < 1.0,
    "checkpoint_present": weight.is_file(),
}
digest = hashlib.sha256()
with weight.open("rb") as handle:
    for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
        digest.update(chunk)
evidence = {
    "schema_version": 1,
    "steps": len(losses),
    "early_train_loss_mean": sum(losses[:10]) / 10,
    "late_train_loss_mean": sum(losses[-10:]) / 10,
    "baseline_heldout_mean_loss": baseline["mean_loss"],
    "adapted_heldout_mean_loss": adapted["mean_loss"],
    "heldout_loss_ratio": ratio,
    "checkpoint_bytes": weight.stat().st_size,
    "checkpoint_sha256": digest.hexdigest(),
    "checks": checks,
    "pass": all(checks.values()),
}
output = root / "adaptation.json"
temporary = output.with_suffix(".json.tmp")
temporary.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
os.replace(temporary, output)
print(json.dumps(evidence, indent=2, sort_keys=True))
raise SystemExit(0 if evidence["pass"] else 1)
PY
