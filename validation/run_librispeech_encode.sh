#!/usr/bin/env bash
set -euo pipefail

source /venv/main/bin/activate
cd /workspace/Hybrid-tts
export HF_HOME=/workspace/.hf_home
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="/workspace/Hybrid-tts/validation/compat${PYTHONPATH:+:${PYTHONPATH}}"

ROOT=/workspace/nano-flash-artifacts/g4/librispeech-subset
CODEC=/workspace/.hf_home/hub/models--OpenMOSS-Team--MOSS-Audio-Tokenizer-Nano/snapshots/6aa02b01e445cc585582cf0ba480bc3ea6c8dd68

python /workspace/external/MOSS-TTS-Nano/finetuning/prepare_data.py \
  --codec-path "$CODEC" \
  --input-jsonl "$ROOT/train.raw.jsonl" \
  --output-jsonl "$ROOT/train.prepared.jsonl" \
  --batch-size 4 \
  --skip-reference-audio-codes

python /workspace/external/MOSS-TTS-Nano/finetuning/prepare_data.py \
  --codec-path "$CODEC" \
  --input-jsonl "$ROOT/valid.raw.jsonl" \
  --output-jsonl "$ROOT/valid.prepared.jsonl" \
  --batch-size 4 \
  --skip-reference-audio-codes

python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path("/workspace/nano-flash-artifacts/g4/librispeech-subset")
summary = {}
checks = {}
for split, expected in (("train", 100), ("valid", 20)):
    path = root / f"{split}.prepared.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    widths = {len(row) for record in records for row in record["audio_codes"]}
    lengths = [len(record["audio_codes"]) for record in records]
    summary[split] = {
        "records": len(records),
        "min_frames": min(lengths),
        "max_frames": max(lengths),
        "mean_frames": sum(lengths) / len(lengths),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    checks[f"{split}_record_count"] = len(records) == expected
    checks[f"{split}_vq_width_16"] = widths == {16}
    checks[f"{split}_nonempty_codes"] = min(lengths) > 0
evidence = {"schema_version": 1, "summary": summary, "checks": checks, "pass": all(checks.values())}
output = root / "encoding.json"
temporary = output.with_suffix(".json.tmp")
temporary.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
os.replace(temporary, output)
print(json.dumps(evidence, indent=2, sort_keys=True))
raise SystemExit(0 if evidence["pass"] else 1)
PY
