#!/usr/bin/env bash
set -euo pipefail

VENV=/workspace/venvs/chatterbox-nano
PYTHON="$VENV/bin/python"
NANO_SOURCE=/workspace/external/chatterbox-nano-demo
NANO_SNAPSHOT=/workspace/.hf_home/hub/models--ResembleAI--chatterbox-nano/snapshots/493317046f21b7e557146a9285a111c050564bb4
EXPECTED_SOURCE_COMMIT=647b4e895d3483995e5a6546999aa5e50490b92b
EXPECTED_SNAPSHOT_REVISION=493317046f21b7e557146a9285a111c050564bb4
OUTPUT_DIR=/workspace/nano-flash-artifacts/g1/chatterbox-nano-setup

test "$(git -C "$NANO_SOURCE" rev-parse HEAD)" = "$EXPECTED_SOURCE_COMMIT"
test "$(basename "$NANO_SNAPSHOT")" = "$EXPECTED_SNAPSHOT_REVISION"
test -f "$NANO_SNAPSHOT/t3_nano_v1.safetensors"
test -f "$NANO_SNAPSHOT/s3gen_meanflow.safetensors"
mkdir -p "$OUTPUT_DIR" "$(dirname "$VENV")"

if [[ ! -x "$PYTHON" ]]; then
  uv venv --python 3.12 "$VENV"
fi

# This matches the official Nano demo Space's framework family while using
# CUDA 13.0 wheels for native Blackwell support. It remains isolated from the
# MOSS and Chatterbox-Flash environments.
uv pip install --python "$PYTHON" \
  --index-url https://download.pytorch.org/whl/cu130 \
  --extra-index-url https://pypi.org/simple \
  --index-strategy unsafe-best-match \
  'torch==2.11.0+cu130' 'torchaudio==2.11.0+cu130'

uv pip install --python "$PYTHON" \
  'numpy>=1.26,<2' \
  'librosa==0.11.0' \
  s3tokenizer \
  'transformers==4.46.3' \
  'huggingface-hub>=0.26,<1.0' \
  'diffusers==0.29.0' \
  'resemble-perth==1.0.1' \
  'setuptools<81' \
  'conformer==0.3.2' \
  'safetensors==0.5.3' \
  spacy-pkuseg \
  'pykakasi==2.3.0' \
  pyloudnorm \
  omegaconf

export PYTHONPATH="$NANO_SOURCE"
export HF_HOME=/workspace/.hf_home
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export OUTPUT_DIR NANO_SOURCE NANO_SNAPSHOT EXPECTED_SNAPSHOT_REVISION

"$PYTHON" - <<'PY'
import json
import os
import subprocess
import time
from pathlib import Path

import torch
import torchaudio
import transformers
from chatterbox.tts_nano import ChatterboxNanoTTS

del ChatterboxNanoTTS

output_path = Path(os.environ["OUTPUT_DIR"]) / "environment.json"
temporary = output_path.with_suffix(".json.tmp")

left = torch.randn((512, 512), device="cuda", dtype=torch.bfloat16)
right = torch.randn((512, 512), device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize()
started = time.perf_counter()
result = left @ right
torch.cuda.synchronize()
cuda_probe_seconds = time.perf_counter() - started

checks = {
    "cuda_available": torch.cuda.is_available(),
    "blackwell_visible": torch.cuda.get_device_capability(0)[0] >= 12,
    "bf16_probe_finite": bool(torch.isfinite(result).all().item()),
    "torch_cuda_13_0": str(torch.version.cuda).startswith("13.0"),
    "torchaudio_abi_matches": torchaudio.__version__.split("+")[0] == torch.__version__.split("+")[0],
    "nano_loader_importable": True,
    "snapshot_revision_pinned": Path(os.environ["NANO_SNAPSHOT"]).name == os.environ["EXPECTED_SNAPSHOT_REVISION"],
}
evidence = {
    "schema_version": 1,
    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "nano_source_commit": subprocess.run(
        ["git", "-C", os.environ["NANO_SOURCE"], "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip(),
    "nano_snapshot_revision": Path(os.environ["NANO_SNAPSHOT"]).name,
    "python": os.sys.version.split()[0],
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "torchaudio": torchaudio.__version__,
    "transformers": transformers.__version__,
    "gpu": torch.cuda.get_device_name(0),
    "compute_capability": list(torch.cuda.get_device_capability(0)),
    "bf16_cuda_probe_seconds": round(cuda_probe_seconds, 6),
    "checks": checks,
    "pass": all(checks.values()),
}
temporary.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
os.replace(temporary, output_path)
print(json.dumps(evidence, indent=2, sort_keys=True))
raise SystemExit(0 if evidence["pass"] else 1)
PY
