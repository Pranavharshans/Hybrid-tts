#!/usr/bin/env bash
set -euo pipefail

VENV=/workspace/venvs/chatterbox-flash
PYTHON="$VENV/bin/python"
CHATTERBOX_REPO=/workspace/external/chatterbox
FLASH_REPO=/workspace/external/chatterbox-flash
EXPECTED_CHATTERBOX_COMMIT=65b18437192794391a0308a8f705b1e33e633948
EXPECTED_FLASH_COMMIT=74e05baa8ce574bf2cc571702391a21f1b0d48c5
OUTPUT_DIR=/workspace/nano-flash-artifacts/g1/chatterbox-flash-setup

test "$(git -C "$CHATTERBOX_REPO" rev-parse HEAD)" = "$EXPECTED_CHATTERBOX_COMMIT"
test "$(git -C "$FLASH_REPO" rev-parse HEAD)" = "$EXPECTED_FLASH_COMMIT"
mkdir -p "$OUTPUT_DIR" "$(dirname "$VENV")"

if [[ ! -x "$PYTHON" ]]; then
  uv venv --python 3.12 "$VENV"
fi

# Torch 2.7.1 + CUDA 12.8 is the newest upstream-tested ABI family and the
# earliest practical Blackwell build for Chatterbox-Flash. Keep it isolated
# from the MOSS environment, which uses Torch 2.12 + CUDA 13.0.
uv pip install --python "$PYTHON" \
  --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple \
  --index-strategy unsafe-best-match \
  'torch==2.7.1+cu128' 'torchaudio==2.7.1+cu128' 'torchvision==0.22.1+cu128'

uv pip install --python "$PYTHON" \
  'numpy>=1.24,<2' \
  'librosa==0.11.0' \
  s3tokenizer \
  'transformers==5.2.0' \
  'diffusers==0.29.0' \
  'resemble-perth @ git+https://github.com/resemble-ai/Perth.git@master' \
  'conformer==0.3.2' \
  'safetensors==0.5.3' \
  spacy-pkuseg \
  'pykakasi==2.3.0' \
  'gradio==6.8.0' \
  pyloudnorm \
  omegaconf \
  'huggingface-hub>=0.24' \
  'tqdm>=4.66' \
  'soundfile>=0.12' \
  'scipy>=1.11' \
  regex \
  'inflect>=7.0' \
  'Unidecode>=1.3' \
  'pyyaml>=6.0' \
  einops

uv pip install --python "$PYTHON" --no-deps -e "$CHATTERBOX_REPO"
uv pip install --python "$PYTHON" --no-deps -e "$FLASH_REPO"

export HF_HOME=/workspace/.hf_home
export CHATTERBOX_FLASH_ENGINE=torch
export OUTPUT_DIR CHATTERBOX_REPO FLASH_REPO

"$PYTHON" - <<'PY'
import importlib.metadata
import json
import os
import subprocess
import time
from pathlib import Path

import torch
import torchaudio
import transformers
from chatterbox_flash import ChatterboxFlashTTS

del ChatterboxFlashTTS

output_dir = Path(os.environ["OUTPUT_DIR"])
output_path = output_dir / "environment.json"
temporary = output_path.with_suffix(".json.tmp")

torch.cuda.empty_cache()
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
    "torch_cuda_12_8": str(torch.version.cuda).startswith("12.8"),
    "torchaudio_abi_matches": torchaudio.__version__.split("+")[0] == torch.__version__.split("+")[0],
    "chatterbox_flash_importable": importlib.metadata.version("chatterbox-flash") == "0.1.0",
}
evidence = {
    "schema_version": 1,
    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "chatterbox_commit": subprocess.run(
        ["git", "-C", os.environ["CHATTERBOX_REPO"], "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip(),
    "chatterbox_flash_commit": subprocess.run(
        ["git", "-C", os.environ["FLASH_REPO"], "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip(),
    "python": os.sys.version.split()[0],
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "torchaudio": torchaudio.__version__,
    "transformers": transformers.__version__,
    "gpu": torch.cuda.get_device_name(0),
    "compute_capability": list(torch.cuda.get_device_capability(0)),
    "bf16_cuda_probe_seconds": round(cuda_probe_seconds, 6),
    "flash_engine": os.environ["CHATTERBOX_FLASH_ENGINE"],
    "checks": checks,
    "pass": all(checks.values()),
}
temporary.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
os.replace(temporary, output_path)
print(json.dumps(evidence, indent=2, sort_keys=True))
raise SystemExit(0 if evidence["pass"] else 1)
PY
