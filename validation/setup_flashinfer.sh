#!/usr/bin/env bash
set -euo pipefail

VENV=/workspace/venvs/chatterbox-flash
PYTHON="$VENV/bin/python"
OUTPUT=/workspace/nano-flash-artifacts/g7/flashinfer-setup.json
mkdir -p /workspace/nano-flash-artifacts/g7 /workspace/.flashinfer

uv pip install --python "$PYTHON" \
  --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple \
  --index-strategy unsafe-best-match \
  'torch==2.7.1+cu128' \
  'flashinfer-python==0.6.11.post3'

export FLASHINFER_CACHEDIR=/workspace/.flashinfer
export OUTPUT
"$PYTHON" - <<'PY'
import importlib.metadata
import json
import os
import time
from pathlib import Path

import torch
import flashinfer
from chatterbox_flash.engines.flashinfer import flashinfer_available

residual = torch.randn(256, 512, device="cuda", dtype=torch.bfloat16)
hidden = torch.randn_like(residual)
weight = torch.ones(512, device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize()
started = time.perf_counter()
flashinfer.norm.fused_add_rmsnorm(hidden, residual, weight, 1e-6)
torch.cuda.synchronize()
kernel_seconds = time.perf_counter() - started
checks = {
    "torch_preserved": torch.__version__ == "2.7.1+cu128",
    "cuda_preserved": str(torch.version.cuda).startswith("12.8"),
    "blackwell_visible": torch.cuda.get_device_capability(0)[0] >= 12,
    "flashinfer_version_pinned": importlib.metadata.version("flashinfer-python") == "0.6.11.post3",
    "flashinfer_engine_available": flashinfer_available(),
    "fused_kernel_finite": bool(torch.isfinite(hidden).all() and torch.isfinite(residual).all()),
}
evidence = {
    "schema_version": 1,
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
    "compute_capability": list(torch.cuda.get_device_capability(0)),
    "flashinfer_python": importlib.metadata.version("flashinfer-python"),
    "fused_add_rmsnorm_cold_seconds": kernel_seconds,
    "checks": checks,
    "pass": all(checks.values()),
}
output = Path(os.environ["OUTPUT"])
temporary = output.with_suffix(".json.tmp")
temporary.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
os.replace(temporary, output)
print(json.dumps(evidence, indent=2, sort_keys=True))
raise SystemExit(0 if evidence["pass"] else 1)
PY
