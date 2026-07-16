#!/usr/bin/env python3
"""Validate the minimum environment guarantees needed for unattended experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import socket
import tempfile
import time
from pathlib import Path
from typing import Any

import torch


MIB = 1024 * 1024
GIB = 1024 * MIB


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(MIB), b""):
            digest.update(chunk)
    return digest.hexdigest()


def io_probe(directory: Path, size_mib: int) -> dict[str, Any]:
    payload = bytes(4 * MIB)
    path = directory / "io-probe.bin"

    started = time.perf_counter()
    with path.open("wb") as handle:
        for _ in range(max(1, size_mib // 4)):
            handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    write_seconds = time.perf_counter() - started

    started = time.perf_counter()
    bytes_read = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * MIB), b""):
            bytes_read += len(chunk)
    read_seconds = time.perf_counter() - started
    path.unlink()

    actual_mib = bytes_read / MIB
    return {
        "size_mib": round(actual_mib, 3),
        "write_seconds": round(write_seconds, 4),
        "write_mib_s": round(actual_mib / write_seconds, 3),
        "read_seconds": round(read_seconds, 4),
        "read_mib_s": round(actual_mib / read_seconds, 3),
    }


def checkpoint_probe(directory: Path, device: torch.device) -> dict[str, Any]:
    torch.manual_seed(20260716)
    model = torch.nn.Sequential(
        torch.nn.Linear(128, 256),
        torch.nn.GELU(),
        torch.nn.Linear(256, 64),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    inputs = torch.randn(16, 128, device=device)
    targets = torch.randn(16, 64, device=device)

    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.mse_loss(model(inputs), targets)
    loss.backward()
    optimizer.step()

    expected = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    final_path = directory / "checkpoint-probe.pt"
    temporary_path = directory / "checkpoint-probe.pt.tmp"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss": float(loss.detach().cpu()),
        },
        temporary_path,
    )
    os.replace(temporary_path, final_path)
    digest_before = sha256(final_path)

    loaded = torch.load(final_path, map_location="cpu", weights_only=False)
    equality = all(torch.equal(expected[key], loaded["model"][key]) for key in expected)
    optimizer_present = bool(loaded["optimizer"]["state"])
    digest_after = sha256(final_path)
    final_path.unlink()

    return {
        "training_loss": round(float(loss.detach().cpu()), 8),
        "model_state_equal": equality,
        "optimizer_state_present": optimizer_present,
        "sha256_stable": digest_before == digest_after,
        "sha256": digest_before,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--disk-floor-gib", type=float, default=35.0)
    parser.add_argument("--io-size-mib", type=int, default=64)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    disk = shutil.disk_usage(args.workdir)
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")

    with tempfile.TemporaryDirectory(prefix="g0-", dir=args.workdir) as temporary:
        tempdir = Path(temporary)
        io_result = io_probe(tempdir, args.io_size_mib)
        checkpoint_result = checkpoint_probe(tempdir, device)

    result: dict[str, Any] = {
        "schema_version": 1,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": cuda_available,
        "gpu": torch.cuda.get_device_name(0) if cuda_available else None,
        "gpu_capability": list(torch.cuda.get_device_capability(0)) if cuda_available else None,
        "bf16_supported": torch.cuda.is_bf16_supported() if cuda_available else False,
        "disk": {
            "total_gib": round(disk.total / GIB, 3),
            "used_gib": round(disk.used / GIB, 3),
            "free_gib": round(disk.free / GIB, 3),
            "safety_floor_gib": args.disk_floor_gib,
        },
        "io": io_result,
        "checkpoint": checkpoint_result,
    }
    checks = {
        "cuda_available": cuda_available,
        "free_disk_above_floor": disk.free / GIB >= args.disk_floor_gib,
        "checkpoint_model_equal": checkpoint_result["model_state_equal"],
        "checkpoint_optimizer_present": checkpoint_result["optimizer_state_present"],
        "checkpoint_digest_stable": checkpoint_result["sha256_stable"],
        "io_nonzero": io_result["write_mib_s"] > 0 and io_result["read_mib_s"] > 0,
    }
    result["checks"] = checks
    result["pass"] = all(checks.values())

    temporary_output = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary_output, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
