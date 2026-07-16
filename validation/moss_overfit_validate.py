#!/usr/bin/env python3
"""Validate deliberate-overfit loss dynamics and checkpoint integrity."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import torch


LOSS_RE = re.compile(r"step=(\d+)/(\d+)\s+loss=([0-9.eE+-]+)")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    matches = LOSS_RE.findall(args.log.read_text(errors="replace"))
    steps = [int(match[0]) for match in matches]
    losses = [float(match[2]) for match in matches]
    weight_path = args.checkpoint / "pytorch_model.bin"
    state = torch.load(weight_path, map_location="cpu", weights_only=True)
    tensor_count = sum(torch.is_tensor(value) for value in state.values())
    parameter_values = [value for value in state.values() if torch.is_tensor(value)]
    checks = {
        "at_least_30_logged_steps": len(losses) >= 30,
        "reached_step_40": max(steps, default=0) == 40,
        "all_losses_finite": bool(losses) and all(torch.isfinite(torch.tensor(losses)).tolist()),
        "final_loss_below_initial": bool(losses) and losses[-1] < losses[0],
        "final_loss_below_80_percent_initial": bool(losses) and losses[-1] < losses[0] * 0.80,
        "best_loss_below_70_percent_initial": bool(losses) and min(losses) < losses[0] * 0.70,
        "checkpoint_tensors_present": tensor_count > 100,
        "checkpoint_tensors_finite": all(torch.isfinite(value).all().item() for value in parameter_values),
        "metadata_present": (args.checkpoint / "finetune_config.json").is_file(),
        "config_present": (args.checkpoint / "config.json").is_file(),
    }
    evidence = {
        "schema_version": 1,
        "logged_steps": len(losses),
        "first_step": steps[0] if steps else None,
        "last_step": steps[-1] if steps else None,
        "initial_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "best_loss": min(losses) if losses else None,
        "loss_ratio_final_to_initial": losses[-1] / losses[0] if losses else None,
        "losses": [{"step": step, "loss": loss} for step, loss in zip(steps, losses)],
        "checkpoint_tensor_count": tensor_count,
        "checkpoint_bytes": weight_path.stat().st_size,
        "checkpoint_sha256": file_sha256(weight_path),
        "checks": checks,
        "pass": all(checks.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0 if evidence["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
