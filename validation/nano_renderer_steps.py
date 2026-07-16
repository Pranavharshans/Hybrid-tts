#!/usr/bin/env python3
"""Compare one-step Nano meanflow rendering with cached two-step targets."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch


def move_to(value: Any, device: str) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to(child, device) for key, child in value.items()}
    if isinstance(value, list):
        return [move_to(child, device) for child in value]
    if isinstance(value, tuple):
        return tuple(move_to(child, device) for child in value)
    return value


def spectral_distance(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    window = torch.hann_window(1024)
    ref = torch.stft(reference.flatten(), 1024, hop_length=256, window=window, return_complex=True).abs()
    cand = torch.stft(candidate.flatten(), 1024, hop_length=256, window=window, return_complex=True).abs()
    return float(torch.mean(torch.abs(torch.log1p(ref) - torch.log1p(cand))))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from chatterbox.tts_nano import ChatterboxNanoTTS

    model = ChatterboxNanoTTS.from_local(args.snapshot.resolve(), "cuda")
    conditionals = torch.load(args.cache_dir / "conditionals.pt", map_location="cpu", weights_only=True)
    ref_dict = move_to(conditionals["gen"], "cuda")
    records = []
    for cache_path in sorted(args.cache_dir.glob("*.pt")):
        if cache_path.name == "conditionals.pt":
            continue
        cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        tokens = cached["tokens"].cuda()
        target = cached["waveform"].float()

        torch.set_rng_state(cached["torch_rng"])
        torch.cuda.set_rng_state_all(cached["cuda_rng"])
        torch.cuda.synchronize()
        started = time.perf_counter()
        two_step, _ = model.s3gen.inference(
            speech_tokens=tokens, ref_dict=ref_dict, n_cfm_timesteps=2
        )
        torch.cuda.synchronize()
        two_seconds = time.perf_counter() - started
        two_step = two_step.detach().cpu().float()

        torch.set_rng_state(cached["torch_rng"])
        torch.cuda.set_rng_state_all(cached["cuda_rng"])
        torch.cuda.synchronize()
        started = time.perf_counter()
        one_step, _ = model.s3gen.inference(
            speech_tokens=tokens, ref_dict=ref_dict, n_cfm_timesteps=1
        )
        torch.cuda.synchronize()
        one_seconds = time.perf_counter() - started
        one_step = one_step.detach().cpu().float()
        duration = target.shape[-1] / model.sr
        error = one_step - target
        reference_rms = float(torch.sqrt(torch.mean(target.square())))
        error_rms = float(torch.sqrt(torch.mean(error.square())))
        cosine = float(torch.nn.functional.cosine_similarity(target.flatten(), one_step.flatten(), dim=0))
        records.append(
            {
                "id": cache_path.stem,
                "duration_seconds": duration,
                "two_step_exact_target": torch.equal(two_step, target),
                "two_step_seconds": two_seconds,
                "two_step_rtf": two_seconds / duration,
                "one_step_seconds": one_seconds,
                "one_step_rtf": one_seconds / duration,
                "speedup": two_seconds / one_seconds,
                "one_step_finite": bool(torch.isfinite(one_step).all()),
                "same_shape": tuple(one_step.shape) == tuple(target.shape),
                "normalized_rmse": error_rms / max(reference_rms, 1e-12),
                "cosine_similarity": cosine,
                "log_spectral_l1": spectral_distance(target, one_step),
                "snr_db": 20 * math.log10(max(reference_rms, 1e-12) / max(error_rms, 1e-12)),
            }
        )
    checks = {
        "ten_pairs": len(records) == 10,
        "two_step_cache_reproduced": all(record["two_step_exact_target"] for record in records),
        "one_step_structurally_valid": all(record["one_step_finite"] and record["same_shape"] for record in records),
        "one_step_rtf_below_point_two": all(record["one_step_rtf"] < 0.2 for record in records),
    }
    evidence = {
        "schema_version": 1,
        "records": records,
        "summary": {
            key: sum(record[key] for record in records) / len(records)
            for key in ("two_step_rtf", "one_step_rtf", "speedup", "normalized_rmse", "cosine_similarity", "log_spectral_l1", "snr_db")
        },
        "checks": checks,
        "pass": all(checks.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"summary": evidence["summary"], "checks": checks, "pass": evidence["pass"]}, indent=2, sort_keys=True))
    return 0 if evidence["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
