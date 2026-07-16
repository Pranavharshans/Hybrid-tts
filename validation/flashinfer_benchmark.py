#!/usr/bin/env python3
"""Benchmark Torch and FlashInfer Chatterbox block generation configurations."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from chatterbox_flash import ChatterboxFlashTTS


TEXT = "Hello. This is a deterministic optimized block generation benchmark."
SEED = 20260716
CONFIGS = (
    {"name": "torch-b16", "backend": "torch", "use_cuda_graph": False, "block_size": 16},
    {"name": "flashinfer-b16-eager", "backend": "flashinfer", "use_cuda_graph": False, "block_size": 16},
    {"name": "flashinfer-b8-graph", "backend": "flashinfer", "use_cuda_graph": True, "block_size": 8},
    {"name": "flashinfer-b16-graph", "backend": "flashinfer", "use_cuda_graph": True, "block_size": 16},
    {"name": "flashinfer-b32-graph", "backend": "flashinfer", "use_cuda_graph": True, "block_size": 32},
)


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    return hashlib.sha256(tensor.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def generate(model: Any, config: dict[str, Any], *, profile: bool = False) -> dict[str, Any]:
    captured = {}
    semantic_seconds = 0.0
    original = model.t3.generate

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal semantic_seconds
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = original(*args, **kwargs)
        torch.cuda.synchronize()
        semantic_seconds += time.perf_counter() - started
        captured["tokens"] = result.detach().cpu().clone()
        return result

    model.t3.generate = wrapper
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.cuda.synchronize()
    started = time.perf_counter()
    try:
        waveform = model.generate(
            TEXT,
            num_steps=10,
            temperature=0.6,
            time_shift_tau=0.1,
            omnivoice_schedule_t_shift=0.5,
            cfg_scale=1.0,
            position_temperature=5.0,
            pmi_uncond_prior_precompute=True,
            use_cuda_graph=config["use_cuda_graph"],
            backend=config["backend"],
            n_cfm_timesteps=2,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
    finally:
        model.t3.generate = original
    waveform = waveform.detach().cpu().float().flatten()
    duration = waveform.numel() / model.sr
    return {
        "elapsed_seconds": elapsed,
        "semantic_seconds": semantic_seconds,
        "acoustic_and_other_seconds": elapsed - semantic_seconds,
        "duration_seconds": duration,
        "rtf": elapsed / duration,
        "semantic_rtf": semantic_seconds / duration,
        "speech_tokens": int(captured["tokens"].numel()),
        "speech_token_sha256": tensor_sha256(captured["tokens"]),
        "waveform_sha256": tensor_sha256(waveform),
        "finite": bool(torch.isfinite(waveform).all()),
        "rms": float(torch.sqrt(torch.mean(waveform.square()))),
        "profile": profile,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--prompt-audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = []
    for config in CONFIGS:
        record = {"configuration": config}
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            load_started = time.perf_counter()
            model = ChatterboxFlashTTS.from_local(
                args.snapshot, "cuda", dtype=torch.bfloat16, drf_block_size=config["block_size"]
            )
            model.prepare_conditionals(args.prompt_audio, exaggeration=0.5)
            record["load_and_condition_seconds"] = time.perf_counter() - load_started
            warmup_started = time.perf_counter()
            warmup = generate(model, config)
            record["warmup_seconds"] = time.perf_counter() - warmup_started
            runs = [generate(model, config) for _ in range(3)]
            token_hashes = {run["speech_token_sha256"] for run in runs}
            waveform_hashes = {run["waveform_sha256"] for run in runs}
            record.update(
                {
                    "status": "pass",
                    "warmup": warmup,
                    "runs": runs,
                    "rtf_p50": float(np.percentile([run["rtf"] for run in runs], 50)),
                    "rtf_p95": float(np.percentile([run["rtf"] for run in runs], 95)),
                    "semantic_rtf_p50": float(np.percentile([run["semantic_rtf"] for run in runs], 50)),
                    "deterministic_tokens": len(token_hashes) == 1,
                    "deterministic_waveform": len(waveform_hashes) == 1,
                    "finite_non_silent": all(run["finite"] and run["rms"] > 1e-5 for run in runs),
                    "peak_allocated_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
                }
            )
        except Exception as error:
            record.update(
                {
                    "status": "error",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
            )
        finally:
            if "model" in locals():
                del model
            gc.collect()
            torch.cuda.empty_cache()
        results.append(record)
    passed = [record for record in results if record["status"] == "pass"]
    torch_baseline = next((record for record in passed if record["configuration"]["name"] == "torch-b16"), None)
    flash_passed = [record for record in passed if record["configuration"]["backend"] == "flashinfer"]
    best_flash = min(flash_passed, key=lambda record: record["rtf_p50"]) if flash_passed else None
    checks = {
        "torch_baseline_passed": torch_baseline is not None,
        "at_least_one_flashinfer_passed": bool(flash_passed),
        "all_passed_outputs_valid": all(record["finite_non_silent"] and record["deterministic_tokens"] for record in passed),
        "best_flash_faster_than_torch": bool(best_flash and best_flash["rtf_p50"] < torch_baseline["rtf_p50"]),
        "fits_16gb": all(record["peak_allocated_vram_gib"] < 14.5 for record in passed),
    }
    evidence = {
        "schema_version": 1,
        "configs": results,
        "summary": {
            "passed_configs": len(passed),
            "best_flash_config": None if best_flash is None else best_flash["configuration"]["name"],
            "best_flash_rtf_p50": None if best_flash is None else best_flash["rtf_p50"],
            "torch_rtf_p50": None if torch_baseline is None else torch_baseline["rtf_p50"],
            "speedup_over_torch": None if best_flash is None else torch_baseline["rtf_p50"] / best_flash["rtf_p50"],
            "meets_block_rtf_target": bool(best_flash and best_flash["rtf_p50"] < 0.2),
        },
        "checks": checks,
        "pass": all(checks.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"summary": evidence["summary"], "statuses": {r["configuration"]["name"]: r["status"] for r in results}, "checks": checks, "pass": evidence["pass"]}, indent=2, sort_keys=True))
    return 0 if evidence["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
