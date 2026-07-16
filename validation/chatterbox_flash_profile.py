#!/usr/bin/env python3
"""Deterministic smoke and warm profile for official Chatterbox-Flash."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch


PROFILE_TEXT = "Hello. This is a deterministic Chatterbox Flash warm latency profile."
SEED = 20260716


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flash-source", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--prompt-audio", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    return parser.parse_args()


def sync() -> None:
    torch.cuda.synchronize()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    return sha256_bytes(
        str((str(tensor.dtype), tuple(tensor.shape))).encode() + tensor.numpy().tobytes()
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def run_once(model: Any, text: str, output: Path, instrument: bool = False) -> dict[str, Any]:
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    sync()
    torch.cuda.reset_peak_memory_stats()
    starting_allocated = torch.cuda.memory_allocated()
    stages = {"semantic_seconds": 0.0, "semantic_calls": 0, "acoustic_seconds": 0.0, "acoustic_calls": 0}
    captured: dict[str, torch.Tensor] = {}
    original_semantic = model.t3.generate
    original_acoustic = model.s3gen.inference

    def semantic_wrapper(*args: Any, **kwargs: Any) -> Any:
        if instrument:
            sync()
            started = time.perf_counter()
        result = original_semantic(*args, **kwargs)
        if instrument:
            sync()
            stages["semantic_seconds"] += time.perf_counter() - started
            stages["semantic_calls"] += 1
        captured["tokens"] = result.detach().cpu().clone()
        return result

    def acoustic_wrapper(*args: Any, **kwargs: Any) -> Any:
        if instrument:
            sync()
            started = time.perf_counter()
        result = original_acoustic(*args, **kwargs)
        if instrument:
            sync()
            stages["acoustic_seconds"] += time.perf_counter() - started
            stages["acoustic_calls"] += 1
        return result

    model.t3.generate = semantic_wrapper
    model.s3gen.inference = acoustic_wrapper
    started = time.perf_counter()
    try:
        waveform = model.generate(
            text,
            num_steps=10,
            temperature=0.6,
            time_shift_tau=0.1,
            omnivoice_schedule_t_shift=0.5,
            cfg_scale=1.0,
            position_temperature=5.0,
            pmi_uncond_prior_precompute=True,
            use_cuda_graph=False,
            backend="torch",
            n_cfm_timesteps=2,
        )
        sync()
        elapsed = time.perf_counter() - started
    finally:
        model.t3.generate = original_semantic
        model.s3gen.inference = original_acoustic

    waveform = waveform.detach().cpu().float().flatten()
    audio = waveform.numpy()
    duration = float(audio.size / model.sr)
    sf.write(str(output), audio, model.sr, subtype="PCM_16")
    tokens = captured["tokens"]
    stage_sum = float(stages["semantic_seconds"] + stages["acoustic_seconds"])
    checks = {
        "finite_audio": bool(np.isfinite(audio).all()),
        "non_silent_audio": float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))) > 1e-5,
        "duration_gt_half_second": duration > 0.5,
        "speech_tokens_present": tokens.numel() > 0,
        "output_exists": output.is_file(),
    }
    return {
        "elapsed_seconds": round(elapsed, 6),
        "time_to_first_audio_seconds": round(elapsed, 6),
        "full_utterance_return_api": True,
        "native_streaming_api_exposed": False,
        "duration_seconds": round(duration, 8),
        "rtf": round(elapsed / duration, 6),
        "sample_rate": int(model.sr),
        "speech_token_count": int(tokens.numel()),
        "speech_token_sha256": tensor_sha256(tokens),
        "wav_sha256": file_sha256(output),
        "peak_allocated_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 4),
        "peak_allocated_delta_vram_gib": round((torch.cuda.max_memory_allocated() - starting_allocated) / 2**30, 4),
        "stages": {
            **{key: round(value, 6) if isinstance(value, float) else value for key, value in stages.items()},
            "measured_stage_sum_seconds": round(stage_sum, 6),
            "other_orchestration_seconds": round(max(0.0, elapsed - stage_sum), 6),
        },
        "checks": checks,
        "pass": all(checks.values()),
    }


def main() -> int:
    args = parse_args()
    if args.repeats < 3:
        raise ValueError("At least three measured repeats are required")
    for path in (args.flash_source, args.snapshot):
        if not path.is_dir():
            raise FileNotFoundError(path)
    if not args.prompt_audio.is_file():
        raise FileNotFoundError(args.prompt_audio)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from chatterbox_flash import ChatterboxFlashTTS

    sync()
    load_started = time.perf_counter()
    model = ChatterboxFlashTTS.from_local(
        args.snapshot.resolve(), "cuda", dtype=torch.bfloat16, drf_block_size=16
    )
    sync()
    model_load_seconds = time.perf_counter() - load_started
    torch.cuda.reset_peak_memory_stats()
    conditioning_started = time.perf_counter()
    model.prepare_conditionals(args.prompt_audio.resolve(), exaggeration=0.5)
    sync()
    conditioning_seconds = time.perf_counter() - conditioning_started
    conditioning_peak = torch.cuda.max_memory_allocated() / 2**30

    warmup = run_once(model, "Warm up.", args.output_dir / "warmup.wav")
    runs = [
        run_once(model, PROFILE_TEXT, args.output_dir / f"warm-{index + 1}.wav")
        for index in range(args.repeats)
    ]
    stage_run = run_once(model, PROFILE_TEXT, args.output_dir / "stage-profile.wav", True)
    elapsed = [float(run["elapsed_seconds"]) for run in runs]
    rtfs = [float(run["rtf"]) for run in runs]
    token_hashes = [run["speech_token_sha256"] for run in runs]
    wav_hashes = [run["wav_sha256"] for run in runs]
    peak = max(float(run["peak_allocated_vram_gib"]) for run in runs)
    checks = {
        "warmup_passed": bool(warmup["pass"]),
        "all_measured_runs_passed": all(bool(run["pass"]) for run in runs),
        "stage_run_passed": bool(stage_run["pass"]),
        "fixed_seed_tokens_reproducible": len(set(token_hashes)) == 1,
        "fits_16gb_with_headroom": peak < 14.5,
        "stage_measurements_accounted": 0 < stage_run["stages"]["measured_stage_sum_seconds"] <= stage_run["elapsed_seconds"] * 1.10,
    }
    evidence = {
        "schema_version": 1,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "flash_source_commit": subprocess.run(
            ["git", "-C", str(args.flash_source), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "flash_snapshot_revision": args.snapshot.resolve().name,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "model_load_seconds": round(model_load_seconds, 6),
        "conditioning_seconds": round(conditioning_seconds, 6),
        "conditioning_peak_allocated_vram_gib": round(conditioning_peak, 4),
        "parameter_counts": {
            "t3": sum(parameter.numel() for parameter in model.t3.parameters()),
            "s3gen": sum(parameter.numel() for parameter in model.s3gen.parameters()),
            "voice_encoder": sum(parameter.numel() for parameter in model.ve.parameters()),
        },
        "configuration": {"drf_block_size": 16, "num_steps": 10, "backend": "torch", "use_cuda_graph": False, "n_cfm_timesteps": 2, "seed": SEED},
        "warmup": warmup,
        "measured_runs": runs,
        "stage_run": stage_run,
        "summary": {
            "warm_completion_p50_seconds": round(float(np.percentile(elapsed, 50)), 6),
            "warm_completion_p95_seconds": round(float(np.percentile(elapsed, 95)), 6),
            "warm_rtf_p50": round(float(np.percentile(rtfs, 50)), 6),
            "warm_rtf_p95": round(float(np.percentile(rtfs, 95)), 6),
            "warm_elapsed_coefficient_of_variation": round(statistics.pstdev(elapsed) / statistics.fmean(elapsed), 6),
            "deterministic_speech_token_sha256": token_hashes[0] if len(set(token_hashes)) == 1 else None,
            "wav_files_byte_identical": len(set(wav_hashes)) == 1,
            "peak_allocated_vram_gib": peak,
            "baseline_meets_ttfa_p50_target": float(np.percentile(elapsed, 50)) < 0.150,
            "baseline_meets_ttfa_p95_target": float(np.percentile(elapsed, 95)) < 0.250,
            "baseline_meets_rtf_p50_target": float(np.percentile(rtfs, 50)) < 0.200,
        },
        "checks": checks,
        "pass": all(checks.values()),
    }
    atomic_json(args.output_dir / "chatterbox-flash-profile.json", evidence)
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0 if evidence["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
