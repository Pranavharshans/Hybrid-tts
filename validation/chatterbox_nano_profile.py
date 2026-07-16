#!/usr/bin/env python3
"""Run deterministic smoke and warm profiling for official Chatterbox-Nano."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio


PROFILE_TEXT = "Hello. This is a deterministic Nano Flash warm latency profile."
SEED = 20260716


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nano-source", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--prompt-audio", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    return parser.parse_args()


def synchronize() -> None:
    torch.cuda.synchronize()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    contiguous = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(str(tuple(contiguous.shape)).encode("utf-8"))
    digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


@dataclass
class StageProfiler:
    semantic_generation_seconds: float = 0.0
    semantic_generation_calls: int = 0
    acoustic_render_seconds: float = 0.0
    acoustic_render_calls: int = 0
    watermark_seconds: float = 0.0
    watermark_calls: int = 0


def waveform_stats(waveform: torch.Tensor, sample_rate: int) -> dict[str, Any]:
    array = waveform.detach().cpu().float().numpy()
    duration = float(array.shape[-1] / sample_rate) if sample_rate else 0.0
    return {
        "sample_rate": int(sample_rate),
        "channels": int(array.shape[0]),
        "samples_per_channel": int(array.shape[-1]),
        "duration_seconds": duration,
        "finite": bool(np.isfinite(array).all()),
        "rms": float(np.sqrt(np.mean(np.square(array, dtype=np.float64)))) if array.size else 0.0,
        "peak_absolute_amplitude": float(np.max(np.abs(array))) if array.size else 0.0,
    }


def run_once(
    *,
    model: Any,
    text: str,
    output_audio: Path,
    profiler: StageProfiler | None = None,
) -> dict[str, Any]:
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    synchronize()
    torch.cuda.reset_peak_memory_stats()
    starting_allocated = torch.cuda.memory_allocated()
    starting_reserved = torch.cuda.memory_reserved()

    captured: dict[str, torch.Tensor] = {}
    original_semantic = model.t3.inference_turbo
    original_acoustic = model.s3gen.inference
    original_watermark = model.watermarker.apply_watermark

    def semantic_wrapper(*args: Any, **kwargs: Any) -> Any:
        if profiler is not None:
            synchronize()
            started = time.perf_counter()
        result = original_semantic(*args, **kwargs)
        if profiler is not None:
            synchronize()
            profiler.semantic_generation_seconds += time.perf_counter() - started
            profiler.semantic_generation_calls += 1
        captured["speech_tokens"] = result.detach().cpu().clone()
        return result

    def acoustic_wrapper(*args: Any, **kwargs: Any) -> Any:
        if profiler is not None:
            synchronize()
            started = time.perf_counter()
        result = original_acoustic(*args, **kwargs)
        if profiler is not None:
            synchronize()
            profiler.acoustic_render_seconds += time.perf_counter() - started
            profiler.acoustic_render_calls += 1
        return result

    def watermark_wrapper(*args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        result = original_watermark(*args, **kwargs)
        if profiler is not None:
            profiler.watermark_seconds += time.perf_counter() - started
            profiler.watermark_calls += 1
        return result

    model.t3.inference_turbo = semantic_wrapper
    model.s3gen.inference = acoustic_wrapper
    model.watermarker.apply_watermark = watermark_wrapper
    started = time.perf_counter()
    try:
        waveform = model.generate(
            text,
            repetition_penalty=1.2,
            min_p=0.0,
            top_p=0.95,
            exaggeration=0.0,
            cfg_weight=0.0,
            temperature=0.8,
            top_k=1000,
        )
        synchronize()
        elapsed = time.perf_counter() - started
    finally:
        model.t3.inference_turbo = original_semantic
        model.s3gen.inference = original_acoustic
        model.watermarker.apply_watermark = original_watermark

    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    ending_allocated = torch.cuda.memory_allocated()
    stats = waveform_stats(waveform, model.sr)
    save_started = time.perf_counter()
    torchaudio.save(str(output_audio), waveform.detach().cpu().float(), model.sr)
    save_seconds = time.perf_counter() - save_started
    speech_tokens = captured.get("speech_tokens")
    token_count = int(speech_tokens.numel()) if speech_tokens is not None else 0
    duration = float(stats["duration_seconds"])
    checks = {
        "finite_audio": bool(stats["finite"]),
        "non_silent_audio": float(stats["rms"]) > 1e-5,
        "duration_gt_half_second": duration > 0.5,
        "speech_tokens_present": token_count > 0,
        "output_exists": output_audio.is_file(),
        "positive_elapsed": elapsed > 0.0,
    }
    return {
        "text": text,
        "elapsed_seconds": round(elapsed, 6),
        "time_to_first_audio_seconds": round(elapsed, 6),
        "full_utterance_return_api": True,
        "rtf": None if duration <= 0.0 else round(elapsed / duration, 6),
        "save_seconds": round(save_seconds, 6),
        "speech_token_count": token_count,
        "speech_token_sha256": None if speech_tokens is None else tensor_sha256(speech_tokens),
        "starting_allocated_vram_gib": round(starting_allocated / 2**30, 4),
        "starting_reserved_vram_gib": round(starting_reserved / 2**30, 4),
        "peak_allocated_vram_gib": round(peak_allocated / 2**30, 4),
        "peak_reserved_vram_gib": round(peak_reserved / 2**30, 4),
        "peak_allocated_delta_vram_gib": round((peak_allocated - starting_allocated) / 2**30, 4),
        "ending_allocated_vram_gib": round(ending_allocated / 2**30, 4),
        **{key: round(value, 8) if isinstance(value, float) else value for key, value in stats.items()},
        "wav_sha256": file_sha256(output_audio),
        "checks": checks,
        "pass": all(checks.values()),
    }


def main() -> int:
    args = parse_args()
    if args.repeats < 3:
        raise ValueError("At least three measured repeats are required")
    for path in (args.nano_source, args.snapshot):
        if not path.is_dir():
            raise FileNotFoundError(path)
    if not args.prompt_audio.is_file():
        raise FileNotFoundError(args.prompt_audio)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from chatterbox.tts_nano import ChatterboxNanoTTS  # noqa: PLC0415

    synchronize()
    load_started = time.perf_counter()
    model = ChatterboxNanoTTS.from_local(args.snapshot.resolve(), "cuda")
    synchronize()
    model_load_seconds = time.perf_counter() - load_started

    torch.cuda.reset_peak_memory_stats()
    conditioning_started = time.perf_counter()
    model.prepare_conditionals(
        args.prompt_audio.resolve(),
        exaggeration=0.0,
        norm_loudness=True,
    )
    synchronize()
    conditioning_seconds = time.perf_counter() - conditioning_started
    conditioning_peak_vram = torch.cuda.max_memory_allocated() / 2**30

    warmup = run_once(
        model=model,
        text="Warm up.",
        output_audio=args.output_dir / "warmup.wav",
    )
    measured_runs = [
        run_once(
            model=model,
            text=PROFILE_TEXT,
            output_audio=args.output_dir / f"warm-{index + 1}.wav",
        )
        for index in range(args.repeats)
    ]
    stage_profiler = StageProfiler()
    stage_run = run_once(
        model=model,
        text=PROFILE_TEXT,
        output_audio=args.output_dir / "stage-profile.wav",
        profiler=stage_profiler,
    )

    elapsed_values = [float(run["elapsed_seconds"]) for run in measured_runs]
    rtf_values = [float(run["rtf"]) for run in measured_runs]
    token_hashes = [str(run["speech_token_sha256"]) for run in measured_runs]
    wav_hashes = [str(run["wav_sha256"]) for run in measured_runs]
    stage_metrics = asdict(stage_profiler)
    stage_sum = sum(
        float(stage_metrics[key])
        for key in (
            "semantic_generation_seconds",
            "acoustic_render_seconds",
            "watermark_seconds",
        )
    )
    stage_metrics["measured_stage_sum_seconds"] = stage_sum
    stage_metrics["other_orchestration_seconds"] = max(
        0.0, float(stage_run["elapsed_seconds"]) - stage_sum
    )
    stage_metrics = {
        key: round(value, 6) if isinstance(value, float) else value
        for key, value in stage_metrics.items()
    }
    peak_vram = max(float(run["peak_allocated_vram_gib"]) for run in measured_runs)
    summary = {
        "warm_completion_p50_seconds": round(float(np.percentile(elapsed_values, 50)), 6),
        "warm_completion_p95_seconds": round(float(np.percentile(elapsed_values, 95)), 6),
        "warm_rtf_p50": round(float(np.percentile(rtf_values, 50)), 6),
        "warm_rtf_p95": round(float(np.percentile(rtf_values, 95)), 6),
        "warm_elapsed_coefficient_of_variation": round(
            statistics.pstdev(elapsed_values) / statistics.fmean(elapsed_values), 6
        ),
        "deterministic_speech_token_sha256": (
            token_hashes[0] if len(set(token_hashes)) == 1 else None
        ),
        "wav_files_byte_identical": len(set(wav_hashes)) == 1,
        "stage_run_tokens_match_warm_runs": (
            str(stage_run["speech_token_sha256"]) == token_hashes[0]
            if len(set(token_hashes)) == 1
            else False
        ),
        "peak_allocated_vram_gib": peak_vram,
        "nano_flash_target_warm_ttfa_p50_seconds": 0.150,
        "nano_flash_target_warm_ttfa_p95_seconds": 0.250,
        "nano_flash_target_rtf": 0.200,
        "baseline_meets_ttfa_p50_target": float(np.percentile(elapsed_values, 50)) < 0.150,
        "baseline_meets_ttfa_p95_target": float(np.percentile(elapsed_values, 95)) < 0.250,
        "baseline_meets_rtf_p50_target": float(np.percentile(rtf_values, 50)) < 0.200,
    }
    checks = {
        "warmup_passed": bool(warmup["pass"]),
        "three_or_more_measured_runs": len(measured_runs) >= 3,
        "all_measured_runs_passed": all(bool(run["pass"]) for run in measured_runs),
        "stage_run_passed": bool(stage_run["pass"]),
        "sampled_tokens_reproducible_with_fixed_seed": len(set(token_hashes)) == 1,
        "fits_16gb_with_headroom": peak_vram < 14.5,
        "stage_measurements_accounted": 0.0 < stage_sum <= float(stage_run["elapsed_seconds"]) * 1.10,
    }
    evidence = {
        "schema_version": 1,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "nano_source_commit": subprocess.run(
            ["git", "-C", str(args.nano_source), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "nano_snapshot_revision": args.snapshot.resolve().name,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "sampling": {
            "seed": SEED,
            "repetition_penalty": 1.2,
            "top_p": 0.95,
            "temperature": 0.8,
            "top_k": 1000,
        },
        "model_load_seconds": round(model_load_seconds, 6),
        "conditioning_seconds": round(conditioning_seconds, 6),
        "conditioning_peak_allocated_vram_gib": round(conditioning_peak_vram, 4),
        "parameter_counts": {
            "t3": sum(parameter.numel() for parameter in model.t3.parameters()),
            "s3gen": sum(parameter.numel() for parameter in model.s3gen.parameters()),
            "voice_encoder": sum(parameter.numel() for parameter in model.ve.parameters()),
        },
        "warmup": warmup,
        "measured_runs": measured_runs,
        "stage_run": stage_run,
        "stage_metrics": stage_metrics,
        "summary": summary,
        "checks": checks,
        "pass": all(checks.values()),
    }
    atomic_json(args.output_dir / "chatterbox-nano-profile.json", evidence)
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0 if evidence["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
