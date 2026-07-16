#!/usr/bin/env python3
"""Profile pinned MOSS-TTS-Nano streaming inference in one warm process."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import time
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import soundfile as sf
import torch


PROFILE_TEXT = "Hello. This is a deterministic Nano Flash warm latency profile."
SEED = 20260716


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--moss-repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--text-tokenizer-snapshot", type=Path, required=True)
    parser.add_argument("--audio-tokenizer-snapshot", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-frames", type=int, default=32)
    parser.add_argument("--max-new-frames", type=int, default=160)
    return parser.parse_args()


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


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@dataclass
class StageProfiler:
    reference_encode_seconds: float = 0.0
    reference_encode_calls: int = 0
    semantic_generation_seconds: float = 0.0
    semantic_generation_next_calls: int = 0
    codec_decode_seconds: float = 0.0
    codec_decode_calls: int = 0

    def timed_call(self, stage: str, function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        synchronize()
        started = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            synchronize()
            elapsed = time.perf_counter() - started
            seconds_name = f"{stage}_seconds"
            calls_name = f"{stage}_calls"
            setattr(self, seconds_name, getattr(self, seconds_name) + elapsed)
            setattr(self, calls_name, getattr(self, calls_name) + 1)


class TimedAudioTokenizer:
    """Transparent timing proxy for the two codec operations used by MOSS."""

    def __init__(self, wrapped: Any, profiler: StageProfiler) -> None:
        self._wrapped = wrapped
        self._profiler = profiler

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)

    def batch_encode(self, *args: Any, **kwargs: Any) -> Any:
        return self._profiler.timed_call(
            "reference_encode", self._wrapped.batch_encode, *args, **kwargs
        )

    def batch_decode(self, *args: Any, **kwargs: Any) -> Any:
        return self._profiler.timed_call(
            "codec_decode", self._wrapped.batch_decode, *args, **kwargs
        )


def install_generation_profiler(model: Any, profiler: StageProfiler) -> Callable[[], None]:
    original = model.generate_stream

    def measured_generate_stream(_self: Any, *args: Any, **kwargs: Any) -> Iterator[dict[str, Any]]:
        iterator = original(*args, **kwargs)
        while True:
            synchronize()
            started = time.perf_counter()
            try:
                event = next(iterator)
            except StopIteration:
                synchronize()
                profiler.semantic_generation_seconds += time.perf_counter() - started
                profiler.semantic_generation_next_calls += 1
                return
            synchronize()
            profiler.semantic_generation_seconds += time.perf_counter() - started
            profiler.semantic_generation_next_calls += 1
            yield event

    model.generate_stream = types.MethodType(measured_generate_stream, model)

    def restore() -> None:
        model.generate_stream = original

    return restore


def audio_statistics(path: Path) -> dict[str, Any]:
    waveform, sample_rate = sf.read(path, always_2d=True, dtype="float32")
    duration = float(waveform.shape[0] / sample_rate) if sample_rate else 0.0
    rms = float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64)))) if waveform.size else 0.0
    peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    return {
        "sample_rate": int(sample_rate),
        "channels": int(waveform.shape[1]),
        "samples_per_channel": int(waveform.shape[0]),
        "duration_seconds": duration,
        "finite": bool(np.isfinite(waveform).all()),
        "rms": rms,
        "peak_absolute_amplitude": peak,
        "wav_sha256": file_sha256(path),
    }


def compare_waveforms(paths: list[Path]) -> dict[str, Any]:
    reference, reference_sample_rate = sf.read(paths[0], always_2d=True, dtype="float32")
    max_absolute_error = 0.0
    squared_error_sum = 0.0
    squared_reference_sum = 0.0
    compared_samples = 0
    same_shape_and_rate = True
    for path in paths[1:]:
        candidate, candidate_sample_rate = sf.read(path, always_2d=True, dtype="float32")
        if candidate_sample_rate != reference_sample_rate or candidate.shape != reference.shape:
            same_shape_and_rate = False
            continue
        difference = candidate.astype(np.float64) - reference.astype(np.float64)
        max_absolute_error = max(max_absolute_error, float(np.max(np.abs(difference))))
        squared_error_sum += float(np.square(difference).sum())
        squared_reference_sum += float(np.square(reference.astype(np.float64)).sum())
        compared_samples += int(reference.size)
    rmse = (squared_error_sum / compared_samples) ** 0.5 if compared_samples else None
    snr_db = None
    if squared_error_sum == 0.0 and compared_samples:
        snr_db = "infinite"
    elif squared_error_sum > 0.0 and squared_reference_sum > 0.0:
        snr_db = 10.0 * np.log10(squared_reference_sum / squared_error_sum)
    return {
        "same_shape_and_sample_rate": same_shape_and_rate,
        "max_absolute_error": round(max_absolute_error, 10),
        "rmse": None if rmse is None else round(rmse, 10),
        "snr_db": round(float(snr_db), 6) if isinstance(snr_db, float) else snr_db,
    }


def run_once(
    *,
    model: Any,
    text_tokenizer: Any,
    audio_tokenizer: Any,
    prompt_audio: Path,
    output_audio: Path,
    max_new_frames: int,
    profiler: StageProfiler | None = None,
) -> dict[str, Any]:
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    synchronize()
    torch.cuda.reset_peak_memory_stats()
    starting_allocated = torch.cuda.memory_allocated()
    starting_reserved = torch.cuda.memory_reserved()

    restore_generation = None
    effective_audio_tokenizer = audio_tokenizer
    if profiler is not None:
        effective_audio_tokenizer = TimedAudioTokenizer(audio_tokenizer, profiler)
        restore_generation = install_generation_profiler(model, profiler)

    first_audio_at: float | None = None
    audio_event_times: list[float] = []
    audio_event_durations: list[float] = []
    emitted_lead_seconds: list[float] = []
    pause_events = 0
    final_result: dict[str, Any] | None = None
    final_result_at: float | None = None
    started = time.perf_counter()
    try:
        for event in model.inference_stream(
            text=PROFILE_TEXT,
            output_audio_path=output_audio,
            mode="voice_clone",
            prompt_audio_path=prompt_audio,
            text_tokenizer=text_tokenizer,
            audio_tokenizer=effective_audio_tokenizer,
            device="cuda",
            max_new_frames=max_new_frames,
            do_sample=False,
            use_kv_cache=True,
            audio_repetition_penalty=1.2,
            voice_clone_max_text_tokens=0,
        ):
            observed_at = time.perf_counter()
            if event["type"] == "audio":
                first_audio_at = observed_at if first_audio_at is None else first_audio_at
                audio_event_times.append(observed_at)
                sample_rate = int(event["sample_rate"])
                sample_count = int(event["waveform"].shape[-1])
                audio_event_durations.append(sample_count / sample_rate)
                emitted_lead_seconds.append(float(event["lead_seconds"]))
                pause_events += int(bool(event["is_pause"]))
            elif event["type"] == "result":
                final_result = event
                final_result_at = observed_at
    finally:
        if restore_generation is not None:
            restore_generation()

    synchronize()
    elapsed = time.perf_counter() - started
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    ending_allocated = torch.cuda.memory_allocated()
    ending_reserved = torch.cuda.memory_reserved()
    stats = audio_statistics(output_audio)
    token_frames = (
        int(final_result["audio_token_ids"].shape[0])
        if final_result is not None and hasattr(final_result.get("audio_token_ids"), "shape")
        else 0
    )
    audio_token_hash = (
        tensor_sha256(final_result["audio_token_ids"])
        if final_result is not None and torch.is_tensor(final_result.get("audio_token_ids"))
        else None
    )
    ttfa = (first_audio_at - started) if first_audio_at is not None else None
    inter_event_gaps = np.diff(np.asarray(audio_event_times, dtype=np.float64)).tolist()
    duration = float(stats["duration_seconds"])
    checks = {
        "result_event_present": final_result is not None,
        "audio_event_present": bool(audio_event_times),
        "finite_audio": bool(stats["finite"]),
        "non_silent_audio": float(stats["rms"]) > 1e-5,
        "duration_gt_half_second": duration > 0.5,
        "token_frames_present": token_frames > 0,
        "positive_ttfa": ttfa is not None and ttfa > 0.0,
    }
    return {
        "elapsed_seconds": round(elapsed, 6),
        "ttfa_seconds": None if ttfa is None else round(ttfa, 6),
        "rtf": None if duration <= 0.0 else round(elapsed / duration, 6),
        "audio_token_frames": token_frames,
        "audio_token_sha256": audio_token_hash,
        "hit_max_new_frames": token_frames == max_new_frames,
        "audio_event_count": len(audio_event_times),
        "pause_event_count": pause_events,
        "time_to_last_audio_event_seconds": (
            None if not audio_event_times else round(audio_event_times[-1] - started, 6)
        ),
        "audio_emission_span_seconds": (
            None
            if len(audio_event_times) < 2
            else round(audio_event_times[-1] - audio_event_times[0], 6)
        ),
        "finalization_after_last_audio_seconds": (
            None
            if not audio_event_times or final_result_at is None
            else round(final_result_at - audio_event_times[-1], 6)
        ),
        "first_audio_event_duration_seconds": (
            None if not audio_event_durations else round(audio_event_durations[0], 6)
        ),
        "mean_audio_event_duration_seconds": (
            None if not audio_event_durations else round(statistics.fmean(audio_event_durations), 6)
        ),
        "max_inter_audio_event_gap_seconds": (
            None if not inter_event_gaps else round(max(inter_event_gaps), 6)
        ),
        "min_reported_lead_seconds": (
            None if not emitted_lead_seconds else round(min(emitted_lead_seconds), 6)
        ),
        "starting_allocated_vram_gib": round(starting_allocated / 2**30, 4),
        "starting_reserved_vram_gib": round(starting_reserved / 2**30, 4),
        "peak_allocated_vram_gib": round(peak_allocated / 2**30, 4),
        "peak_reserved_vram_gib": round(peak_reserved / 2**30, 4),
        "peak_allocated_delta_vram_gib": round((peak_allocated - starting_allocated) / 2**30, 4),
        "ending_allocated_vram_gib": round(ending_allocated / 2**30, 4),
        "ending_reserved_vram_gib": round(ending_reserved / 2**30, 4),
        **{key: round(value, 8) if isinstance(value, float) else value for key, value in stats.items()},
        "checks": checks,
        "pass": all(checks.values()),
    }


def main() -> int:
    args = parse_args()
    if args.repeats < 3:
        raise ValueError("At least three measured repeats are required")
    for required_path in (
        args.moss_repo,
        args.model_snapshot,
        args.text_tokenizer_snapshot,
        args.audio_tokenizer_snapshot,
    ):
        if not required_path.is_dir():
            raise FileNotFoundError(required_path)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompt_audio = args.moss_repo / "assets/audio/en_6.wav"
    if not prompt_audio.is_file():
        raise FileNotFoundError(prompt_audio)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the G1 profile")

    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    device = torch.device("cuda")
    synchronize()
    load_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_snapshot.resolve()),
        trust_remote_code=True,
        local_files_only=True,
    )
    model.to(device=device, dtype=torch.bfloat16)
    model._set_attention_implementation("sdpa")
    model.eval()
    synchronize()
    model_load_seconds = time.perf_counter() - load_started

    components_started = time.perf_counter()
    text_tokenizer = AutoTokenizer.from_pretrained(
        str(args.text_tokenizer_snapshot.resolve()),
        trust_remote_code=True,
        use_fast=False,
        local_files_only=True,
    )
    if text_tokenizer.pad_token_id is None and text_tokenizer.eos_token is not None:
        text_tokenizer.pad_token = text_tokenizer.eos_token
    audio_tokenizer = AutoModel.from_pretrained(
        str(args.audio_tokenizer_snapshot.resolve()),
        trust_remote_code=True,
        local_files_only=True,
    )
    audio_tokenizer.to(device)
    audio_tokenizer.eval()
    synchronize()
    component_load_seconds = time.perf_counter() - components_started

    warmup = run_once(
        model=model,
        text_tokenizer=text_tokenizer,
        audio_tokenizer=audio_tokenizer,
        prompt_audio=prompt_audio,
        output_audio=args.output_dir / "warmup.wav",
        max_new_frames=args.warmup_frames,
    )

    measured_runs: list[dict[str, Any]] = []
    for index in range(args.repeats):
        measured_runs.append(
            run_once(
                model=model,
                text_tokenizer=text_tokenizer,
                audio_tokenizer=audio_tokenizer,
                prompt_audio=prompt_audio,
                output_audio=args.output_dir / f"warm-{index + 1}.wav",
                max_new_frames=args.max_new_frames,
            )
        )

    stage_profiler = StageProfiler()
    stage_run = run_once(
        model=model,
        text_tokenizer=text_tokenizer,
        audio_tokenizer=audio_tokenizer,
        prompt_audio=prompt_audio,
        output_audio=args.output_dir / "stage-profile.wav",
        max_new_frames=args.max_new_frames,
        profiler=stage_profiler,
    )
    stage_metrics = asdict(stage_profiler)
    stage_sum = (
        stage_profiler.reference_encode_seconds
        + stage_profiler.semantic_generation_seconds
        + stage_profiler.codec_decode_seconds
    )
    stage_metrics["measured_stage_sum_seconds"] = stage_sum
    stage_metrics["other_orchestration_and_io_seconds"] = max(
        0.0, float(stage_run["elapsed_seconds"]) - stage_sum
    )
    stage_metrics = {
        key: round(value, 6) if isinstance(value, float) else value
        for key, value in stage_metrics.items()
    }

    ttfas = [float(run["ttfa_seconds"]) for run in measured_runs]
    rtfs = [float(run["rtf"]) for run in measured_runs]
    elapsed_values = [float(run["elapsed_seconds"]) for run in measured_runs]
    token_hashes = [str(run["audio_token_sha256"]) for run in measured_runs]
    wav_hashes = [str(run["wav_sha256"]) for run in measured_runs]
    measured_wav_paths = [args.output_dir / f"warm-{index + 1}.wav" for index in range(args.repeats)]
    peak_vram = max(float(run["peak_allocated_vram_gib"]) for run in measured_runs)
    summary = {
        "warm_ttfa_p50_seconds": round(percentile(ttfas, 50), 6),
        "warm_ttfa_p95_seconds": round(percentile(ttfas, 95), 6),
        "warm_rtf_p50": round(percentile(rtfs, 50), 6),
        "warm_rtf_p95": round(percentile(rtfs, 95), 6),
        "warm_elapsed_mean_seconds": round(statistics.fmean(elapsed_values), 6),
        "warm_elapsed_coefficient_of_variation": round(
            statistics.pstdev(elapsed_values) / statistics.fmean(elapsed_values), 6
        ),
        "peak_allocated_vram_gib": peak_vram,
        "deterministic_audio_token_sha256": (
            token_hashes[0] if len(set(token_hashes)) == 1 else None
        ),
        "wav_files_byte_identical": len(set(wav_hashes)) == 1,
        "waveform_repeat_comparison": compare_waveforms(measured_wav_paths),
        "measured_runs_hitting_frame_cap": sum(
            int(bool(run["hit_max_new_frames"])) for run in measured_runs
        ),
        "stage_run_tokens_match_warm_runs": (
            str(stage_run["audio_token_sha256"]) == token_hashes[0]
            if len(set(token_hashes)) == 1
            else False
        ),
        "nano_flash_target_warm_ttfa_p50_seconds": 0.150,
        "nano_flash_target_warm_ttfa_p95_seconds": 0.250,
        "nano_flash_target_rtf": 0.200,
        "baseline_meets_ttfa_p50_target": percentile(ttfas, 50) < 0.150,
        "baseline_meets_ttfa_p95_target": percentile(ttfas, 95) < 0.250,
        "baseline_meets_rtf_p50_target": percentile(rtfs, 50) < 0.200,
    }
    checks = {
        "warmup_passed": bool(warmup["pass"]),
        "three_or_more_measured_runs": len(measured_runs) >= 3,
        "all_measured_runs_passed": all(bool(run["pass"]) for run in measured_runs),
        "stage_run_passed": bool(stage_run["pass"]),
        "greedy_audio_tokens_deterministic": len(set(token_hashes)) == 1,
        "fits_16gb_with_headroom": peak_vram < 14.5,
        "stage_measurements_accounted": 0.0 < stage_sum <= float(stage_run["elapsed_seconds"]) * 1.10,
    }
    output = {
        "schema_version": 1,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "upstream_commit": subprocess.run(
            ["git", "-C", str(args.moss_repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "model_snapshot_revision": args.model_snapshot.resolve().name,
        "text_tokenizer_snapshot_revision": args.text_tokenizer_snapshot.resolve().name,
        "audio_tokenizer_snapshot_revision": args.audio_tokenizer_snapshot.resolve().name,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "dtype": "bfloat16",
        "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "audio_tokenizer_parameter_count": sum(
            parameter.numel() for parameter in audio_tokenizer.parameters()
        ),
        "audio_tokenizer_dtype": str(next(audio_tokenizer.parameters()).dtype),
        "attention_implementation": "sdpa",
        "audio_repetition_penalty": 1.2,
        "concurrency": 1,
        "seed": SEED,
        "text": PROFILE_TEXT,
        "max_new_frames": args.max_new_frames,
        "warmup_frames": args.warmup_frames,
        "model_load_seconds": round(model_load_seconds, 6),
        "component_load_seconds": round(component_load_seconds, 6),
        "warmup": warmup,
        "measured_runs": measured_runs,
        "stage_run": stage_run,
        "stage_metrics": stage_metrics,
        "summary": summary,
        "checks": checks,
        "pass": all(checks.values()),
    }
    output_path = args.output_dir / "moss-profile.json"
    atomic_json(output_path, output)
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if output["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
