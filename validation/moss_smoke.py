#!/usr/bin/env python3
"""Run deterministic MOSS-TTS-Nano CUDA smoke inference and emit compact evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--moss-repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default="OpenMOSS-Team/MOSS-TTS-Nano")
    parser.add_argument("--audio-tokenizer", default="OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano")
    parser.add_argument("--max-new-frames", type=int, default=96)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    moss_repo = args.moss_repo.resolve()
    prompt_audio = moss_repo / "assets/audio/en_6.wav"
    output_audio = args.output_dir / "moss-smoke.wav"
    output_json = args.output_dir / "moss-smoke.json"

    if not prompt_audio.is_file():
        raise FileNotFoundError(prompt_audio)
    if str(moss_repo) not in sys.path:
        sys.path.insert(0, str(moss_repo))

    import infer  # noqa: PLC0415

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the G1 smoke test")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    result = infer.main(
        [
            "--checkpoint",
            args.checkpoint,
            "--audio-tokenizer-pretrained-name-or-path",
            args.audio_tokenizer,
            "--output-audio-path",
            str(output_audio),
            "--mode",
            "voice_clone",
            "--prompt-audio-path",
            str(prompt_audio),
            "--text",
            "Hello. This is a deterministic Nano Flash baseline validation.",
            "--device",
            "cuda",
            "--dtype",
            "bfloat16",
            "--max-new-frames",
            str(args.max_new_frames),
            "--voice-clone-max-text-tokens",
            "0",
            "--do-sample",
            "0",
            "--disable-wetext-processing",
            "--seed",
            "20260716",
        ]
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    waveform, sample_rate = sf.read(output_audio, always_2d=True, dtype="float32")
    finite = bool(np.isfinite(waveform).all())
    peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64)))) if waveform.size else 0.0
    duration = float(waveform.shape[0] / sample_rate) if sample_rate else 0.0
    audio_tokens = result.get("audio_token_ids")
    token_frames = int(audio_tokens.shape[0]) if hasattr(audio_tokens, "shape") else None

    checks = {
        "output_exists": output_audio.is_file(),
        "finite_audio": finite,
        "non_silent_audio": rms > 1e-5,
        "duration_gt_half_second": duration > 0.5,
        "valid_sample_rate": sample_rate >= 16000,
        "token_frames_present": token_frames is not None and token_frames > 0,
    }
    evidence: dict[str, Any] = {
        "schema_version": 1,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "upstream_commit": subprocess.run(
            ["git", "-C", str(moss_repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "dtype": "bfloat16",
        "elapsed_seconds_cold": round(elapsed, 4),
        "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 4),
        "sample_rate": int(sample_rate),
        "channels": int(waveform.shape[1]),
        "samples_per_channel": int(waveform.shape[0]),
        "duration_seconds": round(duration, 6),
        "audio_token_frames": token_frames,
        "peak_absolute_amplitude": round(peak, 8),
        "rms": round(rms, 8),
        "wav_sha256": file_sha256(output_audio),
        "checks": checks,
        "pass": all(checks.values()),
    }
    atomic_json(output_json, evidence)
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0 if evidence["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
