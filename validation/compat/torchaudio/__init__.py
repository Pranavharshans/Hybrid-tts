"""Minimal torchaudio compatibility surface for Blackwell PyTorch images.

MOSS-TTS-Nano uses only ``load``, ``save``, and ``functional.resample``. The
Vast PyTorch 2.12 image intentionally has no matching torchaudio wheel, so this
module supplies those operations without installing a mismatched binary build.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import soundfile as sf
import torch

from . import functional

__version__ = "0.0.0-nano-flash-compat"


def load(path: str | Path, *_: Any, **__: Any) -> tuple[torch.Tensor, int]:
    samples, sample_rate = sf.read(str(path), always_2d=True, dtype="float32")
    waveform = torch.from_numpy(samples.T.copy())
    return waveform, int(sample_rate)


def save(
    path: str | Path,
    src: torch.Tensor,
    sample_rate: int,
    *_: Any,
    **__: Any,
) -> None:
    waveform = src.detach().to(device="cpu", dtype=torch.float32)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim != 2:
        raise ValueError(f"Expected [channels, time] waveform, got {tuple(waveform.shape)}")
    sf.write(str(path), waveform.transpose(0, 1).contiguous().numpy(), int(sample_rate))


__all__ = ["functional", "load", "save"]
