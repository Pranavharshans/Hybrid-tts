"""Signal operations required by the Nano Flash torchaudio compatibility layer."""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy.signal import resample_poly


def resample(
    waveform: torch.Tensor,
    orig_freq: int,
    new_freq: int,
    *_: object,
    **__: object,
) -> torch.Tensor:
    if int(orig_freq) == int(new_freq):
        return waveform
    if waveform.ndim not in {1, 2}:
        raise ValueError(f"Expected [time] or [channels, time], got {tuple(waveform.shape)}")

    original_device = waveform.device
    original_dtype = waveform.dtype
    source = waveform.detach().to(device="cpu", dtype=torch.float32).numpy()
    divisor = math.gcd(int(orig_freq), int(new_freq))
    up = int(new_freq) // divisor
    down = int(orig_freq) // divisor
    converted = resample_poly(source, up, down, axis=-1).astype(np.float32, copy=False)
    return torch.from_numpy(converted.copy()).to(device=original_device, dtype=original_dtype)
