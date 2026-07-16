#!/usr/bin/env python3
"""Self-test for the narrow MOSS torchaudio compatibility API."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import torch

from compat import torchaudio


def main() -> int:
    sample_rate = 16000
    timeline = torch.arange(sample_rate, dtype=torch.float32) / sample_rate
    source = (0.25 * torch.sin(2 * math.pi * 440 * timeline)).unsqueeze(0)

    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "tone.wav"
        torchaudio.save(path, source, sample_rate)
        loaded, loaded_rate = torchaudio.load(path)
        downsampled = torchaudio.functional.resample(loaded, loaded_rate, 8000)

    assert loaded_rate == sample_rate
    assert loaded.shape == source.shape
    assert downsampled.shape == (1, 8000)
    assert bool(torch.isfinite(downsampled).all())
    assert float(downsampled.abs().max()) > 0.1
    print("torchaudio compatibility self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
