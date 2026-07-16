"""Minimal hybrid TTS packetizer and scheduling state machine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class Mode(str, Enum):
    AR = "ar"
    BLOCK = "block"


@dataclass(frozen=True)
class TextBudget:
    speculative_audio_seconds: float
    stable_audio_seconds: float


class PCM24kPacketizer:
    def __init__(self, packet_ms: int = 80, sample_rate: int = 24000) -> None:
        if packet_ms <= 0 or sample_rate <= 0:
            raise ValueError("packet duration and sample rate must be positive")
        self.sample_rate = sample_rate
        self.packet_samples = sample_rate * packet_ms // 1000
        self._pending = np.empty(0, dtype=np.float32)

    def push(self, audio: np.ndarray, *, final: bool = False) -> list[np.ndarray]:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        self._pending = np.concatenate((self._pending, audio))
        packets = []
        while len(self._pending) >= self.packet_samples:
            packets.append(self._pending[: self.packet_samples].copy())
            self._pending = self._pending[self.packet_samples :]
        if final and len(self._pending):
            packets.append(self._pending.copy())
            self._pending = np.empty(0, dtype=np.float32)
        return packets


class HybridScheduler:
    def __init__(
        self,
        *,
        packet_seconds: float = 0.08,
        block_seconds: float = 0.64,
        switch_buffer_seconds: float = 0.24,
    ) -> None:
        if min(packet_seconds, block_seconds, switch_buffer_seconds) <= 0:
            raise ValueError("scheduler durations must be positive")
        self.packet_seconds = packet_seconds
        self.block_seconds = block_seconds
        self.switch_buffer_seconds = switch_buffer_seconds
        self.queued_seconds = 0.0
        self.generated_seconds = 0.0
        self.cancelled = False

    def choose_mode(self, text: TextBudget) -> Mode:
        if self.cancelled:
            raise RuntimeError("request is cancelled")
        stable_remaining = text.stable_audio_seconds - self.generated_seconds
        if self.queued_seconds >= self.switch_buffer_seconds and stable_remaining >= self.block_seconds:
            return Mode.BLOCK
        return Mode.AR

    def generation_quantum(self, mode: Mode) -> float:
        return self.block_seconds if mode is Mode.BLOCK else self.packet_seconds

    def can_schedule(self, mode: Mode, text: TextBudget) -> bool:
        available = text.stable_audio_seconds if mode is Mode.BLOCK else text.speculative_audio_seconds
        return available - self.generated_seconds >= self.generation_quantum(mode) - 1e-9

    def complete_generation(self, seconds: float) -> None:
        if self.cancelled or seconds <= 0:
            raise RuntimeError("invalid generation completion")
        self.generated_seconds += seconds
        self.queued_seconds += seconds

    def consume(self, seconds: float) -> float:
        if seconds < 0:
            raise ValueError("consume duration must be nonnegative")
        consumed = min(seconds, self.queued_seconds)
        self.queued_seconds -= consumed
        return consumed

    def cancel(self) -> float:
        dropped = self.queued_seconds
        self.queued_seconds = 0.0
        self.cancelled = True
        return dropped
