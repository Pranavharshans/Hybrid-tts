from __future__ import annotations

import unittest

import numpy as np

from streaming_scheduler import HybridScheduler, Mode, PCM24kPacketizer, TextBudget


class PacketizerTest(unittest.TestCase):
    def test_packetizes_80ms_and_flushes_tail_losslessly(self) -> None:
        source = np.arange(5000, dtype=np.float32)
        packetizer = PCM24kPacketizer(packet_ms=80)
        packets = packetizer.push(source[:2500]) + packetizer.push(source[2500:], final=True)
        self.assertEqual([len(packet) for packet in packets], [1920, 1920, 1160])
        np.testing.assert_array_equal(np.concatenate(packets), source)

    def test_fragmentation_does_not_change_packets(self) -> None:
        source = np.linspace(-1, 1, 7680, dtype=np.float32)
        whole = PCM24kPacketizer().push(source, final=True)
        fragmented_packetizer = PCM24kPacketizer()
        fragmented = []
        for start in range(0, len(source), 137):
            fragmented.extend(fragmented_packetizer.push(source[start : start + 137]))
        fragmented.extend(fragmented_packetizer.push(np.empty(0), final=True))
        self.assertEqual(len(whole), len(fragmented))
        for left, right in zip(whole, fragmented):
            np.testing.assert_array_equal(left, right)


class SchedulerTest(unittest.TestCase):
    def test_speculative_text_can_feed_ar_only(self) -> None:
        scheduler = HybridScheduler()
        text = TextBudget(speculative_audio_seconds=0.8, stable_audio_seconds=0.0)
        self.assertEqual(scheduler.choose_mode(text), Mode.AR)
        self.assertTrue(scheduler.can_schedule(Mode.AR, text))
        self.assertFalse(scheduler.can_schedule(Mode.BLOCK, text))

    def test_block_requires_buffer_and_stable_span(self) -> None:
        scheduler = HybridScheduler()
        scheduler.complete_generation(0.24)
        text = TextBudget(speculative_audio_seconds=2.0, stable_audio_seconds=2.0)
        self.assertEqual(scheduler.choose_mode(text), Mode.BLOCK)
        self.assertTrue(scheduler.can_schedule(Mode.BLOCK, text))

    def test_low_buffer_falls_back_to_ar(self) -> None:
        scheduler = HybridScheduler()
        scheduler.complete_generation(0.16)
        text = TextBudget(speculative_audio_seconds=2.0, stable_audio_seconds=2.0)
        self.assertEqual(scheduler.choose_mode(text), Mode.AR)

    def test_consumption_never_makes_buffer_negative(self) -> None:
        scheduler = HybridScheduler()
        scheduler.complete_generation(0.16)
        self.assertAlmostEqual(scheduler.consume(0.20), 0.16)
        self.assertEqual(scheduler.queued_seconds, 0.0)

    def test_cancel_drops_all_future_audio_and_disables_generation(self) -> None:
        scheduler = HybridScheduler()
        scheduler.complete_generation(0.32)
        self.assertAlmostEqual(scheduler.cancel(), 0.32)
        self.assertEqual(scheduler.queued_seconds, 0.0)
        with self.assertRaises(RuntimeError):
            scheduler.choose_mode(TextBudget(1.0, 1.0))


if __name__ == "__main__":
    unittest.main()
