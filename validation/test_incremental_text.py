from __future__ import annotations

import json
import random
import unittest
from pathlib import Path

from incremental_text import IncrementalCommitter, normalize_surface


SUITE = Path(__file__).with_name("incremental_prompt_suite.json")


class IncrementalCommitterTest(unittest.TestCase):
    def test_all_chunkings_are_monotonic_and_lossless(self) -> None:
        prompts = json.loads(SUITE.read_text())
        for prompt in prompts:
            text = prompt["text"]
            for seed in range(25):
                randomizer = random.Random(seed)
                committer = IncrementalCommitter(lookahead_words=2)
                offset = 0
                previous = ""
                while offset < len(text):
                    width = randomizer.randint(1, 9)
                    state = committer.push(text[offset : offset + width])
                    self.assertTrue(state.committed.startswith(previous), prompt["id"])
                    self.assertEqual(state.received, state.committed + state.pending)
                    previous = state.committed
                    offset += width
                state = committer.push("", final=True)
                self.assertEqual(state.committed, normalize_surface(text), prompt["id"])
                self.assertEqual(state.pending, "")

    def test_holds_incomplete_tail(self) -> None:
        committer = IncrementalCommitter(lookahead_words=2)
        state = committer.push("The account number is twenty four")
        self.assertEqual(state.committed, "The account number is ")
        self.assertEqual(state.pending, "twenty four")

    def test_sentence_boundary_can_commit(self) -> None:
        committer = IncrementalCommitter(lookahead_words=3)
        state = committer.push("This is ready. ")
        self.assertEqual(state.committed, "This is ready. ")
        self.assertEqual(state.reason, "sentence")

    def test_surface_normalization_is_incremental(self) -> None:
        committer = IncrementalCommitter()
        for chunk in ["She said ", "“hello”", "—then left."]:
            committer.push(chunk)
        state = committer.push("", final=True)
        self.assertEqual(state.committed, 'She said "hello" -- then left.')


if __name__ == "__main__":
    unittest.main()
