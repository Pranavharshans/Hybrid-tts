"""Conservative incremental English text commitment for streaming TTS tests."""

from __future__ import annotations

import re
from dataclasses import dataclass


TOKEN_RE = re.compile(r"\S+\s*")
STRONG_BOUNDARY_RE = re.compile(r"[.!?](?:[\"')\]]+)?\s+$")
CLAUSE_BOUNDARY_RE = re.compile(r"[,;:](?:[\"')\]]+)?\s+$")


@dataclass(frozen=True)
class Commitment:
    received: str
    committed: str
    pending: str
    reason: str


def normalize_surface(text: str) -> str:
    """Apply only prefix-stable normalization; defer semantic expansion."""
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2014", " -- ").replace("\u2013", "-")
    return re.sub(r"[ \t\r\f\v]+", " ", text)


class IncrementalCommitter:
    """Commit an append-only prefix while retaining configurable lookahead."""

    def __init__(self, lookahead_words: int = 2, commit_clauses: bool = True) -> None:
        if lookahead_words < 1:
            raise ValueError("lookahead_words must be positive")
        self.lookahead_words = lookahead_words
        self.commit_clauses = commit_clauses
        self._received = ""
        self._committed = ""

    @property
    def committed(self) -> str:
        return self._committed

    def push(self, append_text: str, *, final: bool = False) -> Commitment:
        if not append_text and not final:
            raise ValueError("push requires text or final=True")
        self._received = normalize_surface(self._received + append_text)
        if not self._received.startswith(self._committed):
            raise RuntimeError("normalization revised already committed text")
        pending = self._received[len(self._committed) :]
        boundary = 0
        reason = "lookahead"

        if final:
            boundary = len(pending)
            reason = "final"
        else:
            tokens = list(TOKEN_RE.finditer(pending))
            for match in tokens:
                candidate = pending[: match.end()]
                if STRONG_BOUNDARY_RE.search(candidate):
                    boundary = match.end()
                    reason = "sentence"
                elif self.commit_clauses and CLAUSE_BOUNDARY_RE.search(candidate):
                    boundary = match.end()
                    reason = "clause"
            safe_token_count = max(0, len(tokens) - self.lookahead_words)
            if safe_token_count and tokens[safe_token_count - 1].end() > boundary:
                boundary = tokens[safe_token_count - 1].end()
                reason = "lookahead"

        if boundary:
            self._committed += pending[:boundary]
        return Commitment(
            received=self._received,
            committed=self._committed,
            pending=self._received[len(self._committed) :],
            reason=reason,
        )
