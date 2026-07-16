#!/usr/bin/env python3
"""Replay incremental prompts at controlled arrival rates and measure commitment lag."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

from incremental_text import IncrementalCommitter, normalize_surface


RATES_CPS = (5, 10, 20, 40)
TICK_SECONDS = 0.02


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def replay(text: str, rate_cps: int) -> dict[str, Any]:
    text = normalize_surface(text)
    committer = IncrementalCommitter(lookahead_words=2)
    arrival_times: list[float] = []
    committed_at: list[float | None] = [None] * len(text)
    sent = 0
    committed = 0
    tick = 0
    first_commit: float | None = None
    max_pending = 0
    commit_events = 0

    while sent < len(text):
        now = tick * TICK_SECONDS
        available = min(len(text), int(math.floor(now * rate_cps)) + 1)
        if available > sent:
            fragment = text[sent:available]
            arrival_times.extend([now] * len(fragment))
            sent = available
            state = committer.push(fragment)
            if len(state.committed) > committed:
                for index in range(committed, len(state.committed)):
                    committed_at[index] = now
                committed = len(state.committed)
                commit_events += 1
                first_commit = now if first_commit is None else first_commit
            max_pending = max(max_pending, len(state.pending))
        tick += 1

    final_time = (tick - 1) * TICK_SECONDS
    state = committer.push("", final=True)
    if len(state.committed) > committed:
        for index in range(committed, len(state.committed)):
            committed_at[index] = final_time
        commit_events += 1
    lags = [float(committed_at[i] - arrival_times[i]) for i in range(len(text))]
    checks = {
        "lossless": state.committed == text,
        "all_characters_committed": all(value is not None for value in committed_at),
        "nonnegative_lag": min(lags) >= -1e-9,
        "bounded_pending": max_pending <= max(48, int(len(text) * 0.8)),
    }
    return {
        "rate_characters_per_second": rate_cps,
        "input_characters": len(text),
        "arrival_duration_seconds": round(final_time, 6),
        "first_commit_seconds": None if first_commit is None else round(first_commit, 6),
        "commit_events": commit_events,
        "max_pending_characters": max_pending,
        "commit_lag_p50_seconds": round(percentile(lags, 0.50), 6),
        "commit_lag_p95_seconds": round(percentile(lags, 0.95), 6),
        "commit_lag_max_seconds": round(max(lags), 6),
        "checks": checks,
        "pass": all(checks.values()),
    }


def main() -> int:
    args = parse_args()
    prompts = json.loads(args.suite.read_text())
    runs = [
        {"prompt_id": prompt["id"], **replay(prompt["text"], rate)}
        for rate in RATES_CPS
        for prompt in prompts
    ]
    by_rate = {}
    for rate in RATES_CPS:
        selected = [run for run in runs if run["rate_characters_per_second"] == rate]
        by_rate[str(rate)] = {
            "runs": len(selected),
            "first_commit_p50_seconds": round(
                percentile([run["first_commit_seconds"] for run in selected], 0.50), 6
            ),
            "commit_lag_p50_seconds": round(
                percentile([run["commit_lag_p50_seconds"] for run in selected], 0.50), 6
            ),
            "commit_lag_p95_seconds": round(
                percentile([run["commit_lag_p95_seconds"] for run in selected], 0.95), 6
            ),
            "max_pending_characters": max(run["max_pending_characters"] for run in selected),
        }
    checks = {
        "forty_scenarios": len(runs) == 40,
        "all_scenarios_pass": all(run["pass"] for run in runs),
        "all_rates_covered": set(map(int, by_rate)) == set(RATES_CPS),
    }
    evidence = {
        "schema_version": 1,
        "tick_seconds": TICK_SECONDS,
        "lookahead_words": 2,
        "rates_characters_per_second": RATES_CPS,
        "runs": runs,
        "summary_by_rate": by_rate,
        "checks": checks,
        "pass": all(checks.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(evidence["summary_by_rate"], indent=2, sort_keys=True))
    return 0 if evidence["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
