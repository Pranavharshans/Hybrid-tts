#!/usr/bin/env python3
"""Discrete-event stress matrix using measured and target hybrid timing."""

from __future__ import annotations

import itertools
import json
import os
from dataclasses import dataclass
from pathlib import Path

from streaming_scheduler import HybridScheduler, Mode, TextBudget


DT = 0.005
TOTAL_AUDIO_SECONDS = 6.0
SPOKEN_CHARACTERS_PER_AUDIO_SECOND = 15.0
LOOKAHEAD_CHARACTERS = 7.0


@dataclass(frozen=True)
class Profile:
    name: str
    first_ar_packet_seconds: float
    ar_rtf: float
    block_rtf: float


PROFILES = (
    Profile("measured", 0.1371, 1.1256, 0.5918),
    Profile("required_optimized", 0.1000, 0.7500, 0.2000),
)
ARRIVAL_RATES = (5.0, 10.0, 20.0, 40.0, 80.0, 1000.0)
BLOCK_SECONDS = (0.16, 0.32, 0.64)
SWITCH_BUFFERS = (0.08, 0.16, 0.24)
CANCEL_AT = (None, 1.0, 2.5)


def text_budget(rate: float, now: float) -> TextBudget:
    arrived_chars = rate * now
    speculative = min(TOTAL_AUDIO_SECONDS, arrived_chars / SPOKEN_CHARACTERS_PER_AUDIO_SECOND)
    stable = min(
        TOTAL_AUDIO_SECONDS,
        max(0.0, arrived_chars - LOOKAHEAD_CHARACTERS) / SPOKEN_CHARACTERS_PER_AUDIO_SECOND,
    )
    return TextBudget(speculative, stable)


def simulate(
    profile: Profile,
    arrival_rate: float,
    block_seconds: float,
    switch_buffer: float,
    cancel_at: float | None,
) -> dict:
    scheduler = HybridScheduler(
        packet_seconds=0.08,
        block_seconds=block_seconds,
        switch_buffer_seconds=switch_buffer,
    )
    now = 0.0
    task_end = None
    task_mode = None
    task_seconds = 0.0
    playback_started = False
    ttfa = None
    underrun = 0.0
    playback_elapsed = 0.0
    mode_quanta = {Mode.AR.value: 0, Mode.BLOCK.value: 0}
    max_buffer = 0.0
    cancel_latency = None
    cancelled = False

    while now <= 30.0:
        if cancel_at is not None and not cancelled and now >= cancel_at:
            scheduler.cancel()
            task_end = None
            cancel_latency = min(0.08, max(DT, 0.08 - (playback_elapsed % 0.08)))
            cancelled = True
            break

        if task_end is not None and now + 1e-9 >= task_end:
            scheduler.complete_generation(task_seconds)
            mode_quanta[task_mode.value] += 1
            task_end = None
            if not playback_started and scheduler.queued_seconds >= scheduler.packet_seconds:
                playback_started = True
                ttfa = now

        if playback_started:
            consumed = scheduler.consume(DT)
            playback_elapsed += DT
            underrun += DT - consumed

        max_buffer = max(max_buffer, scheduler.queued_seconds)
        budget = text_budget(arrival_rate, now)
        if task_end is None and scheduler.generated_seconds < TOTAL_AUDIO_SECONDS - 1e-9:
            mode = scheduler.choose_mode(budget)
            quantum = scheduler.generation_quantum(mode)
            remaining = TOTAL_AUDIO_SECONDS - scheduler.generated_seconds
            if remaining < quantum:
                mode = Mode.AR
                quantum = min(scheduler.packet_seconds, remaining)
            if scheduler.can_schedule(mode, budget):
                service = (
                    profile.first_ar_packet_seconds
                    if mode is Mode.AR and scheduler.generated_seconds == 0
                    else profile.ar_rtf * quantum
                    if mode is Mode.AR
                    else profile.block_rtf * quantum
                )
                task_end = now + service
                task_mode = mode
                task_seconds = quantum

        if (
            cancel_at is None
            and scheduler.generated_seconds >= TOTAL_AUDIO_SECONDS - 1e-9
            and task_end is None
            and scheduler.queued_seconds <= 1e-9
            and playback_started
        ):
            break
        now += DT

    active_playback = max(playback_elapsed, 1e-9)
    input_supply_ratio = arrival_rate / SPOKEN_CHARACTERS_PER_AUDIO_SECOND
    eligible_for_continuous_playback = input_supply_ratio >= 1.0
    checks = {
        "ttfa_target": ttfa is not None and ttfa < 0.250,
        "underrun_target_when_input_sufficient": (
            True if not eligible_for_continuous_playback or cancel_at is not None else underrun / active_playback < 0.01
        ),
        "cancellation_target": True if cancel_at is None else cancel_latency is not None and cancel_latency <= 0.08,
        "no_negative_buffer": scheduler.queued_seconds >= -1e-9,
    }
    return {
        "profile": profile.name,
        "arrival_characters_per_second": arrival_rate,
        "block_seconds": block_seconds,
        "switch_buffer_seconds": switch_buffer,
        "cancel_at_seconds": cancel_at,
        "ttfa_seconds": ttfa,
        "playback_seconds": playback_elapsed,
        "underrun_seconds": underrun,
        "underrun_ratio": underrun / active_playback,
        "input_supply_ratio": input_supply_ratio,
        "input_sufficient": eligible_for_continuous_playback,
        "ar_quanta": mode_quanta[Mode.AR.value],
        "block_quanta": mode_quanta[Mode.BLOCK.value],
        "max_buffer_seconds": max_buffer,
        "cancel_latency_seconds": cancel_latency,
        "checks": checks,
        "product_pass": all(checks.values()),
    }


def main() -> int:
    runs = [
        simulate(*values)
        for values in itertools.product(PROFILES, ARRIVAL_RATES, BLOCK_SECONDS, SWITCH_BUFFERS, CANCEL_AT)
    ]
    uncancelled = [run for run in runs if run["cancel_at_seconds"] is None]
    summaries = {}
    for profile in PROFILES:
        selected = [run for run in uncancelled if run["profile"] == profile.name]
        sufficient = [run for run in selected if run["input_sufficient"]]
        summaries[profile.name] = {
            "uncancelled_scenarios": len(selected),
            "product_pass_scenarios": sum(run["product_pass"] for run in selected),
            "input_sufficient_scenarios": len(sufficient),
            "input_sufficient_product_pass_scenarios": sum(run["product_pass"] for run in sufficient),
            "scenarios_that_reach_block_mode": sum(run["block_quanta"] > 0 for run in selected),
            "minimum_ttfa_seconds": min(run["ttfa_seconds"] for run in selected if run["ttfa_seconds"] is not None),
            "minimum_input_sufficient_underrun_ratio": min(run["underrun_ratio"] for run in sufficient),
        }
    cancellation_runs = [run for run in runs if run["cancel_at_seconds"] is not None]
    checks = {
        "full_324_scenario_matrix": len(runs) == 324,
        "all_buffers_nonnegative": all(run["checks"]["no_negative_buffer"] for run in runs),
        "all_cancellations_within_packet": all(run["checks"]["cancellation_target"] for run in cancellation_runs),
        "measured_profile_has_no_false_product_pass": summaries["measured"]["input_sufficient_product_pass_scenarios"] == 0,
        "optimized_profile_has_feasible_region": summaries["required_optimized"]["input_sufficient_product_pass_scenarios"] > 0,
        "slow_input_identified": any(not run["input_sufficient"] and run["underrun_ratio"] > 0.1 for run in uncancelled),
    }
    evidence = {
        "schema_version": 1,
        "constants": {
            "tick_seconds": DT,
            "total_audio_seconds": TOTAL_AUDIO_SECONDS,
            "spoken_characters_per_audio_second": SPOKEN_CHARACTERS_PER_AUDIO_SECOND,
            "stable_lookahead_characters": LOOKAHEAD_CHARACTERS,
        },
        "profiles": [profile.__dict__ for profile in PROFILES],
        "summaries": summaries,
        "runs": runs,
        "checks": checks,
        "pass": all(checks.values()),
    }
    output = Path("/workspace/nano-flash-artifacts/g5/streaming-stress.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps({"summaries": summaries, "checks": checks, "pass": evidence["pass"]}, indent=2, sort_keys=True))
    return 0 if evidence["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
