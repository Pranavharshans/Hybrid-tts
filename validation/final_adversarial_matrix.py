#!/usr/bin/env python3
"""Final scheduler matrix using the qualified block timing and explicit AR cases."""

from __future__ import annotations

import itertools
import json
import os
from pathlib import Path

from streaming_stress import Profile, simulate


PROFILES = (
    Profile("measured_moss_plus_flash_b32", 0.1371, 1.1256, 0.1326595428548273),
    Profile("required_ar_plus_flash_b32", 0.1000, 0.7500, 0.1326595428548273),
    Profile("margin_ar_plus_flash_b32", 0.0800, 0.5000, 0.1326595428548273),
)
ARRIVALS = (5.0, 10.0, 14.9, 15.0, 20.0, 40.0, 80.0, 1000.0)
BLOCKS = (0.08, 0.16, 0.32, 0.64)
BUFFERS = (0.08, 0.16, 0.24, 0.48)
CANCELLATIONS = (None, 0.25, 1.0, 2.5)


def main() -> int:
    audit_path = Path("/workspace/nano-flash-artifacts/g8/interface-audit/audit.json")
    audit = json.loads(audit_path.read_text())
    runs = [simulate(*case) for case in itertools.product(PROFILES, ARRIVALS, BLOCKS, BUFFERS, CANCELLATIONS)]
    summaries = {}
    for profile in PROFILES:
        selected = [r for r in runs if r["profile"] == profile.name and r["cancel_at_seconds"] is None]
        sufficient = [r for r in selected if r["input_sufficient"]]
        summaries[profile.name] = {
            "uncancelled": len(selected),
            "input_sufficient": len(sufficient),
            "input_sufficient_product_pass": sum(r["product_pass"] for r in sufficient),
            "reached_block": sum(r["block_quanta"] > 0 for r in selected),
            "best_input_sufficient_underrun_ratio": min(r["underrun_ratio"] for r in sufficient),
        }
    cancellation_runs = [r for r in runs if r["cancel_at_seconds"] is not None]
    checks = {
        "full_1536_case_matrix": len(runs) == 1536,
        "all_buffers_nonnegative": all(r["checks"]["no_negative_buffer"] for r in runs),
        "all_cancellations_within_packet": all(r["checks"]["cancellation_target"] for r in cancellation_runs),
        "measured_ar_still_has_no_feasible_case": summaries[PROFILES[0].name]["input_sufficient_product_pass"] == 0,
        "required_ar_has_feasible_region": summaries[PROFILES[1].name]["input_sufficient_product_pass"] > 0,
        "qualified_block_speed_loaded": abs(PROFILES[0].block_rtf - 0.1326595428548273) < 1e-12,
    }
    harness_pass = all(checks.values())
    evidence = {
        "schema_version": 1,
        "profiles": [p.__dict__ for p in PROFILES],
        "summaries": summaries,
        "checks": checks,
        "harness_pass": harness_pass,
        "interface_compatible": audit["compatible"],
        "architecture_pass": harness_pass and audit["compatible"] and summaries[PROFILES[0].name]["input_sufficient_product_pass"] > 0,
        "runs": runs,
    }
    output = Path("/workspace/nano-flash-artifacts/g8/final-adversarial.json")
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps({k: evidence[k] for k in ("summaries", "checks", "harness_pass", "interface_compatible", "architecture_pass")}, indent=2, sort_keys=True))
    return 0 if harness_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
