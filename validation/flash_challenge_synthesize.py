#!/usr/bin/env python3
"""Synthesize the English challenge suite with one Flash configuration."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from chatterbox_flash import ChatterboxFlashTTS


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--prompt-audio", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--backend", choices=("torch", "flashinfer"), required=True)
    parser.add_argument("--block-size", type=int, required=True)
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = ChatterboxFlashTTS.from_local(
        args.snapshot, "cuda", dtype=torch.bfloat16, drf_block_size=args.block_size
    )
    model.prepare_conditionals(args.prompt_audio, exaggeration=0.5)

    def generate(text: str) -> torch.Tensor:
        torch.manual_seed(20260716)
        torch.cuda.manual_seed_all(20260716)
        return model.generate(
            text,
            num_steps=10,
            temperature=0.6,
            time_shift_tau=0.1,
            omnivoice_schedule_t_shift=0.5,
            cfg_scale=1.0,
            position_temperature=5.0,
            use_cuda_graph=args.cuda_graph,
            backend=args.backend,
            n_cfm_timesteps=2,
        )

    generate("Warm up.")
    records = []
    for item in json.loads(args.suite.read_text()):
        torch.cuda.synchronize()
        started = time.perf_counter()
        waveform = generate(item["text"])
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        audio = waveform.detach().cpu().float().flatten().numpy()
        output = args.output_dir / f"{item['id']}.wav"
        sf.write(output, audio, model.sr, subtype="PCM_16")
        duration = len(audio) / model.sr
        records.append(
            {
                **item,
                "audio": str(output.resolve()),
                "duration_seconds": duration,
                "elapsed_seconds": elapsed,
                "rtf": elapsed / duration,
                "finite": bool(np.isfinite(audio).all()),
                "rms": float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))),
            }
        )
    checks = {
        "ten_outputs": len(records) == 10,
        "all_finite": all(record["finite"] for record in records),
        "all_non_silent": all(record["rms"] > 1e-5 for record in records),
        "all_reasonable_duration": all(0.5 < record["duration_seconds"] < 30 for record in records),
    }
    evidence = {
        "schema_version": 1,
        "configuration": {"backend": args.backend, "block_size": args.block_size, "cuda_graph": args.cuda_graph},
        "records": records,
        "mean_rtf": sum(record["rtf"] for record in records) / len(records),
        "checks": checks,
        "pass": all(checks.values()),
    }
    output = args.output_dir / "synthesis.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps({"configuration": evidence["configuration"], "mean_rtf": evidence["mean_rtf"], "checks": checks, "pass": evidence["pass"]}, indent=2, sort_keys=True))
    return 0 if evidence["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
