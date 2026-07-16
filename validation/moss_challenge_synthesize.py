#!/usr/bin/env python3
"""Synthesize a deterministic English challenge suite with MOSS."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--audio-tokenizer", type=Path, required=True)
    parser.add_argument("--prompt-audio", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suite = json.loads(args.suite.read_text())
    device = torch.device("cuda")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True
    ).to(device=device, dtype=torch.bfloat16)
    model._set_attention_implementation("sdpa")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, use_fast=False, local_files_only=True
    )
    audio_tokenizer = AutoModel.from_pretrained(
        args.audio_tokenizer, trust_remote_code=True, local_files_only=True
    ).to(device)
    audio_tokenizer.eval()
    records = []
    for item in suite:
        output = args.output_dir / f"{item['id']}.wav"
        torch.manual_seed(20260716)
        torch.cuda.manual_seed_all(20260716)
        torch.cuda.synchronize()
        started = time.perf_counter()
        result_seen = False
        for event in model.inference_stream(
            text=item["text"],
            output_audio_path=output,
            mode="voice_clone",
            prompt_audio_path=args.prompt_audio,
            text_tokenizer=tokenizer,
            audio_tokenizer=audio_tokenizer,
            device="cuda",
            max_new_frames=192,
            do_sample=False,
            use_kv_cache=True,
            audio_repetition_penalty=1.2,
            voice_clone_max_text_tokens=0,
        ):
            result_seen = result_seen or event["type"] == "result"
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        audio, sample_rate = sf.read(output, dtype="float32", always_2d=True)
        records.append(
            {
                **item,
                "audio": str(output.resolve()),
                "sample_rate": sample_rate,
                "duration_seconds": len(audio) / sample_rate,
                "elapsed_seconds": elapsed,
                "rtf": elapsed / (len(audio) / sample_rate),
                "finite": bool(np.isfinite(audio).all()),
                "rms": float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))),
                "result_event_seen": result_seen,
            }
        )
    checks = {
        "ten_outputs": len(records) == 10,
        "all_result_events": all(record["result_event_seen"] for record in records),
        "all_finite": all(record["finite"] for record in records),
        "all_non_silent": all(record["rms"] > 1e-5 for record in records),
        "all_reasonable_duration": all(0.5 < record["duration_seconds"] < 30 for record in records),
    }
    evidence = {"schema_version": 1, "records": records, "checks": checks, "pass": all(checks.values())}
    path = args.output_dir / "synthesis.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)
    print(json.dumps({"checks": checks, "pass": evidence["pass"]}, indent=2))
    return 0 if evidence["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
