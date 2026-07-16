#!/usr/bin/env python3
"""Cache Nano renderer inputs/targets and prove deterministic rerendering."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import torch


SEED = 20260716


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def atomic_torch(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--prompt-audio", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    from chatterbox.tts_nano import ChatterboxNanoTTS

    model = ChatterboxNanoTTS.from_local(args.snapshot.resolve(), "cuda")
    model.prepare_conditionals(args.prompt_audio.resolve(), exaggeration=0.0, norm_loudness=True)
    atomic_torch(
        args.output_dir / "conditionals.pt",
        {"t3": model.conds.t3.__dict__, "gen": model.conds.gen},
    )
    suite = json.loads(args.suite.read_text())
    records = []
    for item in suite:
        captured = {}
        original = model.s3gen.inference

        def wrapper(*wrapper_args: Any, **wrapper_kwargs: Any) -> Any:
            tokens = wrapper_kwargs.get("speech_tokens", wrapper_args[0] if wrapper_args else None)
            captured["torch_rng"] = torch.get_rng_state().clone()
            captured["cuda_rng"] = torch.cuda.get_rng_state_all()
            result = original(*wrapper_args, **wrapper_kwargs)
            captured["tokens"] = tokens.detach().cpu().clone()
            captured["waveform"] = result[0].detach().cpu().clone()
            return result

        model.s3gen.inference = wrapper
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        try:
            model.generate(item["text"])
        finally:
            model.s3gen.inference = original
        cache_path = args.output_dir / f"{item['id']}.pt"
        atomic_torch(cache_path, captured)

        cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        torch.set_rng_state(cached["torch_rng"])
        torch.cuda.set_rng_state_all(cached["cuda_rng"])
        torch.cuda.synchronize()
        started = time.perf_counter()
        rerendered, _ = model.s3gen.inference(
            speech_tokens=cached["tokens"].cuda(),
            ref_dict=model.conds.gen,
            n_cfm_timesteps=2,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        rerendered = rerendered.detach().cpu()
        duration = rerendered.shape[-1] / model.sr
        records.append(
            {
                "id": item["id"],
                "speech_tokens": int(cached["tokens"].numel()),
                "speech_token_sha256": tensor_sha256(cached["tokens"]),
                "cached_waveform_sha256": tensor_sha256(cached["waveform"]),
                "rerendered_waveform_sha256": tensor_sha256(rerendered),
                "exact_rerender": torch.equal(cached["waveform"], rerendered),
                "duration_seconds": duration,
                "rerender_seconds": elapsed,
                "renderer_rtf": elapsed / duration,
                "cache_bytes": cache_path.stat().st_size,
            }
        )
    checks = {
        "ten_cache_entries": len(records) == 10,
        "all_tokens_present": all(record["speech_tokens"] > 0 for record in records),
        "all_exact_rerenders": all(record["exact_rerender"] for record in records),
        "all_renderer_rtfs_below_point_two": all(record["renderer_rtf"] < 0.2 for record in records),
        "conditionals_cached": (args.output_dir / "conditionals.pt").is_file(),
    }
    evidence = {
        "schema_version": 1,
        "snapshot_revision": args.snapshot.resolve().name,
        "seed": SEED,
        "records": records,
        "mean_renderer_rtf": sum(record["renderer_rtf"] for record in records) / len(records),
        "total_cache_bytes": sum(record["cache_bytes"] for record in records) + (args.output_dir / "conditionals.pt").stat().st_size,
        "checks": checks,
        "pass": all(checks.values()),
    }
    output = args.output_dir / "renderer-cache.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps({"mean_renderer_rtf": evidence["mean_renderer_rtf"], "total_cache_bytes": evidence["total_cache_bytes"], "checks": checks, "pass": evidence["pass"]}, indent=2, sort_keys=True))
    return 0 if evidence["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
