#!/usr/bin/env python3
"""Evaluate mean teacher-forced MOSS loss on a prepared held-out manifest."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--moss-repo", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prepared-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.moss_repo))
    from finetuning.dataset import MossTTSNanoSFTDataset
    from finetuning.sft import compute_supervised_loss

    records = [json.loads(line) for line in args.prepared_jsonl.read_text().splitlines() if line]
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True
    ).to(device="cuda", dtype=torch.bfloat16)
    model._set_attention_implementation("sdpa")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, use_fast=False, local_files_only=True
    )
    dataset = MossTTSNanoSFTDataset(records, tokenizer=tokenizer, model_config=model.config, max_length=512)
    torch.cuda.reset_peak_memory_stats()
    losses = []
    started = time.perf_counter()
    with torch.inference_mode():
        for index in range(len(dataset)):
            batch = {
                key: value.cuda()
                for key, value in dataset.collate_fn([dataset[index]]).items()
            }
            loss = compute_supervised_loss(
                model,
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                channelwise_loss_weight=[1.0] + [2.0] * int(model.config.n_vq),
            )
            losses.append(float(loss.cpu()))
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    checks = {
        "records_present": len(losses) == len(records) and len(losses) > 0,
        "losses_finite": all(torch.isfinite(torch.tensor(losses)).tolist()),
        "losses_positive": min(losses) > 0,
        "fits_16gb_with_headroom": torch.cuda.max_memory_allocated() / 2**30 < 14.5,
    }
    evidence = {
        "schema_version": 1,
        "model": str(args.model.resolve()),
        "records": len(losses),
        "mean_loss": sum(losses) / len(losses),
        "min_loss": min(losses),
        "max_loss": max(losses),
        "losses": losses,
        "elapsed_seconds": elapsed,
        "peak_allocated_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
        "checks": checks,
        "pass": all(checks.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0 if evidence["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
