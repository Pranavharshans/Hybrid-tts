#!/usr/bin/env python3
"""Prove exact single-GPU MOSS continuation with model/optimizer/RNG state."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoModelForCausalLM, AutoTokenizer


SEED = 20260716


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--moss-repo", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def tensor_tree_sha256(value: Any) -> str:
    digest = hashlib.sha256()

    def visit(item: Any, prefix: str) -> None:
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            digest.update(prefix.encode())
            digest.update(str(tensor.dtype).encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                visit(item[key], f"{prefix}/{key}")
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                visit(child, f"{prefix}/{index}")
        else:
            digest.update(f"{prefix}={item!r}".encode())

    visit(value, "root")
    return digest.hexdigest()


def load_batch(args: argparse.Namespace, model: Any) -> dict[str, torch.Tensor]:
    sys.path.insert(0, str(args.moss_repo))
    from finetuning.dataset import MossTTSNanoSFTDataset

    records = [json.loads(line) for line in args.train_jsonl.read_text().splitlines() if line]
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, use_fast=False, local_files_only=True
    )
    dataset = MossTTSNanoSFTDataset(
        records, tokenizer=tokenizer, model_config=model.config, max_length=512
    )
    batch = dataset.collate_fn([dataset[0]])
    return {key: value.cuda(non_blocking=False) for key, value in batch.items()}


def load_training_state(args: argparse.Namespace) -> tuple[Any, AdamW, LambdaLR, dict[str, torch.Tensor]]:
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True
    ).to(device="cuda", dtype=torch.bfloat16)
    model._set_attention_implementation("sdpa")
    model.train()
    optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.0)
    scheduler = LambdaLR(optimizer, lambda _: 1.0)
    batch = load_batch(args, model)
    return model, optimizer, scheduler, batch


def train_step(model: Any, optimizer: AdamW, scheduler: LambdaLR, batch: dict[str, torch.Tensor]) -> float:
    from finetuning.sft import compute_supervised_loss

    optimizer.zero_grad(set_to_none=True)
    loss = compute_supervised_loss(
        model,
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
        channelwise_loss_weight=[1.0] + [2.0] * int(model.config.n_vq),
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    torch.cuda.synchronize()
    return float(loss.detach().cpu())


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    model, optimizer, scheduler, batch = load_training_state(args)
    loss_step_1 = train_step(model, optimizer, scheduler, batch)
    state_path = args.output_dir / "recovery-state.pt"
    temporary = state_path.with_suffix(".pt.tmp")
    torch.save(
        {
            "schema_version": 1,
            "completed_steps": 1,
            "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
        },
        temporary,
    )
    os.replace(temporary, state_path)
    loss_uninterrupted = train_step(model, optimizer, scheduler, batch)
    uninterrupted_model_hash = tensor_tree_sha256(model.state_dict())
    uninterrupted_optimizer_hash = tensor_tree_sha256(optimizer.state_dict())
    peak_first_process = torch.cuda.max_memory_allocated() / 2**30

    del batch, scheduler, optimizer, model
    gc.collect()
    torch.cuda.empty_cache()

    model, optimizer, scheduler, batch = load_training_state(args)
    recovered = torch.load(state_path, map_location="cpu", weights_only=False)
    model.load_state_dict(recovered["model"])
    optimizer.load_state_dict(recovered["optimizer"])
    scheduler.load_state_dict(recovered["scheduler"])
    torch.set_rng_state(recovered["torch_rng"])
    torch.cuda.set_rng_state_all(recovered["cuda_rng"])
    loss_resumed = train_step(model, optimizer, scheduler, batch)
    resumed_model_hash = tensor_tree_sha256(model.state_dict())
    resumed_optimizer_hash = tensor_tree_sha256(optimizer.state_dict())
    peak_second_process = torch.cuda.max_memory_allocated() / 2**30

    checks = {
        "losses_finite": all(torch.isfinite(torch.tensor([loss_step_1, loss_uninterrupted, loss_resumed])).tolist()),
        "step_two_loss_exact": loss_uninterrupted == loss_resumed,
        "step_two_model_exact": uninterrupted_model_hash == resumed_model_hash,
        "step_two_optimizer_exact": uninterrupted_optimizer_hash == resumed_optimizer_hash,
        "state_is_atomic_file": state_path.is_file() and not temporary.exists(),
        "state_under_2_gib": state_path.stat().st_size < 2 * 2**30,
        "fits_16gb_with_headroom": max(peak_first_process, peak_second_process) < 14.5,
    }
    evidence = {
        "schema_version": 1,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": SEED,
        "loss_step_1": loss_step_1,
        "loss_step_2_uninterrupted": loss_uninterrupted,
        "loss_step_2_resumed": loss_resumed,
        "uninterrupted_model_sha256": uninterrupted_model_hash,
        "resumed_model_sha256": resumed_model_hash,
        "uninterrupted_optimizer_sha256": uninterrupted_optimizer_hash,
        "resumed_optimizer_sha256": resumed_optimizer_hash,
        "recovery_state_bytes": state_path.stat().st_size,
        "peak_first_process_vram_gib": round(peak_first_process, 4),
        "peak_second_process_vram_gib": round(peak_second_process, 4),
        "checks": checks,
        "pass": all(checks.values()),
    }
    output = args.output_dir / "checkpoint-repro.json"
    output_tmp = output.with_suffix(".json.tmp")
    output_tmp.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    os.replace(output_tmp, output)
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0 if evidence["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
