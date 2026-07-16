#!/usr/bin/env python3
"""Validate prepared MOSS records and official teacher-forcing packing."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--moss-repo", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--text-tokenizer-snapshot", type=Path, required=True)
    parser.add_argument("--prepared-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import sys

    sys.path.insert(0, str(args.moss_repo))
    from finetuning.dataset import MossTTSNanoSFTDataset

    records = [json.loads(line) for line in args.prepared_jsonl.read_text().splitlines() if line]
    tensors = [torch.tensor(record["audio_codes"], dtype=torch.long) for record in records]
    hashes = [hashlib.sha256(tensor.numpy().tobytes()).hexdigest() for tensor in tensors]
    config = AutoConfig.from_pretrained(args.model_snapshot, trust_remote_code=True, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.text_tokenizer_snapshot,
        trust_remote_code=True,
        use_fast=False,
        local_files_only=True,
    )
    dataset = MossTTSNanoSFTDataset(records, tokenizer=tokenizer, model_config=config, max_length=512)
    first = dataset[0]
    last = dataset[len(dataset) - 1]
    batch = dataset.collate_fn([first, last])
    labels = batch["labels"]
    active_labels = labels[labels.ne(-100)]
    checks = {
        "exactly_100_records": len(records) == 100,
        "english_only": all(record.get("language") == "en" for record in records),
        "unique_ids": len({record.get("id") for record in records}) == 100,
        "all_codes_rank_two": all(tensor.ndim == 2 for tensor in tensors),
        "model_vq_width": all(tensor.shape[1] == int(config.n_vq) for tensor in tensors),
        "codes_nonnegative": min(int(tensor.min()) for tensor in tensors) >= 0,
        "codes_repeat_deterministically": len(set(hashes)) == 1,
        "packed_inputs_finite_shape": tuple(batch["input_ids"].shape) == (2, 511, int(config.n_vq) + 1),
        "supervised_labels_present": active_labels.numel() > 0,
        "padding_or_prompt_labels_masked": labels.eq(-100).any().item(),
        "first_last_packing_identical": torch.equal(first["full_input_ids"], last["full_input_ids"]),
    }
    evidence = {
        "schema_version": 1,
        "records": len(records),
        "audio_code_shape": list(tensors[0].shape),
        "audio_code_min": min(int(tensor.min()) for tensor in tensors),
        "audio_code_max": max(int(tensor.max()) for tensor in tensors),
        "audio_code_sha256": hashes[0],
        "packed_sequence_length": int(first["seq_len"]),
        "packed_prompt_length": int(first["prompt_length"]),
        "collated_input_shape": list(batch["input_ids"].shape),
        "active_supervised_values": int(active_labels.numel()),
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
