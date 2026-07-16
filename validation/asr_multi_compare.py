#!/usr/bin/env python3
"""Compare ASR intelligibility for multiple synthesis directories."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from asr_compare import ASR_REPO, ASR_REVISION, load_16khz, score


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", action="append", required=True, help="name=/absolute/directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    groups = dict(value.split("=", 1) for value in args.group)
    snapshot = Path(
        snapshot_download(
            repo_id=ASR_REPO,
            revision=ASR_REVISION,
            allow_patterns=["*.json", "*.txt", "merges.txt", "vocab.json", "model.safetensors"],
        )
    ).resolve()
    processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        snapshot, local_files_only=True, torch_dtype=torch.float16
    ).cuda().eval()
    results = {}
    for name, directory in groups.items():
        synthesis = json.loads((Path(directory) / "synthesis.json").read_text())
        records = []
        we_total = wt_total = ce_total = ct_total = 0
        for item in synthesis["records"]:
            audio = load_16khz(item["audio"])
            features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(
                device="cuda", dtype=torch.float16
            )
            with torch.inference_mode():
                tokens = model.generate(features, max_new_tokens=128)
            transcript = processor.batch_decode(tokens, skip_special_tokens=True)[0]
            we, wt, ce, ct = score(item["text"], transcript)
            we_total += we
            wt_total += wt
            ce_total += ce
            ct_total += ct
            records.append({"id": item["id"], "reference": item["text"], "transcript": transcript, "word_errors": we, "reference_words": wt, "char_errors": ce, "reference_chars": ct})
        results[name] = {
            "wer": we_total / wt_total,
            "cer": ce_total / ct_total,
            "word_errors": we_total,
            "reference_words": wt_total,
            "char_errors": ce_total,
            "reference_chars": ct_total,
            "mean_synthesis_rtf": synthesis["mean_rtf"],
            "records": records,
        }
    torch_ref = results["torch-b16"]
    checks = {
        "three_groups": len(results) == 3,
        "thirty_transcripts": sum(len(result["records"]) for result in results.values()) == 30,
        "all_wer_below_one": all(result["wer"] < 1.0 for result in results.values()),
        "flash_b16_not_worse_by_5_points": results["flash-b16"]["wer"] <= torch_ref["wer"] + 0.05,
        "flash_b32_not_worse_by_5_points": results["flash-b32"]["wer"] <= torch_ref["wer"] + 0.05,
        "both_flash_meet_rtf_target": results["flash-b16"]["mean_synthesis_rtf"] < 0.2 and results["flash-b32"]["mean_synthesis_rtf"] < 0.2,
    }
    evidence = {"schema_version": 1, "asr_repo": ASR_REPO, "asr_revision": ASR_REVISION, "groups": results, "checks": checks, "pass": all(checks.values())}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"groups": {name: {key: value[key] for key in ("wer", "cer", "mean_synthesis_rtf")} for name, value in results.items()}, "checks": checks, "pass": evidence["pass"]}, indent=2, sort_keys=True))
    return 0 if evidence["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
