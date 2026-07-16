#!/usr/bin/env python3
"""Transcribe paired synthesis outputs and compare normalized WER/CER."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import librosa
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


ASR_REPO = "openai/whisper-tiny.en"
ASR_REVISION = "87c7102498dcde7456f24cfd30239ca606ed9063"


def normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, ref in enumerate(reference, 1):
        current = [row]
        for column, hyp in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[column] + 1, previous[column - 1] + (ref != hyp)))
        previous = current
    return previous[-1]


def score(reference: str, hypothesis: str) -> tuple[int, int, int, int]:
    ref = normalize(reference)
    hyp = normalize(hypothesis)
    ref_words, hyp_words = ref.split(), hyp.split()
    ref_chars, hyp_chars = list(ref.replace(" ", "")), list(hyp.replace(" ", ""))
    return edit_distance(ref_words, hyp_words), len(ref_words), edit_distance(ref_chars, hyp_chars), len(ref_chars)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--adapted", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    snapshot = Path(
        snapshot_download(
            repo_id=ASR_REPO,
            revision=ASR_REVISION,
            allow_patterns=[
                "*.json",
                "*.txt",
                "merges.txt",
                "vocab.json",
                "model.safetensors",
            ],
        )
    ).resolve()
    processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        snapshot, local_files_only=True, torch_dtype=torch.float16
    ).cuda().eval()
    groups = {}
    for name, root in (("baseline", args.baseline), ("adapted", args.adapted)):
        synthesis = json.loads((root / "synthesis.json").read_text())
        records = []
        word_errors = word_total = char_errors = char_total = 0
        for item in synthesis["records"]:
            audio, _ = librosa.load(item["audio"], sr=16000, mono=True)
            inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
            features = inputs.input_features.cuda(dtype=torch.float16)
            with torch.inference_mode():
                tokens = model.generate(features, max_new_tokens=128)
            transcript = processor.batch_decode(tokens, skip_special_tokens=True)[0]
            we, wt, ce, ct = score(item["text"], transcript)
            word_errors += we
            word_total += wt
            char_errors += ce
            char_total += ct
            records.append({"id": item["id"], "reference": item["text"], "transcript": transcript, "word_errors": we, "reference_words": wt, "char_errors": ce, "reference_chars": ct})
        groups[name] = {"records": records, "wer": word_errors / word_total, "cer": char_errors / char_total, "word_errors": word_errors, "reference_words": word_total, "char_errors": char_errors, "reference_chars": char_total}
    checks = {
        "twenty_transcripts": sum(len(group["records"]) for group in groups.values()) == 20,
        "baseline_wer_below_one": groups["baseline"]["wer"] < 1.0,
        "adapted_wer_below_one": groups["adapted"]["wer"] < 1.0,
        "adapted_wer_not_worse_by_5_points": groups["adapted"]["wer"] <= groups["baseline"]["wer"] + 0.05,
        "adapted_cer_not_worse_by_5_points": groups["adapted"]["cer"] <= groups["baseline"]["cer"] + 0.05,
    }
    evidence = {"schema_version": 1, "asr_repo": ASR_REPO, "asr_revision": ASR_REVISION, "groups": groups, "checks": checks, "pass": all(checks.values())}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"baseline": {"wer": groups["baseline"]["wer"], "cer": groups["baseline"]["cer"]}, "adapted": {"wer": groups["adapted"]["wer"], "cer": groups["adapted"]["cer"]}, "checks": checks, "pass": evidence["pass"]}, indent=2, sort_keys=True))
    return 0 if evidence["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
