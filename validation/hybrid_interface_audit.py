#!/usr/bin/env python3
"""Audit whether the retained pretrained AR and block checkpoints can truly hand off."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import soundfile as sf
from safetensors import safe_open


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--moss-config", type=Path, required=True)
    parser.add_argument("--flash-checkpoint", type=Path, required=True)
    parser.add_argument("--flash-audio", type=Path, required=True)
    parser.add_argument("--moss-source", type=Path, required=True)
    parser.add_argument("--flash-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    moss = json.loads(args.moss_config.read_text())
    with safe_open(args.flash_checkpoint, framework="pt", device="cpu") as tensors:
        speech_embedding = list(tensors.get_slice("speech_emb.weight").get_shape())
        speech_head = list(tensors.get_slice("speech_head.weight").get_shape())
    _, flash_sample_rate = sf.read(args.flash_audio, dtype="float32")
    moss_stream = (args.moss_source / "moss_tts_nano").read_text(errors="ignore") if (args.moss_source / "moss_tts_nano").is_file() else ""
    flash_model_text = (args.flash_source / "chatterbox_flash/model.py").read_text(errors="ignore")

    facts = {
        "moss": {
            "architecture": moss["architectures"][0],
            "hidden_size": moss["hidden_size"],
            "audio_codebooks": len(moss["audio_codebook_sizes"]),
            "audio_codebook_sizes": moss["audio_codebook_sizes"],
            "audio_sample_rate": moss["audio_tokenizer_sample_rate"],
            "config_sha256": sha256(args.moss_config),
        },
        "flash": {
            "speech_embedding_shape": speech_embedding,
            "speech_head_shape": speech_head,
            "hidden_size": speech_embedding[1],
            "rendered_sample_rate": flash_sample_rate,
            "checkpoint_sha256": sha256(args.flash_checkpoint),
        },
    }
    checks = {
        "same_hidden_width": facts["moss"]["hidden_size"] == facts["flash"]["hidden_size"],
        "same_semantic_representation": facts["moss"]["audio_codebooks"] == 1
        and facts["moss"]["audio_codebook_sizes"][0] == speech_head[0],
        "same_native_audio_rate": facts["moss"]["audio_sample_rate"] == facts["flash"]["rendered_sample_rate"],
        "flash_exposes_continuation_state": "continuation_state" in flash_model_text,
    }
    evidence = {
        "schema_version": 1,
        "goal": "direct pretrained AR-to-block semantic-state handoff",
        "facts": facts,
        "checks": checks,
        "compatible": all(checks.values()),
        "interpretation": "Waveform concatenation is not semantic-state continuation and is excluded.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0 if evidence["compatible"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
