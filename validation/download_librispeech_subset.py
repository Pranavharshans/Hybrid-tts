#!/usr/bin/env python3
"""Download a deterministic, distributed LibriSpeech subset via Dataset Viewer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


DATASET = "openslr/librispeech_asr"
CONFIG = "clean"
TRAIN_OFFSETS = (0, 5000, 10000, 15000, 20000)
VALID_OFFSETS = (0, 500, 1000, 1500)


def get_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def rows_url(split: str, offset: int, length: int) -> str:
    query = urllib.parse.urlencode(
        {"dataset": DATASET, "config": CONFIG, "split": split, "offset": offset, "length": length}
    )
    return f"https://datasets-server.huggingface.co/rows?{query}"


def download_split(root: Path, split: str, offsets: tuple[int, ...], length: int) -> list[dict[str, Any]]:
    destination = root / split.replace(".", "-")
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for offset in offsets:
        response = get_json(rows_url(split, offset, length))
        for entry in response["rows"]:
            row = entry["row"]
            audio_url = row["audio"][0]["src"]
            audio_path = destination / f"{row['id']}.flac"
            if not audio_path.is_file():
                temporary = audio_path.with_suffix(".flac.tmp")
                urllib.request.urlretrieve(audio_url, temporary)
                os.replace(temporary, audio_path)
            info = sf.info(audio_path)
            audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
            records.append(
                {
                    "id": row["id"],
                    "audio": str(audio_path.resolve()),
                    "text": row["text"].strip().title(),
                    "language": "en",
                    "speaker_id": int(row["speaker_id"]),
                    "chapter_id": int(row["chapter_id"]),
                    "dataset_row_index": int(entry["row_idx"]),
                    "sample_rate": int(sample_rate),
                    "duration_seconds": float(info.duration),
                    "finite": bool(np.isfinite(audio).all()),
                }
            )
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(".jsonl.tmp")
    with temporary.open("w") as handle:
        for record in records:
            training_record = {key: value for key, value in record.items() if key != "finite"}
            handle.write(json.dumps(training_record, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def aggregate_audio_hash(records: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: item["id"]):
        digest.update(record["id"].encode())
        with Path(record["audio"]).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train = download_split(args.output_dir, "train.100", TRAIN_OFFSETS, 20)
    valid = download_split(args.output_dir, "validation", VALID_OFFSETS, 5)
    write_jsonl(args.output_dir / "train.raw.jsonl", train)
    write_jsonl(args.output_dir / "valid.raw.jsonl", valid)
    train_speakers = {record["speaker_id"] for record in train}
    valid_speakers = {record["speaker_id"] for record in valid}
    checks = {
        "exactly_100_train": len(train) == 100,
        "exactly_20_valid": len(valid) == 20,
        "unique_ids": len({record["id"] for record in train + valid}) == 120,
        "speaker_disjoint": train_speakers.isdisjoint(valid_speakers),
        "all_16khz": all(record["sample_rate"] == 16000 for record in train + valid),
        "all_finite": all(record["finite"] for record in train + valid),
        "all_nonempty_text": all(record["text"] for record in train + valid),
        "all_reasonable_duration": all(0.5 < record["duration_seconds"] < 30 for record in train + valid),
    }
    evidence = {
        "schema_version": 1,
        "dataset": DATASET,
        "config": CONFIG,
        "license": "CC-BY-4.0",
        "source_card": f"https://huggingface.co/datasets/{DATASET}",
        "selection": {
            "train_split": "train.100",
            "train_offsets": TRAIN_OFFSETS,
            "rows_per_train_offset": 20,
            "valid_split": "validation",
            "valid_offsets": VALID_OFFSETS,
            "rows_per_valid_offset": 5,
        },
        "train_records": len(train),
        "valid_records": len(valid),
        "train_speakers": len(train_speakers),
        "valid_speakers": len(valid_speakers),
        "train_duration_seconds": sum(record["duration_seconds"] for record in train),
        "valid_duration_seconds": sum(record["duration_seconds"] for record in valid),
        "aggregate_audio_sha256": aggregate_audio_hash(train + valid),
        "checks": checks,
        "pass": all(checks.values()),
    }
    output = args.output_dir / "provenance.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0 if evidence["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
