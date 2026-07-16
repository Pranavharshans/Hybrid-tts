#!/usr/bin/env python3
"""Create the deliberate English 100-record overfit manifest."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


TEXT = (
    "OpenMOSS Team is jointly established by Shanghai Innovation Institution, "
    "Fudan University NLP Lab, and MOSI Intelligence, exploring an innovative "
    "development model centered on deep integration of industry, academia, and research."
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.audio.is_file():
        raise FileNotFoundError(args.audio)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".jsonl.tmp")
    with temporary.open("w") as handle:
        for index in range(100):
            handle.write(
                json.dumps(
                    {
                        "id": f"overfit-en-{index:03d}",
                        "audio": str(args.audio.resolve()),
                        "text": TEXT,
                        "language": "en",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    os.replace(temporary, args.output)
    print(f"records=100 output={args.output}")


if __name__ == "__main__":
    main()
