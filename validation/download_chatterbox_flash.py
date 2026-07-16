#!/usr/bin/env python3
"""Download the exact public Chatterbox-Flash snapshot and record its path."""

from __future__ import annotations

import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download


REPO_ID = "ResembleAI/chatterbox-flash"
REVISION = "4385507288b8197e6dab8b4e6b1603328d549d9d"


def main() -> None:
    snapshot = Path(
        snapshot_download(
            repo_id=REPO_ID,
            revision=REVISION,
            allow_patterns=[
                "t3_flash.safetensors",
                "s3gen.safetensors",
                "ve.safetensors",
                "tokenizer.json",
            ],
        )
    ).resolve()
    required = {
        name: (snapshot / name).stat().st_size
        for name in (
            "t3_flash.safetensors",
            "s3gen.safetensors",
            "ve.safetensors",
            "tokenizer.json",
        )
    }
    output = Path("/workspace/nano-flash-artifacts/g1/chatterbox-flash-download.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            {
                "repo_id": REPO_ID,
                "requested_revision": REVISION,
                "snapshot": str(snapshot),
                "resolved_revision": snapshot.name,
                "files": required,
                "total_bytes": sum(required.values()),
                "pass": snapshot.name == REVISION and all(required.values()),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    os.replace(temporary, output)
    print(output.read_text(), end="")


if __name__ == "__main__":
    main()
