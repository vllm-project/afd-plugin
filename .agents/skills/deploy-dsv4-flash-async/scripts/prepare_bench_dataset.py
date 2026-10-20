#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Materialize the bundled workload without changing its frozen JSONL bytes."""

import argparse
import gzip
import hashlib
import json
import os
import tempfile
from pathlib import Path

CHUNK_BYTES = 1024 * 1024


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(output: Path) -> Path:
    assets = Path(__file__).resolve().parent.parent / "assets" / "workloads"
    manifest = json.loads((assets / "manifest.json").read_text())
    expected = manifest["sha256"]
    output = output.absolute()
    if output.exists():
        if checksum(output) != expected:
            raise ValueError(f"Existing dataset has a different checksum: {output}")
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output.parent, prefix=output.name + ".", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            digest = hashlib.sha256()
            size = 0
            with gzip.open(assets / manifest["file"], "rb") as source:
                for chunk in iter(lambda: source.read(CHUNK_BYTES), b""):
                    handle.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        if digest.hexdigest() != expected or size != manifest["uncompressed_bytes"]:
            raise ValueError(
                "Bundled dataset failed its frozen checksum/size validation"
            )
        temporary.replace(output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(prepare(args.output))


if __name__ == "__main__":
    main()
