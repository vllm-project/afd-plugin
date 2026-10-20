#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Choose the native Decode draft from local checkpoint metadata."""

import argparse
import json
from pathlib import Path

DEFAULT_MTP_TOKENS = 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--config", default="auto")
    args = parser.parse_args()
    if args.config == "none":
        return
    try:
        checkpoint = json.loads((args.model / "config.json").read_text())
        if not isinstance(checkpoint, dict):
            parser.error("checkpoint config.json must contain an object")
        block_size = checkpoint.get("dspark_block_size")
        if block_size is not None and (type(block_size) is not int or block_size <= 0):
            parser.error("dspark_block_size must be a positive integer")
        if args.config == "auto":
            if block_size is not None:
                config = {
                    "method": "dspark",
                    "num_speculative_tokens": block_size,
                    "enforce_eager": True,
                }
            elif (checkpoint.get("num_nextn_predict_layers") or 0) > 0:
                config = {
                    "method": "mtp",
                    "num_speculative_tokens": DEFAULT_MTP_TOKENS,
                    "enforce_eager": True,
                }
            else:
                return
        else:
            config = json.loads(args.config)
            if not isinstance(config, dict):
                parser.error(
                    "DECODE_SPECULATIVE_CONFIG must be a JSON object, auto, or none"
                )
        if (
            config.get("method") == "mtp"
            and block_size is not None
            and config.get("model") in (None, str(args.model))
        ):
            parser.error("DSpark checkpoint requires method=dspark, not ordinary MTP")
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(config))


if __name__ == "__main__":
    main()
