#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Run one reproducible Qwen3.6 CUDA AFD capability gate.

The runner deliberately delegates the CUDA lifecycle to the opt-in pytest
cases.  This keeps all assertions (exact token IDs, top-5 token sets, and
logprobs) in one test module while recording a small machine-readable summary
outside the Git worktree by default.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from xml.etree import ElementTree as ET

TEST_FILE = "tests/e2e/models/qwen3_6/test_e2e_gpu.py"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--topology",
        choices=("1a1f", "2a2f"),
        required=True,
    )
    parser.add_argument("--gate-side", choices=("attention", "ffn"), required=True)
    parser.add_argument(
        "--mode",
        choices=("eager", "graph"),
        required=True,
    )
    parser.add_argument("--batch-size", choices=(1, 2), type=int, required=True)
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--compare-native", action="store_true")
    parser.add_argument("--cleanup", action="store_true")
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path(os.environ.get("AFD_QWEN36_ARTIFACT_DIR", "/tmp/qwen36-afd")),
    )
    return parser


def _node_id(args: argparse.Namespace) -> str:
    if args.topology == "1a1f":
        if args.gate_side != "ffn" or args.mode != "eager" or args.batch_size != 1:
            raise ValueError("1a1f validates the FFN-local-router eager gate only")
        return f"{TEST_FILE}::test_qwen3_6_afd_1a1f_eager_matches_native_tp1"

    if args.topology == "2a2f" and args.mode == "eager":
        if args.gate_side != "ffn" or args.batch_size != 2:
            raise ValueError("2a2f eager validates the FFN-local-router batch-size 2")
        return f"{TEST_FILE}::test_qwen3_6_afd_2a2f_tp2_eager_matches_native_tp2"

    if args.topology == "2a2f" and args.mode == "graph":
        if args.gate_side != "ffn" or args.batch_size != 1:
            raise ValueError("2a2f graph validates FFN-local-router batch-size 1")
        return f"{TEST_FILE}::test_qwen3_6_afd_2a2f_tp2_graph_matches_native[b1]"

    raise ValueError(
        "requested topology, gate-side, mode, and batch-size are not gated",
    )


def _junit_result(path: Path) -> dict[str, int]:
    root = ET.parse(path).getroot()
    suites = root.findall("testsuite")
    return {
        key: sum(int(suite.attrib.get(key, "0")) for suite in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }


def main() -> int:
    args = _parser().parse_args()
    try:
        node_id = _node_id(args)
    except ValueError as exc:
        _parser().error(str(exc))

    if "AFD_QWEN3_6_E2E_MODEL" not in os.environ:
        _parser().error("set AFD_QWEN3_6_E2E_MODEL to the original BF16 checkpoint")

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    junit_path = args.artifacts_dir / f"{stamp}-{args.topology}-{args.mode}.xml"
    env = os.environ | {"AFD_QWEN3_6_E2E_MAX_TOKENS": str(args.max_tokens)}
    command = [
        sys.executable,
        "-m",
        "pytest",
        node_id,
        "-q",
        f"--junitxml={junit_path}",
    ]
    result = subprocess.run(command, check=False, env=env)
    summary = {
        "command": command,
        "topology": args.topology,
        "gate_side": args.gate_side,
        "mode": args.mode,
        "batch_size": args.batch_size,
        "max_tokens": args.max_tokens,
        "compare_native": args.compare_native,
        "returncode": result.returncode,
        "junit": str(junit_path),
    }
    if junit_path.exists():
        summary.update(_junit_result(junit_path))
    summary_path = args.artifacts_dir / f"{stamp}-{args.topology}-{args.mode}.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(summary_path)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
