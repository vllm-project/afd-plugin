#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Build only a target-side CS helper, without changing installed/native send ops."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OP = "afd_async_combine_send"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New, nonexistent build directory (never overwrites)",
    )
    parser.add_argument("--stage-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    # No in-place patching and no recursive deletion of user directories.
    output.mkdir(parents=True, exist_ok=False)
    staged = output / "csrc/npu"
    shutil.copytree(
        ROOT / "csrc/npu",
        staged,
        ignore=shutil.ignore_patterns("build", "output", "__pycache__"),
    )
    kernels = staged / "ascend_kernels"
    registry = kernels / "operator_registry.json"
    registry.write_text(
        json.dumps(
            {
                "soc_versions": {"ascend910_93": [OP]},
                "operator_meta": {OP: {"requires_shmem": False}},
            },
            indent=2,
        )
        + "\n"
    )
    helper = Path(__file__).with_name("cs_window.cpp")
    shutil.copyfile(helper, kernels / OP / "op_kernel" / f"{OP}.cpp")
    # Windows checkouts may contain CRLF; staged shell and kernel text is LF.
    for path in staged.rglob("*"):
        if path.is_file() and path.suffix in (
            ".sh",
            ".py",
            ".cpp",
            ".h",
            ".txt",
            ".json",
        ):
            path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n"))
    manifest = {
        "purpose": "issue487_cs_target_only",
        "helper_sha256": hashlib.sha256(
            helper.read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest(),
        "native_header_sha256": hashlib.sha256(
            (kernels / OP / "op_kernel" / f"{OP}.h").read_bytes()
        ).hexdigest(),
        "source_root": str(ROOT),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.stage_only:
        print(staged)
        return
    subprocess.run(
        ["bash", str(staged / "build_aclnn.sh"), str(output), "910c"], check=True
    )
    vendor = output / "afd_plugin/_cann_ops_custom/vendors/afd-plugin"
    if not (vendor / "op_api/lib/libcust_opapi.so").is_file():
        raise RuntimeError(f"Build did not produce a helper vendor at {vendor}")
    (vendor / "issue487_helper_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(f"\n--cs-helper {vendor}")


if __name__ == "__main__":
    main()
