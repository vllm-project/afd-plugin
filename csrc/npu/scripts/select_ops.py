#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Resolve the final AFD operator list to compile for a given SOC generation.

Reads operator_registry.json, validates the user selection, applies the SHMEM
availability filter, then prints the selected operator directory names (one
per line).

Notes:
  - Operator identity = source directory name under ascend_kernels/.
  - "utils" is a shared header directory, always copied by the build script
    and therefore never emitted here.
"""

import argparse
import json
import os
import sys


def die(msg):
    sys.stderr.write("ERROR: " + msg + "\n")
    sys.exit(1)


def load_registry(path):
    if not os.path.isfile(path):
        die(f"registry file not found: {path}")
    try:
        with open(path) as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        die(f"registry JSON parse error: {path}: {e}")


def get_soc_list(registry, soc):
    soc_versions = registry.get("soc_versions", {})
    if soc not in soc_versions:
        supported = ", ".join(soc_versions.keys()) if soc_versions else "(none)"
        die(f"SOC '{soc}' not registered; supported: [{supported}]")
    return list(soc_versions[soc])


def get_meta(registry, op):
    return registry.get("operator_meta", {}).get(op, {})


def requires_shmem(registry, op):
    return bool(get_meta(registry, op).get("requires_shmem", False))


def split_ops(raw):
    """Split the selection list ('a;b;c') and validate no empty entries."""
    if raw is None:
        return None
    parts = [p.strip() for p in raw.split(";")]
    if any(p == "" for p in parts):
        die("operator list contains empty entry")
    return parts


def validate_user_ops(ops, soc_list):
    """Every selected entry must be in the SOC support list."""
    for op in ops:
        if op not in soc_list:
            valid = ", ".join(soc_list)
            die(f"operator '{op}' not in SOC support list; valid: [{valid}]")


def resolve(registry, soc_list, shmem, user_ops):
    """Return the final ordered list of operator directory names to compile."""
    # 1. Determine the candidate set: full SOC list or explicit selection.
    candidates = list(soc_list) if user_ops is None else list(user_ops)

    # 2. SHMEM filter: drop requires_shmem operators when SHMEM is not installed.
    if not shmem:
        dropped = [o for o in candidates if requires_shmem(registry, o)]
        for o in dropped:
            sys.stderr.write(
                f"NOTE: dropping {o} (requires SHMEM, which is not installed)\n"
            )
        candidates = [o for o in candidates if not requires_shmem(registry, o)]

    # 3. Preserve registry order for determinism.
    ordered = [o for o in soc_list if o in candidates]
    return ordered


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Resolve the AFD Ascend operator list to compile for a SOC generation."
        )
    )
    parser.add_argument(
        "--registry", required=True, help="path to operator_registry.json"
    )
    parser.add_argument(
        "--soc", required=True, help="target SOC generation, e.g. ascend910_93"
    )
    parser.add_argument(
        "--shmem",
        choices=["1", "0"],
        required=True,
        help="whether SHMEM is installed (1/0)",
    )
    parser.add_argument(
        "--ops",
        default=None,
        help="semicolon-separated operator list; omit to compile the full SOC set",
    )
    args = parser.parse_args()

    registry = load_registry(args.registry)
    soc_list = get_soc_list(registry, args.soc)
    user_ops = split_ops(args.ops)
    if user_ops is not None:
        validate_user_ops(user_ops, soc_list)

    final_ops = resolve(registry, soc_list, args.shmem == "1", user_ops)

    if not final_ops:
        die(
            "no operators to compile after filtering "
            "(check SHMEM installation or operator selection)"
        )

    for op in final_ops:
        print(op)


if __name__ == "__main__":
    main()
