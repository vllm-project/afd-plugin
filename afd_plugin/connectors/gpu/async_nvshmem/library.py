# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Locate the NVSHMEM host library for both CMake and the runtime loader."""

import os
import sys
from importlib import import_module
from pathlib import Path


def find_host_library(configured_home: str | None = None) -> Path:
    home = configured_home or os.environ.get("NVSHMEM_HOME")
    if home:
        roots = [home]
    else:
        try:
            roots = list(import_module("nvidia.nvshmem").__path__)
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "Install nvidia-nvshmem-cu13 or set NVSHMEM_HOME to its installation."
            ) from error

    for root in roots:
        for name in ("libnvshmem_host.so.3", "libnvshmem_host.so"):
            library = Path(root, "lib", name)
            if library.is_file():
                return library.absolute()
    raise RuntimeError(f"No NVSHMEM host shared library found under {roots}")


if __name__ == "__main__":
    print(find_host_library(*sys.argv[1:]))
