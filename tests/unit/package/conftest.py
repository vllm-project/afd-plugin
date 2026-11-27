# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Keep build-file tests independent of the host SoC.

``setup.py`` adds the split MegaMoE extension on 950 SoCs. The build-file tests
assert which extension set each flag produces, so an SoC exported by the host
toolchain must not leak in. Tests that care about the SoC set it themselves.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def no_host_soc_version(monkeypatch):
    monkeypatch.delenv("SOC_VERSION", raising=False)
