# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""A stand-in for the afdEP package where it is not installed.

``DeepEPM2NAFDConnector`` imports ``afdep.deepep`` when its module loads. The
stand-in provides the names it imports, so connector tests run on hosts
without afdEP; tests that build a lane substitute ``FakeElasticBuffer``.
"""

from __future__ import annotations

import sys
from importlib.util import find_spec
from types import ModuleType


class FakeElasticBuffer:
    """Records the arguments one lane's transport is built with."""

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


def install_fake_afdep() -> None:
    """Register the stand-in ``afdep.deepep`` unless afdEP is importable."""
    if "afdep" in sys.modules or find_spec("afdep") is not None:
        return
    package = ModuleType("afdep")
    deepep = ModuleType("afdep.deepep")
    deepep.ElasticBuffer = FakeElasticBuffer  # type: ignore[attr-defined]
    deepep.DispatchHandle = type("DispatchHandle", (), {})  # type: ignore[attr-defined]
    deepep.ReceiveResult = type("ReceiveResult", (), {})  # type: ignore[attr-defined]
    package.deepep = deepep  # type: ignore[attr-defined]
    sys.modules["afdep"] = package
    sys.modules["afdep.deepep"] = deepep
