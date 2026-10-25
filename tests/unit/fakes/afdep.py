# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""A stand-in for afdEP-comm's ``afdep`` package where it is not installed.

``DeepEPM2NAFDConnector`` imports ``afdep.deepep`` when its module loads. The
stand-in provides the names it imports, so connector tests run on hosts
without afdEP-comm; tests that build a lane substitute ``FakeElasticBuffer``.
"""

from __future__ import annotations

import sys
from importlib.util import find_spec
from types import ModuleType

# Keyword-only parameters of afdep.deepep.ElasticBuffer at the pinned afdEP-comm
# (afdEP/afdEP-comm 2b74d57). The stand-in requires the required ones and
# rejects any other keyword, so a lane built with arguments the real transport
# would refuse fails here as it would on a GPU.
ELASTIC_BUFFER_REQUIRED = (
    "group",
    "capacity_tokens_per_rank",
    "hidden_size",
    "num_experts",
    "top_k",
    "allow_hybrid_mode",
    "rdma_traffic_class",
    "cpu_timeout_s",
    "gpu_timeout_s",
)
ELASTIC_BUFFER_OPTIONAL = ("use_fp8_dispatch", "num_sms", "num_qps", "ffn_ranks")


class FakeElasticBuffer:
    """Records the arguments one lane's transport is built with."""

    def __init__(self, **kwargs: object) -> None:
        missing = sorted(set(ELASTIC_BUFFER_REQUIRED) - set(kwargs))
        unknown = sorted(
            set(kwargs) - set(ELASTIC_BUFFER_REQUIRED) - set(ELASTIC_BUFFER_OPTIONAL)
        )
        if missing or unknown:
            raise TypeError(
                f"ElasticBuffer() missing keywords {missing}, unexpected {unknown}"
            )
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
