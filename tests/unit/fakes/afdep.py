# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""A stand-in for afdEP-comm's ``afdep`` package where it is not installed.

``DeepEPM2NAFDConnector`` imports ``afdep`` when its module loads. The stand-in
provides the names it imports, so connector tests run on hosts without
afdEP-comm; tests that build the lanes substitute ``FakeConfig`` and
``FakeLanes``.
"""

from __future__ import annotations

import sys
from importlib.machinery import ModuleSpec
from importlib.util import find_spec
from types import ModuleType

# Fields of afdep.Config at the pinned afdEP-comm. The stand-in requires the
# required ones and rejects any other keyword, so a transport built with
# arguments the real Config would refuse fails here as it would on a GPU.
CONFIG_REQUIRED = (
    "capacity_tokens_per_rank",
    "hidden_size",
    "num_experts",
    "top_k",
    "ffn_ranks",
    "attention_ranks",
)
CONFIG_OPTIONAL = (
    "expert_alignment",
    "payload",
    "allow_hybrid_mode",
    "num_sms",
    "num_qps",
    "rdma_traffic_class",
    "cpu_timeout_s",
    "gpu_timeout_s",
    "ffn_ranks_per_domain",
)


class FakeConfig:
    """Records the fields the transport is configured with."""

    def __init__(self, **kwargs: object) -> None:
        missing = sorted(set(CONFIG_REQUIRED) - set(kwargs))
        unknown = sorted(set(kwargs) - set(CONFIG_REQUIRED) - set(CONFIG_OPTIONAL))
        if missing or unknown:
            raise TypeError(
                f"Config() missing keywords {missing}, unexpected {unknown}"
            )
        self.kwargs = kwargs


class FakeLanes:
    """Records what the lanes are built with; ``afdep.Lanes`` is keyword-only."""

    def __init__(self, *, group: object, config: FakeConfig, lanes: int) -> None:
        self.group, self.config, self.num_lanes = group, config, lanes

    def __len__(self) -> int:
        return self.num_lanes


_FAKE_MARKER = "__afd_fake_afdep__"


def install_fake_afdep() -> None:
    """Register the stand-in ``afdep`` unless afdEP is importable."""
    if "afdep" in sys.modules or find_spec("afdep") is not None:
        return
    package = ModuleType("afdep")
    package.Buffer = type("Buffer", (), {})  # type: ignore[attr-defined]
    package.Config = FakeConfig  # type: ignore[attr-defined]
    package.DispatchHandle = type("DispatchHandle", (), {})  # type: ignore[attr-defined]
    package.Lanes = FakeLanes  # type: ignore[attr-defined]
    # A module without a spec makes a later find_spec("afdep") raise instead of
    # reporting the package as missing.
    package.__spec__ = ModuleSpec("afdep", None)
    setattr(package, _FAKE_MARKER, True)
    sys.modules["afdep"] = package


def uninstall_fake_afdep() -> None:
    """Remove the stand-in again, leaving a real afdEP install untouched.

    A stand-in left in ``sys.modules`` has no ``__spec__``, so a later
    ``find_spec("afdep")`` raises instead of reporting "not installed".
    """
    module = sys.modules.get("afdep")
    if module is not None and getattr(module, _FAKE_MARKER, False):
        sys.modules.pop("afdep", None)
