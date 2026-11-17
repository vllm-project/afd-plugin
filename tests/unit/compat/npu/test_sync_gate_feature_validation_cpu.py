# mypy: disable-error-code="attr-defined, name-defined"
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU coverage for the sync gate-on-Attention feature contract.

The public validator is the seam every AFD NPU run passes before execution, so it is
driven directly rather than through its private helpers. Only the two upstream modules
the validator reaches for -- the connector factory and the connector's extra-info type
-- are stand-ins; the validator itself is the real one.

Keeping the assertions on the public entry matters: it fails both when the helper
starts rejecting supported shapes and when the entry stops calling the helper.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

from afd_plugin.compat.npu import feature_validation
from afd_plugin.config import CAMP2P_CONNECTOR
from tests.unit.fakes.stub_modules import (
    drop_stub_bound_modules as _drop_stub_bound_modules,
)

_DSV4 = "deepseek_v4"


class _FakeExtraInfo:
    """Stands in for the connector extra info the factory would build."""

    def __init__(self) -> None:
        self.validated = 0

    def validate_supported(self) -> None:
        self.validated += 1


@pytest.fixture
def extra_info(monkeypatch):
    info = _FakeExtraInfo()
    factory = types.ModuleType("afd_plugin.connectors.factory")
    factory.AFDConnectorFactory = SimpleNamespace(
        parse_connector_extra_info=lambda connector, config: info,
    )
    camp2p = types.ModuleType("afd_plugin.connectors.npu.camp2p")
    camp2p.CAMP2PExtraInfo = _FakeExtraInfo
    monkeypatch.setitem(sys.modules, "afd_plugin.connectors.factory", factory)
    monkeypatch.setitem(sys.modules, "afd_plugin.connectors.npu.camp2p", camp2p)
    return info


def _afd_config(*, role: str = "attention"):
    return SimpleNamespace(
        role=role,
        connector=CAMP2P_CONNECTOR,
        compute_gate_on_attention=True,
    )


def _vllm_config(*, tp: int = 1, dbo: bool = False, ubatching: bool = False):
    hf_config = SimpleNamespace(model_type=_DSV4)
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=hf_config,
            hf_text_config=SimpleNamespace(model_type=_DSV4),
        ),
        parallel_config=SimpleNamespace(
            enable_dbo=dbo,
            use_ubatching=ubatching,
            num_ubatches=1,
            tensor_parallel_size=tp,
        ),
        compilation_config=SimpleNamespace(cudagraph_mode=None),
        speculative_config=None,
    )


@pytest.mark.parametrize("role", ["attention", "ffn"])
def test_the_supported_shape_is_accepted_for_both_roles(extra_info, role):
    """Gate-on-Attention at TP=1 on the sync connector must pass validation."""

    feature_validation.fail_if_unsupported_npu_afd_features(
        _vllm_config(),
        afd_config=_afd_config(role=role),
    )

    assert extra_info.validated == 1


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"dbo": True}, "DBO or ubatching"),
        ({"ubatching": True}, "DBO or ubatching"),
        ({"tp": 2}, "tensor-parallel size 1"),
    ],
)
@pytest.mark.parametrize("role", ["attention", "ffn"])
def test_unsupported_shapes_are_rejected(extra_info, kwargs, match, role):
    with pytest.raises(RuntimeError, match=match):
        feature_validation.fail_if_unsupported_npu_afd_features(
            _vllm_config(**kwargs),
            afd_config=_afd_config(role=role),
        )


def test_gate_on_ffn_keeps_its_previous_behaviour(extra_info):
    """Turning the Attention-side gate off must not pick up the new restrictions."""

    afd_config = SimpleNamespace(
        role="ffn",
        connector=CAMP2P_CONNECTOR,
        compute_gate_on_attention=False,
    )

    feature_validation.fail_if_unsupported_npu_afd_features(
        _vllm_config(tp=2),
        afd_config=afd_config,
    )

    assert extra_info.validated == 1


def _real_extra_info():
    """Return the real ``CAMP2PExtraInfo`` with only its imports stubbed.

    ``camp2p`` pulls ``vllm`` at module scope, so a bare CPU host cannot import
    it. Stubbing the couple of runtime modules it touches keeps the class under
    test -- the parsing and the validation -- instead of skipping the cases that
    pin the connector-level contract.
    """

    # The class lives in a module that imports torch unconditionally; the gate
    # tests above run without it, so these two keep skipping on a torch-less host.
    pytest.importorskip("torch")

    import sys
    import types

    stubs = {}

    def module(name, package=False, **attrs):
        stub = types.ModuleType(name)
        if package:
            stub.__path__ = []
        for key, value in attrs.items():
            setattr(stub, key, value)
        stubs[name] = stub
        return stub

    def missing(name):
        try:
            __import__(name)
        except ImportError:
            return True
        return False

    if missing("vllm.forward_context"):
        vllm = module("vllm", package=True)
        module(
            "vllm.forward_context",
            DPMetadata=type("DPMetadata", (), {}),
            get_forward_context=lambda: None,
        )
        module(
            "vllm.logger",
            init_logger=lambda *args: __import__("logging").getLogger("vllm"),
        )
        module("vllm.utils", package=True)
        module(
            "vllm.utils.torch_utils",
            direct_register_custom_op=lambda **kwargs: None,
            is_torch_equal_or_newer=lambda *args: True,
        )
        module("vllm.distributed", package=True)
        module("vllm.distributed.parallel_state")
        vllm.forward_context = stubs["vllm.forward_context"]
        vllm.logger = stubs["vllm.logger"]
        vllm.utils = stubs["vllm.utils"]
        vllm.distributed = stubs["vllm.distributed"]

    stub_objects = {
        id(value) for stub in stubs.values() for value in vars(stub).values()
    }
    saved = {name: sys.modules.get(name) for name in stubs}
    before = set(sys.modules)
    sys.modules.update(stubs)
    try:
        from afd_plugin.connectors.npu.camp2p import CAMP2PExtraInfo
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
        # A module imported under the stubs keeps the stub callables alive -- the
        # connector is one, but so is everything else the connector pulled in
        # (``afd_plugin.connectors.metadata`` holds ``DPMetadata``), and the
        # parent package keeps the child as an attribute. Drop both (the leak
        # #568 describes).
        _drop_stub_bound_modules(before, stub_objects)

    return CAMP2PExtraInfo


def test_a_connector_level_gate_flag_does_not_block():
    """The switch is read from ``afd``; the connector-local copy has no say.

    Regression test for a rejection that used to live in ``validate_supported``: a
    supported
        shape was refused at startup whenever the flag also appeared under
        ``connector_extra_config``.
    """

    _real_extra_info().from_mapping(
        {"compute_gate_on_attention": True}
    ).validate_supported()


def test_quant_mode_is_still_rejected():
    """The transport mode really is unsupported, so that check has to stay."""

    with pytest.raises(RuntimeError, match="only quant_mode=0"):
        _real_extra_info().from_mapping({"quant_mode": 4}).validate_supported()
