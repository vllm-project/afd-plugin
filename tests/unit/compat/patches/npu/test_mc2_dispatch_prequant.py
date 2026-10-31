# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

"""The pre-quantized branch must accept only what the device established.

Three things are easy to get wrong here and each has its own case:

* the branch is chosen by "a scale is present", so a scale that does not describe
  the activation would be forwarded as though it did;
* an explicit communication quant mode disagrees with the pre-quantized one, and
  silently overriding it changes the caller's quantisation scheme;
* the routing and quant objects are shared with the combine step, so anything the
  branch writes back would change a later stage.

The module resolves vLLM-Ascend on first use, so these run on a CPU-only host
against stand-ins.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from afd_plugin.compat.patches.npu import mc2_dispatch_prequant as patch  # noqa: E402

_ROWS = 2
_HIDDEN = 4096
_SCALE_COLUMNS = _HIDDEN // 32
_TOP_K = 2
_EXPERT_COUNT = 8


def _activation(dtype=torch.float8_e4m3fn):
    return torch.zeros((_ROWS, _HIDDEN), dtype=dtype)


def _scale(dtype=torch.float8_e8m0fnu):
    return torch.zeros((_ROWS, _SCALE_COLUMNS), dtype=dtype)


def _dispatch_input(*, scale=None, comm_quant_mode=None, hidden=None, mxfp_act=None):
    if hidden is None:
        hidden = _activation()
    return SimpleNamespace(
        hidden_states=hidden,
        topk_weights=torch.zeros((_ROWS, _TOP_K), dtype=torch.float32),
        topk_ids=torch.zeros((_ROWS, _TOP_K), dtype=torch.int32),
        routing=SimpleNamespace(
            expert_map=torch.zeros(_EXPERT_COUNT, dtype=torch.int32),
            global_redundant_expert_num=0,
            pertoken_scale=scale,
            mc2_mask=torch.ones(_ROWS, dtype=torch.bool),
        ),
        quant=SimpleNamespace(
            comm_quant_mode=comm_quant_mode,
            dispatch_with_quant=True,
            is_mxfp=True,
            is_fp8=True,
            mxfp=SimpleNamespace(act_quant_type=mxfp_act),
        ),
    )


def _dispatcher(**overrides):
    defaults = {
        "need_shared_expert_args": True,
        "need_extra_args": False,
        "need_expert_scale": True,
        "global_bs": 0,
        "moe_all_to_all_group_name": "group",
        "ep_world_size": 4,
        "ep_rank_id": 0,
        "mc2_comm_alg": "alg",
        "moe_expert_num": _EXPERT_COUNT,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


@pytest.fixture(autouse=True)
def _stand_ins(monkeypatch):
    """Resolve the upstream names to stand-ins before the module caches anything."""

    patch._for_testing_reset()
    monkeypatch.setattr(
        patch,
        "TokenDispatcherWithMC2",
        SimpleNamespace(
            __qualname__="TokenDispatcherWithMC2",
            get_dispatch_mc2_kwargs=lambda self, x: {},
        ),
    )
    monkeypatch.setattr(patch, "_get_expert_token_nums_type", lambda _input: 1)
    yield
    patch._for_testing_reset()


def test_a_verified_prequantised_input_is_forwarded_as_is():
    scale = _scale()
    kwargs = patch.get_dispatch_mc2_kwargs(
        _dispatcher(),
        _dispatch_input(scale=scale),
    )

    assert kwargs["scales"] is scale
    assert kwargs["quant_mode"] == patch.PREQUANT_QUANT_MODE
    assert kwargs["y_dtype"] is torch.float8_e4m3fn


def test_an_explicit_mode_of_zero_is_accepted():
    """Zero agrees with the pre-quantized branch, so nothing conflicts."""

    kwargs = patch.get_dispatch_mc2_kwargs(
        _dispatcher(),
        _dispatch_input(scale=_scale(), comm_quant_mode=0),
    )

    assert kwargs["quant_mode"] == patch.PREQUANT_QUANT_MODE


def test_without_a_scale_upstream_behaviour_is_kept():
    """The FFN-side gate path: no scale, so every field upstream set stays as it was."""

    dispatch_input = _dispatch_input(scale=None)
    kwargs = patch.get_dispatch_mc2_kwargs(_dispatcher(), dispatch_input)

    assert kwargs["scales"] is None
    # MXFP dispatch with quant enabled resolves to 4 upstream, not to 0.
    assert kwargs["quant_mode"] == 4
    assert kwargs["x_active_mask"] is dispatch_input.routing.mc2_mask
    assert kwargs["group_ep"] == "group"
    assert kwargs["ep_world_size"] == 4
    assert kwargs["ep_rank_id"] == 0
    assert kwargs["comm_alg"] == "alg"


def test_a_conflicting_explicit_mode_is_refused():
    with pytest.raises(RuntimeError, match="conflicts with a pre-quantized activation"):
        patch.get_dispatch_mc2_kwargs(
            _dispatcher(),
            _dispatch_input(scale=_scale(), comm_quant_mode=2),
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("scale_dtype", torch.float32, "expects an E8M0 scale"),
        ("scale_dim", 3, "two-dimensional scale"),
        ("activation", torch.bfloat16, "expects an E4M3 activation"),
        ("mxfp_act", torch.bfloat16, "expects an E4M3 activation type"),
    ],
)
def test_a_scale_or_activation_outside_the_contract_is_refused(
    field,
    value,
    message,
):
    kwargs = {}
    if field == "scale_dtype":
        kwargs["scale"] = _scale(dtype=value)
    elif field == "scale_dim":
        kwargs["scale"] = torch.zeros((_ROWS, 64, 2), dtype=torch.float8_e8m0fnu)
    elif field == "activation":
        kwargs["scale"] = _scale()
        kwargs["hidden"] = _activation(dtype=value)
    else:
        kwargs["scale"] = _scale()
        kwargs["mxfp_act"] = value

    with pytest.raises(RuntimeError, match=message):
        patch.get_dispatch_mc2_kwargs(
            _dispatcher(),
            _dispatch_input(**kwargs),
        )


def test_the_shared_quant_and_routing_objects_are_not_written_back():
    """Combine reads the same objects, so the dispatch branch must not edit them."""

    dispatch_input = _dispatch_input(scale=_scale(), comm_quant_mode=0)
    quant_before = dict(vars(dispatch_input.quant))
    routing_before = dict(vars(dispatch_input.routing))

    patch.get_dispatch_mc2_kwargs(_dispatcher(), dispatch_input)

    assert vars(dispatch_input.quant) == quant_before
    assert vars(dispatch_input.routing) == routing_before


def test_install_refuses_when_the_method_was_already_replaced(monkeypatch):
    """A replacement made before the first install must not be mistaken for upstream."""

    patch._for_testing_reset()

    def foreign(self, x):
        return {}

    foreign.__qualname__ = "SomeOtherPatch.get_dispatch_mc2_kwargs"
    monkeypatch.setattr(
        patch,
        "TokenDispatcherWithMC2",
        SimpleNamespace(
            __qualname__="TokenDispatcherWithMC2",
            get_dispatch_mc2_kwargs=foreign,
        ),
    )
    monkeypatch.setattr(patch, "_get_expert_token_nums_type", lambda _input: 1)

    with pytest.raises(RuntimeError, match="already replaced"):
        patch.apply_afd_mc2_dispatch_prequant_patch()
