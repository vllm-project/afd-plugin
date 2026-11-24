# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
# Description: Host tests for compact receive/compute/return and MXFP4 layer
#              selection.
# Create: 2026-11-23
# Note: The native expert compute boundary is stubbed for host execution.
# History: 2026-11-23 create
"""Host tests for compact receive/compute/return and MXFP4 layer selection."""

from types import SimpleNamespace

import pytest

from tests.unit.fakes.async_ffn import (
    eager_mxfp_runner,
    load_async_modules,
    mxfp_runtime,
)

torch = pytest.importorskip("torch")


@pytest.fixture
def runtime(monkeypatch):
    _, _, runner = load_async_modules(monkeypatch)
    executor, layers = mxfp_runtime(monkeypatch, torch)
    return runner, executor, layers


@pytest.mark.parametrize(
    "layer_id,counts,expected",
    [(7, [2, 1], [7, 7, 8]), (3, [0, 2], [4, 4]), (3, [0, 0], [0])],
)
def test_950_push_returns_only_received_rows_and_the_original_header(
    monkeypatch, runtime, layer_id, counts, expected
):
    runner, _, layers = runtime
    header = torch.tensor([12, 0, layer_id, 0, 1])
    states = SimpleNamespace(
        token_nums_rankid_layeridx=header,
        group_list=torch.tensor(counts),
        dynamic_scales=None,
    )
    context = SimpleNamespace(states=states)
    payload = SimpleNamespace(
        hidden_states=torch.ones((6, 2), dtype=torch.bfloat16), context=context
    )
    sends = []
    session = eager_mxfp_runner(
        monkeypatch,
        torch,
        runner,
        layers,
        lambda: payload,
        lambda rows, ctx: sends.append((rows, ctx)),
    )
    assert session.execute_connector_driven_step()
    assert len(sends) == len(layers)
    rows, returned = sends[0]
    assert all(ctx is context for _, ctx in sends)
    torch.testing.assert_close(
        rows, torch.tensor(expected, dtype=torch.bfloat16)[:, None].expand(-1, 2)
    )
    assert torch.equal(returned.states.token_nums_rankid_layeridx, header)


@pytest.mark.parametrize(
    "layer,counts,error",
    [
        (1, [2, 1], "unknown FFN layer"),
        (3, [2, 4], "receive capacity"),
        (3, [1], "one count per local expert"),
    ],
)
def test_950_rejects_invalid_received_metadata(runtime, layer, counts, error):
    _, executor, layers = runtime
    with pytest.raises(ValueError, match=error):
        executor(layers, 2)(
            torch.ones((4, 2), dtype=torch.bfloat16),
            torch.tensor(counts),
            torch.tensor([9, 0, layer]),
        )


@pytest.mark.parametrize("bad", ["quant", "shared", "mega"])
def test_950_rejects_unsupported_expert_weights_at_startup(runtime, bad):
    _, executor, layers = runtime
    experts = layers[7].experts
    if bad == "quant":
        experts.quant_type = "w4a8"
    elif bad == "shared":
        experts._shared_experts = object()
    else:
        experts.routed_experts.w13_weight = None
    with pytest.raises(ValueError, match="layer 7 requires"):
        executor(layers, 2)
