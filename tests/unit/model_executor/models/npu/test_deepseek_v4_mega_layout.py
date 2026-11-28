# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""DSV4 routing/layout through CAMAsyncMegaAFDConnector only."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("vllm_ascend.models.deepseek_v4.model")

from afd_plugin.connectors.npu.async_cam_runtime import (  # noqa: E402
    async_cam_layout as async_cam,
)
from afd_plugin.model_executor.models.npu import deepseek_v4 as adapter  # noqa: E402
from afd_plugin.model_executor.models.npu import (  # noqa: E402
    deepseek_v4_attention_gate,
)
from tests.unit.fakes.npu_deepseek_v4 import (  # noqa: E402
    make_attention_gate_proxy as _make_proxy,
)


@pytest.mark.parametrize("tokens,rank", [(5, 0), (5, 1), (1, 1)])
def test_mega_routes_only_real_local_rows(monkeypatch, tokens, rank):
    from afd_plugin.model_executor.models.npu import deepseek_v4_mega

    proxy, _, _ = _make_proxy(
        monkeypatch, use_sequence_parallel=False, in_profile_run=False, mega=True
    )
    proxy.top_k = 2
    proxy.gate = SimpleNamespace(tid2eid=True)
    hidden = torch.arange(tokens * 4).reshape(tokens, 4).to(torch.bfloat16)
    normalized = hidden.float() + 0.125
    token_ids = torch.arange(tokens) + 10
    width = (tokens + 1) // 2
    rows = slice(rank * width, (rank + 1) * width)
    monkeypatch.setattr(
        deepseek_v4_mega,
        "get_tp_group",
        lambda: SimpleNamespace(world_size=2, rank_in_group=rank),
    )
    # Replace the proxy itself: accessing its attributes requires an active
    # vLLM forward context, which this CPU routing test does not create.
    monkeypatch.setattr(
        deepseek_v4_mega, "_EXTRA_CTX", SimpleNamespace(in_profile_run=False)
    )
    # Restore gathers the per-rank outputs back to the padded layout, so the
    # stand-in must return tp_size x width rows instead of the parent-length
    # input: two ranks of locally routed rows.
    monkeypatch.setattr(
        async_cam,
        "tensor_model_parallel_all_gather",
        lambda x, dim: torch.cat([x, x], dim=dim),
    )
    seen = []

    def route(_, x, *, input_ids, hidden_states_fp32):
        seen.append(len(x))
        assert torch.equal(x, hidden[rows])
        assert torch.equal(input_ids, token_ids[rows])
        assert torch.equal(hidden_states_fp32, normalized[rows])
        return torch.ones(len(x), 2), torch.zeros(len(x), 2, dtype=torch.int32)

    def send(_, x, context, **kwargs):
        valid = len(hidden[rows])
        assert len(x) == width
        assert torch.equal(x[:valid], hidden[rows])
        if valid < width:
            assert kwargs["topk_weights"][valid:].count_nonzero() == 0
            assert kwargs["topk_ids"][-1].tolist() == [0, 1]
        return lambda: x

    monkeypatch.setattr(
        deepseek_v4_attention_gate, "compute_attention_gate_topk", route
    )
    monkeypatch.setattr(adapter, "send_and_build_receive", send)
    proxy(hidden, input_ids=token_ids, hidden_states_fp32=normalized)
    assert sum(seen) == len(hidden[rows])
