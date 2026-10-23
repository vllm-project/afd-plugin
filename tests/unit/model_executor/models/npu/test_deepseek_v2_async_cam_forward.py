# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""FaaS marks only TP padding at the model-to-connector boundary."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from afd_plugin.config import AFDConfig  # noqa: E402
from afd_plugin.model_executor.models.npu import (  # noqa: E402
    async_cam_dbo,
    async_cam_layout,
)
from afd_plugin.model_executor.models.npu import (  # noqa: E402
    deepseek_v2_async_cam_forward as schedule,
)


@pytest.fixture
def dispatch_one_layer(monkeypatch):
    monkeypatch.setattr(async_cam_dbo, "dbo_enabled", lambda: False)
    monkeypatch.setattr(
        async_cam_layout,
        "get_tp_group",
        lambda: SimpleNamespace(world_size=2, rank_in_group=1),
    )
    monkeypatch.setattr(
        async_cam_layout,
        "tensor_model_parallel_all_gather",
        lambda tensor, dim: torch.cat((tensor, tensor)),
    )
    monkeypatch.setattr(schedule, "maybe_apply_dbo_yield", lambda tensor, **kw: tensor)
    hidden = torch.zeros(3, 2, dtype=torch.bfloat16)
    probabilities = torch.zeros(3, 6)
    ids = torch.zeros(3, 6, dtype=torch.int32)
    layer = SimpleNamespace(
        is_moe_layer=True,
        use_sequence_parallel_moe=False,
        layer_idx=1,
        mlp=SimpleNamespace(shared_experts=None),
        compute_attn_output=lambda *a, **kw: (hidden, None, probabilities, ids, None),
    )

    def dispatch(connector_name):
        sent = []
        connector = SimpleNamespace(
            send_attn_output=lambda states, context, **kw: sent.append(
                (states, context.metadata, kw["topk_ids"])
            ),
            recv_ffn_output=lambda ref_tensor, **kw: torch.zeros_like(ref_tensor),
        )
        model = SimpleNamespace(
            layers=[layer],
            start_layer=0,
            end_layer=1,
            afd_config=AFDConfig(connector=connector_name),
        )
        schedule.run_attention_gate_afd_forward(
            model,
            hidden,
            None,
            torch.arange(3),
            SimpleNamespace(stage_idx=0, connector=connector),
        )
        return sent[0]

    return dispatch


@pytest.mark.parametrize(
    "connector_name,padding_id",
    [("FaaSAsyncAFDConnector", -1), ("CAMAsyncAFDConnector", 0)],
)
def test_connector_receives_real_expert_zero_and_mode_specific_padding(
    dispatch_one_layer, connector_name, padding_id
):
    states, metadata, ids = dispatch_one_layer(connector_name)
    assert states.shape == (2, 2)
    assert metadata.seq_lens == [2]
    assert metadata.layer_idx == 1
    assert torch.equal(
        ids, torch.tensor([[0] * 6, [padding_id] * 6], dtype=torch.int32)
    )
