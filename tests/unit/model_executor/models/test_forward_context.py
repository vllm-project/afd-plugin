# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from afd_plugin.model_executor.models import (  # noqa: E402
    get_afd_metadata_from_forward_context,
)


def test_get_afd_metadata_from_additional_kwargs():
    forward_context = SimpleNamespace(
        additional_kwargs={"afd_metadata": {"stage": 0}},
        afd_metadata={"stage": 1},
    )

    assert get_afd_metadata_from_forward_context(forward_context) == {"stage": 0}


def test_get_afd_metadata_ignores_forward_context_attribute():
    forward_context = SimpleNamespace(
        additional_kwargs={},
        afd_metadata={"stage": 0},
    )

    assert get_afd_metadata_from_forward_context(forward_context) is None


def test_async_cam_profile_forward_runs_matched_connector_io(
    monkeypatch,
):  # long-test: profile communication must stay paired across consecutive layers
    from afd_plugin.model_executor.models.npu import (
        deepseek_v2_async_cam_forward as async_forward,
    )

    forward_context = SimpleNamespace(
        in_profile_run=True,
        ubatch_idx=0,
        flash_comm_v1_enabled=True,
    )
    monkeypatch.setattr(async_forward, "get_forward_context", lambda: forward_context)
    monkeypatch.setattr(
        async_forward,
        "maybe_apply_dbo_yield",
        lambda hidden_states, **_kwargs: hidden_states,
    )

    connector_calls: list[str] = []
    from afd_plugin.model_executor.models.npu import async_cam_layout

    monkeypatch.setattr(
        async_cam_layout,
        "get_tp_group",
        lambda: SimpleNamespace(world_size=1, rank_in_group=0),
    )

    def send_attn_output(*args, **kwargs):
        connector_calls.append("send")

    def recv_ffn_output(ref_tensor, ubatch_idx, context):
        connector_calls.append("recv")
        return ref_tensor

    connector = SimpleNamespace(
        send_attn_output=send_attn_output,
        recv_ffn_output=recv_ffn_output,
    )
    afd_metadata = SimpleNamespace(connector=connector, stage_idx=0)

    class _ProfileMoELayer:
        is_moe_layer = True
        use_sequence_parallel_moe = False

        layer_idx = 0
        mlp = SimpleNamespace(shared_experts=lambda x: 2 * x)

        def compute_attn_output(
            self,
            positions,
            hidden_states,
            residual,
            llama_4_scaling,
            *,
            already_sequence_parallel=False,
        ):
            return (
                hidden_states + 1,
                residual,
                torch.ones((hidden_states.shape[0], 1)),
                torch.zeros((hidden_states.shape[0], 1), dtype=torch.int32),
                torch.ones((hidden_states.shape[0], 1)),
            )

    model = SimpleNamespace(
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(use_sequence_parallel_moe=True),
        ),
        layers=[_ProfileMoELayer(), _ProfileMoELayer()],
        start_layer=0,
        end_layer=2,
    )
    hidden_states = torch.zeros((2, 4))

    output, residual = async_forward.run_attention_gate_afd_forward(
        model,
        hidden_states,
        None,
        torch.arange(2),
        afd_metadata,
    )

    assert torch.equal(output, (hidden_states + 1) * 9 + 3)
    assert residual is None
    assert connector_calls == ["send", "recv", "send", "recv"]
