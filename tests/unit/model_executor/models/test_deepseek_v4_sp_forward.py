# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Exercise DSV4's model/Attention/CAM boundaries with CPU collectives."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import Tensor  # noqa: E402

pytest.importorskip("vllm_ascend.models.deepseek_v4.model")

from afd_plugin.model_executor.models.npu import (  # noqa: E402
    async_cam_layout,
)
from afd_plugin.model_executor.models.npu import (  # noqa: E402
    deepseek_v4_async_cam_forward as forward,
)
from afd_plugin.model_executor.models.npu import (  # noqa: E402
    deepseek_v4_attention_gate as gate,
)
from afd_plugin.model_executor.npu.async_cam_ubatching import (  # noqa: E402
    AsyncMoeStage,
)


@pytest.mark.parametrize("tp_rank", range(4))
def test_two_stage_sp_keeps_ids_and_attention_positions_aligned(monkeypatch, tp_rank):
    metadata = async_cam_layout.AsyncMoeUbatchMetadata(
        attn_metadata=[{"stage": 0}, {"stage": 1}],
        stages=(
            AsyncMoeStage(slice(0, 1), slice(0, 5), 8),
            AsyncMoeStage(slice(0, 1), slice(5, 12), 8),
        ),
        parent_input_tokens=13,
        use_sequence_parallel=True,
    )
    ids = torch.arange(100, 113)
    group = SimpleNamespace(world_size=4, rank_in_group=tp_rank)
    monkeypatch.setattr(forward, "get_tp_group", lambda: group)
    monkeypatch.setattr(async_cam_layout, "get_tp_group", lambda: group)
    monkeypatch.setattr(
        forward,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    context = SimpleNamespace(additional_kwargs={}, input_ids=ids)
    active = [context]
    monkeypatch.setattr(forward, "get_forward_context", lambda: active[-1])

    @contextmanager
    def override(child):
        active.append(child)
        try:
            yield
        finally:
            active.pop()

    monkeypatch.setattr(forward, "override_forward_context", override)
    pending = {}
    transfers = []

    def send(hidden, transfer, **payload):
        stage = transfer.metadata.stage_idx
        assert hidden.shape == (2, 1)
        assert payload["topk_ids"].shape == (2, 1)
        assert stage not in pending
        pending[stage] = hidden + 1
        transfers.append((transfer.metadata.layer_idx, stage))

    def receive(*, ref_tensor, ubatch_idx):
        assert ref_tensor.shape == (2, 1)
        return pending.pop(ubatch_idx)

    monkeypatch.setattr(
        forward,
        "get_afd_metadata_from_forward_context",
        lambda _ctx: SimpleNamespace(
            connector=SimpleNamespace(
                send_attn_output=send,
                recv_ffn_output=receive,
            )
        ),
    )
    layer_index = [0]
    attention_collectives = []

    def global_stage(stage, completed_layers):
        real_ids = ids[metadata.stages[stage].token_slice].float().unsqueeze(-1)
        values = torch.nn.functional.pad(real_ids, (0, 0, 0, 8 - len(real_ids)))
        return values * (2**completed_layers) + (2**completed_layers - 1)

    def gather_attention(local):
        stage = active[-1].ubatch_idx
        full = global_stage(stage, layer_index[0])
        assert torch.equal(local, full[tp_rank * 2 : (tp_rank + 1) * 2])
        attention_collectives.append(("gather", layer_index[0], stage))
        return full

    def reduce_attention(full):
        stage = active[-1].ubatch_idx
        assert len(full) == metadata.stages[stage].actual_tokens
        padded = torch.nn.functional.pad(full, (0, 0, 0, 8 - len(full)))
        attention_collectives.append(("scatter", layer_index[0], stage))
        return padded[tp_rank * 2 : (tp_rank + 1) * 2]

    monkeypatch.setattr(forward.native, "sp_all_gather", gather_attention)
    monkeypatch.setattr(forward.native, "sp_reduce_scatter", reduce_attention)

    class LocalMoE:
        shared_experts = None

    monkeypatch.setattr(forward, "AFDDeepseekV4AttentionGateRemoteMoE", LocalMoE)

    class Layer:
        enable_dsa_cp = False
        hc_attn_fn = hc_attn_scale = hc_attn_base = None
        hc_ffn_fn = hc_ffn_scale = hc_ffn_base = None
        input_layernorm = staticmethod(lambda x: x)
        rms_norm_cast = staticmethod(lambda x: (x, x.float()))
        hc_post = staticmethod(lambda x, *_args: x.unsqueeze(1).repeat(1, 2, 1))

        def __init__(self, index):
            self.layer_idx = index
            self.mlp = LocalMoE()

        def hc_pre(self, x, *_args):
            layer_index[0] = self.layer_idx
            return x.mean(1), None, None

        def self_attn(self, *, positions, hidden_states, llama_4_scaling):
            stage = metadata.stages[active[-1].ubatch_idx]
            assert positions.tolist() == list(
                range(stage.token_slice.start, stage.token_slice.stop)
            )
            assert hidden_states.shape[0] == len(positions)
            return hidden_states * 2

    routed_ids = []

    def topk(_moe, hidden, *, input_ids, hidden_states_fp32):
        stage = active[-1].ubatch_idx
        expected_ids = ids[metadata.stages[stage].token_slice]
        expected_ids = torch.nn.functional.pad(
            expected_ids, (0, 8 - len(expected_ids)), value=-1
        )
        expected_ids = expected_ids[tp_rank * 2 : (tp_rank + 1) * 2]
        assert torch.equal(input_ids, expected_ids)
        assert torch.equal(hidden_states_fp32, hidden.float())
        routed_ids.append(input_ids)
        return torch.ones(2, 1), torch.zeros(2, 1, dtype=torch.int32)

    monkeypatch.setattr(gate, "compute_attention_gate_topk", topk)
    final_gathers: list[Tensor] = []

    def gather_output(local, token_dim):
        stage = len(final_gathers)
        final_gathers.append(local)
        result = global_stage(stage, 2).unsqueeze(1).repeat(1, 2, 1)
        real_local = max(0, min(2, metadata.stages[stage].actual_tokens - tp_rank * 2))
        assert torch.equal(
            local[:real_local], result[tp_rank * 2 : tp_rank * 2 + real_local]
        )
        return result

    monkeypatch.setattr(
        async_cam_layout, "tensor_model_parallel_all_gather", gather_output
    )
    model = SimpleNamespace(
        use_sequence_parallel_moe=True,
        embed_input_ids=lambda token_ids: token_ids.float().unsqueeze(-1),
        hc_mult=2,
        layers=[Layer(0), Layer(1)],
        start_layer=0,
        end_layer=2,
        _needs_mtp_hidden_states=True,
        _mtp_hidden_buffer=None,
        _mtp_buffer_shape=(13, 2),
        _mtp_buffer_dtype=torch.float32,
        device="cpu",
        hc_head=lambda x, *_args: x.mean(1),
        hc_head_fn=None,
        hc_head_scale=None,
        hc_head_base=None,
        norm=lambda x: x,
    )
    output = forward.run_async_moe_ubatch_forward(
        model,
        ids,
        torch.arange(13),
        None,
        metadata,
        None,
    )
    expected = ids.float().unsqueeze(-1) * 4 + 3
    expected[-1] = 0  # Parent padding is restored after removing stage padding.
    assert torch.equal(output, expected)
    assert torch.equal(model._mtp_hidden_buffer, expected.repeat(1, 2))
    assert transfers == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert len(routed_ids) == 4
    assert len(attention_collectives) == 8
    assert len(final_gathers) == 2
    assert not pending
