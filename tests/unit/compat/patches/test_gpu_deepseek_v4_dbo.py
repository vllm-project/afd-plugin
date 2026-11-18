# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from torch import nn  # noqa: E402
from vllm.forward_context import (  # noqa: E402
    ForwardContext,
    get_forward_context,
    override_forward_context,
)
from vllm.models.deepseek_v4.attention import DeepseekV4Attention  # noqa: E402
from vllm.models.deepseek_v4.nvidia.flashinfer_sparse import (  # noqa: E402
    DeepseekV4FlashInferMLAAttention,
    DeepseekV4FlashInferSM120Attention,
)
from vllm.models.deepseek_v4.nvidia.flashmla import (  # noqa: E402
    DeepseekV4FlashMLAAttention,
)
from vllm.v1.worker.workspace import use_workspace_lane  # noqa: E402

from afd_plugin.compat.patches import gpu_deepseek_v4_dbo as patches  # noqa: E402
from tests.unit.fakes.deepseek_v4 import (  # noqa: E402
    WorkspaceCase,
    afd_stage_context,
)


def test_v4_workspace_failure_restores_owner_and_context(monkeypatch):
    case = WorkspaceCase(monkeypatch, error=ValueError("native reservation failed"))
    case.ubatch = 1
    outer = ForwardContext(
        no_compile_layers={}, attn_metadata={"prior": None}, slot_mapping={}
    )
    with use_workspace_lane(1), override_forward_context(outer):
        before = case.manager.get_simultaneous(((1,), torch.uint8))[0]
        before.fill_(71)
        with pytest.raises(ValueError, match="native reservation failed"):
            patches._reserve_flashmla_ubatch_workspace(case.model, case.config)
        after = case.manager.get_simultaneous(((1,), torch.uint8))[0]
        assert get_forward_context() is outer
        assert after.data_ptr() == before.data_ptr()
        assert after.item() == 71


@pytest.mark.parametrize(
    "native_cls",
    [
        DeepseekV4FlashMLAAttention,
        DeepseekV4FlashInferMLAAttention,
        DeepseekV4FlashInferSM120Attention,
    ],
    ids=["flashmla", "flashinfer", "flashinfer-sm120"],
)
@pytest.mark.parametrize(
    "tokens,stages,expected", [(0, 2, 0), (2, 2, 3), (0, 1, 3), (None, 2, 3)]
)
def test_v4_padding_guard_skips_only_empty_dbo_attention(
    monkeypatch, native_cls, tokens, stages, expected
):
    def native_forward(self, positions, hidden_states, llama_4_scaling=None):
        return hidden_states * 3

    # Stub only the selected class's forward: a wrong class mapping would
    # reach the real native forward and fail.
    monkeypatch.setattr(native_cls, "forward", native_forward)
    attention = object.__new__(patches.afd_v4_attention_cls(native_cls))
    nn.Module.__init__(attention)
    hidden = torch.ones((4, 4))
    with override_forward_context(afd_stage_context(tokens=tokens, stages=stages)):
        output = attention(torch.arange(4), hidden, None)
    assert torch.equal(output, torch.full_like(hidden, expected))


def test_v4_padding_guard_rejects_unknown_attention_class():
    class FutureAttention(DeepseekV4Attention):
        pass

    with pytest.raises(ValueError, match="no padding-stage guard for FutureAttention"):
        patches.afd_v4_attention_cls(FutureAttention)
