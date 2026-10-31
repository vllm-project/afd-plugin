# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from vllm.forward_context import (  # noqa: E402
    ForwardContext,
    get_forward_context,
    override_forward_context,
)
from vllm.v1.worker.workspace import use_workspace_lane  # noqa: E402

from afd_plugin.compat.patches import gpu_deepseek_v4_dbo as patches  # noqa: E402
from tests.unit.fakes.deepseek_v4 import WorkspaceCase  # noqa: E402


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
    "tokens,stages,expected", [(0, 2, 0), (2, 2, 3), (0, 1, 3), (None, 2, 3)]
)
def test_v4_padding_guard_skips_only_empty_dbo_attention(tokens, stages, expected):
    metadata = (
        SimpleNamespace(num_stages=stages, tokens_unpadded_lens=[tokens])
        if tokens is not None
        else None
    )
    context = ForwardContext(
        no_compile_layers={},
        attn_metadata={},
        slot_mapping={},
        additional_kwargs={"afd_metadata": metadata},
        is_padding=torch.ones(4, dtype=torch.bool),
    )
    hidden = torch.ones((4, 4))
    with override_forward_context(context):
        output = patches.forward_v4_attention(
            lambda positions, x, unused: x * 3, torch.arange(4), hidden
        )
    assert torch.equal(output, torch.full_like(hidden, expected))
