# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Target and draft contexts publish independent AFD transaction metadata."""

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("vllm_ascend")

from vllm.v1.worker.dp_utils import should_skip_dp_coordination  # noqa: E402

from afd_plugin.v1.worker.npu.attention_model_runner_v2 import (  # noqa: E402
    AFDNPUAttentionModelRunnerV2,
)
from tests.unit.fakes.vllm_config import npu_vllm_config  # noqa: E402


@pytest.fixture
def metadata_runner():
    runner = object.__new__(AFDNPUAttentionModelRunnerV2)
    runner.vllm_config = npu_vllm_config()
    runner.vllm_config.kv_transfer_config = None
    runner.connector = SimpleNamespace(control_plane=None)
    runner._afd_pending_metadata = None
    runner._afd_suppress_metadata_send = False
    runner._afd_transaction_counter = 0
    runner._is_warmup = runner._afd_is_profile = False
    return runner


def _context(num_tokens):
    return SimpleNamespace(
        batch_descriptor=SimpleNamespace(num_tokens=num_tokens),
        attn_metadata={},
        additional_kwargs={},
        ubatch_slices=None,
    )


@pytest.mark.parametrize("fail_draft", [False, True])
def test_draft_uses_own_batch_and_restores_target_transaction(
    metadata_runner, fail_draft
):
    target = _context(12)
    metadata_runner.install_afd_metadata_on_forward_context(target)
    target_metadata = target.additional_kwargs["afd_metadata"]
    draft = _context(10)
    try:
        with metadata_runner._afd_draft_context(False, False):
            assert should_skip_dp_coordination()
            metadata_runner.install_afd_metadata_on_forward_context(draft)
            if fail_draft:
                raise RuntimeError("proposal failed")
    except RuntimeError:
        if not fail_draft:
            raise
    draft_metadata = draft.additional_kwargs["afd_metadata"]
    assert not should_skip_dp_coordination()
    assert draft_metadata.tokens_lens == [10]
    assert target_metadata.tokens_lens == [12]
    assert draft_metadata.transaction_id != target_metadata.transaction_id
    following_target = _context(12)
    metadata_runner.install_afd_metadata_on_forward_context(following_target)
    assert following_target.additional_kwargs["afd_metadata"] is target_metadata
