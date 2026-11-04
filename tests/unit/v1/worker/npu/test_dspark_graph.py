# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Draft graph scopes preserve independent AFD batches and native preparation."""

from contextlib import nullcontext

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("vllm_ascend")

from vllm.config import CUDAGraphMode  # noqa: E402
from vllm.v1.worker.gpu.cudagraph_utils import (  # noqa: E402
    AttentionState,
    BatchExecutionDescriptor,
)

from afd_plugin.v1.worker.npu import dspark_graph  # noqa: E402
from tests.unit.fakes.npu_attention import (  # noqa: E402
    DraftGraphManager,
    forward_context,
    metadata_runner,
)


@pytest.mark.parametrize("fail", [False, True])
def test_draft_replay_isolates_padding_and_restores_target_context(fail):
    runner = metadata_runner()
    target = forward_context(12)
    runner.install_afd_metadata_on_forward_context(target)
    target_metadata = target.additional_kwargs["afd_metadata"]
    manager = DraftGraphManager(runner, fail=fail)
    desc = BatchExecutionDescriptor(CUDAGraphMode.FULL, num_tokens=15, num_reqs=3)
    expected = (
        pytest.raises(RuntimeError, match="native replay failed")
        if fail
        else nullcontext()
    )
    with (
        expected,
        runner._afd_draft_context(False, False),
        dspark_graph.use_afd_draft_replay(runner, manager, real_tokens=10),
    ):
        manager.run_fullgraph(desc)
    draft = manager.emitted[0]
    assert draft.tokens_lens == [15]
    assert draft.tokens_unpadded_lens == [10]
    assert draft.transaction_id != target_metadata.transaction_id
    following = forward_context(12)
    runner.install_afd_metadata_on_forward_context(following)
    assert following.additional_kwargs["afd_metadata"] is target_metadata
    manager.fail = False
    assert manager.run_fullgraph(desc) is target_metadata


@pytest.mark.parametrize("fail", [False, True])
def test_draft_capture_creates_distinct_batches_and_restores_native_prepare(
    monkeypatch, fail
):
    runner = metadata_runner()
    desc = BatchExecutionDescriptor(CUDAGraphMode.FULL, num_tokens=10, num_reqs=2)
    manager = DraftGraphManager(runner, [desc])
    native_result = AttentionState({}, {})
    monkeypatch.setattr(
        dspark_graph.dflash_cudagraph,
        "_prepare_dflash_inputs_to_capture",
        lambda *args, **kwargs: native_result,
    )
    emitted = []
    expected = (
        pytest.raises(RuntimeError, match="native capture failed")
        if fail
        else nullcontext()
    )
    with (
        expected,
        runner._afd_draft_context(True, False),
        dspark_graph.use_afd_draft_capture(runner, manager),
    ):
        for _ in range(2):
            result = dspark_graph.dflash_cudagraph._prepare_dflash_inputs_to_capture(
                2, 10, None, None, None, None, 4096, False, False
            )
            assert result is native_result
            context = forward_context(10)
            runner.install_afd_metadata_on_forward_context(context)
            emitted.append(context.additional_kwargs["afd_metadata"])
        if fail:
            raise RuntimeError("native capture failed")
    assert [item.tokens_lens for item in emitted] == [[10], [10]]
    assert emitted[0].transaction_id != emitted[1].transaction_id
    assert (
        dspark_graph.dflash_cudagraph._prepare_dflash_inputs_to_capture()
        is native_result
    )


def test_draft_full_rejects_missing_native_capture_descriptors():
    runner = metadata_runner()
    with (
        pytest.raises(RuntimeError, match="no native capture descriptors"),
        dspark_graph.use_afd_draft_capture(runner, DraftGraphManager(runner)),
    ):
        pass
