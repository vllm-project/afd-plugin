# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

"""One table owns upstream signature checks for scoped runtime patches."""

import inspect

import pytest


@pytest.mark.parametrize("hook", ["capture-inputs", "camp2p-dispatch"])
def test_scoped_patch_preserves_upstream_signature(hook):
    pytest.importorskip("torch")
    pytest.importorskip("vllm")
    if hook == "capture-inputs":
        from vllm.v1.worker.gpu import cudagraph_utils as target
        from vllm.v1.worker.gpu.pcp_manager import PCPManager  # noqa: F401

        from afd_plugin.v1.worker import attention_model_runner_v2 as adapter

        symbol = "prepare_inputs_to_capture"
        tracker = adapter._AFDCaptureEventTracker([])
        scope = adapter._use_afd_capture_input_preparation(None, tracker)
    else:
        pytest.importorskip("torch_npu")
        pytest.importorskip("vllm_ascend")
        from vllm.v1.worker.gpu import model_runner as target

        from afd_plugin.v1.worker.npu import attention_model_runner_v2 as adapter

        symbol = "dispatch_cg_and_sync_dp"
        scope = adapter._use_camp2p_dp_padding(4, 2)
    original = getattr(target, symbol)
    with scope:
        patched = getattr(target, symbol)
        assert patched is not original
        assert inspect.signature(
            patched, eval_str=True, locals=locals()
        ) == inspect.signature(original, eval_str=True, locals=locals())
    assert getattr(target, symbol) is original
