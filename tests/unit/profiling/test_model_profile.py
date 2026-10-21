# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from afd_plugin.profiling.model import (
    attention_profile_annotation,
    begin_profile_call,
    install_pipeline_hook,
    set_attention_profile_batch,
)


@pytest.mark.parametrize(
    "num_requests,num_tokens,expected", [(3, 9, (0, 3, 9)), (0, 0, (0, 0, 0))]
)
def test_attention_single_batch_uses_cpu_counts(num_requests, num_tokens, expected):
    metadata = SimpleNamespace(profile_batch=[], stage_idx=0)
    set_attention_profile_batch(metadata, num_requests, num_tokens, None)
    assert metadata.profile_batch == [expected]
    assert attention_profile_annotation(metadata, 7) == (
        f"0|7|{expected[1]}|{expected[2]}"
    )


def test_attention_microbatch_counts_clip_padding_and_keep_split_request():
    # The first request spans the split; the final stage contains only padding.
    slices = [
        SimpleNamespace(request_slice=slice(0, 1), token_slice=slice(0, 4)),
        SimpleNamespace(request_slice=slice(0, 4), token_slice=slice(4, 10)),
        SimpleNamespace(request_slice=slice(4, 6), token_slice=slice(10, 12)),
    ]
    metadata = SimpleNamespace(profile_batch=[], stage_idx=1)
    set_attention_profile_batch(metadata, 3, 9, slices)
    assert metadata.profile_batch == [(0, 1, 4), (1, 3, 5), (2, 0, 0)]
    assert attention_profile_annotation(metadata, 7) == "1|7|3|5"
    child = SimpleNamespace(profile_batch=[metadata.profile_batch[1]], stage_idx=1)
    assert attention_profile_annotation(child, 8) == "1|8|3|5"


def test_unknown_batch_is_unannotated():
    assert attention_profile_annotation(SimpleNamespace(profile_batch=[]), 7) == ""


def test_idle_pipeline_has_no_layer_lookup():
    assert begin_profile_call(SimpleNamespace(profile_pipeline=None)) is None


def test_pipeline_caches_track_but_checks_admission_each_layer():
    tracer = SimpleNamespace(retain_execution=Mock(return_value=True))
    execution = SimpleNamespace(tracer=tracer)
    metadata = SimpleNamespace(profile_pipeline=execution)
    assert begin_profile_call(metadata) is execution
    assert begin_profile_call(metadata) is execution
    assert metadata.profile_pipeline is execution
    tracer.retain_execution.return_value = False
    assert begin_profile_call(metadata) is None


def test_model_hook_refreshes_stream_for_each_dbo_child_and_generation():
    layer = SimpleNamespace(register_forward_pre_hook=Mock())
    module = SimpleNamespace(layers=[None, layer], start_layer=1)
    tracer = SimpleNamespace(execution=Mock(), transition=Mock())
    metadata = SimpleNamespace(
        connector=SimpleNamespace(profile_session=SimpleNamespace(tracer=tracer)),
        profile_pipeline=None,
    )
    install_pipeline_hook(module, lambda: metadata)
    hook = layer.register_forward_pre_hook.call_args.args[0]
    for _ in range(3):
        execution = SimpleNamespace(tracer=tracer)
        tracer.execution.return_value = execution
        hook(module, ())
        assert metadata.profile_pipeline is execution
        tracer.transition.assert_called_with(execution, "COMPUTE")
    tracer.execution.return_value = None
    hook(module, ())
    assert metadata.profile_pipeline is None
