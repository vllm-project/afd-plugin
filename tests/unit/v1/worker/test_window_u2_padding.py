# SPDX-License-Identifier: Apache-2.0
"""Run the real Window padding methods on CPU, without an Ascend worker."""

from __future__ import annotations

import ast
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch


@pytest.fixture
def runner():
    # Importing the full runner requires NPU extensions. Compile the unchanged
    # methods from its source instead; expectations below use real CPU buffers.
    path = (
        Path(__file__).resolve().parents[4]
        / "afd_plugin/v1/worker/npu/attention_model_runner.py"
    )
    tree = ast.parse(path.read_text())
    runner_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "AFDNPUAttentionModelRunner"
    )
    methods = [
        node
        for node in runner_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in {
            "_uses_fixed_window_u2",
            "_prepare_fixed_window_u2_dummy_padding",
            "_build_attention_metadata",
        }
    ]
    namespace = {
        "CUDAGraphMode": SimpleNamespace(NONE="none"),
        "np": np,
        "UBatchSlice": namedtuple("UBatchSlice", "request_slice token_slice"),
    }
    helpers_path = path.with_name("ubatch_utils.py")
    helpers = [
        node
        for node in ast.parse(helpers_path.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "create_ubatch_slices"
    ]
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[future, *helpers, *methods], type_ignores=[])
            ),
            str(path),
            "exec",
        ),
        namespace,
    )
    runner_type = type(
        "PaddingRunner",
        (),
        {method.name: namespace[method.name] for method in methods},
    )
    instance = runner_type()
    instance.connector = SimpleNamespace(is_window_connector=True, micro_batch_num=2)
    instance.query_start_loc = SimpleNamespace(
        np=np.full(9, -1, dtype=np.int32), copy_to_gpu=Mock()
    )
    instance.arange_np = np.arange(9, dtype=np.int32)
    instance.input_ids = SimpleNamespace(gpu=torch.full((8,), 77))
    instance.input_ids.gpu[0] = 42
    instance.positions = torch.full((8,), 99)
    instance.use_compress = True
    instance._dsa_positions_cpu_buf = torch.full((8,), 99)
    instance.optimistic_seq_lens_cpu = torch.full((8,), 99)
    instance.seq_lens = torch.full((8,), 99)
    instance.input_batch = SimpleNamespace(
        num_computed_tokens_cpu_tensor=torch.full((8,), 99),
        num_prompt_tokens_cpu_tensor=torch.full((8,), 99),
    )
    instance._pad_query_start_loc_for_fia = Mock(return_value=2)
    instance.speculative_config = None
    instance.afd_async_extra_info = SimpleNamespace(async_moe_ubatching=False)
    instance._build_afd_metadata = Mock(return_value=None)
    instance._build_attention_metadata_with_ubatches = Mock(
        side_effect=lambda **kwargs: kwargs
    )
    return instance


@pytest.mark.parametrize("live_requests", [1, 3])
def test_window_u2_slices_include_the_padded_request(runner, live_requests):
    capacity = live_requests + 1
    runner.query_start_loc.np[: live_requests + 1] = np.arange(live_requests + 1)
    result = runner._build_attention_metadata(
        num_tokens=live_requests,
        num_reqs=live_requests,
        max_query_len=1,
        num_tokens_padded=capacity,
        num_reqs_padded=capacity,
        ubatch_slices=[SimpleNamespace(token_slice=slice(0, live_requests))],
    )
    first, second = result["ubatch_slices"]
    assert first.token_slice == first.request_slice == slice(0, capacity // 2)
    assert second.token_slice == second.request_slice == slice(capacity // 2, capacity)
    assert runner.query_start_loc.np[capacity] == capacity


@pytest.mark.parametrize(
    ("live_boundaries", "dummy_requests"),
    [([0, 1], 1), ([0, 4], 1), ([0, 1, 5], 2)],
)
@pytest.mark.parametrize("use_compress", [False, True])
def test_pd_and_dspark_padding_pairs_each_dummy_request_with_one_token(
    runner, live_boundaries, dummy_requests, use_compress
):
    live_tokens = live_boundaries[-1]
    live_requests = len(live_boundaries) - 1
    runner.query_start_loc.np[: live_requests + 1] = live_boundaries
    runner.use_compress = use_compress
    runner._prepare_fixed_window_u2_dummy_padding(
        num_tokens=live_tokens,
        num_tokens_padded=live_tokens + dummy_requests,
        num_reqs=live_requests,
        num_reqs_padded=live_requests + dummy_requests,
    )

    assert runner.query_start_loc.np[: live_requests + dummy_requests + 1].tolist() == (
        live_boundaries + list(range(live_tokens + 1, live_tokens + dummy_requests + 1))
    )
    runner.query_start_loc.copy_to_gpu.assert_called_once_with()
    runner._pad_query_start_loc_for_fia.assert_not_called()
    token_slice = slice(live_tokens, live_tokens + dummy_requests)
    assert runner.input_ids.gpu[token_slice].tolist() == [42] * dummy_requests
    assert runner.positions[token_slice].tolist() == [127] * dummy_requests
    expected_positions = 127 if use_compress else 99
    assert runner._dsa_positions_cpu_buf[token_slice].tolist() == (
        [expected_positions] * dummy_requests
    )
    request_slice = slice(live_requests, live_requests + dummy_requests)
    for buffer in (
        runner.optimistic_seq_lens_cpu,
        runner.seq_lens,
        runner.input_batch.num_computed_tokens_cpu_tensor,
        runner.input_batch.num_prompt_tokens_cpu_tensor,
    ):
        assert buffer[request_slice].tolist() == [0] * dummy_requests
        assert buffer[:live_requests].tolist() == [99] * live_requests
    assert runner.positions[:live_tokens].tolist() == [99] * live_tokens


def test_invalid_live_boundary_fails_before_modifying_buffers(runner):
    runner.query_start_loc.np[:2] = [0, 2]
    with pytest.raises(RuntimeError, match="invalid live request boundary"):
        runner._prepare_fixed_window_u2_dummy_padding(
            num_tokens=1, num_tokens_padded=2, num_reqs=1, num_reqs_padded=2
        )
    runner.query_start_loc.copy_to_gpu.assert_not_called()
    assert runner.positions.tolist() == [99] * 8


@pytest.mark.parametrize("helper_capacity", [2, 3])
def test_other_padding_shapes_use_native_helper(runner, helper_capacity):
    runner._pad_query_start_loc_for_fia.return_value = helper_capacity
    arguments = dict(num_tokens=1, num_tokens_padded=4, num_reqs=1, num_reqs_padded=2)
    if helper_capacity == 2:
        runner._prepare_fixed_window_u2_dummy_padding(**arguments)
    else:
        with pytest.raises(RuntimeError, match="changed the request capacity"):
            runner._prepare_fixed_window_u2_dummy_padding(**arguments)
    runner._pad_query_start_loc_for_fia.assert_called_once_with(
        runner.query_start_loc, 4, 2, 1, "none", 2
    )


@pytest.mark.parametrize(
    ("window", "micro_batches", "token_capacity", "request_capacity"),
    [
        (False, 2, 2, 2),
        (True, 1, 2, 2),
        (True, 2, None, 2),
        (True, 2, 2, None),
        (True, 2, 1, 1),
    ],
)
def test_other_connectors_u1_and_unpadded_shapes_are_unchanged(
    runner, window, micro_batches, token_capacity, request_capacity
):
    runner.connector.is_window_connector = window
    runner.connector.micro_batch_num = micro_batches
    runner._prepare_fixed_window_u2_dummy_padding(
        num_tokens=1,
        num_tokens_padded=token_capacity,
        num_reqs=1,
        num_reqs_padded=request_capacity,
    )
    runner._pad_query_start_loc_for_fia.assert_not_called()
    runner.query_start_loc.copy_to_gpu.assert_not_called()
    assert runner.positions.tolist() == [99] * 8


def test_padding_smaller_than_live_batch_is_rejected(runner):
    with pytest.raises(RuntimeError, match="smaller than the live batch"):
        runner._prepare_fixed_window_u2_dummy_padding(
            num_tokens=3, num_tokens_padded=2, num_reqs=2, num_reqs_padded=1
        )
