# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU coverage of model-specific PP transport for TP4/PP2."""

import ast
import inspect
import textwrap
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")

from vllm.sequence import IntermediateTensors  # noqa: E402

from afd_plugin.compat.npu.runtime_config import (  # noqa: E402
    npu_model_uses_sharded_pp_tensors,
)
from afd_plugin.v1.worker.npu import attention_model_runner  # noqa: E402
from afd_plugin.v1.worker.npu.npu_ubatch_wrapper import (  # noqa: E402
    AscendUBatchWrapper,
)


def _config(model_type, use_sp, pipeline_parallel_size=2):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=4,
            pipeline_parallel_size=pipeline_parallel_size,
            use_sequence_parallel_moe=use_sp,
        ),
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type=model_type)
        ),
    )


def _allocate(*, batch_size, dtype, device):
    return IntermediateTensors(
        {"hidden_states": torch.zeros(batch_size, 2, 3, dtype=dtype, device=device)}
    )


@pytest.mark.parametrize("model_type", ["deepseek_v4", "deepseek_v2"])
@pytest.mark.parametrize("use_sp", [False, True])
@pytest.mark.parametrize("pipeline_parallel_size", [1, 2])
def test_pp_dummy_allocation_uses_model_wire_layout(
    model_type, use_sp, pipeline_parallel_size
):
    runner = SimpleNamespace(
        vllm_config=_config(model_type, use_sp, pipeline_parallel_size),
        max_num_tokens=13,
        intermediate_tensors=None,
        model=SimpleNamespace(make_empty_intermediate_tensors=_allocate),
        dtype=torch.float32,
        device="cpu",
        sync_and_slice_intermediate_tensors=lambda *_args: None,
    )
    # Run the actual allocation branch independently of the dummy forward's
    # NPU graph/Attention setup. This protects the first allocation as well as
    # subsequent sync_and_slice growth tested below.
    source = textwrap.dedent(
        inspect.getsource(
            attention_model_runner.AFDNPUAttentionModelRunner._dummy_run_with_ubatches
        )
    )
    tree = ast.parse(source)
    allocation = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "self.intermediate_tensors is None"
    )
    namespace = {
        "self": runner,
        "get_tensor_model_parallel_world_size": lambda: 4,
        "npu_model_uses_sharded_pp_tensors": npu_model_uses_sharded_pp_tensors,
    }
    exec(
        compile(
            ast.Module(body=[allocation], type_ignores=[]), "<PP-allocation>", "exec"
        ),
        namespace,
    )
    expected = (
        4
        if use_sp and model_type == "deepseek_v2" and pipeline_parallel_size == 1
        else 13
    )
    assert runner.intermediate_tensors["hidden_states"].shape[0] == expected


@pytest.mark.parametrize("model_type", ["deepseek_v4", "deepseek_v2"])
@pytest.mark.parametrize("use_sp", [False, True])
@pytest.mark.parametrize("ubatching", [False, True])
@pytest.mark.parametrize("pipeline_parallel_size", [1, 2])
def test_pp_copy_and_slice_preserves_every_transported_token(
    model_type, use_sp, ubatching, pipeline_parallel_size
):
    sharded = use_sp and model_type == "deepseek_v2" and pipeline_parallel_size == 1
    expected_rows = (4 if ubatching else 3) if sharded else 12
    runner = SimpleNamespace(
        vllm_config=_config(model_type, use_sp, pipeline_parallel_size),
        ubatch_slices=[SimpleNamespace(num_tokens=5), SimpleNamespace(num_tokens=7)]
        if ubatching
        else None,
        intermediate_tensors=_allocate(batch_size=2, dtype=torch.float32, device="cpu"),
        model=SimpleNamespace(make_empty_intermediate_tensors=_allocate),
        dtype=torch.float32,
        device="cpu",
    )
    incoming = IntermediateTensors(
        {
            "hidden_states": torch.arange(expected_rows * 6)
            .reshape(expected_rows, 2, 3)
            .float(),
            "pp_transport_aux_hidden_states_0": torch.arange(expected_rows * 3)
            .reshape(expected_rows, 3)
            .float(),
        }
    )
    runner_class = attention_model_runner.AFDNPUAttentionModelRunner
    method = runner_class.sync_and_slice_intermediate_tensors
    result = method(runner, 12, incoming, True)
    for key, values in incoming.items():
        assert torch.equal(result[key], values)
        assert result[key].shape[0] == expected_rows
    no_copy = method(runner, 12, None, False)
    assert torch.equal(no_copy["hidden_states"], incoming["hidden_states"])


@pytest.mark.parametrize("model_type", ["deepseek_v4", "deepseek_v2"])
@pytest.mark.parametrize("use_sp", [False, True])
@pytest.mark.parametrize("pipeline_parallel_size", [1, 2])
def test_pp_ubatch_slices_and_merge_retain_model_layout(
    model_type, use_sp, pipeline_parallel_size
):
    config = _config(model_type, use_sp, pipeline_parallel_size)
    sharded = use_sp and model_type == "deepseek_v2" and pipeline_parallel_size == 1
    rows = 4 if sharded else 12
    tensors = IntermediateTensors(
        {"hidden_states": torch.arange(rows * 6).reshape(rows, 2, 3).float()}
    )
    wrapper = object.__new__(AscendUBatchWrapper)
    wrapper.vllm_config = config
    slices = [slice(0, 5), slice(5, 12)]
    outputs = []
    for index, token_slice in enumerate(slices):
        ids, positions, embeds, child = wrapper._slice_model_inputs(
            token_slice,
            torch.arange(12),
            torch.arange(12),
            None,
            tensors,
        )
        expected_slice = slice(index * 2, (index + 1) * 2) if sharded else token_slice
        assert torch.equal(
            child["hidden_states"], tensors["hidden_states"][expected_slice]
        )
        assert (
            ids.shape[0] == positions.shape[0] == token_slice.stop - token_slice.start
        )
        assert embeds is None
        outputs.append(child)
    merged = wrapper._merge_intermediate_tensors(outputs)
    assert torch.equal(merged["hidden_states"], tensors["hidden_states"])
