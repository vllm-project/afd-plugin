# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU tensor checks for the NPU transport contracts retained from MRV1."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from afd_plugin.v1.worker.npu.model_io import (  # noqa: E402
    cat_ubatch_outputs,
    pp_transport_rows,
    pp_transport_slice,
    prepare_pp_transport_buffers,
)


def _config(model_type, use_sp, pp_size):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=4,
            pipeline_parallel_size=pp_size,
            use_sequence_parallel_moe=use_sp,
        ),
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type=model_type)
        ),
    )


@pytest.mark.parametrize("model_type", ["deepseek_v4", "deepseek_v2"])
@pytest.mark.parametrize("use_sp", [False, True])
@pytest.mark.parametrize("pp_size", [1, 2])
def test_allocation_and_stage_padding_follow_effective_model_wire_layout(
    model_type, use_sp, pp_size
):
    config = _config(model_type, use_sp, pp_size)
    sharded = model_type == "deepseek_v2" and use_sp and pp_size == 1
    assert pp_transport_rows(config, 13) == (4 if sharded else 13)
    stages = [SimpleNamespace(num_tokens=5), SimpleNamespace(num_tokens=7)]
    # Physical stage padding is independent: ceil(5/4)+ceil(7/4), not
    # ceil((5+7)/4). DSV4 and DSV2 PP2 carry every global token.
    assert pp_transport_rows(config, 12, stages) == (4 if sharded else 12)


@pytest.mark.parametrize("model_type", ["deepseek_v4", "deepseek_v2"])
@pytest.mark.parametrize("use_sp", [False, True])
@pytest.mark.parametrize("pp_size", [1, 2])
def test_two_stage_slices_retain_all_transported_fields(model_type, use_sp, pp_size):
    config = _config(model_type, use_sp, pp_size)
    sharded = model_type == "deepseek_v2" and use_sp and pp_size == 1
    rows = 4 if sharded else 12
    tensors = {
        "hidden_states": torch.arange(rows * 6).reshape(rows, 2, 3).float(),
        "pp_transport_aux_hidden_states_0": torch.arange(rows * 3)
        .reshape(rows, 3)
        .float(),
    }
    stages = [slice(0, 5), slice(5, 12)]
    for tensor in tensors.values():
        pieces = [tensor[pp_transport_slice(config, stage)] for stage in stages]
        assert [len(piece) for piece in pieces] == ([2, 2] if sharded else [5, 7])
        torch.testing.assert_close(torch.cat(pieces), tensor)
        assert (
            pieces[0].untyped_storage().data_ptr()
            == tensor.untyped_storage().data_ptr()
        )


def test_pp_growth_uses_incoming_capacity_and_retains_auxiliary_fields():
    previous = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
    incoming = torch.ones(12, 2, 3)
    auxiliary = torch.ones(12, 3, dtype=torch.bfloat16)
    buffers = {"hidden_states": previous}
    received = {
        "hidden_states": incoming,
        "pp_transport_aux_hidden_states_0": auxiliary,
    }
    grown = prepare_pp_transport_buffers(buffers, received, capacity=4)
    assert grown["hidden_states"].shape == (12, 2, 3)
    torch.testing.assert_close(grown["hidden_states"][:2], previous)
    assert grown["pp_transport_aux_hidden_states_0"].shape == (12, 3)
    assert grown["pp_transport_aux_hidden_states_0"].dtype == torch.bfloat16
    assert buffers == {"hidden_states": previous}
    # Native execute, rather than this preparation seam, owns the incoming copy.
    assert grown["hidden_states"].data_ptr() != incoming.data_ptr()
    reused = prepare_pp_transport_buffers(grown, received, capacity=12)
    assert all(reused[key] is value for key, value in grown.items())


@pytest.mark.parametrize("auxiliary", [False, True])
def test_merge_preserves_ascend_nested_auxiliary_outputs(auxiliary):
    first, second = torch.arange(5), torch.arange(5, 12)
    outputs = (
        [(first, [first + 10, first + 20]), (second, [second + 10, second + 20])]
        if auxiliary
        else [first, second]
    )
    merged = cat_ubatch_outputs(outputs)
    if auxiliary:
        hidden, aux = merged
        torch.testing.assert_close(hidden, torch.arange(12))
        assert isinstance(aux, list)
        torch.testing.assert_close(aux[0], torch.arange(12) + 10)
        torch.testing.assert_close(aux[1], torch.arange(12) + 20)
    else:
        torch.testing.assert_close(merged, torch.arange(12))
