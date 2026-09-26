# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Dispatcher tests for ``grouped_matmul_layered``.

Build the extension against an Ascend 910C toolchain, then run::

    SOC_VERSION=910c pytest tests/unit/compat/npu/test_gmm_layered.py

The Meta tests are exercised through the real PyTorch dispatcher on meta
tensors, so only the Python interface contract and the resulting output
shapes/dtypes are validated. No NPU kernel is launched and no numerical
correctness is asserted.

``grouped_matmul_layered`` is the layered A8W4 grouped matmul: the ``all_*``
inputs are per-layer TensorLists (one element per layer, each element covering
all experts of that layer) and a device-side ``layer_index`` selects which one
the kernel consumes. The Meta contract uses the shapes the A8W4 MSD dequant
path expects:

- ``x``: ``[M, K]`` int8 per-token quantized activation
- ``all_weight``: one layer weight ``[E, K, N // 8]`` int32 holding eight int4
  nibbles per word along the last axis
- ``all_bias``: one layer bias ``[E, N]`` float32
- ``all_scale``: one layer per-channel dequant scale ``[E, N]`` int64 (uint64 on
  the wire)
- ``per_token_scale``: ``[M]`` float32
- ``group_list``: ``[E]`` int64 token counts (``group_list_type=1``)
- ``layer_index``: ``[1]`` int64

Meta dispatch never reaches the op_api input validation, so the layer-count and
layer_index checks are only observable on the device path;
``test_gmm_layered_runtime_multi_layer`` covers that and needs a 910C device
plus the AFD CANN run package, so it is opt-in::

    SOC_VERSION=910c AFD_RUN_ASCEND_OP_RUNTIME=1 \\
        pytest tests/unit/compat/npu/test_gmm_layered.py
"""

from __future__ import annotations

import importlib
import importlib.util
import os
from types import ModuleType

import pytest

from afd_plugin.compat.npu.ops import ensure_afd_ascend_ops_loaded

pytestmark = pytest.mark.npu

_OP_NAME = "afd_ascend::grouped_matmul_layered"
_RUNTIME_ENV_VAR = "AFD_RUN_ASCEND_OP_RUNTIME"

# Meta shapes: deliberately small and dimension-true so the dispatcher contract
# is what is being checked, not the arithmetic.
_M = 4  # tokens
_K = 32  # reduction dimension
_E = 4  # experts
_N = 64  # output features
_INT4_PER_INT32 = 8

# Device shapes for the opt-in runtime test. Two layers are supplied so the
# device-side layer_index has a real choice.
_RUNTIME_LAYERS = 2
_RUNTIME_M = 128
_RUNTIME_K = 256
_RUNTIME_E = 8
_RUNTIME_N = 256


@pytest.fixture(scope="module")
def gmm_runtime() -> ModuleType:
    soc = os.environ.get("SOC_VERSION", "").lower()
    if soc != "910c" and not soc.startswith("ascend910_93"):
        pytest.skip("Set SOC_VERSION=910c to test the compiled gmm extension")
    for module in ("torch", "torch_npu", "afd_plugin._C_ascend"):
        if importlib.util.find_spec(module) is None:
            pytest.skip(f"{module} is required for compiled gmm dispatcher tests")
    # Optional runtime imports stay inside the fixture, so CPU test collection
    # remains independent of torch/torch_npu and their process-wide state.
    torch = importlib.import_module("torch")
    importlib.import_module("torch_npu")
    ensure_afd_ascend_ops_loaded()
    return torch


def _build_meta_inputs(torch: ModuleType, layers: int = 1) -> dict:
    """Return a valid A8W4 input set on the meta device.

    ``layers`` is the TensorList length. The host tiling requires every list to
    share one length, which is why all three are built from the same argument.
    """
    meta = {"device": "meta"}
    return {
        "x": [torch.empty((_M, _K), dtype=torch.int8, **meta)],
        "all_weight": [
            torch.empty((_E, _K, _N // _INT4_PER_INT32), dtype=torch.int32, **meta)
            for _ in range(layers)
        ],
        "all_bias": [
            torch.empty((_E, _N), dtype=torch.float32, **meta) for _ in range(layers)
        ],
        "all_scale": [
            torch.empty((_E, _N), dtype=torch.int64, **meta) for _ in range(layers)
        ],
        "layer_index": torch.zeros((1,), dtype=torch.int64, **meta),
        "per_token_scale": torch.empty((_M,), dtype=torch.float32, **meta),
        "group_list": torch.empty((_E,), dtype=torch.int64, **meta),
    }


def _invoke_gmm(torch: ModuleType, inputs: dict) -> list:
    return torch.ops.afd_ascend.grouped_matmul_layered(
        inputs["x"],
        inputs["all_weight"],
        inputs["all_bias"],
        inputs["all_scale"],
        inputs["layer_index"],
        inputs["group_list"],
        inputs["per_token_scale"],
        1,  # group_list_type: count
        3,  # split_item: NO_SEPARATED
        None,  # output_dtype: derive from the A8W4 scenario
    )


def test_gmm_layered_registers_inference_and_meta_kernels(
    gmm_runtime: ModuleType,
) -> None:
    schema = str(gmm_runtime.ops.afd_ascend.grouped_matmul_layered.default._schema)
    assert "Tensor group_list" in schema
    assert "int[]? group_list" not in schema
    for dispatch_key in ("PrivateUse1", "AutogradPrivateUse1", "Meta"):
        assert gmm_runtime._C._dispatch_has_kernel_for_dispatch_key(
            _OP_NAME, dispatch_key
        )


def test_gmm_layered_meta_contract(gmm_runtime: ModuleType) -> None:
    torch = gmm_runtime
    outputs = _invoke_gmm(torch, _build_meta_inputs(torch))

    # The layered output model is one [rows, n] tensor: the kernel writes each
    # group's rows at their offsets in a single buffer, and rows come from x
    # because x is a single (non-per-group) activation here.
    assert len(outputs) == 1
    y_out = outputs[0]

    assert tuple(y_out.shape) == (_M, _N)
    # A8W4 antiquant yields bf16 unless output_dtype overrides it.
    assert y_out.dtype == torch.bfloat16
    assert y_out.device.type == "meta"


def test_gmm_layered_meta_multi_layer_lists(gmm_runtime: ModuleType) -> None:
    """A multi-layer call must succeed on the meta path too.

    The all_* lengths are the layer count, not the expert count, so a list of
    two elements is the two-layer shape and the output geometry must not change
    with it.
    """
    torch = gmm_runtime
    outputs = _invoke_gmm(torch, _build_meta_inputs(torch, layers=_RUNTIME_LAYERS))

    assert len(outputs) == 1
    y_out = outputs[0]
    assert tuple(y_out.shape) == (_M, _N)
    assert y_out.dtype == torch.bfloat16


def test_gmm_layered_rejects_mismatched_list_lengths(gmm_runtime: ModuleType) -> None:
    """all_weight / all_bias / all_scale must agree on the layer count."""
    torch = gmm_runtime
    inputs = _build_meta_inputs(torch, layers=_RUNTIME_LAYERS)
    inputs["all_bias"] = inputs["all_bias"][:1]  # 1 vs 2 layers

    with pytest.raises(RuntimeError, match="same length"):
        _invoke_gmm(torch, inputs)


def test_gmm_layered_rejects_bad_layer_index_shape(gmm_runtime: ModuleType) -> None:
    """layer_index is a single int64 element, not a vector."""
    torch = gmm_runtime
    for bad in (
        torch.zeros((2,), dtype=torch.int64, device="meta"),  # not one element
        torch.zeros((1,), dtype=torch.int32, device="meta"),  # wrong dtype
    ):
        inputs = _build_meta_inputs(torch)
        inputs["layer_index"] = bad
        with pytest.raises(RuntimeError, match="layer_index"):
            _invoke_gmm(torch, inputs)


def test_gmm_layered_rejects_bad_split_item(gmm_runtime: ModuleType) -> None:
    torch = gmm_runtime
    inputs = _build_meta_inputs(torch)
    with pytest.raises(RuntimeError, match="split_item"):
        torch.ops.afd_ascend.grouped_matmul_layered(
            inputs["x"],
            inputs["all_weight"],
            inputs["all_bias"],
            inputs["all_scale"],
            inputs["layer_index"],
            inputs["group_list"],
            inputs["per_token_scale"],
            1,  # group_list_type
            9,  # split_item out of range
            None,
        )


def test_gmm_layered_runtime_multi_layer(gmm_runtime: ModuleType) -> None:
    """Run the A8W4 device path for two layers and check the output geometry.

    This is the only layer that reaches the op_api validation and the kernel, so
    it covers what Meta cannot: the per-layer TensorList indexing driven by the
    device-side layer_index and group list. It needs a 910C
    device and the AFD CANN run package, hence the opt-in gate.
    """
    if os.environ.get(_RUNTIME_ENV_VAR) != "1":
        pytest.skip(
            f"set {_RUNTIME_ENV_VAR}=1 on a 910C host with the AFD run package "
            "installed to run the A8W4 device invocation",
        )

    torch = gmm_runtime
    torch.npu.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    device = "npu"

    x = torch.zeros((_RUNTIME_M, _RUNTIME_K), dtype=torch.int8, device=device)
    all_weight = [
        torch.zeros(
            (_RUNTIME_E, _RUNTIME_K, _RUNTIME_N // _INT4_PER_INT32),
            dtype=torch.int32,
            device=device,
        )
        for _ in range(_RUNTIME_LAYERS)
    ]
    all_bias = [
        torch.zeros((_RUNTIME_E, _RUNTIME_N), dtype=torch.float32, device=device)
        for _ in range(_RUNTIME_LAYERS)
    ]
    all_scale = [
        torch.zeros((_RUNTIME_E, _RUNTIME_N), dtype=torch.int64, device=device)
        for _ in range(_RUNTIME_LAYERS)
    ]
    per_token_scale = torch.ones((_RUNTIME_M,), dtype=torch.float32, device=device)
    group_list = torch.full(
        (_RUNTIME_E,), _RUNTIME_M // _RUNTIME_E, dtype=torch.int64, device=device
    )

    for layer in range(_RUNTIME_LAYERS):
        outputs = torch.ops.afd_ascend.grouped_matmul_layered(
            [x],
            all_weight,
            all_bias,
            all_scale,
            torch.tensor([layer], dtype=torch.int64, device=device),
            group_list,
            per_token_scale,
            1,  # group_list_type: count
            3,  # split_item: NO_SEPARATED
            None,
        )
        torch.npu.synchronize()

        assert len(outputs) == 1
        y_out = outputs[0]
        assert tuple(y_out.shape) == (_RUNTIME_M, _RUNTIME_N)
        assert y_out.dtype == torch.bfloat16
        assert y_out.device.type == "npu"


def test_gmm_layered_device_group_list_meta(gmm_runtime: ModuleType) -> None:
    torch = gmm_runtime
    inputs = _build_meta_inputs(torch, layers=2)
    inputs["group_list"] = torch.empty((_E,), dtype=torch.int64, device="meta")
    # A slice with a nonzero storage offset is the production layer-index contract.
    inputs["layer_index"] = torch.empty((5,), dtype=torch.int64, device="meta")[2:3]
    output = torch.ops.afd_ascend.grouped_matmul_layered(
        **inputs,
        output_dtype=torch.bfloat16,
    )[0]
    assert output.shape == (_M, _N)
    assert output.dtype == torch.bfloat16
    cumulative = torch.ops.afd_ascend.grouped_matmul_layered(
        **inputs, group_list_type=0
    )[0]
    assert cumulative.shape == output.shape
    with pytest.raises(RuntimeError, match="group_list_type=0/1"):
        torch.ops.afd_ascend.grouped_matmul_layered(**inputs, group_list_type=2)
    inputs["group_list"] = torch.empty((_E,), dtype=torch.int32, device="meta")
    with pytest.raises(RuntimeError, match="1D int64"):
        torch.ops.afd_ascend.grouped_matmul_layered(**inputs)
