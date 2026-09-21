# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Dispatcher tests for ``gmm_swiglu_quant_v2_layered``.

Build the extension against an Ascend 910C toolchain, then run::

    SOC_VERSION=910c pytest tests/unit/compat/npu/test_gmm_swiglu_quant_v2_layered.py

The Meta tests are exercised through the real PyTorch dispatcher on meta
tensors, so only the Python interface contract and the resulting output
shapes/dtypes are validated. No NPU kernel is launched and no numerical
correctness is asserted.

``grouped_matmul_swiglu_quant_v2_layered`` fuses a layered grouped matmul,
SiLU gating, and dynamic quantization into one kernel. The Meta test uses the
per-token quant path (``quant_mode=0``), whose input format and infershape
contract are self-consistent for the Meta binding:

- ``x``: ``[M, K]`` int8
- ``all_weight``: one layer weight ``[E, K, N]`` int8
- ``all_weight_scale``: one per-channel scale ``[E, N]`` float32
- ``all_weight_assist_matrix``: empty (bias semantics, unsupported)
- ``x_scale``: per-token scale ``[M]`` float32
- ``group_list``: token counts per expert ``[E]`` int64
- ``layer_index``: current layer ``[1]`` int64

Meta dispatch never reaches the op_api input validation, so the A4W4 rule that
rejects a non-null assist matrix is only observable on the device path.
``test_gmm_swiglu_quant_v2_layered_a4w4_empty_assist_matrix_runtime`` covers it
with a two-layer call and an empty assist list (the ``Tensor[]`` schema cannot
express ``None``). It needs a 910C device and the AFD CANN run package, so it
is opt-in::

    SOC_VERSION=910c AFD_RUN_ASCEND_OP_RUNTIME=1 \\
        pytest tests/unit/compat/npu/test_gmm_swiglu_quant_v2_layered.py
"""

from __future__ import annotations

import importlib
import importlib.util
import os
from types import ModuleType

import pytest

from afd_plugin.compat.npu.ops import ensure_afd_ascend_ops_loaded

pytestmark = pytest.mark.npu

_OP_NAME = "afd_ascend::gmm_swiglu_quant_v2_layered"
_RUNTIME_ENV_VAR = "AFD_RUN_ASCEND_OP_RUNTIME"
_M = 4  # tokens
_K = 32  # hidden / reduction dimension
_E = 4  # experts
_N = 64  # swiglu halves this into the output feature dimension

# A4W4 device-invocation shapes. int4 activations and weights are carried by an
# int32 storage whose last axis holds eight packed int4 values per element, and
# the op_api unpacks them to int4 before validation; the weight therefore keeps
# K int4 values per row and N // 8 packed elements. Two layers are supplied so
# the empty assist placeholder must match the per-layer list length.
_INT4_PER_INT32 = 8
_A4W4_LAYERS = 2
_A4W4_M = 128
_A4W4_K = 256
_A4W4_E = 8
_A4W4_N = 256
# fp32 1.0 in the low word of the wire-format uint64 per-channel scale.
_A4W4_SCALE_ONE = 0x3F800000


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


def _build_pertoken_inputs(torch: ModuleType) -> tuple:
    """Return a valid per-token (quant_mode=0) input set on the meta device."""
    x = torch.empty((_M, _K), device="meta", dtype=torch.int8)
    weight = torch.empty((_E, _K, _N), device="meta", dtype=torch.int8)
    weight_scale = torch.empty((_E, _N), device="meta", dtype=torch.float32)
    x_scale = torch.empty((_M,), device="meta", dtype=torch.float32)
    group_list = torch.empty((_E,), device="meta", dtype=torch.int64)
    layer_index = torch.empty((1,), device="meta", dtype=torch.int64)
    return x, [weight], [weight_scale], [], x_scale, group_list, layer_index


def _build_a4w4_inputs(torch: ModuleType) -> tuple:
    """Return a two-layer A4W4 (int4 x int4) input set on the NPU.

    The assist matrix is absent, which the ``Tensor[]`` schema can only express
    as an empty list.
    """
    device = "npu"
    x = torch.zeros(
        (_A4W4_M, _A4W4_K // _INT4_PER_INT32), device=device, dtype=torch.int32
    )
    weight = torch.zeros(
        (_A4W4_E, _A4W4_K, _A4W4_N // _INT4_PER_INT32),
        device=device,
        dtype=torch.int32,
    )
    # A4W4 per-channel scales are uint64 on the wire. torch int64 is the
    # torch_npu-compatible spelling that the op_api converts in place.
    weight_scale = torch.full(
        (_A4W4_E, _A4W4_N), _A4W4_SCALE_ONE, device=device, dtype=torch.int64
    )
    x_scale = torch.ones((_A4W4_M,), device=device, dtype=torch.float32)
    group_list = torch.full(
        (_A4W4_E,), _A4W4_M // _A4W4_E, device=device, dtype=torch.int64
    )
    layer_index = torch.ones((1,), device=device, dtype=torch.int64)
    return (
        x,
        [weight.clone() for _ in range(_A4W4_LAYERS)],
        [weight_scale.clone() for _ in range(_A4W4_LAYERS)],
        [],  # absent assist matrix: Tensor[] cannot express None
        x_scale,
        group_list,
        layer_index,
    )


def _invoke_gmm(torch: ModuleType, inputs: tuple) -> list:
    x, all_weight, all_weight_scale, all_assist, x_scale, group_list, layer_index = (
        inputs
    )
    return torch.ops.afd_ascend.gmm_swiglu_quant_v2_layered(
        x,
        all_weight,
        all_weight_scale,
        all_assist,
        x_scale,
        group_list,
        layer_index,
        0,  # dequant_mode
        0,  # quant_mode (per-token)
        0,  # group_list_type
        None,  # tuning_config
    )


def test_gmm_swiglu_quant_v2_layered_registers_inference_and_meta_kernels(
    gmm_runtime: ModuleType,
) -> None:
    for dispatch_key in ("PrivateUse1", "AutogradPrivateUse1", "Meta"):
        assert gmm_runtime._C._dispatch_has_kernel_for_dispatch_key(
            _OP_NAME, dispatch_key
        )


def test_gmm_swiglu_quant_v2_layered_meta_contract(gmm_runtime: ModuleType) -> None:
    torch = gmm_runtime
    outputs = _invoke_gmm(torch, _build_pertoken_inputs(torch))

    assert len(outputs) == 2
    y_out, y_scale_out = outputs

    # y is the swiglu result quantized to int8: [M, N // 2].
    assert tuple(y_out.shape) == (_M, _N // 2)
    assert y_out.dtype == torch.int8

    # Per-token output scale: [M] float32.
    assert tuple(y_scale_out.shape) == (_M,)
    assert y_scale_out.dtype == torch.float32

    assert all(tensor.device.type == "meta" for tensor in outputs)


def test_gmm_swiglu_quant_v2_layered_a4w4_empty_assist_matrix_runtime(
    gmm_runtime: ModuleType,
) -> None:
    """A4W4 must accept an empty assist list on the real CANN device path.

    This is the only layer that reaches the op_api validation. It covers both
    halves of the fix: an absent assist matrix must be normalized to nullptr
    before the A4W4 check rejects it, and the empty placeholder must carry one
    element per layer so the host tiling list-length check accepts a two-layer
    call.
    """
    if os.environ.get(_RUNTIME_ENV_VAR) != "1":
        pytest.skip(
            f"set {_RUNTIME_ENV_VAR}=1 on a 910C host with the AFD run package "
            "installed to run the A4W4 device invocation",
        )

    torch = gmm_runtime
    torch.npu.set_device(int(os.environ.get("LOCAL_RANK", "0")))

    outputs = _invoke_gmm(torch, _build_a4w4_inputs(torch))
    torch.npu.synchronize()

    assert len(outputs) == 2
    y_out, y_scale_out = outputs

    # y is the swiglu result quantized to int8: [M, N // 2].
    assert tuple(y_out.shape) == (_A4W4_M, _A4W4_N // 2)
    assert y_out.dtype == torch.int8

    # Per-token output scale: [M] float32.
    assert tuple(y_scale_out.shape) == (_A4W4_M,)
    assert y_scale_out.dtype == torch.float32

    assert all(tensor.device.type == "npu" for tensor in outputs)
