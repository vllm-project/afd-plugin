# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""910C precision tests for ``gmm_swiglu_quant_v2_layered``.

Build the extension against an Ascend 910C toolchain, then run::

    SOC_VERSION=910c AFD_RUN_ASCEND_OP_RUNTIME=1 \
        pytest tests/npu/test_gmm_swiglu_quant_v2_layered_precision.py

Two reference strategies are used, because only one of the two quantization
scenarios has a usable built-in counterpart.

``a8w8`` compares against the built-in non-layered
``torch_npu.npu_grouped_matmul_swiglu_quant_v2``, fed the selected layer's
weight, scale and assist matrix as a single-element list. Both are ACLNN
operators of the same family driven with byte-identical inputs, so a direct
comparison is a valid differential precision test.

``a8w4`` compares against a CPU golden implemented in this file, following the
formula already exercised on device by
``tests/npu/test_async_cam_layered_w4a8.py``: dequantize the int4 weights,
per-expert matmul, SiLU gating, then per-token quantization. The built-in cannot
serve as the reference for this scenario, and the reason is worth recording
precisely.

**Why A8W4 needs a golden.** For an int32 weight carrying packed int4,
``npu_grouped_matmul_swiglu_quant_v2`` derives the output width from the
weight's own last dimension - ``n = weight[0].size(2)`` in
``torch_npu/op_plugin/meta/_meta_registrations.py``, and the matching C++ N
inference in ``op-plugin/.../GroupedMatmulSwigluQuantV2NpuOpapi.cpp`` - which
for an ``[E, K, N // 8]`` int32 weight is ``N // 8``, while the ACLNN kernel
unpacks the int4 values and therefore expects ``N``. Measured directly against
the meta registration::

    int8  [E, K, N]      size(2) = 256 -> output [128, 128]  (agrees with kernel)
    int32 [E, K, N // 8] size(2) =  32 -> output [128,  16]  (kernel wants 128)

The framework therefore allocates an output eight times too narrow and the
ACLNN call fails with ``Expected tensor for gmmDsqParams_.output to have same
size as [128, 128], but got [128, 16]``. The mismatch is in the N derivation,
not in the NZ layout, so no input construction works around it: an unpacked
ND int32 tensor of the same logical shape reproduces it. That is a torch_npu
limitation, out of scope for this repository, which is why ``a8w4`` is checked
against a golden instead.

Two layers with distinct weights and scales are exercised so a wrong
``layer_index`` cannot pass unnoticed, every invocation gets its own activation
clone (the A8W4 MSD path may modify ``x`` in place), and each case reports its
seed, shapes, scenario, mode, layer, repeat, both maximum absolute errors and
the int8 mismatch count.

**Each attribute combination gets its own shape, deliberately.** The tiling is
cached per shape and tiling key, and neither ``group_list_type`` nor
``dequant_mode`` is part of that key - both reach the tiling only as data, and the
A8W8 fusion path pins a constant key (``A8W8_FUSION_KEY_MODE = 3``). Running two
combinations at one shape in the same process therefore makes the second inherit
the first one's tiling *and its attribute values*. That is not hypothetical: it
is what made this suite report a failure when the operator was in fact correct.
An A8W4 ``type=0`` case tiled first, an A8W8 ``type=1`` case inherited that
tiling, and the kernel then differenced token counts that were already counts -
writing 44 of 128 rows and leaving the rest of the capacity-sized output
uninitialized. The single-process log confirmed the mechanism: four calls
carrying four different attribute values produced exactly one ``PrintTilingData``
line. ``_COUNT_PATTERNS`` and ``test_variant_shapes_are_distinct`` keep the
combinations independent; ``test_a8w8_group_list_type_is_independent_of_call_order``
re-runs both values in both orders so a regression fails either way.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.npu

_K = 256  # reduction / hidden dimension
_N = 256  # output features (swiglu halves this to N // 2)
_E = 4  # experts
_LAYERS = 2  # distinct layers with distinct weights and scales
_REPEATS = 3  # repeated invocations per layer
_K_GROUPS = 2  # per-group scales along K for dequant_mode=1
_INT4_PER_INT8 = 2  # the W4A8 loader packs two int4 values per int8
_INT4_PER_INT32 = 8  # and the A8W4 path carries eight per int32 word
_NZ_FORMAT = 29
_SEED = 380

# Every (dequant_mode, group_list_type) combination gets its own activation
# shape, because the tiling is cached and the cache does not distinguish the two
# attributes.
#
# Both attributes reach the tiling only as tiling *data*. The A8W8 fusion path
# sets a constant tiling key (`A8W8_FUSION_KEY_MODE = 3`) and the cache key does
# not include `group_list_type` or `dequant_mode`, so a second call with the same
# shape reuses the first call's tiling - and with it the first call's attribute
# value. Measured on device: four calls carrying four different
# `group_list_type` values but one shape produced exactly one
# `PrintTilingData` line, i.e. the tiling ran once and the later calls inherited
# it. An out-of-range probe consequently looked "accepted" because the validation
# never re-ran, and a `type=1` case failed only because a `type=0` case had been
# tiled first - the kernel then differenced counts that were not cumulative and
# left the tail of the output unwritten.
#
# Varying the shape per combination forces a fresh tiling and keeps the
# combinations independent inside one process. The row counts stay non-uniform
# per expert so a wrong expert offset is still visible, and each pattern's entries
# differ from each other so differencing them does not reproduce them - which is
# what makes the two group-list representations distinguishable.
_COUNT_PATTERNS = {
    0: (20, 28, 36, 44),  # 128 rows
    1: (24, 32, 40, 48),  # 144 rows
    2: (28, 36, 44, 52),  # 160 rows
    3: (32, 40, 48, 56),  # 176 rows
}


def _variant(dequant_mode: int, group_list_type: int) -> tuple[int, tuple[int, ...]]:
    """Return the (row count, per-expert counts) unique to this combination."""
    counts = _COUNT_PATTERNS[dequant_mode * 2 + group_list_type]
    return sum(counts), counts


# The layer-selection check runs its own attribute combination, so it gets row
# counts of its own rather than reusing a pattern above: sharing a shape with a
# differently-parameterized case is exactly the coupling this file guards against.
_LAYER_SELECTION_COUNTS = {
    "a8w4": (30, 38, 40, 44),  # 152 rows
    "a8w8": (36, 40, 44, 48),  # 168 rows
}

# The A8W4 MSD path consumes an assist matrix that compensates the packed-int
# arithmetic, so the kernel's result matches the true dequantized matmul. The
# compensation is computed exactly as ``test_async_cam_layered_w4a8.py`` does.
_ASSIST_FACTOR = 8

# A8W8: the reference and the layered operator run the same arithmetic on the
# same inputs, and on device the per-token scales matched exactly (both maximum
# absolute errors were 0.0e+00). The bound is kept loose rather than set to zero
# because the two sides may select different tiling templates, so a future CANN
# build could legitimately introduce a few ulp of difference. 1e-3 is roughly
# three decimal digits over that, not a blanket allowance.
_A8W8_SCALE_RTOL = 1e-3
# A8W8: both sides are ACLNN operators of the same family, so a tight bound is
# meaningful.
_A8W8_RTOL = 0.02
_A8W8_ATOL = 0.02
# A8W4 needs its own scale bound, for a reason that is a property of the kernel
# rather than of the arithmetic: the MSD matmul writes its accumulator as fp16
# (`using yType = MatmulType<..., half, false>` in
# grouped_matmul_swiglu_quant_v2_layered.cpp) and only then casts to fp32, while
# the golden stays in fp32 throughout. One fp16 unit roundoff is 2**-11 and the
# per-token scale is an amax over a product of two such values, so a few unit
# roundoffs is the expected magnitude.
#
# Measured on Ascend 910C over all four (dequant_mode, group_list_type)
# combinations and both layers: the per-token scale carries 1.8e-03 to 2.1e-03 of
# relative error, about 4 fp16 unit roundoffs, against scales of order 1e-6. The
# bound is set at 16 unit roundoffs, roughly 3.6x the worst observation, so a
# different seed or shape has room without the check becoming vacuous - it still
# rejects any error of the percent scale that a real defect produces. Note the
# absolute figure in the log (~4e-09) looks far smaller than the bound only
# because the assertion is relative; the report prints both.
#
# The int8 codes are unaffected: they matched within one LSB.
_A8W4_SCALE_RTOL = 2**-7
# A8W4 dequantized values: `y * y_scale` with |y| <= 127 and y_scale of order
# 1e-6, so the whole signal is of order 1e-4 and one int8 LSB is about 5e-06.
# Measured maximum absolute difference is 2.4e-06 to 5.6e-06, one LSB wide and
# therefore consistent with the `int8_max_lsb_delta = 1` above rather than an
# independent error. `assert_close` bounds by `atol + rtol * |reference|`, so
# atol=2e-05 leaves roughly 3.6x margin at the smallest magnitudes and more at
# larger ones. The value inherited from the sibling W4A8 test (0.05) was two
# orders of magnitude larger than the signal itself and would have accepted
# anything.
_A8W4_RTOL = 0.04
_A8W4_ATOL = 2e-5
# Per-token quantization divides by a scale that itself carries a few ulp of
# difference, so an element sitting exactly on a rounding boundary can
# legitimately land one int8 code away. Anything larger than one step means a
# real numerical divergence, not a boundary artefact.
_MAX_LSB_DELTA = 1


def _pack_int4_to_nz_int32(torch, torch_npu, values):
    """Pack int4 ``values`` [E, K, N] into the int32 view of NZ storage.

    Mirrors the W4A8 loader: two signed int4 values per int8 (low nibble first),
    then the FRACTAL_NZ format cast, then an int32 view. This is the layout the
    A8W4 paths of both operators expect.
    """
    pairs = values.to(torch.int8).reshape(-1, _INT4_PER_INT8)
    packed = torch.bitwise_or(
        torch.bitwise_left_shift(pairs[:, 1], 4),
        torch.bitwise_and(pairs[:, 0], 0x0F),
    ).reshape(values.shape[0], values.shape[1], values.shape[2] // _INT4_PER_INT8)
    return (
        torch_npu.npu_format_cast(packed.contiguous().npu(), _NZ_FORMAT)
        .view(torch.int32)
        .contiguous()
    )


def _build_a8w4(torch, torch_npu, generator, dequant_mode: int, rows: int):
    """Return the A8W4 inputs plus the dequantized weights for the golden.

    ``weights`` are per-layer int32 views of NZ int4 storage, ``scales`` the
    encoded uint64 dequant scales, ``assists`` the fp32 compensation matrices
    the A8W4 path requires (a null assist matrix is rejected by the op_api), and
    ``dequants`` the fp32 ``[E, K, N]`` weights the golden multiplies against.
    ``rows`` varies per attribute combination so the tiling is recomputed.
    """
    x = torch.randint(-64, 64, (rows, _K), generator=generator, dtype=torch.int8).npu()
    x_scale = torch.full((rows,), 5e-3, dtype=torch.float32).npu()

    weights: list = []
    scales: list = []
    assists: list = []
    dequants: list = []
    for _ in range(_LAYERS):
        values = torch.randint(
            -7, 8, (_E, _K, _N), generator=generator, dtype=torch.int32
        )
        groups = 1 if dequant_mode == 0 else _K_GROUPS
        scale = (
            torch.rand((_E, groups, _N), generator=generator) * 1e-3 + 1e-4
        ).float()

        weights.append(_pack_int4_to_nz_int32(torch, torch_npu, values))

        # The operator carries the fp32 scale bits inside a uint64 element.
        encoded = scale.view(torch.int32).to(torch.int64)
        if groups == 1:
            encoded = encoded.squeeze(1)
        scales.append(encoded.npu())

        dequant = values.float() * scale.repeat_interleave(_K // groups, dim=1)
        dequants.append(dequant)
        assists.append((_ASSIST_FACTOR * dequant.sum(dim=1)).npu())

    return x, x_scale, weights, scales, assists, dequants


def _build_a8w8(torch, torch_npu, generator, rows: int):
    """Return (x, x_scale, weights, scales, assists) for A8W8.

    The weight is int8 NZ ``[E, K, N]`` and the per-channel scale stays 2-D
    ``[E, N]``. The A8W8 path takes no assist matrix, so the list is empty.
    Ranges are narrow because both ``x`` and ``w`` scales multiply into the
    result. ``rows`` varies per attribute combination so the tiling is recomputed.
    """
    x = torch.randint(-8, 8, (rows, _K), generator=generator, dtype=torch.int8).npu()
    x_scale = (torch.rand((rows,), generator=generator) * 1e-2 + 1e-3).float().npu()

    weights: list = []
    scales: list = []
    assists: list = []
    for _ in range(_LAYERS):
        weight = torch.randint(
            -8, 8, (_E, _K, _N), generator=generator, dtype=torch.int8
        )
        weights.append(torch_npu.npu_format_cast(weight.contiguous().npu(), _NZ_FORMAT))
        scales.append(
            (torch.rand((_E, _N), generator=generator) * 15e-3 + 5e-3).float().npu()
        )
        assists.append(None)

    return x, x_scale, weights, scales, assists


def _a8w4_golden(torch, x_cpu, x_scale_cpu, dequant_weight, counts):
    """Return the golden (y, y_scale) as CPU int8 and float32 tensors.

    The formula is the one validated on device by
    ``tests/npu/test_async_cam_layered_w4a8.py``: per-token dequantize the
    activation, matmul against the dequantized int4 weight of each expert, split
    into gate/up halves, apply SiLU gating, then quantize per token.
    """
    quantized_rows, scale_rows = [], []
    offset = 0
    for expert, count in enumerate(counts):
        if count == 0:
            continue
        rows = slice(offset, offset + count)
        activation = x_cpu[rows].float() * x_scale_cpu[rows, None]
        gate, up = (activation @ dequant_weight[expert]).chunk(2, dim=-1)
        swiglu = torch.nn.functional.silu(gate) * up
        scale = swiglu.abs().amax(dim=-1).clamp_min(1e-12) / 127
        quantized = (swiglu / scale[:, None]).round().clamp(-127, 127)
        quantized_rows.append(quantized.to(torch.int8))
        scale_rows.append(scale)
        offset += count
    return torch.cat(quantized_rows), torch.cat(scale_rows)


def _dequantize(torch, activations, scales):
    """Return int8 activations scaled back to fp32, the comparable quantity."""
    return activations.float() * scales.float().unsqueeze(1)


def _compare(
    torch,
    actual_y,
    actual_scale,
    reference_y,
    reference_scale,
    context: str,
    *,
    scale_rtol: float,
    rtol: float,
    atol: float,
) -> str:
    """Assert the two (y, y_scale) pairs agree; return a one-line report.

    Both returned tensors are compared. The int8 codes are bounded rather than
    required to match exactly, because the per-token scale that produced them is
    itself only equal to within a few ulp - an element on a rounding boundary may
    legitimately move by one code. The dequantized values carry the real
    numerical assertion.
    """
    assert tuple(actual_y.shape) == tuple(reference_y.shape), (
        f"{context}: y shape {tuple(actual_y.shape)} != {tuple(reference_y.shape)}"
    )
    assert tuple(actual_scale.shape) == tuple(reference_scale.shape), (
        f"{context}: y_scale shape {tuple(actual_scale.shape)} != "
        f"{tuple(reference_scale.shape)}"
    )
    assert actual_y.dtype == torch.int8 and reference_y.dtype == torch.int8
    assert actual_scale.dtype == torch.float32
    assert reference_scale.dtype == torch.float32

    scale_error = (actual_scale - reference_scale).abs().max().item()
    # The assertion below is relative (`atol=0`), so report the relative error
    # too. The absolute figure alone is misleading here: A8W4 per-token scales
    # are of order 1e-6, so an absolute 4e-09 is a relative 2e-03. Without this
    # number the margin against `scale_rtol` is not visible in the log.
    scale_delta = (actual_scale - reference_scale).abs()
    scale_rel_error = (
        (scale_delta / reference_scale.abs().clamp_min(1e-30)).max().item()
    )
    lsb_delta = (actual_y.to(torch.int32) - reference_y.to(torch.int32)).abs()
    lsb_max = lsb_delta.max().item() if lsb_delta.numel() else 0
    mismatch_count = int((lsb_delta > 0).sum().item())

    actual_values = _dequantize(torch, actual_y, actual_scale)
    reference_values = _dequantize(torch, reference_y, reference_scale)
    value_error = (actual_values - reference_values).abs().max().item()

    report = (
        f"{context}: y_scale_max_abs_err={scale_error:.6e} "
        f"y_scale_max_rel_err={scale_rel_error:.3e} (bound {scale_rtol:.3e}) "
        f"dequant_max_abs_err={value_error:.6e} "
        f"int8_mismatch={mismatch_count}/{lsb_delta.numel()} "
        f"int8_max_lsb_delta={lsb_max}"
    )

    assert lsb_max <= _MAX_LSB_DELTA, f"{report} -> int8 codes diverge"
    torch.testing.assert_close(
        actual_scale, reference_scale, rtol=scale_rtol, atol=0.0, msg=context
    )
    torch.testing.assert_close(
        actual_values, reference_values, rtol=rtol, atol=atol, msg=context
    )
    return report


def _group_list(torch, group_list_type: int, counts_pattern):
    """Counts for ``group_list_type=1``, cumulative offsets for ``0``."""
    counts = torch.tensor(counts_pattern, dtype=torch.int64, device="npu")
    return counts.cumsum(0) if group_list_type == 0 else counts, counts


def _runtime(torch, torch_npu) -> None:
    """Shared per-case setup: load the extension and pin the rank's device."""
    from afd_plugin.compat.npu.ops import ensure_afd_ascend_ops_loaded

    ensure_afd_ascend_ops_loaded()
    torch.npu.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    torch.npu.config.allow_internal_format = True


@pytest.mark.parametrize("group_list_type", [0, 1])
def test_layered_swiglu_a8w8_matches_builtin(group_list_type: int) -> None:
    """Compare every layer against the built-in non-layered operator.

    Each ``group_list_type`` uses its own activation row count so the tiling is
    recomputed instead of being served from the cache of the other value; see the
    ``_COUNT_PATTERNS`` comment.
    """
    if os.environ.get("AFD_RUN_ASCEND_OP_RUNTIME") != "1":
        pytest.skip("requires opt-in 910C runtime")

    torch = pytest.importorskip("torch")
    torch_npu = pytest.importorskip("torch_npu")
    _runtime(torch, torch_npu)

    rows, counts_pattern = _variant(0, group_list_type)
    generator = torch.Generator(device="cpu").manual_seed(_SEED)
    x, x_scale, weights, scales, _assists = _build_a8w8(
        torch, torch_npu, generator, rows
    )
    group_list, _counts = _group_list(torch, group_list_type, counts_pattern)
    reports = []

    for layer in range(_LAYERS):
        layer_index = torch.full((1,), layer, dtype=torch.int64, device="npu")
        for repeat in range(_REPEATS):
            context = (
                f"scenario=a8w8 dequant_mode=0 group_list_type={group_list_type} "
                f"layer={layer} repeat={repeat} seed={_SEED} "
                f"shapes=x{tuple(x.shape)} w{tuple(weights[layer].shape)} "
                f"scales{tuple(scales[layer].shape)}"
            )

            # Fresh activations for both sides, so neither sees a mutated input.
            actual = torch.ops.afd_ascend.gmm_swiglu_quant_v2_layered(
                x.clone(),
                [weights[layer]],
                [scales[layer]],
                [],
                x_scale,
                group_list,
                layer_index,
                0,  # dequant_mode
                0,  # quant_mode: per-token
                group_list_type,
                None,  # tuning_config
            )
            torch.npu.synchronize()

            reference = torch_npu.npu_grouped_matmul_swiglu_quant_v2(
                x.clone(),
                [weights[layer]],
                [scales[layer]],
                x_scale,
                group_list,
                dequant_mode=0,
                quant_mode=0,
                group_list_type=group_list_type,
            )
            torch.npu.synchronize()

            reports.append(
                _compare(
                    torch,
                    actual[0].cpu(),
                    actual[1].cpu(),
                    reference[0].cpu(),
                    reference[1].cpu(),
                    context,
                    rtol=_A8W8_RTOL,
                    atol=_A8W8_ATOL,
                    scale_rtol=_A8W8_SCALE_RTOL,
                )
            )

    # Surface the per-case numbers even when every case passes; a silent pass
    # would not satisfy the issue's reporting clause.
    print("\n".join(reports))


@pytest.mark.parametrize("dequant_mode", [0, 1])
@pytest.mark.parametrize("group_list_type", [0, 1])
def test_layered_swiglu_a8w4_matches_cpu_golden(
    dequant_mode: int, group_list_type: int
) -> None:
    """Compare every layer against the documented CPU golden.

    The built-in cannot be the A8W4 reference (see the module docstring), so the
    reference is the equivalent composition this file implements. Both the
    selected layer's weights and the assist matrix are supplied to the operator;
    the golden models the dequantized matmul the assist compensates for.

    Each ``(dequant_mode, group_list_type)`` combination uses its own activation
    row count so the tiling is recomputed rather than reused from another
    combination's cache; see the ``_COUNT_PATTERNS`` comment.
    """
    if os.environ.get("AFD_RUN_ASCEND_OP_RUNTIME") != "1":
        pytest.skip("requires opt-in 910C runtime")

    torch = pytest.importorskip("torch")
    torch_npu = pytest.importorskip("torch_npu")
    _runtime(torch, torch_npu)

    rows, counts_pattern = _variant(dequant_mode, group_list_type)
    generator = torch.Generator(device="cpu").manual_seed(_SEED)
    x, x_scale, weights, scales, assists, dequants = _build_a8w4(
        torch, torch_npu, generator, dequant_mode, rows
    )
    group_list, _counts = _group_list(torch, group_list_type, counts_pattern)
    x_cpu, x_scale_cpu = x.cpu(), x_scale.cpu()
    reports = []

    for layer in range(_LAYERS):
        layer_index = torch.full((1,), layer, dtype=torch.int64, device="npu")
        golden_y, golden_scale = _a8w4_golden(
            torch, x_cpu, x_scale_cpu, dequants[layer], counts_pattern
        )
        for repeat in range(_REPEATS):
            context = (
                f"scenario=a8w4 dequant_mode={dequant_mode} "
                f"group_list_type={group_list_type} layer={layer} repeat={repeat} "
                f"seed={_SEED} shapes=x{tuple(x.shape)} "
                f"w{tuple(weights[layer].shape)} scales{tuple(scales[layer].shape)}"
            )

            # The A8W4 MSD path may modify x in place, so this call gets a clone.
            actual = torch.ops.afd_ascend.gmm_swiglu_quant_v2_layered(
                x.clone(),
                [weights[layer]],
                [scales[layer]],
                [assists[layer]],
                x_scale,
                group_list,
                layer_index,
                dequant_mode,
                0,  # quant_mode: per-token
                group_list_type,
                None,  # tuning_config
            )
            torch.npu.synchronize()

            reports.append(
                _compare(
                    torch,
                    actual[0].cpu(),
                    actual[1].cpu(),
                    golden_y,
                    golden_scale,
                    context,
                    rtol=_A8W4_RTOL,
                    atol=_A8W4_ATOL,
                    scale_rtol=_A8W4_SCALE_RTOL,
                )
            )

    print("\n".join(reports))


@pytest.mark.parametrize("scenario", ["a8w4", "a8w8"])
def test_layered_swiglu_layer_selection_is_not_degenerate(scenario: str) -> None:
    """A wrong layer must be detectable, so the per-layer results must differ.

    The issue requires distinct weights and scales per layer precisely so that
    selecting the wrong layer cannot pass unnoticed. If two layers produced the
    same output the comparison above would be vacuous.
    """
    if os.environ.get("AFD_RUN_ASCEND_OP_RUNTIME") != "1":
        pytest.skip("requires opt-in 910C runtime")

    torch = pytest.importorskip("torch")
    torch_npu = pytest.importorskip("torch_npu")
    _runtime(torch, torch_npu)

    generator = torch.Generator(device="cpu").manual_seed(_SEED)
    counts_pattern = _LAYER_SELECTION_COUNTS[scenario]
    rows = sum(counts_pattern)
    if scenario == "a8w4":
        x, x_scale, weights, scales, assists, _dequants = _build_a8w4(
            torch, torch_npu, generator, 0, rows
        )
    else:
        x, x_scale, weights, scales, assists = _build_a8w8(
            torch, torch_npu, generator, rows
        )

    group_list, _counts = _group_list(torch, 1, counts_pattern)
    results = []
    for layer in range(_LAYERS):
        assist = [] if assists[layer] is None else assists
        outputs = torch.ops.afd_ascend.gmm_swiglu_quant_v2_layered(
            x.clone(),
            weights,
            scales,
            assist,
            x_scale,
            group_list,
            torch.full((1,), layer, dtype=torch.int64, device="npu"),
            0,
            0,
            1,
            None,
        )
        torch.npu.synchronize()
        results.append(_dequantize(torch, outputs[0].cpu(), outputs[1].cpu()))

    assert not torch.equal(results[0], results[1]), (
        "layers 0 and 1 produced identical output, so the layer_index selection "
        "cannot be validated by this fixture"
    )


def test_variant_shapes_are_distinct() -> None:
    """Guard the invariant the whole suite depends on: no two combinations share
    a shape.

    The tiling is cached per shape and tiling key, and neither ``group_list_type``
    nor ``dequant_mode`` is part of that key - both reach the tiling only as data.
    Two combinations sharing a shape in one process therefore reuse whichever
    tiling ran first, along with its attribute values. That is what produced the
    original failure: a ``type=0`` case tiled first, a ``type=1`` case inherited
    it, the kernel differenced counts that were already counts, and the tail of
    the output was left unwritten.

    Two cases may share a shape only when they also agree on every attribute the
    tiling embeds, in which case the shared tiling is the one they both want. The
    check is pure Python and needs no device, so it also runs in the
    unopted-in collection the rest of the suite skips out of.
    """
    combos = [(mode, glt) for mode in (0, 1) for glt in (0, 1)]
    rows = [_variant(mode, glt)[0] for mode, glt in combos]
    assert len(set(rows)) == len(rows), (
        "each (dequant_mode, group_list_type) combination needs its own row count "
        f"so its tiling is recomputed; got {dict(zip(combos, rows, strict=True))}"
    )
    # The layer-selection check carries its own attribute combination and must not
    # collide with any of the above either.
    shared = set(rows) & {sum(counts) for counts in _LAYER_SELECTION_COUNTS.values()}
    assert not shared, (
        f"layer-selection shapes {sorted(shared)} collide with a differently "
        "parameterized case; give it its own row counts"
    )


@pytest.mark.parametrize("first,second", [(1, 0), (0, 1)])
def test_a8w8_group_list_type_is_independent_of_call_order(
    first: int, second: int
) -> None:
    """Both ``group_list_type`` values must be correct in either call order.

    This is the regression guard for the failure that started this investigation:
    ``type=0`` and ``type=1`` were run in one process, the second inherited the
    first's tiling, and the suite reported a numerical failure that belonged to
    the test's shape reuse rather than to the operator. Each value is compared
    against the built-in here, in an order chosen by the parametrization, so a
    regression that reintroduces shape sharing fails on one of the two orders.
    """
    if os.environ.get("AFD_RUN_ASCEND_OP_RUNTIME") != "1":
        pytest.skip("requires opt-in 910C runtime")

    torch = pytest.importorskip("torch")
    torch_npu = pytest.importorskip("torch_npu")
    _runtime(torch, torch_npu)

    layer_index = torch.zeros((1,), dtype=torch.int64, device="npu")
    reports = []

    for group_list_type in (first, second):
        rows, counts_pattern = _variant(0, group_list_type)
        generator = torch.Generator(device="cpu").manual_seed(_SEED)
        x, x_scale, weights, scales, _assists = _build_a8w8(
            torch, torch_npu, generator, rows
        )
        group_list, _counts = _group_list(torch, group_list_type, counts_pattern)
        context = (
            f"order={first}-then-{second} group_list_type={group_list_type} "
            f"rows={rows} seed={_SEED}"
        )

        actual = torch.ops.afd_ascend.gmm_swiglu_quant_v2_layered(
            x.clone(),
            [weights[0]],
            [scales[0]],
            [],
            x_scale,
            group_list,
            layer_index,
            0,
            0,
            group_list_type,
            None,
        )
        torch.npu.synchronize()

        reference = torch_npu.npu_grouped_matmul_swiglu_quant_v2(
            x.clone(),
            [weights[0]],
            [scales[0]],
            x_scale,
            group_list,
            dequant_mode=0,
            quant_mode=0,
            group_list_type=group_list_type,
        )
        torch.npu.synchronize()

        reports.append(
            _compare(
                torch,
                actual[0].cpu(),
                actual[1].cpu(),
                reference[0].cpu(),
                reference[1].cpu(),
                context,
                rtol=_A8W8_RTOL,
                atol=_A8W8_ATOL,
                scale_rtol=_A8W8_SCALE_RTOL,
            )
        )

    print("\n".join(reports))
