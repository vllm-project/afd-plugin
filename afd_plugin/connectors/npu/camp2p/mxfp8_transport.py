# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CAMP2P MXFP8 transport: capability checks, quantization and A2E packet handling.

The connector keeps transport-mode selection, lifecycle, the cross-rank handshake
and microbatch group selection. This module owns what the E4M3/E8M0 packet needs,
so the default BF16 path stays small. Quantization still runs inside the opaque
send op, which keeps the Graph capture boundary unchanged, and BF16 keeps the
original twelve-argument call.
"""

from __future__ import annotations

from typing import Final

import torch

from afd_plugin.compat.npu.ops import a2e_accepts_mxfp8_arguments
from afd_plugin.config import AFDConfig

A2E_QUANT_MODE_MXFP8: Final[int] = 4

# MX block size the A2E contract is defined against, and the smallest AIV count the
# quantized transport was verified with.
A2E_MX_GROUP_SIZE: Final[int] = 32
A2E_MX_MIN_AIV_NUM: Final[int] = 3
# Each side of the A2E pair must fit the operator's rank addressing.
A2E_MX_MAX_RANKS_PER_SIDE: Final[int] = 32

#: Activation dtype carried end to end in MXFP8 mode.
A2E_MX_ACTIVATION_DTYPE: Final[torch.dtype] = torch.float8_e4m3fn
#: Scale dtype the A2E operator validates. This is the PyTorch dtype, not the
#: torch_npu ScalarType enum, because the operator schema is written in dtypes.
A2E_MX_SCALE_DTYPE: Final[torch.dtype] = torch.float8_e8m0fnu
#: Return values the MXFP8 contract needs: the BF16 four past the activation plus
#: the received E8M0 scale.
A2E_MX_RETURN_VALUES: Final[int] = 6


def _require_a5_device() -> None:
    """Refuse the MXFP8 transport on a device whose operator lacks the path.

    The operator compiles its MXFP8 kernel for A5 only (``AFD_TARGET_A5`` in its
    binding) while registering the same fourteen-argument schema on every build,
    so the schema probe alone would accept a non-A5 device and the failure would
    surface at the first transfer instead of at construction.
    """
    from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

    device_type = get_ascend_device_type()
    if device_type != AscendDeviceType.A5:
        raise RuntimeError(
            "quant_mode=4 requires an Ascend A5 device; the a2e MXFP8 path is "
            f"built for A5 only and this one is {device_type.name}",
        )


def require_supported(
    afd_config: AFDConfig,
    *,
    attn_size: int,
    ffn_size: int,
    aiv_num: int,
) -> None:
    """Refuse ``quant_mode=4`` when this deployment cannot carry it.

    The checks are the ones the transport depends on, and they run before any
    communication starts so a bad configuration cannot fail halfway through a
    request. ``quant_mode=0`` keeps the original twelve-argument call and gains no
    new requirement.
    """

    if not afd_config.compute_gate_on_attention:
        raise RuntimeError(
            "quant_mode=4 carries a pre-quantized Attention activation, so it "
            "needs the common compute_gate_on_attention=true on both roles",
        )
    if attn_size != ffn_size:
        raise RuntimeError(
            "quant_mode=4 requires equal Attention and FFN rank counts; got "
            f"A={attn_size}, F={ffn_size}",
        )
    for side, size in (("Attention", attn_size), ("FFN", ffn_size)):
        if not 1 <= size <= A2E_MX_MAX_RANKS_PER_SIDE:
            raise RuntimeError(
                "quant_mode=4 requires 1.."
                f"{A2E_MX_MAX_RANKS_PER_SIDE} {side} ranks; got {size}",
            )
    if aiv_num < A2E_MX_MIN_AIV_NUM:
        raise RuntimeError(
            "quant_mode=4 requires at least "
            f"{A2E_MX_MIN_AIV_NUM} AIV cores; got {aiv_num}",
        )
    _require_a5_device()
    supported = a2e_accepts_mxfp8_arguments()
    if supported is False:
        raise RuntimeError(
            "quant_mode=4 requires an a2e operator that takes x_scale and "
            "quant_mode; this build does not",
        )
    if supported is None:
        raise RuntimeError(
            "quant_mode=4 cannot confirm that this a2e build takes x_scale and "
            "quant_mode, so the quantized transport stays disabled",
        )


def _check_a2e_mx_transport_pair(
    quantized: torch.Tensor,
    scale: torch.Tensor,
    hidden_size: int,
) -> None:
    """Name which part of the A2E ``x_scale`` contract was broken.

    The operator answers six conditions with one opaque failure, which says
    nothing about what actually arrived. Checking both directions here reports
    the observed shapes and dtypes instead.
    """
    groups = (hidden_size + A2E_MX_GROUP_SIZE - 1) // A2E_MX_GROUP_SIZE
    problems: list[str] = []
    if quantized.dtype != A2E_MX_ACTIVATION_DTYPE:
        problems.append(
            f"activation dtype {quantized.dtype} != {A2E_MX_ACTIVATION_DTYPE}",
        )
    if scale.dtype != A2E_MX_SCALE_DTYPE:
        problems.append(f"scale dtype {scale.dtype} != {A2E_MX_SCALE_DTYPE}")
    if scale.dim() != 2:
        problems.append(f"scale dim {scale.dim()} != 2")
    if not scale.is_contiguous():
        problems.append("scale is not contiguous")
    if scale.device != quantized.device:
        problems.append(
            f"scale device {scale.device} != activation device {quantized.device}",
        )
    if scale.shape[0] != quantized.shape[0]:
        problems.append(
            f"scale rows {scale.shape[0]} != activation rows {quantized.shape[0]}",
        )
    if scale.dim() == 2 and scale.shape[1] != groups:
        problems.append(
            f"scale cols {scale.shape[1]} != "
            f"ceil({hidden_size}/{A2E_MX_GROUP_SIZE})={groups}",
        )
    if problems:
        raise RuntimeError(
            "A2E MXFP8 transport pair violates its contract: "
            f"{'; '.join(problems)}"
            f" | activation {tuple(quantized.shape)} {quantized.dtype}"
            f" | scale {tuple(scale.shape)} {scale.dtype}",
        )


def quantize_activation(
    hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize an Attention activation into the A2E MXFP8 transport pair.

    Returns the E4M3 activation and its E8M0 MX scale, the scale shaped
    ``(tokens, ceil(hidden_size / A2E_MX_GROUP_SIZE))``. The FFN side hands both
    to the experts without dequantizing, so the row count of the two must agree.
    """
    import torch_npu
    from vllm_ascend.quantization.utils import get_dynamic_mx_quant_scale_alg

    quantized, scale = torch_npu.npu_dynamic_mx_quant(
        hidden_states,
        dst_type=A2E_MX_ACTIVATION_DTYPE,
        scale_alg=get_dynamic_mx_quant_scale_alg(),
    )
    scale = scale.flatten(-2)
    if scale.dtype != A2E_MX_SCALE_DTYPE:
        scale = scale.view(A2E_MX_SCALE_DTYPE)
    _check_a2e_mx_transport_pair(quantized, scale, int(hidden_states.shape[-1]))
    return quantized, scale


def send(
    activation: torch.Tensor,
    expert_ids: torch.Tensor | None,
    expert_scales: torch.Tensor | None,
    transfer_state,
    *,
    ffn_size: int,
    attn_size: int,
    world_rank: int,
    group_ep,
    compute_gate: int,
) -> tuple:
    """Quantize and send the wire activation on the group the connector selected."""
    quantized, scale = quantize_activation(activation)
    return torch.ops.afd_ascend.a2e(
        quantized,
        expert_ids,
        expert_scales,
        transfer_state.batch_size,
        transfer_state.h,
        transfer_state.k,
        ffn_size,
        attn_size,
        world_rank,
        group_ep,
        transfer_state.aiv_num,
        compute_gate,
        scale,
        A2E_QUANT_MODE_MXFP8,
    )


def receive_packet(
    transfer_state,
    *,
    ffn_size: int,
    attn_size: int,
    world_rank: int,
    group_ep,
    compute_gate_mode: int,
) -> tuple:
    """Run the FFN-side A2E call and return its raw outputs.

    The FFN rank receives both payloads from its paired Attention rank, so it
    supplies an empty FP8 activation placeholder and omits ``x_scale``: the
    operator returns the received activation in ``outputs[0]`` and the received
    E8M0 MX scale in ``outputs[5]``.
    """
    return torch.ops.afd_ascend.a2e(
        torch.empty(
            (0, transfer_state.h),
            dtype=A2E_MX_ACTIVATION_DTYPE,
            device="npu",
        ),
        torch.tensor([], dtype=torch.int32, device="npu"),
        torch.tensor([], dtype=torch.float32, device="npu"),
        transfer_state.batch_size,
        transfer_state.h,
        transfer_state.k,
        ffn_size,
        attn_size,
        world_rank,
        group_ep,
        transfer_state.aiv_num,
        compute_gate_mode,
        None,
        A2E_QUANT_MODE_MXFP8,
    )


def receive(outputs: tuple, hidden_size: int) -> torch.Tensor:
    """Read and validate the received E8M0 scale from the operator outputs."""
    if len(outputs) < A2E_MX_RETURN_VALUES:
        raise RuntimeError(
            "CAMP2P MXFP8 transport expects "
            f"{A2E_MX_RETURN_VALUES} operator return values including the "
            f"received scale; this a2e build returned {len(outputs)}. The "
            "operator does not implement the MXFP8 path",
        )
    scale = outputs[5]
    _check_a2e_mx_transport_pair(outputs[0], scale, hidden_size)
    return scale
