# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""AFD-owned MoE execution for the external-routing FFN path.

Attention computes the gate when ``compute_gate_on_attention`` is on, so the FFN
side receives routing for its own input rows instead of computing it. Upstream
prepares the routing inside the prepare stage because upstream owns the gate;
here the routing arrives as an explicit input, is aligned by the same padding,
gathering and sharding rules as the activation, and comes back as an explicit
result.

The AFD prepare/finalize subclasses are selected through AFD comm implementations
that only override the factory method, so ordinary execution keeps the upstream
classes. The MoE component owns the per-token state that expert execution reads
back, and clears it on both success and failure.

MC2 gets a second seam for the same reason: when the transport carried a
quantized activation, the scale it carried has to reach the dispatch operator
instead of being replaced by a scale computed here. The dispatcher subclass
changes only the two kwargs that describe the activation.
"""

from __future__ import annotations

from typing import Any, Final

import torch
import torch.nn as nn
from vllm.distributed.parallel_state import get_pcp_group
from vllm_ascend.ops.fused_moe.dataclass.prepare_finalize import MoEPrepareOutput
from vllm_ascend.ops.fused_moe.dataclass.token_dispatcher import MoETokenDispatchInput
from vllm_ascend.ops.fused_moe.moe_comm_method import (
    AllGatherCommImpl,
    MC2CommImpl,
)
from vllm_ascend.ops.fused_moe.prepare_finalize import (
    PrepareAndFinalizeWithAllGather,
    PrepareAndFinalizeWithMC2,
)
from vllm_ascend.ops.fused_moe.token_dispatcher import TokenDispatcherWithMC2
from vllm_ascend.quantization.quant_type import QuantType

from afd_plugin.connectors.npu.camp2p.external_routing import (
    AFDExternalRoutingInputs,
)

#: Quant mode that tells the MC2 dispatch operator the activation is already
#: quantized and its scale arrives with it. The device smoke established this
#: number for the pre-quantized path; every other path leaves the mode to upstream.
_MC2_PREQUANT_QUANT_MODE: Final[int] = 0


class AFDPrepareAndFinalizeWithAllGather(PrepareAndFinalizeWithAllGather):
    """ALLGATHER prepare that accepts routing from the Attention side."""

    def prepare_external(
        self,
        hidden_states: torch.Tensor,
        routing: AFDExternalRoutingInputs,
        *,
        router_logits: torch.Tensor | None = None,
        replace_allreduce: bool = False,
        quant_type: QuantType = QuantType.NONE,
    ) -> tuple[MoEPrepareOutput, AFDExternalRoutingInputs]:
        """Pad and gather the routing exactly like the activation.

        The native ``prepare`` stays untouched for callers that own the gate; this
        entry returns the aligned routing instead of publishing it on ``self``.
        """
        from vllm_ascend.ascend_forward_context import _EXTRA_CTX

        external = routing
        if external.token_count != hidden_states.shape[0]:
            raise ValueError(
                "external routing covers a different token count than the activation: "
                f"{external.token_count} routing rows for "
                f"{hidden_states.shape[0]} activation rows",
            )

        if self.moe_config.dp_size > 1:
            max_tokens_across_dp = _EXTRA_CTX.max_tokens_across_dp

            self.num_tokens = hidden_states.shape[0]
            pad_size = max_tokens_across_dp - self.num_tokens
            if pad_size > 0:
                hidden_states = nn.functional.pad(hidden_states, (0, 0, 0, pad_size))
                if router_logits is not None:
                    router_logits = nn.functional.pad(
                        router_logits, (0, 0, 0, pad_size)
                    )
                external = external.padded(pad_size)

            hidden_states = self.moe_config.dp_group.all_gather(hidden_states, 0)
            if router_logits is not None:
                router_logits = self.moe_config.dp_group.all_gather(router_logits, 0)
            external = external.gathered(
                lambda tensor: self.moe_config.dp_group.all_gather(tensor, 0),
            )

        if self.moe_config.pcp_size > 1:
            max_tokens_across_pcp = _EXTRA_CTX.max_tokens_across_pcp

            self.num_tokens_pcp = hidden_states.shape[0]
            pad_size = max_tokens_across_pcp - self.num_tokens_pcp
            if pad_size > 0:
                hidden_states = nn.functional.pad(hidden_states, (0, 0, 0, pad_size))
                if router_logits is not None:
                    router_logits = nn.functional.pad(
                        router_logits, (0, 0, 0, pad_size)
                    )
                external = external.padded(pad_size)

            hidden_states = get_pcp_group().all_gather(hidden_states, dim=0)
            if router_logits is not None:
                router_logits = get_pcp_group().all_gather(router_logits, dim=0)
            external = external.gathered(
                lambda tensor: get_pcp_group().all_gather(tensor, dim=0),
            )

        return (
            MoEPrepareOutput(
                hidden_states=hidden_states,
                router_logits=router_logits,
                mc2_mask=None,
                padded_hidden_states_shape=None,
                pertoken_scale=None,
            ),
            external,
        )


class AFDPrepareAndFinalizeWithMC2(PrepareAndFinalizeWithMC2):
    """MC2 prepare that accepts routing from the Attention side."""

    def prepare_external(
        self,
        hidden_states: torch.Tensor,
        routing: AFDExternalRoutingInputs,
        *,
        router_logits: torch.Tensor | None = None,
        replace_allreduce: bool = False,
        quant_type: QuantType = QuantType.NONE,
    ) -> tuple[MoEPrepareOutput, AFDExternalRoutingInputs]:
        """Pad and TP-shard the routing exactly like the activation."""
        from vllm_ascend.ascend_forward_context import _EXTRA_CTX
        from vllm_ascend.ops.fused_moe.moe_utils import _pad_tokens_with_cat

        external = routing
        if external.token_count != hidden_states.shape[0]:
            raise ValueError(
                "external routing covers a different token count than the activation: "
                f"{external.token_count} routing rows for "
                f"{hidden_states.shape[0]} activation rows",
            )

        self.replace_allreduce = replace_allreduce
        mc2_mask = _EXTRA_CTX.mc2_mask
        if self.replace_allreduce:
            self.num_tokens = hidden_states.shape[0]
            start = self.tp_rank * self.num_tokens
            mc2_mask = mc2_mask[start : start + self.num_tokens]
            pad_size = _EXTRA_CTX.padded_num_tokens // self.tp_size - self.num_tokens
            if pad_size > 0:
                hidden_states = nn.functional.pad(hidden_states, (0, 0, 0, pad_size))
                if router_logits is not None:
                    router_logits = nn.functional.pad(
                        router_logits, (0, 0, 0, pad_size)
                    )
                external = external.padded(pad_size)
                mc2_mask = nn.functional.pad(mc2_mask, (0, pad_size), value=False)
        elif self.tp_size > 1:
            split_mc2_mask = torch.tensor_split(mc2_mask, self.tp_size, dim=0)
            mc2_mask = split_mc2_mask[self.tp_rank]

        padded_hidden_states_shape = hidden_states.shape
        if not self.replace_allreduce:
            self.num_tokens, _ = hidden_states.shape
            target_pad_length = _EXTRA_CTX.padded_num_tokens
            pad_size = target_pad_length - self.num_tokens

            if pad_size > 0:
                hidden_states = _pad_tokens_with_cat(hidden_states, target_pad_length)
                if router_logits is not None:
                    router_logits = _pad_tokens_with_cat(
                        router_logits, target_pad_length
                    )
                external = external.padded(pad_size)
                padded_hidden_states_shape = hidden_states.shape

            if self.tp_size > 1:
                split_hidden_states = torch.tensor_split(
                    hidden_states, self.tp_size, dim=0
                )
                if router_logits is not None:
                    split_router_logits = torch.tensor_split(
                        router_logits, self.tp_size, dim=0
                    )
                    router_logits = split_router_logits[self.tp_rank]
                hidden_states = split_hidden_states[self.tp_rank]
                external = external.split_rows(
                    lambda tensor: torch.tensor_split(tensor, self.tp_size, dim=0)[
                        self.tp_rank
                    ],
                )

        return (
            MoEPrepareOutput(
                hidden_states=hidden_states,
                router_logits=router_logits,
                mc2_mask=mc2_mask,
                padded_hidden_states_shape=padded_hidden_states_shape,
                pertoken_scale=None,
            ),
            external,
        )


class AFDAllGatherCommImpl(AllGatherCommImpl):
    """ALLGATHER comm that prepares external routing instead of computing it."""

    def _get_prepare_finalize(self) -> AFDPrepareAndFinalizeWithAllGather:
        return AFDPrepareAndFinalizeWithAllGather(self.moe_config)


def _require_prequant_dispatch_contract(
    token_dispatch_input: MoETokenDispatchInput,
    scale: torch.Tensor,
) -> None:
    """Reject a scale that does not describe the activation being dispatched.

    Only the combination the device smoke established is accepted. A scale with a
    different shape or dtype would otherwise reach the operator and surface as a
    shape error inside it, or as silently wrong values.
    """
    problems: list[str] = []
    if not token_dispatch_input.quant.is_mxfp:
        problems.append(
            f"got is_mxfp={token_dispatch_input.quant.is_mxfp}",
        )
    if scale.dim() != 2:
        problems.append(f"got scale dim {scale.dim()}")
    if scale.dtype is not torch.float8_e8m0fnu:
        problems.append(f"got scale dtype {scale.dtype}")
    if token_dispatch_input.hidden_states.dtype is not torch.float8_e4m3fn:
        problems.append(
            f"got activation dtype {token_dispatch_input.hidden_states.dtype}",
        )
    if problems:
        raise RuntimeError(
            "AFD MC2 pre-quantized dispatch expects an MXFP quant feature, an E4M3 "
            "activation and a two-dimensional E8M0 scale; " + "; ".join(problems),
        )


class AFDTokenDispatcherWithMC2(TokenDispatcherWithMC2):
    """MC2 dispatcher that accepts an already-quantized activation.

    When the Attention side quantized the activation it also sent its E8M0 MX
    scale, which prepare publishes as the routing's ``pertoken_scale``.
    Upstream's dispatch kwargs hardcode ``scales=None`` and resolve the quant mode
    as if this rank still had to quantize, so the transported scale would be
    dropped and the operator would quantize a tensor that already is quantized.

    Only a routing that carries a scale changes the two kwargs that describe the
    activation; upstream builds every other field.
    """

    def get_dispatch_mc2_kwargs(
        self,
        token_dispatch_input: MoETokenDispatchInput,
    ) -> dict[str, Any]:
        kwargs = super().get_dispatch_mc2_kwargs(token_dispatch_input)
        scale = token_dispatch_input.routing.pertoken_scale
        if scale is None:
            return kwargs
        # An explicit communication quant mode that disagrees is a configuration
        # conflict, not something to overrule silently: the caller asked for one
        # quantisation scheme while the routing carries another.
        comm_quant_mode = token_dispatch_input.quant.comm_quant_mode
        if (
            comm_quant_mode is not None
            and int(comm_quant_mode) != _MC2_PREQUANT_QUANT_MODE
        ):
            raise RuntimeError(
                "AFD MC2 pre-quantized dispatch: comm_quant_mode="
                f"{comm_quant_mode} conflicts with a pre-quantized activation, "
                f"which requires quant_mode={_MC2_PREQUANT_QUANT_MODE}",
            )
        _require_prequant_dispatch_contract(token_dispatch_input, scale)
        kwargs["scales"] = scale
        kwargs["quant_mode"] = _MC2_PREQUANT_QUANT_MODE
        return kwargs


class AFDMC2CommImpl(MC2CommImpl):
    """MC2 comm that prepares external routing instead of computing it."""

    def _get_prepare_finalize(self) -> AFDPrepareAndFinalizeWithMC2:
        return AFDPrepareAndFinalizeWithMC2(self.moe_config)

    def _get_token_dispatcher(self) -> AFDTokenDispatcherWithMC2:
        return AFDTokenDispatcherWithMC2()


# Native comm types the external-routing FFN path supports, and the AFD
# counterparts that own the external-routing prepare/finalize.
_AFD_COMM_BY_NATIVE: dict[type, type] = {
    AllGatherCommImpl: AFDAllGatherCommImpl,
    MC2CommImpl: AFDMC2CommImpl,
}
_NATIVE_COMM_BY_AFD: dict[type, type] = {
    afd: native for native, afd in _AFD_COMM_BY_NATIVE.items()
}


def _afd_types_by_comm_type() -> dict[object, type]:
    """Map the runtime comm type to the AFD class that owns its prepare stage."""
    from vllm_ascend.ascend_forward_context import MoECommType

    return {
        MoECommType.ALLGATHER: AFDAllGatherCommImpl,
        MoECommType.MC2: AFDMC2CommImpl,
    }


def _prebuilt_comms(moe_config) -> dict[type, object]:
    """Build the AFD comm instances the native selector can pick from.

    The native selector is keyed by ``MoECommType`` and returns ``None`` when this
    configuration was not initialised with that method, so a missing candidate means
    the type is simply absent here. Construction happens in the caller's config
    context, which is why it runs at component creation and not per forward.
    """
    from vllm_ascend.ascend_forward_context import MoECommType
    from vllm_ascend.ops.fused_moe.moe_comm_method import get_moe_comm_method

    candidates: dict[type, object] = {}
    for comm_type, afd_type in (
        (MoECommType.ALLGATHER, AFDAllGatherCommImpl),
        (MoECommType.MC2, AFDMC2CommImpl),
    ):
        native = get_moe_comm_method(comm_type, moe_config)
        if native is None:
            continue
        candidates[afd_type] = afd_type(native.moe_config)
    return candidates


def _require_activation_scale_pair(
    hidden_states: torch.Tensor,
    activation_scale: torch.Tensor | None,
) -> None:
    """Refuse anything but a matching E4M3 activation and E8M0 scale pair.

    The activation and its scale are a pair or nothing. A quantized activation
    without its scale cannot be interpreted, and a scale next to a model-dtype
    activation would make the MoE skip quantizing a tensor that still needs it.

    Only the target format is accepted. E5M2 would take the same route while the
    weight path is defined for E4M3.
    """
    if activation_scale is None:
        if hidden_states.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            raise RuntimeError(
                "DSV4 external routing received a quantized activation without its "
                f"MX scale; got activation={hidden_states.dtype}",
            )
        return
    if hidden_states.dtype is not torch.float8_e4m3fn:
        raise RuntimeError(
            "DSV4 external routing expects an E4M3 activation for the MXFP8 "
            f"pass-through; got {hidden_states.dtype}",
        )
    if activation_scale.shape[0] != hidden_states.shape[0]:
        raise RuntimeError(
            "DSV4 external routing needs one scale row per activation row; got "
            f"{activation_scale.shape[0]} scale rows for "
            f"{hidden_states.shape[0]} activation rows",
        )


class AFDExternalRoutingMoE:
    """Run the routed experts on routing supplied by the Attention side.

    The AFD comm instances are built while the component is constructed, in a config
    context. A forward only looks up the instance matching the native selection; it
    never builds a missing one.
    """

    def __init__(self, routed_experts):
        self.routed_experts = routed_experts
        self._candidates = _prebuilt_comms(routed_experts.moe_config)
        if not self._candidates:
            raise RuntimeError(
                "DSV4 external routing found no AFD comm candidate for this "
                "configuration; the ALLGATHER and MC2 paths are the supported ones",
            )

    def _comm_for(self, comm_type, native_comm):
        if native_comm is None:
            # The MRV2 hook resolves the method without a config, which misses the
            # (type, expert-count) registry key and leaves the instance unset. The
            # comm type is always present, so fall back to it.
            afd_type = _afd_types_by_comm_type().get(comm_type)
            afd_comm = self._candidates.get(afd_type) if afd_type is not None else None
            if afd_comm is not None:
                return afd_comm
            raise RuntimeError(
                "DSV4 external routing could not resolve the MoE comm method for "
                f"{comm_type} with this layer's config; no AFD candidate was built",
            )
        for afd_type, afd_comm in self._candidates.items():
            if isinstance(native_comm, _NATIVE_COMM_BY_AFD[afd_type]):
                return afd_comm
        raise RuntimeError(
            "DSV4 external routing is wired for the ALLGATHER and MC2 MoE paths; "
            f"got {type(native_comm).__name__}. The all-to-all and fused-MC2 paths "
            "shard differently and are not supported.",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        activation_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from vllm_ascend.ascend_forward_context import _EXTRA_CTX
        from vllm_ascend.ops.fused_moe.moe_utils import (
            maybe_normalize_mxfp_scale_layout,
        )

        _require_activation_scale_pair(hidden_states, activation_scale)

        experts = self.routed_experts
        previous_comm = _EXTRA_CTX.moe_comm_method
        comm = self._comm_for(_EXTRA_CTX.moe_comm_type, previous_comm)
        prepare_finalize = comm.prepare_finalize

        try:
            # Scope switch, not a registry change: expert execution reads the comm
            # from this context attribute, so both sides use the same object.
            _EXTRA_CTX.moe_comm_method = comm
            prepare_output, aligned = prepare_finalize.prepare_external(
                hidden_states,
                AFDExternalRoutingInputs(
                    topk_ids=topk_ids,
                    topk_weights=topk_weights,
                    scale=activation_scale,
                ),
                router_logits=None,
                replace_allreduce=experts.moe_config.is_sequence_parallel,
                quant_type=experts.quant_type,
            )
            # The scale comes back on the aligned routing rather than on the prepare
            # output: prepare does not quantize, so ``pertoken_scale`` stays None
            # while the transported scale follows the activation through padding,
            # gathering and sharding.
            #
            # What consumes it fixes the layout, and the paths do not agree:
            #
            #   ALLGATHER  the fused GMM wants ``(rows, cols // 2, 2)``, which is
            #              what the quant method would have produced had it
            #              quantized here.
            #   MC2        the dispatcher's pre-quantized contract takes the scale
            #              two-dimensionally, so normalising here would hand it the
            #              wrong shape.
            #   BF16       no scale at all.
            #
            # The condition names ALLGATHER rather than testing for "not MC2": a
            # path added later should have to state its layout instead of
            # inheriting one.
            pertoken_scale = aligned.scale
            if pertoken_scale is not None and isinstance(comm, AllGatherCommImpl):
                pertoken_scale = maybe_normalize_mxfp_scale_layout(pertoken_scale)
            experts.ascend_pertoken_scale = pertoken_scale
            experts.ascend_mc2_mask = prepare_output.mc2_mask
            try:
                fused = experts.quant_method.apply(
                    layer=experts,
                    x=prepare_output.hidden_states,
                    topk_weights=aligned.topk_weights,
                    topk_ids=aligned.topk_ids,
                    shared_experts=None,
                    shared_experts_input=None,
                )
            finally:
                experts.ascend_pertoken_scale = None
                experts.ascend_mc2_mask = None
            return comm.finalize(
                hidden_states=fused.routed_out,
                reduce_results=isinstance(comm, AllGatherCommImpl),
                padded_hidden_states_shape=prepare_output.padded_hidden_states_shape,
            )
        finally:
            experts.ascend_pertoken_scale = None
            experts.ascend_mc2_mask = None
            _EXTRA_CTX.moe_comm_method = previous_comm
