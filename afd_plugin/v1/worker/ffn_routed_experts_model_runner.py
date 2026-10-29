# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""FFN-side model runner for connectors that deliver routed expert rows."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from vllm.config import VllmConfig
from vllm.forward_context import DPMetadata

from afd_plugin.connectors import (
    AFDConnectorBase,
    AFDDPMetadata,
    AFDRoutedExpertsConnector,
    AFDTransferContext,
    FFNLocalExperts,
    LocalExpertWeights,
)
from afd_plugin.connectors.metadata import FP8_DTYPE
from afd_plugin.v1.worker.ffn_model_runner import GPUFFNModelRunner

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts


# vLLM MoE backends whose ``process_weights_after_loading`` keeps the FusedMoE
# layout the connector's executor reads: w13 as [gate; up] with block-128
# scales. Others reorder in place without changing dtype or shape (FlashInfer
# swaps w13 to [up; gate]; TRT-LLM and Marlin also shuffle), which yields wrong
# outputs with no error.
ROUTED_MOE_BACKEND = "triton"
_LAYOUT_PRESERVING_BACKENDS = frozenset({"TRITON"})


def select_routed_moe_backend(vllm_config: VllmConfig) -> None:
    """Pin the FFN's vLLM MoE backend to one that keeps the expert layout.

    vLLM's MoE forward never runs on a routed-experts FFN; its backend only
    decides how the expert weights are processed at load. ``auto`` picks
    FlashInfer CUTLASS on Hopper with EP > 1 for block FP8, which reorders
    w13, so ``auto`` becomes ``triton`` and any other explicit choice fails.
    """
    kernel_config = vllm_config.kernel_config
    backend = str(kernel_config.moe_backend)
    if backend == "auto":
        kernel_config.moe_backend = ROUTED_MOE_BACKEND
    elif backend != ROUTED_MOE_BACKEND:
        raise RuntimeError(
            f"routed-experts AFD FFN needs --moe-backend {ROUTED_MOE_BACKEND} (or "
            f"auto); moe_backend={backend!r} reorders the expert weights the "
            "connector reads"
        )


def _check_expert_layout(routed_experts: RoutedExperts) -> None:
    quant_method = routed_experts.quant_method
    backend = getattr(quant_method, "fp8_backend", None) or getattr(
        quant_method, "unquantized_backend", None
    )
    name = getattr(backend, "name", None)
    if name not in _LAYOUT_PRESERVING_BACKENDS:
        raise RuntimeError(
            f"vLLM processed the FFN experts for MoE backend {name or backend!r}, "
            f"which may reorder them; the routed-experts connector reads them "
            f"as [gate; up] FusedMoE weights (supported: "
            f"{sorted(_LAYOUT_PRESERVING_BACKENDS)}; set --moe-backend "
            f"{ROUTED_MOE_BACKEND})"
        )


def local_experts_from_vllm(routed_experts: RoutedExperts) -> FFNLocalExperts:
    """This rank's experts of one vLLM MoE layer, as views of its parameters.

    Read after ``process_weights_after_loading``, which must have kept the
    FusedMoE layout (``select_routed_moe_backend``); checked here because a
    reordered layout keeps dtypes and shapes.
    """
    _check_expert_layout(routed_experts)
    w13 = routed_experts.w13_weight.data
    w2 = routed_experts.w2_weight.data
    w13_scale = w2_scale = None
    if w13.dtype == FP8_DTYPE:
        w13_scale = routed_experts.w13_weight_scale_inv.data
        w2_scale = routed_experts.w2_weight_scale_inv.data
    return FFNLocalExperts(
        expert_ids=tuple(routed_experts.expert_map_manager.get_local_expert_ids()),
        weights=LocalExpertWeights(
            w13=w13, w2=w2, w13_scale=w13_scale, w2_scale=w2_scale
        ),
    )


def _require_routed(connector: AFDConnectorBase) -> AFDRoutedExpertsConnector:
    if not isinstance(connector, AFDRoutedExpertsConnector):
        raise TypeError(
            f"{type(connector).__name__} is not an AFDRoutedExpertsConnector; "
            "use GPUFFNModelRunner",
        )
    return connector


class GPUFFNRoutedExpertsModelRunner(GPUFFNModelRunner):
    """FFN runner for connectors that deliver routed (token, expert) rows.

    Routing (gate and top-k) runs on Attention. For each MoE layer and stage,
    in the same order as Attention, the FFN receives the rows routed to its
    local experts, runs those experts with the connector's executor and sends
    the weighted rows back just before that stage's next receive. vLLM still
    builds the model and loads and places the expert weights; its MoE forward
    never runs, so the P2P path's forward-context and DP metadata setup is
    skipped.

    The worker selects this runner for an ``AFDRoutedExpertsConnector``
    (DeepEP M2N).
    """

    def __init__(self, vllm_config: VllmConfig, device: torch.device) -> None:
        super().__init__(vllm_config, device)
        _require_routed(self.connector)
        self.moe_layer_indices: tuple[int, ...] = ()
        self.local_experts: dict[int, FFNLocalExperts] = {}

    @property
    def routed_connector(self) -> AFDRoutedExpertsConnector:
        """The connector, typed as the routed-experts contract it must meet."""
        return _require_routed(self.connector)

    def load_model(self, *, load_dummy_weights: bool = False, **kwargs: Any) -> None:
        """Load with the native loader, then take each MoE layer's experts.

        Runs again on ``reload_weights``, whose parameters replace the old ones.
        """
        super().load_model(load_dummy_weights=load_dummy_weights, **kwargs)
        model = self.get_model()
        self.moe_layer_indices = tuple(model.get_experts_layer_indices())
        self.local_experts = {
            layer_idx: local_experts_from_vllm(model.get_routed_experts(layer_idx))
            for layer_idx in self.moe_layer_indices
        }

    def initialize_afd_connector(self) -> None:
        """Collective: share the loaded experts and, with EPLB, the expert
        maps and rearrangement schedule the whole world routes by."""
        self.routed_connector.init_afd_connector(
            moe_layer_indices=self.moe_layer_indices,
            local_experts=self.local_experts,
            eplb_state=(
                self.eplb.state
                if self.vllm_config.parallel_config.enable_eplb
                else None
            ),
        )

    def _ffn_forward(
        self,
        *,
        dp_metadata_list: dict[int, DPMetadata | AFDDPMetadata],
        is_graph_capturing: bool = False,
        is_warmup: bool = False,
        is_graph_replaying: bool = False,
        update_connector_state: bool = True,
    ) -> torch.Tensor | None:
        if update_connector_state:
            self._update_connector_state(
                dp_metadata_list,
                is_graph_capturing=is_graph_capturing,
                is_warmup=is_warmup,
                is_graph_replaying=is_graph_replaying,
            )
        connector = self.routed_connector
        stage_ids = sorted(int(stage_idx) for stage_idx in dp_metadata_list) or [0]
        ffn_output = None
        # Each stage's combine waits until just before that stage's next
        # dispatch, matching Attention's DBO micro-batch order
        # D0(0) D1(0) C0(0) D0(1) C1(0) D1(1) ... C0(n) C1(n). Dispatch and
        # combine can block the host, so a combine issued earlier would wait
        # on an Attention combine that only follows the next stage's dispatch.
        # On the first layer no stage has a result to send back yet.
        pending_outputs: dict[int, tuple[torch.Tensor, AFDTransferContext]] = {}
        for layer_idx in self.moe_layer_indices:
            weights = self.local_experts[layer_idx].weights
            for stage_idx in stage_ids:
                if stage_idx in pending_outputs:
                    connector.send_ffn_output(*pending_outputs.pop(stage_idx))
                payload = connector.recv_attn_output(ubatch_idx=stage_idx)
                ffn_output = connector.run_experts(payload, weights)
                pending_outputs[stage_idx] = (ffn_output, payload.context)
        for stage_idx in stage_ids:
            if stage_idx in pending_outputs:
                connector.send_ffn_output(*pending_outputs.pop(stage_idx))
        # Lane streams forked by dispatch/combine must rejoin the caller's
        # stream before a CUDA-graph capture ends.
        connector.synchronize_lanes()
        return ffn_output

    # The FFN role of an EPLB deployment never routes tokens, so nothing on
    # this side records expert load and EplbState.step has no input. Instead
    # the connector's FFN EPLB driver steps the rearrangement counter (every
    # connector step, real or dummy, exactly like the Attention recorder) and
    # at each boundary exchanges the Attention-recorded load window, runs
    # EplbState.rearrange and broadcasts the committed expert maps. Without
    # EPLB there is nothing to step. `is_dummy` is part of the base seam's
    # signature; the counter advances on dummy steps too, so it is unused.
    def eplb_step(self, is_dummy: bool = False) -> None:
        """Step FFN-side EPLB for the routed-experts path."""
        self.routed_connector.step_ffn_eplb()


__all__ = ["GPUFFNRoutedExpertsModelRunner", "local_experts_from_vllm"]
