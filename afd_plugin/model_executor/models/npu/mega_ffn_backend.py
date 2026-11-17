# SPDX-License-Identifier: Apache-2.0
"""F-side MegaFFN adapter for the synchronous A-side-routing protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as functional
from vllm.logger import init_logger

if TYPE_CHECKING:
    from blockrt.models.mega_ffn.ffn_runtime import MegaFfnRunner
    from transformers import PretrainedConfig

logger = init_logger("vllm.afd.mega_ffn")


class MegaFFNBackend:
    """Lazy conversion after vLLM weight loading; fixed-shape normal-mode graphs.

    Shared experts stay on A. The received top-k weights already contain the
    routing scale. This component performs only the routed FFN on F.
    Instances are serialized by the synchronous FFN loop, not thread-safe.
    """

    def __init__(
        self,
        experts: torch.nn.Module,
        config: PretrainedConfig,
        layer_idx: int,
        *,
        max_tokens: int,
    ) -> None:
        if max_tokens < 1:
            raise ValueError("MegaFFN capacity must be positive")
        self.max_tokens = max_tokens
        self.experts = experts
        self.config = config
        self.layer_idx = layer_idx
        self.weights: dict[str, torch.Tensor] | None = None
        self.runners: dict[int, MegaFfnRunner] = {}

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        # Optional native extension: native AFD must import without blockrt.
        from blockrt.models.mega_ffn.ffn_args import MegaFfnArgs
        from blockrt.models.mega_ffn.ffn_runtime import MegaFfnRunner
        from blockrt.models.mega_ffn.weight_loader import convert_routed_expert_weights

        args = MegaFfnArgs(
            hidden_size=self.config.hidden_size,
            moe_intermediate_size=self.config.moe_intermediate_size,
            n_routed_experts=self.config.n_routed_experts,
            num_experts_per_tok=self.config.num_experts_per_tok,
            routed_scaling_factor=1.0,
        )
        num_tokens = hidden_states.shape[0]
        if num_tokens > self.max_tokens:
            raise ValueError(f"MegaFFN token count exceeds capacity {self.max_tokens}")
        if (
            hidden_states.shape != (num_tokens, args.hidden_size)
            or hidden_states.dtype != torch.bfloat16
        ):
            raise ValueError("MegaFFN hidden_states must be BF16 [M,H]")
        if (
            topk_ids.shape != (num_tokens, args.num_experts_per_tok)
            or topk_weights.shape != topk_ids.shape
        ):
            raise ValueError("MegaFFN routing tensors must both have shape [M,K]")
        if num_tokens == 0:
            return torch.empty_like(hidden_states)
        if self.weights is None:
            self.weights = convert_routed_expert_weights(
                args,
                self.experts.w13_weight,
                self.experts.w2_weight,
            )
            logger.info(
                "MegaFFN layer=%d converted loaded BF16 experts to MXFP8",
                self.layer_idx,
            )
        # Normal graphs currently retain native allocations until process exit.
        # Use two fixed capacities; never evict/rebuild on changing prompt sizes.
        graph_tokens = 1 if num_tokens == 1 else self.max_tokens
        runner = self.runners.get(graph_tokens)
        if runner is None:
            runner = MegaFfnRunner(
                args, device_id=hidden_states.device.index, weights=self.weights
            )
            runner.build_graph(graph_tokens)
            self.runners[graph_tokens] = runner
            logger.info(
                "MegaFFN layer=%d captured tokens=%d normal mode",
                self.layer_idx,
                graph_tokens,
            )
        padding = graph_tokens - num_tokens
        if padding:
            # Rows are independent. Padded rows carry zero routing weight and
            # valid expert IDs; only the caller's rows are returned to E2A.
            hidden_states = functional.pad(hidden_states, (0, 0, 0, padding))
            topk_weights = functional.pad(topk_weights, (0, 0, 0, padding))
            topk_ids = functional.pad(topk_ids, (0, 0, 0, padding))
        result = runner.run_ffn(
            hidden_states, topk_weights.to(torch.float32), topk_ids
        )[:num_tokens]
        if not bool(torch.isfinite(result).all()):
            raise RuntimeError(
                f"MegaFFN layer {self.layer_idx} produced non-finite output"
            )
        logger.debug("MegaFFN layer=%d completed tokens=%d", self.layer_idx, num_tokens)
        return result
