# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU-only description of the Qwen3 MoE checkpoints the GPU FFN service serves.

Reads ``config.json`` only. Rejects every Qwen3 shape the routed-experts M2N
path cannot serve before any device, process group or weight is touched.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# NPU parallel: afd_plugin/ffn_faas/platforms/npu/v4_model.py::
#   V4CheckpointDescription (CPU-only checkpoint facts for /status and
#   startup validation).
# Reuse: PARTIAL - the read-validate-describe pattern and the positive-int
#   helper are the same; the V4 description also maps W4A8 safetensors names
#   for its own loader, while Qwen3 loads through vLLM's native loader.
#   REFACTOR-CANDIDATE(R6): a shared CPU checkpoint-config reader (config
#   JSON parse, positive-int fields, dtype check).

QWEN3_MOE_ARCHITECTURE = "Qwen3MoeForCausalLM"
BF16_DTYPE = "bfloat16"
BF16_BYTES = 2
# gate_proj, up_proj and down_proj per routed expert.
PROJECTIONS_PER_EXPERT = 3
# Qwen3 MoE places an MoE block on every layer whose index + 1 is a multiple of
# decoder_sparse_step; the routed-experts path requires every layer to be MoE.
DENSE_FREE_SPARSE_STEP = 1


def _positive_int(config: dict[str, Any], name: str) -> int:
    """Read a required positive integer field, rejecting bools and absent keys."""
    value = config.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"Qwen3 MoE checkpoint {name} must be a positive integer")
    return value


@dataclass(frozen=True)
class Qwen3CheckpointDescription:
    """Facts about one supported Qwen3 MoE checkpoint, read on the CPU."""

    root: Path
    num_layers: int
    num_experts: int
    hidden_size: int
    moe_intermediate_size: int
    top_k: int

    @property
    def layer_indices(self) -> tuple[int, ...]:
        """MoE layer indices; every layer is MoE for a supported checkpoint."""
        return tuple(range(self.num_layers))

    @classmethod
    def read(cls, model_path: str | Path) -> Qwen3CheckpointDescription:
        """Parse ``config.json`` and accept only the routed-experts Qwen3 shape.

        Accepts ``Qwen3MoeForCausalLM`` in BF16 without quantization, with an
        MoE block on every layer (no ``mlp_only_layers``, sparse step 1) and
        no shared expert. ``shared_expert_intermediate_size`` that is absent or
        ``null`` counts as zero, as vLLM reads it; the stock Qwen3-30B-A3B
        config omits it. Newer configs name the dtype ``dtype`` instead of
        ``torch_dtype``; either is read. Raises ``ValueError`` for anything
        else, and ``OSError`` when the path or file is missing.
        """
        root = Path(model_path).resolve(strict=True)
        try:
            config = json.loads((root / "config.json").read_text())
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid Qwen3 config.json: {error}") from error
        if not isinstance(config, dict):
            raise ValueError("Qwen3 config.json must be a JSON object")
        if config.get("architectures") != [QWEN3_MOE_ARCHITECTURE]:
            raise ValueError(
                f"GPU standalone FFN supports only {QWEN3_MOE_ARCHITECTURE}, got "
                f"{config.get('architectures')!r}"
            )
        dtype = config.get("torch_dtype", config.get("dtype"))
        if dtype != BF16_DTYPE:
            raise ValueError(
                f"GPU standalone FFN requires a BF16 checkpoint, got {dtype!r}"
            )
        if config.get("quantization_config") is not None:
            raise ValueError(
                "GPU standalone FFN does not support quantized checkpoints"
            )
        if config.get("mlp_only_layers"):
            raise ValueError("GPU standalone FFN requires every layer to be MoE")
        if config.get("decoder_sparse_step", DENSE_FREE_SPARSE_STEP) != (
            DENSE_FREE_SPARSE_STEP
        ):
            raise ValueError("GPU standalone FFN requires decoder_sparse_step=1")
        if (config.get("shared_expert_intermediate_size") or 0) > 0:
            raise ValueError("GPU standalone FFN does not support shared experts")
        return cls(
            root=root,
            num_layers=_positive_int(config, "num_hidden_layers"),
            num_experts=_positive_int(config, "num_experts"),
            hidden_size=_positive_int(config, "hidden_size"),
            moe_intermediate_size=_positive_int(config, "moe_intermediate_size"),
            top_k=_positive_int(config, "num_experts_per_tok"),
        )

    def local_experts(self, ffn_ranks: int) -> int:
        """Experts per F rank under linear placement; rejects an uneven split."""
        if ffn_ranks <= 0 or self.num_experts % ffn_ranks:
            raise ValueError(
                f"{self.num_experts} routed experts do not divide across "
                f"{ffn_ranks} FFN ranks"
            )
        return self.num_experts // ffn_ranks

    def estimated_local_expert_bytes(self, ffn_ranks: int) -> int:
        """BF16 bytes of this rank's routed experts over every layer.

        ``layers * local experts * 3 projections * hidden * intermediate * 2``;
        48 x 64 x 3 x 2048 x 768 x 2 B is about 27 GiB for Qwen3-30B-A3B at 2F.
        The MoE gate (``num_experts x hidden`` per layer) is not counted.
        """
        return (
            self.num_layers
            * self.local_experts(ffn_ranks)
            * PROJECTIONS_PER_EXPERT
            * self.hidden_size
            * self.moe_intermediate_size
            * BF16_BYTES
        )
