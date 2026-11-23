# SPDX-License-Identifier: Apache-2.0
"""Experimental MegaFFN scope validation without device imports."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig


def validate_mega_ffn_config(
    config: VllmConfig, *, execution_mode: str = "normal"
) -> None:
    model, parallel = config.model_config, config.parallel_config
    hf = model.hf_config
    if execution_mode == "persistent" and hf.num_hidden_layers < 1:
        raise ValueError("mega_ffn persistent requires a positive layer count")
    if hf.model_type != "deepseek_v4":
        raise ValueError("mega_ffn currently supports DeepSeek V4 only")
    if not model.enforce_eager or str(model.dtype) != "torch.bfloat16":
        raise ValueError("mega_ffn requires eager BF16 execution")
    if config.quant_config is not None:
        raise ValueError("mega_ffn requires unquantized BF16 source weights")
    for field in (
        "tensor_parallel_size",
        "data_parallel_size",
        "pipeline_parallel_size",
        "prefill_context_parallel_size",
        "decode_context_parallel_size",
    ):
        if getattr(parallel, field) != 1:
            raise ValueError(f"mega_ffn requires {field}=1")
    if (
        parallel.enable_dbo
        or parallel.use_ubatching
        or config.speculative_config is not None
    ):
        raise ValueError("mega_ffn does not support DBO, ubatching or speculation")
    if (getattr(hf, "swiglu_limit", None) or 0) != 0:
        raise ValueError("mega_ffn lacks SwiGLU clipping; require swiglu_limit=0")
    if hf.hidden_size % 64 or hf.moe_intermediate_size % 64:
        raise ValueError("mega_ffn requires H and I divisible by 64")
