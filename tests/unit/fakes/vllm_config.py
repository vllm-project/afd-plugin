# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any


def camp2p_config(
    *,
    num_ubatches: int = 1,
    n_shared_experts: int = 0,
    extra_config=None,
):
    return SimpleNamespace(
        additional_config={"afd": {"connector_extra_config": extra_config or {}}},
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            data_parallel_rank=0,
            prefill_context_parallel_size=1,
            tensor_parallel_size=1,
            num_ubatches=num_ubatches,
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=8),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                hidden_size=16,
                num_experts_per_tok=2,
                n_routed_experts=4,
                n_shared_experts=n_shared_experts,
            ),
        ),
    )


def npu_parallel_config(**overrides):
    values = {
        "data_parallel_size": 1,
        "data_parallel_rank": 0,
        "enable_dbo": False,
        "use_ubatching": False,
        "num_ubatches": 1,
        "ubatch_size": 0,
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "use_sequence_parallel_moe": False,
        "prefill_context_parallel_size": 1,
        "decode_context_parallel_size": 1,
        "dbo_decode_token_threshold": 1,
        "dbo_prefill_token_threshold": 1,
        "worker_cls": "unused",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def npu_vllm_config(
    *,
    role="attention",
    connector="CAMP2pAFDConnector",
    extra_config=None,
    use_mla=False,
    use_v2_model_runner=True,
    cudagraph_mode="FULL",
    speculative_config=None,
    **parallel_overrides,
):
    async_dp = bool(parallel_overrides.pop("async_dp", False))
    compute_gate_on_attention = bool(
        parallel_overrides.pop("compute_gate_on_attention", False),
    )
    num_attention_ranks = int(parallel_overrides.pop("num_attention_ranks", 1))
    num_ffn_ranks = int(parallel_overrides.pop("num_ffn_ranks", 1))
    return SimpleNamespace(
        use_v2_model_runner=use_v2_model_runner,
        additional_config={
            "afd": {
                "role": role,
                "connector": connector,
                "async": async_dp,
                "compute_gate_on_attention": compute_gate_on_attention,
                "num_attention_ranks": num_attention_ranks,
                "num_ffn_ranks": num_ffn_ranks,
                "connector_extra_config": extra_config or {},
            },
        },
        parallel_config=npu_parallel_config(**parallel_overrides),
        model_config=SimpleNamespace(
            enforce_eager=True,
            hf_text_config=SimpleNamespace(model_type="deepseek_v2"),
            use_mla=use_mla,
        ),
        compilation_config=SimpleNamespace(
            cudagraph_mode=SimpleNamespace(
                name=cudagraph_mode,
                has_full_cudagraphs=lambda: (
                    cudagraph_mode in {"FULL", "FULL_DECODE_ONLY", "FULL_AND_PIECEWISE"}
                ),
            ),
            fast_moe_cold_start=False,
        ),
        speculative_config=speculative_config,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16),
    )


def async_moe_config(
    *,
    role="attention",
    compute_gate_on_attention=True,
    num_attention_ranks=1,
    num_ffn_ranks=1,
    tensor_parallel_size=1,
    prefill_context_parallel_size=1,
    decode_context_parallel_size=1,
    num_ubatches=2,
    **extra_config,
):
    return npu_vllm_config(
        role=role,
        connector="CAMAsyncAFDConnector",
        use_mla=True,
        async_dp=True,
        compute_gate_on_attention=compute_gate_on_attention,
        num_attention_ranks=num_attention_ranks,
        num_ffn_ranks=num_ffn_ranks,
        tensor_parallel_size=tensor_parallel_size,
        prefill_context_parallel_size=prefill_context_parallel_size,
        decode_context_parallel_size=decode_context_parallel_size,
        use_ubatching=role == "attention",
        num_ubatches=num_ubatches,
        extra_config={
            "attn_ranks_per_dp": tensor_parallel_size,
            **extra_config,
        },
    )


# DeepEP M2N connector tests: Qwen3-30B-A3B MoE shapes.
M2N_HIDDEN_SIZE = 2048
M2N_NUM_EXPERTS = 128
M2N_TOP_K = 8
M2N_INTERMEDIATE_SIZE = 768
# Fake vLLM limits: capacity defaults to the larger of the two.
M2N_MAX_NUM_BATCHED_TOKENS = 8192
M2N_MAX_CUDAGRAPH_CAPTURE_SIZE = 512
# Fake CUDA-graph capture sizes: 1, 2, 4, then every 8 up to the maximum above.
M2N_CUDAGRAPH_CAPTURE_SIZES = (
    1,
    2,
    4,
    *range(8, M2N_MAX_CUDAGRAPH_CAPTURE_SIZE + 1, 8),
)


@dataclass(frozen=True)
class M2NTopology:
    ffn_size: int
    attn_size: int
    # Attention TP peers (consecutive Attention ranks) share one DP rank's
    # tokens; each dispatches its slice.
    attn_tp_size: int = 1

    @property
    def world_size(self) -> int:
        return self.ffn_size + self.attn_size

    @property
    def attn_dp_size(self) -> int:
        return self.attn_size // self.attn_tp_size

    def role(self, world_rank: int) -> tuple[str, int]:
        if world_rank < self.ffn_size:
            return "ffn", world_rank
        return "attention", world_rank - self.ffn_size


def m2n_vllm_config(
    topology: M2NTopology,
    role: str,
    role_rank: int,
    extra_config: dict[str, Any],
    port: int,
    host: str = "127.0.0.1",
) -> SimpleNamespace:
    """A ``vllm_config`` for one DeepEP M2N rank of ``topology``."""
    # Deferred: this fakes module must import on CPU-only hosts without torch.
    import torch
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config

    text_config = SimpleNamespace(
        hidden_size=M2N_HIDDEN_SIZE,
        num_hidden_layers=2,
        num_experts=M2N_NUM_EXPERTS,
        num_experts_per_tok=M2N_TOP_K,
        moe_intermediate_size=M2N_INTERMEDIATE_SIZE,
    )
    # vLLM's view: FFN ranks are DP ranks; Attention ranks are DP x TP.
    tp_size = 1 if role == "ffn" else topology.attn_tp_size
    dp_size = topology.ffn_size if role == "ffn" else topology.attn_dp_size
    # The FFN deployment the handshake requires: linear EP over the FFN ranks,
    # with checkpoint weights in the payload's format.
    quant_config = (
        Fp8Config(is_checkpoint_fp8_serialized=True, weight_block_size=[128, 128])
        if extra_config.get("payload_format") == "fp8_block128_fp32"
        else None
    )
    return SimpleNamespace(
        additional_config={
            "afd": {
                "connector": "DeepEPM2NAFDConnector",
                "role": role,
                "host": host,
                "port": port,
                "num_attention_ranks": topology.attn_size,
                "num_ffn_ranks": topology.ffn_size,
                "compute_gate_on_attention": True,
                "connector_extra_config": extra_config,
            }
        },
        quant_config=quant_config,
        compilation_config=SimpleNamespace(
            max_cudagraph_capture_size=M2N_MAX_CUDAGRAPH_CAPTURE_SIZE,
            cudagraph_capture_sizes=list(M2N_CUDAGRAPH_CAPTURE_SIZES),
        ),
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            enforce_eager=True,
            get_num_experts=lambda: M2N_NUM_EXPERTS,
            hf_config=text_config,
            hf_text_config=text_config,
        ),
        parallel_config=SimpleNamespace(
            data_parallel_size=dp_size,
            data_parallel_rank=role_rank // tp_size,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            tensor_parallel_size=tp_size,
            enable_expert_parallel=True,
            expert_placement_strategy="linear",
            enable_eplb=False,
            all2all_backend="allgather_reducescatter",
            use_ubatching=False,
            num_ubatches=1,
        ),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=M2N_MAX_NUM_BATCHED_TOKENS
        ),
    )
