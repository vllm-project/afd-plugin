# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from types import SimpleNamespace


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
        extra_config={
            "async_moe_ubatching": True,
            "attn_ranks_per_dp": tensor_parallel_size,
            **extra_config,
        },
    )
