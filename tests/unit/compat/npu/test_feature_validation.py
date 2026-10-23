# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""NPU AFD feature validation rejects unsupported combinations before execution."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from afd_plugin.compat.npu import (  # noqa: E402
    fail_if_unsupported_npu_afd_features,
    npu_afd_num_ubatches,
)
from tests.unit.fakes.vllm_config import (  # noqa: E402
    async_moe_config as _async_moe_config,
)
from tests.unit.fakes.vllm_config import (  # noqa: E402
    npu_vllm_config as _vllm_config,
)

TWO_UBATCH_DBO = dict(
    enable_dbo=True, use_ubatching=True, num_ubatches=2, ubatch_size=4
)
ASYNC_CAM = dict(connector="CAMAsyncAFDConnector", async_dp=True)


@pytest.fixture
def vllm_runtime():
    # Non-FaaS validators parse upstream connector/router configuration.
    # FaaS model validation is CPU-only and must not acquire that dependency.
    pytest.importorskip("vllm")


@pytest.mark.parametrize(
    "model_type,architecture,quantization,use_sp,error",
    [
        ("deepseek_v4", "DeepseekV4ForCausalLM", "ascend", False, None),
        ("deepseek_v4", "AFDDeepseekV4ForCausalLM", "ascend", True, None),
        (
            "deepseek_v4",
            "DeepseekV4ForCausalLM",
            None,
            False,
            "Ascend W4A8 DeepSeek-V4",
        ),
        (
            "deepseek_v2",
            "DeepseekV2ForCausalLM",
            None,
            False,
            "Ascend W4A8 DeepSeek-V4",
        ),
        (
            "deepseek_v2",
            "AFDDeepseekV2ForCausalLM",
            None,
            False,
            "Ascend W4A8 DeepSeek-V4",
        ),
        (
            "deepseek_v2",
            "DeepseekV2ForCausalLM",
            "ascend",
            False,
            "Ascend W4A8 DeepSeek-V4",
        ),
        (
            "deepseek_v2",
            "DeepseekV2ForCausalLM",
            None,
            True,
            "Ascend W4A8 DeepSeek-V4",
        ),
        (
            "deepseek_v3",
            "DeepseekV3ForCausalLM",
            None,
            False,
            "Ascend W4A8 DeepSeek-V4",
        ),
        (
            "deepseek_v2",
            "DeepseekV4ForCausalLM",
            "ascend",
            False,
            "Ascend W4A8 DeepSeek-V4",
        ),
        (
            "deepseek_v4",
            "DeepseekV2ForCausalLM",
            "ascend",
            False,
            "Ascend W4A8 DeepSeek-V4",
        ),
    ],
)
def test_faas_validation_matches_model_quantization_and_layout(
    model_type, architecture, quantization, use_sp, error
):
    config = _config_with(
        connector="FaaSAsyncAFDConnector",
        async_dp=True,
        compute_gate_on_attention=True,
        enable_eplb=False,
        enable_elastic_ep=False,
        use_sequence_parallel_moe=use_sp,
        model={
            "hf_config": SimpleNamespace(
                model_type=model_type, architectures=[architecture]
            ),
            "quantization": quantization,
            "dtype": "torch.bfloat16",
        },
    )
    with pytest.raises(RuntimeError, match=error) if error else nullcontext():
        fail_if_unsupported_npu_afd_features(config)


@pytest.mark.parametrize(
    "model,parallel,error",
    [
        ({"dtype": "torch.float16"}, {}, "in eager mode"),
        ({"enforce_eager": False}, {}, "in eager mode"),
        ({}, {"enable_dbo": True}, "without DBO"),
        ({}, {"enable_eplb": True}, "static routed experts"),
        ({}, {"prefill_context_parallel_size": 2}, "without DBO, PP, CP"),
    ],
)
def test_faas_v4_retains_precision_and_static_execution_boundaries(
    model, parallel, error
):
    config = _config_with(
        connector="FaaSAsyncAFDConnector",
        async_dp=True,
        compute_gate_on_attention=True,
        enable_elastic_ep=False,
        model={
            "hf_config": SimpleNamespace(
                model_type="deepseek_v4", architectures=["DeepseekV4ForCausalLM"]
            ),
            "quantization": "ascend",
            "dtype": "torch.bfloat16",
            **model,
        },
        **({"enable_eplb": False} | parallel),
    )
    with pytest.raises(RuntimeError, match=error):
        fail_if_unsupported_npu_afd_features(config)


def _config_with(*, model=None, max_num_batched_tokens=None, **kwargs):
    config = _vllm_config(**kwargs)
    for name, value in (model or {}).items():
        setattr(config.model_config, name, value)
    if max_num_batched_tokens is not None:
        config.scheduler_config.max_num_batched_tokens = max_num_batched_tokens
    return config


def _mla_dbo(cudagraph_mode, **model_fields):
    return _config_with(
        use_mla=True,
        cudagraph_mode=cudagraph_mode,
        model={
            "hf_text_config": SimpleNamespace(model_type="deepseek_v2", **model_fields)
        },
        **TWO_UBATCH_DBO,
    )


@pytest.mark.parametrize(
    ("config", "error"),
    [
        pytest.param(_config_with(model={"enforce_eager": False}), None, id="graph"),
        pytest.param(_config_with(**TWO_UBATCH_DBO), None, id="two-ubatches"),
        pytest.param(
            _config_with(**{**TWO_UBATCH_DBO, "num_ubatches": 4}),
            "exactly two ubatches",
            id="four-ubatches",
        ),
        pytest.param(
            _config_with(extra_config={"compute_gate_on_attention": True}),
            "compute_gate_on_attention",
            id="gate-on-attention",
        ),
        pytest.param(
            _config_with(extra_config={"quant_mode": 1}), "quant_mode=0", id="quant"
        ),
        pytest.param(
            _config_with(connector="P2pNcclAFDConnector", extra_config={"core_num": 8}),
            "does not support connector_extra_config",
            id="connector-extra-parser",
        ),
        pytest.param(_mla_dbo("FULL"), "FULL_DECODE_ONLY", id="mla-full"),
        pytest.param(
            _mla_dbo("FULL_AND_PIECEWISE"), "FULL_DECODE_ONLY", id="mla-mixed"
        ),
        pytest.param(_mla_dbo("FULL_DECODE_ONLY"), None, id="mla-decode"),
        pytest.param(_mla_dbo("FULL", index_topk=8), None, id="sparse"),
        # Compressor models select the DSA backend, whose graph-params update is
        # a no-op, so the MLA DBO graph rules do not apply.
        pytest.param(
            _mla_dbo("FULL", index_topk=512, compress_ratios=[0, 4, 128]),
            None,
            id="compressor",
        ),
        pytest.param(
            _config_with(
                use_mla=True,
                cudagraph_mode="FULL_DECODE_ONLY",
                speculative_config=object(),
                **TWO_UBATCH_DBO,
            ),
            "does not support speculative decoding",
            id="mla-draft",
        ),
        pytest.param(
            _config_with(**{**ASYNC_CAM, "async_dp": False}),
            "async=true",
            id="async-off",
        ),
        pytest.param(
            _config_with(model={"enforce_eager": False}, **ASYNC_CAM),
            "only eager",
            id="async-graph",
        ),
        pytest.param(
            _config_with(use_ubatching=True, **ASYNC_CAM),
            "DBO requires exactly two ubatches",
            id="async-native-ubatching",
        ),
        pytest.param(
            _config_with(enable_dbo=True, **ASYNC_CAM),
            "DBO requires exactly two ubatches",
            id="async-native-dbo",
        ),
        *(
            pytest.param(
                _config_with(
                    use_mla=True, extra_config={"dynamicQuant": value}, **ASYNC_CAM
                ),
                error,
                id=f"async-dynamic-quant-{value}",
            )
            for value, error in ((0, None), ("1", None), (2, "dynamicQuant"))
        ),
        pytest.param(
            _config_with(
                compute_gate_on_attention=True,
                use_v2_model_runner=True,
                use_mla=True,
                tensor_parallel_size=2,
                num_attention_ranks=2,
                extra_config={"attn_ranks_per_dp": 2},
                max_num_batched_tokens=5,
                **ASYNC_CAM,
            ),
            "divisible by attn_ranks_per_dp",
            id="async-mrv2-fractional-tp-capacity",
        ),
        pytest.param(_async_moe_config(), None, id="moe-request-tp1"),
        pytest.param(
            _async_moe_config(tensor_parallel_size=2, num_attention_ranks=2),
            None,
            id="moe-token-attention-tp2",
        ),
        pytest.param(
            _async_moe_config(role="ffn"),
            None,
            id="moe-token-ffn-tp1",
        ),
        pytest.param(
            _async_moe_config(compute_gate_on_attention=False),
            "compute_gate_on_attention",
            id="moe-missing-attention-gate",
        ),
        pytest.param(
            _async_moe_config(num_ubatches=3),
            "exactly two",
            id="moe-three-stages",
        ),
        pytest.param(
            _async_moe_config(decode_context_parallel_size=2),
            "CP",
            id="moe-decode-context-parallel",
        ),
        pytest.param(
            _async_moe_config(prefill_context_parallel_size=2),
            "CP",
            id="moe-prefill-context-parallel",
        ),
        pytest.param(
            _async_moe_config(tensor_parallel_size=2, attn_ranks_per_dp=1),
            "attn_ranks_per_dp",
            id="moe-attention-cam-tp-mismatch",
        ),
    ],
)
def test_npu_feature_validation(config, error, vllm_runtime):
    with (
        pytest.raises((RuntimeError, ValueError), match=error)
        if error
        else nullcontext()
    ):
        fail_if_unsupported_npu_afd_features(config)


def test_npu_afd_num_ubatches_reads_two_ubatch_dbo():
    assert npu_afd_num_ubatches(_vllm_config(**TWO_UBATCH_DBO)) == 2


@pytest.mark.parametrize("role", ["attention", "ffn"])
@pytest.mark.parametrize(
    ("factor", "max_num_batched_tokens", "rejected"),
    [
        ("0.015625", 8192, True),
        ("0.015625", 4096, False),
        ("0.5", 8192, False),
        (None, 262144, False),
        ("0", 16, True),
        ("nan", 16, True),
        ("invalid", 16, True),
        ("0.01_5625", 4096, True),
    ],
)
def test_npu_async_feature_validation_checks_receive_capacity(
    monkeypatch,
    role,
    factor,
    max_num_batched_tokens,
    rejected,
    vllm_runtime,
):
    if factor is None:
        monkeypatch.delenv("BATCH_SIZE_FACTOR", raising=False)
    else:
        monkeypatch.setenv("BATCH_SIZE_FACTOR", factor)
    config = _vllm_config(
        role=role,
        connector="CAMAsyncAFDConnector",
        async_dp=True,
        use_mla=True,
        use_v2_model_runner=True,
        num_attention_ranks=2,
        tensor_parallel_size=2 if role == "attention" else 1,
        extra_config={"attn_ranks_per_dp": 2},
    )
    config.scheduler_config.max_num_batched_tokens = max_num_batched_tokens
    if rejected:
        with pytest.raises(RuntimeError, match="BATCH_SIZE_FACTOR"):
            fail_if_unsupported_npu_afd_features(config)
    else:
        fail_if_unsupported_npu_afd_features(config)


@pytest.mark.parametrize(
    "extra_config,additional_config,model_type,use_mla,error",
    [
        (
            {},
            {"enable_shared_expert_dp": True},
            "deepseek_v2",
            True,
            "shared-expert DP",
        ),
        *[
            (
                {},
                {"finegrained_tp_config": {key: 2}},
                "deepseek_v2",
                True,
                "finegrained TP",
            )
            for key in (
                "oproj_tensor_parallel_size",
                "mlp_tensor_parallel_size",
                "embedding_tensor_parallel_size",
                "lmhead_tensor_parallel_size",
            )
        ],
        ({}, {}, "qwen3_moe", True, "DeepSeek-V2/V3 MLA"),
        ({}, {}, "deepseek_v2", False, "DeepSeek-V2/V3 MLA"),
    ],
)
def test_npu_async_mrv2_feature_boundaries(
    extra_config, additional_config, model_type, use_mla, error, vllm_runtime
):
    config = _vllm_config(
        connector="CAMAsyncAFDConnector",
        async_dp=True,
        compute_gate_on_attention=True,
        use_v2_model_runner=True,
        use_mla=use_mla,
        extra_config=extra_config,
    )
    config.additional_config.update(additional_config)
    config.model_config.hf_text_config.model_type = model_type
    with pytest.raises(RuntimeError, match=error):
        fail_if_unsupported_npu_afd_features(config)


@pytest.mark.parametrize(
    "model_type,use_mla",
    [("deepseek_v2", True), ("deepseek_v3", True), ("deepseek_v4", False)],
)
@pytest.mark.parametrize("tp_size,use_sp", [(1, False), (2, False), (2, True)])
@pytest.mark.parametrize("local_ubatching", [False, True])
def test_npu_async_mrv2_features_allow_model_layouts(
    model_type, use_mla, tp_size, use_sp, local_ubatching, vllm_runtime
):
    config = _vllm_config(
        connector="CAMAsyncAFDConnector",
        async_dp=True,
        compute_gate_on_attention=True,
        use_v2_model_runner=True,
        use_mla=use_mla,
        tensor_parallel_size=tp_size,
        use_sequence_parallel_moe=use_sp,
        use_ubatching=local_ubatching,
        num_ubatches=2,
        num_attention_ranks=2 * tp_size,
        data_parallel_size=2,
        extra_config={
            "attn_ranks_per_dp": tp_size,
            "dynamicQuant": 1,
        },
    )
    config.model_config.hf_text_config.model_type = model_type
    config.additional_config["enable_flashcomm1"] = use_sp
    fail_if_unsupported_npu_afd_features(config)
