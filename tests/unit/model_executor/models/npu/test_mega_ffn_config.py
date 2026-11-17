# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest

from afd_plugin.model_executor.models.npu.mega_ffn_config import (
    validate_mega_ffn_config,
)


def valid_config():
    torch = pytest.importorskip("torch")
    return SimpleNamespace(
        model_config=SimpleNamespace(
            enforce_eager=True,
            dtype=torch.bfloat16,
            hf_config=SimpleNamespace(
                model_type="deepseek_v4",
                hidden_size=4096,
                moe_intermediate_size=1024,
                swiglu_limit=0,
            ),
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            data_parallel_size=1,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            enable_dbo=False,
            use_ubatching=False,
        ),
        quant_config=None,
        speculative_config=None,
    )


def test_unclipped_bf16_single_rank_is_supported():
    validate_mega_ffn_config(valid_config())


@pytest.mark.parametrize(
    "group, field, value",
    [
        ("model_config", "enforce_eager", False),
        ("parallel_config", "tensor_parallel_size", 2),
        ("parallel_config", "data_parallel_size", 2),
        ("parallel_config", "pipeline_parallel_size", 2),
        ("parallel_config", "prefill_context_parallel_size", 2),
        ("parallel_config", "decode_context_parallel_size", 2),
        ("parallel_config", "enable_dbo", True),
        ("parallel_config", "use_ubatching", True),
    ],
)
def test_unsupported_execution_is_rejected(group, field, value):
    config = valid_config()
    setattr(getattr(config, group), field, value)
    with pytest.raises(ValueError, match="mega_ffn"):
        validate_mega_ffn_config(config)


@pytest.mark.parametrize(
    "field, value",
    [
        ("swiglu_limit", 10),
        ("hidden_size", 65),
        ("moe_intermediate_size", 65),
        ("model_type", "deepseek_v2"),
    ],
)
def test_unsupported_model_semantics_are_rejected(field, value):
    config = valid_config()
    setattr(config.model_config.hf_config, field, value)
    with pytest.raises(ValueError, match="mega_ffn"):
        validate_mega_ffn_config(config)


@pytest.mark.parametrize("field", ["quant_config", "speculative_config"])
def test_quantization_and_speculation_are_rejected(field):
    config = valid_config()
    setattr(config, field, SimpleNamespace())
    with pytest.raises(ValueError, match="mega_ffn"):
        validate_mega_ffn_config(config)
