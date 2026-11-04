# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

"""Runtime gate for the sync Attention-side gate prototype.

The sync shell does not call the DBO yield, so a DBO config would interleave
microbatches against a transport that never yields. Not calling the yield is not
the same as forbidding the feature, so the limit is enforced here.

The entry-level tests matter more than the helper ones: a helper test keeps passing
if the call is deleted from the validator, while a test that goes through the
validator fails. Both roles are covered because both run this validator.
"""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from afd_plugin import validation
from afd_plugin.validation import _fail_if_unsupported_sync_gate_runtime


def _runner_config(*, enable_dbo=False, use_ubatching=False):
    """A ``VllmConfig`` surface complete enough for a clean return.

    Every field the validator reads on the accepted path is present, so the
    "allowed" test can assert a real return instead of tolerating an exception.
    """

    return SimpleNamespace(
        use_v2_model_runner=True,
        model_config=SimpleNamespace(enforce_eager=True),
        speculative_config=None,
        parallel_config=SimpleNamespace(
            enable_dbo=enable_dbo,
            use_ubatching=use_ubatching,
            num_ubatches=2,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            data_parallel_size=4,
            tensor_parallel_size=1,
            enable_expert_parallel=True,
            enable_elastic_ep=False,
            enable_eplb=False,
            use_sequence_parallel_moe=False,
        ),
        compilation_config=SimpleNamespace(
            pass_config=SimpleNamespace(enable_sp=False),
        ),
    )


@pytest.fixture()
def sync_gate_afd_config(monkeypatch):
    """Make the validator see the sync gate-on-Attention configuration."""

    from afd_plugin.config import CAMP2P_CONNECTOR

    afd_config = SimpleNamespace(
        connector=CAMP2P_CONNECTOR,
        async_dp=False,
        compute_gate_on_attention=True,
        num_attention_ranks=4,
        num_ffn_ranks=4,
    )
    monkeypatch.setattr(
        validation,
        "parse_afd_config",
        lambda _vllm_config, expected_role=None: afd_config,
    )
    # The validator imports this lazily, and the real module path pulls vLLM in.
    # Inject the leaf module so the accepted path can run on a CPU host.
    for name, attributes in (
        ("afd_plugin.model_executor", {}),
        ("afd_plugin.model_executor.models", {}),
        (
            "afd_plugin.model_executor.models.model_utils",
            {"has_afd_model_registration": lambda _model_config: True},
        ),
    ):
        module = ModuleType(name)
        module.__path__ = []
        for attribute, value in attributes.items():
            setattr(module, attribute, value)
        monkeypatch.setitem(sys.modules, name, module)
    return afd_config


@pytest.mark.parametrize("role", ["attention", "ffn"])
@pytest.mark.parametrize("unsupported", ["dbo", "ubatching"])
def test_validator_rejects_unsupported_parallel_features(
    sync_gate_afd_config, role, unsupported
):
    """Both roles refuse DBO and ubatching: the sync shell never yields."""

    settings = {"enable_dbo": True} if unsupported == "dbo" else {"use_ubatching": True}
    with pytest.raises(RuntimeError, match="DBO or ubatching"):
        validation.validate_npu_model_runner_v2_config(
            _runner_config(**settings),
            expected_role=role,
            device_type="npu",
        )


@pytest.mark.parametrize("role", ["attention", "ffn"])
def test_validator_accepts_a_plain_sync_gate_config(sync_gate_afd_config, role):
    """A supported configuration must return cleanly, not merely avoid one message."""

    validation.validate_npu_model_runner_v2_config(
        _runner_config(),
        expected_role=role,
        device_type="npu",
    )
