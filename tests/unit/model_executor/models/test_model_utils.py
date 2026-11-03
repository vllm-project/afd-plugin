# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Draft model registration keeps native configuration aliases intact."""

import pytest

from tests.unit.fakes.async_ffn import load_host_module
from tests.unit.fakes.vllm_config import npu_dspark_config


@pytest.fixture
def configure_dspark(monkeypatch):
    # The parent models namespace imports vLLM forward context. Load this pure
    # configuration module directly, using the existing CPU host-test loader.
    module = load_host_module(
        monkeypatch, "afd_plugin.model_executor.models.model_utils"
    )
    return module.configure_afd_dspark


@pytest.fixture
def dspark_config():
    return npu_dspark_config()


def test_dspark_registration_preserves_native_text_config_alias(
    configure_dspark, dspark_config
):
    original = dspark_config.speculative_config.draft_model_config
    configure_dspark(dspark_config)
    draft = dspark_config.speculative_config.draft_model_config
    assert draft.hf_config.architectures == ["AFDDSparkDraftModel"]
    assert draft.hf_text_config is draft.hf_config
    assert original.hf_config.architectures == ["DSparkDraftModel"]


def test_dspark_rejects_connector_without_draft_ffn_loader(
    configure_dspark, dspark_config
):
    dspark_config.additional_config["afd"]["connector"] = "CAMP2pAFDConnector"
    dspark_config.additional_config["afd"]["async"] = False
    dspark_config.additional_config["afd"]["compute_gate_on_attention"] = False
    with pytest.raises(ValueError, match="requires CAMAsyncAFDConnector"):
        configure_dspark(dspark_config)


@pytest.mark.parametrize(
    "field", ["hidden_size", "num_experts_per_tok", "n_routed_experts"]
)
def test_dspark_rejects_incompatible_remote_transport(
    configure_dspark, dspark_config, field
):
    draft = dspark_config.speculative_config.draft_model_config.hf_text_config
    setattr(draft, field, getattr(draft, field) + 1)
    with pytest.raises(ValueError, match=f"transport {field} mismatch"):
        configure_dspark(dspark_config)
