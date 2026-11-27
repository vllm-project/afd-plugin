# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

import sys
from types import SimpleNamespace

import pytest

from tests.unit.fakes.platform import use_cpu_moe_factory


@pytest.fixture
def cam_factory_on_cpu(monkeypatch):
    """Use the real vLLM factory and connector registration with CPU weight owners."""
    pytest.importorskip("torch")
    pytest.importorskip("vllm")
    from vllm.distributed import parallel_state

    from afd_plugin.model_executor.ffn import remote

    use_cpu_moe_factory(monkeypatch)
    monkeypatch.setattr(remote, "current_platform", SimpleNamespace(device_type="npu"))
    monkeypatch.setitem(
        sys.modules,
        "afd_plugin.model_executor.ffn.npu.remote",
        SimpleNamespace(AFDAscendRemoteRoutedExperts=remote.AFDRemoteRoutedExperts),
    )

    monkeypatch.setattr(
        parallel_state, "_TP", SimpleNamespace(rank_in_group=0, world_size=1)
    )


@pytest.fixture
def cam_moe(cam_factory_on_cpu, monkeypatch):
    from vllm.config import set_current_vllm_config

    from afd_plugin.model_executor.models.npu.deepseek_v4_moe import AFDDeepseekV4MoE
    from tests.unit.fakes.vllm_config import attention_moe_vllm_config

    def build(
        *,
        gate,
        vllm_config=None,
        layer_idx=3,
        is_draft_layer=False,
        routing_config=None,
        shared_cls=None,
    ):
        config = SimpleNamespace(
            hidden_size=4,
            moe_intermediate_size=8,
            num_experts_per_tok=2,
            n_routed_experts=3,
            n_shared_experts=int(shared_cls is not None),
            hidden_act="silu",
            norm_topk_prob=True,
            vocab_size=4,
            **(routing_config or {}),
        )
        if shared_cls is not None:
            monkeypatch.setitem(
                sys.modules,
                "afd_plugin.model_executor.models.npu.deepseek_v4_shared_experts",
                SimpleNamespace(AFDDeepseekV4SharedExperts=shared_cls),
            )
        if vllm_config is None:
            vllm_config = attention_moe_vllm_config(
                connector="CAMP2pAFDConnector", compute_gate_on_attention=gate
            )
        with set_current_vllm_config(vllm_config):
            return AFDDeepseekV4MoE(
                vllm_config=vllm_config,
                config=config,
                layer_idx=layer_idx,
                prefix=f"model.layers.{layer_idx}.mlp",
                compute_gate_on_attention=gate,
                is_draft_layer=is_draft_layer,
            )

    return build
