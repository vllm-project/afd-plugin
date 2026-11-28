# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

import importlib.util
import sys
from types import SimpleNamespace

import pytest

from tests.unit.fakes.platform import use_cpu_moe_factory


@pytest.fixture
def cam_factory_on_cpu(monkeypatch):
    """Build the CAM runners with the real vLLM factory and CPU weight owners."""
    pytest.importorskip("torch")
    pytest.importorskip("vllm")
    from vllm.distributed import parallel_state

    from afd_plugin.model_executor.ffn import remote

    use_cpu_moe_factory(monkeypatch)
    monkeypatch.setattr(remote, "current_platform", SimpleNamespace(device_type="npu"))
    monkeypatch.setattr(
        parallel_state, "_TP", SimpleNamespace(rank_in_group=0, world_size=1)
    )


@pytest.fixture
def ascend_v4_model(cam_factory_on_cpu, monkeypatch):
    """Load the model with CPU base classes without caching a fake Ascend decoder."""
    from torch import nn

    native = SimpleNamespace(
        DeepseekV4DecoderLayer=nn.Module,
        DeepseekV4Model=nn.Module,
        AscendDeepseekV4ForCausalLM=nn.Module,
        support_torch_compile=lambda cls: cls,
        DeepseekV4Attention=nn.Identity,
        RMSNorm=nn.Identity,
    )
    with monkeypatch.context() as patch:
        patch.setitem(
            sys.modules, "vllm_ascend.models.deepseek_v4", SimpleNamespace(model=native)
        )
        spec = importlib.util.find_spec(
            "afd_plugin.model_executor.models.npu.deepseek_v4"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


@pytest.fixture
def cam_moe(ascend_v4_model, monkeypatch):
    from vllm.config import set_current_vllm_config

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
            return ascend_v4_model.AFDDeepseekV4MoE(
                vllm_config=vllm_config,
                config=config,
                layer_idx=layer_idx,
                prefix=f"model.layers.{layer_idx}.mlp",
                compute_gate_on_attention=gate,
                is_draft_layer=is_draft_layer,
            )

    return build
