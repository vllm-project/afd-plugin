# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from types import SimpleNamespace

import torch
from torch import nn


class Stage(nn.Module):
    """Layer stage double that records the prefix of each constructed `kind`."""

    kind = "stage"

    def __init__(self, calls: dict[str, list[str]], *args, prefix="", **kwargs):
        super().__init__()
        calls[self.kind].append(prefix)
        self.weight = nn.Parameter(torch.empty(1))


class GateLinear(nn.Module):
    """Native gate's public shape/dtype contract for BF16 model fixtures."""

    def __init__(self, input_size, output_size, *, out_dtype=None, prefix=""):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(output_size, input_size, dtype=out_dtype or torch.bfloat16)
        )

    def forward(self, hidden_states):
        return hidden_states @ self.weight.t(), None


def stage_type(kind: str):
    return type(f"Fake{kind.title()}", (Stage,), {"kind": kind})


class DeepseekV4Decoder(nn.Module):
    pass


class DeepseekV4CausalLM(nn.Module):
    def __init__(self, *, vllm_config, prefix=""):
        super().__init__()
        self.model = self.model_cls(vllm_config=vllm_config, prefix=prefix)

    def get_mtp_target_hidden_states(self):
        # The native public getter exposes the model-owned residual buffer.
        return self.model._mtp_hidden_buffer


class DSAImpl:
    _oproj_buffer_pool: SimpleNamespace

    def reserve_oproj_buffers(self, device, dtype):
        pool = self._oproj_buffer_pool
        if pool.output is None:
            pool.output = torch.empty(1, device=device, dtype=dtype)
        self.output = pool.output

    def project(self, value):
        self.output.fill_(value)
        return self.output


class DSAAttention(nn.Module):
    def __init__(self, *, implementation, max_position_embeddings, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(1))
        self.max_position_embeddings = max_position_embeddings
        self.dsa_attn = SimpleNamespace(dsa_attn=SimpleNamespace(impl=implementation()))

    def forward(self, positions):
        return positions < self.max_position_embeddings

    def project(self, value):
        return self.dsa_attn.dsa_attn.impl.project(value)


class DeepseekV4MoE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(1))
