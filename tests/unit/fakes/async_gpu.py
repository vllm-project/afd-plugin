# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU protocol recorder for async GPU connector unit tests.

This records wire writes; it supplies no evidence about CUDA memory ordering.
"""

from collections import deque
from types import SimpleNamespace

import torch

from afd_plugin.config import parse_afd_config
from afd_plugin.connectors.gpu.async_nvshmem.connector import GpuAsyncAFDConnector
from tests.unit.fakes.vllm_config import npu_vllm_config


def recording_async_connector(*, role="attention", role_rank=0, store=None):
    config = npu_vllm_config(
        role=role,
        connector="GpuAsyncAFDConnector",
        num_attention_ranks=2,
        num_ffn_ranks=2,
        use_ubatching=True,
        num_ubatches=2,
    )
    config.model_config.dtype = torch.float32
    vars(config.model_config.hf_text_config).update(
        hidden_size=3,
        num_experts_per_tok=2,
        n_routed_experts=4,
        n_shared_experts=0,
        num_hidden_layers=4,
    )
    instance = GpuAsyncAFDConnector(
        role_rank,
        role_rank,
        config,
        parse_afd_config(config, validate=False),
        role_rank,
    )
    instance.window = RecordingAsyncWindow(store=store)
    instance._initialized = True
    instance._seq_device = torch.zeros(2, dtype=torch.int32)
    return instance


class RecordingAsyncWindow:
    def __init__(self, *, arrivals=(), replies=None, defer_headers=False, store=None):
        self.device = torch.device("cpu")
        self.defer_headers = defer_headers
        self.messages = []
        self.arrivals = deque(arrivals)
        self.replies = replies or {}
        self.events = []
        self.runtime = SimpleNamespace(
            store=store if store is not None else torch.distributed.HashStore()
        )
        self.closed = False

    def close(self):
        self.closed = True

    def write_slot(self, **fields):
        self.messages.append(
            {
                name: (
                    value
                    if name == "header" and self.defer_headers
                    else value.clone()
                    if isinstance(value, torch.Tensor)
                    else value
                )
                for name, value in fields.items()
            }
        )

    def stream_wait(self, region, ring, value):
        self.events.append(("wait", region, ring, value))

    def local_routed(self, region, ring, count):
        self.events.append(("read", region, ring))
        return self.replies[region][:count]

    def clear_flag(self, region, ring):
        self.events.append(("clear", region, ring))

    def wait(self, *, timeout_s):
        return self.arrivals.popleft() if self.arrivals else None
