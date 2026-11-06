# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""In-memory Store and rank world for the host startup protocol."""

from threading import Condition

from afd_plugin.connectors.npu.async_cam_runtime.async_cam_startup import (
    AsyncCAMGraphStartup,
)


class MemoryStore:
    def __init__(self):
        self.values = {}
        self.condition = Condition()
        self.closed = False

    def set(self, key, value):
        assert not self.closed, "Store used after release"
        with self.condition:
            self.values[key] = value.encode() if isinstance(value, str) else value
            self.condition.notify_all()

    def get(self, key):
        assert not self.closed, "Store used after release"
        with self.condition:
            assert self.condition.wait_for(lambda: key in self.values, timeout=3)
            return self.values[key]

    def check(self, keys):
        assert not self.closed, "Store used after release"
        with self.condition:
            return all(key in self.values for key in keys)


def startup_world(attn=2, ffn=2, store=None):
    store = store or MemoryStore()
    owners = []
    for rank in range(attn + ffn):
        owner = AsyncCAMGraphStartup(rank, attn, attn + ffn)
        owner._store = store
        owner._prefix = "ffn_graph/test/"
        owners.append(owner)
    return owners
