# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU protocol recorder for async GPU connector unit tests.

This records wire writes; it supplies no evidence about CUDA memory ordering.
"""

from collections import deque

import torch


class RecordingAsyncWindow:
    def __init__(self, *, arrivals=(), replies=None, defer_headers=False):
        self.device = torch.device("cpu")
        self.defer_headers = defer_headers
        self.messages = []
        self.arrivals = deque(arrivals)
        self.replies = replies or {}
        self.events = []

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
