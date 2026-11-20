# Copyright (c) 2026 Huawei Technologies Co., Ltd. Licensed under CANN OSL v2.
"""MTE registered payload geometry; mirrors arch35/common/*_peermem.h."""
from dataclasses import dataclass


CONTROL_ROW_BYTES = 64


def align(value, multiple):
    return (value + multiple - 1) // multiple * multiple


@dataclass(frozen=True)
class SplitLayout:
    attention_ranks: int
    moe_ranks: int
    experts: int
    capacity: int
    hidden: int
    top_k: int
    prefetch: bool = False
    combine_quant_mode: int = 0
    is_attention: bool = True

    def __post_init__(self):
        if (self.attention_ranks < 1 or self.moe_ranks < 1 or self.world_size > 768 or
            self.experts < self.moe_ranks or self.experts > 2048 or self.experts % self.moe_ranks or
            self.local_experts > 1024 or self.local_experts * self.world_size > 16384 or not 1 <= self.capacity < 2**24 or
            not 1024 <= self.hidden <= 8192 or self.hidden % 64 or
            not 1 <= self.top_k <= min(32, self.experts) or self.combine_quant_mode not in (0, 3, 4)):
            raise ValueError("Invalid MTE split geometry; see docs/mega_split/README.md")

    @property
    def world_size(self):
        return self.attention_ranks + self.moe_ranks

    @property
    def local_experts(self):
        return self.experts // self.moe_ranks

    @property
    def sections(self):
        route_stride = align(self.capacity * 4, 32)
        record_bytes = align(self.hidden, 256) + align((self.hidden + 31) // 32, 32)
        if self.prefetch:
            record_bytes += align(self.top_k * 4, 32)
        result_bytes = self.hidden * 2
        if self.combine_quant_mode:
            result_bytes = align(align(self.hidden, 256) + align((self.hidden + 31) // 32, 2), 32)
        prefix = (
            ("exception", 60 * 1024),
            ("control", 60 * 1024),
        )
        if not self.is_attention:
            return prefix + (
                ("routes", align(self.local_experts * self.attention_ranks * route_stride, 512)),
                ("route_task_meta", align(self.attention_ranks * 64, 512)),
                ("counts", align(self.local_experts * self.attention_ranks * 4, 512)),
            )
        return prefix + (
            ("inputs", align(self.capacity * record_bytes, 512)),
            ("results", align(self.capacity * self.top_k * result_bytes, 512)),
        )

    @property
    def control_offset(self):
        return dict(self.sections)["exception"]

    @property
    def buffer_bytes(self):
        return sum(length for _, length in self.sections)
