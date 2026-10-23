# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU-safe connector settings for the independent FFN service."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from afd_plugin.config_utils import coerce_extra_positive_int


@dataclass(frozen=True)
class FaaSOptions:
    attn_ranks_per_dp: int = 1
    capacity: int = 128
    slots: int = 2
    hccl_buffer_size: int = 64
    timeout_s: int = 30
    startup_timeout_s: int = 300

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> FaaSOptions:
        raw = {} if raw is None else raw
        unknown = set(raw) - cls.__dataclass_fields__.keys()
        if unknown:
            raise ValueError(f"unknown FFN FaaS settings: {sorted(unknown)}")
        return cls(
            **{
                name: coerce_extra_positive_int(value, field_name=name)
                for name, value in raw.items()
            }
        )

    def to_mapping(self) -> dict[str, int]:
        return asdict(self)
