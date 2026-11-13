# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Immutable declarations exchanged by model entries and the E2E executor.

Importing declarations does not resolve checkpoints or import a device runtime.
New fields represent shared capabilities and require their own review.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeAlias

JsonValue: TypeAlias = (
    str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
)

EXECUTION_EAGER = "eager"
EXECUTION_FULL_DECODE_GRAPH = "full_decode_graph"
CLEANUP_PROFILE_STRICT = "strict"
CLEANUP_PROFILE_NPU_ASYNC_CAM_LEGACY = "npu_async_cam_legacy"
BASELINE_ROLE = "baseline"
ATTENTION_ROLE = "attention"
FFN_ROLE = "ffn"
DEFAULT_GENERATED_TOKENS = 512
DEFAULT_FEWSHOT = 8
DEFAULT_API_PORT = 8000
DEFAULT_CONNECTOR_PORT = 1239
DEFAULT_STARTUP_TIMEOUT_S = 900
DEFAULT_SHUTDOWN_TIMEOUT_S = 10
DEFAULT_TERMINATION_TIMEOUT_S = 20
WORKER_SHUTDOWN_TIMEOUT_ENV = "VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS"
NPU_ASYNC_VLLM_SHUTDOWN_TIMEOUT_S = 90
NPU_ASYNC_WORKER_SHUTDOWN_TIMEOUT_S = 60
NPU_ASYNC_TERMINATION_TIMEOUT_S = 120


@dataclass(frozen=True)
class Role:
    dp: int
    tp: int = 1
    execution: str = EXECUTION_EAGER
    dbo: DboConfig | None = None
    vllm_args: tuple[str, ...] = ()
    additional_config: tuple[tuple[str, bool], ...] = ()

    def __post_init__(self) -> None:
        if self.dp < 1 or self.tp < 1:
            raise ValueError("DP and TP must be positive")
        if self.execution not in (EXECUTION_EAGER, EXECUTION_FULL_DECODE_GRAPH):
            raise ValueError(f"unknown execution mode {self.execution!r}")

    @property
    def rank_count(self) -> int:
        return self.dp * self.tp

    @property
    def uses_graph(self) -> bool:
        return self.execution == EXECUTION_FULL_DECODE_GRAPH


@dataclass(frozen=True)
class DboConfig:
    # None keeps the runtime default, as the existing async Attention path does.
    decode_token_threshold: int | None = None
    prefill_token_threshold: int | None = None


@dataclass(frozen=True)
class Deployment:
    backend: str
    attention: Role
    ffn: Role | None = None
    connector: str | None = None
    cleanup_profile: str = CLEANUP_PROFILE_STRICT
    dbo: DboConfig | None = None
    cudagraph_capture_size: int | None = None
    max_num_seqs: int | None = None
    max_num_batched_tokens: int | None = None
    compute_gate_on_attention: bool = False
    connector_extra_config: tuple[tuple[str, int], ...] = ()
    pin_attention_address: bool = False
    compilation_config: tuple[tuple[str, str | tuple[int, ...]], ...] = ()
    shutdown_timeout_s: int = DEFAULT_SHUTDOWN_TIMEOUT_S
    # Existing worker environment overrides take precedence over this default.
    worker_shutdown_timeout_s: int | None = None
    termination_timeout_s: float = DEFAULT_TERMINATION_TIMEOUT_S

    @property
    def is_baseline(self) -> bool:
        return self.ffn is None

    @property
    def roles(self) -> tuple[tuple[str, Role], ...]:
        if self.ffn is None:
            return ((BASELINE_ROLE, self.attention),)
        return ((ATTENTION_ROLE, self.attention), (FFN_ROLE, self.ffn))

    def dbo_for_role(self, role: str) -> DboConfig | None:
        """A role's DBO overrides the deployment default; None inherits it."""
        override = dict(self.roles)[role].dbo
        return self.dbo if override is None else override


@dataclass(frozen=True)
class Gsm8kSettings:
    sample_limit: int | None
    threshold: float
    num_fewshot: int = DEFAULT_FEWSHOT
    num_concurrent: int | None = None
    max_gen_toks: int = DEFAULT_GENERATED_TOKENS
    minimum_samples: int = 1
    # Existing smoke profiles raise their threshold for larger workloads.
    smoke_max_samples: int | None = None
    smoke_threshold: float | None = None
    larger_sample_threshold: float | None = None


@dataclass(frozen=True)
class Gsm8kTask:
    settings: Gsm8kSettings

    def resolve(
        self, environment: Mapping[str, str]
    ) -> tuple[Gsm8kTask, tuple[str, ...]]:
        from tests.e2e.accuracy.gsm8k import resolve_settings

        settings, reasons = resolve_settings(self.settings, environment)
        return Gsm8kTask(settings), reasons

    def execute(
        self, *, base_url: str, model_name: str, checkpoint: str, output_path: Path
    ) -> dict[str, JsonValue]:
        from tests.e2e.accuracy.gsm8k import evaluate_settings

        return evaluate_settings(
            self.settings,
            base_url=base_url,
            model_name=model_name,
            checkpoint=checkpoint,
            output_path=output_path,
        )


class Task(Protocol):
    """A serializable task declaration, resolved and executed in the child."""

    def resolve(
        self, environment: Mapping[str, str]
    ) -> tuple[Task, tuple[str, ...]]: ...

    def execute(
        self, *, base_url: str, model_name: str, checkpoint: str, output_path: Path
    ) -> dict[str, JsonValue]: ...


@dataclass(frozen=True)
class Mrv2Check:
    roles: tuple[str, ...]
    graph: bool = False
    dbo: bool = False


@dataclass(frozen=True)
class HashInputCheck:
    roles: tuple[str, ...] = (ATTENTION_ROLE, FFN_ROLE)
    graph: bool = False
    dbo: bool = False
    compare_with: str | None = None


@dataclass(frozen=True)
class Phase:
    name: str
    task: Task
    checks: tuple[Mrv2Check | HashInputCheck, ...] = ()


@dataclass(frozen=True)
class Case:
    id: str
    model: str
    deployment: Deployment
    phases: tuple[Phase, ...]
    served_model_name_prefix: str
    common_vllm_args: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.id or not self.phases:
            raise ValueError("Case requires an ID and at least one phase")
        names = [phase.name for phase in self.phases]
        if len(names) != len(set(names)):
            raise ValueError("phase names must be unique within a Case")


@dataclass(frozen=True)
class CaseEnvironment:
    model: str
    vllm_bin: str
    device_backend: str
    devices: tuple[str, ...]
    output_path: Path
    api_host: str = "127.0.0.1"
    api_port_base: int = DEFAULT_API_PORT
    afd_host: str = "127.0.0.1"
    afd_port: int = DEFAULT_CONNECTOR_PORT
    startup_timeout: float = DEFAULT_STARTUP_TIMEOUT_S

    def __post_init__(self) -> None:
        if len(self.devices) != len(set(self.devices)):
            raise ValueError("devices must be unique")
