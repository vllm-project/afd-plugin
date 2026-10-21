# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Plugin-owned NPU profiler helpers for AFD Ascend runners."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, replace
from types import MethodType
from typing import Final, Literal, Protocol

logger = logging.getLogger(__name__)

AFDNPUProfilerRole = Literal["attention", "ffn"]

_ENV_PREFIX: Final[dict[AFDNPUProfilerRole, str]] = {
    "attention": "AFD_NPU_ATTENTION_PROFILER",
    "ffn": "AFD_NPU_FFN_PROFILER",
}
_DEFAULT_DIR: Final[dict[AFDNPUProfilerRole, str]] = {
    "attention": "/tmp/profile/attn",
    "ffn": "/tmp/profile/ffn",
}
_DEFAULT_ACTIVE_STEPS: Final[dict[AFDNPUProfilerRole, int]] = {
    "attention": 10,
    "ffn": 20,
}
_DEFAULT_WAIT_STEPS: Final[int] = 2
_DEFAULT_WARMUP_STEPS: Final[int] = 1
_DEFAULT_REPEAT: Final[int] = 1
_DEFAULT_SKIP_FIRST_STEPS: Final[int] = 1500
_VLLM_TORCH_PROFILER_DIR_ENV: Final[str] = "VLLM_TORCH_PROFILER_DIR"


@dataclass(frozen=True)
class AFDNPUProfilerConfig:
    enabled: bool
    wait: int
    warmup: int
    active: int
    repeat: int
    skip_first: int
    trace_dir: str
    with_stack: bool


class AFDNPUProfiler(Protocol):
    def step(self) -> None: ...

    def stop(self) -> None: ...


def afd_npu_profiler_config(role: AFDNPUProfilerRole) -> AFDNPUProfilerConfig:
    """Read plugin-owned profiler settings for an AFD NPU runner role."""

    prefix = _ENV_PREFIX[role]
    return AFDNPUProfilerConfig(
        enabled=_env_bool(f"{prefix}_ENABLE", default=False),
        wait=_env_int(f"{prefix}_WAIT", default=_DEFAULT_WAIT_STEPS),
        warmup=_env_int(f"{prefix}_WARMUP", default=_DEFAULT_WARMUP_STEPS),
        active=_env_int(f"{prefix}_ACTIVE", default=_DEFAULT_ACTIVE_STEPS[role]),
        repeat=_env_int(f"{prefix}_REPEAT", default=_DEFAULT_REPEAT),
        skip_first=_env_int(
            f"{prefix}_SKIP_FIRST",
            default=_DEFAULT_SKIP_FIRST_STEPS,
        ),
        trace_dir=_env_dir(f"{prefix}_DIR", default=_DEFAULT_DIR[role]),
        with_stack=_env_bool(f"{prefix}_WITH_STACK", default=False),
    )


def create_afd_npu_profiler(role: AFDNPUProfilerRole) -> AFDNPUProfiler | None:
    """Create a torch-npu profiler when the plugin-owned env enables it."""

    config = afd_npu_profiler_config(role)
    if not config.enabled:
        return None
    return _create_npu_profiler(role, config, runtime=False)


def create_runtime_npu_profiler(
    role: AFDNPUProfilerRole,
    trace_dir: str,
    *,
    lightweight: bool = False,
    graph: bool = False,
) -> AFDNPUProfiler:
    config = replace(afd_npu_profiler_config(role), enabled=True, trace_dir=trace_dir)
    return _create_npu_profiler(
        role, config, runtime=True, lightweight=lightweight, graph=graph
    )


def _create_npu_profiler(
    role: AFDNPUProfilerRole,
    config: AFDNPUProfilerConfig,
    *,
    runtime: bool,
    lightweight: bool = False,
    graph: bool = False,
) -> AFDNPUProfiler:

    import torch_npu

    profiler_level = torch_npu.profiler.ProfilerLevel.Level2
    if lightweight:
        profiler_level = (
            torch_npu.profiler.ProfilerLevel.Level0
            if graph
            else torch_npu.profiler.ProfilerLevel.Level_none
        )
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        export_type=torch_npu.profiler.ExportType.Text,
        profiler_level=profiler_level,
        mstx=lightweight and not graph,
        mstx_domain_include=["afd.cam"] if lightweight and not graph else [],
        data_simplification=True,
        record_op_args=False,
        op_attr=False,
        l2_cache=False,
        aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
    )
    logger.info(
        "AFD NPU %s profiler enabled. Traces will be saved to: %s; with_stack=%s",
        role,
        config.trace_dir,
        config.with_stack,
    )
    profiler = torch_npu.profiler.profile(
        activities=[
            *([torch_npu.profiler.ProfilerActivity.CPU] if not lightweight else []),
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        schedule=None
        if runtime
        else torch_npu.profiler.schedule(
            wait=config.wait,
            warmup=config.warmup,
            active=config.active,
            repeat=config.repeat,
            skip_first=config.skip_first,
        ),
        with_stack=config.with_stack and not lightweight,
        with_modules=config.with_stack and not lightweight,
        record_shapes=not lightweight,
        profile_memory=False,
        with_flops=False,
        experimental_config=experimental_config,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
            config.trace_dir,
            analyse_flag=not lightweight,
        ),
    )
    if runtime and role == "ffn":
        from torch_npu.profiler._profiler_action_controller import ProfActionController

        profiler.prof_if.stop_trace = MethodType(_stop_ffn_trace, profiler.prof_if)
        # The controller captures bound methods when constructed.
        profiler.action_controller = ProfActionController(
            profiler, profiler.prof_if, profiler.on_trace_ready
        )
    profiler.start()
    if lightweight:
        from torch_npu.profiler.profiler_interface import _disable_event_record

        # NPU-only capture still installs CPU record_function on every Event query.
        _disable_event_record()
    return profiler


# TorchNPU 2.10 _ProfInterface.stop_trace waits for the permanently idle CAM recv.
# Runtime FFN's model thread instead checks the pre-receive fence before stopping.
# shortcut: pinned private API; remove when TorchNPU supports stream-scoped stop.
def _stop_ffn_trace(self):
    from torch_npu.profiler.profiler_interface import (
        FlopsHookManager,
        _disable_event_record,
        _stop_profiler,
    )

    if not self._is_env_valid:
        return
    # ### PATCH START: completed FFN work fence replaces device synchronize
    # The owner has verified completion of all work preceding the pending recv.
    # ### PATCH END: completed FFN work fence replaces device synchronize
    if self.with_flops and self.experimental_config.tx_enabled:
        FlopsHookManager.uninstall()
    _stop_profiler()
    self.stop_gc_detect()
    _disable_event_record()


def step_afd_npu_profiler(profiler: AFDNPUProfiler | None) -> None:
    if profiler is not None:
        profiler.step()


def stop_afd_npu_profiler(profiler: AFDNPUProfiler | None) -> None:
    if profiler is not None:
        profiler.stop()


def _env_bool(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value, got {value!r}")


def _env_int(name: str, *, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def _env_dir(name: str, *, default: str) -> str:
    return os.getenv(name) or os.getenv(_VLLM_TORCH_PROFILER_DIR_ENV) or default


__all__ = [
    "AFDNPUProfilerConfig",
    "AFDNPUProfiler",
    "afd_npu_profiler_config",
    "create_afd_npu_profiler",
    "create_runtime_npu_profiler",
    "step_afd_npu_profiler",
    "stop_afd_npu_profiler",
]
