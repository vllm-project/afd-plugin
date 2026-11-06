# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Metadata fixtures and host-only lifecycle stand-ins for standalone FFN."""

from __future__ import annotations

import atexit
import json
import os
import sys
import threading
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

from afd_plugin.ffn_faas.interfaces import FFNLaunch
from afd_plugin.ffn_faas.service import FFNService, ServicePhase


@dataclass(frozen=True)
class MetadataModelAdapter:
    """CPU loading-boundary facts, never a qualified model or device stand-in."""

    root: Path
    layer_indices: tuple[int, ...] = (2, 7)
    num_experts: int = 4

    def estimated_parameter_bytes(self, local_experts: int) -> int:
        return 1024 * local_experts

    def local_sources(self, layer_index, owned):
        raise RuntimeError("metadata-only adapter has no checkpoint tensors")

    def create_owner(self, config):
        raise RuntimeError("metadata-only adapter has no native parameter owner")


def write_v4_checkpoint(root: Path, *, layers: int = 2, experts: int = 4) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    config = {
        "architectures": ["DeepseekV4ForCausalLM"],
        "hidden_act": "silu",
        "scoring_func": "sqrtsoftplus",
        "torch_dtype": "bfloat16",
        "num_hidden_layers": layers,
        "n_routed_experts": experts,
        "num_experts_per_tok": 2,
        "num_hash_layers": 1,
        "hidden_size": 256,
        "moe_intermediate_size": 256,
        "routed_scaling_factor": 1.5,
        "swiglu_limit": 10.0,
    }
    shard = "weights.safetensors"
    weights = {
        f"layers.{layer}.ffn.experts.{expert}.w{projection}.{suffix}": shard
        for layer in range(layers)
        for expert in range(experts)
        for projection in (1, 2, 3)
        for suffix in ("weight", "weight_scale", "weight_offset", "scale_bias")
    }
    (root / "config.json").write_text(json.dumps(config))
    (root / "quant_model_weights.safetensors.index.json").write_text(
        json.dumps({"weight_map": weights})
    )
    (root / "quant_model_description.json").write_text(
        json.dumps({"group_size": 0, **dict.fromkeys(weights, "W4A8_DYNAMIC")})
    )
    # Metadata tests never pass this placeholder to the native loader.
    (root / "weights.safetensors").write_bytes(b"metadata-only fixture")
    return root


class HostBackend:
    """Exercise host ownership rules, without simulating device completion."""

    def __init__(
        self, *, stop_during_prepare: bool = False, fail: str = "", block: bool = False
    ) -> None:
        self.stop_during_prepare = stop_during_prepare
        self.fail = fail
        self.block = block
        self.entered = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()
        self.released = False
        self.service: FFNService | None = None

    def prepare(self, service: FFNService) -> None:
        self.service = service
        if self.fail == "prepare":
            raise RuntimeError("prepare failed")
        service.phase(ServicePhase.CONNECTING, model_loaded=True)
        if self.stop_during_prepare:
            service.request_stop()

    def start_receiver(self) -> None:
        if self.fail == "activate":
            raise RuntimeError("receiver activation failed")

    def execute_step(self) -> None:
        assert self.service is not None
        self.entered.set()
        if self.block:
            self.release.wait()
        if self.fail == "execute":
            self.finished.set()
            raise RuntimeError("execute failed")
        self.service.request_stop()

    def receiver_exited(self) -> None:
        self.finished.set()

    def close(self) -> None:
        if not self.finished.is_set():
            raise RuntimeError("host receiver has not exited")
        if self.fail == "close":
            raise RuntimeError("host-only resource release failed")
        self.released = True


class HostPlatformAdapter:
    """An independent CPU host contract; it has no model/device requirements."""

    def add_arguments(self, parser) -> None:
        pass

    def configure(self, args) -> FFNLaunch:
        return FFNLaunch(
            rank=3,
            local_rank=0,
            model_path=str(args.model),
            backend_name="host-contract",
            shutdown_timeout=args.shutdown_timeout,
            backend_factory=HostBackend,
        )


def run_cpu_platform(model: str, status: str) -> int:
    """Run the actual service/loop after the child entry installed its import gate."""
    from afd_plugin.ffn_faas import __main__ as cli

    args = [
        "ffn_faas",
        "serve",
        model,
        "--backend",
        "host-contract",
        "--status-port",
        "0",
        "--status-file",
        status,
        "--shutdown-timeout",
        "0.1",
    ]
    with (
        patch.object(cli, "SUPPORTED_BACKENDS", ("host-contract",)),
        patch.object(cli, "get_platform_adapter", lambda _: HostPlatformAdapter()),
        patch.object(sys, "argv", args),
    ):
        return cli.main()


def run_cli_host_case(model: str, marker: str, fault: str, released: bool) -> int:
    """Run the real CLI exit path in a child; no native backend is constructed.

    The atexit marker distinguishes normal interpreter teardown from os._exit.
    Faults start only after the host-only runner has reported its outcome.
    """
    from afd_plugin.ffn_faas import __main__ as cli

    atexit.register(lambda: Path(marker).write_text("ordinary shutdown"))
    os.environ.update(
        RANK="0",
        LOCAL_RANK="0",
        WORLD_SIZE="1",
        LOCAL_WORLD_SIZE="1",
        VLLM_USE_V2_MODEL_RUNNER="1",
        AFD_ASYNC_CAM_LAYERED_GMM="1",
    )
    args = [
        "ffn_faas",
        "serve",
        model,
        "--attention-ranks",
        "1",
        "--attention-tp",
        "1",
        "--ffn-ranks",
        "1",
        "--status-port",
        "0",
    ]
    if fault == "status_file":
        blocked = Path(marker).with_name("blocked-parent")
        blocked.write_text("not a directory")
        args += ["--status-file", str(blocked / "status.json")]

    with ExitStack() as patches:

        class FailedOutput:
            def __init__(self, original, operation):
                self.original, self.operation = original, operation

            def write(self, text):
                if self.operation == "write":
                    raise BrokenPipeError("host-only reporting fault")
                return self.original.write(text)

            def flush(self):
                if self.operation == "flush":
                    raise BrokenPipeError("host-only flush fault")
                return self.original.flush()

        class HostService(FFNService):
            def snapshot(self) -> dict[str, str | int | bool]:
                if fault == "snapshot":
                    raise OSError("host-only snapshot fault")
                return super().snapshot()

            def close_http(self) -> None:
                super().close_http()
                if fault == "close_http":
                    raise OSError("host-only HTTP close fault")

        class HostRunner:
            def __init__(self, service, backend_factory, *, shutdown_timeout):
                self.service = service

            def run(self):
                if released:
                    self.service.phase(ServicePhase.STOPPED, resources_released=True)
                else:
                    self.service.phase(ServicePhase.FAILED, error="host-only failure")
                if fault.startswith(("stdout_", "stderr_")):
                    stream_name, operation = fault.split("_", 1)
                    original = sys.stdout if stream_name == "stdout" else sys.stderr
                    patches.enter_context(
                        patch.object(
                            sys, stream_name, FailedOutput(original, operation)
                        )
                    )
                if not released and fault != "returned_unreleased":
                    raise RuntimeError("host-only unreleased session")

        patches.enter_context(patch.object(cli, "FFNService", HostService))
        patches.enter_context(patch.object(cli, "FFNModelRunner", HostRunner))
        patches.enter_context(patch.object(sys, "argv", args))
        return cli.main()
