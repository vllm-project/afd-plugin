# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Shared E2E test infrastructure – server lifecycle, launch helpers, port utils."""

from __future__ import annotations

import json
import subprocess
import threading
import time
import urllib.error
import urllib.request
from typing import Any

from tests.e2e.runner import (
    build_env,
    build_vllm_command,
    served_model_name,
    start_process,
    stream_output,
    terminate_processes,
)

# ---------------------------------------------------------------------------
# Helper: lightweight namespace to mimic runner's argparse result
# ---------------------------------------------------------------------------


def _make_args(
    *,
    model: str,
    vllm_bin: str = "vllm",
    num_attention_ranks: int = 1,
    num_ffn_ranks: int = 1,
    api_host: str = "127.0.0.1",
    api_port_base: int = 8000,
    afd_host: str = "127.0.0.1",
    afd_port: int = 1239,
    startup_timeout: float = 900,
    served_model_name_prefix: str = "deepseek-v2-lite-afd",
    cuda_graph_full_decode_only: bool = False,
    cudagraph_capture_size: int = 64,
    enable_dbo: bool = False,
    common_vllm_args: list[str] | None = None,
    attention_vllm_args: list[str] | None = None,
    ffn_vllm_args: list[str] | None = None,
    use_decode_bench_connector: bool = False,
    dbo_decode_token_threshold: int = 1,
    dbo_prefill_token_threshold: int | None = None,
    tp_size: int = 1,
    attention_tp_size: int | None = None,
    ffn_tp_size: int | None = None,
    afd_connector: str | None = None,
    afd_async: bool = False,
    compute_gate_on_attention: bool = False,
    afd_connector_extra_config: list[str] | None = None,
    device_backend: str = "gpu",
) -> dict[str, Any]:
    """Build a dict mimicking the runner's argparse Namespace."""
    return {
        "model": model,
        "vllm_bin": vllm_bin,
        "num_attention_ranks": num_attention_ranks,
        "num_ffn_ranks": num_ffn_ranks,
        "api_host": api_host,
        "api_port_base": api_port_base,
        "afd_host": afd_host,
        "afd_port": afd_port,
        "startup_timeout": startup_timeout,
        "served_model_name_prefix": served_model_name_prefix,
        "cuda_graph_full_decode_only": cuda_graph_full_decode_only,
        "cudagraph_capture_size": cudagraph_capture_size,
        "enable_dbo": enable_dbo,
        "dbo_decode_token_threshold": dbo_decode_token_threshold,
        "dbo_prefill_token_threshold": dbo_prefill_token_threshold,
        "use_decode_bench_connector": use_decode_bench_connector,
        "common_vllm_arg": common_vllm_args or [],
        "attention_vllm_arg": attention_vllm_args or [],
        "ffn_vllm_arg": ffn_vllm_args or [],
        "prompt": "",
        "max_tokens": 16,
        "temperature": 0.0,
        "num_requests": None,
        "request_concurrency": None,
        "tp_size": tp_size,
        "attention_tp_size": attention_tp_size,
        "ffn_tp_size": ffn_tp_size,
        "afd_connector": afd_connector,
        "afd_async": afd_async,
        "compute_gate_on_attention": compute_gate_on_attention,
        "afd_connector_extra_config": afd_connector_extra_config or [],
        "device_backend": device_backend,
    }


class _ArgsNs:
    """Simple object-from-dict namespace for runner functions."""

    def __init__(self, d: dict[str, Any]) -> None:
        self.__dict__.update(d)


# ---------------------------------------------------------------------------
# Unified AFD server management
# ---------------------------------------------------------------------------


class AFDServer:
    """Manages a running pair of AFD attention + FFN servers (GPU or NPU)."""

    def __init__(
        self,
        base_url: str,
        attention_port: int,
        served_model: str,
        processes: list[subprocess.Popen[str]],
        log_threads: list[threading.Thread],
    ) -> None:
        self.base_url = base_url
        self.attention_port = attention_port
        self.served_model = served_model
        self._processes = processes
        self._log_threads = log_threads

    def request_completion(
        self,
        prompt: str,
        *,
        max_tokens: int = 16,
        temperature: float = 0.0,
        stream: bool = False,
        logprobs: int | None = None,
    ) -> dict[str, Any]:
        """Send a /v1/completions request and return the parsed JSON body."""
        url = f"{self.base_url}/v1/completions"
        payload: dict[str, Any] = {
            "model": self.served_model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }
        if logprobs is not None:
            payload["logprobs"] = logprobs

        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace") or "<empty>"
            raise RuntimeError(
                f"HTTP {exc.code} {exc.reason}: {body}",
            ) from exc

    def shutdown(self) -> None:
        terminate_processes(self._processes)
        for thread in self._log_threads:
            thread.join(timeout=2)


# ---------------------------------------------------------------------------
# Process helpers
# ---------------------------------------------------------------------------


def _ensure_alive(process: subprocess.Popen[str], message: str) -> None:
    rc = process.poll()
    if rc is not None:
        raise RuntimeError(f"{message} (returncode={rc})")


# ---------------------------------------------------------------------------
# Unified AFD server launcher
# ---------------------------------------------------------------------------


def _launch_afd_server(
    model: str,
    *,
    backend: str,
    vllm_bin: str = "vllm",
    attention_devices: list[str] | None = None,
    ffn_devices: list[str] | None = None,
    api_port_base: int = 8000,
    afd_port: int = 1239,
    startup_timeout: float = 900,
    cuda_graph_full_decode_only: bool = False,
    cudagraph_capture_size: int = 64,
    enable_dbo: bool = False,
    common_vllm_args: list[str] | None = None,
    served_model_name_prefix: str = "deepseek-v2-lite-afd",
    connector: str | None = None,
    attention_tp_size: int | None = None,
    ffn_tp_size: int | None = None,
    afd_async: bool = False,
    compute_gate_on_attention: bool = False,
    afd_connector_extra_config: list[str] | None = None,
    attention_env: dict[str, str] | None = None,
    ffn_env: dict[str, str] | None = None,
) -> AFDServer:
    """Start AFD servers and return an AFDServer once the API is ready.

    Parameters
    ----------
    backend:
        ``"gpu"`` uses CUDA workers with ``P2pNcclAFDConnector``.
        ``"npu"`` uses Ascend workers with ``CAMP2pAFDConnector`` by default;
        pass ``connector`` to override it.
    attention_devices:
        Device IDs for the attention worker (e.g. ``["0"]``).
    ffn_devices:
        Device IDs for the FFN worker (e.g. ``["1"]``).
    attention_env:
        Extra environment variables applied only to the Attention process.
    ffn_env:
        Extra environment variables applied only to the FFN process.
    """
    attention_devices = attention_devices or ["0"]
    ffn_devices = ffn_devices or ["1"]

    is_npu = backend == "npu"

    args = _ArgsNs(
        _make_args(
            model=model,
            vllm_bin=vllm_bin,
            num_attention_ranks=len(attention_devices),
            num_ffn_ranks=len(ffn_devices),
            api_port_base=api_port_base,
            afd_port=afd_port,
            startup_timeout=startup_timeout,
            cuda_graph_full_decode_only=cuda_graph_full_decode_only,
            cudagraph_capture_size=cudagraph_capture_size,
            enable_dbo=enable_dbo,
            common_vllm_args=common_vllm_args,
            served_model_name_prefix=served_model_name_prefix,
            device_backend=backend,
            attention_tp_size=attention_tp_size,
            ffn_tp_size=ffn_tp_size,
            afd_connector=connector,
            afd_async=afd_async,
            compute_gate_on_attention=compute_gate_on_attention,
            afd_connector_extra_config=afd_connector_extra_config,
        ),
    )

    processes: list[subprocess.Popen[str]] = []
    log_threads: list[threading.Thread] = []

    try:
        # --- FFN ---
        ffn_cmd = build_vllm_command(args, role="ffn")
        ffn_devices_str = ",".join(ffn_devices)
        device_label = (
            f"ASCEND_RT_VISIBLE_DEVICES={ffn_devices_str}"
            if is_npu
            else f"CUDA_VISIBLE_DEVICES={ffn_devices_str}"
        )
        print(f"\n[conftest] Starting FFN ({device_label})")
        ffn_process_env = build_env(ffn_devices_str, args, role="ffn")
        ffn_process_env.update(ffn_env or {})
        ffn_proc = start_process(
            "ffn",
            ffn_cmd,
            ffn_process_env,
        )
        processes.append(ffn_proc)
        log_threads.append(stream_output("ffn", ffn_proc))

        _ensure_alive(ffn_proc, "FFN process exited during startup")

        # --- Attention ---
        attn_cmd = build_vllm_command(args, role="attention")
        attn_devices_str = ",".join(attention_devices)
        device_label = (
            f"ASCEND_RT_VISIBLE_DEVICES={attn_devices_str}"
            if is_npu
            else f"CUDA_VISIBLE_DEVICES={attn_devices_str}"
        )
        print(f"[conftest] Starting ATTN ({device_label})")
        attn_process_env = build_env(attn_devices_str, args, role="attention")
        attn_process_env.update(attention_env or {})
        attn_proc = start_process(
            "attention",
            attn_cmd,
            attn_process_env,
        )
        processes.append(attn_proc)
        log_threads.append(stream_output("attention", attn_proc))

        _ensure_alive(attn_proc, "Attention process exited during startup")

        # --- Wait for API ---
        deadline = time.monotonic() + startup_timeout
        api_url = f"http://{args.api_host}:{api_port_base}/v1/models"
        while time.monotonic() < deadline:
            # Fail fast if a worker died while waiting for the API to come up;
            # otherwise we'd burn the whole startup_timeout polling a dead process.
            _ensure_alive(ffn_proc, "FFN process died while waiting for API")
            _ensure_alive(attn_proc, "Attention process died while waiting for API")
            try:
                with urllib.request.urlopen(api_url, timeout=5) as resp:
                    if resp.status == 200:
                        print(f"[conftest] API ready at {api_url}")
                        break
            except (OSError, urllib.error.URLError):
                pass
            time.sleep(2)
        else:
            raise TimeoutError(
                f"API not ready at {api_url} after {startup_timeout}s",
            )

        return AFDServer(
            base_url=f"http://{args.api_host}:{api_port_base}",
            attention_port=api_port_base,
            served_model=served_model_name(args, "attention"),
            processes=processes,
            log_threads=log_threads,
        )
    except BaseException:
        terminate_processes(processes)
        for thread in log_threads:
            thread.join(timeout=2)
        raise
