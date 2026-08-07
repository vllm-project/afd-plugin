# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Opt-in CUDA eager correctness test for Qwen3.6 MoE AFD.

Set ``AFD_QWEN3_6_E2E_MODEL`` to an original BF16 checkpoint. The test runs a
native TP2 oracle on GPUs 0,1 and then an AFD 2A2F TP2 stack on GPUs 0,1/2,3.
It requires exact generated token IDs, top-5 token sets, and logprobs.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

PROMPT = "Explain why reproducibility matters in systems research."
ALTERNATE_PROMPT = "Summarize the role of deterministic tests in one sentence."
MODEL_NAME = "qwen36-afd"
NATIVE_PORT = 18080
ATTENTION_PORT = 18081
FFN_PORT = 18082
AFD_PORT = 6280


@dataclass(frozen=True)
class _ServerProcess:
    process: subprocess.Popen[str]
    log_path: Path


def _devices() -> list[str]:
    return [
        device.strip()
        for device in os.environ.get("AFD_QWEN3_6_E2E_GPUS", "0,1,2,3").split(",")
        if device.strip()
    ]


def _model_path() -> str:
    model = os.environ.get("AFD_QWEN3_6_E2E_MODEL")
    if not model:
        pytest.skip("set AFD_QWEN3_6_E2E_MODEL to run Qwen3.6 CUDA E2E")
    return model


def _common_args(
    model: str,
    *,
    tp_size: int,
    data_parallel_size: int = 1,
    use_cuda_graph: bool = False,
    enable_dbo: bool = False,
    enable_expert_parallel: bool = False,
    all2all_backend: str | None = None,
) -> list[str]:
    args = [
        os.environ.get("AFD_GPU_E2E_VLLM_BIN", "vllm"),
        "serve",
        model,
        "--tensor-parallel-size",
        str(tp_size),
        "--data-parallel-size",
        str(data_parallel_size),
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "4096",
        "--language-model-only",
        "--generation-config",
        "vllm",
        "--seed",
        "0",
        "--disable-cascade-attn",
        "--moe-backend",
        "triton",
    ]
    if use_cuda_graph:
        args.extend(
            [
                "--max-num-seqs",
                "64",
                "--max-num-batched-tokens",
                "64",
                "--max-cudagraph-capture-size",
                "64",
                "--cudagraph-capture-sizes",
                "64",
                "--compilation-config",
                '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}',
            ],
        )
    else:
        args.append("--enforce-eager")
    if enable_dbo:
        args.extend(
            [
                "--enable-dbo",
                "--dbo-decode-token-threshold",
                "2",
                "--dbo-prefill-token-threshold",
                "12",
            ],
        )
    args.append(
        "--enable-expert-parallel"
        if enable_expert_parallel
        else "--no-enable-expert-parallel",
    )
    if all2all_backend is not None:
        args.extend(["--all2all-backend", all2all_backend])
    return args


def _launch(command: list[str], devices: list[str], log_path: Path) -> _ServerProcess:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
    env["VLLM_PLUGINS"] = "afd"
    env["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    env["PYTHONUNBUFFERED"] = "1"
    env.pop("HTTP_PROXY", None)
    env.pop("HTTPS_PROXY", None)
    env.pop("ALL_PROXY", None)
    with log_path.open("w", encoding="utf-8") as output:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
    return _ServerProcess(process=process, log_path=log_path)


def _log_tail(log_path: Path, limit: int = 4096) -> str:
    with log_path.open("rb") as log:
        log.seek(0, os.SEEK_END)
        log.seek(max(0, log.tell() - limit))
        return log.read().decode(errors="replace")


def _wait_for_api(port: int, servers: list[_ServerProcess]) -> None:
    deadline = time.monotonic() + 900
    url = f"http://127.0.0.1:{port}/v1/models"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    while time.monotonic() < deadline:
        for server in servers:
            process = server.process
            if process.poll() is not None:
                raise RuntimeError(
                    f"vLLM exited during startup; log={server.log_path}:\n"
                    f"{_log_tail(server.log_path)}"
                )
        try:
            with opener.open(url, timeout=5) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            time.sleep(2)
    tails = "\n".join(
        f"{server.log_path}:\n{_log_tail(server.log_path)}" for server in servers
    )
    raise TimeoutError(f"timed out waiting for {url}; log tails:\n{tails}")


def _request(
    port: int,
    *,
    batch_size: int = 1,
    prompt: str = PROMPT,
) -> dict[str, Any]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt if batch_size == 1 else [prompt] * batch_size,
        "max_tokens": int(os.environ.get("AFD_QWEN3_6_E2E_MAX_TOKENS", "8")),
        "temperature": 0,
        "top_p": 1.0,
        "seed": 0,
        "logprobs": 5,
        "return_tokens_as_token_ids": True,
        "return_token_ids": True,
    }
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=300) as response:
        return json.load(response)


def _stop(servers: list[_ServerProcess]) -> None:
    for server in reversed(servers):
        process = server.process
        if process.poll() is None:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
    for server in reversed(servers):
        process = server.process
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=30)
        if process.poll() is None:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


def _assert_exact(oracle: dict[str, Any], candidate: dict[str, Any]) -> None:
    oracle_choices = oracle["choices"]
    candidate_choices = candidate["choices"]
    assert len(oracle_choices) == len(candidate_choices)
    for expected, actual in zip(oracle_choices, candidate_choices, strict=True):
        assert expected["token_ids"] == actual["token_ids"]
        for step, (expected_top5, actual_top5) in enumerate(
            zip(
                expected["logprobs"]["top_logprobs"],
                actual["logprobs"]["top_logprobs"],
                strict=True,
            )
        ):
            assert set(expected_top5) == set(actual_top5)
            for token_id, expected_logprob in expected_top5.items():
                assert actual_top5[token_id] == expected_logprob, (
                    f"step={step}, token_id={token_id}, "
                    f"expected={expected_logprob}, actual={actual_top5[token_id]}"
                )


def _afd_config(
    *,
    role: str,
    num_attention_ranks: int,
    num_ffn_ranks: int,
    port: int,
    compute_gate_on_attention: bool,
) -> str:
    return json.dumps(
        {
            "afd": {
                "role": role,
                "connector": "P2pNcclAFDConnector",
                "host": "127.0.0.1",
                "port": port,
                "num_attention_ranks": num_attention_ranks,
                "num_ffn_ranks": num_ffn_ranks,
                "compute_gate_on_attention": compute_gate_on_attention,
            }
        },
        separators=(",", ":"),
    )


@pytest.mark.gpu
@pytest.mark.e2e
@pytest.mark.slow
def test_qwen3_6_afd_2a2f_tp2_eager_matches_native_tp2(tmp_path: Path):
    """Check batch-two/repeat/A-B-A state isolation with Attention routing."""
    devices = _devices()
    if len(devices) < 4:
        pytest.skip(f"Qwen3.6 2A2F TP2 requires 4 GPUs; got {len(devices)}")
    model = _model_path()
    native: list[_ServerProcess] = []
    afd: list[_ServerProcess] = []
    try:
        native_command = [
            *_common_args(model, tp_size=2),
            "--served-model-name",
            MODEL_NAME,
            "--host",
            "127.0.0.1",
            "--port",
            str(NATIVE_PORT),
        ]
        native.append(_launch(native_command, devices[:2], tmp_path / "native.log"))
        _wait_for_api(NATIVE_PORT, native)
        oracle_batch2 = _request(NATIVE_PORT, batch_size=2)
        oracle_a = _request(NATIVE_PORT)
        oracle_b = _request(NATIVE_PORT, prompt=ALTERNATE_PROMPT)
    finally:
        _stop(native)

    ffn_config = _afd_config(
        role="ffn",
        num_attention_ranks=2,
        num_ffn_ranks=2,
        port=AFD_PORT,
        compute_gate_on_attention=True,
    )
    attention_config = _afd_config(
        role="attention",
        num_attention_ranks=2,
        num_ffn_ranks=2,
        port=AFD_PORT,
        compute_gate_on_attention=True,
    )
    try:
        ffn_command = [
            *_common_args(model, tp_size=2),
            "--served-model-name",
            MODEL_NAME,
            "--host",
            "127.0.0.1",
            "--port",
            str(FFN_PORT),
            "--additional-config",
            ffn_config,
        ]
        attention_command = [
            *_common_args(model, tp_size=2),
            "--served-model-name",
            MODEL_NAME,
            "--host",
            "127.0.0.1",
            "--port",
            str(ATTENTION_PORT),
            "--additional-config",
            attention_config,
        ]
        afd.append(_launch(ffn_command, devices[2:4], tmp_path / "ffn.log"))
        afd.append(_launch(attention_command, devices[:2], tmp_path / "attention.log"))
        _wait_for_api(ATTENTION_PORT, afd)
        _assert_exact(oracle_batch2, _request(ATTENTION_PORT, batch_size=2))
        _assert_exact(oracle_a, _request(ATTENTION_PORT))
        _assert_exact(
            oracle_b,
            _request(ATTENTION_PORT, prompt=ALTERNATE_PROMPT),
        )
        _assert_exact(oracle_a, _request(ATTENTION_PORT))
    finally:
        _stop(afd)


@pytest.mark.gpu
@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.parametrize("compute_gate_on_attention", [True, False])
def test_qwen3_6_afd_1a1f_eager_matches_native_tp1(
    tmp_path: Path,
    compute_gate_on_attention: bool,
):
    """Exercise both Qwen routing placements at the native experts boundary."""
    devices = _devices()
    if len(devices) < 2:
        pytest.skip(f"Qwen3.6 1A1F requires 2 GPUs; got {len(devices)}")
    model = _model_path()
    suffix = "attention_gate" if compute_gate_on_attention else "ffn_gate"
    native_port = 18180 if compute_gate_on_attention else 18183
    attention_port = native_port + 1
    ffn_port = native_port + 2
    afd_port = 6380 if compute_gate_on_attention else 6381
    native: list[_ServerProcess] = []
    afd: list[_ServerProcess] = []
    try:
        native.append(
            _launch(
                [
                    *_common_args(model, tp_size=1),
                    "--served-model-name",
                    MODEL_NAME,
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(native_port),
                ],
                devices[:1],
                tmp_path / f"native_{suffix}.log",
            ),
        )
        _wait_for_api(native_port, native)
        oracle = _request(native_port)
    finally:
        _stop(native)

    try:
        ffn_config = _afd_config(
            role="ffn",
            num_attention_ranks=1,
            num_ffn_ranks=1,
            port=afd_port,
            compute_gate_on_attention=compute_gate_on_attention,
        )
        attention_config = _afd_config(
            role="attention",
            num_attention_ranks=1,
            num_ffn_ranks=1,
            port=afd_port,
            compute_gate_on_attention=compute_gate_on_attention,
        )
        afd.append(
            _launch(
                [
                    *_common_args(model, tp_size=1),
                    "--served-model-name",
                    MODEL_NAME,
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(ffn_port),
                    "--additional-config",
                    ffn_config,
                ],
                devices[1:2],
                tmp_path / f"ffn_{suffix}.log",
            ),
        )
        afd.append(
            _launch(
                [
                    *_common_args(model, tp_size=1),
                    "--served-model-name",
                    MODEL_NAME,
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(attention_port),
                    "--additional-config",
                    attention_config,
                ],
                devices[:1],
                tmp_path / f"attention_{suffix}.log",
            ),
        )
        _wait_for_api(attention_port, afd)
        _assert_exact(oracle, _request(attention_port))
        _assert_exact(oracle, _request(attention_port))
    finally:
        _stop(afd)


@pytest.mark.gpu
@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.parametrize(
    ("enable_expert_parallel", "batch_size"),
    [(False, 1), (False, 2), (True, 1)],
    ids=["tp2-b1", "tp2-b2", "tp2ep2-b1"],
)
def test_qwen3_6_afd_2a2f_tp2ep2_graph_matches_native(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    batch_size: int,
    enable_expert_parallel: bool,
):
    """Check Qwen TP2 and TP2/EP2 FULL_DECODE_ONLY Graph exactly."""
    devices = _devices()
    if len(devices) < 4:
        pytest.skip("Qwen3.6 2A2F TP2/EP2 graph requires 4 GPUs")
    model = _model_path()
    monkeypatch.setenv("AFD_E2E_GRAPH_AUDIT", "1")
    suffix = f"graph_b{batch_size}_{'ep2' if enable_expert_parallel else 'tp2'}"
    native_port = 18220 + batch_size + (10 if enable_expert_parallel else 0)
    attention_port = native_port + 1
    ffn_port = native_port + 2
    afd_port = 6400 + (10 if enable_expert_parallel else 0) + batch_size
    native: list[_ServerProcess] = []
    afd: list[_ServerProcess] = []
    try:
        native.append(
            _launch(
                [
                    *_common_args(
                        model,
                        tp_size=2,
                        use_cuda_graph=True,
                        enable_expert_parallel=enable_expert_parallel,
                    ),
                    "--served-model-name",
                    MODEL_NAME,
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(native_port),
                ],
                devices[:2],
                tmp_path / f"native_{suffix}.log",
            ),
        )
        _wait_for_api(native_port, native)
        oracle = _request(native_port, batch_size=batch_size)
    finally:
        _stop(native)

    config_args = dict(
        num_attention_ranks=2,
        num_ffn_ranks=2,
        port=afd_port,
        compute_gate_on_attention=True,
    )
    try:
        ffn_config = _afd_config(role="ffn", **config_args)
        attention_config = _afd_config(role="attention", **config_args)
        shared_args = _common_args(
            model,
            tp_size=2,
            use_cuda_graph=True,
            enable_expert_parallel=enable_expert_parallel,
        )
        afd.append(
            _launch(
                [
                    *shared_args,
                    "--served-model-name",
                    MODEL_NAME,
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(ffn_port),
                    "--additional-config",
                    ffn_config,
                ],
                devices[2:4],
                tmp_path / f"ffn_{suffix}.log",
            ),
        )
        afd.append(
            _launch(
                [
                    *shared_args,
                    "--served-model-name",
                    MODEL_NAME,
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(attention_port),
                    "--additional-config",
                    attention_config,
                ],
                devices[:2],
                tmp_path / f"attention_{suffix}.log",
            ),
        )
        _wait_for_api(attention_port, afd)
        _assert_exact(oracle, _request(attention_port, batch_size=batch_size))
        _assert_exact(oracle, _request(attention_port, batch_size=batch_size))
        logs = {
            server.log_path.name: _log_tail(server.log_path, 65536) for server in afd
        }
        assert "Capturing CUDA graphs" in logs[f"attention_{suffix}.log"]
        assert "cuda graph addresses" in logs[f"ffn_{suffix}.log"]
        assert "AFD FFN replayed CUDA Graph" in logs[f"ffn_{suffix}.log"]
    finally:
        _stop(afd)


@pytest.mark.gpu
@pytest.mark.e2e
@pytest.mark.slow
def test_qwen3_6_afd_dp2ep2_eager_matches_native(tmp_path: Path):
    """Exercise the existing CUDA DP2/EP2 transport without a Qwen protocol.

    Each service uses two local DP ranks with TP1/EP2. The colocated native
    service is the oracle; the AFD attention and FFN services each use the
    same DP/EP layout on their respective GPU pairs.
    """
    devices = _devices()
    if len(devices) < 4:
        pytest.skip("Qwen3.6 DP2/EP2 requires 4 GPUs")
    model = _model_path()
    native_port, attention_port, ffn_port, afd_port = 18420, 18421, 18422, 6420
    native: list[_ServerProcess] = []
    afd: list[_ServerProcess] = []
    try:
        native.append(
            _launch(
                [
                    *_common_args(
                        model,
                        tp_size=1,
                        data_parallel_size=2,
                        enable_expert_parallel=True,
                    ),
                    "--served-model-name",
                    MODEL_NAME,
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(native_port),
                ],
                devices[:2],
                tmp_path / "native_dp2ep2.log",
            ),
        )
        _wait_for_api(native_port, native)
        oracle = _request(native_port)
    finally:
        _stop(native)

    try:
        config_args = dict(
            num_attention_ranks=2,
            num_ffn_ranks=2,
            port=afd_port,
            compute_gate_on_attention=True,
        )
        shared_args = _common_args(
            model,
            tp_size=1,
            data_parallel_size=2,
            enable_expert_parallel=True,
        )
        afd.append(
            _launch(
                [
                    *shared_args,
                    "--served-model-name",
                    MODEL_NAME,
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(ffn_port),
                    "--additional-config",
                    _afd_config(role="ffn", **config_args),
                ],
                devices[2:4],
                tmp_path / "ffn_dp2ep2.log",
            ),
        )
        afd.append(
            _launch(
                [
                    *shared_args,
                    "--served-model-name",
                    MODEL_NAME,
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(attention_port),
                    "--additional-config",
                    _afd_config(role="attention", **config_args),
                ],
                devices[:2],
                tmp_path / "attention_dp2ep2.log",
            ),
        )
        _wait_for_api(attention_port, afd)
        _assert_exact(oracle, _request(attention_port))
        _assert_exact(oracle, _request(attention_port))
    finally:
        _stop(afd)


@pytest.mark.gpu
@pytest.mark.e2e
@pytest.mark.slow
def test_qwen3_6_afd_2a2f_tp2_dbo_graph_runs_two_ubatches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Exercise Qwen's AFD-owned two-ubatch DBO + Graph path.

    vLLM's colocated DBO oracle needs an optional DeepEP/NIXL kernel. The
    exact native comparison therefore remains conditionally gated on that
    external kernel, while this test proves that AFD uses two real stages,
    captures its FFN graph, and replays it for a batch-two request.
    """
    devices = _devices()
    if len(devices) < 4:
        pytest.skip("Qwen3.6 2A2F TP2 DBO graph requires 4 GPUs")
    model = _model_path()
    monkeypatch.setenv("AFD_E2E_GRAPH_AUDIT", "1")
    monkeypatch.setenv("AFD_E2E_DBO_AUDIT", "1")
    attention_port, ffn_port, afd_port = 18431, 18432, 6430
    config_args = dict(
        num_attention_ranks=2,
        num_ffn_ranks=2,
        port=afd_port,
        compute_gate_on_attention=True,
    )
    shared_args = _common_args(
        model,
        tp_size=2,
        use_cuda_graph=True,
        enable_dbo=True,
    )
    afd: list[_ServerProcess] = []
    try:
        afd.append(
            _launch(
                [
                    *shared_args,
                    "--served-model-name",
                    MODEL_NAME,
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(ffn_port),
                    "--additional-config",
                    _afd_config(role="ffn", **config_args),
                ],
                devices[2:4],
                tmp_path / "ffn_dbo_graph.log",
            ),
        )
        afd.append(
            _launch(
                [
                    *shared_args,
                    "--served-model-name",
                    MODEL_NAME,
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(attention_port),
                    "--additional-config",
                    _afd_config(role="attention", **config_args),
                ],
                devices[:2],
                tmp_path / "attention_dbo_graph.log",
            ),
        )
        _wait_for_api(attention_port, afd)
        assert len(_request(attention_port, batch_size=2)["choices"]) == 2
        assert len(_request(attention_port, batch_size=2)["choices"]) == 2
        ffn_log = _log_tail(tmp_path / "ffn_dbo_graph.log", 65536)
        assert "AFD FFN executed two DBO ubatches; stages=[0, 1]" in ffn_log
        assert "cuda graph addresses" in ffn_log
        assert "AFD FFN replayed CUDA Graph" in ffn_log
    finally:
        _stop(afd)
