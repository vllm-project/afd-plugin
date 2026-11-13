# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Shared pytest helpers for the AFD plugin test suite."""

from __future__ import annotations

import json
import os
import signal
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from importlib.util import find_spec
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# Covers the roughly 64-second nested lm-eval/vLLM cleanup bound plus the NPU
# async runner's 120-second procfs cleanup window with buffer.
RUNNER_CLEANUP_TIMEOUT_S = 240

# vLLM-Ascend 80d8c194f (v0.26 baseline) has an inherent circular import:
# device/device_op.py -> ops/__init__ -> fused_moe -> experts_selector.py ->
# back to the partially initialized device_op. Real ``vllm serve`` imports
# ``vllm_ascend.ops`` first and never trips it, but pytest's import order does.
# Pre-importing ``vllm_ascend.ops`` here breaks the cycle for the test run.
# Environments without vllm_ascend (CPU-only CI) simply skip this. When the
# package is installed, import and ABI failures must remain visible.
if find_spec("vllm_ascend") is not None:  # pragma: no cover - environment-dependent
    import vllm_ascend.ops  # noqa: F401


@contextmanager
def preserve_environment_variable(name: str) -> Iterator[None]:
    """Restore an environment variable after a temporary E2E asset override."""
    original_value = os.environ.get(name)
    try:
        yield
    finally:
        if original_value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = original_value


def run_runner(command: list[str], env: dict[str, str] | None = None) -> None:
    """Run an E2E runner command and forward cancellation to the process group."""
    handled_signals = (signal.SIGTERM, signal.SIGINT)
    previous_handlers = {signum: signal.getsignal(signum) for signum in handled_signals}
    process: subprocess.Popen | None = None
    received_signal: int | None = None
    forwarded = False

    def forward_received_signal() -> None:
        nonlocal forwarded
        if process is None or received_signal is None or forwarded:
            return
        forwarded = True
        with suppress(ProcessLookupError):
            os.killpg(process.pid, received_signal)
        raise SystemExit(128 + received_signal)

    def forward_cancellation(signum, _frame) -> None:
        nonlocal received_signal
        if received_signal is not None:
            return
        received_signal = signum
        forward_received_signal()

    for signum in handled_signals:
        signal.signal(signum, forward_cancellation)

    try:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            start_new_session=True,
        )
        forward_received_signal()
        returncode = process.wait()
        if returncode != 0:
            raise subprocess.CalledProcessError(returncode, command)
    finally:
        try:
            if process is not None and process.poll() is None:
                try:
                    process.wait(timeout=RUNNER_CLEANUP_TIMEOUT_S)
                except subprocess.TimeoutExpired:
                    try:
                        with suppress(ProcessLookupError):
                            os.killpg(process.pid, signal.SIGKILL)
                    finally:
                        process.wait()
        finally:
            for signum, previous_handler in previous_handlers.items():
                signal.signal(signum, previous_handler)


def run_case_entry(
    command: list[str], output_path: Path, env: dict[str, str] | None = None
) -> None:
    """Require both successful execution and canonical acceptance at pytest gates."""
    run_runner(command, env=env)
    record = json.loads((output_path / "run.json").read_text())
    assert record["acceptance_passed"], (
        "task execution succeeded but the fixed scenario did not qualify: "
        f"{record['configuration']}"
    )


def download_dataset(dataset_id: str, dataset_config: str | None = None) -> None:
    """Download/cache a Hugging Face dataset.

    Args:
        dataset_id: Dataset repo id, e.g. ``openai/gsm8k``.
        dataset_config: Optional dataset configuration name, e.g. ``main``.
    """
    # Must be set before ``datasets`` transitively imports huggingface_hub,
    # which reads HF_HUB_DISABLE_XET once at import time.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    from datasets import load_dataset

    if dataset_config is None:
        load_dataset(dataset_id)
    else:
        load_dataset(dataset_id, dataset_config)
    print(f"[e2e] Dataset ready: {dataset_id}", flush=True)


def download_model(repo_id: str) -> Path:
    """Download/cache a Hugging Face model repo and return its local path.

    Args:
        repo_id: Model repo id, e.g. ``deepseek-ai/DeepSeek-V2-Lite``.
    """
    # Must be set before importing huggingface_hub: Xet chunk caches can fill
    # the container root filesystem in CI even when HF_HOME is on a large volume.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to download models; "
            "install it in the E2E environment",
        ) from exc

    print(f"[e2e] Downloading model {repo_id}", flush=True)
    model_path = Path(snapshot_download(repo_id=repo_id))
    print(f"[e2e] Model ready at {model_path}", flush=True)
    return model_path
