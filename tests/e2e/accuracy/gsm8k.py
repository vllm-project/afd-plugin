# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Shared GSM8K lm-eval integration helpers."""

from __future__ import annotations

import json
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import FrameType

from tests.e2e.process_utils import terminate_process_groups

LM_EVAL_TIMEOUT_S = 7200
LM_EVAL_TERMINATION_TIMEOUT_S = 20
LM_EVAL_PROCESS_POLL_INTERVAL_S = 0.2
LM_EVAL_REAP_TIMEOUT_S = 5
LM_EVAL_READER_JOIN_TIMEOUT_S = 5


def _finish_lm_eval_cleanup(
    process: subprocess.Popen[str],
    reader: threading.Thread | None,
    *,
    terminate_group: bool,
) -> None:
    if terminate_group:
        failures = terminate_process_groups(
            [process],
            termination_timeout_s=LM_EVAL_TERMINATION_TIMEOUT_S,
            poll_interval_s=LM_EVAL_PROCESS_POLL_INTERVAL_S,
            reap_timeout_s=LM_EVAL_REAP_TIMEOUT_S,
            process_name="lm-eval",
        )
    else:
        failures = []
        try:
            process.wait(timeout=LM_EVAL_REAP_TIMEOUT_S)
        except Exception as exc:
            failures.append(f"wait failed for lm-eval process {process.pid}: {exc}")

    if reader is not None:
        try:
            reader.join(timeout=LM_EVAL_READER_JOIN_TIMEOUT_S)
        except Exception as exc:
            failures.append(f"lm-eval reader thread join failed: {exc}")
        else:
            if reader.is_alive():
                failures.append("lm-eval reader thread did not stop")

    if failures:
        raise RuntimeError("; ".join(failures))


def _run_lm_eval(
    base_url: str,
    model_name: str,
    *,
    output_path: str,
    num_fewshot: int | None = None,
    batch_size: int | None = None,
    num_concurrent: int | None = None,
    max_gen_toks: int = 512,
    tokenizer: str | None = None,
    limit: int | None = None,
    timeout_s: float = LM_EVAL_TIMEOUT_S,
) -> dict:
    """Run lm-eval against the AFD attention server and return results dict."""
    tokenizer_arg = f",tokenizer={tokenizer}" if tokenizer else ""
    num_concurrent_arg = (
        f",num_concurrent={num_concurrent}" if num_concurrent is not None else ""
    )
    cmd = [
        sys.executable,
        "-m",
        "lm_eval",
        "--model",
        "local-completions",
        "--model_args",
        (
            f"model={model_name},"
            f"base_url={base_url}/v1/completions,"
            f"max_gen_toks={max_gen_toks},"
            f"tokenized_requests=False"
            f"{num_concurrent_arg}"
            f"{tokenizer_arg}"
        ),
    ]
    cmd.extend(
        [
            "--tasks",
            "gsm8k",
            "--output_path",
            output_path,
            "--log_samples",
        ]
    )
    if num_fewshot is not None:
        cmd.extend(["--num_fewshot", str(num_fewshot)])
    if batch_size is not None:
        cmd.extend(["--batch_size", str(batch_size)])
    if limit is not None:
        cmd.extend(["--limit", str(limit)])

    print(f"\n[lm-eval] Running: {' '.join(cmd)}")
    env = os.environ.copy()
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    env.setdefault("USE_MODELSCOPE_HUB", "0")
    env["PYTHONUNBUFFERED"] = "1"
    # Stream lm-eval output live (pytest -s surfaces it) instead of capturing.
    # capture_output=True swallows everything until exit, which makes a slow run
    # indistinguishable from a deadlock — a real footgun on NPU eager-mode runs.
    handled_signals = (signal.SIGTERM, signal.SIGINT)
    previous_handlers = {signum: signal.getsignal(signum) for signum in handled_signals}
    proc: subprocess.Popen[str] | None = None
    reader: threading.Thread | None = None
    completed = False
    cleanup_state_registered = False
    received_signal: tuple[int, FrameType | None] | None = None
    delegated = False
    cleanup_in_progress = False

    def delegate_received_signal() -> None:
        nonlocal delegated
        if (
            not cleanup_state_registered
            or received_signal is None
            or delegated
            or cleanup_in_progress
        ):
            return
        delegated = True
        signum, frame = received_signal
        previous_handler = previous_handlers[signal.Signals(signum)]
        if callable(previous_handler):
            previous_handler(signum, frame)
        elif previous_handler != signal.SIG_IGN:
            raise SystemExit(128 + signum)

    def pending_signal_proxy(signum: int, frame: FrameType | None) -> None:
        nonlocal received_signal
        if received_signal is not None:
            return
        received_signal = (signum, frame)
        delegate_received_signal()

    try:
        for signum in handled_signals:
            signal.signal(signum, pending_signal_proxy)

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
            start_new_session=True,
        )
        deadline = time.monotonic() + timeout_s
        stdout_lines: list[str] = []

        # Pump stdout through a queue on a daemon thread so a blocking
        # readline() on a hung/deadlocked lm-eval subprocess cannot defeat the
        # deadline below.
        line_queue: queue.Queue[str | None] = queue.Queue()

        def pump() -> None:
            assert proc is not None
            assert proc.stdout is not None
            for line in proc.stdout:
                line_queue.put(line)
            line_queue.put(None)  # EOF sentinel

        reader = threading.Thread(target=pump, daemon=True)
        reader.start()
        cleanup_state_registered = True
        delegate_received_signal()

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"lm-eval exceeded {timeout_s}s budget")
            try:
                line = line_queue.get(timeout=min(remaining, 5.0))
            except queue.Empty:
                continue
            if line is None:  # EOF — reader drained the pipe
                break
            stdout_lines.append(line)
            sys.stdout.write(line)
            sys.stdout.flush()

        proc.wait(timeout=max(deadline - time.monotonic(), 0))
        stdout_text = "".join(stdout_lines)

        if proc.returncode != 0:
            raise RuntimeError(
                f"lm-eval exited with code {proc.returncode}:\n{stdout_text[-3000:]}",
            )

        # lm-eval writes timestamped results below a model-specific directory.
        # Keep the fixed filename as a fallback for older versions.
        op = Path(output_path)
        results_file = None
        if op.is_file():
            results_file = op
        elif op.is_dir():
            hits = list(op.rglob("results_*.json"))
            if not hits:
                hits = list(op.rglob("results.json"))
            if hits:
                results_file = max(hits, key=lambda path: path.stat().st_mtime_ns)
        if results_file is not None:
            with open(results_file) as f:
                results = json.load(f)
        else:
            results = _parse_lm_eval_stdout(stdout_text)
        completed = True
        return results
    finally:
        body_error = sys.exc_info()[1]
        cleanup_error: BaseException | None = None
        cleanup_in_progress = True
        try:
            try:
                try:
                    if proc is not None:
                        _finish_lm_eval_cleanup(
                            proc,
                            reader,
                            terminate_group=not completed,
                        )
                finally:
                    for signum, previous_handler in previous_handlers.items():
                        signal.signal(signum, previous_handler)
            except BaseException as exc:
                cleanup_error = exc
        finally:
            cleanup_in_progress = False

        try:
            delegate_received_signal()
        except BaseException as signal_error:
            if cleanup_error is not None:
                raise signal_error from cleanup_error
            raise
        if cleanup_error is not None:
            if body_error is not None:
                raise body_error from cleanup_error
            raise cleanup_error


def _parse_lm_eval_stdout(stdout: str) -> dict:
    """Parse lm-eval results from stdout.

    Prefers a trailing JSON block (some versions print one). Falls back to the
    pipe-delimited results table lm-eval always prints, e.g. a strict-match row
    like: | strict-match | 5 | exact_match | 0.33 | +/- | 0.0473 |. Returns a
    dict shaped like results.json for _extract_gsm8k_accuracy.
    """
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    m = re.search(
        r"strict-match.*?exact_match[^0-9.\-+]*([0-9]*\.?[0-9]+)",
        stdout,
        re.DOTALL,
    )
    strict_val = float(m.group(1)) if m else None
    m2 = re.search(r"exact_match[^0-9.\-+]*([0-9]*\.?[0-9]+)", stdout, re.DOTALL)
    flex_val = float(m2.group(1)) if m2 else None
    if strict_val is None and flex_val is None:
        raise RuntimeError("Could not parse lm-eval results from stdout")
    gsm8k = {}
    if strict_val is not None:
        gsm8k["exact_match,strict-match"] = strict_val
    if flex_val is not None:
        gsm8k["exact_match,flexible-extract"] = flex_val
    exact_match = strict_val if strict_val is not None else flex_val
    assert exact_match is not None
    gsm8k["exact_match"] = exact_match
    return {"results": {"gsm8k": gsm8k}}


def _extract_gsm8k_accuracy(results: dict) -> float:
    """Extract the GSM8K exact_match,strict-match score from lm-eval output."""
    # Navigate the nested results structure
    # results["results"]["gsm8k"]["exact_match,strict-match"]
    task_results = results.get("results", results)

    gsm8k = task_results.get("gsm8k", task_results)
    for key in ("exact_match,strict-match", "exact_match"):
        if key in gsm8k:
            return float(gsm8k[key])

    raise KeyError(
        f"Could not find GSM8K accuracy in results. "
        f"Available keys: {list(gsm8k.keys())}",
    )


def _extract_gsm8k_sample_count(results: dict) -> int:
    """Extract the effective GSM8K sample count from lm-eval results."""
    try:
        sample_count = results["n-samples"]["gsm8k"]["effective"]
    except (KeyError, TypeError) as exc:
        raise KeyError(
            "Could not find GSM8K sample count at "
            "results['n-samples']['gsm8k']['effective']",
        ) from exc
    try:
        count = int(sample_count)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"GSM8K sample count must be an integer, got {sample_count!r}",
        ) from exc
    if isinstance(sample_count, bool) or (
        isinstance(sample_count, float) and not sample_count.is_integer()
    ):
        raise ValueError(
            f"GSM8K sample count must be an integer, got {sample_count!r}",
        )
    return count
