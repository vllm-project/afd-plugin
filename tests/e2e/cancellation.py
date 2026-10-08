# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Shared cancellation handling for the E2E runners.

Both the single-host and the multi-pod runner need the same SIGTERM/SIGINT
precedence: a signal during the body unwinds immediately, a signal during
cleanup is deferred until cleanup finishes, and the resulting SystemExit then
wins over a cleanup error, which in turn wins over the body error.
"""

from __future__ import annotations

import signal
from collections.abc import Callable, Iterator
from contextlib import contextmanager

HANDLED_SIGNALS = (signal.SIGTERM, signal.SIGINT)


@contextmanager
def cancellable_run(cleanup: Callable[[], None]) -> Iterator[None]:
    """Run a body with cancellation handling and a guaranteed cleanup.

    ``cleanup`` always runs exactly once, with the cancellation handlers still
    installed, and is restored to the previous handlers before this returns.
    """
    previous_handlers = {signum: signal.getsignal(signum) for signum in HANDLED_SIGNALS}
    received_signal: int | None = None
    cleanup_in_progress = False

    def exit_after_cleanup(signum: int, _frame: object) -> None:
        nonlocal received_signal
        if received_signal is not None:
            return
        received_signal = signum
        if not cleanup_in_progress:
            raise SystemExit(128 + signum)

    for signum in HANDLED_SIGNALS:
        signal.signal(signum, exit_after_cleanup)

    body_error: BaseException | None = None
    try:
        try:
            yield
        except BaseException as exc:
            body_error = exc
            raise
    finally:
        cleanup_error: BaseException | None = None
        cleanup_in_progress = True
        try:
            try:
                try:
                    cleanup()
                finally:
                    for signum, previous_handler in previous_handlers.items():
                        # Preloaded native libraries can install handlers
                        # unknown to Python (getsignal returns None). Python
                        # cannot restore those; reset to the OS default.
                        signal.signal(
                            signum,
                            signal.SIG_DFL
                            if previous_handler is None
                            else previous_handler,
                        )
            except BaseException as exc:
                cleanup_error = exc
        finally:
            cleanup_in_progress = False

        if received_signal is not None:
            signal_error = SystemExit(128 + received_signal)
            if cleanup_error is not None:
                raise signal_error from cleanup_error
            raise signal_error
        if cleanup_error is not None:
            if body_error is not None:
                raise body_error from cleanup_error
            raise cleanup_error
