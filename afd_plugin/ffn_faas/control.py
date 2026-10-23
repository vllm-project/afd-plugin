# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Classify failures only at session Store I/O boundaries."""

from __future__ import annotations

from typing import Protocol


class SessionPeerFailureError(RuntimeError):
    """A session peer explicitly failed, disconnected or lost heartbeat."""


class ControlStore(Protocol):
    def set(self, key: str, value: str) -> None: ...
    def get(self, key: str) -> bytes: ...
    def check(self, keys: list[str]) -> bool: ...


class SessionControl:
    """Borrow the executor's Store, preserving cause and failing I/O phase."""

    def __init__(self, store: ControlStore) -> None:
        self.store = store

    def write(self, key: str, value: str) -> None:
        try:
            self.store.set(key, value)
        except RuntimeError as error:
            raise SessionPeerFailureError(
                f"session Store write failed: {key}"
            ) from error

    def read(self, key: str) -> bytes:
        try:
            return self.store.get(key)
        except RuntimeError as error:
            raise SessionPeerFailureError(
                f"session Store read failed: {key}"
            ) from error

    def ready(self, keys: list[str]) -> bool:
        try:
            return self.store.check(keys)
        except RuntimeError as error:
            raise SessionPeerFailureError(
                f"session Store check failed: {keys}"
            ) from error
