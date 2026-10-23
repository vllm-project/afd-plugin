# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Failure-only Store seam; never represents device execution."""


class DisconnectedStore:
    def __init__(self, error: RuntimeError) -> None:
        self.error = error

    def set(self, key: str, value: str) -> None:
        raise self.error

    def get(self, key: str) -> bytes:
        raise self.error

    def check(self, keys: list[str]) -> bool:
        raise self.error
