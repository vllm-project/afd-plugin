# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Standalone FFN CPU fixtures."""

from __future__ import annotations

import pytest

from tests.unit.fakes.ffn_faas import write_v4_checkpoint


@pytest.fixture
def checkpoint_dir(tmp_path):
    return write_v4_checkpoint(tmp_path / "model")
