# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Package-level AFD forward metadata reads."""

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")

from afd_plugin.forward_context import (
    get_afd_metadata_from_forward_context,  # noqa: E402
)


def test_get_afd_metadata_from_additional_kwargs():
    forward_context = SimpleNamespace(
        additional_kwargs={"afd_metadata": {"stage": 0}},
        afd_metadata={"stage": 1},
    )

    assert get_afd_metadata_from_forward_context(forward_context) == {"stage": 0}


def test_get_afd_metadata_ignores_forward_context_attribute():
    forward_context = SimpleNamespace(
        additional_kwargs={},
        afd_metadata={"stage": 0},
    )

    assert get_afd_metadata_from_forward_context(forward_context) is None
