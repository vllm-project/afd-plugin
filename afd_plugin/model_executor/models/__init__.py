# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""AFD model wrappers; shared metadata access remains a lazy compatibility export."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from afd_plugin.forward_context import get_afd_metadata_from_forward_context


def __getattr__(name: str):
    if name == "get_afd_metadata_from_forward_context":
        from afd_plugin.forward_context import get_afd_metadata_from_forward_context

        return get_afd_metadata_from_forward_context
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["get_afd_metadata_from_forward_context"]
