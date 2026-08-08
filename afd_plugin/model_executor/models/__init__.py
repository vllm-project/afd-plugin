# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""AFD model wrapper namespace."""

from afd_plugin.model_executor.models.forward_context import (
    get_afd_metadata_from_forward_context,
)

__all__ = [
    "get_afd_metadata_from_forward_context",
]
